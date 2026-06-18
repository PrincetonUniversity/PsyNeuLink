"""Backend-neutral input/parameter normalization for batched execution.

These helpers shape user inputs and parameter sets into the structure-of-arrays
buffers the batched kernels consume.  They are independent of any execution
backend (they used to live in the now-removed `ir_debug` numpy executor) and are
shared by the Triton GPU and CPU-interpret runtimes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from psyneulink.core.batched.ir import BatchedCompositionIR


def normalize_parameter_sets(parameter_sets, ir: BatchedCompositionIR) -> list[dict[str, float]]:
    defaults = dict(ir.param_defaults)
    if parameter_sets is None:
        return [defaults]

    if isinstance(parameter_sets, Mapping):
        values = list(parameter_sets.values())
        if values and all(_is_vector_value(v) for v in values):
            lengths = {len(np.asarray(v).reshape(-1)) for v in values}
            if len(lengths) == 1:
                rows = []
                for idx in range(next(iter(lengths))):
                    row = dict(defaults)
                    for key, value in parameter_sets.items():
                        row[_normalize_param_key(key)] = float(np.asarray(value).reshape(-1)[idx])
                    rows.append(_canonicalize_param_set(row, ir))
                return rows

        row = dict(defaults)
        for key, value in parameter_sets.items():
            row[_normalize_param_key(key)] = _as_scalar(value)
        return [_canonicalize_param_set(row, ir)]

    rows = []
    for parameter_set in parameter_sets:
        row = dict(defaults)
        if isinstance(parameter_set, Mapping):
            for key, value in parameter_set.items():
                row[_normalize_param_key(key)] = _as_scalar(value)
        else:
            values = np.asarray(parameter_set, dtype=float).reshape(-1)
            if len(values) != len(ir.params):
                raise ValueError(
                    "Array parameter sets must have one value for each batched IR parameter."
                )
            for spec, value in zip(ir.params, values):
                row[spec.name] = float(value)
        rows.append(_canonicalize_param_set(row, ir))
    return rows


def prepare_inputs(ir: BatchedCompositionIR, inputs, subject_slices=None) -> dict[str, np.ndarray]:
    if ir.graph is None:
        raise ValueError("Batched execution requires a graph IR.")

    values = {}
    for input_spec in ir.graph.inputs:
        raw_value = _extract_named_input(inputs, (input_spec.node,), fallback_first=len(ir.graph.inputs) == 1)
        coerced = _coerce_trials(raw_value, width=input_spec.width)
        values[input_spec.node] = coerced[:, 0] if input_spec.width == 1 else coerced

    return _split_subject_trials(values, subject_slices)


def lca_max_steps(ir: BatchedCompositionIR, inputs: dict[str, np.ndarray]) -> int:
    metadata_limit = int(ir.metadata.get("lca_max_steps", 0) or 0)
    input_limit = 0
    if ir.graph is not None:
        for node in ir.graph.nodes:
            termination_input_node = node.attrs.get("termination_input_node")
            if termination_input_node is None:
                continue
            if termination_input_node in inputs and np.size(inputs[termination_input_node]):
                input_limit = max(input_limit, int(np.ceil(np.max(inputs[termination_input_node]))))
    return max(1, metadata_limit, input_limit)


def _split_subject_trials(values: dict[str, np.ndarray], subject_slices) -> dict[str, np.ndarray]:
    first = next(iter(values.values()))
    num_trials = len(first)
    if subject_slices is None:
        subject_slices = [slice(0, num_trials)]

    result = {}
    for name, array in values.items():
        subject_arrays = [np.asarray(array[subject_slice], dtype=np.float32) for subject_slice in subject_slices]
        max_trials = max(len(subject_array) for subject_array in subject_arrays)
        if subject_arrays[0].ndim == 1:
            padded = np.zeros((len(subject_arrays), max_trials), dtype=np.float32)
            for idx, subject_array in enumerate(subject_arrays):
                padded[idx, : len(subject_array)] = subject_array
        else:
            padded = np.zeros((len(subject_arrays), max_trials, subject_arrays[0].shape[1]), dtype=np.float32)
            for idx, subject_array in enumerate(subject_arrays):
                padded[idx, : len(subject_array), :] = subject_array
        result[name] = padded
    return result


def _extract_named_input(inputs, name_prefixes: Sequence[str], fallback_first: bool = False):
    if not isinstance(inputs, Mapping):
        return inputs

    for key, value in inputs.items():
        key_name = getattr(key, "name", str(key))
        if any(key_name.startswith(prefix) for prefix in name_prefixes):
            return value

    if fallback_first and len(inputs) == 1:
        return next(iter(inputs.values()))

    raise KeyError(f"Could not find an input keyed by one of {tuple(name_prefixes)}.")


def _coerce_trials(value, width: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != object and array.ndim == 2 and array.shape[1] == width:
        return array.astype(np.float32)
    if array.dtype != object and array.ndim == 1 and width == 1:
        return array.reshape(-1, 1).astype(np.float32)

    rows = []
    for trial in value:
        flattened = np.asarray(_unwrap_singletons(trial), dtype=float).reshape(-1)
        if len(flattened) != width:
            raise ValueError(f"Expected trial input width {width}, got {len(flattened)}.")
        rows.append(flattened)
    return np.asarray(rows, dtype=np.float32)


def _unwrap_singletons(value):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def _canonicalize_param_set(row: dict[str, float], ir: BatchedCompositionIR) -> dict[str, float]:
    canonical = dict(ir.param_defaults)
    matched = set()
    for spec in ir.params:
        candidate_names = (spec.name,) + spec.aliases
        for name in candidate_names:
            if name in row:
                canonical[spec.name] = float(row[name])
                matched.add(name)
        for name, value in row.items():
            if isinstance(name, str) and any(name.endswith(f".{candidate}") for candidate in candidate_names):
                canonical[spec.name] = float(value)
                matched.add(name)
    unknown = sorted(
        str(name)
        for name in row
        if name not in matched and name not in canonical
    )
    if unknown:
        available = sorted({spec.name for spec in ir.params} | {alias for spec in ir.params for alias in spec.aliases})
        raise ValueError(
            "Unknown batched parameter(s): "
            f"{', '.join(unknown)}. Available parameters: {', '.join(available)}"
        )
    return canonical


def _normalize_param_key(key) -> str:
    if isinstance(key, tuple) and key:
        return str(key[0])
    return str(key)


def _is_vector_value(value) -> bool:
    if isinstance(value, str):
        return False
    try:
        array = np.asarray(value)
    except Exception:
        return False
    return array.ndim > 0 and array.size > 1


def _as_scalar(value) -> float:
    return float(np.asarray(value, dtype=float).reshape(-1)[0])
