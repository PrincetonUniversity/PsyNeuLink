from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from psyneulink.core.batched import specs
from psyneulink.core.batched.graph import projection_inputs
from psyneulink.core.batched.ir import BatchedCompositionIR, BatchedSimulationResult
from psyneulink.core.batched.kernel_ir import KernelIR, lower_to_kernel_ir


def run_ir_debug(
    ir: BatchedCompositionIR,
    inputs,
    parameter_sets,
    num_estimates: int,
    subject_slices=None,
    seed=None,
    common_random_numbers: bool = True,
) -> BatchedSimulationResult:
    params = normalize_parameter_sets(parameter_sets, ir)
    prepared_inputs = prepare_inputs(ir, inputs, subject_slices)
    kernel = lower_to_kernel_ir(ir)

    values = _run_kernel(kernel, prepared_inputs, params, num_estimates, seed, common_random_numbers)

    return BatchedSimulationResult(
        values=values,
        output_names=ir.output_names,
        backend="ir_debug",
        metadata={"model_kind": ir.model_kind, "fusion_kind": kernel.fusion_kind},
    )


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
        raise ValueError("Batched IR debug execution requires a graph IR.")

    values = {}
    for input_spec in ir.graph.inputs:
        raw_value = _extract_named_input(inputs, (input_spec.node,), fallback_first=len(ir.graph.inputs) == 1)
        coerced = _coerce_trials(raw_value, width=input_spec.width)
        values[input_spec.node] = coerced[:, 0] if input_spec.width == 1 else coerced

    return _split_subject_trials(values, subject_slices)


def _run_kernel(
    kernel: KernelIR,
    inputs: dict[str, np.ndarray],
    params: list[dict[str, float]],
    num_estimates: int,
    seed,
    common_random_numbers: bool,
) -> np.ndarray:
    graph = kernel.graph
    if graph is None:
        raise ValueError("Batched IR debug execution requires a graph IR.")

    first_input = inputs[graph.inputs[0].node]
    num_subjects, num_trials = first_input.shape[:2]
    num_params = len(params)
    outcome_width = sum(output.width for output in graph.outputs)
    result = np.zeros((num_params, num_subjects, num_trials, num_estimates, outcome_width), dtype=np.float32)

    for p_idx, param in enumerate(params):
        for s_idx in range(num_subjects):
            for e_idx in range(num_estimates):
                state = _initial_state(graph)
                for t_idx in range(num_trials):
                    node_outputs = {}
                    rng = _rng_for(seed, p_idx, s_idx, t_idx, e_idx, common_random_numbers)
                    for node_name in graph.execution_order:
                        node = graph.node(node_name)
                        node_outputs[node_name] = _execute_node(
                            graph,
                            node,
                            inputs,
                            node_outputs,
                            state,
                            param,
                            s_idx,
                            t_idx,
                            rng,
                            kernel.max_steps,
                        )
                    cursor = 0
                    for output in graph.outputs:
                        value = node_outputs[output.node][output.port].reshape(-1)
                        result[p_idx, s_idx, t_idx, e_idx, cursor: cursor + output.width] = value
                        cursor += output.width
    return result


def _execute_node(
    graph,
    node,
    inputs: dict[str, np.ndarray],
    node_outputs: dict[str, dict[str, np.ndarray]],
    state: dict[str, np.ndarray],
    params: dict[str, float],
    subject_idx: int,
    trial_idx: int,
    rng: np.random.Generator,
    max_steps: int,
) -> dict[str, np.ndarray]:
    spec_key = node.attrs.get("spec_key")
    if not spec_key:
        raise ValueError(f"Unsupported batched graph node '{node.name}' ({node.component_type}).")
    spec = specs.lookup_spec(spec_key)
    input_value = _node_input(graph, node, inputs, node_outputs, subject_idx, trial_idx)

    if isinstance(spec, specs.ElementwiseFunctionSpec):
        args = [_param(params, node, binding.arg) for binding in spec.params]
        value = spec.body(input_value, *args)
        return {_primary_output_port(node): np.asarray(value, dtype=np.float32).reshape(-1)}

    if isinstance(spec, specs.MechanismOpSpec):
        resolved_params = {binding.arg: _param(params, node, binding.arg) for binding in spec.params}
        if spec.cpu_execute is not None:
            return spec.cpu_execute(
                node,
                input_value,
                inputs,
                state,
                resolved_params,
                subject_idx,
                trial_idx,
                rng,
                max_steps,
            )
        if spec.cpu_body is None:
            raise ValueError(f"Batched op for '{node.name}' has no CPU implementation.")

        args = []
        for binding in spec.cpu_bindings:
            if binding.role == "input":
                args.append(float(input_value.reshape(-1)[0]) if node.input_width == 1 else input_value)
            elif binding.role == "param":
                args.append(resolved_params[binding.name])
            elif binding.role == "rng":
                args.append(rng)
            elif binding.role == "max_steps":
                args.append(max_steps)
            elif binding.role == "state":
                args.append(state[f"{node.name}.{binding.name}"])
            else:
                raise ValueError(
                    f"Batched op for '{node.name}' has an unsupported CPU arg role "
                    f"'{binding.role}'."
                )
        results = spec.cpu_body(*args)
        if not isinstance(results, tuple):
            results = (results,)
        op_outputs = tuple(node.attrs.get("op_outputs", ()))
        if len(results) != len(op_outputs):
            raise ValueError(
                f"Batched op for '{node.name}' returned {len(results)} values, "
                f"expected {len(op_outputs)}."
            )
        return {
            port: np.asarray(value, dtype=np.float32).reshape(-1)
            for (port, _width), value in zip(op_outputs, results)
        }

    raise ValueError(f"Unsupported batched graph node '{node.name}' ({node.component_type}).")


def _node_input(graph, node, inputs, node_outputs, subject_idx: int, trial_idx: int) -> np.ndarray:
    projections = projection_inputs(graph, node.name)
    if not projections:
        return np.asarray(inputs[node.name][subject_idx, trial_idx], dtype=np.float32).reshape(-1)

    contributions = []
    for projection in projections:
        sender_outputs = node_outputs.get(projection.sender)
        if sender_outputs is None:
            continue
        sender_value = sender_outputs[projection.sender_port].reshape(-1)
        contributions.append(np.asarray(sender_value @ projection.matrix, dtype=np.float32).reshape(-1))

    if not contributions:
        return np.zeros(node.input_width, dtype=np.float32)
    if node.combine == "product":
        value = np.ones_like(contributions[0], dtype=np.float32)
        for contribution in contributions:
            value = value * contribution
        return value
    value = np.zeros_like(contributions[0], dtype=np.float32)
    for contribution in contributions:
        value = value + contribution
    return value


def _initial_state(graph) -> dict[str, np.ndarray]:
    return {
        state.name: np.asarray(state.initial_value, dtype=np.float32)
        for state in graph.states
    }


def _primary_output_port(node) -> str:
    output_ports = tuple(node.attrs.get("output_ports", ()))
    if output_ports:
        return output_ports[0]
    return "RESULT"


def _param(params: dict[str, float], node, local_name: str) -> float:
    return float(params[node.params[local_name]])


def _lca_max_steps(ir: BatchedCompositionIR, inputs: dict[str, np.ndarray]) -> int:
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


def _rng_for(seed, p_idx: int, s_idx: int, t_idx: int, e_idx: int, common_random_numbers: bool):
    base_seed = 0 if seed is None else int(seed)
    p_component = 0 if common_random_numbers else p_idx
    mixed = (
        base_seed
        + 1000003 * p_component
        + 10007 * s_idx
        + 1009 * t_idx
        + 9176 * e_idx
    )
    return np.random.default_rng(mixed % (2 ** 32))
