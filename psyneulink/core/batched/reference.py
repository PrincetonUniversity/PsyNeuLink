from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from psyneulink.core.batched.ir import BatchedCompositionIR, BatchedSimulationResult
from psyneulink.core.batched.registry import DDM_MODEL, STABILITY_FLEXIBILITY_MODEL


def run_reference(
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

    if ir.model_kind == DDM_MODEL:
        values = _run_ddm(ir, prepared_inputs, params, num_estimates, seed, common_random_numbers)
    elif ir.model_kind == STABILITY_FLEXIBILITY_MODEL:
        values = _run_stability_flexibility(
            ir,
            prepared_inputs,
            params,
            num_estimates,
            seed,
            common_random_numbers,
        )
    else:
        raise ValueError(f"Unknown batched model kind '{ir.model_kind}'.")

    return BatchedSimulationResult(
        values=values,
        output_names=ir.output_names,
        backend="reference",
        metadata={"model_kind": ir.model_kind},
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
    if ir.model_kind == DDM_MODEL:
        stimulus = _extract_named_input(inputs, ("DDM",), fallback_first=True)
        stimulus = _coerce_trials(stimulus, width=1)[:, 0]
        return _split_subject_trials({"stimulus": stimulus}, subject_slices)

    task = _coerce_trials(_extract_named_input(inputs, ("Task Input [I1, I2]",)), width=2)
    stimulus = _coerce_trials(_extract_named_input(inputs, ("Stimulus Input [S1, S2]",)), width=2)
    cue = _coerce_trials(_extract_named_input(inputs, ("Cue-Stimulus Interval",)), width=1)[:, 0]
    try:
        correct = _coerce_trials(_extract_named_input(inputs, ("Correct Response Info",)), width=1)[:, 0]
    except KeyError:
        correct = np.sum(task * stimulus, axis=1)

    return _split_subject_trials(
        {
            "task": task,
            "stimulus": stimulus,
            "cue": cue,
            "correct": correct,
        },
        subject_slices,
    )


def _run_ddm(
    ir: BatchedCompositionIR,
    inputs: dict[str, np.ndarray],
    params: list[dict[str, float]],
    num_estimates: int,
    seed,
    common_random_numbers: bool,
) -> np.ndarray:
    stimulus = inputs["stimulus"]
    num_params = len(params)
    num_subjects, num_trials = stimulus.shape
    values = np.zeros((num_params, num_subjects, num_trials, num_estimates, 2), dtype=np.float32)

    for p_idx, param in enumerate(params):
        for s_idx in range(num_subjects):
            for t_idx in range(num_trials):
                drift_input = stimulus[s_idx, t_idx]
                for e_idx in range(num_estimates):
                    rng = _rng_for(seed, p_idx, s_idx, t_idx, e_idx, common_random_numbers)
                    decision, response_time = _simulate_ddm_trial(
                        drift_input=drift_input,
                        rate=param["rate"],
                        noise=param["noise"],
                        threshold=param["threshold"],
                        non_decision_time=param["non_decision_time"],
                        time_step_size=param["time_step_size"],
                        starting_value=param["starting_value"],
                        offset=param["offset"],
                        max_steps=ir.max_steps,
                        rng=rng,
                    )
                    values[p_idx, s_idx, t_idx, e_idx, 0] = decision
                    values[p_idx, s_idx, t_idx, e_idx, 1] = response_time

    return values


def _run_stability_flexibility(
    ir: BatchedCompositionIR,
    inputs: dict[str, np.ndarray],
    params: list[dict[str, float]],
    num_estimates: int,
    seed,
    common_random_numbers: bool,
) -> np.ndarray:
    task = inputs["task"]
    stimulus = inputs["stimulus"]
    cue = inputs["cue"]
    correct = inputs["correct"]
    num_params = len(params)
    num_subjects, num_trials, _ = task.shape
    lca_max_steps = _stability_flexibility_lca_max_steps(ir, inputs)
    values = np.zeros((num_params, num_subjects, num_trials, num_estimates, 2), dtype=np.float32)

    for p_idx, param in enumerate(params):
        for s_idx in range(num_subjects):
            for e_idx in range(num_estimates):
                lca_pre = np.zeros(2, dtype=float)
                lca_activity = np.zeros(2, dtype=float)
                for t_idx in range(num_trials):
                    rng = _rng_for(seed, p_idx, s_idx, t_idx, e_idx, common_random_numbers)
                    lca_pre, lca_activity = _simulate_lca_cue_period(
                        task=task[s_idx, t_idx],
                        pre_activity=lca_pre,
                        activity=lca_activity,
                        cue=cue[s_idx, t_idx],
                        gain=param["gain"],
                        leak=param["leak"],
                        competition=param["competition"],
                        self_excitation=param["self_excitation"],
                        noise=param["lca_noise"],
                        time_step_size=param["lca_time_step_size"],
                        max_steps=lca_max_steps,
                        rng=rng,
                    )
                    drift = (
                        np.sum(stimulus[s_idx, t_idx] * lca_activity)
                        + param["automaticity"] * np.sum(stimulus[s_idx, t_idx])
                    )
                    drift *= param["scale"] * correct[s_idx, t_idx]
                    decision, response_time = _simulate_ddm_trial(
                        drift_input=drift,
                        rate=1.0,
                        noise=param["ddm_noise"],
                        threshold=param["threshold"],
                        non_decision_time=param["non_decision_time"],
                        time_step_size=param["ddm_time_step_size"],
                        starting_value=param["starting_value"],
                        offset=param["ddm_offset"],
                        max_steps=ir.max_steps,
                        rng=rng,
                    )
                    values[p_idx, s_idx, t_idx, e_idx, 0] = decision
                    values[p_idx, s_idx, t_idx, e_idx, 1] = response_time

    return values


def _simulate_ddm_trial(
    *,
    drift_input: float,
    rate: float,
    noise: float,
    threshold: float,
    non_decision_time: float,
    time_step_size: float,
    starting_value: float,
    offset: float,
    max_steps: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    value = float(starting_value)
    steps = 0
    sqrt_dt = np.sqrt(time_step_size)
    boundary_tolerance = max(1e-7, abs(threshold) * 1e-6)
    for _ in range(max_steps):
        if abs(value) + boundary_tolerance >= threshold:
            break
        random_draw = rng.normal()
        value = value + rate * drift_input * time_step_size + noise * sqrt_dt * random_draw
        value = float(np.clip(value + offset, -threshold, threshold))
        steps += 1
    return (1.0 if value > 0 else 0.0), float(non_decision_time + steps * time_step_size)


def _simulate_lca_cue_period(
    *,
    task: np.ndarray,
    pre_activity: np.ndarray,
    activity: np.ndarray,
    cue: float,
    gain: float,
    leak: float,
    competition: float,
    self_excitation: float,
    noise: float,
    time_step_size: float,
    max_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    steps = min(max(int(np.ceil(cue)), 0), max_steps)
    sqrt_dt = np.sqrt(time_step_size)
    pre = np.array(pre_activity, dtype=float, copy=True)
    act = np.array(activity, dtype=float, copy=True)
    for _ in range(steps):
        recurrent = np.array(
            [
                self_excitation * act[0] - competition * act[1],
                -competition * act[0] + self_excitation * act[1],
            ]
        )
        update = (task + recurrent - leak * pre) * time_step_size
        if noise:
            update = update + noise * sqrt_dt * rng.normal(size=2)
        pre = pre + update
        act = 1.0 / (1.0 + np.exp(-gain * pre))
    return pre, act


def _stability_flexibility_lca_max_steps(ir: BatchedCompositionIR, inputs: dict[str, np.ndarray]) -> int:
    metadata_limit = int(ir.metadata.get("lca_max_steps", 0) or 0)
    input_limit = 0
    if "cue" in inputs and np.size(inputs["cue"]):
        input_limit = int(np.ceil(np.max(inputs["cue"])))
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
    for spec in ir.params:
        candidate_names = (spec.name,) + spec.aliases
        for name in candidate_names:
            if name in row:
                canonical[spec.name] = float(row[name])
        for name, value in row.items():
            if isinstance(name, str) and any(name.endswith(f".{candidate}") for candidate in candidate_names):
                canonical[spec.name] = float(value)
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
