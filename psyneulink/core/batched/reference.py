from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from psyneulink.core.batched.ir import BatchedCompositionIR, BatchedSimulationResult
from psyneulink.core.batched.graph import DDM_MODEL, STABILITY_FLEXIBILITY_MODEL, projection_inputs


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

    values = _run_graph(ir, prepared_inputs, params, num_estimates, seed, common_random_numbers)

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
    if ir.graph is None:
        raise ValueError("Batched reference execution requires a graph IR.")

    values = {}
    for input_spec in ir.graph.inputs:
        raw_value = _extract_named_input(inputs, (input_spec.node,), fallback_first=len(ir.graph.inputs) == 1)
        coerced = _coerce_trials(raw_value, width=input_spec.width)
        values[input_spec.node] = coerced[:, 0] if input_spec.width == 1 else coerced

    prepared = _split_subject_trials(values, subject_slices)
    roles = ir.graph.metadata.get("stability_flexibility_roles", {})
    if ir.graph.fusion_kind == DDM_MODEL and ir.graph.inputs:
        prepared["stimulus"] = prepared[ir.graph.inputs[0].node]
    elif ir.graph.fusion_kind == STABILITY_FLEXIBILITY_MODEL and roles:
        prepared["task"] = prepared[roles["task_node"]]
        prepared["stimulus"] = prepared[roles["stimulus_node"]]
        prepared["cue"] = prepared[roles["cue_node"]]
        prepared["correct"] = prepared[roles["correct_node"]]
    return prepared


def _run_graph(
    ir: BatchedCompositionIR,
    inputs: dict[str, np.ndarray],
    params: list[dict[str, float]],
    num_estimates: int,
    seed,
    common_random_numbers: bool,
) -> np.ndarray:
    graph = ir.graph
    if graph is None:
        raise ValueError("Batched reference execution requires a graph IR.")

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
                            ir.max_steps,
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
    if node.component_type in {"TransferMechanism", "ProcessingMechanism"}:
        input_value = _node_input(graph, node, inputs, node_outputs, subject_idx, trial_idx)
        if node.function_type == "Linear":
            value = _param(params, node, "slope") * input_value + _param(params, node, "intercept")
        elif node.function_type == "Logistic":
            value = 1.0 / (1.0 + np.exp(-_param(params, node, "gain") * input_value))
        else:
            raise ValueError(f"Unsupported stateless function '{node.function_type}'.")
        return {_primary_output_port(node): np.asarray(value, dtype=np.float32).reshape(-1)}

    if node.component_type == "LCAMechanism":
        task = _node_input(graph, node, inputs, node_outputs, subject_idx, trial_idx)
        termination_input_node = node.attrs.get("termination_input_node")
        if termination_input_node is not None and termination_input_node in inputs:
            cue = float(np.asarray(inputs[termination_input_node][subject_idx, trial_idx]).reshape(-1)[0])
        else:
            cue = float(node.attrs.get("termination_threshold", 1.0))
        pre_key = f"{node.name}.pre"
        act_key = f"{node.name}.act"
        pre, act = _simulate_lca_cue_period(
            task=task,
            pre_activity=state[pre_key],
            activity=state[act_key],
            cue=cue,
            gain=_param(params, node, "gain"),
            leak=_param(params, node, "leak"),
            competition=_param(params, node, "competition"),
            self_excitation=_param(params, node, "self_excitation"),
            noise=_param(params, node, "noise"),
            time_step_size=_param(params, node, "time_step_size"),
            max_steps=max(int(np.ceil(cue)), 1),
            rng=rng,
        )
        state[pre_key] = pre
        state[act_key] = act
        return {_primary_output_port(node): act.astype(np.float32)}

    if node.component_type == "DDM":
        drift_input = float(_node_input(graph, node, inputs, node_outputs, subject_idx, trial_idx).reshape(-1)[0])
        decision, response_time = _simulate_ddm_trial(
            drift_input=drift_input,
            rate=_param(params, node, "rate"),
            noise=_param(params, node, "noise"),
            threshold=_param(params, node, "threshold"),
            non_decision_time=_param(params, node, "non_decision_time"),
            time_step_size=_param(params, node, "time_step_size"),
            starting_value=_param(params, node, "starting_value"),
            offset=_param(params, node, "offset"),
            max_steps=max_steps,
            rng=rng,
        )
        return {
            "DECISION_OUTCOME": np.asarray([decision], dtype=np.float32),
            "RESPONSE_TIME": np.asarray([response_time], dtype=np.float32),
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
    elif ir.graph is not None:
        for node in ir.graph.nodes:
            if node.component_type != "LCAMechanism":
                continue
            termination_input_node = node.attrs.get("termination_input_node")
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
