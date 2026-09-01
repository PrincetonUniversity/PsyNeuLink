"""CSI-only observed-history likelihood for a deterministic persistent LCA.

This is deliberately a specialization, not a general stateful-composition
contract.  The first kernel follows the participant's observed response-time
history with one deterministic LCA lane per parameter set and stores the
time-varying DDM drift for every trial.  The second kernel simulates all DDM
trials and estimates in parallel.  Simulated stopping times therefore affect
the current trial's histogram density but do not create artificial uncertainty
about the next LCA state.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np

from psyneulink.core.batched.backend.triton.cache import (
    interpret_scope,
    load_triton_kernel_module,
)
from psyneulink.core.batched.backend.triton.runtime import (
    _check_step_caps,
    _compiler_launch_options,
    _import_torch_triton,
    _normalize_launch_options,
    _report_truncation,
)
from psyneulink.core.batched.graph import COEVOLVING_GRAPH_FUSION
from psyneulink.core.batched.likelihood import histogram_log_likelihood
from psyneulink.core.batched.prep import (
    _dynamic_lca_effective_counts,
    normalize_parameter_sets,
    prepare_inputs,
    prepare_parameter_values,
)


_CSI_HISTORY_SOURCE = r'''
import triton
import triton.language as tl


@triton.jit
def _sigmoid(value):
    return 1.0 / (1.0 + tl.exp(-value))


@triton.jit
def _lca_step(
    input0, input1, pre0, pre1, act0, act1, initialized,
    gain, leak, competition, self_excitation, dt, noise,
    bias, x_0, scale, offset, active,
):
    noise_step = noise * tl.sqrt(dt)
    initial_act = scale * _sigmoid(gain * (noise_step + bias - x_0)) + offset
    initialize_sender = initialized == 0.0
    act0 = tl.where(initialize_sender, initial_act, act0)
    act1 = tl.where(initialize_sender, initial_act, act1)
    rec0 = self_excitation * act0 - competition * act1
    rec1 = -competition * act0 + self_excitation * act1
    pre0 = tl.where(
        active,
        pre0 + (input0 + rec0 - leak * pre0) * dt + noise_step,
        pre0,
    )
    pre1 = tl.where(
        active,
        pre1 + (input1 + rec1 - leak * pre1) * dt + noise_step,
        pre1,
    )
    act0 = tl.where(
        active,
        scale * _sigmoid(gain * (pre0 + bias - x_0)) + offset,
        act0,
    )
    act1 = tl.where(
        active,
        scale * _sigmoid(gain * (pre1 + bias - x_0)) + offset,
        act1,
    )
    initialized = tl.where(active, 1.0, initialized)
    return pre0, pre1, act0, act1, initialized


@triton.jit
def _csi_drift(s0, s1, s2, s3, c0, c1, correct):
    a = _sigmoid((s0 - s1) + 4.0 * c0 - 4.0)
    b = _sigmoid((s1 - s0) + 4.0 * c0 - 4.0)
    c = _sigmoid((s2 - s3) + 4.0 * c1 - 4.0)
    d = _sigmoid((s3 - s2) + 4.0 * c1 - 4.0)
    positive = _sigmoid(a - b + c - d)
    negative = _sigmoid(-a + b - c + d)
    return (positive - negative) * correct


@triton.jit(
    do_not_specialize=['num_trials', 'onset_steps'],
    do_not_specialize_on_alignment=['num_trials', 'onset_steps'],
)
def pnl_csi_deterministic_history_kernel(
    task, stimulus, correct, effective_steps, observed_steps,
    gain, leak, competition, self_excitation, lca_dt, lca_noise,
    bias, x_0, scale, offset,
    drift_out, state_out,
    num_params: tl.constexpr, num_trials, onset_steps,
    MAX_STEPS: tl.constexpr, BLOCK: tl.constexpr,
):
    param_idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = param_idx < num_params
    pre0 = tl.zeros((BLOCK,), tl.float32)
    pre1 = tl.zeros((BLOCK,), tl.float32)
    act0 = tl.zeros((BLOCK,), tl.float32)
    act1 = tl.zeros((BLOCK,), tl.float32)
    initialized = tl.zeros((BLOCK,), tl.float32)

    trial_idx = 0
    while trial_idx < num_trials:
        lane = param_idx * num_trials + trial_idx
        task0 = tl.load(task + trial_idx * 2)
        task1 = tl.load(task + trial_idx * 2 + 1)
        settle_steps = tl.load(effective_steps + lane, mask=mask, other=0)
        gain_value = tl.load(gain + lane, mask=mask, other=1.0)
        leak_value = tl.load(leak + lane, mask=mask, other=0.0)
        competition_value = tl.load(competition + lane, mask=mask, other=0.0)
        self_excitation_value = tl.load(
            self_excitation + lane, mask=mask, other=0.0
        )
        dt_value = tl.load(lca_dt + lane, mask=mask, other=0.01)
        noise_value = tl.load(lca_noise + lane, mask=mask, other=0.0)
        bias_value = tl.load(bias + lane, mask=mask, other=0.0)
        x_0_value = tl.load(x_0 + lane, mask=mask, other=0.0)
        scale_value = tl.load(scale + lane, mask=mask, other=1.0)
        offset_value = tl.load(offset + lane, mask=mask, other=0.0)

        # The scheduler's first DDM execution shares the pass on which a
        # positive LCA count becomes finished.  Integrate only the passes before
        # that overlap here; DDM path step zero performs the final onset step.
        pre_ddm_steps = tl.maximum(settle_steps - 1, 0)
        settle_step = 0
        block_pre_ddm_steps = tl.max(tl.where(mask, pre_ddm_steps, 0))
        while settle_step < block_pre_ddm_steps:
            active = mask & (settle_step < pre_ddm_steps)
            after_onset = settle_step >= onset_steps
            input0 = tl.where(after_onset, task0, 0.0)
            input1 = tl.where(after_onset, task1, 0.0)
            pre0, pre1, act0, act1, initialized = _lca_step(
                input0, input1, pre0, pre1, act0, act1, initialized,
                gain_value, leak_value, competition_value,
                self_excitation_value, dt_value, noise_value,
                bias_value, x_0_value, scale_value, offset_value, active,
            )
            settle_step += 1

        canonical_pre0 = pre0
        canonical_pre1 = pre1
        canonical_act0 = act0
        canonical_act1 = act1
        canonical_initialized = initialized
        required_steps = tl.load(observed_steps + lane, mask=mask, other=0)

        path_pre0 = pre0
        path_pre1 = pre1
        path_act0 = act0
        path_act1 = act1
        path_initialized = initialized
        s0 = tl.load(stimulus + trial_idx * 4)
        s1 = tl.load(stimulus + trial_idx * 4 + 1)
        s2 = tl.load(stimulus + trial_idx * 4 + 2)
        s3 = tl.load(stimulus + trial_idx * 4 + 3)
        correct_value = tl.load(correct + trial_idx)

        ddm_step = 0
        first_ddm_pass = tl.maximum(settle_steps - 1, 0)
        while ddm_step < MAX_STEPS:
            ddm_after_onset = first_ddm_pass + ddm_step >= onset_steps
            ddm_input0 = tl.where(ddm_after_onset, task0, 0.0)
            ddm_input1 = tl.where(ddm_after_onset, task1, 0.0)
            path_pre0, path_pre1, path_act0, path_act1, path_initialized = _lca_step(
                ddm_input0, ddm_input1,
                path_pre0, path_pre1, path_act0, path_act1, path_initialized,
                gain_value, leak_value, competition_value,
                self_excitation_value, dt_value, noise_value,
                bias_value, x_0_value, scale_value, offset_value, mask,
            )
            drift = _csi_drift(
                s0, s1, s2, s3, path_act0, path_act1, correct_value
            )
            drift_index = lane * MAX_STEPS + ddm_step
            tl.store(drift_out + drift_index, drift, mask=mask)
            observed_here = mask & (
                (required_steps == ddm_step + 1)
                | (
                    (required_steps == 0)
                    & (settle_steps > 0)
                    & (ddm_step == 0)
                )
            )
            canonical_pre0 = tl.where(observed_here, path_pre0, canonical_pre0)
            canonical_pre1 = tl.where(observed_here, path_pre1, canonical_pre1)
            canonical_act0 = tl.where(observed_here, path_act0, canonical_act0)
            canonical_act1 = tl.where(observed_here, path_act1, canonical_act1)
            canonical_initialized = tl.where(
                observed_here, path_initialized, canonical_initialized
            )
            ddm_step += 1

        pre0 = canonical_pre0
        pre1 = canonical_pre1
        act0 = canonical_act0
        act1 = canonical_act1
        initialized = canonical_initialized
        state_base = lane * 5
        tl.store(state_out + state_base, pre0, mask=mask)
        tl.store(state_out + state_base + 1, pre1, mask=mask)
        tl.store(state_out + state_base + 2, act0, mask=mask)
        tl.store(state_out + state_base + 3, act1, mask=mask)
        tl.store(state_out + state_base + 4, initialized, mask=mask)
        trial_idx += 1


@triton.jit(
    do_not_specialize=['SEED', 'csi_time_per_step'],
    do_not_specialize_on_alignment=['SEED', 'csi_time_per_step'],
)
def pnl_csi_deterministic_ddm_kernel(
    drift, csi_steps, include, rate, noise, threshold, threshold_collapse,
    non_decision_time, time_step_size, starting_value, ddm_offset,
    out, truncated, csi_time_per_step,
    total_lanes: tl.constexpr, num_trials: tl.constexpr,
    num_estimates: tl.constexpr, MAX_STEPS: tl.constexpr,
    COMMON_RANDOM: tl.constexpr, SEED, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_lanes
    estimate_idx = offsets % num_estimates
    tmp = offsets // num_estimates
    trial_idx = tmp % num_trials
    param_idx = tmp // num_trials
    lane = param_idx * num_trials + trial_idx
    trial_included = tl.load(include + trial_idx, mask=mask, other=0) != 0
    active_mask = mask & trial_included

    if COMMON_RANDOM:
        random_base = (
            (estimate_idx * num_trials + trial_idx).to(tl.int64) * 4294967296
        )
    else:
        random_base = (
            ((param_idx * num_estimates + estimate_idx) * num_trials + trial_idx)
            .to(tl.int64) * 4294967296
        )

    rate_value = tl.load(rate + lane, mask=mask, other=1.0)
    noise_value = tl.load(noise + lane, mask=mask, other=0.0)
    threshold_value = tl.load(threshold + lane, mask=mask, other=1.0)
    collapse_value = tl.load(threshold_collapse + lane, mask=mask, other=0.0)
    ndt_value = tl.load(non_decision_time + lane, mask=mask, other=0.0)
    dt_value = tl.load(time_step_size + lane, mask=mask, other=0.01)
    value = tl.load(starting_value + lane, mask=mask, other=0.0)
    offset_value = tl.load(ddm_offset + lane, mask=mask, other=0.0)
    csi_value = tl.load(csi_steps + lane, mask=mask, other=0).to(tl.float32)
    finished = ~active_mask
    steps = tl.zeros((BLOCK,), tl.float32)

    step = 0
    while step < MAX_STEPS:
        active = active_mask & ~finished
        draw = tl.randn(SEED, random_base + step)
        drift_value = tl.load(
            drift + lane * MAX_STEPS + step, mask=mask, other=0.0
        )
        boundary = threshold_value + collapse_value * step
        tolerance = tl.maximum(1.0e-7, threshold_value * 1.0e-6)
        updated = (
            value + rate_value * drift_value * dt_value
            + noise_value * tl.sqrt(dt_value) * draw + offset_value
        )
        updated = tl.minimum(tl.maximum(updated, -boundary), boundary)
        value = tl.where(active, updated, value)
        steps = tl.where(active, steps + 1.0, steps)
        finished = finished | (
            active & (tl.abs(value) + tolerance >= boundary)
        )
        step += 1

    decision = tl.where(value > 0.0, 1.0, 0.0)
    response_time = (
        ndt_value + steps * dt_value + csi_value * csi_time_per_step
    )
    output_base = offsets * 2
    tl.store(out + output_base, decision, mask=mask)
    tl.store(out + output_base + 1, response_time, mask=mask)
    tl.store(
        truncated + offsets,
        (active_mask & ~finished).to(tl.uint8),
        mask=mask,
    )
'''


def _base_name(name: str) -> str:
    return re.sub(r"-\d+$", "", str(name))


def _unique_node(graph, name: str):
    matches = tuple(node for node in graph.nodes if _base_name(node.name) == name)
    if len(matches) != 1:
        raise ValueError(
            f"CSI deterministic-history likelihood requires one '{name}' node; "
            f"found {len(matches)}."
        )
    return matches[0]


def _require_affine(node, rows, defaults, expected=None) -> None:
    if expected is None:
        expected = {
            "slope": 1.0,
            "intercept": 0.0,
            "scale": 1.0,
            "offset": 0.0,
        }
    if node.function_type != "Linear" or set(node.params) != set(expected):
        raise ValueError(
            f"CSI deterministic-history likelihood requires an identity Linear "
            f"transform for '{node.name}'."
        )
    for argument, expected_value in expected.items():
        parameter = node.params[argument]
        for row in rows:
            value = row.get(parameter, defaults[parameter])
            array = np.asarray(value)
            if array.size != 1 or float(array.reshape(-1)[0]) != expected_value:
                raise ValueError(
                    f"CSI deterministic-history likelihood requires "
                    f"'{parameter}'={expected_value}."
                )


def _projection(graph, sender, sender_port, receiver):
    matches = tuple(
        projection
        for projection in graph.projections
        if _base_name(projection.sender) == sender
        and projection.sender_port == sender_port
        and _base_name(projection.receiver) == receiver
    )
    if len(matches) != 1:
        raise ValueError(
            "CSI deterministic-history likelihood requires exactly one "
            f"{sender}.{sender_port} -> {receiver} projection; found "
            f"{len(matches)}."
        )
    return matches[0]


def _require_projection_matrix(projection, expected) -> None:
    actual = np.asarray(projection.matrix, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError(
            "CSI deterministic-history likelihood found an unsupported matrix "
            f"for {projection.sender} -> {projection.receiver}."
        )


def _dense_parameter(ir, buffers, name: str, num_trials: int) -> np.ndarray:
    index = next(
        (index for index, spec in enumerate(ir.params) if spec.name == name),
        None,
    )
    if index is None:
        raise ValueError(f"CSI deterministic-history parameter '{name}' is missing.")
    value = buffers[index]
    if value.ndim == 1:
        return np.broadcast_to(value[:, None], (len(value), num_trials)).copy()
    if value.ndim == 3 and value.shape[1] == 1:
        return np.asarray(value[:, 0, :], dtype=np.float32)
    raise ValueError(
        f"CSI deterministic-history parameter '{name}' has unsupported shape "
        f"{value.shape}."
    )


def _observed_endpoint_steps(
    response_time,
    non_decision_time,
    csi_steps,
    csi_time_per_step,
    ddm_time_step,
) -> np.ndarray:
    """Invert the fp32 response readout without boundary roundoff steps."""

    onset_time = (
        np.asarray(non_decision_time, dtype=np.float64)
        + np.asarray(csi_steps, dtype=np.float64) * csi_time_per_step
    )
    dt = np.asarray(ddm_time_step, dtype=np.float64)
    response = np.asarray(response_time, dtype=np.float64)
    ratio = np.maximum((response - onset_time) / dt, 0.0)
    nearest = np.rint(ratio)
    reconstructed = onset_time + nearest * dt
    # A response emitted by the fp32 GPU readout may invert to just above an
    # integer even though it was produced by that exact scheduler step.  Snap
    # only within a few fp32 ULPs of the reconstructed RT; ordinary recorded
    # RTs and real parameter-dependent cell crossings retain ceil semantics.
    spacing = np.spacing(np.asarray(reconstructed, dtype=np.float32)).astype(
        np.float64
    )
    snap = np.abs(response - reconstructed) <= 8.0 * np.abs(spacing)
    return np.where(snap, nearest, np.ceil(ratio)).astype(np.int32)


def _validate_csi_contract(ir, rows):
    graph = ir.graph
    if graph is None or graph.fusion_kind != COEVOLVING_GRAPH_FUSION:
        raise ValueError(
            "CSI deterministic-history likelihood requires a co-evolving graph."
        )
    lca = _unique_node(graph, "Task Activations [C1, C2]")
    ddm = _unique_node(graph, "DDM")
    cue = _unique_node(graph, "Cue Stimulus Interval")
    task = _unique_node(graph, "Task Input")
    stimulus = _unique_node(graph, "Stimulus Input")
    correct = _unique_node(graph, "Correct Response")
    drift = _unique_node(graph, "Drift Rate Value")
    decision_gate = _unique_node(graph, "DECISION_GATE")
    response_gate = _unique_node(graph, "RESPONSE_GATE")
    csi_override = _unique_node(graph, "CSI Override")
    if (
        lca.component_type != "LCAMechanism"
        or lca.output_width != 2
        or ddm.component_type != "DDM"
        or drift.function_type != "UserDefinedFunction"
        or drift.input_width != 7
        or drift.output_width != 1
    ):
        raise ValueError("The compiled graph is not the authenticated CSI LCA/DDM shape.")
    if lca.attrs.get("termination_input_node") != cue.name:
        raise ValueError("CSI LCA termination must be driven by the cue interval node.")
    if not lca.attrs.get("initialize_noise_sender", False):
        raise ValueError(
            "CSI deterministic-history likelihood requires canonical LCA sender "
            "initialization."
        )
    defaults = dict(ir.param_defaults)
    for node in (task, stimulus, correct, decision_gate, response_gate):
        _require_affine(node, rows, defaults)
    onset_steps = int(task.attrs.get("onset_step", 0))
    if onset_steps < 0:
        raise ValueError("CSI task onset must be nonnegative.")
    if tuple(task.attrs.get("integrator_pre", ())) != (1.0, 0.0):
        raise ValueError(
            "CSI task input requires the canonical held-value integrator."
        )
    _require_affine(
        csi_override,
        rows,
        defaults,
        {
            "slope": 1.0,
            "intercept": float(onset_steps),
            "scale": 1.0,
            "offset": 0.0,
        },
    )
    if (
        len(graph.modulations) != 1
        or _base_name(graph.modulations[0].controller) != "CSI Override"
        or _base_name(graph.modulations[0].source) != "Cue Stimulus Interval"
        or _base_name(graph.modulations[0].target)
        != "Task Activations [C1, C2]"
        or graph.modulations[0].target_parameter != "termination_threshold"
        or graph.modulations[0].mode != "OVERRIDE"
    ):
        raise ValueError(
            "CSI deterministic-history likelihood requires the canonical CSI "
            "termination override."
        )
    actual_states = {
        (
            _base_name(state.node),
            state.name.removeprefix(state.node),
            state.width,
        )
        for state in graph.states
    }
    expected_states = {
        ("Task Activations [C1, C2]", ".pre", 2),
        ("Task Activations [C1, C2]", ".act", 2),
        ("Task Activations [C1, C2]", ".initialized", 1),
    }
    if actual_states != expected_states:
        raise ValueError(
            "CSI deterministic-history likelihood requires only the canonical "
            "persistent LCA state."
        )
    if (
        len(graph.folded_affine_controls) != 1
        or _base_name(graph.folded_affine_controls[0].target) != "DDM"
        or graph.folded_affine_controls[0].target_parameter != "threshold"
        or graph.folded_affine_controls[0].base_parameter
        != ddm.params["threshold"]
        or graph.folded_affine_controls[0].delta_parameter
        != ddm.params["threshold_collapse"]
    ):
        raise ValueError(
            "CSI deterministic-history likelihood requires the canonical folded "
            "DDM threshold controller."
        )

    expected_inputs = {
        "Correct Response",
        "Cue Stimulus Interval",
        "Stimulus Input",
        "Task Input",
    }
    if {_base_name(spec.node) for spec in graph.inputs} != expected_inputs:
        raise ValueError(
            "CSI deterministic-history likelihood requires the four canonical "
            "CSI inputs."
        )
    expected_outputs = (
        ("DECISION_GATE", "OutputPort-0"),
        ("RESPONSE_GATE", "OutputPort-0"),
    )
    if tuple(
        (_base_name(spec.node), spec.port) for spec in graph.outputs
    ) != expected_outputs:
        raise ValueError(
            "CSI deterministic-history likelihood requires decision and "
            "response-time gate outputs in canonical order."
        )

    matrices = (
        (_projection(graph, "Task Input", "RESULT", "Task Activations [C1, C2]"), np.eye(2)),
        (_projection(graph, "Correct Response", "OutputPort-0", "Drift Rate Value"), [[0, 0, 0, 0, 0, 0, 1]]),
        (_projection(graph, "Stimulus Input", "OutputPort-0", "Drift Rate Value"), np.pad(np.eye(4), ((0, 0), (0, 3)))),
        (_projection(graph, "Task Activations [C1, C2]", "RESULT", "Drift Rate Value"), [[0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0]]),
        (_projection(graph, "Drift Rate Value", "OutputPort-0", "DDM"), [[1]]),
        (_projection(graph, "DDM", "DECISION_OUTCOME", "DECISION_GATE"), [[1]]),
        (_projection(graph, "DDM", "RESPONSE_TIME", "RESPONSE_GATE"), [[1]]),
    )
    for projection, expected in matrices:
        _require_projection_matrix(projection, expected)
    csi_response_projection = _projection(
        graph,
        "Cue Stimulus Interval",
        "OutputPort-0",
        "RESPONSE_GATE",
    )
    csi_matrix = np.asarray(csi_response_projection.matrix, dtype=float)
    if (
        csi_matrix.shape != (1, 1)
        or not np.isfinite(csi_matrix[0, 0])
        or csi_matrix[0, 0] <= 0.0
    ):
        raise ValueError(
            "CSI response-time projection must be one finite positive scalar."
        )
    if len(graph.projections) != len(matrices) + 1:
        raise ValueError(
            "CSI deterministic-history likelihood found unsupported extra "
            "pathway projections."
        )
    return (
        lca,
        ddm,
        cue,
        task,
        stimulus,
        correct,
        onset_steps,
        float(csi_matrix[0, 0]),
    )


def run_csi_deterministic_history_likelihood(
    ir,
    inputs,
    parameter_sets,
    num_estimates: int,
    data,
    categorical_dims,
    *,
    bins: int,
    bin_range: Sequence | None,
    smoothing_sigma: float,
    pseudocount: float,
    categorical_cardinalities,
    include_mask,
    seed,
    common_random_numbers: bool,
    strict_truncation: bool,
    component_bindings,
    launch_options: Mapping | None,
    return_debug: bool = False,
):
    """Run the authenticated CSI deterministic-history GPU specialization."""

    if num_estimates < 1:
        raise ValueError("num_estimates must be positive.")
    rows = normalize_parameter_sets(parameter_sets, ir)
    (
        lca,
        ddm,
        cue,
        task_node,
        stimulus_node,
        correct_node,
        onset_steps,
        csi_time_per_step,
    ) = (
        _validate_csi_contract(ir, rows)
    )
    prepared = prepare_inputs(
        ir,
        inputs,
        subject_slices=None,
        parameter_sets=rows,
        component_bindings=component_bindings,
    )
    first_input = next(iter(prepared.values()))
    num_subjects, num_trials = first_input.shape[:2]
    if num_subjects != 1:
        raise NotImplementedError(
            "CSI deterministic-history likelihood currently accepts one subject."
        )
    exp_data = np.asarray(data, dtype=float)
    if exp_data.shape != (num_trials, 2):
        raise ValueError(
            "CSI deterministic-history data must have shape "
            f"({num_trials}, 2), got {exp_data.shape}."
        )
    if not np.all(np.isfinite(exp_data)):
        raise ValueError("CSI deterministic-history data must be finite.")
    from psyneulink.core.batched.likelihood import _as_categorical_mask

    categorical = _as_categorical_mask(categorical_dims, 2)
    if categorical.shape != (2,) or not np.array_equal(categorical, [True, False]):
        raise ValueError(
            "CSI deterministic-history likelihood requires categorical choice "
            "and continuous response time."
        )
    include = (
        np.ones(num_trials, dtype=bool)
        if include_mask is None
        else np.asarray(include_mask, dtype=bool).reshape(-1)
    )
    if include.shape != (num_trials,):
        raise ValueError("include_mask must have one value per CSI trial.")

    buffers, _ = prepare_parameter_values(
        ir,
        rows,
        num_subjects=1,
        num_trials=num_trials,
    )
    num_params = len(rows)
    dense = {
        spec.name: _dense_parameter(ir, buffers, spec.name, num_trials)
        for spec in ir.params
    }
    lca_noise = dense[lca.params["noise"]]
    if np.any(lca_noise != 0.0):
        raise ValueError(
            "CSI deterministic-history likelihood requires deterministic LCA noise=0."
        )

    cue_values = np.asarray(prepared[cue.name], dtype=np.float32)
    effective = _dynamic_lca_effective_counts(ir, cue.name, cue_values, rows)
    effective = np.asarray(effective, dtype=np.int32).reshape(num_params, num_trials)
    csi_steps = effective - onset_steps
    if np.any(csi_steps < 0):
        raise ValueError("CSI effective duration cannot precede task onset.")

    ddm_dt = dense[ddm.params["time_step_size"]]
    ndt = dense[ddm.params["non_decision_time"]]
    if np.any(~np.isfinite(ddm_dt)) or np.any(ddm_dt <= 0.0):
        raise ValueError(
            "CSI deterministic-history DDM time_step_size must be finite and "
            "positive."
        )
    observed_steps = _observed_endpoint_steps(
        exp_data[None, :, 1],
        ndt,
        csi_steps,
        csi_time_per_step,
        ddm_dt,
    )
    if np.any(observed_steps > ir.max_steps):
        maximum = int(observed_steps.max())
        raise ValueError(
            f"Observed CSI history requires {maximum} DDM steps but max_steps="
            f"{ir.max_steps}."
        )

    torch, triton = _import_torch_triton(False)
    if not torch.cuda.is_available():
        raise RuntimeError("CSI deterministic-history likelihood requires CUDA.")
    launch = _normalize_launch_options(launch_options, interpret=False)
    _check_step_caps(max_steps=ir.max_steps)
    device = "cuda"

    def tensor(value, *, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=device).contiguous()

    task = tensor(prepared[task_node.name][0])
    stimulus = tensor(prepared[stimulus_node.name][0])
    correct = tensor(prepared[correct_node.name][0])
    effective_tensor = tensor(effective, dtype=torch.int32)
    observed_tensor = tensor(observed_steps, dtype=torch.int32)
    csi_tensor = tensor(csi_steps, dtype=torch.int32)
    include_tensor = tensor(include.astype(np.uint8), dtype=torch.uint8)

    def parameter(name):
        return tensor(dense[name])

    drift_paths = torch.empty(
        (num_params, num_trials, ir.max_steps),
        dtype=torch.float32,
        device=device,
    )
    history_states = torch.empty(
        (num_params, num_trials, 5), dtype=torch.float32, device=device
    )
    values = torch.empty(
        (num_params, num_trials, num_estimates, 2),
        dtype=torch.float32,
        device=device,
    )
    truncated = torch.empty(
        (num_params, num_trials, num_estimates),
        dtype=torch.uint8,
        device=device,
    )

    with interpret_scope(False):
        module = load_triton_kernel_module(
            _CSI_HISTORY_SOURCE,
            module_kind="csi_deterministic_history",
            model_kind=ir.model_kind,
            interpret=False,
        )
        history_block = min(128, triton.next_power_of_2(num_params))
        module.pnl_csi_deterministic_history_kernel[
            (triton.cdiv(num_params, history_block),)
        ](
            task,
            stimulus,
            correct,
            effective_tensor,
            observed_tensor,
            parameter(lca.params["gain"]),
            parameter(lca.params["leak"]),
            parameter(lca.params["competition"]),
            parameter(lca.params["self_excitation"]),
            parameter(lca.params["time_step_size"]),
            parameter(lca.params["noise"]),
            parameter(lca.params["bias"]),
            parameter(lca.params["x_0"]),
            parameter(lca.params["scale"]),
            parameter(lca.params["offset"]),
            drift_paths,
            history_states,
            num_params=num_params,
            num_trials=num_trials,
            onset_steps=onset_steps,
            MAX_STEPS=ir.max_steps,
            BLOCK=history_block,
            num_warps=1,
        )

        total_lanes = num_params * num_trials * num_estimates
        block = launch["block_size"]
        module.pnl_csi_deterministic_ddm_kernel[
            (triton.cdiv(total_lanes, block),)
        ](
            drift_paths,
            csi_tensor,
            include_tensor,
            parameter(ddm.params["rate"]),
            parameter(ddm.params["noise"]),
            parameter(ddm.params["threshold"]),
            parameter(ddm.params["threshold_collapse"]),
            parameter(ddm.params["non_decision_time"]),
            parameter(ddm.params["time_step_size"]),
            parameter(ddm.params["starting_value"]),
            parameter(ddm.params["offset"]),
            values,
            truncated,
            csi_time_per_step,
            total_lanes=total_lanes,
            num_trials=num_trials,
            num_estimates=num_estimates,
            MAX_STEPS=ir.max_steps,
            COMMON_RANDOM=bool(common_random_numbers),
            SEED=0 if seed is None else int(seed),
            BLOCK=block,
            **_compiler_launch_options(launch),
        )

    nonfinite = int((~torch.isfinite(values)).sum().item())
    if nonfinite:
        raise FloatingPointError(
            f"CSI deterministic-history simulation produced {nonfinite} "
            "non-finite outcomes."
        )
    included_lanes = include_tensor[None, :, None].expand_as(truncated) != 0
    included_count = int(included_lanes.sum().item())
    truncated_count = int(((truncated != 0) & included_lanes).sum().item())
    fraction = 0.0 if included_count == 0 else truncated_count / included_count
    _report_truncation({ddm.name: fraction}, ir.max_steps, strict_truncation)

    score = histogram_log_likelihood(
        values,
        exp_data,
        categorical_dims,
        bins=bins,
        bin_range=bin_range,
        smoothing_sigma=smoothing_sigma,
        pseudocount=pseudocount,
        categorical_cardinalities=categorical_cardinalities,
        include_mask=include,
    )
    score_array = np.asarray(score, dtype=float).reshape(-1)
    result = float(score_array[0]) if num_params == 1 else score_array
    if return_debug:
        return result, {
            "drift_paths": drift_paths,
            "history_states": history_states,
            "observed_steps": observed_steps,
            "csi_steps": csi_steps,
            "values": values,
            "truncation_fraction": fraction,
        }
    return result


__all__ = ["run_csi_deterministic_history_likelihood"]
