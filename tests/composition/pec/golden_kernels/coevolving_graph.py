import triton
import triton.language as tl


@triton.jit
def _pnl_triton_linear(x, slope, intercept):
    return slope * x + intercept


@triton.jit
def _pnl_triton_projection_term(x, coefficient):
    return x * coefficient


@triton.jit
def _pnl_triton_lca_width2_recurrence(input0, input1, pre0, pre1, act0, act1, active, gain, leak, competition, self_excitation, noise, dt, n0, n1):
    sqrt_dt = tl.sqrt(dt)
    rec0 = self_excitation * act0 - competition * act1
    rec1 = -competition * act0 + self_excitation * act1
    pre0 = tl.where(active, pre0 + (input0 + rec0 - leak * pre0) * dt + noise * sqrt_dt * n0, pre0)
    pre1 = tl.where(active, pre1 + (input1 + rec1 - leak * pre1) * dt + noise * sqrt_dt * n1, pre1)
    act0 = tl.where(active, 1.0 / (1.0 + tl.exp(-gain * pre0)), act0)
    act1 = tl.where(active, 1.0 / (1.0 + tl.exp(-gain * pre1)), act1)
    return (pre0, pre1, act0, act1)


@triton.jit
def _pnl_triton_lca_width2_step(input0, input1, pre0, pre1, act0, act1, finished, gain, leak, competition, self_excitation, noise, dt, seed, random_base, step, stream0: tl.constexpr, stream1: tl.constexpr, lca_max_steps: tl.constexpr):
    active = finished == 0.0
    n0 = tl.randn(seed, random_base + stream0 * lca_max_steps + step)
    n1 = tl.randn(seed, random_base + stream1 * lca_max_steps + step)
    pre0, pre1, act0, act1 = _pnl_triton_lca_width2_recurrence(input0, input1, pre0, pre1, act0, act1, active, gain, leak, competition, self_excitation, noise, dt, n0, n1)
    return (pre0, pre1, act0, act1)


@triton.jit
def _pnl_triton_instance_Drift_Rate_Value(x0, x1, x2, x3, x4, x5, x6):
    a = 1.0 / (1.0 + tl.exp(-(x0 - x1 + 4.0 * x4 - 4.0)))
    b = 1.0 / (1.0 + tl.exp(-(x1 - x0 + 4.0 * x4 - 4.0)))
    c = 1.0 / (1.0 + tl.exp(-(x2 - x3 + 4.0 * x5 - 4.0)))
    d = 1.0 / (1.0 + tl.exp(-(x3 - x2 + 4.0 * x5 - 4.0)))
    pos = 1.0 / (1.0 + tl.exp(-(a - b + c - d)))
    neg = 1.0 / (1.0 + tl.exp(-(-a + b - c + d)))
    return (pos - neg) * x6


@triton.jit
def _pnl_triton_ddm_update(value, steps, finished, drift, rate, noise, threshold, threshold_collapse, time_step_size, offset, draw, step):
    sqrt_dt = tl.sqrt(time_step_size)
    thr = threshold + threshold_collapse * step
    boundary_tolerance = tl.maximum(1e-07, threshold * 1e-06)
    active = (finished == 0.0) & (tl.abs(value) + boundary_tolerance < thr)
    updated = value + rate * drift * time_step_size + noise * sqrt_dt * draw
    updated = tl.minimum(tl.maximum(updated + offset, -thr), thr)
    value = tl.where(active, updated, value)
    steps = tl.where(active, steps + 1.0, steps)
    finished = tl.where(tl.abs(value) + boundary_tolerance >= thr, 1.0, finished)
    return (value, steps, finished)


@triton.jit
def _pnl_triton_ddm_step(value, steps, finished, drift, rate, noise, threshold, threshold_collapse, time_step_size, offset, seed, rng_base, step, start):
    active_time = step >= start
    draw = tl.randn(seed, rng_base + step)
    new_value, new_steps, new_finished = _pnl_triton_ddm_update(value, steps, finished, drift, rate, noise, threshold, threshold_collapse, time_step_size, offset, draw, tl.maximum(step - start, 0))
    value = tl.where(active_time, new_value, value)
    steps = tl.where(active_time, new_steps, steps)
    finished = tl.where(active_time, new_finished, finished)
    return (value, steps, finished)


@triton.jit
def pnl_batched_coevolving_graph_kernel(
    input_0,
    input_1,
    input_2,
    input_3,
    param_0,
    param_1,
    param_2,
    param_3,
    param_4,
    param_5,
    param_6,
    param_7,
    param_8,
    param_9,
    param_10,
    param_11,
    param_12,
    param_13,
    param_14,
    param_15,
    param_16,
    param_17,
    param_18,
    param_19,
    param_20,
    param_21,
    param_22,
    param_23,
    param_24,
    param_25,
    out,
    diag,
    total_lanes: tl.constexpr,
    num_subjects: tl.constexpr,
    num_estimates: tl.constexpr,
    num_trials,
    LCA_MAX_STEPS: tl.constexpr,
    MAX_STEPS: tl.constexpr,
    COMMON_RANDOM: tl.constexpr,
    SEED: tl.constexpr,
    BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_lanes
    estimate_idx = offsets % num_estimates
    tmp = offsets // num_estimates
    trial_idx = tmp % num_trials
    tmp = tmp // num_trials
    subject_idx = tmp % num_subjects
    param_idx = tmp // num_subjects

    param_0_value = tl.load(param_0 + param_idx, mask=mask, other=1.0)
    param_1_value = tl.load(param_1 + param_idx, mask=mask, other=0.0)
    param_2_value = tl.load(param_2 + param_idx, mask=mask, other=1.0)
    param_3_value = tl.load(param_3 + param_idx, mask=mask, other=0.0)
    param_4_value = tl.load(param_4 + param_idx, mask=mask, other=0.0)
    param_5_value = tl.load(param_5 + param_idx, mask=mask, other=0.0)
    param_6_value = tl.load(param_6 + param_idx, mask=mask, other=1.0)
    param_7_value = tl.load(param_7 + param_idx, mask=mask, other=0.0)
    param_8_value = tl.load(param_8 + param_idx, mask=mask, other=10.0)
    param_9_value = tl.load(param_9 + param_idx, mask=mask, other=7.0)
    param_10_value = tl.load(param_10 + param_idx, mask=mask, other=3.0)
    param_11_value = tl.load(param_11 + param_idx, mask=mask, other=0.0)
    param_12_value = tl.load(param_12 + param_idx, mask=mask, other=0.0)
    param_13_value = tl.load(param_13 + param_idx, mask=mask, other=0.01)
    param_14_value = tl.load(param_14 + param_idx, mask=mask, other=1.0)
    param_15_value = tl.load(param_15 + param_idx, mask=mask, other=0.0)
    param_16_value = tl.load(param_16 + param_idx, mask=mask, other=0.06)
    param_17_value = tl.load(param_17 + param_idx, mask=mask, other=-0.001)
    param_18_value = tl.load(param_18 + param_idx, mask=mask, other=0.3)
    param_19_value = tl.load(param_19 + param_idx, mask=mask, other=0.01)
    param_20_value = tl.load(param_20 + param_idx, mask=mask, other=0.0)
    param_21_value = tl.load(param_21 + param_idx, mask=mask, other=0.0)
    param_22_value = tl.load(param_22 + param_idx, mask=mask, other=1.0)
    param_23_value = tl.load(param_23 + param_idx, mask=mask, other=0.0)
    param_24_value = tl.load(param_24 + param_idx, mask=mask, other=1.0)
    param_25_value = tl.load(param_25 + param_idx, mask=mask, other=0.0)

    n_Task_Activations__C1__C2__pre_0 = tl.full((BLOCK,), 0.0, tl.float32)
    n_Task_Activations__C1__C2__pre_1 = tl.full((BLOCK,), 0.0, tl.float32)
    n_Task_Activations__C1__C2__act_0 = tl.full((BLOCK,), 0.0, tl.float32)
    n_Task_Activations__C1__C2__act_1 = tl.full((BLOCK,), 0.0, tl.float32)

    trial_idx = 0
    while trial_idx < num_trials:
        random_stride = (2) * LCA_MAX_STEPS + (1) * MAX_STEPS
        if COMMON_RANDOM:
            random_base = ((subject_idx * num_estimates + estimate_idx) * num_trials + trial_idx) * random_stride
        else:
            random_base = (((param_idx * num_subjects + subject_idx) * num_estimates + estimate_idx) * num_trials + trial_idx) * random_stride

        n_DDM_value_0 = tl.full((BLOCK,), 0.0, tl.float32)
        n_DDM_steps_0 = tl.full((BLOCK,), 0.0, tl.float32)
        n_DDM_finished_0 = tl.full((BLOCK,), 0.0, tl.float32)

        n_Stimulus_Input_OutputPort_0_0 = _pnl_triton_linear(tl.load(input_0 + (subject_idx * num_trials + trial_idx) * 4 + 0, mask=mask, other=0.0), param_0_value, param_1_value)
        n_Stimulus_Input_OutputPort_0_1 = _pnl_triton_linear(tl.load(input_0 + (subject_idx * num_trials + trial_idx) * 4 + 1, mask=mask, other=0.0), param_0_value, param_1_value)
        n_Stimulus_Input_OutputPort_0_2 = _pnl_triton_linear(tl.load(input_0 + (subject_idx * num_trials + trial_idx) * 4 + 2, mask=mask, other=0.0), param_0_value, param_1_value)
        n_Stimulus_Input_OutputPort_0_3 = _pnl_triton_linear(tl.load(input_0 + (subject_idx * num_trials + trial_idx) * 4 + 3, mask=mask, other=0.0), param_0_value, param_1_value)

        n_Task_Input_RESULT_0 = _pnl_triton_linear((1.0 * tl.load(input_1 + (subject_idx * num_trials + trial_idx) * 2 + 0, mask=mask, other=0.0) + 0.0), param_2_value, param_3_value)
        n_Task_Input_RESULT_1 = _pnl_triton_linear((1.0 * tl.load(input_1 + (subject_idx * num_trials + trial_idx) * 2 + 1, mask=mask, other=0.0) + 0.0), param_2_value, param_3_value)

        n_Cue_Stimulus_Interval_OutputPort_0_0 = _pnl_triton_linear(tl.load(input_2 + (subject_idx * num_trials + trial_idx) * 1 + 0, mask=mask, other=0.0), param_4_value, param_5_value)

        n_Correct_Response_OutputPort_0_0 = _pnl_triton_linear(tl.load(input_3 + (subject_idx * num_trials + trial_idx) * 1 + 0, mask=mask, other=0.0), param_6_value, param_7_value)

        n_Task_Activations__C1__C2__projection_0_0 = _pnl_triton_projection_term(n_Task_Input_RESULT_0, 1.0)
        n_Task_Activations__C1__C2__projection_0_1 = _pnl_triton_projection_term(n_Task_Input_RESULT_1, 1.0)

        n_Task_Activations__C1__C2__input_0 = (n_Task_Activations__C1__C2__projection_0_0)
        n_Task_Activations__C1__C2__input_1 = (n_Task_Activations__C1__C2__projection_0_1)

        n_Drift_Rate_Value_projection_0_0 = _pnl_triton_projection_term(n_Stimulus_Input_OutputPort_0_0, 1.0)
        n_Drift_Rate_Value_projection_0_1 = _pnl_triton_projection_term(n_Stimulus_Input_OutputPort_0_1, 1.0)
        n_Drift_Rate_Value_projection_0_2 = _pnl_triton_projection_term(n_Stimulus_Input_OutputPort_0_2, 1.0)
        n_Drift_Rate_Value_projection_0_3 = _pnl_triton_projection_term(n_Stimulus_Input_OutputPort_0_3, 1.0)
        n_Drift_Rate_Value_projection_0_4 = tl.zeros((BLOCK,), dtype=tl.float32)
        n_Drift_Rate_Value_projection_0_5 = tl.zeros((BLOCK,), dtype=tl.float32)
        n_Drift_Rate_Value_projection_0_6 = tl.zeros((BLOCK,), dtype=tl.float32)

        n_Drift_Rate_Value_projection_1_0 = tl.zeros((BLOCK,), dtype=tl.float32)
        n_Drift_Rate_Value_projection_1_1 = tl.zeros((BLOCK,), dtype=tl.float32)
        n_Drift_Rate_Value_projection_1_2 = tl.zeros((BLOCK,), dtype=tl.float32)
        n_Drift_Rate_Value_projection_1_3 = tl.zeros((BLOCK,), dtype=tl.float32)
        n_Drift_Rate_Value_projection_1_4 = tl.zeros((BLOCK,), dtype=tl.float32)
        n_Drift_Rate_Value_projection_1_5 = tl.zeros((BLOCK,), dtype=tl.float32)
        n_Drift_Rate_Value_projection_1_6 = _pnl_triton_projection_term(n_Correct_Response_OutputPort_0_0, 1.0)

        for step in tl.range(0, MAX_STEPS, 1, loop_unroll_factor=1):
            n_Task_Activations__C1__C2__pre_0, n_Task_Activations__C1__C2__pre_1, n_Task_Activations__C1__C2__act_0, n_Task_Activations__C1__C2__act_1 = _pnl_triton_lca_width2_step(n_Task_Activations__C1__C2__input_0, n_Task_Activations__C1__C2__input_1, n_Task_Activations__C1__C2__pre_0, n_Task_Activations__C1__C2__pre_1, n_Task_Activations__C1__C2__act_0, n_Task_Activations__C1__C2__act_1, n_DDM_finished_0, param_8_value, param_9_value, param_10_value, param_11_value, param_12_value, param_13_value, SEED, random_base, step, 0, 1, LCA_MAX_STEPS)
            n_Drift_Rate_Value_projection_2_0 = tl.zeros((BLOCK,), dtype=tl.float32)
            n_Drift_Rate_Value_projection_2_1 = tl.zeros((BLOCK,), dtype=tl.float32)
            n_Drift_Rate_Value_projection_2_2 = tl.zeros((BLOCK,), dtype=tl.float32)
            n_Drift_Rate_Value_projection_2_3 = tl.zeros((BLOCK,), dtype=tl.float32)
            n_Drift_Rate_Value_projection_2_4 = _pnl_triton_projection_term(n_Task_Activations__C1__C2__act_0, 1.0)
            n_Drift_Rate_Value_projection_2_5 = _pnl_triton_projection_term(n_Task_Activations__C1__C2__act_1, 1.0)
            n_Drift_Rate_Value_projection_2_6 = tl.zeros((BLOCK,), dtype=tl.float32)

            n_Drift_Rate_Value_input_0 = (n_Drift_Rate_Value_projection_0_0) + (n_Drift_Rate_Value_projection_1_0) + (n_Drift_Rate_Value_projection_2_0)
            n_Drift_Rate_Value_input_1 = (n_Drift_Rate_Value_projection_0_1) + (n_Drift_Rate_Value_projection_1_1) + (n_Drift_Rate_Value_projection_2_1)
            n_Drift_Rate_Value_input_2 = (n_Drift_Rate_Value_projection_0_2) + (n_Drift_Rate_Value_projection_1_2) + (n_Drift_Rate_Value_projection_2_2)
            n_Drift_Rate_Value_input_3 = (n_Drift_Rate_Value_projection_0_3) + (n_Drift_Rate_Value_projection_1_3) + (n_Drift_Rate_Value_projection_2_3)
            n_Drift_Rate_Value_input_4 = (n_Drift_Rate_Value_projection_0_4) + (n_Drift_Rate_Value_projection_1_4) + (n_Drift_Rate_Value_projection_2_4)
            n_Drift_Rate_Value_input_5 = (n_Drift_Rate_Value_projection_0_5) + (n_Drift_Rate_Value_projection_1_5) + (n_Drift_Rate_Value_projection_2_5)
            n_Drift_Rate_Value_input_6 = (n_Drift_Rate_Value_projection_0_6) + (n_Drift_Rate_Value_projection_1_6) + (n_Drift_Rate_Value_projection_2_6)

            n_Drift_Rate_Value_OutputPort_0_0 = _pnl_triton_instance_Drift_Rate_Value(n_Drift_Rate_Value_input_0, n_Drift_Rate_Value_input_1, n_Drift_Rate_Value_input_2, n_Drift_Rate_Value_input_3, n_Drift_Rate_Value_input_4, n_Drift_Rate_Value_input_5, n_Drift_Rate_Value_input_6)

            n_DDM_projection_0_0 = _pnl_triton_projection_term(n_Drift_Rate_Value_OutputPort_0_0, 1.0)

            n_DDM_input_0 = (n_DDM_projection_0_0)

            n_DDM_value_0, n_DDM_steps_0, n_DDM_finished_0 = _pnl_triton_ddm_step(n_DDM_value_0, n_DDM_steps_0, n_DDM_finished_0, n_DDM_input_0, param_14_value, param_15_value, param_16_value, param_17_value, param_19_value, param_21_value, SEED, random_base, step, 0)

        n_DDM_DECISION_OUTCOME_0 = tl.where(n_DDM_value_0 > 0.0, 1.0, 0.0)
        n_DDM_RESPONSE_TIME_0 = param_18_value + n_DDM_steps_0 * param_19_value
        n_DDM_diag_n_truncated = tl.where(n_DDM_finished_0 == 0.0, 1.0, 0.0)

        diag_lane = (((param_idx * num_subjects + subject_idx) * num_trials + trial_idx) * num_estimates + estimate_idx) * 1
        tl.store(diag + diag_lane + 0, n_DDM_diag_n_truncated, mask=mask)
        n_DECISION_GATE_projection_0_0 = _pnl_triton_projection_term(n_DDM_DECISION_OUTCOME_0, 1.0)

        n_DECISION_GATE_input_0 = (n_DECISION_GATE_projection_0_0)

        n_DECISION_GATE_OutputPort_0_0 = _pnl_triton_linear(n_DECISION_GATE_input_0, param_22_value, param_23_value)

        n_RESPONSE_GATE_projection_0_0 = _pnl_triton_projection_term(n_DDM_RESPONSE_TIME_0, 1.0)

        n_RESPONSE_GATE_projection_1_0 = _pnl_triton_projection_term(n_Cue_Stimulus_Interval_OutputPort_0_0, 0.009999999776482582)

        n_RESPONSE_GATE_input_0 = (n_RESPONSE_GATE_projection_0_0) + (n_RESPONSE_GATE_projection_1_0)

        n_RESPONSE_GATE_OutputPort_0_0 = _pnl_triton_linear(n_RESPONSE_GATE_input_0, param_24_value, param_25_value)

        lane_out = (((param_idx * num_subjects + subject_idx) * num_trials + trial_idx) * num_estimates + estimate_idx) * 2
        tl.store(out + lane_out + 0, n_DECISION_GATE_OutputPort_0_0, mask=mask)
        tl.store(out + lane_out + 1, n_RESPONSE_GATE_OutputPort_0_0, mask=mask)
        trial_idx += 1