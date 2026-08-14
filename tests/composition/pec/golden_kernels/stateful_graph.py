import triton
import triton.language as tl


@triton.jit
def _pnl_triton_linear(x, slope, intercept, scale, offset):
    return scale * (x * slope + intercept) + offset


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
def _pnl_triton_lca_width2_integrate(input0, input1, pre0, pre1, act0, act1, gain, leak, competition, self_excitation, noise, dt, lca_steps, seed, random_base, stream0: tl.constexpr, stream1: tl.constexpr, lca_max_steps: tl.constexpr, lane_mask):
    block_steps = tl.minimum(tl.max(tl.where(lane_mask, lca_steps, 0.0)), lca_max_steps)
    step = 0
    while step < block_steps:
        active = step < lca_steps
        n0 = tl.randn(seed, random_base + stream0 + step)
        n1 = tl.randn(seed, random_base + stream1 + step)
        pre0, pre1, act0, act1 = _pnl_triton_lca_width2_recurrence(input0, input1, pre0, pre1, act0, act1, active, gain, leak, competition, self_excitation, noise, dt, n0, n1)
        step += 1
    return (pre0, pre1, act0, act1)


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
def _pnl_triton_ddm(x, rate, noise, threshold, threshold_collapse, non_decision_time, time_step_size, starting_value, offset, seed, rng_base, max_steps: tl.constexpr, lane_mask):
    value = starting_value
    steps = tl.zeros_like(x)
    finished = tl.zeros_like(x)
    step = 0
    while (step < max_steps) & (tl.max(tl.where(lane_mask & (finished == 0.0), 1, 0)) > 0):
        draw = tl.randn(seed, rng_base + step)
        value, steps, finished = _pnl_triton_ddm_update(value, steps, finished, x, rate, noise, threshold, threshold_collapse, time_step_size, offset, draw, step)
        step += 1
    truncated = tl.where(finished == 0.0, 1.0, 0.0)
    return (tl.where(value > 0.0, 1.0, 0.0), non_decision_time + steps * time_step_size, truncated)


@triton.jit
def pnl_batched_stateful_graph_kernel(
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
    param_26,
    param_27,
    param_28,
    param_29,
    param_30,
    param_31,
    param_32,
    param_33,
    param_34,
    param_35,
    param_36,
    param_37,
    param_38,
    param_39,
    param_40,
    param_41,
    param_42,
    param_43,
    param_44,
    param_45,
    param_46,
    param_47,
    param_48,
    param_49,
    param_50,
    param_51,
    param_52,
    param_53,
    param_54,
    param_55,
    param_56,
    param_57,
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
    subject_idx = tmp % num_subjects
    param_idx = tmp // num_subjects

    param_0_value = tl.load(param_0 + param_idx, mask=mask, other=1.0)
    param_1_value = tl.load(param_1 + param_idx, mask=mask, other=0.0)
    param_2_value = tl.load(param_2 + param_idx, mask=mask, other=1.0)
    param_3_value = tl.load(param_3 + param_idx, mask=mask, other=0.0)
    param_4_value = tl.load(param_4 + param_idx, mask=mask, other=1.0)
    param_5_value = tl.load(param_5 + param_idx, mask=mask, other=0.0)
    param_6_value = tl.load(param_6 + param_idx, mask=mask, other=1.0)
    param_7_value = tl.load(param_7 + param_idx, mask=mask, other=0.0)
    param_8_value = tl.load(param_8 + param_idx, mask=mask, other=1.0)
    param_9_value = tl.load(param_9 + param_idx, mask=mask, other=0.0)
    param_10_value = tl.load(param_10 + param_idx, mask=mask, other=1.0)
    param_11_value = tl.load(param_11 + param_idx, mask=mask, other=0.0)
    param_12_value = tl.load(param_12 + param_idx, mask=mask, other=1.0)
    param_13_value = tl.load(param_13 + param_idx, mask=mask, other=0.0)
    param_14_value = tl.load(param_14 + param_idx, mask=mask, other=1.0)
    param_15_value = tl.load(param_15 + param_idx, mask=mask, other=0.0)
    param_16_value = tl.load(param_16 + param_idx, mask=mask, other=3.0)
    param_17_value = tl.load(param_17 + param_idx, mask=mask, other=3.0)
    param_18_value = tl.load(param_18 + param_idx, mask=mask, other=2.0)
    param_19_value = tl.load(param_19 + param_idx, mask=mask, other=0.0)
    param_20_value = tl.load(param_20 + param_idx, mask=mask, other=0.0)
    param_21_value = tl.load(param_21 + param_idx, mask=mask, other=0.01)
    param_22_value = tl.load(param_22 + param_idx, mask=mask, other=1.0)
    param_23_value = tl.load(param_23 + param_idx, mask=mask, other=0.0)
    param_24_value = tl.load(param_24 + param_idx, mask=mask, other=1.0)
    param_25_value = tl.load(param_25 + param_idx, mask=mask, other=0.0)
    param_26_value = tl.load(param_26 + param_idx, mask=mask, other=0.01)
    param_27_value = tl.load(param_27 + param_idx, mask=mask, other=0.0)
    param_28_value = tl.load(param_28 + param_idx, mask=mask, other=1.0)
    param_29_value = tl.load(param_29 + param_idx, mask=mask, other=0.0)
    param_30_value = tl.load(param_30 + param_idx, mask=mask, other=1.0)
    param_31_value = tl.load(param_31 + param_idx, mask=mask, other=0.0)
    param_32_value = tl.load(param_32 + param_idx, mask=mask, other=1.0)
    param_33_value = tl.load(param_33 + param_idx, mask=mask, other=0.0)
    param_34_value = tl.load(param_34 + param_idx, mask=mask, other=1.0)
    param_35_value = tl.load(param_35 + param_idx, mask=mask, other=0.0)
    param_36_value = tl.load(param_36 + param_idx, mask=mask, other=1.0)
    param_37_value = tl.load(param_37 + param_idx, mask=mask, other=0.0)
    param_38_value = tl.load(param_38 + param_idx, mask=mask, other=0.2)
    param_39_value = tl.load(param_39 + param_idx, mask=mask, other=0.0)
    param_40_value = tl.load(param_40 + param_idx, mask=mask, other=1.0)
    param_41_value = tl.load(param_41 + param_idx, mask=mask, other=0.0)
    param_42_value = tl.load(param_42 + param_idx, mask=mask, other=1.0)
    param_43_value = tl.load(param_43 + param_idx, mask=mask, other=0.0)
    param_44_value = tl.load(param_44 + param_idx, mask=mask, other=0.05)
    param_45_value = tl.load(param_45 + param_idx, mask=mask, other=0.0)
    param_46_value = tl.load(param_46 + param_idx, mask=mask, other=0.2)
    param_47_value = tl.load(param_47 + param_idx, mask=mask, other=0.01)
    param_48_value = tl.load(param_48 + param_idx, mask=mask, other=0.0)
    param_49_value = tl.load(param_49 + param_idx, mask=mask, other=0.0)
    param_50_value = tl.load(param_50 + param_idx, mask=mask, other=1.0)
    param_51_value = tl.load(param_51 + param_idx, mask=mask, other=0.0)
    param_52_value = tl.load(param_52 + param_idx, mask=mask, other=1.0)
    param_53_value = tl.load(param_53 + param_idx, mask=mask, other=0.0)
    param_54_value = tl.load(param_54 + param_idx, mask=mask, other=1.0)
    param_55_value = tl.load(param_55 + param_idx, mask=mask, other=0.0)
    param_56_value = tl.load(param_56 + param_idx, mask=mask, other=1.0)
    param_57_value = tl.load(param_57 + param_idx, mask=mask, other=0.0)

    n6_state_0_0 = tl.full((BLOCK,), 0.0, tl.float32)
    n6_state_0_1 = tl.full((BLOCK,), 0.0, tl.float32)
    n6_state_1_0 = tl.full((BLOCK,), 0.0, tl.float32)
    n6_state_1_1 = tl.full((BLOCK,), 0.0, tl.float32)

    trial_idx = 0
    while trial_idx < num_trials:
        if COMMON_RANDOM:
            random_base = ((subject_idx * num_estimates + estimate_idx) * num_trials + trial_idx).to(tl.int64) * 12884901888
        else:
            random_base = (((param_idx * num_subjects + subject_idx) * num_estimates + estimate_idx) * num_trials + trial_idx).to(tl.int64) * 12884901888

        n_n0_output_0_0 = _pnl_triton_linear(tl.load(input_3 + (subject_idx * num_trials + trial_idx) * 1 + 0, mask=mask, other=0.0), param_12_value, param_13_value, param_14_value, param_15_value)

        n_n1_output_0_0 = _pnl_triton_linear(tl.load(input_2 + (subject_idx * num_trials + trial_idx) * 1 + 0, mask=mask, other=0.0), param_8_value, param_9_value, param_10_value, param_11_value)

        n_n3_output_0_0 = _pnl_triton_linear(tl.load(input_1 + (subject_idx * num_trials + trial_idx) * 2 + 0, mask=mask, other=0.0), param_4_value, param_5_value, param_6_value, param_7_value)
        n_n3_output_0_1 = _pnl_triton_linear(tl.load(input_1 + (subject_idx * num_trials + trial_idx) * 2 + 1, mask=mask, other=0.0), param_4_value, param_5_value, param_6_value, param_7_value)

        n_n4_projection_0_0 = _pnl_triton_projection_term(n_n3_output_0_0, 1.0)
        n_n4_projection_0_1 = _pnl_triton_projection_term(n_n3_output_0_1, 1.0)

        n_n4_input_0 = (n_n4_projection_0_0)
        n_n4_input_1 = (n_n4_projection_0_1)

        n_n4_output_0_0 = _pnl_triton_linear(n_n4_input_0, param_26_value, param_27_value, param_28_value, param_29_value)
        n_n4_output_0_1 = _pnl_triton_linear(n_n4_input_1, param_26_value, param_27_value, param_28_value, param_29_value)

        n_n5_output_0_0 = _pnl_triton_linear(tl.load(input_0 + (subject_idx * num_trials + trial_idx) * 2 + 0, mask=mask, other=0.0), param_0_value, param_1_value, param_2_value, param_3_value)
        n_n5_output_0_1 = _pnl_triton_linear(tl.load(input_0 + (subject_idx * num_trials + trial_idx) * 2 + 1, mask=mask, other=0.0), param_0_value, param_1_value, param_2_value, param_3_value)

        n_n6_projection_0_0 = _pnl_triton_projection_term(n_n5_output_0_0, 1.0)
        n_n6_projection_0_1 = _pnl_triton_projection_term(n_n5_output_0_1, 1.0)

        n_n6_input_0 = (n_n6_projection_0_0)
        n_n6_input_1 = (n_n6_projection_0_1)

        n6_lca_steps = tl.minimum(tl.maximum(tl.ceil(tl.load(input_2 + (subject_idx * num_trials + trial_idx) * 1 + 0, mask=mask, other=0.0)), 0.0), LCA_MAX_STEPS)
        n6_state_0_0, n6_state_0_1, n6_state_1_0, n6_state_1_1 = _pnl_triton_lca_width2_integrate(n_n6_input_0, n_n6_input_1, n6_state_0_0, n6_state_0_1, n6_state_1_0, n6_state_1_1, param_16_value, param_17_value, param_18_value, param_19_value, param_20_value, param_21_value, n6_lca_steps, SEED, random_base, 0, 4294967296, LCA_MAX_STEPS, mask)

        n_n7_projection_0_0 = _pnl_triton_projection_term(n6_state_1_0, 1.0)
        n_n7_projection_0_1 = _pnl_triton_projection_term(n6_state_1_1, 1.0)

        n_n7_projection_1_0 = _pnl_triton_projection_term(n_n3_output_0_0, 1.0)
        n_n7_projection_1_1 = _pnl_triton_projection_term(n_n3_output_0_1, 1.0)

        n_n7_input_0 = (n_n7_projection_0_0) * (n_n7_projection_1_0)
        n_n7_input_1 = (n_n7_projection_0_1) * (n_n7_projection_1_1)

        n_n7_output_0_0 = _pnl_triton_linear(n_n7_input_0, param_22_value, param_23_value, param_24_value, param_25_value)
        n_n7_output_0_1 = _pnl_triton_linear(n_n7_input_1, param_22_value, param_23_value, param_24_value, param_25_value)

        n_n8_projection_0_0 = _pnl_triton_projection_term(n_n7_output_0_0, 1.0) + _pnl_triton_projection_term(n_n7_output_0_1, 1.0)

        n_n8_projection_1_0 = _pnl_triton_projection_term(n_n4_output_0_0, 1.0) + _pnl_triton_projection_term(n_n4_output_0_1, 1.0)

        n_n8_input_0 = (n_n8_projection_0_0) + (n_n8_projection_1_0)

        n_n8_output_0_0 = _pnl_triton_linear(n_n8_input_0, param_30_value, param_31_value, param_32_value, param_33_value)

        n_n9_projection_0_0 = _pnl_triton_projection_term(n_n8_output_0_0, 1.0)

        n_n9_projection_1_0 = _pnl_triton_projection_term(n_n0_output_0_0, 1.0)

        n_n9_input_0 = (n_n9_projection_0_0) * (n_n9_projection_1_0)

        n_n9_output_0_0 = _pnl_triton_linear(n_n9_input_0, param_34_value, param_35_value, param_36_value, param_37_value)

        n_n10_projection_0_0 = _pnl_triton_projection_term(n_n9_output_0_0, 1.0)

        n_n10_input_0 = (n_n10_projection_0_0)

        n_n10_output_0_0 = _pnl_triton_linear(n_n10_input_0, param_38_value, param_39_value, param_40_value, param_41_value)

        n_n11_projection_0_0 = _pnl_triton_projection_term(n_n10_output_0_0, 1.0)

        n_n11_input_0 = (n_n11_projection_0_0)

        n_n11_output_0_0, n_n11_output_1_0, n_n11_diagnostic_0_0 = _pnl_triton_ddm(n_n11_input_0, param_42_value, param_43_value, param_44_value, param_45_value, param_46_value, param_47_value, param_48_value, param_49_value, SEED, random_base + 8589934592, MAX_STEPS, mask)

        diag_lane = (((param_idx * num_subjects + subject_idx) * num_trials + trial_idx) * num_estimates + estimate_idx) * 1
        tl.store(diag + diag_lane + 0, n_n11_diagnostic_0_0, mask=mask)
        n_n12_projection_0_0 = _pnl_triton_projection_term(n_n11_output_0_0, 1.0)

        n_n12_input_0 = (n_n12_projection_0_0)

        n_n12_output_0_0 = _pnl_triton_linear(n_n12_input_0, param_50_value, param_51_value, param_52_value, param_53_value)

        n_n13_projection_0_0 = _pnl_triton_projection_term(n_n11_output_1_0, 1.0)

        n_n13_input_0 = (n_n13_projection_0_0)

        n_n13_output_0_0 = _pnl_triton_linear(n_n13_input_0, param_54_value, param_55_value, param_56_value, param_57_value)

        lane_out = (((param_idx * num_subjects + subject_idx) * num_trials + trial_idx) * num_estimates + estimate_idx) * 2
        tl.store(out + lane_out + 0, n_n12_output_0_0, mask=mask)
        tl.store(out + lane_out + 1, n_n13_output_0_0, mask=mask)
        trial_idx += 1