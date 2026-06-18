import triton
import triton.language as tl


@triton.jit
def _pnl_triton_linear(x, slope, intercept):
    return slope * x + intercept


@triton.jit
def _pnl_triton_projection_term(x, coefficient):
    return x * coefficient


@triton.jit
def _pnl_triton_ddm(x, rate, noise, threshold, non_decision_time, time_step_size, starting_value, offset, seed, rng_base, max_steps: tl.constexpr):
    value = starting_value
    steps = tl.zeros_like(x)
    sqrt_dt = tl.sqrt(time_step_size)
    boundary_tolerance = tl.maximum(1e-07, threshold * 1e-06)
    for step in tl.range(0, max_steps, 1, loop_unroll_factor=1):
        active = tl.abs(value) + boundary_tolerance < threshold
        draw = tl.randn(seed, rng_base + step)
        updated = value + rate * x * time_step_size + noise * sqrt_dt * draw
        updated = tl.minimum(tl.maximum(updated + offset, -threshold), threshold)
        value = tl.where(active, updated, value)
        steps += tl.where(active, 1.0, 0.0)
    return (tl.where(value > 0.0, 1.0, 0.0), non_decision_time + steps * time_step_size)


@triton.jit
def pnl_batched_ddm_graph_kernel(
    input_0,
    param_0,
    param_1,
    param_2,
    param_3,
    param_4,
    param_5,
    param_6,
    param_7,
    param_8,
    out,
    total_lanes: tl.constexpr,
    num_subjects: tl.constexpr,
    num_trials: tl.constexpr,
    num_estimates: tl.constexpr,
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
    param_4_value = tl.load(param_4 + param_idx, mask=mask, other=0.05)
    param_5_value = tl.load(param_5 + param_idx, mask=mask, other=0.0)
    param_6_value = tl.load(param_6 + param_idx, mask=mask, other=0.01)
    param_7_value = tl.load(param_7 + param_idx, mask=mask, other=0.0)
    param_8_value = tl.load(param_8 + param_idx, mask=mask, other=0.0)

    n_stimulus_RESULT_0 = _pnl_triton_linear(tl.load(input_0 + (subject_idx * num_trials + trial_idx) * 1 + 0, mask=mask, other=0.0), param_0_value, param_1_value)

    n_DDM_projection_0_0 = _pnl_triton_projection_term(n_stimulus_RESULT_0, 1.0)

    n_DDM_input_0 = (n_DDM_projection_0_0)

    if COMMON_RANDOM:
        random_base = ((subject_idx * num_estimates + estimate_idx) * num_trials + trial_idx) * MAX_STEPS
    else:
        random_base = (((param_idx * num_subjects + subject_idx) * num_estimates + estimate_idx) * num_trials + trial_idx) * MAX_STEPS
    n_DDM_DECISION_OUTCOME_0, n_DDM_RESPONSE_TIME_0 = _pnl_triton_ddm(n_DDM_input_0, param_2_value, param_3_value, param_4_value, param_5_value, param_6_value, param_7_value, param_8_value, SEED, random_base, MAX_STEPS)

    lane_out = (((param_idx * num_subjects + subject_idx) * num_trials + trial_idx) * num_estimates + estimate_idx) * 2
    tl.store(out + lane_out + 0, n_DDM_DECISION_OUTCOME_0, mask=mask)
    tl.store(out + lane_out + 1, n_DDM_RESPONSE_TIME_0, mask=mask)