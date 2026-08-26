import triton
import triton.language as tl


@triton.jit
def _pnl_triton_linear(x, slope, intercept, scale, offset):
    return scale * (x * slope + intercept) + offset


@triton.jit
def _pnl_triton_projection_term(x, coefficient):
    return x * coefficient


@triton.jit
def pnl_batched_stateless_graph_kernel(
    input_0,
    param_0,
    param_1,
    param_2,
    param_3,
    param_4,
    param_5,
    param_6,
    param_7,
    param_0_set_stride: tl.constexpr,
    param_0_trial_stride: tl.constexpr,
    param_1_set_stride: tl.constexpr,
    param_1_trial_stride: tl.constexpr,
    param_2_set_stride: tl.constexpr,
    param_2_trial_stride: tl.constexpr,
    param_3_set_stride: tl.constexpr,
    param_3_trial_stride: tl.constexpr,
    param_4_set_stride: tl.constexpr,
    param_4_trial_stride: tl.constexpr,
    param_5_set_stride: tl.constexpr,
    param_5_trial_stride: tl.constexpr,
    param_6_set_stride: tl.constexpr,
    param_6_trial_stride: tl.constexpr,
    param_7_set_stride: tl.constexpr,
    param_7_trial_stride: tl.constexpr,
    out,
    total_lanes: tl.constexpr,
    num_subjects: tl.constexpr,
    num_trials: tl.constexpr,
    num_estimates: tl.constexpr,
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

    param_0_value = tl.load(param_0 + param_idx * param_0_set_stride + (subject_idx * num_trials + trial_idx) * param_0_trial_stride, mask=mask, other=1.0)
    param_1_value = tl.load(param_1 + param_idx * param_1_set_stride + (subject_idx * num_trials + trial_idx) * param_1_trial_stride, mask=mask, other=0.0)
    param_2_value = tl.load(param_2 + param_idx * param_2_set_stride + (subject_idx * num_trials + trial_idx) * param_2_trial_stride, mask=mask, other=1.0)
    param_3_value = tl.load(param_3 + param_idx * param_3_set_stride + (subject_idx * num_trials + trial_idx) * param_3_trial_stride, mask=mask, other=0.0)
    param_4_value = tl.load(param_4 + param_idx * param_4_set_stride + (subject_idx * num_trials + trial_idx) * param_4_trial_stride, mask=mask, other=3.0)
    param_5_value = tl.load(param_5 + param_idx * param_5_set_stride + (subject_idx * num_trials + trial_idx) * param_5_trial_stride, mask=mask, other=1.0)
    param_6_value = tl.load(param_6 + param_idx * param_6_set_stride + (subject_idx * num_trials + trial_idx) * param_6_trial_stride, mask=mask, other=1.0)
    param_7_value = tl.load(param_7 + param_idx * param_7_set_stride + (subject_idx * num_trials + trial_idx) * param_7_trial_stride, mask=mask, other=0.0)

    n_n0_output_0_0 = _pnl_triton_linear(tl.load(input_0 + (subject_idx * num_trials + trial_idx) * 2 + 0, mask=mask, other=0.0), param_0_value, param_1_value, param_2_value, param_3_value)
    n_n0_output_0_1 = _pnl_triton_linear(tl.load(input_0 + (subject_idx * num_trials + trial_idx) * 2 + 1, mask=mask, other=0.0), param_0_value, param_1_value, param_2_value, param_3_value)

    n_n1_projection_0_0 = _pnl_triton_projection_term(n_n0_output_0_0, 1.0) + _pnl_triton_projection_term(n_n0_output_0_1, 2.0)

    n_n1_input_0 = (n_n1_projection_0_0)

    n_n1_output_0_0 = _pnl_triton_linear(n_n1_input_0, param_4_value, param_5_value, param_6_value, param_7_value)

    lane_out = (((param_idx * num_subjects + subject_idx) * num_trials + trial_idx) * num_estimates + estimate_idx) * 1
    tl.store(out + lane_out + 0, n_n1_output_0_0, mask=mask)