from __future__ import annotations

import numpy as np

from psyneulink.core.batched.bindings import (
    EMPTY_COMPONENT_BINDINGS,
    BatchedComponentBindings,
)
from psyneulink.core.batched.graph import (
    DDM_GRAPH_FUSION,
    DDM_MODEL,
    STATELESS_GRAPH_FUSION,
    STATEFUL_GRAPH_FUSION,
)
from psyneulink.core.batched.ir import BatchedCompositionIR, BatchedSimulationResult
from psyneulink.core.batched.ir_debug import (
    _lca_max_steps,
    normalize_parameter_sets,
    prepare_inputs,
)
from psyneulink.core.batched.kernel_ir import lower_to_kernel_ir
from psyneulink.core.batched.backend.triton.cache import load_triton_kernel_module
from psyneulink.core.batched.backend.triton.graph_emit import triton_graph_kernel_source


def run_triton(
    ir: BatchedCompositionIR,
    inputs,
    parameter_sets,
    num_estimates: int,
    subject_slices=None,
    seed=None,
    common_random_numbers: bool = True,
    component_bindings: BatchedComponentBindings = EMPTY_COMPONENT_BINDINGS,
) -> BatchedSimulationResult:
    torch, triton = _import_torch_triton()
    if not torch.cuda.is_available():
        raise RuntimeError("The Triton batched backend requires an available CUDA device.")

    params = normalize_parameter_sets(parameter_sets, ir)
    prepared_inputs = prepare_inputs(ir, inputs, subject_slices)
    module = _load_kernel_module(ir, component_bindings)
    fusion_kind = None if ir.graph is None else ir.graph.fusion_kind

    if fusion_kind == DDM_MODEL:
        values = _run_ddm_kernel(
            torch,
            triton,
            module,
            ir,
            prepared_inputs,
            params,
            num_estimates,
            seed,
            common_random_numbers,
        )
    elif fusion_kind == STATELESS_GRAPH_FUSION:
        values = _run_stateless_graph_kernel(
            torch,
            triton,
            module,
            ir,
            prepared_inputs,
            params,
            num_estimates,
        )
    elif fusion_kind == DDM_GRAPH_FUSION:
        values = _run_ddm_graph_kernel(
            torch,
            triton,
            module,
            ir,
            prepared_inputs,
            params,
            num_estimates,
            seed,
            common_random_numbers,
        )
    elif fusion_kind == STATEFUL_GRAPH_FUSION:
        values = _run_stateful_graph_kernel(
            torch,
            triton,
            module,
            ir,
            prepared_inputs,
            params,
            num_estimates,
            seed,
            common_random_numbers,
        )
    else:
        raise ValueError(f"Unsupported Triton batched graph fusion kind '{fusion_kind}'.")

    return BatchedSimulationResult(
        values=values,
        output_names=ir.output_names,
        backend="triton",
        metadata={"model_kind": ir.model_kind},
    )


def _run_ddm_kernel(
    torch,
    triton,
    module,
    ir: BatchedCompositionIR,
    inputs: dict[str, np.ndarray],
    params: list[dict[str, float]],
    num_estimates: int,
    seed,
    common_random_numbers: bool,
) -> np.ndarray:
    stimulus = torch.as_tensor(inputs["stimulus"], dtype=torch.float32, device="cuda").contiguous()
    param_tensors = _ddm_param_tensors(torch, params)
    num_params = len(params)
    num_subjects, num_trials = inputs["stimulus"].shape
    total_lanes = num_params * num_subjects * num_trials * num_estimates
    out = torch.empty((num_params, num_subjects, num_trials, num_estimates, 2), dtype=torch.float32, device="cuda")
    block = 128
    grid = (triton.cdiv(total_lanes, block),)
    module.pnl_batched_ddm_kernel[grid](
        stimulus,
        *param_tensors,
        out,
        total_lanes,
        num_subjects,
        num_trials,
        num_estimates,
        MAX_STEPS=ir.max_steps,
        COMMON_RANDOM=bool(common_random_numbers),
        SEED=0 if seed is None else int(seed),
        BLOCK=block,
    )
    torch.cuda.synchronize()
    return out.cpu().numpy()


def _run_stateless_graph_kernel(
    torch,
    triton,
    module,
    ir: BatchedCompositionIR,
    inputs: dict[str, np.ndarray],
    params: list[dict[str, float]],
    num_estimates: int,
) -> np.ndarray:
    graph = ir.graph
    input_tensors = []
    for input_spec in graph.inputs:
        value = np.asarray(inputs[input_spec.node], dtype=np.float32)
        if input_spec.width == 1 and value.ndim == 2:
            value = value[:, :, None]
        input_tensors.append(torch.as_tensor(value, dtype=torch.float32, device="cuda").contiguous())

    param_tensors = tuple(
        torch.as_tensor([row[param_spec.name] for row in params], dtype=torch.float32, device="cuda").contiguous()
        for param_spec in ir.params
    )
    num_params = len(params)
    first_input = input_tensors[0]
    num_subjects, num_trials = first_input.shape[:2]
    output_width = sum(output.width for output in graph.outputs)
    total_lanes = num_params * num_subjects * num_trials * num_estimates
    out = torch.empty(
        (num_params, num_subjects, num_trials, num_estimates, output_width),
        dtype=torch.float32,
        device="cuda",
    )
    block = 128
    grid = (triton.cdiv(total_lanes, block),)
    module.pnl_batched_stateless_graph_kernel[grid](
        *input_tensors,
        *param_tensors,
        out,
        total_lanes,
        num_subjects,
        num_trials,
        num_estimates,
        BLOCK=block,
    )
    torch.cuda.synchronize()
    return out.cpu().numpy()


def _run_ddm_graph_kernel(
    torch,
    triton,
    module,
    ir: BatchedCompositionIR,
    inputs: dict[str, np.ndarray],
    params: list[dict[str, float]],
    num_estimates: int,
    seed,
    common_random_numbers: bool,
) -> np.ndarray:
    graph = ir.graph
    input_tensors = []
    for input_spec in graph.inputs:
        value = np.asarray(inputs[input_spec.node], dtype=np.float32)
        if input_spec.width == 1 and value.ndim == 2:
            value = value[:, :, None]
        input_tensors.append(torch.as_tensor(value, dtype=torch.float32, device="cuda").contiguous())

    param_tensors = tuple(
        torch.as_tensor([row[param_spec.name] for row in params], dtype=torch.float32, device="cuda").contiguous()
        for param_spec in ir.params
    )
    num_params = len(params)
    first_input = input_tensors[0]
    num_subjects, num_trials = first_input.shape[:2]
    output_width = sum(output.width for output in graph.outputs)
    total_lanes = num_params * num_subjects * num_trials * num_estimates
    out = torch.empty(
        (num_params, num_subjects, num_trials, num_estimates, output_width),
        dtype=torch.float32,
        device="cuda",
    )
    block = 128
    grid = (triton.cdiv(total_lanes, block),)
    module.pnl_batched_ddm_graph_kernel[grid](
        *input_tensors,
        *param_tensors,
        out,
        total_lanes,
        num_subjects,
        num_trials,
        num_estimates,
        MAX_STEPS=ir.max_steps,
        COMMON_RANDOM=bool(common_random_numbers),
        SEED=0 if seed is None else int(seed),
        BLOCK=block,
    )
    torch.cuda.synchronize()
    return out.cpu().numpy()


def _run_stateful_graph_kernel(
    torch,
    triton,
    module,
    ir: BatchedCompositionIR,
    inputs: dict[str, np.ndarray],
    params: list[dict[str, float]],
    num_estimates: int,
    seed,
    common_random_numbers: bool,
) -> np.ndarray:
    graph = ir.graph
    input_tensors = []
    for input_spec in graph.inputs:
        value = np.asarray(inputs[input_spec.node], dtype=np.float32)
        if input_spec.width == 1 and value.ndim == 2:
            value = value[:, :, None]
        input_tensors.append(torch.as_tensor(value, dtype=torch.float32, device="cuda").contiguous())

    param_tensors = tuple(
        torch.as_tensor([row[param_spec.name] for row in params], dtype=torch.float32, device="cuda").contiguous()
        for param_spec in ir.params
    )
    num_params = len(params)
    first_input = input_tensors[0]
    num_subjects, num_trials = first_input.shape[:2]
    output_width = sum(output.width for output in graph.outputs)
    total_lanes = num_params * num_subjects * num_estimates
    out = torch.empty(
        (num_params, num_subjects, num_trials, num_estimates, output_width),
        dtype=torch.float32,
        device="cuda",
    )
    block = 128
    grid = (triton.cdiv(total_lanes, block),)
    module.pnl_batched_stateful_graph_kernel[grid](
        *input_tensors,
        *param_tensors,
        out,
        total_lanes,
        num_subjects,
        num_estimates,
        num_trials,
        LCA_MAX_STEPS=_lca_max_steps(ir, inputs),
        MAX_STEPS=ir.max_steps,
        COMMON_RANDOM=bool(common_random_numbers),
        SEED=0 if seed is None else int(seed),
        BLOCK=block,
    )
    torch.cuda.synchronize()
    return out.cpu().numpy()


def _ddm_param_tensors(torch, params: list[dict[str, float]]):
    names = ("rate", "noise", "threshold", "non_decision_time", "time_step_size", "starting_value", "offset")
    return tuple(
        torch.as_tensor([row[name] for row in params], dtype=torch.float32, device="cuda").contiguous()
        for name in names
    )


def _import_torch_triton():
    try:
        import torch
        import triton
    except ImportError as error:
        raise RuntimeError(
            "The Triton batched backend requires torch and triton. "
            "Install the optional triton extra to use it."
        ) from error
    return torch, triton


def _load_kernel_module(
    ir: BatchedCompositionIR,
    component_bindings: BatchedComponentBindings = EMPTY_COMPONENT_BINDINGS,
):
    source = _kernel_source(ir, component_bindings)
    module_kind = None if ir.graph is None else ir.graph.fusion_kind
    return load_triton_kernel_module(source, module_kind, ir.model_kind)


def _kernel_source(
    ir: BatchedCompositionIR,
    component_bindings: BatchedComponentBindings = EMPTY_COMPONENT_BINDINGS,
) -> str:
    if ir.graph is not None and ir.graph.fusion_kind == STATELESS_GRAPH_FUSION:
        return triton_graph_kernel_source(lower_to_kernel_ir(ir), component_bindings)
    if ir.graph is not None and ir.graph.fusion_kind == DDM_GRAPH_FUSION:
        return triton_graph_kernel_source(lower_to_kernel_ir(ir), component_bindings)
    if ir.graph is not None and ir.graph.fusion_kind == STATEFUL_GRAPH_FUSION:
        return triton_graph_kernel_source(lower_to_kernel_ir(ir), component_bindings)

    return r'''
import triton
import triton.language as tl


@triton.jit
def pnl_batched_ddm_kernel(
    stimulus,
    rates,
    noises,
    thresholds,
    non_decision_times,
    time_step_sizes,
    starting_values,
    offsets_param,
    out,
    total_lanes: tl.constexpr,
    num_subjects: tl.constexpr,
    num_trials: tl.constexpr,
    num_estimates: tl.constexpr,
    MAX_STEPS: tl.constexpr,
    COMMON_RANDOM: tl.constexpr,
    SEED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_lanes

    estimate_idx = offsets % num_estimates
    tmp = offsets // num_estimates
    trial_idx = tmp % num_trials
    tmp = tmp // num_trials
    subject_idx = tmp % num_subjects
    param_idx = tmp // num_subjects

    drift_input = tl.load(stimulus + subject_idx * num_trials + trial_idx, mask=mask, other=0.0)
    rate = tl.load(rates + param_idx, mask=mask, other=0.0)
    noise = tl.load(noises + param_idx, mask=mask, other=0.0)
    threshold = tl.load(thresholds + param_idx, mask=mask, other=1.0)
    non_decision_time = tl.load(non_decision_times + param_idx, mask=mask, other=0.0)
    dt = tl.load(time_step_sizes + param_idx, mask=mask, other=1.0)
    value = tl.load(starting_values + param_idx, mask=mask, other=0.0)
    step_offset = tl.load(offsets_param + param_idx, mask=mask, other=0.0)
    steps = tl.zeros((BLOCK,), dtype=tl.float32)
    sqrt_dt = tl.sqrt(dt)
    boundary_tolerance = tl.maximum(1.0e-7, threshold * 1.0e-6)

    if COMMON_RANDOM:
        random_base = ((subject_idx * num_estimates + estimate_idx) * num_trials + trial_idx) * MAX_STEPS
    else:
        random_base = (((param_idx * num_subjects + subject_idx) * num_estimates + estimate_idx) * num_trials + trial_idx) * MAX_STEPS

    for step in tl.static_range(0, MAX_STEPS):
        active = tl.abs(value) + boundary_tolerance < threshold
        random_draw = tl.randn(SEED, random_base + step)
        updated = value + rate * drift_input * dt + noise * sqrt_dt * random_draw
        updated = tl.minimum(tl.maximum(updated + step_offset, -threshold), threshold)
        value = tl.where(active, updated, value)
        steps += tl.where(active, 1.0, 0.0)

    lane_out = offsets * 2
    decision = tl.where(value > 0.0, 1.0, 0.0)
    response_time = non_decision_time + steps * dt
    tl.store(out + lane_out, decision, mask=mask)
    tl.store(out + lane_out + 1, response_time, mask=mask)
'''
