from __future__ import annotations

import numpy as np

from psyneulink.core.batched.graph import (
    DDM_GRAPH_FUSION,
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
) -> BatchedSimulationResult:
    torch, triton = _import_torch_triton()
    if not torch.cuda.is_available():
        raise RuntimeError("The Triton batched backend requires an available CUDA device.")

    params = normalize_parameter_sets(parameter_sets, ir)
    prepared_inputs = prepare_inputs(ir, inputs, subject_slices)
    module = _load_kernel_module(ir)
    fusion_kind = None if ir.graph is None else ir.graph.fusion_kind

    if fusion_kind == STATELESS_GRAPH_FUSION:
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


def _load_kernel_module(ir: BatchedCompositionIR):
    source = _kernel_source(ir)
    module_kind = None if ir.graph is None else ir.graph.fusion_kind
    return load_triton_kernel_module(source, module_kind, ir.model_kind)


def _kernel_source(ir: BatchedCompositionIR) -> str:
    if ir.graph is None or ir.graph.fusion_kind is None:
        raise ValueError(
            "The Triton batched backend requires a lowered graph with a "
            "supported fusion kind."
        )
    return triton_graph_kernel_source(lower_to_kernel_ir(ir))
