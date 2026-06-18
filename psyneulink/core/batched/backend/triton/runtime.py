from __future__ import annotations

import os

import numpy as np

from psyneulink.core.batched.graph import (
    DDM_GRAPH_FUSION,
    STATELESS_GRAPH_FUSION,
    STATEFUL_GRAPH_FUSION,
)
from psyneulink.core.batched.ir import BatchedCompositionIR, BatchedSimulationResult
from psyneulink.core.batched.prep import (
    lca_max_steps,
    normalize_parameter_sets,
    prepare_inputs,
)
from psyneulink.core.batched.kernel_ir import lower_to_kernel_ir
from psyneulink.core.batched.backend.triton.cache import (
    interpret_scope,
    load_triton_kernel_module,
)
from psyneulink.core.batched.backend.triton.graph_emit import triton_graph_kernel_source


def run_triton(
    ir: BatchedCompositionIR,
    inputs,
    parameter_sets,
    num_estimates: int,
    subject_slices=None,
    seed=None,
    common_random_numbers: bool = True,
    device: str = "cuda",
) -> BatchedSimulationResult:
    """Execute the generated batched kernels.

    ``device="cuda"`` compiles and runs on the GPU.  ``device="cpu"`` runs the
    *same* generated kernels through Triton's interpreter (no CUDA required); this
    is the CPU test/debug path and is slow, so keep cases small.
    """

    interpret = device == "cpu"
    torch, triton = _import_torch_triton(interpret)
    if not interpret and not torch.cuda.is_available():
        raise RuntimeError("The Triton batched backend requires an available CUDA device.")

    params = normalize_parameter_sets(parameter_sets, ir)
    prepared_inputs = prepare_inputs(ir, inputs, subject_slices)
    fusion_kind = None if ir.graph is None else ir.graph.fusion_kind

    with interpret_scope(interpret):
        module = _load_kernel_module(ir, interpret=interpret)

        if fusion_kind == STATELESS_GRAPH_FUSION:
            values = _run_stateless_graph_kernel(
                torch, triton, module, ir, prepared_inputs, params, num_estimates, device,
            )
        elif fusion_kind == DDM_GRAPH_FUSION:
            values = _run_ddm_graph_kernel(
                torch, triton, module, ir, prepared_inputs, params, num_estimates,
                seed, common_random_numbers, device,
            )
        elif fusion_kind == STATEFUL_GRAPH_FUSION:
            values = _run_stateful_graph_kernel(
                torch, triton, module, ir, prepared_inputs, params, num_estimates,
                seed, common_random_numbers, device,
            )
        else:
            raise ValueError(f"Unsupported Triton batched graph fusion kind '{fusion_kind}'.")

    return BatchedSimulationResult(
        values=values,
        output_names=ir.output_names,
        backend="triton" if device == "cuda" else "triton_cpu",
        metadata={"model_kind": ir.model_kind, "device": device},
    )


def _input_tensors(torch, graph, inputs, device):
    tensors = []
    for input_spec in graph.inputs:
        value = np.asarray(inputs[input_spec.node], dtype=np.float32)
        if input_spec.width == 1 and value.ndim == 2:
            value = value[:, :, None]
        tensors.append(torch.as_tensor(value, dtype=torch.float32, device=device).contiguous())
    return tensors


def _param_tensors(torch, ir, params, device):
    return tuple(
        torch.as_tensor([row[param_spec.name] for row in params], dtype=torch.float32, device=device).contiguous()
        for param_spec in ir.params
    )


def _sync(torch, device):
    if device == "cuda":
        torch.cuda.synchronize()


def _run_stateless_graph_kernel(
    torch, triton, module, ir, inputs, params, num_estimates, device,
) -> np.ndarray:
    graph = ir.graph
    input_tensors = _input_tensors(torch, graph, inputs, device)
    param_tensors = _param_tensors(torch, ir, params, device)
    num_params = len(params)
    num_subjects, num_trials = input_tensors[0].shape[:2]
    output_width = sum(output.width for output in graph.outputs)
    total_lanes = num_params * num_subjects * num_trials * num_estimates
    out = torch.empty(
        (num_params, num_subjects, num_trials, num_estimates, output_width),
        dtype=torch.float32, device=device,
    )
    block = 128
    grid = (triton.cdiv(total_lanes, block),)
    module.pnl_batched_stateless_graph_kernel[grid](
        *input_tensors, *param_tensors, out,
        total_lanes, num_subjects, num_trials, num_estimates, BLOCK=block,
    )
    _sync(torch, device)
    return out.cpu().numpy()


def _run_ddm_graph_kernel(
    torch, triton, module, ir, inputs, params, num_estimates, seed, common_random_numbers, device,
) -> np.ndarray:
    graph = ir.graph
    input_tensors = _input_tensors(torch, graph, inputs, device)
    param_tensors = _param_tensors(torch, ir, params, device)
    num_params = len(params)
    num_subjects, num_trials = input_tensors[0].shape[:2]
    output_width = sum(output.width for output in graph.outputs)
    total_lanes = num_params * num_subjects * num_trials * num_estimates
    out = torch.empty(
        (num_params, num_subjects, num_trials, num_estimates, output_width),
        dtype=torch.float32, device=device,
    )
    block = 128
    grid = (triton.cdiv(total_lanes, block),)
    module.pnl_batched_ddm_graph_kernel[grid](
        *input_tensors, *param_tensors, out,
        total_lanes, num_subjects, num_trials, num_estimates,
        MAX_STEPS=ir.max_steps, COMMON_RANDOM=bool(common_random_numbers),
        SEED=0 if seed is None else int(seed), BLOCK=block,
    )
    _sync(torch, device)
    return out.cpu().numpy()


def _run_stateful_graph_kernel(
    torch, triton, module, ir, inputs, params, num_estimates, seed, common_random_numbers, device,
) -> np.ndarray:
    graph = ir.graph
    input_tensors = _input_tensors(torch, graph, inputs, device)
    param_tensors = _param_tensors(torch, ir, params, device)
    num_params = len(params)
    num_subjects, num_trials = input_tensors[0].shape[:2]
    output_width = sum(output.width for output in graph.outputs)
    total_lanes = num_params * num_subjects * num_estimates
    out = torch.empty(
        (num_params, num_subjects, num_trials, num_estimates, output_width),
        dtype=torch.float32, device=device,
    )
    block = 128
    grid = (triton.cdiv(total_lanes, block),)
    module.pnl_batched_stateful_graph_kernel[grid](
        *input_tensors, *param_tensors, out,
        total_lanes, num_subjects, num_estimates, num_trials,
        LCA_MAX_STEPS=lca_max_steps(ir, inputs), MAX_STEPS=ir.max_steps,
        COMMON_RANDOM=bool(common_random_numbers),
        SEED=0 if seed is None else int(seed), BLOCK=block,
    )
    _sync(torch, device)
    return out.cpu().numpy()


def _import_torch_triton(interpret: bool):
    # Triton decorates its own library `@triton.jit` functions (tl.randn, philox,
    # ...) at import time, baking interpret-vs-compiled per process.  The CPU path
    # therefore must enable interpret mode *before* triton is first imported.
    if interpret:
        os.environ.setdefault("TRITON_INTERPRET", "1")
    try:
        import torch
        import triton
        import triton.language as tl
    except ImportError as error:
        raise RuntimeError(
            "The Triton batched backend requires torch and triton. "
            "Install the optional triton extra to use it."
        ) from error

    if interpret and type(tl.randn).__name__ != "InterpretedFunction":
        raise RuntimeError(
            "The triton_cpu (interpret) backend requires Triton interpret mode to be "
            "enabled before `triton` is first imported in this process. Set the "
            "environment variable TRITON_INTERPRET=1 before importing torch/triton "
            "(for example at the start of the process or test session). It cannot be "
            "enabled after a compiled (GPU) Triton run in the same process."
        )
    if not interpret and type(tl.randn).__name__ == "InterpretedFunction":
        raise RuntimeError(
            "The triton (GPU) backend cannot run after Triton interpret mode was "
            "enabled in this process (TRITON_INTERPRET=1). Run GPU and CPU-interpret "
            "execution in separate processes."
        )
    return torch, triton


def _load_kernel_module(ir: BatchedCompositionIR, interpret: bool = False):
    if ir.graph is None or ir.graph.fusion_kind is None:
        raise ValueError(
            "The Triton batched backend requires a lowered graph with a "
            "supported fusion kind."
        )
    source = triton_graph_kernel_source(lower_to_kernel_ir(ir))
    return load_triton_kernel_module(source, ir.graph.fusion_kind, ir.model_kind, interpret=interpret)
