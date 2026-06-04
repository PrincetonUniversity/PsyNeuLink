from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from psyneulink.core.batched.graph import (
    DDM_GRAPH_FUSION,
    STATELESS_GRAPH_FUSION,
    STATEFUL_GRAPH_FUSION,
    projection_inputs,
)
from psyneulink.core.batched.ir import BatchedCompositionIR, BatchedSimulationResult
from psyneulink.core.batched.reference import (
    _stability_flexibility_lca_max_steps,
    normalize_parameter_sets,
    prepare_inputs,
)
from psyneulink.core.batched.registry import DDM_MODEL, STABILITY_FLEXIBILITY_MODEL


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
    elif fusion_kind == STABILITY_FLEXIBILITY_MODEL:
        values = _run_stability_flexibility_kernel(
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


def _run_stability_flexibility_kernel(
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
    task = torch.as_tensor(inputs["task"], dtype=torch.float32, device="cuda").contiguous()
    stimulus = torch.as_tensor(inputs["stimulus"], dtype=torch.float32, device="cuda").contiguous()
    cue = torch.as_tensor(inputs["cue"], dtype=torch.float32, device="cuda").contiguous()
    correct = torch.as_tensor(inputs["correct"], dtype=torch.float32, device="cuda").contiguous()
    param_tensors = _stability_flexibility_param_tensors(torch, params)
    num_params = len(params)
    num_subjects, num_trials, _ = inputs["task"].shape
    total_lanes = num_params * num_subjects * num_estimates
    out = torch.empty((num_params, num_subjects, num_trials, num_estimates, 2), dtype=torch.float32, device="cuda")
    block = 128
    grid = (triton.cdiv(total_lanes, block),)
    module.pnl_batched_stability_flexibility_kernel[grid](
        task,
        stimulus,
        cue,
        correct,
        *param_tensors,
        out,
        total_lanes,
        num_subjects,
        num_estimates,
        num_trials,
        LCA_MAX_STEPS=_stability_flexibility_lca_max_steps(ir, inputs),
        DDM_MAX_STEPS=ir.max_steps,
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
        LCA_MAX_STEPS=_stability_flexibility_lca_max_steps(ir, inputs),
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


def _stability_flexibility_param_tensors(torch, params: list[dict[str, float]]):
    names = (
        "gain",
        "leak",
        "competition",
        "self_excitation",
        "lca_noise",
        "lca_time_step_size",
        "automaticity",
        "scale",
        "starting_value",
        "threshold",
        "ddm_noise",
        "ddm_time_step_size",
        "non_decision_time",
        "ddm_offset",
    )
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


def _load_kernel_module(ir: BatchedCompositionIR):
    cache_dir = Path(os.environ.get("PNL_TRITON_CACHE_DIR", Path(tempfile.gettempdir()) / "psyneulink_triton_batch"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    source = _kernel_source(ir)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    module_kind = None if ir.graph is None else ir.graph.fusion_kind
    module_path = cache_dir / f"pnl_batched_{module_kind or ir.model_kind}_{digest}.py"
    if not module_path.exists():
        module_path.write_text(source, encoding="utf-8")

    module_name = f"pnl_batched_{module_kind or ir.model_kind}_{digest}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _stateless_graph_kernel_source(ir: BatchedCompositionIR) -> str:
    graph = ir.graph
    input_args = [f"input_{idx}" for idx, _ in enumerate(graph.inputs)]
    param_args = [f"param_{idx}" for idx, _ in enumerate(ir.params)]
    signature_args = input_args + param_args + [
        "out",
        "total_lanes: tl.constexpr",
        "num_subjects: tl.constexpr",
        "num_trials: tl.constexpr",
        "num_estimates: tl.constexpr",
        "BLOCK: tl.constexpr",
    ]
    lines = [
        "import triton",
        "import triton.language as tl",
        "",
        "",
        "@triton.jit",
        "def pnl_batched_stateless_graph_kernel(",
    ]
    for idx, arg in enumerate(signature_args):
        suffix = "," if idx < len(signature_args) - 1 else ""
        lines.append(f"    {arg}{suffix}")
    lines.extend(
        [
            "):",
            "    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
            "    mask = offsets < total_lanes",
            "    estimate_idx = offsets % num_estimates",
            "    tmp = offsets // num_estimates",
            "    trial_idx = tmp % num_trials",
            "    tmp = tmp // num_trials",
            "    subject_idx = tmp % num_subjects",
            "    param_idx = tmp // num_subjects",
            "",
        ]
    )

    param_vars = {}
    for idx, param_spec in enumerate(ir.params):
        var = f"param_{idx}_value"
        param_vars[param_spec.name] = var
        lines.append(
            f"    {var} = tl.load(param_{idx} + param_idx, mask=mask, other={_float_literal(param_spec.default)})"
        )
    if ir.params:
        lines.append("")

    input_index = {input_spec.node: idx for idx, input_spec in enumerate(graph.inputs)}
    value_vars: dict[tuple[str, str], list[str]] = {}
    for node_name in graph.execution_order:
        node = graph.node(node_name)
        projections = projection_inputs(graph, node.name)
        input_values: list[str]
        if projections:
            projected = []
            for proj_idx, projection in enumerate(projections):
                sender_values = value_vars[(projection.sender, projection.sender_port)]
                projected_components = []
                for col_idx in range(projection.matrix.shape[1]):
                    terms = []
                    for row_idx, sender_var in enumerate(sender_values):
                        coeff = float(projection.matrix[row_idx, col_idx])
                        if coeff:
                            terms.append(f"({sender_var}) * {_float_literal(coeff)}")
                    projected_components.append(" + ".join(terms) if terms else "tl.zeros((BLOCK,), dtype=tl.float32)")
                projected.append(projected_components)

            input_values = []
            for col_idx in range(node.input_width):
                components = [projection[col_idx] for projection in projected]
                if node.combine == "product":
                    expr = " * ".join(f"({component})" for component in components)
                else:
                    expr = " + ".join(f"({component})" for component in components)
                input_values.append(expr or "tl.zeros((BLOCK,), dtype=tl.float32)")
        else:
            input_spec = graph.inputs[input_index[node.name]]
            base = f"(subject_idx * num_trials + trial_idx) * {input_spec.width}"
            input_values = [
                f"tl.load(input_{input_index[node.name]} + {base} + {idx}, mask=mask, other=0.0)"
                for idx in range(input_spec.width)
            ]

        output_port = _primary_output_port_name(node)
        if node.function_type == "Linear":
            slope = param_vars[node.params["slope"]]
            intercept = param_vars[node.params["intercept"]]
            output_values = [f"({slope}) * ({input_value}) + ({intercept})" for input_value in input_values]
        elif node.function_type == "Logistic":
            gain = param_vars[node.params["gain"]]
            output_values = [f"1.0 / (1.0 + tl.exp(-({gain}) * ({input_value})))" for input_value in input_values]
        else:
            raise ValueError(f"Unsupported stateless Triton function '{node.function_type}'.")
        value_vars[(node.name, output_port)] = [f"{_safe_ident(node.name)}_{idx}" for idx in range(len(output_values))]
        for idx, expr in enumerate(output_values):
            lines.append(f"    {value_vars[(node.name, output_port)][idx]} = {expr}")
        lines.append("")

    output_width = sum(output.width for output in graph.outputs)
    lines.append(
        "    lane_out = (((param_idx * num_subjects + subject_idx) * num_trials + trial_idx) * "
        f"num_estimates + estimate_idx) * {output_width}"
    )
    cursor = 0
    for output in graph.outputs:
        source_values = value_vars[(output.node, output.port)]
        for idx in range(output.width):
            lines.append(f"    tl.store(out + lane_out + {cursor + idx}, {source_values[idx]}, mask=mask)")
        cursor += output.width
    lines.append("")
    return "\n".join(lines)


def _ddm_graph_kernel_source(ir: BatchedCompositionIR) -> str:
    graph = ir.graph
    input_args = [f"input_{idx}" for idx, _ in enumerate(graph.inputs)]
    param_args = [f"param_{idx}" for idx, _ in enumerate(ir.params)]
    signature_args = input_args + param_args + [
        "out",
        "total_lanes: tl.constexpr",
        "num_subjects: tl.constexpr",
        "num_trials: tl.constexpr",
        "num_estimates: tl.constexpr",
        "MAX_STEPS: tl.constexpr",
        "COMMON_RANDOM: tl.constexpr",
        "SEED: tl.constexpr",
        "BLOCK: tl.constexpr",
    ]
    lines = [
        "import triton",
        "import triton.language as tl",
        "",
        "",
        "@triton.jit",
        "def pnl_batched_ddm_graph_kernel(",
    ]
    for idx, arg in enumerate(signature_args):
        suffix = "," if idx < len(signature_args) - 1 else ""
        lines.append(f"    {arg}{suffix}")
    lines.extend(
        [
            "):",
            "    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
            "    mask = offsets < total_lanes",
            "    estimate_idx = offsets % num_estimates",
            "    tmp = offsets // num_estimates",
            "    trial_idx = tmp % num_trials",
            "    tmp = tmp // num_trials",
            "    subject_idx = tmp % num_subjects",
            "    param_idx = tmp // num_subjects",
            "",
        ]
    )

    param_vars = {}
    for idx, param_spec in enumerate(ir.params):
        var = f"param_{idx}_value"
        param_vars[param_spec.name] = var
        lines.append(
            f"    {var} = tl.load(param_{idx} + param_idx, mask=mask, other={_float_literal(param_spec.default)})"
        )
    if ir.params:
        lines.append("")

    input_index = {input_spec.node: idx for idx, input_spec in enumerate(graph.inputs)}
    value_vars: dict[tuple[str, str], list[str]] = {}
    for node_name in graph.execution_order:
        node = graph.node(node_name)
        projections = projection_inputs(graph, node.name)
        if projections:
            projected = []
            for projection in projections:
                sender_values = value_vars[(projection.sender, projection.sender_port)]
                projected_components = []
                for col_idx in range(projection.matrix.shape[1]):
                    terms = []
                    for row_idx, sender_var in enumerate(sender_values):
                        coeff = float(projection.matrix[row_idx, col_idx])
                        if coeff:
                            terms.append(f"({sender_var}) * {_float_literal(coeff)}")
                    projected_components.append(" + ".join(terms) if terms else "tl.zeros((BLOCK,), dtype=tl.float32)")
                projected.append(projected_components)

            input_values = []
            for col_idx in range(node.input_width):
                components = [projection[col_idx] for projection in projected]
                if node.combine == "product":
                    expr = " * ".join(f"({component})" for component in components)
                else:
                    expr = " + ".join(f"({component})" for component in components)
                input_values.append(expr or "tl.zeros((BLOCK,), dtype=tl.float32)")
        else:
            input_spec = graph.inputs[input_index[node.name]]
            base = f"(subject_idx * num_trials + trial_idx) * {input_spec.width}"
            input_values = [
                f"tl.load(input_{input_index[node.name]} + {base} + {idx}, mask=mask, other=0.0)"
                for idx in range(input_spec.width)
            ]

        if node.component_type in {"TransferMechanism", "ProcessingMechanism"}:
            output_port = _primary_output_port_name(node)
            if node.function_type == "Linear":
                slope = param_vars[node.params["slope"]]
                intercept = param_vars[node.params["intercept"]]
                output_values = [f"({slope}) * ({input_value}) + ({intercept})" for input_value in input_values]
            elif node.function_type == "Logistic":
                gain = param_vars[node.params["gain"]]
                output_values = [f"1.0 / (1.0 + tl.exp(-({gain}) * ({input_value})))" for input_value in input_values]
            else:
                raise ValueError(f"Unsupported Triton function '{node.function_type}' in DDM graph.")
            value_vars[(node.name, output_port)] = [f"{_safe_ident(node.name)}_{idx}" for idx in range(len(output_values))]
            for idx, expr in enumerate(output_values):
                lines.append(f"    {value_vars[(node.name, output_port)][idx]} = {expr}")
            lines.append("")
            continue

        if node.component_type != "DDM":
            raise ValueError(f"Unsupported Triton node '{node.component_type}' in DDM graph.")

        drift_input = input_values[0]
        rate = param_vars[node.params["rate"]]
        noise = param_vars[node.params["noise"]]
        threshold = param_vars[node.params["threshold"]]
        non_decision_time = param_vars[node.params["non_decision_time"]]
        dt = param_vars[node.params["time_step_size"]]
        starting_value = param_vars[node.params["starting_value"]]
        step_offset = param_vars[node.params["offset"]]
        safe_name = _safe_ident(node.name)
        decision_var = f"{safe_name}_decision"
        response_time_var = f"{safe_name}_response_time"
        lines.extend(
            [
                f"    {safe_name}_value = {starting_value}",
                f"    {safe_name}_steps = tl.zeros((BLOCK,), dtype=tl.float32)",
                f"    {safe_name}_sqrt_dt = tl.sqrt({dt})",
                f"    {safe_name}_boundary_tolerance = tl.maximum(1.0e-7, {threshold} * 1.0e-6)",
                "    if COMMON_RANDOM:",
                "        random_base = ((subject_idx * num_estimates + estimate_idx) * num_trials + trial_idx) * MAX_STEPS",
                "    else:",
                "        random_base = (((param_idx * num_subjects + subject_idx) * num_estimates + estimate_idx) * num_trials + trial_idx) * MAX_STEPS",
                "    for step in tl.range(0, MAX_STEPS, 1, loop_unroll_factor=1):",
                f"        {safe_name}_active = tl.abs({safe_name}_value) + {safe_name}_boundary_tolerance < {threshold}",
                "        random_draw = tl.randn(SEED, random_base + step)",
                (
                    f"        {safe_name}_updated = {safe_name}_value + ({rate}) * ({drift_input}) * ({dt}) "
                    f"+ ({noise}) * {safe_name}_sqrt_dt * random_draw"
                ),
                f"        {safe_name}_updated = tl.minimum(tl.maximum({safe_name}_updated + ({step_offset}), -({threshold})), {threshold})",
                f"        {safe_name}_value = tl.where({safe_name}_active, {safe_name}_updated, {safe_name}_value)",
                f"        {safe_name}_steps += tl.where({safe_name}_active, 1.0, 0.0)",
                f"    {decision_var} = tl.where({safe_name}_value > 0.0, 1.0, 0.0)",
                f"    {response_time_var} = ({non_decision_time}) + {safe_name}_steps * ({dt})",
                "",
            ]
        )
        value_vars[(node.name, "DECISION_OUTCOME")] = [decision_var]
        value_vars[(node.name, "RESPONSE_TIME")] = [response_time_var]
        value_vars[(node.name, _primary_output_port_name(node))] = [decision_var]

    output_width = sum(output.width for output in graph.outputs)
    lines.append(
        "    lane_out = (((param_idx * num_subjects + subject_idx) * num_trials + trial_idx) * "
        f"num_estimates + estimate_idx) * {output_width}"
    )
    cursor = 0
    for output in graph.outputs:
        source_values = value_vars[(output.node, output.port)]
        for idx in range(output.width):
            lines.append(f"    tl.store(out + lane_out + {cursor + idx}, {source_values[idx]}, mask=mask)")
        cursor += output.width
    lines.append("")
    return "\n".join(lines)


def _stateful_graph_kernel_source(ir: BatchedCompositionIR) -> str:
    graph = ir.graph
    input_args = [f"input_{idx}" for idx, _ in enumerate(graph.inputs)]
    param_args = [f"param_{idx}" for idx, _ in enumerate(ir.params)]
    signature_args = input_args + param_args + [
        "out",
        "total_lanes: tl.constexpr",
        "num_subjects: tl.constexpr",
        "num_estimates: tl.constexpr",
        "num_trials",
        "LCA_MAX_STEPS: tl.constexpr",
        "MAX_STEPS: tl.constexpr",
        "COMMON_RANDOM: tl.constexpr",
        "SEED: tl.constexpr",
        "BLOCK: tl.constexpr",
    ]
    lines = [
        "import triton",
        "import triton.language as tl",
        "",
        "",
        "@triton.jit",
        "def pnl_batched_stateful_graph_kernel(",
    ]
    for idx, arg in enumerate(signature_args):
        suffix = "," if idx < len(signature_args) - 1 else ""
        lines.append(f"    {arg}{suffix}")
    lines.extend(
        [
            "):",
            "    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
            "    mask = offsets < total_lanes",
            "    estimate_idx = offsets % num_estimates",
            "    tmp = offsets // num_estimates",
            "    subject_idx = tmp % num_subjects",
            "    param_idx = tmp // num_subjects",
            "",
        ]
    )

    param_vars = {}
    for idx, param_spec in enumerate(ir.params):
        var = f"param_{idx}_value"
        param_vars[param_spec.name] = var
        lines.append(
            f"    {var} = tl.load(param_{idx} + param_idx, mask=mask, other={_float_literal(param_spec.default)})"
        )
    if ir.params:
        lines.append("")

    state_vars = {}
    for state in graph.states:
        safe_state = _safe_ident(state.name)
        for idx, value in enumerate(state.initial_value):
            var = f"{safe_state}_{idx}"
            state_vars[(state.name, idx)] = var
            lines.append(f"    {var} = tl.full((BLOCK,), {_float_literal(value)}, tl.float32)")
    if graph.states:
        lines.append("")

    lca_stream_index = {}
    ddm_stream_index = {}
    lca_stream_count = 0
    ddm_stream_count = 0
    for node_name in graph.execution_order:
        node = graph.node(node_name)
        if node.component_type == "LCAMechanism":
            lca_stream_index[node.name] = lca_stream_count
            lca_stream_count += node.output_width
        elif node.component_type == "DDM":
            ddm_stream_index[node.name] = ddm_stream_count
            ddm_stream_count += 1

    input_index = {input_spec.node: idx for idx, input_spec in enumerate(graph.inputs)}

    def raw_input_value(node_name: str, component_idx: int = 0) -> str:
        if node_name in input_index:
            input_spec = graph.inputs[input_index[node_name]]
            base = f"(subject_idx * num_trials + trial_idx) * {input_spec.width}"
            return f"tl.load(input_{input_index[node_name]} + {base} + {component_idx}, mask=mask, other=0.0)"
        node = graph.node(node_name)
        return value_vars[(node_name, _primary_output_port_name(node))][component_idx]

    def node_input_values(node) -> list[str]:
        projections = projection_inputs(graph, node.name)
        if projections:
            projected = []
            for projection in projections:
                sender_values = value_vars[(projection.sender, projection.sender_port)]
                projected_components = []
                for col_idx in range(projection.matrix.shape[1]):
                    terms = []
                    for row_idx, sender_var in enumerate(sender_values):
                        coeff = float(projection.matrix[row_idx, col_idx])
                        if coeff:
                            terms.append(f"({sender_var}) * {_float_literal(coeff)}")
                    projected_components.append(
                        " + ".join(terms) if terms else "tl.zeros((BLOCK,), dtype=tl.float32)"
                    )
                projected.append(projected_components)

            values = []
            for col_idx in range(node.input_width):
                components = [projection[col_idx] for projection in projected]
                if node.combine == "product":
                    expr = " * ".join(f"({component})" for component in components)
                else:
                    expr = " + ".join(f"({component})" for component in components)
                values.append(expr or "tl.zeros((BLOCK,), dtype=tl.float32)")
            return values

        input_spec = graph.inputs[input_index[node.name]]
        base = f"(subject_idx * num_trials + trial_idx) * {input_spec.width}"
        return [
            f"tl.load(input_{input_index[node.name]} + {base} + {idx}, mask=mask, other=0.0)"
            for idx in range(input_spec.width)
        ]

    random_stride = (
        f"({lca_stream_count}) * LCA_MAX_STEPS + ({ddm_stream_count}) * MAX_STEPS"
    )
    lines.extend(
        [
            "    trial_idx = 0",
            "    while trial_idx < num_trials:",
            f"        random_stride = {random_stride}",
            "        if COMMON_RANDOM:",
            "            random_base = ((subject_idx * num_estimates + estimate_idx) * num_trials + trial_idx) * random_stride",
            "        else:",
            "            random_base = (((param_idx * num_subjects + subject_idx) * num_estimates + estimate_idx) * num_trials + trial_idx) * random_stride",
            "",
        ]
    )

    value_vars: dict[tuple[str, str], list[str]] = {}
    for node_name in graph.execution_order:
        node = graph.node(node_name)
        input_values = node_input_values(node)
        indent = "        "

        if node.component_type in {"TransferMechanism", "ProcessingMechanism"}:
            output_port = _primary_output_port_name(node)
            if node.function_type == "Linear":
                slope = param_vars[node.params["slope"]]
                intercept = param_vars[node.params["intercept"]]
                output_values = [f"({slope}) * ({input_value}) + ({intercept})" for input_value in input_values]
            elif node.function_type == "Logistic":
                gain = param_vars[node.params["gain"]]
                output_values = [f"1.0 / (1.0 + tl.exp(-({gain}) * ({input_value})))" for input_value in input_values]
            else:
                raise ValueError(f"Unsupported Triton function '{node.function_type}' in stateful graph.")
            value_vars[(node.name, output_port)] = [f"{_safe_ident(node.name)}_{idx}" for idx in range(len(output_values))]
            for idx, expr in enumerate(output_values):
                lines.append(f"{indent}{value_vars[(node.name, output_port)][idx]} = {expr}")
            lines.append("")
            continue

        if node.component_type == "LCAMechanism":
            if node.output_width != 2:
                raise ValueError(f"Stateful Triton graph supports LCAMechanism width 2, got {node.output_width}.")
            pre0 = state_vars[(f"{node.name}.pre", 0)]
            pre1 = state_vars[(f"{node.name}.pre", 1)]
            act0 = state_vars[(f"{node.name}.act", 0)]
            act1 = state_vars[(f"{node.name}.act", 1)]
            gain = param_vars[node.params["gain"]]
            leak = param_vars[node.params["leak"]]
            competition = param_vars[node.params["competition"]]
            self_excitation = param_vars[node.params["self_excitation"]]
            noise = param_vars[node.params["noise"]]
            dt = param_vars[node.params["time_step_size"]]
            termination_node = node.attrs.get("termination_input_node")
            if termination_node:
                cue_value = raw_input_value(termination_node)
            else:
                cue_value = _float_literal(node.attrs.get("termination_threshold", 1.0))
            stream0 = lca_stream_index[node.name]
            stream1 = stream0 + 1
            lines.extend(
                [
                    f"{indent}{_safe_ident(node.name)}_lca_steps = tl.minimum(tl.maximum(tl.ceil({cue_value}), 0.0), LCA_MAX_STEPS)",
                    f"{indent}{_safe_ident(node.name)}_sqrt_dt = tl.sqrt({dt})",
                    f"{indent}for step in tl.range(0, LCA_MAX_STEPS, 1, loop_unroll_factor=1):",
                    f"{indent}    active_lca = step < {_safe_ident(node.name)}_lca_steps",
                    f"{indent}    rec0 = ({self_excitation}) * {act0} - ({competition}) * {act1}",
                    f"{indent}    rec1 = -({competition}) * {act0} + ({self_excitation}) * {act1}",
                    f"{indent}    n0 = tl.randn(SEED, random_base + ({stream0}) * LCA_MAX_STEPS + step)",
                    f"{indent}    n1 = tl.randn(SEED, random_base + ({stream1}) * LCA_MAX_STEPS + step)",
                    (
                        f"{indent}    upd0 = (({input_values[0]}) + rec0 - ({leak}) * {pre0}) * ({dt}) "
                        f"+ ({noise}) * {_safe_ident(node.name)}_sqrt_dt * n0"
                    ),
                    (
                        f"{indent}    upd1 = (({input_values[1]}) + rec1 - ({leak}) * {pre1}) * ({dt}) "
                        f"+ ({noise}) * {_safe_ident(node.name)}_sqrt_dt * n1"
                    ),
                    f"{indent}    {pre0} = tl.where(active_lca, {pre0} + upd0, {pre0})",
                    f"{indent}    {pre1} = tl.where(active_lca, {pre1} + upd1, {pre1})",
                    f"{indent}    {act0} = tl.where(active_lca, 1.0 / (1.0 + tl.exp(-({gain}) * {pre0})), {act0})",
                    f"{indent}    {act1} = tl.where(active_lca, 1.0 / (1.0 + tl.exp(-({gain}) * {pre1})), {act1})",
                    "",
                ]
            )
            value_vars[(node.name, _primary_output_port_name(node))] = [act0, act1]
            continue

        if node.component_type == "DDM":
            drift_input = input_values[0]
            rate = param_vars[node.params["rate"]]
            noise = param_vars[node.params["noise"]]
            threshold = param_vars[node.params["threshold"]]
            non_decision_time = param_vars[node.params["non_decision_time"]]
            dt = param_vars[node.params["time_step_size"]]
            starting_value = param_vars[node.params["starting_value"]]
            step_offset = param_vars[node.params["offset"]]
            safe_name = _safe_ident(node.name)
            decision_var = f"{safe_name}_decision"
            response_time_var = f"{safe_name}_response_time"
            ddm_random_base = (
                f"random_base + ({lca_stream_count}) * LCA_MAX_STEPS "
                f"+ ({ddm_stream_index[node.name]}) * MAX_STEPS"
            )
            lines.extend(
                [
                    f"{indent}{safe_name}_value = {starting_value}",
                    f"{indent}{safe_name}_steps = tl.zeros((BLOCK,), dtype=tl.float32)",
                    f"{indent}{safe_name}_sqrt_dt = tl.sqrt({dt})",
                    f"{indent}{safe_name}_boundary_tolerance = tl.maximum(1.0e-7, {threshold} * 1.0e-6)",
                    f"{indent}for step in tl.range(0, MAX_STEPS, 1, loop_unroll_factor=1):",
                    f"{indent}    {safe_name}_active = tl.abs({safe_name}_value) + {safe_name}_boundary_tolerance < {threshold}",
                    f"{indent}    random_draw = tl.randn(SEED, {ddm_random_base} + step)",
                    (
                        f"{indent}    {safe_name}_updated = {safe_name}_value + ({rate}) * ({drift_input}) * ({dt}) "
                        f"+ ({noise}) * {safe_name}_sqrt_dt * random_draw"
                    ),
                    f"{indent}    {safe_name}_updated = tl.minimum(tl.maximum({safe_name}_updated + ({step_offset}), -({threshold})), {threshold})",
                    f"{indent}    {safe_name}_value = tl.where({safe_name}_active, {safe_name}_updated, {safe_name}_value)",
                    f"{indent}    {safe_name}_steps += tl.where({safe_name}_active, 1.0, 0.0)",
                    f"{indent}{decision_var} = tl.where({safe_name}_value > 0.0, 1.0, 0.0)",
                    f"{indent}{response_time_var} = ({non_decision_time}) + {safe_name}_steps * ({dt})",
                    "",
                ]
            )
            value_vars[(node.name, "DECISION_OUTCOME")] = [decision_var]
            value_vars[(node.name, "RESPONSE_TIME")] = [response_time_var]
            value_vars[(node.name, _primary_output_port_name(node))] = [decision_var]
            continue

        raise ValueError(f"Unsupported Triton node '{node.component_type}' in stateful graph.")

    output_width = sum(output.width for output in graph.outputs)
    lines.append(
        "        lane_out = (((param_idx * num_subjects + subject_idx) * num_trials + trial_idx) * "
        f"num_estimates + estimate_idx) * {output_width}"
    )
    cursor = 0
    for output in graph.outputs:
        source_values = value_vars[(output.node, output.port)]
        for idx in range(output.width):
            lines.append(f"        tl.store(out + lane_out + {cursor + idx}, {source_values[idx]}, mask=mask)")
        cursor += output.width
    lines.extend(["        trial_idx += 1", ""])
    return "\n".join(lines)


def _primary_output_port_name(node) -> str:
    output_ports = tuple(node.attrs.get("output_ports", ()))
    if output_ports:
        return output_ports[0]
    return "RESULT"


def _safe_ident(name: str) -> str:
    return "n_" + "".join(ch if ch.isalnum() else "_" for ch in name)


def _float_literal(value: float) -> str:
    return repr(float(value))


def _kernel_source(ir: BatchedCompositionIR) -> str:
    if ir.graph is not None and ir.graph.fusion_kind == STATELESS_GRAPH_FUSION:
        return _stateless_graph_kernel_source(ir)
    if ir.graph is not None and ir.graph.fusion_kind == DDM_GRAPH_FUSION:
        return _ddm_graph_kernel_source(ir)
    if ir.graph is not None and ir.graph.fusion_kind == STATEFUL_GRAPH_FUSION:
        return _stateful_graph_kernel_source(ir)

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


@triton.jit
def pnl_batched_stability_flexibility_kernel(
    task,
    stimulus,
    cue,
    correct,
    gains,
    leaks,
    competitions,
    self_excitations,
    lca_noises,
    lca_time_step_sizes,
    automaticities,
    scales,
    starting_values,
    thresholds,
    ddm_noises,
    ddm_time_step_sizes,
    non_decision_times,
    ddm_offsets,
    out,
    total_lanes: tl.constexpr,
    num_subjects: tl.constexpr,
    num_estimates: tl.constexpr,
    num_trials,
    LCA_MAX_STEPS: tl.constexpr,
    DDM_MAX_STEPS: tl.constexpr,
    COMMON_RANDOM: tl.constexpr,
    SEED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_lanes

    estimate_idx = offsets % num_estimates
    tmp = offsets // num_estimates
    subject_idx = tmp % num_subjects
    param_idx = tmp // num_subjects

    gain = tl.load(gains + param_idx, mask=mask, other=1.0)
    leak = tl.load(leaks + param_idx, mask=mask, other=1.0)
    competition = tl.load(competitions + param_idx, mask=mask, other=0.0)
    self_excitation = tl.load(self_excitations + param_idx, mask=mask, other=0.0)
    lca_noise = tl.load(lca_noises + param_idx, mask=mask, other=0.0)
    lca_dt = tl.load(lca_time_step_sizes + param_idx, mask=mask, other=0.01)
    automaticity = tl.load(automaticities + param_idx, mask=mask, other=0.0)
    scale = tl.load(scales + param_idx, mask=mask, other=1.0)
    starting_value = tl.load(starting_values + param_idx, mask=mask, other=0.0)
    threshold = tl.load(thresholds + param_idx, mask=mask, other=1.0)
    ddm_noise = tl.load(ddm_noises + param_idx, mask=mask, other=0.0)
    ddm_dt = tl.load(ddm_time_step_sizes + param_idx, mask=mask, other=0.01)
    non_decision_time = tl.load(non_decision_times + param_idx, mask=mask, other=0.0)
    ddm_offset = tl.load(ddm_offsets + param_idx, mask=mask, other=0.0)

    lca_pre0 = tl.zeros((BLOCK,), dtype=tl.float32)
    lca_pre1 = tl.zeros((BLOCK,), dtype=tl.float32)
    lca_act0 = tl.zeros((BLOCK,), dtype=tl.float32)
    lca_act1 = tl.zeros((BLOCK,), dtype=tl.float32)

    trial_idx = 0
    while trial_idx < num_trials:
        trial_base = subject_idx * num_trials + trial_idx
        task0 = tl.load(task + trial_base * 2, mask=mask, other=0.0)
        task1 = tl.load(task + trial_base * 2 + 1, mask=mask, other=0.0)
        stim0 = tl.load(stimulus + trial_base * 2, mask=mask, other=0.0)
        stim1 = tl.load(stimulus + trial_base * 2 + 1, mask=mask, other=0.0)
        cue_value = tl.load(cue + trial_base, mask=mask, other=0.0)
        correct_value = tl.load(correct + trial_base, mask=mask, other=1.0)
        lca_steps = tl.minimum(tl.maximum(tl.ceil(cue_value), 0.0), LCA_MAX_STEPS)
        sqrt_lca_dt = tl.sqrt(lca_dt)
        random_stride = LCA_MAX_STEPS * 2 + DDM_MAX_STEPS

        if COMMON_RANDOM:
            random_base = ((subject_idx * num_estimates + estimate_idx) * num_trials + trial_idx) * random_stride
        else:
            random_base = (((param_idx * num_subjects + subject_idx) * num_estimates + estimate_idx) * num_trials + trial_idx) * random_stride

        for step in tl.range(0, LCA_MAX_STEPS, 1, loop_unroll_factor=1):
            active_lca = step < lca_steps
            rec0 = self_excitation * lca_act0 - competition * lca_act1
            rec1 = -competition * lca_act0 + self_excitation * lca_act1
            n0 = tl.randn(SEED, random_base + step)
            n1 = tl.randn(SEED, random_base + LCA_MAX_STEPS + step)
            upd0 = (task0 + rec0 - leak * lca_pre0) * lca_dt + lca_noise * sqrt_lca_dt * n0
            upd1 = (task1 + rec1 - leak * lca_pre1) * lca_dt + lca_noise * sqrt_lca_dt * n1
            lca_pre0 = tl.where(active_lca, lca_pre0 + upd0, lca_pre0)
            lca_pre1 = tl.where(active_lca, lca_pre1 + upd1, lca_pre1)
            lca_act0 = tl.where(active_lca, 1.0 / (1.0 + tl.exp(-gain * lca_pre0)), lca_act0)
            lca_act1 = tl.where(active_lca, 1.0 / (1.0 + tl.exp(-gain * lca_pre1)), lca_act1)

        drift = (stim0 * lca_act0 + stim1 * lca_act1 + automaticity * (stim0 + stim1)) * scale * correct_value
        value = starting_value
        steps = tl.zeros((BLOCK,), dtype=tl.float32)
        sqrt_ddm_dt = tl.sqrt(ddm_dt)
        boundary_tolerance = tl.maximum(1.0e-7, threshold * 1.0e-6)
        for step in tl.range(0, DDM_MAX_STEPS, 1, loop_unroll_factor=1):
            active_ddm = tl.abs(value) + boundary_tolerance < threshold
            random_draw = tl.randn(SEED, random_base + 2 * LCA_MAX_STEPS + step)
            updated = value + drift * ddm_dt + ddm_noise * sqrt_ddm_dt * random_draw
            updated = tl.minimum(tl.maximum(updated + ddm_offset, -threshold), threshold)
            value = tl.where(active_ddm, updated, value)
            steps += tl.where(active_ddm, 1.0, 0.0)

        lane_out = (((param_idx * num_subjects + subject_idx) * num_trials + trial_idx) * num_estimates + estimate_idx) * 2
        tl.store(out + lane_out, tl.where(value > 0.0, 1.0, 0.0), mask=mask)
        tl.store(out + lane_out + 1, non_decision_time + steps * ddm_dt, mask=mask)
        trial_idx += 1
'''
