from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

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
    module = _load_kernel_module(ir.model_kind)

    if ir.model_kind == DDM_MODEL:
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
    elif ir.model_kind == STABILITY_FLEXIBILITY_MODEL:
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
    else:
        raise ValueError(f"Unknown batched model kind '{ir.model_kind}'.")

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


def _load_kernel_module(model_kind: str):
    cache_dir = Path(os.environ.get("PNL_TRITON_CACHE_DIR", Path(tempfile.gettempdir()) / "psyneulink_triton_batch"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    source = _kernel_source()
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    module_path = cache_dir / f"pnl_batched_{model_kind}_{digest}.py"
    if not module_path.exists():
        module_path.write_text(source, encoding="utf-8")

    module_name = f"pnl_batched_{model_kind}_{digest}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _kernel_source() -> str:
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
