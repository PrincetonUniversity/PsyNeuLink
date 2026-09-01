"""Lazy native kernels for the research-local direct-likelihood prototype."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import shutil

import torch


def native_lca_available() -> bool:
    """Whether the optional PyTorch extension build helper is installed."""
    compiler = os.environ.get("CXX", "c++")
    return (
        shutil.which("ninja") is not None
        and shutil.which(compiler) is not None
    )


@lru_cache(maxsize=1)
def _native_lca_module():
    """Build or load the small CPU extension outside the source tree."""
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("native_lca.cpp")
    return load(
        name="csi_direct_likelihood_native_lca_v3",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG", "-fopenmp"],
        extra_ldflags=["-fopenmp"],
        verbose=False,
    )


class _NativeLCASubjectScan(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        task,
        gain,
        csi_duration,
        state_duration,
        csi_steps,
        state_steps,
        iti_duration,
        iti_steps,
        leak,
        competition,
    ):
        module = _native_lca_module()
        onset, after, history = module.forward(
            task.contiguous(),
            gain.contiguous(),
            csi_duration.contiguous(),
            state_duration.contiguous(),
            csi_steps.contiguous(),
            state_steps.contiguous(),
            iti_duration,
            iti_steps,
            leak,
            competition,
        )
        ctx.module = module
        ctx.iti_duration = iti_duration
        ctx.iti_steps = iti_steps
        ctx.leak = leak
        ctx.competition = competition
        ctx.save_for_backward(
            history,
            task,
            gain,
            csi_duration,
            state_duration,
            csi_steps,
            state_steps,
        )
        return onset, after

    @staticmethod
    def backward(ctx, gradient_onset, gradient_after):
        (
            history,
            task,
            gain,
            csi_duration,
            state_duration,
            csi_steps,
            state_steps,
        ) = ctx.saved_tensors
        if gradient_onset is None:
            gradient_onset = torch.zeros(
                (task.shape[0], 2), dtype=task.dtype, device=task.device
            )
        if gradient_after is None:
            gradient_after = torch.zeros(
                (task.shape[0], 2), dtype=task.dtype, device=task.device
            )
        gradients = ctx.module.backward(
            history,
            task.contiguous(),
            gain.contiguous(),
            csi_duration.contiguous(),
            state_duration.contiguous(),
            csi_steps.contiguous(),
            state_steps.contiguous(),
            gradient_onset.contiguous(),
            gradient_after.contiguous(),
            ctx.iti_duration,
            ctx.iti_steps,
            ctx.leak,
            ctx.competition,
        )
        return (*gradients, None, None, None, None, None, None)


def native_lca_subject_scan(
    task: torch.Tensor,
    gain: torch.Tensor,
    csi_duration: torch.Tensor,
    state_duration: torch.Tensor,
    csi_steps: torch.Tensor,
    state_steps: torch.Tensor,
    *,
    iti_duration: float,
    iti_steps: int,
    leak: float,
    competition: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return decision-onset and post-trial states for a complete participant."""
    if task.device.type != "cpu":
        raise ValueError("The native LCA subject scan currently supports only CPU tensors.")
    return _NativeLCASubjectScan.apply(
        task,
        gain,
        csi_duration,
        state_duration,
        csi_steps,
        state_steps,
        iti_duration,
        iti_steps,
        leak,
        competition,
    )


class _NativeLCADriftPath(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        state,
        task,
        gain,
        stimulus,
        correct_response,
        steps,
        step_size,
        leak,
        competition,
    ):
        module = _native_lca_module()
        drift, final_state, history = module.drift_forward(
            state.contiguous(),
            task.contiguous(),
            gain.contiguous(),
            stimulus.contiguous(),
            correct_response.contiguous(),
            steps,
            step_size,
            leak,
            competition,
        )
        ctx.module = module
        ctx.steps = steps
        ctx.step_size = step_size
        ctx.leak = leak
        ctx.competition = competition
        ctx.save_for_backward(
            history, task, gain, stimulus, correct_response
        )
        return drift, final_state

    @staticmethod
    def backward(ctx, gradient_drift, gradient_final_state):
        history, task, gain, stimulus, correct_response = ctx.saved_tensors
        if gradient_drift is None:
            gradient_drift = torch.zeros(
                (task.shape[0], ctx.steps),
                dtype=task.dtype,
                device=task.device,
            )
        if gradient_final_state is None:
            gradient_final_state = torch.zeros(
                (task.shape[0], 2), dtype=task.dtype, device=task.device
            )
        gradients = ctx.module.drift_backward(
            history,
            task.contiguous(),
            gain.contiguous(),
            stimulus.contiguous(),
            correct_response.contiguous(),
            gradient_drift.contiguous(),
            gradient_final_state.contiguous(),
            ctx.steps,
            ctx.step_size,
            ctx.leak,
            ctx.competition,
        )
        return (*gradients, None, None, None, None)


def native_lca_drift_path(
    state: torch.Tensor,
    task: torch.Tensor,
    gain: torch.Tensor,
    stimulus: torch.Tensor,
    correct_response: torch.Tensor,
    *,
    steps: int,
    step_size: float,
    leak: float,
    competition: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return batched midpoint drifts from one fused native CPU call."""
    if state.device.type != "cpu":
        raise ValueError("The native LCA drift path currently supports only CPU tensors.")
    return _NativeLCADriftPath.apply(
        state,
        task,
        gain,
        stimulus,
        correct_response,
        steps,
        step_size,
        leak,
        competition,
    )


def native_ddm_forward(
    drift: torch.Tensor,
    threshold: torch.Tensor,
    collapse_rate: torch.Tensor,
    interval_low: torch.Tensor,
    interval_high: torch.Tensor,
    choice: torch.Tensor,
    *,
    time_step: float,
    spatial_points: int,
    noise: float,
    boundary_floor: float,
    rannacher_steps: int,
    store_history: bool,
) -> tuple[torch.Tensor, ...]:
    """Run the moving-boundary DDM time loop in one native CPU call."""
    if drift.device.type != "cpu":
        raise ValueError("The native DDM forward solve currently supports only CPU tensors.")
    return tuple(
        _native_lca_module().ddm_forward(
            drift.contiguous(),
            threshold.contiguous(),
            collapse_rate.contiguous(),
            interval_low.contiguous(),
            interval_high.contiguous(),
            choice.contiguous(),
            time_step,
            spatial_points,
            noise,
            boundary_floor,
            rannacher_steps,
            store_history,
        )
    )


def native_ddm_backward(
    history: torch.Tensor,
    drift: torch.Tensor,
    threshold: torch.Tensor,
    collapse_rate: torch.Tensor,
    interval_low: torch.Tensor,
    interval_high: torch.Tensor,
    choice: torch.Tensor,
    invalid: torch.Tensor,
    gradient_probability: torch.Tensor,
    *,
    time_step: float,
    spatial_points: int,
    noise: float,
    rannacher_steps: int,
) -> tuple[torch.Tensor, ...]:
    """Apply the native implicit adjoint to a stored DDM density history."""
    return tuple(
        _native_lca_module().ddm_backward(
            history.contiguous(),
            drift.contiguous(),
            threshold.contiguous(),
            collapse_rate.contiguous(),
            interval_low.contiguous(),
            interval_high.contiguous(),
            choice.contiguous(),
            invalid.contiguous(),
            gradient_probability.contiguous(),
            time_step,
            spatial_points,
            noise,
            rannacher_steps,
        )
    )
