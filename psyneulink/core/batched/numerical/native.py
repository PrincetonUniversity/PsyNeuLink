"""Lazy compiled scalar first-passage kernels; no CSI imports or equations."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import shutil

import torch


def native_kernels_available() -> bool:
    """Whether the optional fused CPU backend can be built or loaded."""
    compiler = os.environ.get("CXX", "c++")
    return (
        shutil.which("ninja") is not None
        and shutil.which(compiler) is not None
    )


@lru_cache(maxsize=1)
def _native_module():
    """Build or load the CPU PDE extension outside the source tree."""
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("first_passage_cpu.cpp")
    return load(
        name="pnl_first_passage_cpu_v1",
        sources=[str(source)],
        extra_cflags=["-O3", "-DNDEBUG", "-fopenmp"],
        extra_ldflags=["-fopenmp"],
        verbose=False,
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
        _native_module().ddm_forward(
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
        _native_module().ddm_backward(
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
