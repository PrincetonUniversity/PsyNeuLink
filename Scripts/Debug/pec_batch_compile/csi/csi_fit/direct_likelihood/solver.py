"""Compatibility imports for the shared first-passage backend.

CSI equations/history remain research-local; the PDE implementation is owned by
the model-independent batched compiler numerical backend.
"""

import math
import torch

from psyneulink.core.batched.numerical.first_passage import (
    DDMBatchResult,
    MovingBoundaryDDMSolver,
    differentiable_tridiagonal_solve,
    solve_tridiagonal_pcr,
)

__all__ = [
    "DDMBatchResult", "MovingBoundaryDDMSolver",
    "differentiable_tridiagonal_solve", "solve_tridiagonal_pcr",
    "wiener_choice_density",
]


def wiener_choice_density(
    time: torch.Tensor,
    *,
    drift: torch.Tensor | float,
    boundary: torch.Tensor | float,
    noise: float = 0.1,
    starting_value: float = 0.0,
    upper: bool = True,
    terms: int = 400,
) -> torch.Tensor:
    """Analytic fixed-boundary Wiener first-passage density.

    This sine-series form is intended as a numerical oracle for moderate and
    long times.  Very short times require the complementary image series.
    """
    if terms < 1:
        raise ValueError("terms must be positive.")
    dtype, device = time.dtype, time.device
    v = torch.as_tensor(drift, dtype=dtype, device=device)
    a = torch.as_tensor(boundary, dtype=dtype, device=device)
    sigma2 = noise ** 2
    width = 2.0 * a
    start = torch.as_tensor(starting_value, dtype=dtype, device=device) + a
    k = torch.arange(1, terms + 1, dtype=dtype, device=device)
    time_e = time[..., None]
    sine = torch.sin(math.pi * k * start[..., None] / width[..., None])
    decay = torch.exp(
        -(k * math.pi).square() * sigma2 * time_e
        / (2.0 * width[..., None].square())
    )
    if upper:
        signs = torch.where(
            (torch.arange(1, terms + 1, device=device) % 2) == 1,
            torch.ones(terms, dtype=dtype, device=device),
            -torch.ones(terms, dtype=dtype, device=device),
        )
        series = torch.sum(k * sine * signs * decay, dim=-1)
        distance = width - start
    else:
        series = torch.sum(k * sine * decay, dim=-1)
        distance = -start
    prefactor = math.pi * sigma2 / width.square()
    exponential = torch.exp(
        v * distance / sigma2 - v.square() * time / (2.0 * sigma2)
    )
    return prefactor * exponential * series
