"""Deterministic likelihood for PNL's discrete endpoint-crossing DDM."""

from __future__ import annotations

import math

import torch

from .solver import DDMBatchResult


class EndpointCrossingDDMSolver:
    """Propagate a Gaussian Euler random walk and absorb at step endpoints.

    PNL's legacy DDM advances evidence once per scheduler pass and tests the
    threshold only after that update. This solver evaluates the corresponding
    discrete-time Markov process deterministically. A fixed evidence grid is
    diffused and shifted with an FFT heat kernel; fractional edge cells reduce
    error where a moving boundary cuts through the grid.

    This is a forward diagnostic, not yet a fitting backend. In particular,
    the number of 10 ms steps remains a discrete topology and no custom adjoint
    is provided.
    """

    def __init__(
        self,
        *,
        time_step: float = 0.01,
        spatial_points: int = 511,
        noise: float = 0.1,
        boundary_floor: float = 1.0e-5,
        evidence_domain: float = 0.35,
    ):
        if time_step <= 0.0:
            raise ValueError("time_step must be positive.")
        if spatial_points < 31 or spatial_points % 2 == 0:
            raise ValueError("spatial_points must be an odd integer at least 31.")
        if noise <= 0.0:
            raise ValueError("noise must be positive.")
        if evidence_domain <= 0.0:
            raise ValueError("evidence_domain must be positive.")
        self.time_step = float(time_step)
        self.spatial_points = int(spatial_points)
        self.noise = float(noise)
        self.boundary_floor = float(boundary_floor)
        self.evidence_domain = float(evidence_domain)

    def solve_observation_batch(
        self,
        *,
        drift: torch.Tensor,
        threshold: torch.Tensor,
        collapse_rate: torch.Tensor,
        interval_low: torch.Tensor,
        interval_high: torch.Tensor,
        choice: torch.Tensor,
    ) -> DDMBatchResult:
        """Return choice/RT-bin probabilities for a batch of drift paths."""
        if drift.ndim != 2:
            raise ValueError("drift must have shape [batch, time].")
        batch, time_steps = drift.shape
        expected = (batch,)
        for name, value in {
            "threshold": threshold,
            "collapse_rate": collapse_rate,
            "interval_low": interval_low,
            "interval_high": interval_high,
            "choice": choice,
        }.items():
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}.")

        dtype, device = drift.dtype, drift.device
        points = self.spatial_points
        domain = self.evidence_domain
        grid = torch.linspace(-domain, domain, points, dtype=dtype, device=device)
        dx = grid[1] - grid[0]
        lower_edge = grid - 0.5 * dx
        upper_edge = grid + 0.5 * dx

        # At least N empty cells are appended so Gaussian mass cannot wrap from
        # one physical edge to the other under the periodic FFT convolution.
        fft_points = 1 << int(math.ceil(math.log2(2 * points)))
        angular_frequency = 2.0 * torch.pi * torch.fft.rfftfreq(
            fft_points, d=float(dx), device=device
        ).to(dtype=dtype)
        increment_sd = self.noise * math.sqrt(self.time_step)
        diffusion = torch.exp(
            -0.5 * increment_sd**2 * angular_frequency**2
        )

        mass = torch.zeros((batch, points), dtype=dtype, device=device)
        mass[:, points // 2] = 1.0
        upper_probability = torch.zeros(batch, dtype=dtype, device=device)
        lower_probability = torch.zeros_like(upper_probability)
        observed_probability = torch.zeros_like(upper_probability)
        invalid_boundary = torch.zeros(batch, dtype=torch.bool, device=device)
        minimum_mass = torch.zeros((), dtype=dtype, device=device)
        dt = torch.as_tensor(self.time_step, dtype=dtype, device=device)

        for step_index in range(time_steps):
            step_time = (step_index + 1) * dt
            # Duration bucketing pads shorter rows to the longest drift path in
            # the bucket. Endpoint likelihood needs no transition after a row's
            # final candidate RT step; leaving it frozen also prevents a later
            # padded boundary collapse from falsely invalidating that row.
            active = step_time < interval_high
            mean_increment = drift[:, step_index] * dt
            padded_mass = torch.nn.functional.pad(
                mass, (0, fft_points - points)
            )
            transformed = torch.fft.rfft(padded_mass, n=fft_points)
            phase = torch.exp(
                -1j * mean_increment[:, None] * angular_frequency[None, :]
            )
            transitioned = torch.fft.irfft(
                transformed * diffusion[None, :] * phase,
                n=fft_points,
            )[:, :points]
            minimum_mass = torch.minimum(minimum_mass, torch.amin(transitioned))
            # Roundoff from the spectral convolution can be a few ulps below
            # zero. It is not physical probability mass.
            transitioned = torch.clamp(transitioned, min=0.0)

            boundary = threshold + collapse_rate * step_time
            step_invalid = boundary <= self.boundary_floor
            invalid_boundary = invalid_boundary | (step_invalid & active)
            safe_boundary = torch.clamp(boundary, min=self.boundary_floor)

            upper_fraction = torch.clamp(
                (upper_edge[None, :] - safe_boundary[:, None]) / dx,
                min=0.0,
                max=1.0,
            )
            lower_fraction = torch.clamp(
                (-safe_boundary[:, None] - lower_edge[None, :]) / dx,
                min=0.0,
                max=1.0,
            )
            survival_fraction = torch.clamp(
                1.0 - upper_fraction - lower_fraction,
                min=0.0,
                max=1.0,
            )
            step_upper = torch.sum(transitioned * upper_fraction, dim=1)
            step_lower = torch.sum(transitioned * lower_fraction, dim=1)
            step_upper = torch.where(
                active, step_upper, torch.zeros_like(step_upper)
            )
            step_lower = torch.where(
                active, step_lower, torch.zeros_like(step_lower)
            )
            upper_probability = upper_probability + step_upper
            lower_probability = lower_probability + step_lower

            in_observation_bin = (
                (step_time >= interval_low) & (step_time < interval_high)
            )
            selected = torch.where(choice >= 0.5, step_upper, step_lower)
            observed_probability = observed_probability + torch.where(
                in_observation_bin,
                selected,
                torch.zeros_like(selected),
            )
            updated_mass = transitioned * survival_fraction
            mass = torch.where(active[:, None], updated_mass, mass)

        survival_probability = torch.sum(mass, dim=1)
        mass_error = torch.abs(
            upper_probability + lower_probability + survival_probability - 1.0
        )
        observed_probability = torch.where(
            invalid_boundary,
            torch.zeros_like(observed_probability),
            observed_probability,
        )
        return DDMBatchResult(
            probability=observed_probability,
            upper_probability=upper_probability,
            lower_probability=lower_probability,
            survival_probability=survival_probability,
            mass_error=mass_error,
            minimum_density=minimum_mass.expand(batch),
            invalid_boundary=invalid_boundary,
        )
