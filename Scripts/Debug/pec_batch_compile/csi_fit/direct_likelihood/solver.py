"""Differentiable moving-boundary first-passage solver for the CSI DDM."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


_PCR_LEVEL_CACHE: dict[tuple[int, torch.device], tuple[tuple[int, torch.Tensor, torch.Tensor], ...]] = {}


def _pcr_levels(
    n: int, device: torch.device
) -> tuple[tuple[int, torch.Tensor, torch.Tensor], ...]:
    cache_key = (n, device)
    levels = _PCR_LEVEL_CACHE.get(cache_key)
    if levels is None:
        indices = torch.arange(n, device=device)
        generated_levels = []
        stride = 1
        while stride < n:
            generated_levels.append(
                (stride, indices >= stride, indices + stride < n)
            )
            stride *= 2
        levels = tuple(generated_levels)
        _PCR_LEVEL_CACHE[cache_key] = levels
    return levels


@dataclass(frozen=True)
class DDMBatchResult:
    probability: torch.Tensor
    upper_probability: torch.Tensor
    lower_probability: torch.Tensor
    survival_probability: torch.Tensor
    mass_error: torch.Tensor
    minimum_density: torch.Tensor
    invalid_boundary: torch.Tensor


def _shift_left(value: torch.Tensor, amount: int) -> torch.Tensor:
    if amount >= value.shape[-1]:
        return torch.zeros_like(value)
    return F.pad(value[..., :-amount], (amount, 0))


def _shift_right(value: torch.Tensor, amount: int) -> torch.Tensor:
    if amount >= value.shape[-1]:
        return torch.zeros_like(value)
    return F.pad(value[..., amount:], (0, amount))


def solve_tridiagonal_pcr(
    lower: torch.Tensor,
    diagonal: torch.Tensor,
    upper: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Solve independent tridiagonal systems with parallel cyclic reduction.

    All tensors have shape ``[..., n]``.  The first lower and final upper
    entries must be zero.  PCR uses logarithmically many tensor operations and
    remains differentiable with ordinary Torch autograd.
    """
    if not (lower.shape == diagonal.shape == upper.shape == rhs.shape):
        raise ValueError("Tridiagonal coefficient and RHS shapes must match.")
    n = diagonal.shape[-1]
    if n == 0:
        return rhs
    a, b, c, d = lower, diagonal, upper, rhs
    one = torch.ones((), dtype=diagonal.dtype, device=diagonal.device)
    levels = _pcr_levels(n, diagonal.device)
    for stride, has_left, has_right in levels:
        a_left = _shift_left(a, stride)
        b_left = _shift_left(b, stride)
        c_left = _shift_left(c, stride)
        d_left = _shift_left(d, stride)
        a_right = _shift_right(a, stride)
        b_right = _shift_right(b, stride)
        c_right = _shift_right(c, stride)
        d_right = _shift_right(d, stride)

        safe_left = torch.where(has_left, b_left, one)
        safe_right = torch.where(has_right, b_right, one)
        alpha = torch.where(has_left, -a / safe_left, 0.0)
        beta = torch.where(has_right, -c / safe_right, 0.0)
        a, b, c, d = (
            alpha * a_left,
            b + alpha * c_left + beta * a_right,
            beta * c_right,
            d + alpha * d_left + beta * d_right,
        )
    return d / b


class _DifferentiableTridiagonalSolve(torch.autograd.Function):
    """Implicit first-order derivative for a batched tridiagonal solve.

    Differentiating through every PCR reduction level creates a large autograd
    graph at every PDE time cell. For ``Ax = rhs``, reverse mode instead needs
    only ``A.T * adjoint = grad_x`` and the local outer-product entries that
    correspond to the three represented diagonals.
    """

    @staticmethod
    def forward(lower, diagonal, upper, rhs):
        solution = solve_tridiagonal_pcr(lower, diagonal, upper, rhs)
        return solution

    @staticmethod
    def setup_context(ctx, inputs, output):
        lower, diagonal, upper, _ = inputs
        ctx.save_for_backward(lower, diagonal, upper, output)

    @staticmethod
    def backward(ctx, gradient):
        lower, diagonal, upper, solution = ctx.saved_tensors
        transpose_lower = _shift_left(upper, 1)
        transpose_upper = _shift_right(lower, 1)
        adjoint = solve_tridiagonal_pcr(
            transpose_lower,
            diagonal,
            transpose_upper,
            gradient,
        )
        gradient_lower = -adjoint * _shift_left(solution, 1)
        gradient_diagonal = -adjoint * solution
        gradient_upper = -adjoint * _shift_right(solution, 1)
        return (
            gradient_lower if ctx.needs_input_grad[0] else None,
            gradient_diagonal if ctx.needs_input_grad[1] else None,
            gradient_upper if ctx.needs_input_grad[2] else None,
            adjoint if ctx.needs_input_grad[3] else None,
        )


def differentiable_tridiagonal_solve(
    lower: torch.Tensor,
    diagonal: torch.Tensor,
    upper: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Solve with a compact exact backward graph when gradients are needed."""
    if torch.is_grad_enabled() and any(
        value.requires_grad for value in (lower, diagonal, upper, rhs)
    ):
        return _DifferentiableTridiagonalSolve.apply(
            lower, diagonal, upper, rhs
        )
    return solve_tridiagonal_pcr(lower, diagonal, upper, rhs)


def _chang_cooper_delta(peclet: torch.Tensor) -> torch.Tensor:
    """Stable Chang--Cooper face weight."""
    small = torch.abs(peclet) < 1.0e-4
    series = 0.5 - peclet / 12.0 + peclet.pow(3) / 720.0
    safe = torch.where(small, torch.ones_like(peclet), peclet)
    exact = 1.0 / safe - 1.0 / torch.expm1(safe)
    return torch.where(small, series, exact)


class MovingBoundaryDDMSolver:
    """Forward Fokker--Planck solver on a normalized fixed spatial grid.

    ``drift`` is supplied at time-step midpoints with shape ``[batch, time]``.
    Boundaries are symmetric and linear in physical time.  Returned observation
    probabilities are boundary flux integrated over each row's RT interval.
    """

    def __init__(
        self,
        *,
        time_step: float = 0.001,
        spatial_points: int = 65,
        noise: float = 0.1,
        starting_value: float = 0.0,
        boundary_floor: float = 1.0e-5,
        rannacher_steps: int = 2,
        checkpoint_steps: int = 32,
        compile_steps: bool = False,
        custom_adjoint: bool = False,
        native_forward: bool = False,
    ):
        if time_step <= 0:
            raise ValueError("time_step must be positive.")
        if spatial_points < 5 or spatial_points % 2 == 0:
            raise ValueError("spatial_points must be an odd integer of at least 5.")
        if noise <= 0:
            raise ValueError("noise must be positive.")
        if starting_value != 0.0:
            raise ValueError(
                "The CSI prototype currently supports only a zero DDM starting value."
            )
        if checkpoint_steps < 0:
            raise ValueError("checkpoint_steps cannot be negative.")
        self.time_step = float(time_step)
        self.spatial_points = int(spatial_points)
        self.noise = float(noise)
        self.starting_value = float(starting_value)
        self.boundary_floor = float(boundary_floor)
        self.rannacher_steps = int(rannacher_steps)
        self.checkpoint_steps = int(checkpoint_steps)
        self.compile_steps = bool(compile_steps)
        self.custom_adjoint = bool(custom_adjoint)
        self.native_forward = bool(native_forward)
        self._grid_cache: dict[
            tuple[torch.dtype, torch.device], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._step_function = (
            torch.compile(self._advance_step, fullgraph=True, dynamic=True)
            if self.compile_steps
            else self._advance_step
        )
        self._adjoint_step_function = (
            torch.compile(
                self._local_step_vjp, fullgraph=True, dynamic=True
            )
            if self.custom_adjoint
            else self._local_step_vjp
        )

    def _grid(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (dtype, device)
        cached = self._grid_cache.get(key)
        if cached is None:
            y = torch.linspace(
                -1.0, 1.0, self.spatial_points, dtype=dtype, device=device
            )
            dy = y[1] - y[0]
            cached = (dy, 0.5 * (y[:-1] + y[1:]))
            self._grid_cache[key] = cached
        return cached

    def _operator(
        self,
        drift: torch.Tensor,
        boundary: torch.Tensor,
        boundary_rate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        batch = drift.shape[0]
        dtype, device = drift.dtype, drift.device
        dy, faces = self._grid(dtype, device)
        face_velocity = (
            drift[:, None] - boundary_rate[:, None] * faces[None, :]
        ) / boundary[:, None]
        diffusion = (self.noise ** 2) / (2.0 * boundary.square())
        peclet = face_velocity * dy / diffusion[:, None]
        delta = _chang_cooper_delta(peclet)

        # M interior nodes have M+1 faces (including both absorbing faces).
        interior = self.spatial_points - 2
        left_b = face_velocity[:, :interior]
        right_b = face_velocity[:, 1:interior + 1]
        left_delta = delta[:, :interior]
        right_delta = delta[:, 1:interior + 1]
        diff_over_dy = diffusion[:, None] / dy

        lower = (left_b * (1.0 - left_delta) + diff_over_dy) / dy
        diagonal = (
            -(right_b * (1.0 - right_delta) + diff_over_dy)
            + (left_b * left_delta - diff_over_dy)
        ) / dy
        upper = (-right_b * right_delta + diff_over_dy) / dy
        lower = torch.cat((torch.zeros((batch, 1), dtype=dtype, device=device), lower[:, 1:]), dim=1)
        upper = torch.cat((upper[:, :-1], torch.zeros((batch, 1), dtype=dtype, device=device)), dim=1)
        boundary_terms = (
            left_b[:, 0], left_delta[:, 0], right_b[:, -1], right_delta[:, -1],
            diffusion, dy,
        )
        return lower, diagonal, upper, boundary_terms

    @staticmethod
    def _apply_tridiagonal(
        lower: torch.Tensor,
        diagonal: torch.Tensor,
        upper: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        left = torch.cat((torch.zeros_like(value[:, :1]), value[:, :-1]), dim=1)
        right = torch.cat((value[:, 1:], torch.zeros_like(value[:, :1])), dim=1)
        return lower * left + diagonal * value + upper * right

    @staticmethod
    def _boundary_flux(
        value: torch.Tensor,
        boundary_terms: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_b, left_delta, right_b, right_delta, diffusion, dy = boundary_terms
        lower = (-left_b * left_delta + diffusion / dy) * value[:, 0]
        upper = (right_b * (1.0 - right_delta) + diffusion / dy) * value[:, -1]
        return lower, upper

    def _advance_step(
        self,
        density: torch.Tensor,
        observed_probability: torch.Tensor,
        upper_probability: torch.Tensor,
        lower_probability: torch.Tensor,
        minimum_density: torch.Tensor,
        drift: torch.Tensor,
        threshold: torch.Tensor,
        collapse_rate: torch.Tensor,
        interval_low: torch.Tensor,
        interval_high: torch.Tensor,
        choice: torch.Tensor,
        invalid: torch.Tensor,
        t0: torch.Tensor,
        theta: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Advance one PDE cell; optionally compiled as one reusable graph."""
        dt = torch.as_tensor(
            self.time_step, dtype=density.dtype, device=density.device
        )
        t1 = t0 + dt
        midpoint_time = t0 + 0.5 * dt
        active = (t0 < interval_high) & ~invalid
        boundary = threshold + collapse_rate * midpoint_time
        safe_boundary = torch.where(
            active, boundary, torch.ones_like(boundary)
        )

        lower, diagonal, upper, boundary_terms = self._operator(
            drift,
            safe_boundary,
            collapse_rate,
        )
        rhs = density + (
            (1.0 - theta)
            * dt
            * self._apply_tridiagonal(lower, diagonal, upper, density)
        )
        next_density = differentiable_tridiagonal_solve(
            -theta * dt * lower,
            1.0 - theta * dt * diagonal,
            -theta * dt * upper,
            rhs,
        )
        next_density = torch.where(
            active[:, None], next_density, density
        )
        midpoint_density = 0.5 * (density + next_density)
        raw_lower_flux, raw_upper_flux = self._boundary_flux(
            midpoint_density, boundary_terms
        )
        minimum_density = torch.minimum(
            minimum_density, torch.amin(next_density, dim=1)
        )
        # Tiny negative boundary flux is a discretization artifact and must
        # not create a negative probability. Its size remains visible through
        # the mass and density diagnostics.
        lower_flux = torch.clamp(raw_lower_flux, min=0.0)
        upper_flux = torch.clamp(raw_upper_flux, min=0.0)
        active_float = active.to(density.dtype)
        lower_probability = (
            lower_probability + active_float * dt * lower_flux
        )
        upper_probability = (
            upper_probability + active_float * dt * upper_flux
        )

        # Select overlap branches using detached comparisons. This keeps the
        # exact forward value while giving interval endpoints a deterministic
        # zero subgradient when they lie exactly on a time-cell edge;
        # torch.clamp/minimum otherwise makes that subgradient depend on
        # whether later padded cells are present.
        overlap_left = torch.where(interval_low > t0, interval_low, t0)
        overlap_right = torch.where(interval_high < t1, interval_high, t1)
        has_overlap = overlap_right.detach() > overlap_left.detach()
        overlap = torch.where(
            has_overlap,
            overlap_right - overlap_left,
            torch.zeros_like(overlap_right),
        )
        selected_flux = torch.where(choice > 0.5, upper_flux, lower_flux)
        observed_probability = observed_probability + overlap * selected_flux
        return (
            next_density,
            observed_probability,
            upper_probability,
            lower_probability,
            minimum_density,
        )

    def _advance_likelihood_step(
        self,
        density: torch.Tensor,
        observed_probability: torch.Tensor,
        drift: torch.Tensor,
        threshold: torch.Tensor,
        collapse_rate: torch.Tensor,
        interval_low: torch.Tensor,
        interval_high: torch.Tensor,
        choice: torch.Tensor,
        invalid: torch.Tensor,
        t0: torch.Tensor,
        theta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros_like(observed_probability)
        result = self._advance_step(
            density,
            observed_probability,
            zeros,
            zeros,
            zeros,
            drift,
            threshold,
            collapse_rate,
            interval_low,
            interval_high,
            choice,
            invalid,
            t0,
            theta,
        )
        return result[0], result[1]

    def _local_step_vjp(
        self,
        density: torch.Tensor,
        observed_probability: torch.Tensor,
        drift: torch.Tensor,
        threshold: torch.Tensor,
        collapse_rate: torch.Tensor,
        interval_low: torch.Tensor,
        interval_high: torch.Tensor,
        choice: torch.Tensor,
        invalid: torch.Tensor,
        t0: torch.Tensor,
        theta: torch.Tensor,
        gradient_density: torch.Tensor,
        gradient_observed: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """VJP of one likelihood-relevant PDE step for the custom adjoint."""

        def differentiable_step(
            local_density,
            local_observed,
            local_drift,
            local_threshold,
            local_collapse_rate,
            local_interval_low,
            local_interval_high,
        ):
            return self._advance_likelihood_step(
                local_density,
                local_observed,
                local_drift,
                local_threshold,
                local_collapse_rate,
                local_interval_low,
                local_interval_high,
                choice,
                invalid,
                t0,
                theta,
            )

        _, pullback = torch.func.vjp(
            differentiable_step,
            density,
            observed_probability,
            drift,
            threshold,
            collapse_rate,
            interval_low,
            interval_high,
        )
        return pullback((gradient_density, gradient_observed))

    def _likelihood_adjoint(
        self,
        density_history: torch.Tensor,
        drift: torch.Tensor,
        threshold: torch.Tensor,
        collapse_rate: torch.Tensor,
        interval_low: torch.Tensor,
        interval_high: torch.Tensor,
        choice: torch.Tensor,
        invalid: torch.Tensor,
        gradient_probability: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        dtype, device = drift.dtype, drift.device
        if self.native_forward and device.type == "cpu":
            from .native import native_ddm_backward

            return native_ddm_backward(
                density_history,
                drift,
                threshold,
                collapse_rate,
                interval_low,
                interval_high,
                choice,
                invalid,
                gradient_probability,
                time_step=self.time_step,
                spatial_points=self.spatial_points,
                noise=self.noise,
                rannacher_steps=self.rannacher_steps,
            )
        dt = torch.as_tensor(self.time_step, dtype=dtype, device=device)
        gradient_density = torch.zeros_like(density_history[0])
        gradient_observed = torch.where(
            invalid,
            torch.zeros_like(gradient_probability),
            gradient_probability,
        )
        gradient_drift = torch.zeros_like(drift)
        gradient_threshold = torch.zeros_like(threshold)
        gradient_collapse_rate = torch.zeros_like(collapse_rate)
        gradient_interval_low = torch.zeros_like(interval_low)
        gradient_interval_high = torch.zeros_like(interval_high)
        observed = torch.zeros_like(gradient_observed)

        for step_index in range(drift.shape[1] - 1, -1, -1):
            t0 = step_index * dt
            theta = torch.as_tensor(
                1.0 if step_index < self.rannacher_steps else 0.5,
                dtype=dtype,
                device=device,
            )
            (
                gradient_density,
                gradient_observed,
                step_drift,
                step_threshold,
                step_collapse_rate,
                step_interval_low,
                step_interval_high,
            ) = self._adjoint_step_function(
                density_history[step_index],
                observed,
                drift[:, step_index],
                threshold,
                collapse_rate,
                interval_low,
                interval_high,
                choice,
                invalid,
                t0,
                theta,
                gradient_density,
                gradient_observed,
            )
            gradient_drift[:, step_index] = step_drift
            gradient_threshold = gradient_threshold + step_threshold
            gradient_collapse_rate = (
                gradient_collapse_rate + step_collapse_rate
            )
            gradient_interval_low = (
                gradient_interval_low + step_interval_low
            )
            gradient_interval_high = (
                gradient_interval_high + step_interval_high
            )
        return (
            gradient_drift,
            gradient_threshold,
            gradient_collapse_rate,
            gradient_interval_low,
            gradient_interval_high,
        )

    def _solve_observation_batch_impl(
        self,
        *,
        drift: torch.Tensor,
        threshold: torch.Tensor,
        collapse_rate: torch.Tensor,
        interval_low: torch.Tensor,
        interval_high: torch.Tensor,
        choice: torch.Tensor,
        store_density_history: bool = False,
    ) -> tuple[DDMBatchResult, torch.Tensor | None]:
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
                raise ValueError(f"{name} must have shape {expected}; got {value.shape}")

        dtype, device = drift.dtype, drift.device
        if (
            self.native_forward
            and device.type == "cpu"
            and (store_density_history or not torch.is_grad_enabled())
        ):
            from .native import native_ddm_forward

            # The compiled Torch adjoint remains the reverse-mode oracle and
            # expects these fixed-grid caches to exist before graph capture.
            self._grid(dtype, device)
            _pcr_levels(self.spatial_points - 2, device)
            native_values = native_ddm_forward(
                drift,
                threshold,
                collapse_rate,
                interval_low,
                interval_high,
                choice,
                time_step=self.time_step,
                spatial_points=self.spatial_points,
                noise=self.noise,
                boundary_floor=self.boundary_floor,
                rannacher_steps=self.rannacher_steps,
                store_history=store_density_history,
            )
            result = DDMBatchResult(*native_values[:7])
            history = native_values[7] if store_density_history else None
            return result, history

        dt = torch.as_tensor(self.time_step, dtype=dtype, device=device)
        dy, _ = self._grid(dtype, device)
        interior = self.spatial_points - 2
        # Populate the Python-side cache before torch.compile captures the
        # numerical step. The masks depend only on grid width and device.
        _pcr_levels(interior, device)
        density = torch.zeros((batch, interior), dtype=dtype, device=device)
        center = interior // 2
        density[:, center] = 1.0 / dy
        density_history = [density] if store_density_history else None

        observed_probability = torch.zeros(batch, dtype=dtype, device=device)
        upper_probability = torch.zeros_like(observed_probability)
        lower_probability = torch.zeros_like(observed_probability)
        invalid = (interval_high <= 0.0) | (interval_high <= interval_low)
        midpoint_times = (
            torch.arange(time_steps, dtype=dtype, device=device) + 0.5
        ) * dt
        active_mesh = midpoint_times[None, :] - 0.5 * dt < interval_high[:, None]
        boundary_mesh = (
            threshold[:, None]
            + collapse_rate[:, None] * midpoint_times[None, :]
        )
        invalid = invalid | torch.any(
            active_mesh & (boundary_mesh <= self.boundary_floor), dim=1
        )
        minimum_density = torch.zeros_like(observed_probability)

        def advance_block(
            block_density,
            block_observed,
            block_upper_probability,
            block_lower_probability,
            block_minimum_density,
            block_drift,
            block_threshold,
            block_collapse_rate,
            block_interval_low,
            block_interval_high,
            block_choice,
            *,
            start_step,
        ):
            for local_index in range(block_drift.shape[1]):
                step_index = start_step + local_index
                t0 = step_index * dt
                theta = torch.as_tensor(
                    1.0 if step_index < self.rannacher_steps else 0.5,
                    dtype=dtype,
                    device=device,
                )
                (
                    block_density,
                    block_observed,
                    block_upper_probability,
                    block_lower_probability,
                    block_minimum_density,
                ) = self._step_function(
                    block_density,
                    block_observed,
                    block_upper_probability,
                    block_lower_probability,
                    block_minimum_density,
                    block_drift[:, local_index],
                    block_threshold,
                    block_collapse_rate,
                    block_interval_low,
                    block_interval_high,
                    block_choice,
                    invalid,
                    t0,
                    theta,
                )
                if density_history is not None:
                    density_history.append(block_density)
            return (
                block_density,
                block_observed,
                block_upper_probability,
                block_lower_probability,
                block_minimum_density,
            )

        block_size = self.checkpoint_steps or max(1, time_steps)
        use_checkpoint = (
            self.checkpoint_steps > 0
            and torch.is_grad_enabled()
            and any(
                value.requires_grad
                for value in (
                    drift,
                    threshold,
                    collapse_rate,
                    interval_low,
                    interval_high,
                )
            )
        )
        if use_checkpoint:
            from torch.utils.checkpoint import checkpoint

        for block_start in range(0, time_steps, block_size):
            block_end = min(time_steps, block_start + block_size)
            block_arguments = (
                density,
                observed_probability,
                upper_probability,
                lower_probability,
                minimum_density,
                drift[:, block_start:block_end],
                threshold,
                collapse_rate,
                interval_low,
                interval_high,
                choice,
            )
            if use_checkpoint:
                def checkpointed_block(*arguments, current_start=block_start):
                    return advance_block(
                        *arguments,
                        start_step=current_start,
                    )

                block_result = checkpoint(
                    checkpointed_block, *block_arguments, use_reentrant=True
                )
            else:
                block_result = advance_block(
                    *block_arguments,
                    start_step=block_start,
                )
            (
                density,
                observed_probability,
                upper_probability,
                lower_probability,
                minimum_density,
            ) = block_result

        survival = torch.sum(density, dim=1) * dy
        mass_error = torch.abs(
            survival + upper_probability + lower_probability - 1.0
        )
        probability = torch.where(
            invalid, torch.zeros_like(observed_probability), observed_probability
        )
        result = DDMBatchResult(
            probability=probability,
            upper_probability=upper_probability,
            lower_probability=lower_probability,
            survival_probability=survival,
            mass_error=mass_error,
            minimum_density=minimum_density,
            invalid_boundary=invalid,
        )
        history = (
            torch.stack(density_history)
            if density_history is not None
            else None
        )
        return result, history

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
        needs_gradient = torch.is_grad_enabled() and any(
            value.requires_grad
            for value in (
                drift,
                threshold,
                collapse_rate,
                interval_low,
                interval_high,
            )
        )
        if self.custom_adjoint and needs_gradient:
            values = _DifferentiableMovingBoundarySolve.apply(
                drift,
                threshold,
                collapse_rate,
                interval_low,
                interval_high,
                choice,
                self,
            )
            return DDMBatchResult(*values)
        result, _ = self._solve_observation_batch_impl(
            drift=drift,
            threshold=threshold,
            collapse_rate=collapse_rate,
            interval_low=interval_low,
            interval_high=interval_high,
            choice=choice,
        )
        return result


class _DifferentiableMovingBoundarySolve(torch.autograd.Function):
    """Whole-solve adjoint that avoids a thousand-node autograd chain."""

    @staticmethod
    def forward(
        ctx,
        drift,
        threshold,
        collapse_rate,
        interval_low,
        interval_high,
        choice,
        solver,
    ):
        result, density_history = solver._solve_observation_batch_impl(
            drift=drift,
            threshold=threshold,
            collapse_rate=collapse_rate,
            interval_low=interval_low,
            interval_high=interval_high,
            choice=choice,
            store_density_history=True,
        )
        if density_history is None:  # pragma: no cover - defensive contract
            raise RuntimeError("The DDM adjoint requires a density history.")
        ctx.solver = solver
        ctx.save_for_backward(
            density_history,
            drift,
            threshold,
            collapse_rate,
            interval_low,
            interval_high,
            choice,
            result.invalid_boundary,
        )
        ctx.mark_non_differentiable(
            result.upper_probability,
            result.lower_probability,
            result.survival_probability,
            result.mass_error,
            result.minimum_density,
            result.invalid_boundary,
        )
        return (
            result.probability,
            result.upper_probability,
            result.lower_probability,
            result.survival_probability,
            result.mass_error,
            result.minimum_density,
            result.invalid_boundary,
        )

    @staticmethod
    def backward(
        ctx,
        gradient_probability,
        gradient_upper,
        gradient_lower,
        gradient_survival,
        gradient_mass_error,
        gradient_minimum_density,
        gradient_invalid,
    ):
        del (
            gradient_upper,
            gradient_lower,
            gradient_survival,
            gradient_mass_error,
            gradient_minimum_density,
            gradient_invalid,
        )
        (
            density_history,
            drift,
            threshold,
            collapse_rate,
            interval_low,
            interval_high,
            choice,
            invalid,
        ) = ctx.saved_tensors
        gradients = ctx.solver._likelihood_adjoint(
            density_history,
            drift,
            threshold,
            collapse_rate,
            interval_low,
            interval_high,
            choice,
            invalid,
            gradient_probability,
        )
        return (*gradients, None, None)


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
