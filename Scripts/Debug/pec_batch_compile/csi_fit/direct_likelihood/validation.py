"""Numerical and PNL refinement checks for the direct likelihood."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import math
import warnings

import numpy as np
import torch

from .model import ContinuousLCA, pnl_euler_lca_step
from .solver import MovingBoundaryDDMSolver, wiener_choice_density


@dataclass(frozen=True)
class FixedBoundaryValidation:
    numerical_probability: tuple[float, float]
    analytic_probability: tuple[float, float]
    relative_error: tuple[float, float]
    mass_error: float


def validate_fixed_boundary(
    *,
    time_step: float = 0.001,
    spatial_points: int = 65,
    observation_time: float = 0.4,
    observation_width: float = 0.004,
    drift: float = 0.03,
    threshold: float = 0.12,
    noise: float = 0.1,
) -> FixedBoundaryValidation:
    """Compare both boundary fluxes with the analytic Wiener density."""
    dtype = torch.float64
    steps = int(math.ceil((observation_time + observation_width / 2) / time_step))
    solver = MovingBoundaryDDMSolver(
        time_step=time_step, spatial_points=spatial_points, noise=noise
    )
    result = solver.solve_observation_batch(
        drift=torch.full((2, steps), drift, dtype=dtype),
        threshold=torch.full((2,), threshold, dtype=dtype),
        collapse_rate=torch.zeros(2, dtype=dtype),
        interval_low=torch.full(
            (2,), observation_time - observation_width / 2, dtype=dtype
        ),
        interval_high=torch.full(
            (2,), observation_time + observation_width / 2, dtype=dtype
        ),
        choice=torch.tensor([1.0, 0.0], dtype=dtype),
    )
    analytic = torch.stack(
        [
            wiener_choice_density(
                torch.tensor(observation_time, dtype=dtype),
                drift=drift,
                boundary=threshold,
                noise=noise,
                upper=upper,
            )
            for upper in (True, False)
        ]
    ) * observation_width
    relative = torch.abs(result.probability - analytic) / analytic
    return FixedBoundaryValidation(
        numerical_probability=tuple(float(value) for value in result.probability),
        analytic_probability=tuple(float(value) for value in analytic),
        relative_error=tuple(float(value) for value in relative),
        mass_error=float(torch.amax(result.mass_error)),
    )


def _torch_euler_endpoint(
    *,
    gain: float,
    task: Sequence[float],
    iti_duration: float,
    active_duration: float,
    step_size: float,
) -> np.ndarray:
    state = torch.zeros(2, dtype=torch.float64)
    gain_tensor = torch.tensor(gain, dtype=torch.float64)
    zero = torch.zeros(2, dtype=torch.float64)
    task_tensor = torch.as_tensor(task, dtype=torch.float64)
    for _ in range(round(iti_duration / step_size)):
        state = pnl_euler_lca_step(
            state, zero, gain_tensor, step_size=step_size
        )
    for _ in range(round(active_duration / step_size)):
        state = pnl_euler_lca_step(
            state, task_tensor, gain_tensor, step_size=step_size
        )
    return state.numpy()


def pnl_lca_endpoint(
    *,
    gain: float,
    task: Sequence[float],
    iti_duration: float,
    active_duration: float,
    step_size: float,
) -> np.ndarray:
    """Execute the original PNL LCA recurrence and return its internal state."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import psyneulink as pnl

    source = pnl.ProcessingMechanism(input_shapes=2)
    lca = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(gain=gain),
        leak=12.0,
        competition=3.0,
        self_excitation=0.0,
        noise=0.0,
        time_step_size=step_size,
        execute_until_finished=False,
    )
    composition = pnl.Composition()
    composition.add_linear_processing_pathway([source, lca])
    inputs = [
        [[0.0, 0.0]] for _ in range(round(iti_duration / step_size))
    ]
    inputs.extend(
        [list(task)] for _ in range(round(active_duration / step_size))
    )
    if inputs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            composition.run(inputs={source: inputs})
    return np.asarray(
        lca.integrator_function.parameters.previous_value.get(composition),
        dtype=float,
    ).reshape(2)


def lca_refinement_report(
    *,
    step_sizes: Sequence[float] = (0.01, 0.005, 0.0025, 0.00125),
    gain: float = 10.0,
    task: Sequence[float] = (1.0, 0.0),
    iti_duration: float = 1.0,
    active_duration: float = 0.4,
    run_pnl: bool = True,
) -> list[dict[str, object]]:
    """Show convergence of PNL's Euler states to the continuous LCA ODE."""
    dtype = torch.float64
    lca = ContinuousLCA()
    reference = torch.zeros(2, dtype=dtype)
    reference = lca.integrate(
        reference,
        torch.zeros(2, dtype=dtype),
        torch.tensor(gain, dtype=dtype),
        iti_duration,
        max_step=min(step_sizes) / 8.0,
    )
    reference = lca.integrate(
        reference,
        torch.as_tensor(task, dtype=dtype),
        torch.tensor(gain, dtype=dtype),
        active_duration,
        max_step=min(step_sizes) / 8.0,
    )
    reference_array = reference.numpy()
    report = []
    for step_size in step_sizes:
        euler = _torch_euler_endpoint(
            gain=gain,
            task=task,
            iti_duration=iti_duration,
            active_duration=active_duration,
            step_size=step_size,
        )
        row: dict[str, object] = {
            "step_size": step_size,
            "continuous_state": reference_array.tolist(),
            "euler_state": euler.tolist(),
            "euler_error": float(np.linalg.norm(euler - reference_array)),
        }
        if run_pnl:
            pnl_state = pnl_lca_endpoint(
                gain=gain,
                task=task,
                iti_duration=iti_duration,
                active_duration=active_duration,
                step_size=step_size,
            )
            row.update(
                pnl_state=pnl_state.tolist(),
                pnl_euler_error=float(np.linalg.norm(pnl_state - euler)),
                pnl_continuous_error=float(np.linalg.norm(pnl_state - reference_array)),
            )
        report.append(row)
    return report


def validation_report(*, run_pnl: bool = True) -> dict[str, object]:
    return {
        "fixed_boundary": asdict(validate_fixed_boundary()),
        "lca_refinement": lca_refinement_report(run_pnl=run_pnl),
    }
