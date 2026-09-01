"""Independent first-passage checks for the synthetic recovery generator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from .native import native_lca_available
from .solver import MovingBoundaryDDMSolver


@dataclass(frozen=True)
class PrescribedDDMCase:
    """A smooth drift and linear symmetric boundary used in validation."""

    name: str
    threshold: float
    collapse_rate: float
    drift_offset: float
    drift_amplitude: float = 0.0
    drift_decay: float = 1.0

    def drift(self, time: np.ndarray) -> np.ndarray:
        """Evaluate drift at the supplied times."""
        return self.drift_offset + self.drift_amplitude * np.exp(
            -time / self.drift_decay
        )


@dataclass(frozen=True)
class FirstPassageSamples:
    """Discrete first-passage samples; choice 1 is the upper boundary."""

    crossing_time: np.ndarray
    choice: np.ndarray


DEFAULT_GENERATOR_CASES = (
    PrescribedDDMCase(
        name="fixed_drift_fixed_boundary",
        threshold=0.12,
        collapse_rate=0.0,
        drift_offset=0.03,
    ),
    PrescribedDDMCase(
        name="decaying_drift_collapsing_boundary",
        threshold=0.15,
        collapse_rate=-0.035,
        drift_offset=0.005,
        drift_amplitude=0.06,
        drift_decay=0.30,
    ),
)


def simulate_prescribed_first_passage(
    case: PrescribedDDMCase,
    *,
    paths: int,
    time_step: float,
    maximum_time: float,
    noise: float = 0.1,
    seed: int = 1,
    bridge_correction: bool = False,
    chunk_size: int = 50_000,
) -> FirstPassageSamples:
    """Simulate a prescribed DDM with endpoint or bridge crossing detection.

    The endpoint method mirrors a conventional Euler--Maruyama generator. The
    optional Brownian-bridge correction also samples crossings that occur and
    return between adjacent endpoints. Linear moving-boundary endpoints are
    used in the bridge calculation; drift cancels after conditioning on the
    two evidence endpoints.
    """
    if paths < 1:
        raise ValueError("paths must be positive.")
    if time_step <= 0.0 or maximum_time <= 0.0 or noise <= 0.0:
        raise ValueError("Time and noise settings must be positive.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    steps = int(math.ceil(maximum_time / time_step))
    final_boundary = case.threshold + case.collapse_rate * steps * time_step
    if final_boundary <= 0.0:
        raise ValueError("The boundary collapses within the simulation horizon.")

    rng = np.random.default_rng(seed)
    crossing_time = np.full(paths, np.inf, dtype=np.float64)
    choice = np.full(paths, -1, dtype=np.int8)
    variance = noise * noise * time_step
    noise_scale = math.sqrt(variance)

    for chunk_start in range(0, paths, chunk_size):
        chunk_stop = min(paths, chunk_start + chunk_size)
        active_ids = np.arange(chunk_start, chunk_stop)
        evidence = np.zeros(chunk_stop - chunk_start, dtype=np.float64)
        for step_index in range(steps):
            if active_ids.size == 0:
                break
            t0 = step_index * time_step
            t1 = (step_index + 1) * time_step
            midpoint = t0 + 0.5 * time_step
            boundary_start = case.threshold + case.collapse_rate * t0
            boundary_end = case.threshold + case.collapse_rate * t1
            old_evidence = evidence
            new_evidence = old_evidence + (
                case.drift(np.asarray(midpoint)) * time_step
                + noise_scale * rng.standard_normal(active_ids.size)
            )
            upper_crossing = new_evidence >= boundary_end
            lower_crossing = new_evidence <= -boundary_end

            if bridge_correction:
                inside = ~(upper_crossing | lower_crossing)
                inside_indices = np.flatnonzero(inside)
                if inside_indices.size:
                    old_inside = old_evidence[inside_indices]
                    new_inside = new_evidence[inside_indices]
                    upper_probability = np.exp(
                        -2.0
                        * (boundary_start - old_inside)
                        * (boundary_end - new_inside)
                        / variance
                    )
                    lower_probability = np.exp(
                        -2.0
                        * (boundary_start + old_inside)
                        * (boundary_end + new_inside)
                        / variance
                    )
                    total_probability = upper_probability + lower_probability
                    bridge_crossing = (
                        rng.random(inside_indices.size)
                        < np.minimum(1.0, total_probability)
                    )
                    bridge_indices = inside_indices[bridge_crossing]
                    if bridge_indices.size:
                        local_upper = upper_probability[bridge_crossing]
                        local_total = total_probability[bridge_crossing]
                        bridge_upper = (
                            rng.random(bridge_indices.size) * local_total
                            < local_upper
                        )
                        upper_crossing[bridge_indices] = bridge_upper
                        lower_crossing[bridge_indices] = ~bridge_upper

            crossed = upper_crossing | lower_crossing
            if np.any(crossed):
                crossed_ids = active_ids[crossed]
                crossing_time[crossed_ids] = t1
                choice[crossed_ids] = upper_crossing[crossed].astype(np.int8)
            retained = ~crossed
            active_ids = active_ids[retained]
            evidence = new_evidence[retained]

    return FirstPassageSamples(crossing_time=crossing_time, choice=choice)


def _pde_reference(
    case: PrescribedDDMCase,
    cutoffs: np.ndarray,
    *,
    time_step: float,
    spatial_points: int,
    noise: float,
) -> dict[str, object]:
    maximum_time = float(np.max(cutoffs))
    steps = int(math.ceil(maximum_time / time_step))
    midpoint_times = (np.arange(steps) + 0.5) * time_step
    drift_values = case.drift(midpoint_times)
    batch = len(cutoffs)
    dtype = torch.float64
    solver = MovingBoundaryDDMSolver(
        time_step=time_step,
        spatial_points=spatial_points,
        noise=noise,
        native_forward=native_lca_available(),
    )
    with torch.no_grad():
        result = solver.solve_observation_batch(
            drift=torch.as_tensor(
                np.broadcast_to(drift_values, (batch, steps)).copy(),
                dtype=dtype,
            ),
            threshold=torch.full((batch,), case.threshold, dtype=dtype),
            collapse_rate=torch.full(
                (batch,), case.collapse_rate, dtype=dtype
            ),
            interval_low=torch.zeros(batch, dtype=dtype),
            interval_high=torch.as_tensor(cutoffs, dtype=dtype),
            choice=torch.ones(batch, dtype=dtype),
        )
    return {
        "upper_cdf": result.upper_probability.numpy(),
        "lower_cdf": result.lower_probability.numpy(),
        "survival_probability": result.survival_probability.numpy(),
        "maximum_mass_error": float(torch.amax(result.mass_error)),
        "minimum_density": float(torch.amin(result.minimum_density)),
    }


def _sample_comparison(
    samples: FirstPassageSamples,
    cutoffs: np.ndarray,
    reference: dict[str, object],
) -> dict[str, object]:
    paths = len(samples.choice)
    simulated = {
        "upper_cdf": np.asarray(
            [
                np.mean((samples.choice == 1) & (samples.crossing_time <= cutoff))
                for cutoff in cutoffs
            ]
        ),
        "lower_cdf": np.asarray(
            [
                np.mean((samples.choice == 0) & (samples.crossing_time <= cutoff))
                for cutoff in cutoffs
            ]
        ),
    }
    errors = []
    standard_errors = []
    rows = []
    for boundary_name in ("upper_cdf", "lower_cdf"):
        reference_values = np.asarray(reference[boundary_name])
        sample_values = simulated[boundary_name]
        boundary_errors = sample_values - reference_values
        boundary_se = np.sqrt(sample_values * (1.0 - sample_values) / paths)
        errors.extend(boundary_errors.tolist())
        standard_errors.extend(boundary_se.tolist())
        for index, cutoff in enumerate(cutoffs):
            rows.append(
                {
                    "boundary": boundary_name.removesuffix("_cdf"),
                    "cutoff": float(cutoff),
                    "pde_probability": reference_values[index],
                    "sample_probability": sample_values[index],
                    "error": boundary_errors[index],
                    "monte_carlo_standard_error": boundary_se[index],
                    "absolute_standardized_error": (
                        abs(boundary_errors[index]) / boundary_se[index]
                        if boundary_se[index] > 0.0
                        else float("inf")
                    ),
                }
            )
    error_array = np.asarray(errors)
    standard_error_array = np.asarray(standard_errors)
    finite_standard_error = standard_error_array > 0.0
    return {
        "rows": rows,
        "maximum_absolute_error": float(np.max(np.abs(error_array))),
        "root_mean_squared_error": float(np.sqrt(np.mean(error_array ** 2))),
        "maximum_absolute_standardized_error": float(
            np.max(
                np.abs(error_array[finite_standard_error])
                / standard_error_array[finite_standard_error]
            )
        ),
        "uncrossed_fraction": float(np.mean(samples.choice < 0)),
    }


def generator_validation_report(
    *,
    paths: int = 50_000,
    seed: int = 9182,
    simulation_time_steps: tuple[float, ...] = (0.002, 0.001, 0.0005),
    reference_time_step: float = 0.00025,
    reference_spatial_points: int = 257,
    cutoffs: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5),
    noise: float = 0.1,
) -> dict[str, object]:
    """Compare seeded first-passage samples with a fine PDE reference."""
    cutoff_array = np.asarray(cutoffs, dtype=np.float64)
    if np.any(np.diff(cutoff_array) <= 0.0) or cutoff_array[0] <= 0.0:
        raise ValueError("cutoffs must be positive and strictly increasing.")
    if not simulation_time_steps:
        raise ValueError("At least one simulation time step is required.")
    cases = []
    for case_index, case in enumerate(DEFAULT_GENERATOR_CASES):
        reference = _pde_reference(
            case,
            cutoff_array,
            time_step=reference_time_step,
            spatial_points=reference_spatial_points,
            noise=noise,
        )
        comparisons = []
        for step_index, simulation_time_step in enumerate(simulation_time_steps):
            for bridge_correction in (False, True):
                samples = simulate_prescribed_first_passage(
                    case,
                    paths=paths,
                    time_step=simulation_time_step,
                    maximum_time=float(cutoff_array[-1]),
                    noise=noise,
                    seed=(
                        seed
                        + 10_000 * case_index
                        + 100 * step_index
                        + int(bridge_correction)
                    ),
                    bridge_correction=bridge_correction,
                )
                comparisons.append(
                    {
                        "method": (
                            "brownian_bridge" if bridge_correction else "endpoint"
                        ),
                        "simulation_time_step": simulation_time_step,
                        **_sample_comparison(samples, cutoff_array, reference),
                    }
                )
        cases.append(
            {
                "case": asdict(case),
                "pde_reference": reference,
                "comparisons": comparisons,
            }
        )
    return {
        "paths_per_comparison": paths,
        "seed": seed,
        "noise": noise,
        "cutoffs": cutoff_array,
        "simulation_time_steps": simulation_time_steps,
        "reference_time_step": reference_time_step,
        "reference_spatial_points": reference_spatial_points,
        "cases": cases,
    }
