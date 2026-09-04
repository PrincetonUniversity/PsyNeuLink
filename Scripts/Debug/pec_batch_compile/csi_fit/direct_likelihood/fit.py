"""Optimizers for the continuous CSI direct likelihood."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import multiprocessing as mp
import os
import time

import numpy as np
import torch

from .likelihood import ContinuousCSILikelihood
from .model import CSITrialData, ContinuousCSIParameters, parameter_bounds


_WORKER_LIKELIHOOD: ContinuousCSILikelihood | None = None
_WORKER_TRIALS: CSITrialData | None = None
_WORKER_PROBABILITY_FLOOR = 1.0e-300


def _initialize_score_worker(
    config,
    trials: CSITrialData,
    probability_floor: float,
) -> None:
    """Create one single-threaded likelihood evaluator per worker process."""
    global _WORKER_LIKELIHOOD, _WORKER_TRIALS, _WORKER_PROBABILITY_FLOOR
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _WORKER_LIKELIHOOD = ContinuousCSILikelihood(config)
    _WORKER_TRIALS = trials
    _WORKER_PROBABILITY_FLOOR = probability_floor


def _worker_negative_log_likelihood(value: np.ndarray) -> float:
    if _WORKER_LIKELIHOOD is None or _WORKER_TRIALS is None:
        raise RuntimeError("Likelihood score worker was not initialized.")
    dtype = _WORKER_TRIALS.response_time.dtype
    with torch.no_grad():
        parameter_tensor = torch.as_tensor(value, dtype=dtype)
        result = _WORKER_LIKELIHOOD.score_vector(
            parameter_tensor, _WORKER_TRIALS
        )
        loss = _finite_negative_log_likelihood(
            result, _WORKER_PROBABILITY_FLOOR
        )
    return float(loss)


class _PoolMap:
    """Map adapter for SciPy numerical differentiation.

    SciPy supplies its scalar objective as the first argument. Workers already
    own an equivalent initialized likelihood, so only parameter vectors cross
    process boundaries.
    """

    def __init__(self, pool):
        self.pool = pool

    def __call__(self, function, iterable):
        del function
        return self.pool.map(_worker_negative_log_likelihood, iterable)


def parallel_finite_difference_objective(
    likelihood: ContinuousCSILikelihood,
    trials: CSITrialData,
    vector: np.ndarray,
    *,
    workers: int | None = None,
    relative_step: float = 1.0e-5,
    probability_floor: float = 1.0e-300,
) -> tuple[float, np.ndarray]:
    """Evaluate an objective and bounded two-point gradient in parallel."""
    if trials.response_time.device.type != "cpu":
        raise ValueError("Parallel finite differences require CPU trial data.")
    if relative_step <= 0.0:
        raise ValueError("relative_step must be positive.")
    worker_count = workers
    if worker_count is None:
        worker_count = max(1, min(13, (os.cpu_count() or 2) - 1))
    if worker_count < 1:
        raise ValueError("workers must be positive.")

    from scipy.optimize._numdiff import approx_derivative

    value = np.asarray(vector, dtype=float)
    lower, upper = parameter_bounds()
    dtype = trials.response_time.dtype

    def scalar_objective(candidate) -> float:
        with torch.no_grad():
            parameter_tensor = torch.as_tensor(candidate, dtype=dtype)
            result = likelihood.score_vector(parameter_tensor, trials)
            loss = _finite_negative_log_likelihood(
                result, probability_floor
            )
        return float(loss)

    objective = scalar_objective(value)
    worker_config = replace(
        likelihood.config,
        compile_ddm_steps=False,
        ddm_checkpoint_steps=0,
        checkpoint_lca=False,
    )
    context = mp.get_context("spawn")
    with context.Pool(
        processes=worker_count,
        initializer=_initialize_score_worker,
        initargs=(worker_config, trials, probability_floor),
    ) as pool:
        gradient = approx_derivative(
            scalar_objective,
            value,
            method="2-point",
            rel_step=relative_step,
            bounds=(lower, upper),
            f0=objective,
            workers=_PoolMap(pool),
        )
    return objective, np.asarray(gradient, dtype=float)


@dataclass(frozen=True)
class FitResult:
    method: str
    parameter_vector: np.ndarray
    log_likelihood: float
    success: bool
    evaluations: int
    iterations: int
    projected_gradient_inf_norm: float
    stationary: bool
    coordinate_stationary: bool
    rejected_start_attempts: int
    run_results: tuple[dict[str, object], ...]
    message: str


@dataclass(frozen=True)
class StagedFitResult:
    """Coarse basin search followed by default and optional fine polishing."""

    parameter_vector: np.ndarray
    log_likelihood: float
    final_mesh: str
    coarse_result: FitResult
    default_result: FitResult
    fine_result: FitResult | None
    stage_seconds: dict[str, float]
    meshes: dict[str, dict[str, float | int]]


def _finite_negative_log_likelihood(result, probability_floor: float) -> torch.Tensor:
    probabilities = result.probability[result.included_row_indices]
    return -torch.sum(torch.log(torch.clamp(probabilities, min=probability_floor)))


def _feasible_start(
    rng: np.random.Generator,
    trials: CSITrialData,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Draw an interior start and make its RT shifts feasible."""
    fraction = rng.uniform(0.15, 0.85, size=lower.shape)
    vector = lower + fraction * (upper - lower)
    # A linked CSI+NDT constraint is not expressible as independent L-BFGS-B
    # bounds.  Pull those values toward their lower bounds until every included
    # row starts with a positive decision-time interval.
    included = trials.include.detach().cpu().numpy().astype(bool)
    condition = trials.condition_index.detach().cpu().numpy()
    switch = trials.is_switch.detach().cpu().numpy()
    rt = trials.response_time.detach().cpu().numpy()
    for _ in range(2):
        for condition_index in range(3):
            rows = included & (condition == condition_index)
            if not np.any(rows):
                continue
            allowance = np.min(rt[rows] - switch[rows] * vector[3]) - 0.002
            vector[10 + condition_index] = min(
                vector[10 + condition_index],
                max(lower[10 + condition_index], allowance),
            )
        switch_rows = included & (switch > 0.5)
        if np.any(switch_rows):
            row_ndt = vector[10 + condition[switch_rows]]
            allowance = np.min(rt[switch_rows] - row_ndt) - 0.002
            vector[3] = min(vector[3], max(lower[3], allowance))
    # A collapsing boundary must remain open through every observed decision
    # interval in its condition. This linked threshold/rate constraint cannot
    # be represented by independent box bounds.
    boundary_margin = 1.0e-4
    for condition_index in range(3):
        rows = included & (condition == condition_index)
        if not np.any(rows):
            continue
        decision_high = (
            rt[rows]
            + 0.5 * trials.rt_resolution
            - vector[10 + condition_index]
            - switch[rows] * vector[3]
        )
        maximum_time = np.max(decision_high)
        if maximum_time > 0.0:
            minimum_rate = (
                boundary_margin - vector[4 + condition_index]
            ) / maximum_time
            vector[7 + condition_index] = max(
                vector[7 + condition_index], minimum_rate
            )
    return np.clip(vector, lower + 1.0e-8, upper - 1.0e-8)


def _valid_start_score(
    likelihood: ContinuousCSILikelihood,
    trials: CSITrialData,
    vector: np.ndarray,
    probability_floor: float,
) -> tuple[bool, float, float]:
    """Return start validity, log likelihood, and its minimum row density."""
    dtype, device = trials.response_time.dtype, trials.response_time.device
    with torch.no_grad():
        parameter_tensor = torch.as_tensor(vector, dtype=dtype, device=device)
        result = likelihood.score_vector(parameter_tensor, trials)
        probabilities = result.probability[result.included_row_indices]
        loss = _finite_negative_log_likelihood(result, probability_floor)
    finite = bool(torch.all(torch.isfinite(probabilities)).detach().cpu())
    positive = bool(torch.all(probabilities > probability_floor).detach().cpu())
    finite_loss = bool(torch.isfinite(loss).detach().cpu())
    minimum = float(torch.amin(probabilities).detach().cpu())
    return finite and positive and finite_loss, -float(loss.detach().cpu()), minimum


def fit_lbfgsb(
    likelihood: ContinuousCSILikelihood,
    trials: CSITrialData,
    *,
    starts: int = 4,
    max_iterations: int = 200,
    seed: int = 1,
    initial_vectors: Sequence[np.ndarray] = (),
    include_default_start: bool = True,
    probability_floor: float = 1.0e-300,
    gradient_method: str = "autograd",
    finite_difference_workers: int | None = None,
    finite_difference_relative_step: float = 1.0e-5,
    max_start_attempts: int = 100,
    random_start_candidates: int | None = None,
    function_tolerance: float = 1.0e-12,
    gradient_tolerance: float = 1.0e-5,
    maximum_line_search_steps: int = 50,
    polish_restarts: int = 1,
    polish_iterations: int = 100,
    polish_method: str = "lbfgsb",
    coordinate_polish: bool = True,
    coordinate_step: float = 1.0e-3,
    coordinate_levels: int = 3,
    coordinate_rounds: int = 2,
    coordinate_cycles: int = 2,
    coordinate_improvement_tolerance: float = 1.0e-7,
    stationarity_tolerance: float = 1.0e-2,
) -> FitResult:
    """Run bounded, gradient-based maximum likelihood from multiple starts."""
    if starts < 1:
        raise ValueError("starts must be at least one.")
    if gradient_method not in {"autograd", "parallel-finite-difference"}:
        raise ValueError(
            "gradient_method must be 'autograd' or "
            "'parallel-finite-difference'."
        )
    if finite_difference_relative_step <= 0.0:
        raise ValueError("finite_difference_relative_step must be positive.")
    if max_start_attempts < starts:
        raise ValueError("max_start_attempts must be at least starts.")
    if random_start_candidates is not None and random_start_candidates < 0:
        raise ValueError("random_start_candidates cannot be negative.")
    if function_tolerance <= 0.0 or gradient_tolerance <= 0.0:
        raise ValueError("Optimizer tolerances must be positive.")
    if maximum_line_search_steps < 1:
        raise ValueError("maximum_line_search_steps must be positive.")
    if polish_restarts < 0 or polish_iterations < 1:
        raise ValueError("Polish settings must be nonnegative and positive.")
    if polish_method not in {"lbfgsb", "powell"}:
        raise ValueError("polish_method must be 'lbfgsb' or 'powell'.")
    if (
        coordinate_step <= 0.0
        or coordinate_levels < 1
        or coordinate_rounds < 1
        or coordinate_cycles < 1
    ):
        raise ValueError("Coordinate-polish settings must be positive.")
    if coordinate_improvement_tolerance <= 0.0:
        raise ValueError("coordinate_improvement_tolerance must be positive.")
    if stationarity_tolerance <= 0.0:
        raise ValueError("stationarity_tolerance must be positive.")
    from scipy.optimize import OptimizeResult, minimize

    lower, upper = parameter_bounds()
    parameter_scale = upper - lower
    rng = np.random.default_rng(seed)
    requested_candidates = [
        ("provided", np.asarray(value, dtype=float))
        for value in initial_vectors
    ]
    if any(candidate.shape != lower.shape for _, candidate in requested_candidates):
        raise ValueError("Every initial vector must contain exactly 13 values.")
    dtype, device = trials.response_time.dtype, trials.response_time.device
    if include_default_start and len(requested_candidates) < starts:
        default_vector = ContinuousCSIParameters.defaults(
            dtype=dtype, device=device
        ).vector()
        requested_candidates.append(
            ("default", default_vector.detach().cpu().numpy())
        )
    candidates: list[np.ndarray] = []
    candidate_metadata: list[dict[str, object]] = []
    rejected_start_attempts = 0
    attempt = 0
    for source, candidate in requested_candidates[:starts]:
        if attempt >= max_start_attempts:
            break
        attempt += 1
        candidate = np.clip(candidate, lower, upper)
        valid, initial_log_likelihood, minimum_probability = _valid_start_score(
            likelihood, trials, candidate, probability_floor
        )
        if not valid:
            rejected_start_attempts += 1
            continue
        candidates.append(candidate)
        candidate_metadata.append(
            {
                "source": source,
                "initial_log_likelihood": initial_log_likelihood,
                "initial_minimum_probability": minimum_probability,
                "initial_parameter_vector": candidate.copy(),
            }
        )
    random_slots = starts - len(candidates)
    random_pool_target = (
        random_slots
        if random_start_candidates is None
        else max(random_slots, random_start_candidates)
    )
    random_pool = []
    while len(random_pool) < random_pool_target and attempt < max_start_attempts:
        attempt += 1
        candidate = _feasible_start(rng, trials, lower, upper)
        valid, initial_log_likelihood, minimum_probability = _valid_start_score(
            likelihood, trials, candidate, probability_floor
        )
        if not valid:
            rejected_start_attempts += 1
            continue
        random_pool.append(
            (
                initial_log_likelihood,
                candidate,
                minimum_probability,
            )
        )
    random_pool.sort(key=lambda item: item[0], reverse=True)
    for pool_rank, (
        initial_log_likelihood,
        candidate,
        minimum_probability,
    ) in enumerate(random_pool[:random_slots], start=1):
        candidates.append(candidate)
        candidate_metadata.append(
            {
                "source": "screened-random",
                "random_pool_rank": pool_rank,
                "random_pool_valid_candidates": len(random_pool),
                "initial_log_likelihood": initial_log_likelihood,
                "initial_minimum_probability": minimum_probability,
                "initial_parameter_vector": candidate.copy(),
            }
        )
    if len(candidates) < starts:
        raise RuntimeError(
            f"Found only {len(candidates)} valid starts after {attempt} attempts."
        )
    results = []

    def autograd_objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        parameter_tensor = torch.tensor(
            value, dtype=dtype, device=device, requires_grad=True
        )
        result = likelihood.score_vector(parameter_tensor, trials)
        loss = _finite_negative_log_likelihood(result, probability_floor)
        loss.backward()
        gradient = parameter_tensor.grad
        if gradient is None:
            return float(loss.detach().cpu()), np.zeros_like(value)
        gradient_array = gradient.detach().cpu().numpy().astype(float, copy=False)
        return float(loss.detach().cpu()), gradient_array

    def scalar_objective(value: np.ndarray) -> float:
        with torch.no_grad():
            parameter_tensor = torch.as_tensor(
                value, dtype=dtype, device=device
            )
            result = likelihood.score_vector(parameter_tensor, trials)
            loss = _finite_negative_log_likelihood(
                result, probability_floor
            )
        return float(loss.detach().cpu())

    def scaled_autograd_objective(
        scaled_value: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        value = lower + parameter_scale * scaled_value
        objective, gradient = autograd_objective(value)
        return objective, gradient * parameter_scale

    def scaled_scalar_objective(scaled_value: np.ndarray) -> float:
        return scalar_objective(lower + parameter_scale * scaled_value)

    bounds = list(zip(lower, upper, strict=True))
    if gradient_method == "autograd":
        scaled_bounds = [(0.0, 1.0)] * len(lower)
        for candidate, metadata in zip(
            candidates, candidate_metadata, strict=True
        ):
            scaled_candidate = (
                np.clip(candidate, lower, upper) - lower
            ) / parameter_scale
            optimized = minimize(
                scaled_autograd_objective,
                scaled_candidate,
                method="L-BFGS-B",
                jac=True,
                bounds=scaled_bounds,
                options={
                    "maxiter": int(max_iterations),
                    "ftol": function_tolerance,
                    "gtol": gradient_tolerance,
                    "maxls": int(maximum_line_search_steps),
                    "maxcor": 20,
                },
            )
            initial_objective = -float(metadata["initial_log_likelihood"])
            optimized_objective = float(optimized.fun)
            retained_initial = (
                not np.isfinite(optimized_objective)
                or optimized_objective > initial_objective
            )
            metadata["optimizer_final_log_likelihood"] = -optimized_objective
            metadata["retained_initial_candidate"] = retained_initial
            if retained_initial:
                optimized = OptimizeResult(
                    x=scaled_candidate,
                    fun=initial_objective,
                    success=False,
                    nfev=optimized.nfev,
                    nit=optimized.nit,
                    message=(
                        f"{optimized.message}; retained the better scored "
                        "initial candidate."
                    ),
                )
            results.append(optimized)
    else:
        if device.type != "cpu":
            raise ValueError(
                "Parallel finite differences currently require CPU trial data."
            )
        workers = finite_difference_workers
        if workers is None:
            workers = max(1, min(13, (os.cpu_count() or 2) - 1))
        if workers < 1:
            raise ValueError("finite_difference_workers must be positive.")
        # Worker scores use eager inference. Compiling the same graph in every
        # process creates a large cold-start and provides little benefit while
        # the workers saturate separate CPU cores.
        worker_config = replace(
            likelihood.config,
            compile_ddm_steps=False,
            ddm_checkpoint_steps=0,
            checkpoint_lca=False,
        )
        context = mp.get_context("spawn")
        with context.Pool(
            processes=workers,
            initializer=_initialize_score_worker,
            initargs=(worker_config, trials, probability_floor),
        ) as pool:
            worker_map = _PoolMap(pool)
            for candidate, metadata in zip(
                candidates, candidate_metadata, strict=True
            ):
                clipped_candidate = np.clip(candidate, lower, upper)
                optimized = minimize(
                    scalar_objective,
                    clipped_candidate,
                    method="L-BFGS-B",
                    jac="2-point",
                    bounds=bounds,
                    options={
                        "maxiter": int(max_iterations),
                        "finite_diff_rel_step": (
                            finite_difference_relative_step
                        ),
                        "workers": worker_map,
                    },
                )
                initial_objective = -float(metadata["initial_log_likelihood"])
                optimized_objective = float(optimized.fun)
                retained_initial = (
                    not np.isfinite(optimized_objective)
                    or optimized_objective > initial_objective
                )
                metadata["optimizer_final_log_likelihood"] = -optimized_objective
                metadata["retained_initial_candidate"] = retained_initial
                if retained_initial:
                    optimized = OptimizeResult(
                        x=clipped_candidate,
                        fun=initial_objective,
                        success=False,
                        nfev=optimized.nfev,
                        nit=optimized.nit,
                        message=(
                            f"{optimized.message}; retained the better scored "
                            "initial candidate."
                        ),
                    )
                results.append(optimized)
    main_results = list(results)
    best = min(results, key=lambda result: float(result.fun))
    polish_results = []

    def run_polish(starting_value: np.ndarray):
        if polish_method == "lbfgsb":
            return minimize(
                scaled_autograd_objective,
                np.asarray(starting_value, dtype=float),
                method="L-BFGS-B",
                jac=True,
                bounds=scaled_bounds,
                options={
                    "maxiter": int(polish_iterations),
                    "ftol": function_tolerance * 0.1,
                    "gtol": gradient_tolerance,
                    "maxls": int(maximum_line_search_steps),
                    "maxcor": 30,
                },
            )
        return minimize(
            scaled_scalar_objective,
            np.asarray(starting_value, dtype=float),
            method="Powell",
            bounds=scaled_bounds,
            options={
                "maxiter": int(polish_iterations),
                "xtol": 1.0e-6,
                "ftol": function_tolerance,
            },
        )

    if gradient_method == "autograd":
        scaled_bounds = [(0.0, 1.0)] * len(lower)
        for _ in range(polish_restarts):
            previous_objective = float(best.fun)
            polished = run_polish(best.x)
            polish_results.append(polished)
            results.append(polished)
            if float(polished.fun) <= float(best.fun):
                best = polished
            improvement = previous_objective - float(best.fun)
            if improvement <= function_tolerance * max(1.0, abs(previous_objective)):
                break
    coordinate_results = []
    post_coordinate_results = []
    coordinate_stationary = False
    if gradient_method == "autograd" and coordinate_polish:
        def run_coordinate_poll(starting_value, starting_objective):
            coordinate_value = np.asarray(starting_value, dtype=float).copy()
            coordinate_objective = float(starting_objective)
            coordinate_evaluations = 0
            coordinate_iterations = 0
            stationary = False
            for level in range(coordinate_levels):
                step = coordinate_step * (0.1 ** level)
                for _ in range(coordinate_rounds):
                    trial_values = []
                    for index in range(len(coordinate_value)):
                        for direction in (-1.0, 1.0):
                            trial_value = coordinate_value.copy()
                            trial_value[index] = np.clip(
                                trial_value[index] + direction * step, 0.0, 1.0
                            )
                            if trial_value[index] != coordinate_value[index]:
                                trial_values.append(trial_value)
                    trial_objectives = [
                        scaled_scalar_objective(value) for value in trial_values
                    ]
                    coordinate_evaluations += len(trial_objectives)
                    best_index = int(np.argmin(trial_objectives))
                    best_trial_objective = trial_objectives[best_index]
                    improvement_threshold = max(
                        coordinate_improvement_tolerance,
                        function_tolerance
                        * max(1.0, abs(coordinate_objective)),
                    )
                    if best_trial_objective < (
                        coordinate_objective - improvement_threshold
                    ):
                        coordinate_value = trial_values[best_index]
                        coordinate_objective = best_trial_objective
                        coordinate_iterations += 1
                        continue
                    if level == coordinate_levels - 1:
                        stationary = True
                    break
            return OptimizeResult(
                x=coordinate_value,
                fun=coordinate_objective,
                success=stationary,
                nfev=coordinate_evaluations,
                nit=coordinate_iterations,
                message=(
                    "No improving coordinate move at the finest poll step."
                    if stationary
                    else "Coordinate-polish round budget completed."
                ),
            )

        for cycle in range(1, coordinate_cycles + 1):
            previous_objective = float(best.fun)
            coordinate_result = run_coordinate_poll(best.x, best.fun)
            coordinate_results.append((cycle, coordinate_result))
            results.append(coordinate_result)
            if float(coordinate_result.fun) <= float(best.fun):
                best = coordinate_result
            coordinate_stationary = bool(coordinate_result.success)
            if coordinate_stationary:
                break
            improvement_threshold = max(
                coordinate_improvement_tolerance,
                function_tolerance * max(1.0, abs(previous_objective)),
            )
            if previous_objective - float(best.fun) <= improvement_threshold:
                break
            polished = run_polish(best.x)
            post_coordinate_results.append((cycle, polished))
            results.append(polished)
            if float(polished.fun) <= float(best.fun):
                best = polished
    if gradient_method == "autograd":
        best_parameter_vector = lower + parameter_scale * np.asarray(best.x)
        _, best_gradient = autograd_objective(best_parameter_vector)
        projected_gradient = best_gradient * parameter_scale
        scaled_value = np.asarray(best.x)
        bound_tolerance = 1.0e-8
        projected_gradient[
            (scaled_value <= bound_tolerance) & (projected_gradient > 0.0)
        ] = 0.0
        projected_gradient[
            (scaled_value >= 1.0 - bound_tolerance)
            & (projected_gradient < 0.0)
        ] = 0.0
    else:
        best_parameter_vector = np.asarray(best.x, dtype=float)
        projected_gradient = np.asarray(best.jac, dtype=float).copy()
        bound_tolerance = 1.0e-8
        projected_gradient[
            (best_parameter_vector <= lower + bound_tolerance)
            & (projected_gradient > 0.0)
        ] = 0.0
        projected_gradient[
            (best_parameter_vector >= upper - bound_tolerance)
            & (projected_gradient < 0.0)
        ] = 0.0
    run_results: list[dict[str, object]] = []
    for start_index, (result, metadata) in enumerate(
        zip(main_results, candidate_metadata, strict=True), start=1
    ):
        if gradient_method == "autograd":
            result_vector = lower + parameter_scale * np.asarray(result.x)
        else:
            result_vector = np.asarray(result.x, dtype=float)
        run_results.append(
            {
                "phase": "start",
                "start_index": start_index,
                **metadata,
                "final_parameter_vector": result_vector,
                "log_likelihood": -float(result.fun),
                "success": bool(result.success),
                "evaluations": int(result.nfev),
                "iterations": int(result.nit),
                "message": str(result.message),
            }
        )
    for polish_index, result in enumerate(polish_results, start=1):
        run_results.append(
            {
                "phase": "polish",
                "polish_index": polish_index,
                "polish_method": polish_method,
                "final_parameter_vector": (
                    lower + parameter_scale * np.asarray(result.x)
                ),
                "log_likelihood": -float(result.fun),
                "success": bool(result.success),
                "evaluations": int(result.nfev),
                "iterations": int(result.nit),
                "message": str(result.message),
            }
        )
    for coordinate_cycle, result in coordinate_results:
        run_results.append(
            {
                "phase": "coordinate-polish",
                "coordinate_cycle": coordinate_cycle,
                "initial_step": coordinate_step,
                "levels": coordinate_levels,
                "rounds_per_level": coordinate_rounds,
                "final_parameter_vector": (
                    lower + parameter_scale * np.asarray(result.x)
                ),
                "log_likelihood": -float(result.fun),
                "success": bool(result.success),
                "evaluations": int(result.nfev),
                "iterations": int(result.nit),
                "message": str(result.message),
            }
        )
    for coordinate_cycle, result in post_coordinate_results:
        run_results.append(
            {
                "phase": "post-coordinate-polish",
                "coordinate_cycle": coordinate_cycle,
                "polish_method": polish_method,
                "final_parameter_vector": (
                    lower + parameter_scale * np.asarray(result.x)
                ),
                "log_likelihood": -float(result.fun),
                "success": bool(result.success),
                "evaluations": int(result.nfev),
                "iterations": int(result.nit),
                "message": str(result.message),
            }
        )
    projected_gradient_inf_norm = float(np.max(np.abs(projected_gradient)))
    return FitResult(
        method=(
            f"L-BFGS-B/{gradient_method}/unit-scaled"
            if gradient_method == "autograd"
            else f"L-BFGS-B/{gradient_method}"
        ),
        parameter_vector=best_parameter_vector,
        log_likelihood=-float(best.fun),
        success=bool(best.success),
        evaluations=int(sum(result.nfev for result in results)) + 1,
        iterations=int(sum(result.nit for result in results)),
        projected_gradient_inf_norm=projected_gradient_inf_norm,
        stationary=projected_gradient_inf_norm <= stationarity_tolerance,
        coordinate_stationary=coordinate_stationary,
        rejected_start_attempts=rejected_start_attempts,
        run_results=tuple(run_results),
        message=str(best.message),
    )


def fit_cmaes(
    likelihood: ContinuousCSILikelihood,
    trials: CSITrialData,
    *,
    evaluations: int = 500,
    seed: int = 1,
    probability_floor: float = 1.0e-300,
    initial_vectors: Sequence[np.ndarray] = (),
) -> FitResult:
    """Run bounded Optuna CMA-ES as a derivative-free cross-check."""
    if evaluations < 1:
        raise ValueError("evaluations must be at least one.")
    try:
        import optuna
    except ImportError as error:  # pragma: no cover - dependency is optional
        raise RuntimeError("CMA-ES fitting requires the optional optuna package.") from error

    from .model import parameter_names

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    lower, upper = parameter_bounds()
    names = parameter_names()
    dtype, device = trials.response_time.dtype, trials.response_time.device
    sampler = optuna.samplers.CmaEsSampler(seed=seed, sigma0=0.2)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    default = ContinuousCSIParameters.defaults(
        dtype=dtype, device=device
    ).vector().detach().cpu().numpy()
    queued_vectors = [default]
    queued_vectors.extend(np.asarray(value, dtype=float) for value in initial_vectors)
    queued_vectors.append(
        _feasible_start(np.random.default_rng(seed), trials, lower, upper)
    )
    for value in queued_vectors:
        if value.shape != lower.shape:
            raise ValueError("Every initial vector must contain exactly 13 values.")
        clipped = np.clip(value, lower, upper)
        study.enqueue_trial(dict(zip(names, clipped, strict=True)))

    def objective(optuna_trial) -> float:
        value = np.asarray(
            [
                optuna_trial.suggest_float(name, float(low), float(high))
                for name, low, high in zip(names, lower, upper, strict=True)
            ]
        )
        with torch.no_grad():
            parameter_tensor = torch.as_tensor(value, dtype=dtype, device=device)
            result = likelihood.score_vector(parameter_tensor, trials)
            loss = _finite_negative_log_likelihood(result, probability_floor)
        return float(loss.detach().cpu())

    study.optimize(objective, n_trials=int(evaluations))
    best = study.best_trial
    best_vector = np.asarray([best.params[name] for name in names], dtype=float)
    return FitResult(
        method="CMA-ES",
        parameter_vector=best_vector,
        log_likelihood=-float(best.value),
        success=True,
        evaluations=len(study.trials),
        iterations=len(study.trials),
        projected_gradient_inf_norm=float("nan"),
        stationary=False,
        coordinate_stationary=False,
        rejected_start_attempts=0,
        run_results=(),
        message="Optuna CMA-ES evaluation budget completed.",
    )


def fit_staged(
    likelihood: ContinuousCSILikelihood,
    trials: CSITrialData,
    *,
    seed: int = 1,
    initial_vectors: Sequence[np.ndarray] = (),
    coarse_ddm_time_step: float = 0.002,
    coarse_ddm_spatial_points: int = 33,
    coarse_lca_max_step: float = 0.02,
    coarse_cma_evaluations: int = 1_000,
    default_starts: int = 2,
    default_max_iterations: int = 100,
    random_start_candidates: int = 32,
    fine: bool = True,
    fine_ddm_time_step: float = 0.0005,
    fine_ddm_spatial_points: int = 129,
    fine_lca_max_step: float = 0.005,
    fine_max_iterations: int = 50,
) -> StagedFitResult:
    """Run the standard coarse-CMA, exact-gradient, fine-polish pipeline."""
    if default_starts < 1:
        raise ValueError("default_starts must be positive.")
    if default_max_iterations < 1 or fine_max_iterations < 1:
        raise ValueError("Stage iteration budgets must be positive.")
    for name, time_step, spatial_points, lca_step in (
        (
            "coarse",
            coarse_ddm_time_step,
            coarse_ddm_spatial_points,
            coarse_lca_max_step,
        ),
        (
            "fine",
            fine_ddm_time_step,
            fine_ddm_spatial_points,
            fine_lca_max_step,
        ),
    ):
        if time_step <= 0.0 or lca_step <= 0.0:
            raise ValueError(f"{name} time steps must be positive.")
        if spatial_points < 5 or spatial_points % 2 == 0:
            raise ValueError(
                f"{name} spatial points must be an odd integer of at least 5."
            )

    coarse_likelihood = ContinuousCSILikelihood(
        replace(
            likelihood.config,
            ddm_time_step=coarse_ddm_time_step,
            ddm_spatial_points=coarse_ddm_spatial_points,
            lca_max_step=coarse_lca_max_step,
        )
    )
    start = time.perf_counter()
    coarse_result = fit_cmaes(
        coarse_likelihood,
        trials,
        evaluations=coarse_cma_evaluations,
        seed=seed,
        initial_vectors=initial_vectors,
    )
    coarse_seconds = time.perf_counter() - start

    default_initial_vectors = [coarse_result.parameter_vector]
    default_initial_vectors.extend(initial_vectors)
    start = time.perf_counter()
    default_result = fit_lbfgsb(
        likelihood,
        trials,
        starts=default_starts,
        max_iterations=default_max_iterations,
        seed=seed + 1,
        initial_vectors=default_initial_vectors,
        include_default_start=True,
        random_start_candidates=random_start_candidates,
    )
    default_seconds = time.perf_counter() - start

    fine_result = None
    fine_seconds = 0.0
    if fine:
        fine_likelihood = ContinuousCSILikelihood(
            replace(
                likelihood.config,
                ddm_time_step=fine_ddm_time_step,
                ddm_spatial_points=fine_ddm_spatial_points,
                lca_max_step=fine_lca_max_step,
            )
        )
        start = time.perf_counter()
        fine_result = fit_lbfgsb(
            fine_likelihood,
            trials,
            starts=1,
            max_iterations=fine_max_iterations,
            seed=seed + 2,
            initial_vectors=[default_result.parameter_vector],
            include_default_start=False,
            random_start_candidates=0,
            polish_restarts=1,
            polish_iterations=min(30, fine_max_iterations),
            coordinate_step=1.0e-4,
            coordinate_levels=2,
            coordinate_rounds=1,
        )
        fine_seconds = time.perf_counter() - start

    final_result = fine_result if fine_result is not None else default_result
    meshes = {
        "coarse": {
            "ddm_time_step": coarse_ddm_time_step,
            "ddm_spatial_points": coarse_ddm_spatial_points,
            "lca_max_step": coarse_lca_max_step,
        },
        "default": {
            "ddm_time_step": likelihood.config.ddm_time_step,
            "ddm_spatial_points": likelihood.config.ddm_spatial_points,
            "lca_max_step": likelihood.config.lca_max_step,
        },
        "fine": {
            "ddm_time_step": fine_ddm_time_step,
            "ddm_spatial_points": fine_ddm_spatial_points,
            "lca_max_step": fine_lca_max_step,
        },
    }
    return StagedFitResult(
        parameter_vector=final_result.parameter_vector,
        log_likelihood=final_result.log_likelihood,
        final_mesh="fine" if fine_result is not None else "default",
        coarse_result=coarse_result,
        default_result=default_result,
        fine_result=fine_result,
        stage_seconds={
            "coarse": coarse_seconds,
            "default": default_seconds,
            "fine": fine_seconds,
            "total": coarse_seconds + default_seconds + fine_seconds,
        },
        meshes=meshes,
    )


def parameters_from_fit(
    fit: FitResult,
    *,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> ContinuousCSIParameters:
    vector = torch.as_tensor(fit.parameter_vector, dtype=dtype, device=device)
    return ContinuousCSIParameters.from_vector(vector)
