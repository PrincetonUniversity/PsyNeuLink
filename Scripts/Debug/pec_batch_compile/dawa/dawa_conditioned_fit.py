"""Staged CMA-ES using complete observation-conditioned particle filters.

A smaller fixed particle count guides exploration. Checkpoints and refinement
restart the full filter at the reference count; no densities, ancestors, or
particles are pooled across counts. Fresh-seed final selection averages complete
masked log scores, a finite-particle criterion, not an unbiased likelihood claim.
"""

from dataclasses import dataclass
import time

import numpy as np
import optuna
from optuna.distributions import FloatDistribution

from dawa_adaptive_fit import CovarianceCmaEsSampler, PENALTY, learned_covariance


@dataclass
class StagedConfig:
    search_estimates: int = 10000
    reference_estimates: int = 100000
    refine_evaluations: int = 600
    check_every: int = 250
    min_evaluations: int = 1000
    patience: int = 2
    progress_tolerance: float = .25
    checkpoint_candidates: int = 4
    selection_candidates: int = 8
    selection_repeats: int = 3

    def validate(self, evaluations, population):
        if not 1 <= self.search_estimates <= self.reference_estimates:
            raise ValueError("Staged counts require 1 <= search particles <= reference particles")
        if population < 2 or min(self.check_every, self.min_evaluations, self.patience,
                                 self.checkpoint_candidates, self.selection_candidates, self.selection_repeats) < 1:
            raise ValueError("Staged population must be at least two; check and selection counts must be positive")
        if min(self.refine_evaluations, evaluations - self.refine_evaluations) < population + 1:
            raise ValueError("Each staged phase needs an initial proposal and at least one complete population")
        if not np.isfinite(self.progress_tolerance) or self.progress_tolerance < 0:
            raise ValueError("Staged progress tolerance must be finite and nonnegative")


def conditioned_scores(pec, candidates, estimates, seed, *, reference_estimates, pseudocount,
                       invalid, work):
    """Use the production PEC closure at a new count, preserving its observation law.

    All observations, including masked history, pass through the normal closure.
    A failed batch is retried individually to identify truncating candidates.
    Other errors abort, and the caller's PEC settings are restored even on failure.
    """
    from psyneulink.core.batched.backend.triton.runtime import BatchedTruncationError

    function = pec.controller.function
    if not function.conditioned_likelihood:
        raise ValueError("Staged conditioned scoring requires a conditioned PEC")
    previous = (pec.controller.num_estimates, function.batched_seed, function.batched_pseudocount)
    try:
        pec.controller.num_estimates = estimates
        function.batched_seed = seed
        function.batched_pseudocount = pseudocount * estimates / reference_estimates
        objective = function._make_objective_func()._batched_parameter_sets

        def run(rows):
            work["filter_batch_calls"] = work.get("filter_batch_calls", 0) + 1
            work["candidate_filter_runs"] = work.get("candidate_filter_runs", 0) + len(rows)
            work["candidate_particles"] = work.get("candidate_particles", 0) + len(rows) * estimates
            return objective(rows)

        try:
            scores = run(candidates)
        except BatchedTruncationError:
            scores = []
            for candidate in candidates:
                try:
                    scores.append(float(run([candidate])[0]))
                except BatchedTruncationError as error:
                    scores.append(PENALTY)
                    invalid.append({"parameters": list(candidate), "estimates": estimates,
                                    "seed": seed, "reason": str(error)})
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
            raise FloatingPointError("Expected one finite complete-filter score per candidate")
        return scores
    finally:
        pec.controller.num_estimates, function.batched_seed, function.batched_pseudocount = previous


def fit_staged(study, bounds, initial, sample, config, *, evaluations, population,
               simulation_seed, optimizer_seed, reserved_seeds, log_batch):
    """Explore cheaply, check and refine at full count, then select with fresh seeds.

    sample(candidates, particle_count, seed) must start a complete filter from
    the model's initial state and return ONE masked log score per candidate.
    Checkpoint plateaus trigger refinement, never a claim of convergence.
    Reserved validation/generation seeds are excluded from selection draws.
    """
    config.validate(evaluations, population)
    names = list(bounds)
    distributions = {name: FloatDistribution(lo, hi, step=step) for name, (lo, hi, step) in bounds.items()}
    cache = {}
    reference_evaluations = 0

    def evaluate(candidates, estimates, seed):
        scores = np.asarray(sample(candidates, estimates, seed), dtype=float)
        if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
            raise FloatingPointError("Staged sampling requires one finite complete-filter score per candidate")
        return scores

    def reference(candidates):
        nonlocal reference_evaluations
        unique = list(dict.fromkeys(tuple(row) for row in candidates if tuple(row) not in cache))
        for offset in range(0, len(unique), population):
            group = unique[offset:offset + population]
            scores = evaluate(group, config.reference_estimates, simulation_seed)
            cache.update(zip(group, map(float, scores), strict=True))
            reference_evaluations += len(group)
        return np.array([cache[tuple(row)] for row in candidates])

    incumbent = tuple(initial[name] for name in names)
    incumbent_score = initial_score = float(reference([incumbent])[0])
    recent = {}
    checkpoints = []
    completed, stale, next_check = 0, 0, config.check_every
    search_limit = evaluations - config.refine_evaluations
    stop_reason = "Search proposal cap reached; convergence not asserted"
    while completed < search_limit:
        count = 1 if completed == 0 else min(population, search_limit - completed)
        trials = [study.ask(distributions) for _ in range(count)]
        candidates = np.array([[trial.params[name] for name in names] for trial in trials])
        started = time.perf_counter()
        scores = evaluate(candidates, config.search_estimates, simulation_seed)
        elapsed = time.perf_counter() - started
        for trial, candidate, score in zip(trials, candidates, scores, strict=True):
            study.tell(trial, float(score))
            if score > PENALTY:
                recent[tuple(candidate)] = float(score)
        completed += count
        log_batch(candidates, scores, elapsed, {"phase": "staged_search", "estimates": config.search_estimates,
                                               "seed": simulation_seed})
        if completed >= next_check or completed == search_limit:
            # Every recent score has the same count and seed. Raw low-count
            # scores never compete with a reference-count incumbent score.
            nominees = sorted(recent, key=lambda row: -recent[row])[:config.checkpoint_candidates]
            shortlist = list(dict.fromkeys([incumbent, *nominees]))
            checked = reference(shortlist)
            best = int(np.argmax(checked))
            improvement = float(checked[best] - incumbent_score)
            incumbent, incumbent_score = shortlist[best], float(checked[best])
            stale = stale + 1 if improvement < config.progress_tolerance else 0
            checkpoints.append({"evaluations": completed, "reference_score": incumbent_score,
                                "improvement": improvement, "parameters": list(incumbent),
                                "candidates": [list(row) for row in shortlist], "scores": checked.tolist(),
                                "stale_checks": stale})
            print({"staged_checkpoint": checkpoints[-1]}, flush=True)
            recent.clear()
            next_check = completed + config.check_every
            if completed >= config.min_evaluations and stale >= config.patience:
                stop_reason = "Reference-check plateau triggered refinement; convergence not asserted"
                break

    covariance = learned_covariance(study, names)
    refinement = optuna.create_study(direction="maximize", sampler=CovarianceCmaEsSampler(
        covariance=covariance, parameter_order=sorted(names), x0=dict(zip(names, incumbent, strict=True)),
        sigma0=.03, lr_adapt=True, popsize=population, seed=optimizer_seed + 1))
    refinement.enqueue_trial(dict(zip(names, incumbent, strict=True)))
    refined = 0
    while refined < config.refine_evaluations:
        count = 1 if refined == 0 else min(population, config.refine_evaluations - refined)
        trials = [refinement.ask(distributions) for _ in range(count)]
        candidates = np.array([[trial.params[name] for name in names] for trial in trials])
        started = time.perf_counter()
        scores = reference(candidates)
        elapsed = time.perf_counter() - started
        for trial, score in zip(trials, scores, strict=True):
            refinement.tell(trial, float(score))
        best = int(np.argmax(scores))
        if scores[best] > incumbent_score:
            incumbent, incumbent_score = tuple(candidates[best]), float(scores[best])
        log_batch(candidates, scores, elapsed, {"phase": "refinement", "estimates": config.reference_estimates,
                                               "seed": simulation_seed})
        refined += count

    finalists = sorted((row for row, score in cache.items() if score > PENALTY),
                       key=lambda row: -cache[row])[:config.selection_candidates]
    if not finalists:
        raise RuntimeError("Staged fitting did not find a valid candidate")
    rng = np.random.default_rng(np.random.SeedSequence([simulation_seed, 69471]))
    used = set(reserved_seeds) | {simulation_seed}
    selection_seeds, selection_scores = [], []
    while len(selection_seeds) < config.selection_repeats:
        seed = int(rng.integers(0, 2**31 - 1))
        if seed in used:
            continue
        used.add(seed)
        selection_seeds.append(seed)
        selection_scores.append(np.concatenate([
            evaluate(finalists[first:first + population], config.reference_estimates, seed)
            for first in range(0, len(finalists), population)
        ]))
    replicate_scores = np.array(selection_scores)
    valid = (replicate_scores > PENALTY).all(axis=0)
    if not valid.any():
        raise RuntimeError("All final candidates truncated on fresh selection seeds")
    means = replicate_scores.mean(axis=0)
    means[~valid] = PENALTY
    winner = int(np.argmax(means))
    selected = finalists[winner]
    result = {"fitted_params": dict(zip(names, selected, strict=True)), "optimal_value": cache[selected]}
    diagnostic = {
        "policy_version": 1, "initial_reference_score": initial_score, "reference_seed": simulation_seed,
        "reference_estimates": config.reference_estimates, "search_estimates": config.search_estimates,
        "checkpoints": checkpoints, "search_evaluations": completed, "refinement_evaluations": refined,
        "reference_evaluations": reference_evaluations, "search_stop_reason": stop_reason,
        "refinement_stop_reason": "Precision-stage proposal budget completed; convergence not asserted",
        "refinement_covariance": {"reused": covariance is not None, "parameter_order": sorted(names),
                                  "matrix": None if covariance is None else covariance.tolist()},
        "best_reference_score": incumbent_score,
        "final_selection": {"seeds": selection_seeds, "estimates_per_seed": config.reference_estimates,
                            "candidates": [list(row) for row in finalists], "valid": valid.tolist(),
                            "reference_scores": [cache[row] for row in finalists],
                            "replicate_log_scores": replicate_scores.tolist(),
                            "mean_log_scores": means.tolist(), "winner": winner,
                            "selected_reference_score": cache[selected]},
        "note": "Fresh-seed selection averages complete masked log scores at the reference particle count. "
                "This is a finite-particle criterion, not pooling trial factors or an unbiased likelihood claim. "
                "Independent validation belongs to the caller and never guides selection.",
    }
    return result, diagnostic, refinement
