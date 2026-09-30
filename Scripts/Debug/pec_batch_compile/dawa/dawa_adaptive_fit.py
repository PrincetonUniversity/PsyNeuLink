"""Adaptive Monte Carlo ranking and CMA-ES refinement for marginal likelihoods.

Sampling callbacks return per-trial densities with constant prior fraction.
Each independent block simulates whole histories. No trial is resampled or
treated as an independent Monte Carlo replicate in the uncertainty estimate.
Conditioned likelihoods use PEC's adaptive policy instead: their resampled state
histories cannot be combined by pooling these per-trial marginal densities.
"""

from dataclasses import dataclass
import time

import numpy as np
import optuna
from optuna.distributions import FloatDistribution

from psyneulink.core.components.functions.nonstateful.adaptivefit import (
    CovarianceCmaEsSampler,
    learned_covariance,
)


PENALTY = -1.0e10


def pooled_scores(blocks, sizes, include, *, return_profile=False):
    """Pool marginal density estimates before scoring; retain paired block uncertainty."""
    weights = np.asarray(sizes, dtype=float) / sum(sizes)
    densities = np.stack(blocks).astype(np.float64)
    if densities.ndim not in (3, 4):
        raise ValueError(
            "Density blocks must be [candidate, trial] or [candidate, profile value, trial]"
        )
    valid = np.isfinite(densities).all(axis=(0, *range(2, densities.ndim)))
    densities[:, ~valid] = 1.0
    pooled = np.einsum("b,bc...->c...", weights, densities)
    profile_indices = None
    if densities.ndim == 4:
        # One nuisance value per complete subject/candidate, chosen AFTER
        # pooling probabilities. Never maximize each block or each trial alone.
        profile_scores = np.log(np.maximum(pooled[..., include], 1.0e-10)).sum(-1)
        profile_indices = np.argmax(profile_scores, axis=1)
        densities = np.take_along_axis(
            densities, profile_indices[None, :, None, None], axis=2
        )[:, :, 0]
        pooled = pooled[np.arange(len(pooled)), profile_indices]
    scores = np.log(np.maximum(pooled[:, include], 1.0e-10)).sum(-1)
    scores[~valid] = PENALTY
    # Delta-method block influence, keeping covariance across trials and across
    # candidates sharing random draws. Unequal blocks have variance ~1/size.
    influence = (
        (densities[..., include] - pooled[None, :, include])
        / np.maximum(pooled[None, :, include], 1.0e-10)
    ).sum(-1)
    difference = influence[:, :, None] - influence[:, None, :]
    variance = np.einsum("b,bij->ij", weights, difference**2) / max(1, len(blocks) - 1)
    result = (scores, np.sqrt(variance), valid)
    return (*result, profile_indices) if return_profile else result


def ranking_uncertainty(scores, pair_se, valid, tolerance):
    """Heuristic weighted regret for inversions anywhere in the population.

    Signed logarithmic rank weights approximate the importance of changes to
    CMA's mean/covariance updates. Leading-candidate swaps matter even when
    top-half membership is certain. Neither this proxy nor the 2.5-SE margin
    is a calibrated confidence bound on the optimizer's update.
    """
    order = np.flatnonzero(valid)[np.argsort(-scores[valid])]
    if len(order) < 2:
        return 0.0, False
    weights = np.log((len(scores) + 1) / 2) - np.log(np.arange(1, len(order) + 1))
    weights /= weights[0]
    i, j = np.triu_indices(len(order), 1)
    gap = scores[order[i]] - scores[order[j]]
    regret = np.maximum(0.0, 2.5 * pair_se[order[i], order[j]] - gap)
    uncertainty = float((np.abs(weights[i] - weights[j]) * regret).max())
    return uncertainty, uncertainty > tolerance


@dataclass
class AdaptiveConfig:
    min_estimates: int = 5000
    max_estimates: int = 100000
    rank_tolerance: float = 1.0
    check_every: int = 250
    min_evaluations: int = 1000
    patience: int = 2
    progress_tolerance: float = 0.25
    refine_evaluations: int = 600
    initial_blocks: int = 4
    max_search_evaluations: int = 2000
    selection_blocks: int = 3
    selection_candidates: int = 8

    def validate(self, evaluations, population):
        if not 2 <= self.initial_blocks <= self.min_estimates <= self.max_estimates:
            raise ValueError(
                "Adaptive estimates require 2 <= initial blocks <= minimum <= maximum"
            )
        if min(self.check_every, self.min_evaluations, self.patience) < 1:
            raise ValueError(
                "Adaptive check interval, minimum proposals, and patience must be positive"
            )
        if not 0 <= self.refine_evaluations < evaluations:
            raise ValueError(
                "Adaptive refinement count must be nonnegative and smaller than the proposal budget"
            )
        if evaluations - self.refine_evaluations < population + 1:
            raise ValueError(
                "Adaptive search needs an initial proposal and at least one complete population"
            )
        if (
            self.max_search_evaluations < population + 1
            or min(self.selection_blocks, self.selection_candidates) < 1
        ):
            raise ValueError(
                "Adaptive search cap must allow a population; selection counts must be positive"
            )
        if (
            not np.isfinite([self.rank_tolerance, self.progress_tolerance]).all()
            or min(self.rank_tolerance, self.progress_tolerance) < 0
        ):
            raise ValueError("Adaptive tolerances must be finite and nonnegative")


class PopulationRacer:
    """Add independent simulation blocks until ranking is stable or the budget is spent."""

    def __init__(
        self, sample, include, config, seed, reserved_seeds=(), *, sample_blocks=None
    ):
        self.sample, self.include, self.config = (
            sample,
            np.asarray(include, dtype=bool),
            config,
        )
        self.sample_blocks = sample_blocks
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 39817]))
        self.used_seeds = set(reserved_seeds) | {seed}

    def next_seed(self):
        while True:
            seed = int(self.rng.integers(0, 2**31 - 1))
            if seed not in self.used_seeds:
                self.used_seeds.add(seed)
                return seed

    def evaluate(self, candidates):
        blocks, sizes, seeds = [], [], []
        minimum = self.config.min_estimates
        count = self.config.initial_blocks
        initial_sizes = [minimum // count + (i < minimum % count) for i in range(count)]

        def append_blocks(new_sizes):
            new_sizes = [size for size in new_sizes if size]
            new_seeds = [self.next_seed() for _ in new_sizes]
            results = (
                [
                    self.sample(candidates, size, seed)
                    for size, seed in zip(new_sizes, new_seeds, strict=True)
                ]
                if self.sample_blocks is None
                else self.sample_blocks(candidates, new_sizes, new_seeds)
            )
            if len(results) != len(new_sizes):
                raise ValueError(
                    "Block callback must return one density array per sampling block"
                )
            blocks.extend(results)
            sizes.extend(new_sizes)
            seeds.extend(new_seeds)

        append_blocks(initial_sizes)
        while True:
            scores, se, valid, profile_indices = pooled_scores(
                blocks, sizes, self.include, return_profile=True
            )
            uncertainty, promote = ranking_uncertainty(
                scores, se, valid, self.config.rank_tolerance
            )
            if not promote or sum(sizes) >= self.config.max_estimates:
                break
            addition = min(sum(sizes), self.config.max_estimates - sum(sizes))
            append_blocks((addition // 2, addition - addition // 2))
        detail = {
            "estimates": sum(sizes),
            "block_sizes": sizes,
            "block_seeds": seeds,
            "ranking_uncertainty": uncertainty,
            "budget_cap_reached": bool(promote),
        }
        if profile_indices is not None:
            detail["profile_indices"] = profile_indices.tolist()
        return scores, detail


def fit_adaptive(
    study,
    bounds,
    initial,
    sample,
    include,
    config,
    *,
    evaluations,
    population,
    simulation_seed,
    optimizer_seed,
    reserved_seeds,
    log_batch,
    profile_parameter=None,
    sample_blocks=None,
):
    """Search adaptively, refine with learned covariance, then rescore finalists.

    Population scores at different budgets guide CMA-ES updates within
    generations and shortlist candidates for reference checks. They never
    directly choose the reported final fit. Fresh selection seeds compare a
    shortlist after optimization; separate validation seeds remain untouched.
    Independent final validation belongs to the caller and is not used here.
    Optional sample_blocks(candidates, sizes, seeds) batches the independent
    blocks within a race. It must preserve their order, densities, and seeds;
    reference checks and precision refinement still use the scalar callback.
    """
    config.validate(evaluations, population)
    names = list(bounds)
    distributions = {
        name: FloatDistribution(lo, hi, step=step)
        for name, (lo, hi, step) in bounds.items()
    }
    include = np.asarray(include, dtype=bool)
    racer = PopulationRacer(
        sample,
        include,
        config,
        simulation_seed,
        reserved_seeds,
        sample_blocks=sample_blocks,
    )
    final_seeds = [racer.next_seed() for _ in range(config.selection_blocks)]
    reference_cache = {}
    reference_grids = {}
    reference_profiles = {}
    selection_evaluations = 0

    def reference(candidates):
        nonlocal selection_evaluations
        unique = list(
            dict.fromkeys(
                tuple(row) for row in candidates if tuple(row) not in reference_cache
            )
        )
        if unique:
            density = sample(unique, config.max_estimates, simulation_seed).astype(
                np.float64
            )
            scores, _, valid, profiles = pooled_scores(
                [density], [config.max_estimates], include, return_profile=True
            )
            if (profiles is None) != (profile_parameter is None):
                raise ValueError(
                    "Profile metadata must match the sampling callback's density shape"
                )
            if profiles is not None:
                density[~valid] = 1.0
                grids = np.log(np.maximum(density[..., include], 1.0e-10)).sum(-1)
                grids[~valid] = PENALTY
                reference_grids.update(zip(unique, grids, strict=True))
                reference_profiles.update(zip(unique, map(int, profiles), strict=True))
            reference_cache.update(zip(unique, map(float, scores), strict=True))
            selection_evaluations += len(unique)
        return np.array([reference_cache[tuple(row)] for row in candidates])

    incumbent = np.array([initial[name] for name in names])
    incumbent_score = float(reference([incumbent])[0])
    initial_score = incumbent_score
    initial_profile = (
        None
        if profile_parameter is None
        else float(profile_parameter[1][reference_profiles[tuple(incumbent)]])
    )
    recent = []
    checkpoints = []
    completed, stale, next_check = 0, 0, config.check_every
    search_limit = min(
        evaluations - config.refine_evaluations, config.max_search_evaluations
    )
    screening_evaluations = 0
    stop_reason = "Search proposal cap reached"

    while completed < search_limit:
        count = 1 if completed == 0 else min(population, search_limit - completed)
        trials = [study.ask(distributions) for _ in range(count)]
        candidates = np.array(
            [[trial.params[name] for name in names] for trial in trials]
        )
        started = time.perf_counter()
        scores, diagnostic = racer.evaluate(candidates)
        elapsed = time.perf_counter() - started
        for trial, score in zip(trials, scores, strict=True):
            study.tell(trial, float(score))
        completed += count
        for index in np.argsort(-scores)[: min(2, count)]:
            if scores[index] > PENALTY:
                recent.append((float(scores[index]), candidates[index].copy()))
        log_batch(
            candidates, scores, elapsed, {"phase": "adaptive_search", **diagnostic}
        )

        if completed >= next_check or completed == search_limit:
            # Recompare nominees at a common budget and seed before reference
            # checks. Raw mixed-budget scores do not compete across generations.
            nominees = list(dict.fromkeys(tuple(candidate) for _, candidate in recent))
            shortlisted = []
            screening = []
            for offset in range(0, len(nominees), population):
                group = nominees[offset : offset + population]
                density = sample(group, config.min_estimates, simulation_seed)
                scores, _, _ = pooled_scores([density], [config.min_estimates], include)
                screening.extend(zip(map(float, scores), group, strict=True))
            screening_evaluations += len(nominees)
            for score, candidate in sorted(screening, key=lambda item: -item[0]):
                if score > PENALTY and not np.array_equal(candidate, incumbent):
                    shortlisted.append(candidate)
                if len(shortlisted) == 3:
                    break
            shortlist = [incumbent, *shortlisted]
            checked = reference(shortlist)
            best = int(np.argmax(checked))
            improvement = float(checked[best] - incumbent_score)
            incumbent, incumbent_score = np.array(shortlist[best]), float(checked[best])
            stale = stale + 1 if improvement < config.progress_tolerance else 0
            checkpoints.append(
                {
                    "evaluations": completed,
                    "reference_score": incumbent_score,
                    "improvement": improvement,
                    "parameters": incumbent.tolist(),
                    "selection_candidates": len(shortlist),
                    "stale_checks": stale,
                }
            )
            print({"adaptive_checkpoint": checkpoints[-1]}, flush=True)
            recent = []
            next_check = completed + config.check_every
            if completed >= config.min_evaluations and stale >= config.patience:
                stop_reason = "Coarse-search plateau triggered the precision stage; convergence not asserted"
                break

    covariance = learned_covariance(study, names)
    refinement = optuna.create_study(
        direction="maximize",
        sampler=CovarianceCmaEsSampler(
            covariance=covariance,
            parameter_order=sorted(names),
            x0=dict(zip(names, incumbent, strict=True)),
            sigma0=0.03,
            lr_adapt=True,
            popsize=population,
            seed=optimizer_seed + 1,
        ),
    )
    refinement.enqueue_trial(dict(zip(names, incumbent, strict=True)))
    refined = 0
    while refined < config.refine_evaluations:
        count = (
            1 if refined == 0 else min(population, config.refine_evaluations - refined)
        )
        trials = [refinement.ask(distributions) for _ in range(count)]
        candidates = np.array(
            [[trial.params[name] for name in names] for trial in trials]
        )
        started = time.perf_counter()
        scores = reference(candidates)
        elapsed = time.perf_counter() - started
        for trial, score in zip(trials, scores, strict=True):
            refinement.tell(trial, float(score))
        best = int(np.argmax(scores))
        if scores[best] > incumbent_score:
            incumbent, incumbent_score = candidates[best].copy(), float(scores[best])
        metadata = {
            "phase": "refinement",
            "estimates": config.max_estimates,
            "block_sizes": [config.max_estimates],
            "block_seeds": [simulation_seed],
        }
        if profile_parameter is not None:
            metadata["profile_indices"] = [
                reference_profiles[tuple(row)] for row in candidates
            ]
        log_batch(candidates, scores, elapsed, metadata)
        refined += count
    if incumbent_score <= PENALTY:
        raise RuntimeError("Adaptive fitting did not find a valid candidate")
    # Final selection uses NEW common seeds and pools probabilities before logs.
    # These are training/selection draws, never the independent validation draws.
    finalists = sorted(
        (row for row, score in reference_cache.items() if score > PENALTY),
        key=lambda row: -reference_cache[row],
    )[: config.selection_candidates]
    selection_densities = [
        sample(finalists, config.max_estimates, seed) for seed in final_seeds
    ]
    selection_scores, _, valid, selected_profiles = pooled_scores(
        selection_densities,
        [config.max_estimates] * len(final_seeds),
        include,
        return_profile=True,
    )
    if not valid.any():
        raise RuntimeError(
            "All final candidates truncated on independent selection seeds"
        )
    winner = int(np.argmax(selection_scores))
    selected = finalists[winner]
    selected_reference = reference_cache[selected]
    if profile_parameter is not None:
        selected_reference = float(reference_grids[selected][selected_profiles[winner]])
    result = {
        "fitted_params": dict(zip(names, selected, strict=True)),
        "optimal_value": selected_reference,
    }
    if profile_parameter is not None:
        result["fitted_params"][profile_parameter[0]] = float(
            profile_parameter[1][selected_profiles[winner]]
        )
    diagnostic = {
        "policy_version": 2,
        "initial_reference_score": initial_score,
        "reference_seed": simulation_seed,
        "reference_estimates": config.max_estimates,
        "checkpoints": checkpoints,
        "search_evaluations": completed,
        "refinement_evaluations": refined,
        "screening_evaluations": screening_evaluations,
        "reference_evaluations": selection_evaluations,
        "search_stop_reason": stop_reason,
        "refinement_stop_reason": "Precision-stage proposal budget completed; convergence not asserted",
        "refinement_covariance": {
            "reused": covariance is not None,
            "parameter_order": sorted(names),
            "matrix": None if covariance is None else covariance.tolist(),
        },
        "best_reference_score": incumbent_score,
        "final_selection": {
            "seeds": final_seeds,
            "estimates_per_seed": config.max_estimates,
            "total_estimates": len(final_seeds) * config.max_estimates,
            "candidates": [list(row) for row in finalists],
            "reference_scores": [reference_cache[row] for row in finalists],
            "scores": selection_scores.tolist(),
            "winner": winner,
            "selected_reference_score": selected_reference,
        },
        "note": "Ranking uncertainty is heuristic. The returned reference score belongs to the independently selected fit; it need not be the largest reference score.",
    }
    if profile_parameter is not None:
        diagnostic["profile"] = {
            "parameter": profile_parameter[0],
            "values": list(map(float, profile_parameter[1])),
            "optimizer_parameter_order": names,
            "initial_value": initial_profile,
            "selection_indices": selected_profiles.tolist(),
            "note": "Ranking SE is conditional on each candidate's pooled profile maximizer; it is not a calibrated bound after profiling.",
        }
    return result, diagnostic, refinement
