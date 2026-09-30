"""Select a particle budget using independent complete-likelihood replicates.

Pilot runs choose one budget for the entire candidate batch. Fresh replicates at
that budget supply the returned mean log scores and Monte Carlo standard errors.
No trial factors, particles, or scores are pooled across particle counts.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

__all__ = ["AdaptiveLikelihoodResult"]


@dataclass(frozen=True)
class AdaptiveLikelihoodResult:
    """Result of ``PEC.log_likelihood_batch(..., adaptive=True)``.

    ``log_likelihood`` is the mean of fresh, complete log scores at
    ``num_estimates`` particles *per replicate*. ``standard_error`` estimates
    the Monte Carlo SE of that mean, not the SD of an individual filter run.
    This is a finite-particle mean-log criterion, not an unbiased log likelihood.

    With ``reference_index``, ``precision_standard_error`` measures paired score
    differences against that row; otherwise it equals ``standard_error``.
    ``target_met`` applies to this precision criterion, separately for each row.
    ``converged`` means all final SE estimates meet ``target_se``. It does not
    establish negligible finite-particle, histogram, smoothing, or model bias.

    The final replicates are independent of pilot budget selection. Their
    empirical SEs need not confirm the pilot result. We return that failure
    explicitly instead of selecting favorable final replicates. ``history``
    contains pilot diagnostics; only ``replicate_log_likelihoods`` contribute to
    the returned score. Seeds allow replay through the fixed-budget interface.
    ``sampling_work`` includes both pilot and final work, in candidate runs and
    particles per complete input history (not time steps).
    """

    log_likelihood: np.ndarray
    standard_error: np.ndarray
    num_estimates: int
    target_se: float
    reference_index: int | None
    log_likelihood_difference: np.ndarray | None
    difference_standard_error: np.ndarray | None
    precision_standard_error: np.ndarray
    target_met: np.ndarray
    stop_reason: str
    replicate_log_likelihoods: np.ndarray
    seeds: tuple[int, ...]
    history: tuple[dict, ...]
    sampling_work: dict

    @property
    def converged(self):
        return bool(np.all(self.target_met))


@dataclass(frozen=True)
class _PrecisionConfig:
    min_estimates: int
    target_se: float = 0.2
    repeats: int = 8
    growth_factor: int = 2
    reference_index: int | None = None
    reserved_seeds: tuple[int, ...] = ()

    def validate(self, max_estimates, candidates):
        for name, value in (
            ("min_estimates", self.min_estimates),
            ("max_estimates", max_estimates),
            ("repeats", self.repeats),
            ("growth_factor", self.growth_factor),
        ):
            if not _integer(value) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.min_estimates > max_estimates:
            raise ValueError("min_estimates must not exceed num_estimates (the cap).")
        if self.repeats < 2 or self.growth_factor < 2:
            raise ValueError("repeats and growth_factor must each be at least 2.")
        if (
            isinstance(self.target_se, (bool, np.bool_))
            or not isinstance(self.target_se, (int, float, np.integer, np.floating))
            or not np.isfinite(self.target_se)
            or self.target_se <= 0
        ):
            raise ValueError("target_se must be finite and positive.")
        if self.reference_index is not None and (
            not _integer(self.reference_index)
            or not 0 <= self.reference_index < candidates
            or candidates < 2
        ):
            raise ValueError(
                "reference_index requires at least two rows and a valid row index."
            )


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _summarize(scores, reference_index):
    mean = scores.mean(axis=0)
    se = scores.std(axis=0, ddof=1) / np.sqrt(len(scores))
    if reference_index is None:
        return mean, se, None, None, se
    # Pair entire histories by replicate seed. Pooling per-trial densities would
    # destroy the conditioning on each filter's own observation history.
    differences = scores - scores[:, reference_index, None]
    difference_se = differences.std(axis=0, ddof=1) / np.sqrt(len(scores))
    return mean, se, differences.mean(axis=0), difference_se, difference_se


def adaptive_log_likelihood(
    evaluate, parameter_values, *, max_estimates, seed, options=None
):
    """Run fixed-size pilot replicates, then an independent final evaluation.

    ``evaluate(rows, count, seed)`` must start complete likelihood runs from the
    initial model state and return one finite total log score per row. Errors
    (including execution truncation) propagate; penalties cannot be interpreted
    as precision evidence. The caller preserves the observation law across N.
    """
    if options is not None and not isinstance(options, Mapping):
        raise ValueError("adaptive_options must be a mapping.")
    options = {} if options is None else dict(options)
    options.setdefault("min_estimates", min(10000, max_estimates))
    config = _PrecisionConfig(**options)
    rows = np.asarray(parameter_values, dtype=float)
    if rows.ndim != 2 or not all(rows.shape) or not np.isfinite(rows).all():
        raise ValueError(
            "parameter_values must be a nonempty, finite matrix of parameter rows."
        )
    config.validate(max_estimates, len(rows))
    if not _integer(seed) or seed < 0:
        raise ValueError(
            "Adaptive likelihood evaluation requires a nonnegative integer seed."
        )
    try:
        reserved = tuple(config.reserved_seeds)
    except TypeError as error:
        raise ValueError(
            "reserved_seeds must contain nonnegative integer seeds."
        ) from error
    if any(not _integer(value) or value < 0 for value in reserved):
        raise ValueError("reserved_seeds must contain nonnegative integer seeds.")

    # Separate streams keep final draws independent of how many pilot stages ran.
    pilot_stream, final_stream = np.random.SeedSequence([int(seed), 479831]).spawn(2)
    pilot_rng, final_rng = map(np.random.default_rng, (pilot_stream, final_stream))
    used_seeds = set(reserved) | {int(seed)}
    work = {"batch_calls": 0, "candidate_runs": 0, "candidate_particles": 0}

    def replicate(count, rng):
        scores, seeds = [], []
        for _ in range(config.repeats):
            while True:
                run_seed = int(rng.integers(0, 2**31 - 1))
                if run_seed not in used_seeds:
                    break
            used_seeds.add(run_seed)
            seeds.append(run_seed)
            values = np.asarray(evaluate(rows, count, run_seed), dtype=float)
            if values.shape != (len(rows),) or not np.isfinite(values).all():
                raise FloatingPointError(
                    "Expected one finite complete log score per parameter row."
                )
            scores.append(values.copy())
            work["batch_calls"] += 1
            work["candidate_runs"] += len(rows)
            work["candidate_particles"] += len(rows) * count
        return np.asarray(scores), tuple(seeds)

    count = int(config.min_estimates)
    history = []
    # At the cap there is no budget decision left to make. Skip a redundant
    # pilot block and spend only the independent final replicates there.
    while count < max_estimates:
        scores, seeds = replicate(count, pilot_rng)
        mean, se, _, _, precision_se = _summarize(scores, config.reference_index)
        target_met = precision_se <= config.target_se
        history.append(
            {
                "num_estimates": count,
                "seeds": seeds,
                "replicate_log_likelihoods": scores,
                "log_likelihood": mean,
                "standard_error": se,
                "precision_standard_error": precision_se,
                "target_met": target_met,
            }
        )
        if target_met.all():
            break
        count = min(int(max_estimates), count * int(config.growth_factor))

    # Do not reuse pilot scores or keep searching on the final scores. Keeping
    # this final block independent avoids selecting unusually quiet replicates.
    scores, seeds = replicate(count, final_rng)
    mean, se, difference, difference_se, precision_se = _summarize(
        scores, config.reference_index
    )
    target_met = precision_se <= config.target_se
    reason = (
        "target_met"
        if target_met.all()
        else "max_estimates"
        if count == max_estimates
        else "precision_not_confirmed"
    )
    return AdaptiveLikelihoodResult(
        log_likelihood=mean,
        standard_error=se,
        num_estimates=count,
        target_se=float(config.target_se),
        reference_index=config.reference_index,
        log_likelihood_difference=difference,
        difference_standard_error=difference_se,
        precision_standard_error=precision_se,
        target_met=target_met,
        stop_reason=reason,
        replicate_log_likelihoods=scores,
        seeds=seeds,
        history=tuple(history),
        sampling_work=work,
    )
