"""Observation kernels and bootstrap filtering, independent of simulation backend.

Continuous kernels describe a smoothed observation likelihood. They are held
fixed across candidates and particle budgets; filtering does not remove the
approximation introduced by the observation kernel.
"""

import numpy as np
from scipy.special import logsumexp


class ParticleSupportError(ValueError):
    """An observation has zero support under the predictive population."""


def _histogram_edges(low, high, bins):
    """Construct the FP32 edges used by the CUDA histogram observation model.

    CUDA linspace calculates each half from its nearest endpoint using a
    fused multiply-add. Evaluate the FP32 operands in FP64 before rounding
    once, so CPU-only scoring uses the same edges without requiring CUDA.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        start = np.float32(low)
        end = np.float32(high + (high - low) * 1e-6)
        step = np.float32((end - start) / np.float32(bins))
        indices = np.arange(bins + 1)
        edges = np.where(
            indices < (bins + 1) // 2,
            float(start) + float(step) * indices.astype(np.float32).astype(float),
            float(end)
            - float(step) * (bins - indices).astype(np.float32).astype(float),
        ).astype(np.float32)
    if not np.isfinite(edges).all() or np.any(edges[1:] <= edges[:-1]):
        raise ValueError("Histogram edges must be finite and distinct in FP32.")
    return edges


class ParticleObservationModel:
    """Normalized mixed categorical/continuous observation kernels.

    ``kernel='gaussian'`` uses a product Gaussian with fixed ``bandwidth`` per
    continuous dimension. The default bandwidth uses the observed standard
    deviation and Silverman's one-dimensional rule, independently by dimension;
    constant columns use unit scale. ``kernel='histogram'`` uses fixed bins and
    Gaussian smoothing in bin units, normalized around each simulated bin.
    Histogram observations, predictions and tables use FP32 to match the GPU
    observation model, independently of simulator precision. Interior edges
    belong to the lower bin; the upper domain bound is expanded by one part
    per million before constructing edges. Histogram categorical matching
    uses absolute tolerance 1e-6 and requires nonoverlapping category supports.
    Gaussian kernels and likelihood accumulation retain FP64 arithmetic.

    ``contamination_probability`` explicitly mixes in a uniform observation
    distribution over ``bin_range`` and ``categorical_values``. Its probability
    is independent of particle count. No contamination or density floor is
    introduced by default. Observations must be finite; histogram observations
    and contamination observations must be inside the declared domain.
    """

    def __init__(
        self,
        data,
        categorical_dims,
        *,
        kernel="gaussian",
        bandwidth=None,
        bins=100,
        bin_range=None,
        smoothing_sigma=0.5,
        contamination_probability=0.0,
        categorical_values=None,
    ):
        self.data = np.array(data, dtype=float, copy=True)
        if (
            self.data.ndim != 2
            or not self.data.size
            or not np.isfinite(self.data).all()
        ):
            raise ValueError(
                "Particle observations must be a nonempty finite [trial, outcome] array."
            )
        categorical_dims = np.asarray(categorical_dims, dtype=bool)
        if categorical_dims.shape != (self.data.shape[1],):
            raise ValueError("categorical_dims must contain one boolean per outcome.")
        self.categorical = np.flatnonzero(categorical_dims)
        self.continuous = np.flatnonzero(~categorical_dims)
        if kernel not in {"gaussian", "histogram"}:
            raise ValueError("kernel must be 'gaussian' or 'histogram'.")
        self.kernel = kernel
        with np.errstate(over="ignore"):
            self._observations = (
                self.data.astype(np.float32) if kernel == "histogram" else self.data
            )
        if not np.isfinite(self._observations).all():
            raise ValueError("Histogram observations must be finite in FP32.")
        if (
            not np.isfinite(contamination_probability)
            or not 0 <= contamination_probability < 1
        ):
            raise ValueError("contamination_probability must be finite and in [0, 1).")
        self.contamination = float(contamination_probability)
        if (
            isinstance(bins, bool)
            or not isinstance(bins, (int, np.integer))
            or bins < 1
        ):
            raise ValueError("bins must be a positive integer.")
        if not np.isfinite(smoothing_sigma) or smoothing_sigma < 0:
            raise ValueError("smoothing_sigma must be finite and nonnegative.")
        if bandwidth is not None and kernel != "gaussian":
            raise ValueError("bandwidth is only used by the Gaussian kernel.")
        values = self._observations[:, self.continuous]
        if bandwidth is None:
            scale = np.std(values, axis=0)
            scale = np.where(np.ptp(values, axis=0) > 0, scale, 1.0)
            bandwidth = 1.06 * scale * len(self.data) ** (-0.2)
        self.bandwidth = np.broadcast_to(
            np.asarray(bandwidth, dtype=float), (len(self.continuous),)
        ).copy()
        if not np.isfinite(self.bandwidth).all() or np.any(self.bandwidth <= 0):
            raise ValueError(
                "bandwidth must be finite and positive for every continuous outcome."
            )
        self.edges = []
        self.bin_lookups = []
        self.bin_volume = np.float32(1.0) if kernel == "histogram" else 1.0
        domain_volume = 1.0
        if bin_range is not None and len(bin_range) != len(self.continuous):
            raise ValueError(
                "bin_range must contain one (low, high) pair per continuous outcome."
            )
        for j, dimension in enumerate(self.continuous):
            if bin_range is None:
                low, high = float(np.min(values[:, j])), float(np.max(values[:, j]))
                margin = 0.02 * (high - low) if high > low else 1.0
                low, high = low - margin, high + margin
            else:
                low, high = bin_range[j]
            if not np.isfinite([low, high]).all() or not high > low:
                raise ValueError("Every bin_range must have finite low < high.")
            edge = (
                _histogram_edges(low, high, bins)
                if kernel == "histogram"
                else np.linspace(low, high, bins + 1)
            )
            if (kernel == "histogram" or self.contamination) and np.any(
                (values[:, j] < edge[0]) | (values[:, j] > edge[-1])
            ):
                raise ValueError(
                    "All assimilated observations, including masked rows, must lie inside bin_range."
                )
            self.edges.append(edge)
            self.bin_volume *= (
                edge[1] - edge[0] if kernel == "histogram" else (high - low) / bins
            )
            domain_volume *= high - low
            observed_bin = np.searchsorted(edge[1:-1], values[:, j], side="left")
            delta = np.arange(bins)[None, :] - observed_bin[:, None]
            dtype = self._observations.dtype
            if smoothing_sigma == 0:
                lookup = (delta == 0).astype(dtype)
            else:
                radius = max(1, int(np.ceil(3 * smoothing_sigma)))
                offsets = np.arange(-radius, radius + 1)
                weights = np.exp(-0.5 * (offsets.astype(dtype) / smoothing_sigma) ** 2)
                valid = (np.arange(bins)[:, None] + offsets >= 0) & (
                    np.arange(bins)[:, None] + offsets < bins
                )
                normalization = (valid * weights).sum(axis=1)
                lookup = np.where(
                    abs(delta) <= radius,
                    np.exp(-0.5 * (delta.astype(dtype) / smoothing_sigma) ** 2),
                    0.0,
                )
                lookup /= normalization
            self.bin_lookups.append(lookup)
        if kernel == "histogram":
            # The GPU's pseudocount mixture is uniform over joint histogram
            # cells, using the same rounded bin volume as the observation law.
            domain_volume = float(self.bin_volume) * bins ** len(self.continuous)
        if categorical_values is None:
            categorical_values = [
                np.unique(self.data[:, dim]) for dim in self.categorical
            ]
        if len(categorical_values) != len(self.categorical):
            raise ValueError(
                "categorical_values must declare each categorical outcome's possible values."
            )
        self.categorical_values = []
        for dimension, support in zip(self.categorical, categorical_values):
            support = np.asarray(support, dtype=float)
            if (
                support.ndim != 1
                or not len(support)
                or not np.isfinite(support).all()
                or len(np.unique(support)) != len(support)
            ):
                raise ValueError(
                    "Categorical support must contain distinct finite values."
                )
            if not np.isin(self.data[:, dimension], support).all():
                raise ValueError("Observed category is absent from categorical_values.")
            if kernel == "histogram":
                with np.errstate(over="ignore"):
                    rounded_support = support.astype(np.float32)
                if not np.isfinite(rounded_support).all() or np.any(
                    np.diff(np.sort(rounded_support).astype(float))
                    <= 2 * np.float32(1e-6)
                ):
                    raise ValueError(
                        "Histogram categorical values must be finite in FP32 and separated by more than 2e-6."
                    )
            self.categorical_values.append(support.copy())
            domain_volume *= len(support)
        self._log_uniform_density = -np.log(domain_volume)

    def evaluate(self, simulated, trial):
        """Return log predictive density, normalized weights, and contamination responsibility."""
        simulated = np.asarray(simulated)
        category_dtype = (
            simulated.dtype
            if np.issubdtype(simulated.dtype, np.floating)
            else np.dtype(float)
        )
        if self.kernel == "histogram":
            category_dtype = np.dtype(np.float32)
        with np.errstate(over="ignore"):
            simulated = np.asarray(
                simulated, dtype=np.float32 if self.kernel == "histogram" else float
            )
        if (
            simulated.ndim != 2
            or simulated.shape[1] != self.data.shape[1]
            or not len(simulated)
        ):
            raise ValueError("Simulated outcomes must have shape [particle, outcome].")
        if not np.isfinite(simulated).all():
            raise ValueError("Simulated outcomes must be finite.")
        # Compare labels at the simulator's precision, without overlapping
        # tolerance neighborhoods that could assign mass to two categories.
        for support in self.categorical_values:
            if len(np.unique(support.astype(category_dtype))) != len(support):
                raise ValueError(
                    "Categorical values are indistinguishable at the simulation precision."
                )
        observed_categories = self._observations[trial, self.categorical].astype(
            category_dtype
        )
        if self.kernel == "histogram":
            match = (
                abs(simulated[:, self.categorical] - observed_categories)
                <= np.float32(1e-6)
            ).all(axis=1)
        else:
            match = (simulated[:, self.categorical] == observed_categories).all(axis=1)
        log_weights = np.where(match, 0.0, -np.inf)
        if self.kernel == "gaussian":
            residual = (
                simulated[:, self.continuous] - self.data[trial, self.continuous]
            ) / self.bandwidth
            with np.errstate(over="ignore"):
                log_weights -= (
                    0.5 * residual**2 + np.log(self.bandwidth) + 0.5 * np.log(2 * np.pi)
                ).sum(axis=1)
        else:
            contributions = match.astype(np.float32)
            with np.errstate(divide="ignore"):
                for dimension, edge, lookup in zip(
                    self.continuous, self.edges, self.bin_lookups
                ):
                    values = simulated[:, dimension]
                    source_bin = np.searchsorted(edge[1:-1], values, side="left")
                    contributions *= lookup[trial, source_bin] * (
                        (values >= edge[0]) & (values <= edge[-1])
                    )
                log_weights = np.log(contributions.astype(float)) - np.log(
                    float(self.bin_volume)
                )
        if self.contamination:
            log_weights = np.logaddexp(
                np.log1p(-self.contamination) + log_weights,
                np.log(self.contamination) + self._log_uniform_density,
            )
        total = logsumexp(log_weights)
        if not np.isfinite(total):
            raise ParticleSupportError(
                f"Zero particle observation support at trial {trial}. Increase num_estimates or the kernel width, "
                "or explicitly specify an observation-contamination model."
            )
        log_density = float(total - np.log(len(simulated)))
        normalized = np.exp(log_weights - total)
        normalized /= normalized.sum()
        responsibility = (
            float(
                np.exp(
                    np.log(self.contamination) + self._log_uniform_density - log_density
                )
            )
            if self.contamination
            else 0.0
        )
        return log_density, normalized, responsibility


def systematic_resample(weights, rng):
    """Resample in O(N log N), using one random offset and a float64 CDF."""
    weights = np.asarray(weights, dtype=float)
    if (
        weights.ndim != 1
        or not len(weights)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError(
            "Resampling requires finite nonnegative weights with positive total."
        )
    cumulative = np.cumsum(weights, dtype=np.float64)
    cumulative /= cumulative[-1]
    cumulative[-1] = 1.0
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    return np.minimum(
        np.searchsorted(cumulative, positions, side="right"), len(weights) - 1
    )


def particle_filter(
    advance,
    resample,
    observation_model,
    *,
    include_mask=None,
    seed=0,
    return_sim_data=False,
):
    """Score one contiguous sequence; masked rows still update the posterior.

    ``advance(trial)`` returns predictive outcomes. ``resample(ancestors)``
    updates the simulator's retained state. Each invocation starts a fresh
    filter; managing the initial simulator state belongs to the caller.
    """
    trials = len(observation_model.data)
    mask = (
        np.ones(trials, dtype=bool)
        if include_mask is None
        else np.asarray(include_mask, dtype=bool)
    )
    if mask.shape != (trials,):
        raise ValueError("include_mask must contain one boolean per trial.")
    rng = np.random.default_rng(seed)
    log_densities, effective_sizes, contamination = [], [], []
    simulations = []
    for trial in range(trials):
        simulated = advance(trial)
        log_density, normalized, responsibility = observation_model.evaluate(
            simulated, trial
        )
        log_densities.append(log_density)
        effective_sizes.append(1.0 / np.sum(normalized**2))
        contamination.append(responsibility)
        if return_sim_data:
            simulations.append(np.array(simulated, copy=True))
        if trial + 1 < trials:
            resample(systematic_resample(normalized, rng))
    log_densities = np.asarray(log_densities)
    diagnostics = {
        "per_trial_log_densities": log_densities,
        "effective_sample_size": np.asarray(effective_sizes),
        "contamination_responsibility": np.asarray(contamination),
    }
    return (
        float(log_densities[mask].sum()),
        diagnostics,
        np.array(simulations) if return_sim_data else None,
    )
