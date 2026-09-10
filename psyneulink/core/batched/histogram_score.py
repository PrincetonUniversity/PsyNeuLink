"""Explicit histogram surrogate scoring for checked scalar observations.

This estimator is distinct from count-domain empirical mass. Bin ranges are
anchored to observations, not simulations. Smoothing, pseudocounts, and the
legacy ZERO_PROB floor are explicit estimator semantics, not a recording model
or an assertion that a discrete simulator has a continuous density.
"""

from dataclasses import dataclass

import numpy as np

from psyneulink.core.batched.likelihood import _as_categorical_mask, _bin_edges, _categorical_cardinalities, ZERO_PROB
from psyneulink.core.batched.observed_sampling import validate_observation_sampling_witness
from psyneulink.core.batched.sampling import StochasticSamplingError


@dataclass(frozen=True)
class HistogramScoreResult:
    bin_counts: np.ndarray  # [candidate, trial, offset from observed bin]
    num_estimates: int
    densities: np.ndarray
    log_factors: np.ndarray
    log_likelihood: np.ndarray
    backend: str
    target: str = "explicit_smoothed_histogram_density_surrogate"
    execution: str = "strict"
    sampled_trials: np.ndarray | None = None  # False: counts=-1, density/log factor=NaN.
    window_stopped: np.ndarray | None = None  # [candidate, trial], not truncation.
    history_timing: str = "exact"


@dataclass(frozen=True)
class HistogramScorePlan:
    observation_plan: object
    categorical_dims: tuple[int, ...]
    bins: int = 100
    bin_range: tuple | None = None
    smoothing_sigma: float = 0.
    pseudocount: float = 0.
    categorical_cardinalities: tuple | None = None

    @property
    def radius(self):
        return min(self.bins - 1, max(1, int(np.ceil(3 * self.smoothing_sigma)))) if self.smoothing_sigma else 0

    @property
    def continuous_dim(self):
        return next(i for i in range(len(self.observation_plan.witness.readouts)) if i not in self.categorical_dims)

    def _validate(self):
        validate_observation_sampling_witness(self.observation_plan.sampler, self.observation_plan.witness)
        fields = tuple(item.observation for item in self.observation_plan.witness.readouts)
        if not all(field.score for field in fields):
            raise StochasticSamplingError("histogram.fields", "This histogram tier requires every declared observation field to be scored.")
        mask = _as_categorical_mask(self.categorical_dims, len(fields))
        if np.count_nonzero(~mask) != 1:
            raise StochasticSamplingError("histogram.dimensions", "Fused histograms currently require one numeric dimension and optional categorical dimensions.")
        if type(self.bins) is not int or self.bins < 1:
            raise ValueError("bins must be a positive integer.")
        if not np.isfinite(self.smoothing_sigma) or self.smoothing_sigma < 0:
            raise ValueError("smoothing_sigma must be finite and nonnegative.")
        if not np.isfinite(self.pseudocount) or self.pseudocount < 0:
            raise ValueError("pseudocount must be finite and nonnegative.")
        if self.radius > 32:
            raise StochasticSamplingError("histogram.radius", "This fused tier supports a smoothing radius of at most 32 bins.")
        if self.bin_range is not None:
            bounds = np.asarray(self.bin_range)
            if bounds.shape != (1, 2) or not np.all(np.isfinite(bounds)) or bounds[0, 1] <= bounds[0, 0]:
                raise ValueError("bin_range must contain one finite increasing (lower, upper) pair.")

    def score(self, inputs, data, parameter_sets=None, *, num_estimates=1024, seed=0,
              common_random_numbers=True, horizon=None, max_buffer_bytes=256 * 1024**2,
              include_mask=None, candidate_batch_size=None, estimate_batch_size=None,
              triton_launch_options=None, reference=False, execution="strict"):
        """Score using integer GPU reductions, or materialized samples as oracle.

        strict: sample every trial to completion. score_only: sample included
        trials only. window: additionally stop after a checked count cutoff
        outside the contributing bins. All modes replay every observed history;
        no mode renormalizes surviving samples. True truncation still raises.
        Faster modes do not provide full-path diagnostics for omitted work.
        """
        self._validate()
        if type(reference) is not bool:
            raise ValueError("reference must be boolean.")
        if execution not in ("strict", "score_only", "window"):
            raise ValueError("execution must be 'strict', 'score_only', or 'window'.")
        if reference and execution != "strict":
            raise ValueError("The materialized oracle requires execution='strict'.")
        if reference and any(value is not None for value in (candidate_batch_size, estimate_batch_size, triton_launch_options)):
            raise ValueError("Chunk/launch controls apply only to fused scoring, not the materialized oracle.")
        from psyneulink.core.batched.backend.triton.scoring import run_reduced_observations

        if not reference:
            return run_reduced_observations(
                self.observation_plan, inputs, data, parameter_sets, num_estimates, seed,
                common_random_numbers, horizon, max_buffer_bytes, histogram=self,
                include_mask=include_mask, candidate_batch_size=candidate_batch_size,
                estimate_batch_size=estimate_batch_size, triton_launch_options=triton_launch_options,
                execution=execution,
            )
        # Deliberately retain the independent materializing sampler/torch
        # bucketize oracle. It has a different memory cost from fused scoring.
        samples = self.observation_plan._run(inputs, data, parameter_sets, num_estimates, seed,
                                             common_random_numbers, horizon, True, max_buffer_bytes,
                                             reference=False, return_device=True)
        prepared = prepare_histogram(self, data, samples.values.device)
        import torch

        observed, edges, observed_bin, valid, weights, joint_bins = prepared
        match = torch.ones(samples.values.shape[:-1], dtype=torch.bool, device=observed.device)
        for column in self.categorical_dims:
            match &= torch.isclose(samples.values[..., column], observed[None, :, None, column], atol=1e-6, rtol=0)
        values = samples.values[..., self.continuous_dim].contiguous()
        sample_bin = torch.bucketize(values, edges[1:-1])
        match &= (values >= edges[0]) & (values <= edges[-1]) & valid[None, :, None]
        counts = torch.stack([
            (match & (sample_bin == observed_bin[None, :, None] + offset)).sum(-1)
            for offset in range(-self.radius, self.radius + 1)
        ], dim=-1)
        return histogram_result(self, counts, weights, edges, joint_bins, num_estimates, include_mask,
                                self.observation_plan.sampler.path_plan.history_plan.simulation_plan.backend)

    def source(self, *, execution="strict"):
        """Inspect the checked fused sampling/reduction kernel."""
        self._validate()
        if execution not in ("strict", "score_only", "window"):
            raise ValueError("Unknown histogram execution mode.")
        from psyneulink.core.batched.backend.triton.scoring import ReducedObservationEmitter

        kernel = self.observation_plan.sampler.path_plan.history_plan.simulation_plan.kernel_ir
        return ReducedObservationEmitter(kernel, self.observation_plan.witness, self, execution=execution).emit()


def prepare_histogram(plan, data, device):
    import torch

    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or not len(data) or data.shape[1] != len(plan.observation_plan.witness.readouts) or not np.all(np.isfinite(data)):
        raise StochasticSamplingError("histogram.data", "Data must be a nonempty finite [trial, observation] array.")
    # Pandas columns often arrive in Fortran order; emitted kernels address
    # observations as packed [trial, field], not using tensor strides.
    observed = torch.tensor(data, dtype=torch.float32, device=device).contiguous()
    if not torch.isfinite(observed).all().item():
        raise StochasticSamplingError("histogram.data", "Observed values exceed finite FP32.")
    numeric = observed[:, plan.continuous_dim:plan.continuous_dim + 1]
    edges = _bin_edges(numeric, numeric, plan.bins, plan.bin_range, torch)[0]
    if not torch.isfinite(edges).all().item() or not (edges[1:] > edges[:-1]).all().item():
        raise StochasticSamplingError("histogram.edges", "Histogram edges must be finite and strictly increasing in FP32.")
    observed_bin = torch.bucketize(numeric[:, 0].contiguous(), edges[1:-1])
    valid = (numeric[:, 0] >= edges[0]) & (numeric[:, 0] <= edges[-1])
    offsets = torch.arange(-plan.radius, plan.radius + 1, device=device)
    kernel = torch.exp(-.5 * (offsets.float() / plan.smoothing_sigma) ** 2) if plan.smoothing_sigma else torch.ones(1, device=device)
    valid_offsets = (observed_bin[:, None] + offsets >= 0) & (observed_bin[:, None] + offsets < plan.bins)
    weights = kernel[None, :] * valid_offsets
    weights = weights / weights.sum(-1, keepdim=True)
    cat_mask = _as_categorical_mask(plan.categorical_dims, data.shape[1])
    cardinalities = _categorical_cardinalities(data, cat_mask, plan.categorical_cardinalities) if plan.pseudocount else ()
    return observed, edges, observed_bin, valid, weights, float(plan.bins * np.prod(cardinalities))


def histogram_result(plan, counts, weights, edges, joint_bins, estimates, include_mask, backend,
                     *, execution="strict", sampled_trials=None, window_stopped=None):
    import torch

    weighted = (counts.to(torch.float32) * weights[None]).sum(-1)
    densities = ((weighted + plan.pseudocount) /
                 ((float(estimates) + plan.pseudocount * joint_bins) * (edges[1] - edges[0])))
    densities = torch.clamp(densities, min=ZERO_PROB).cpu().numpy()
    counts = counts.cpu().numpy().astype(np.int64)
    logs = np.log(densities)
    include = np.ones(logs.shape[1], dtype=bool) if include_mask is None else np.asarray(include_mask, dtype=bool).reshape(-1)
    if len(include) != logs.shape[1]:
        raise ValueError("include_mask must have one entry per trial.")
    sampled_trials = np.ones(logs.shape[1], dtype=bool) if sampled_trials is None else np.array(sampled_trials, dtype=bool, copy=True)
    window_stopped = np.zeros(logs.shape, dtype=np.int64) if window_stopped is None else np.array(window_stopped, dtype=np.int64, copy=True)
    counts[:, ~sampled_trials] = -1
    densities[:, ~sampled_trials] = np.nan
    logs[:, ~sampled_trials] = np.nan
    total = logs[:, include].sum(-1)
    for array in (counts, densities, logs, total, sampled_trials, window_stopped):
        array.flags.writeable = False
    return HistogramScoreResult(counts, estimates, densities, logs, total, backend,
                                execution=execution, sampled_trials=sampled_trials, window_stopped=window_stopped,
                                history_timing=plan.observation_plan.witness.sampler.boundary.history.endpoint.observation.history_timing)


def compile_histogram_score(observation_plan, *, categorical_dims, bins=100, bin_range=None,
                            smoothing_sigma=0., pseudocount=0., categorical_cardinalities=None):
    width = len(observation_plan.witness.readouts)
    categorical = tuple(np.flatnonzero(_as_categorical_mask(categorical_dims, width)).tolist())
    plan = HistogramScorePlan(observation_plan, categorical, bins,
                              None if bin_range is None else tuple(tuple(pair) for pair in bin_range),
                              smoothing_sigma, pseudocount,
                              None if categorical_cardinalities is None else tuple(categorical_cardinalities))
    plan._validate()
    return plan
