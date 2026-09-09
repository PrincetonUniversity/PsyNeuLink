"""Explicit empirical mass in the checked event-count observation domain.

No bandwidth, pseudocount, density conversion, or probability floor is implicit.
Endpoint compatibility is the existing numerical count guard, not an assertion
of exact floating-point readout support or a measurement-error distribution.
"""

from dataclasses import dataclass

import numpy as np

from psyneulink.core.batched.observed_sampling import validate_observation_sampling_witness
from psyneulink.core.batched.sampling import StochasticSamplingError


@dataclass(frozen=True)
class EmpiricalMassResult:
    successes: np.ndarray  # [candidate, trial]
    num_estimates: int
    probabilities: np.ndarray
    log_factors: np.ndarray
    log_likelihood: np.ndarray  # [candidate]
    zero_hits: np.ndarray
    backend: str
    target: str = "checked_event_count_and_exact_fp32_observation_fields"


def _validate_mass_plan(observation_plan):
    validate_observation_sampling_witness(observation_plan.sampler, observation_plan.witness)
    fields = tuple(readout.observation for readout in observation_plan.witness.readouts)
    if not any(field.score for field in fields):
        raise StochasticSamplingError("mass.no_scored_fields", "At least one observation field must be scored.")
    if any(field.score and field.measure != "counting" for field in fields):
        raise StochasticSamplingError("mass.measure", "Empirical mass requires counting measure; no continuous density is inferred.")
    endpoint = observation_plan.sampler.witness.boundary.history.endpoint.observation
    if any(field.score and field.role == "event_time" and field.port_id != endpoint.port_id for field in fields):
        raise StochasticSamplingError("mass.event_field", "A scored event_time must be the checked conditioning endpoint; additional times need separate consistency checks.")
    return fields


@dataclass(frozen=True)
class EmpiricalMassPlan:
    observation_plan: object

    def score(self, inputs, data, parameter_sets=None, *, num_estimates=1024, seed=0,
              common_random_numbers=True, horizon=None, max_buffer_bytes=256 * 1024**2, reference=False):
        """Estimate conditional event-count/observation masses with fresh lanes.

        A scored event_time matches its unique reconstructed integration count;
        other scored fields match exact FP32 values. All declared observed times
        still determine subsequent history, even if score=False. Truncated lanes
        raise rather than being discarded or renormalized. Zero hits return zero
        mass and -inf log mass. The log estimate is biased and can be unstable;
        this is not a claim of pseudo-marginal MCMC suitability.
        """
        import torch

        fields = _validate_mass_plan(self.observation_plan)
        if type(reference) is not bool:
            raise StochasticSamplingError("mass.reference", "reference must be a boolean.")
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != len(fields) or not np.all(np.isfinite(data)):
            raise StochasticSamplingError("mass.data", "Data must be a finite [trial, observation] array.")
        # Reserve matching/reduction workspace separately from sampler buffers.
        from psyneulink.core.batched.prep import normalize_parameter_sets

        simulation = self.observation_plan.sampler.path_plan.history_plan.simulation_plan
        rows = normalize_parameter_sets(parameter_sets, simulation.ir)
        if type(num_estimates) is not int or num_estimates < 1:
            raise StochasticSamplingError("sampling.estimates", "num_estimates must be a positive integer.")
        if type(max_buffer_bytes) is not int or max_buffer_bytes <= 0:
            raise StochasticSamplingError("sampling.memory_budget", "Buffer budget must be a positive integer.")
        workspace = len(rows) * len(data) * (num_estimates * 8 + 64)
        if workspace >= max_buffer_bytes:
            raise StochasticSamplingError("sampling.memory_budget", "Empirical matching buffers exceed the budget.")
        samples = self.observation_plan._run(
            inputs, data, rows, num_estimates, seed, common_random_numbers, horizon, True,
            max_buffer_bytes - workspace, reference=reference, return_device=True,
        )
        device = samples.values.device
        observed = torch.tensor(data, dtype=torch.float32, device=device)
        if not torch.isfinite(observed).all().item():
            raise StochasticSamplingError("mass.data", "Observed values exceed the finite FP32 domain.")
        matched = torch.ones(samples.values.shape[:-1], dtype=torch.bool, device=device)
        for field in fields:
            if not field.score:
                continue
            if field.role == "event_time":
                count = torch.tensor(np.array(samples.event_counts), device=device)
                target = torch.tensor(np.array(samples.history.event_counts), device=device)
                matched &= count == target[..., None]
            else:
                matched &= samples.values[..., field.column_start] == observed[None, :, None, field.column_start]
        successes = matched.sum(dim=-1).cpu().numpy()
        probabilities = successes.astype(np.float64) / num_estimates
        with np.errstate(divide="ignore"):
            log_factors = np.log(probabilities)
        totals = log_factors.sum(axis=-1)
        zero_hits = successes == 0
        for array in (successes, probabilities, log_factors, totals, zero_hits):
            array.flags.writeable = False
        return EmpiricalMassResult(successes, num_estimates, probabilities, log_factors, totals, zero_hits, simulation.backend)


def compile_empirical_mass(observation_plan):
    _validate_mass_plan(observation_plan)
    return EmpiricalMassPlan(observation_plan)
