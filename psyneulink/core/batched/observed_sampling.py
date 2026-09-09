"""Checked scalar observation gates for conditional stochastic sampling.

This layer uses the same affine/publication derivation as endpoint inversion.
It does not infer a measurement model from a floating-point output or a bin size.
"""

from dataclasses import dataclass

import numpy as np

from psyneulink.core.batched.endpoints import derive_scalar_readout
from psyneulink.core.batched.history import HistoryTrace
from psyneulink.core.batched.likelihood_ir import ObservationSamplingWitness
from psyneulink.core.batched.observation import ResolvedObservationField, resolve_observations
from psyneulink.core.batched.sampling import StochasticSamplingError, validate_sampler_witness


def derive_observation_sampling_witness(sampler):
    validate_sampler_witness(sampler.path_plan, sampler.witness)
    history = sampler.path_plan.history_plan
    simulation = history.simulation_plan
    fields = resolve_observations(history.observations, simulation.ir.graph, simulation.component_bindings)
    if any(field.recording != "exact" or field.availability != "complete" or field.width != 1 for field in fields):
        raise StochasticSamplingError("observation.recording", "Observation sampling currently requires complete exact scalar fields.")
    ports = tuple(field.port_id for field in sampler.witness.outputs)
    readouts = tuple(derive_scalar_readout(simulation.kernel_ir, field, primitive_ports=ports) for field in fields)
    if any(readout.clock_component_id != sampler.witness.boundary.consumer_component_id for readout in readouts):
        raise StochasticSamplingError("observation.clock", "All observation readouts must use the checked stochastic region.")
    return ObservationSamplingWitness(sampler.witness, readouts)


def validate_observation_sampling_witness(sampler, witness):
    if type(witness) is not ObservationSamplingWitness or witness != derive_observation_sampling_witness(sampler):
        raise StochasticSamplingError("observation.witness_mismatch", "Observation witness does not match the frozen readouts and sampler.")


@dataclass(frozen=True)
class ObservationSamples:
    history: HistoryTrace
    values: np.ndarray  # [candidate, trial, estimate, declared observation column]
    event_counts: np.ndarray
    truncated: np.ndarray
    outputs: tuple[ResolvedObservationField, ...]
    mode: str


@dataclass(frozen=True)
class ObservationSamplerPlan:
    sampler: object
    witness: ObservationSamplingWitness

    def compile_empirical_mass(self):
        from psyneulink.core.batched.empirical_mass import compile_empirical_mass

        return compile_empirical_mass(self)

    def source(self, *, reference=False):
        from psyneulink.core.batched.backend.triton.sampling import CanonicalTrialReferenceEmitter, ObservationRegionEmitter

        validate_observation_sampling_witness(self.sampler, self.witness)
        kernel = self.sampler.path_plan.history_plan.simulation_plan.kernel_ir
        return (CanonicalTrialReferenceEmitter(kernel, self.sampler.witness, self.witness)
                if reference else ObservationRegionEmitter(kernel, self.witness)).emit()

    def sample(self, inputs, data, parameter_sets=None, *, num_estimates=1, seed=0,
               common_random_numbers=True, horizon=None, strict_truncation=True, max_buffer_bytes=256 * 1024**2):
        return self._run(inputs, data, parameter_sets, num_estimates, seed, common_random_numbers,
                         horizon, strict_truncation, max_buffer_bytes, reference=False)

    def simulate_reference(self, inputs, data, parameter_sets=None, *, num_estimates=1, seed=0,
                           common_random_numbers=True, horizon=None, strict_truncation=True, max_buffer_bytes=256 * 1024**2):
        return self._run(inputs, data, parameter_sets, num_estimates, seed, common_random_numbers,
                         horizon, strict_truncation, max_buffer_bytes, reference=True)

    def _run(self, inputs, data, rows, estimates, seed, common_random, horizon, strict, budget, *, reference, return_device=False):
        from psyneulink.core.batched.backend.triton.sampling import run_stochastic_samples

        validate_observation_sampling_witness(self.sampler, self.witness)
        return run_stochastic_samples(self.sampler, inputs, data, rows, estimates, seed, common_random,
                                      horizon, strict, budget, reference=reference,
                                      observation_witness=self.witness, return_device=return_device)


def compile_observation_sampler(sampler):
    return ObservationSamplerPlan(sampler, derive_observation_sampling_witness(sampler))
