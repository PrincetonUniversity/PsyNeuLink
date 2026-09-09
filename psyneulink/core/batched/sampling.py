"""Generated conditional sampling of a checked stochastic primitive region.

Raw primitive outputs only: observation gates and likelihood scoring have not
been lowered. Simulation lanes never update the canonical observed history.
"""

from dataclasses import dataclass

import numpy as np

from psyneulink.core.batched.history import HistoryTrace, replay_program
from psyneulink.core.batched.likelihood_ir import PrimitiveSampleOutput, StochasticSamplerWitness
from psyneulink.core.batched.trajectories import validate_boundary_witness


class StochasticSamplingError(ValueError):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def stochastic_step(kernel, clock):
    return next(op for group in replay_program(kernel).consideration_sets
                for member in group.members if member.component_id == clock
                for op in member.body if op.kind == "StepMechanism")


def derive_sampler_witness(path_plan):
    validate_boundary_witness(path_plan.history_plan, path_plan.witness)
    kernel = path_plan.history_plan.simulation_plan.kernel_ir
    clock = path_plan.witness.consumer_component_id
    op = stochastic_step(kernel, clock)
    spec = kernel.op_specs.lookup_spec(op.attrs["spec_key"])
    if spec.step_emit is None or spec.readout_emit is None:
        raise StochasticSamplingError("sampling.primitive_hooks", "A registered step and readout implementation are required.")
    node = kernel.graph.node(op.target)
    ports = {port.name: port for port in kernel.graph.ports if port.port_id in node.output_port_ids}
    fields, column = [], 0
    for declaration in spec.outputs:
        port = ports[declaration.port]
        fields.append(PrimitiveSampleOutput(port.port_id, port.name, port.width, column))
        column += port.width
    model_outputs = op.outputs[:len(op.outputs) - len(op.attrs["trial_state_ids"]) - 1]
    if tuple(output.width for output in model_outputs) != tuple(field.width for field in fields):
        raise StochasticSamplingError("sampling.output_layout", "Primitive outputs do not match the frozen step layout.")
    if any(stream.component_id != clock for stream in kernel.rng_streams):
        raise StochasticSamplingError("sampling.random_region", "The first sampler supports one stochastic primitive region.")
    return StochasticSamplerWitness(path_plan.witness, spec.key, tuple(fields), tuple(op.attrs["rng_stream_ids"]))


def validate_sampler_witness(path_plan, witness):
    if type(witness) is not StochasticSamplerWitness or witness != derive_sampler_witness(path_plan):
        raise StochasticSamplingError("sampling.witness_mismatch", "Sampler witness does not match the frozen boundary and primitive implementation.")


@dataclass(frozen=True)
class PrimitiveSamples:
    history: HistoryTrace
    values: np.ndarray  # [candidate, trial, estimate, primitive-output column]
    event_counts: np.ndarray  # [candidate, trial, estimate]
    truncated: np.ndarray
    outputs: tuple[PrimitiveSampleOutput, ...]
    mode: str


@dataclass(frozen=True)
class StochasticSamplerPlan:
    path_plan: object
    witness: StochasticSamplerWitness

    def source(self, *, reference=False):
        from psyneulink.core.batched.backend.triton.sampling import CanonicalTrialReferenceEmitter, StochasticRegionEmitter

        validate_sampler_witness(self.path_plan, self.witness)
        emitter = CanonicalTrialReferenceEmitter if reference else StochasticRegionEmitter
        return emitter(self.path_plan.history_plan.simulation_plan.kernel_ir, self.witness).emit()

    def sample(self, inputs, data, parameter_sets=None, *, num_estimates=1, seed=0,
               common_random_numbers=True, horizon=None, strict_truncation=True, max_buffer_bytes=256 * 1024**2):
        """Sample raw primitive outputs conditioned on the observed sequence.

        This CPU reference is not a likelihood evaluator. Truncation is explicit
        and rejected by default; no observation-gate mapping or score is implied.
        """
        return self._run(inputs, data, parameter_sets, num_estimates, seed, common_random_numbers,
                         horizon, strict_truncation, max_buffer_bytes, False)

    def simulate_reference(self, inputs, data, parameter_sets=None, *, num_estimates=1, seed=0,
                           common_random_numbers=True, horizon=None, strict_truncation=True, max_buffer_bytes=256 * 1024**2):
        """Run full coupled trials from the same canonical starts for comparison.

        Unlike unconditional multi-trial simulation, sampled RTs do not become
        the history of later trials. Both mechanism state and held controls are
        restored independently for each trial/estimate lane.
        """
        return self._run(inputs, data, parameter_sets, num_estimates, seed, common_random_numbers,
                         horizon, strict_truncation, max_buffer_bytes, True)

    def _run(self, inputs, data, rows, estimates, seed, common_random, horizon, strict, budget, reference):
        from psyneulink.core.batched.backend.triton.sampling import run_stochastic_samples

        validate_sampler_witness(self.path_plan, self.witness)
        return run_stochastic_samples(self, inputs, data, rows, estimates, seed, common_random,
                                      horizon, strict, budget, reference=reference)


def compile_stochastic_sampler(path_plan):
    return StochasticSamplerPlan(path_plan, derive_sampler_witness(path_plan))
