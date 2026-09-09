"""Checked single-event history replay using the source scheduler.

This is an inspection/reference interface, not a likelihood objective or a
parallel trial transformation. Primitive semantics remain trusted contracts.
"""

from dataclasses import dataclass, field

import numpy as np

from psyneulink.core.batched.dependency import analyze_axis_dependencies
from psyneulink.core.batched.kernel_ir import iter_kernel_ops
from psyneulink.core.batched.likelihood_analysis import _random_dependency_path
from psyneulink.core.batched.likelihood_ir import HistoryReplayWitness
from psyneulink.core.batched.observation import ObservationSpec


class HistoryReplayError(ValueError):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def _reject(code, detail):
    raise HistoryReplayError(code, detail)


def replay_program(kernel):
    programs = [op.attrs["program"] for op in iter_kernel_ops(kernel)
                if op.kind == "ForPasses" and "program" in op.attrs]
    if kernel.fusion_kind != "coevolving_graph" or len(programs) != 1:
        _reject("history.schedule_unsupported", "History replay currently requires one coupled dynamic scheduler program.")
    return programs[0]


def derive_history_witness(simulation_plan, observations):
    report = simulation_plan.diagnose_likelihood(observations)
    if report.factorization_status == "blocked":
        _reject("history.dependencies_unresolved", "; ".join(d.detail for d in report.diagnostics))
    kernel = simulation_plan.kernel_ir
    program = replay_program(kernel)
    if len(report.endpoint_witnesses) != 1:
        _reject("history.single_event_required", "History replay requires exactly one checked conditioning event.")
    endpoint = report.endpoint_witnesses[0]
    clock_id = endpoint.clock_component_id
    if set(report.stochastic_component_ids) != {clock_id}:
        _reject("history.random_region_unsupported", "All declared randomness must belong to the observed event primitive.")
    spec = kernel.op_specs.lookup_spec(endpoint.clock_spec_key)
    if spec.likelihood_contract.event_readout.execution_rule != "one_step_until_finished":
        _reject("history.execution_rule_missing", "The event readout lacks a one-step, absorbing-finished execution contract.")
    if any(state.component_id == clock_id for state in kernel.states):
        _reject("history.random_state_retained", "The event primitive cannot retain unobserved state across trials.")
    members = [member for group in program.consideration_sets for member in group.members]
    clock_steps = [op for member in members if member.component_id == clock_id
                   for op in member.body if op.kind == "StepMechanism"]
    if len(clock_steps) != 1 or clock_steps[0].attrs.get("finished_trial_state_id") is None:
        _reject("history.event_step_unsupported", "The event must have one explicit scheduled step and finished publication.")
    # The prior effect analysis conservatively rejects any stochastic influence
    # on retained state/controls except trial termination. Do not weaken the
    # simulation graph: only this separately validated execution substitutes it.
    axis = analyze_axis_dependencies(kernel.graph, kernel.params)
    # Inspection promises every declared state, including states that reset on
    # the next trial. Those are outside retained-state factorization analysis
    # but may not be replaced by dummy stochastic outputs in this interface.
    residual = _random_dependency_path(
        (clock_id,), {state.component_id for state in kernel.states},
        tuple(edge for edge in axis.edges if edge.kind != "schedule_termination_control"),
    )
    if residual is not None:
        _reject("history.state_dependency_unresolved", "Unobserved event values influence a state included in the history trace.")
    return HistoryReplayWitness(
        endpoint=endpoint,
        state_ids=tuple(state.state_id for state in kernel.states),
        effective_parameter_ids=tuple(item.effective_parameter_id for item in kernel.effective_parameters),
        component_ids=tuple(sorted(member.component_id for member in members)),
        resolved_termination_edges=tuple(edge for edge in axis.edges if edge.kind == "schedule_termination_control"),
    )


def validate_history_witness(simulation_plan, observations, witness):
    if type(witness) is not HistoryReplayWitness or witness != derive_history_witness(simulation_plan, observations):
        _reject("history.witness_mismatch", "History witness does not match the frozen source and observation contracts.")


@dataclass(frozen=True)
class HistoryTrace:
    """Read-only single-subject arrays; state starts precede trial resets.

    State arrays: [candidate, trial, flattened state]. Held-value arrays use
    the same axes with one column per effective parameter. Execution counts
    have one column per component, including post-finish scheduled calls.
    ``observations`` is populated only by unmodified forward simulation.
    """

    start_states: np.ndarray
    end_states: np.ndarray
    start_effective_parameters: np.ndarray
    end_effective_parameters: np.ndarray
    execution_counts: np.ndarray
    scheduler_rounds: np.ndarray
    event_counts: np.ndarray
    observations: np.ndarray | None


@dataclass(frozen=True)
class HistoryReplayPlan:
    simulation_plan: object = field(repr=False, compare=False)
    observations: ObservationSpec = field(repr=False, compare=False)
    witness: HistoryReplayWitness

    def source(self, *, replay=True):
        """Inspectable generated source; simulation mode changes no events."""
        from psyneulink.core.batched.backend.triton.history import HistoryTraceEmitter

        validate_history_witness(self.simulation_plan, self.observations, self.witness)
        return HistoryTraceEmitter(self.simulation_plan.kernel_ir, self.witness, replay=replay).emit()

    def reconstruct(self, inputs, data, parameter_sets=None):
        """Replay one subject from model defaults, conditioning on exact events."""
        from psyneulink.core.batched.backend.triton.history import run_history_trace

        validate_history_witness(self.simulation_plan, self.observations, self.witness)
        counts = self.simulation_plan.compile_observed_endpoints(self.observations).reconstruct(
            inputs, data, parameter_sets,
        )
        return run_history_trace(self, inputs, parameter_sets, counts=counts[..., 0])

    def simulate_reference(self, inputs, parameter_sets=None, *, seed=0):
        """Instrument ordinary coupled simulation, with one estimate per candidate.

        This does not force its stopping time and is a differential test oracle
        for scheduler/count/state replay, not an independent component oracle.
        """
        from psyneulink.core.batched.backend.triton.history import run_history_trace

        validate_history_witness(self.simulation_plan, self.observations, self.witness)
        return run_history_trace(self, inputs, parameter_sets, seed=seed)


def compile_history_replay(simulation_plan, observations):
    return HistoryReplayPlan(simulation_plan, observations, derive_history_witness(simulation_plan, observations))
