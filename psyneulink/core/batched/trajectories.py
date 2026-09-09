"""Checked deterministic boundary paths from observation-conditioned starts.

History replay and hypothetical within-trial tails are separate computations.
Only the former may update the canonical start state of the next trial.
"""

from dataclasses import dataclass

import numpy as np

from psyneulink.core.batched.dependency import analyze_axis_dependencies
from psyneulink.core.batched.history import HistoryTrace, replay_program, validate_history_witness
from psyneulink.core.batched.likelihood_analysis import _random_dependency_path
from psyneulink.core.batched.likelihood_ir import BoundaryField, BoundaryTrajectoryWitness


class BoundaryTrajectoryError(ValueError):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def derive_boundary_witness(history_plan):
    validate_history_witness(history_plan.simulation_plan, history_plan.observations, history_plan.witness)
    kernel = history_plan.simulation_plan.kernel_ir
    clock = history_plan.witness.endpoint.clock_component_id
    program = replay_program(kernel)
    group, member = next((group, member) for group in program.consideration_sets
                         for member in group.members if member.component_id == clock)
    step = next(op for op in member.body if op.kind == "StepMechanism")
    budget = next(budget for budget in program.execution_budgets if budget.component_id == clock)
    if budget.post_finish != "stop":
        raise BoundaryTrajectoryError("boundary.clock_policy", "Boundary paths currently require an event that stops scheduled execution when finished.")
    effective_ids = tuple(step.attrs["sampled_effective_parameter_ids"])
    values = step.inputs[:1 + len(effective_ids)]
    fields, column = [], 0
    for index, value in enumerate(values):
        fields.append(BoundaryField(
            "input" if index == 0 else "held_modulation", value.name, value.width,
            column, None if index == 0 else effective_ids[index - 1],
        ))
        column += value.width
    # Cross-region inputs and held controls must not depend on unobserved
    # stochastic values, including effects that never reach retained state.
    sources = {projection.sender_component_id for projection in kernel.graph.projections
               if projection.receiver_component_id == clock}
    # Step eligibility is also a boundary dependency: a deterministic value
    # sampled at an unobserved stochastic execution time is not a known path.
    sources.update(member.predicate.dependency_component_ids)
    sources.update(control.controller_component_id
                   for control in (*kernel.graph.modulations, *kernel.graph.folded_affine_controls)
                   if control.effective_parameter_id in effective_ids)
    axis = analyze_axis_dependencies(kernel.graph, kernel.params)
    edges = tuple(edge for edge in axis.edges if edge.kind != "schedule_termination_control")
    residual = _random_dependency_path((clock,), sources, edges)
    if residual is not None:
        raise BoundaryTrajectoryError("boundary.stochastic_dependency", "A boundary value depends on an unobserved stochastic value or scheduler effect.")
    # Preserve the full dependency closure in the checked artifact. Source
    # topology and parameter identity are rederived, never supplied by name.
    ancestors = set(sources)
    while True:
        expanded = ancestors | {edge.producer_component_id for edge in edges if edge.consumer_component_id in ancestors}
        if expanded == ancestors:
            break
        ancestors = expanded
    node = kernel.graph.node(step.target)
    parameter_names = set(node.params.values())
    return BoundaryTrajectoryWitness(
        history_plan.witness, clock, group.consideration_set_id, tuple(fields),
        tuple(sorted(ancestors)), tuple(p.parameter_id for p in kernel.params if p.name in parameter_names),
    )


def validate_boundary_witness(history_plan, witness):
    if type(witness) is not BoundaryTrajectoryWitness or witness != derive_boundary_witness(history_plan):
        raise BoundaryTrajectoryError("boundary.witness_mismatch", "Boundary witness does not match the frozen source step and dependencies.")


@dataclass(frozen=True)
class BoundaryTrajectories:
    history: HistoryTrace
    values: np.ndarray  # [candidate, trial, active step, field column]
    valid: np.ndarray  # [candidate, trial, active step]
    pass_indices: np.ndarray  # zero-based scheduler pass for each active step
    fields: tuple[BoundaryField, ...]
    mode: str


@dataclass(frozen=True)
class BoundaryTrajectoryPlan:
    history_plan: object
    witness: BoundaryTrajectoryWitness

    def compile_stochastic_sampler(self):
        from psyneulink.core.batched.sampling import compile_stochastic_sampler

        return compile_stochastic_sampler(self)

    def source(self, *, parallel_trials=True):
        """Generate deterministic paths, or instrument unmodified simulation."""
        from psyneulink.core.batched.backend.triton.trajectories import BoundaryTrajectoryEmitter

        validate_boundary_witness(self.history_plan, self.witness)
        return BoundaryTrajectoryEmitter(
            self.history_plan.simulation_plan.kernel_ir, self.witness, parallel_trials=parallel_trials,
        ).emit()

    def generate(self, inputs, data, parameter_sets=None, *, horizon=None, max_buffer_bytes=256 * 1024**2):
        """Generate paths in trial-parallel lanes from internally replayed starts.

        Supports interpreter and compiled GPU plans. No externally supplied start state or
        hypothetical path endpoint can replace the observed sequence history.
        """
        from psyneulink.core.batched.backend.triton.trajectories import run_boundary_trajectories

        validate_boundary_witness(self.history_plan, self.witness)
        return run_boundary_trajectories(self, inputs, data, parameter_sets, horizon, max_buffer_bytes)

    def simulate_reference(self, inputs, parameter_sets=None, *, seed=0, horizon=None, max_buffer_bytes=256 * 1024**2):
        """Capture the actual pre-step boundary in one coupled simulation lane."""
        from psyneulink.core.batched.backend.triton.trajectories import run_boundary_trajectories

        validate_boundary_witness(self.history_plan, self.witness)
        return run_boundary_trajectories(
            self, inputs, None, parameter_sets, horizon, max_buffer_bytes, reference=True, seed=seed,
        )


def compile_boundary_trajectories(history_plan):
    return BoundaryTrajectoryPlan(history_plan, derive_boundary_witness(history_plan))
