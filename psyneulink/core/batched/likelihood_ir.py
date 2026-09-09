"""Immutable contracts and diagnostic records for likelihood compilation.

These records describe structural evidence, not a likelihood implementation or
a formal probability certificate. Primitive contracts are trusted declarations
bound to the same implementation snapshot as simulation code generation.
"""

from dataclasses import asdict, dataclass

from psyneulink.core.batched.dependency import AxisDependencyEdge
from psyneulink.core.batched.diagnostics import BatchedCapabilityReport
from psyneulink.core.batched.observation import ResolvedObservationField


@dataclass(frozen=True)
class EventCountReadout:
    """Registered scalar readout: offset_parameter + counter * step_parameter.

    The counter is a primitive's completed active integration steps, not the
    enclosing scheduler's pass count. Relating those clocks is a separate rule.

    ``one_step_until_finished`` additionally asserts a zero counter and false
    finished flag on each trial, exactly one counter increment per scheduled
    execution while unfinished, and an absorbing finished flag/counter. Other
    trial-local values remain unobserved. This stronger author assertion is
    required for observed-event substitution; it is not inferred from the name
    of the counter or the algebraic readout alone.
    """

    output_port: str
    counter_state: str
    step_parameter: str
    offset_parameter: str
    minimum_count: int = 1
    execution_rule: str | None = None

    def __post_init__(self):
        if any(type(value) is not str or not value for value in (
            self.output_port, self.counter_state, self.step_parameter, self.offset_parameter,
        )):
            raise ValueError("Event readouts require nonempty port, state, and parameter names.")
        if type(self.minimum_count) is not int or self.minimum_count < 0:
            raise ValueError("Event readout minimum_count must be a nonnegative integer.")
        if self.execution_rule not in (None, "one_step_until_finished"):
            raise ValueError("Unknown event execution rule.")


@dataclass(frozen=True)
class HistoryReplayWitness:
    """Checked one-clock substitution, not a likelihood factorization proof."""

    endpoint: "EndpointWitness"
    state_ids: tuple[int, ...]
    effective_parameter_ids: tuple[int, ...]
    component_ids: tuple[int, ...]
    resolved_termination_edges: tuple[AxisDependencyEdge, ...]
    guarantee: str = "registered_contracts_with_checked_event_replay"


@dataclass(frozen=True)
class BoundaryField:
    """A typed value read at the stochastic step boundary, before its update."""

    kind: str
    value_name: str
    width: int
    column_start: int
    effective_parameter_id: int | None = None


@dataclass(frozen=True)
class BoundaryTrajectoryWitness:
    history: HistoryReplayWitness
    consumer_component_id: int
    consideration_set_id: int
    fields: tuple[BoundaryField, ...]
    source_component_ids: tuple[int, ...]
    static_parameter_ids: tuple[int, ...]
    guarantee: str = "registered_contracts_with_checked_pre_finish_boundary_prefix"


@dataclass(frozen=True)
class LikelihoodEffectContract:
    """Assert that an implementation's declared effects are complete.

    All retained state, initialization, control effects, and randomness must
    be represented by the lowered graph and registered state/RNG declarations.
    There may be no hidden state, external effects, or undeclared randomness.
    ``declared_streams`` also asserts fresh trial draws under the runtime RNG
    addressing contract; it does not assert independence of coupled outputs.

    This is an explicit author assertion, not a body-analysis or Lean proof.
    ``version`` identifies the contract revision within the frozen op spec.
    """

    randomness: str = "none"
    version: str = "1"
    value_rule: str | None = None
    event_readout: EventCountReadout | None = None

    def __post_init__(self):
        if self.randomness not in ("none", "declared_streams"):
            raise ValueError("Contract randomness must be 'none' or 'declared_streams'.")
        if type(self.version) is not str or not self.version:
            raise ValueError("A likelihood contract requires a nonempty version.")
        if self.value_rule not in (None, "affine", "dense_projection"):
            raise ValueError("Unknown likelihood value rule.")
        if self.event_readout is not None and type(self.event_readout) is not EventCountReadout:
            raise ValueError("event_readout must be an EventCountReadout.")
        if self.value_rule is not None and self.event_readout is not None:
            raise ValueError("A primitive cannot declare both an affine value and event readout.")


@dataclass(frozen=True)
class LikelihoodDiagnostic:
    code: str
    detail: str
    component_ids: tuple[int, ...] = ()
    dependency_path: tuple[AxisDependencyEdge, ...] = ()


@dataclass(frozen=True)
class PrimitiveLikelihoodEvidence:
    spec_key: str
    contract_version: str
    randomness: str
    trust: str = "registered_contract"


@dataclass(frozen=True)
class EndpointExpression:
    """Small scalar arithmetic IR; identities refer to the frozen model."""

    kind: str
    arguments: tuple["EndpointExpression", ...] = ()
    identity: int = -1
    value: float = 0.0


@dataclass(frozen=True)
class EndpointWitness:
    observation: ResolvedObservationField
    clock_component_id: int
    clock_spec_key: str
    counter_state: str
    minimum_count: int
    step_parameter_id: int
    expression: EndpointExpression
    component_ids: tuple[int, ...]
    projection_ids: tuple[int, ...]
    parameter_ids: tuple[int, ...]
    guarantee: str = "registered_readout_with_runtime_roundoff_guard"


@dataclass(frozen=True)
class LikelihoodCapabilityReport:
    """Separate structural diagnosis from likelihood codegen and execution.

    ``eligible`` currently means independent trial factorization under the
    declared contracts. ``candidate`` requires additional reconstruction and
    scheduler proofs. Neither means a likelihood evaluator was generated.
    """

    simulation: BatchedCapabilityReport
    observations: tuple[ResolvedObservationField, ...]
    history_kind: str
    factorization_status: str
    candidate_strategy: str | None = None
    retained_state_ids: tuple[int, ...] = ()
    held_parameter_ids: tuple[int, ...] = ()
    held_projection_ids: tuple[int, ...] = ()
    stochastic_component_ids: tuple[int, ...] = ()
    contracts: tuple[PrimitiveLikelihoodEvidence, ...] = ()
    diagnostics: tuple[LikelihoodDiagnostic, ...] = ()
    obligations: tuple[LikelihoodDiagnostic, ...] = ()
    endpoint_witnesses: tuple[EndpointWitness, ...] = ()
    input_policy: str = "conditioned"
    initial_state: str = "model_defaults"
    sequence_policy: str = "explicit_subject_boundaries"
    semantic_target: str = "source_discrete_execution"

    @property
    def codegen_ready(self) -> bool:
        # Event readouts can be reconstructed, but no likelihood is generated.
        return False

    @property
    def can_execute(self) -> bool:
        return False

    def to_dict(self) -> dict:
        result = asdict(self)
        result["simulation"] = self.simulation.to_dict()
        result["codegen_ready"] = self.codegen_ready
        result["can_execute"] = self.can_execute
        return result
