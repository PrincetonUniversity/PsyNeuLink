from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Every integer through this bound is represented exactly by IEEE-754 fp32.
# Batched inputs and parameter buffers currently use fp32, so values that carry
# discrete execution semantics must stay within this range before conversion.
FP32_EXACT_INTEGER_LIMIT = 2 ** 24


@dataclass(frozen=True)
class BatchedParamSpec:
    name: str
    default: float
    aliases: tuple[str, ...] = ()
    parameter_id: int = -1
    minimum: float | None = None
    minimum_inclusive: bool = True
    maximum: float | None = None
    maximum_inclusive: bool = True
    runtime_mutable: bool = True
    runtime_constraint: str = ""


@dataclass(frozen=True)
class BatchedInputSpec:
    name: str
    node: str
    width: int
    component_id: int = -1
    port_id: int = -1
    port: str = ""


@dataclass(frozen=True)
class BatchedOutputSpec:
    name: str
    node: str
    port: str
    width: int
    component_id: int = -1
    port_id: int = -1
    flat_start: int = -1
    flat_stop: int = -1

    def __post_init__(self) -> None:
        if self.flat_start == -1 and self.flat_stop == -1:
            return
        if self.flat_start < 0 or self.flat_stop < 0:
            raise ValueError(
                f"Batched output '{self.name}' requires both flattened bounds."
            )
        if self.flat_stop < self.flat_start:
            raise ValueError(
                f"Batched output '{self.name}' has flat_stop {self.flat_stop} "
                f"before flat_start {self.flat_start}."
            )
        if self.flat_stop - self.flat_start != self.width:
            raise ValueError(
                f"Batched output '{self.name}' flattened slice width "
                f"{self.flat_stop - self.flat_start} does not match output width "
                f"{self.width}."
            )

    @property
    def flat_slice(self) -> slice:
        if self.flat_start < 0 or self.flat_stop < 0:
            raise ValueError(
                f"Batched output '{self.name}' has no flattened slice assignment."
            )
        return slice(self.flat_start, self.flat_stop)


@dataclass(frozen=True)
class BatchedProjectionSpec:
    sender: str
    sender_port: str
    receiver: str
    receiver_port: str
    matrix: np.ndarray
    spec_key: str = ""
    projection_id: int = -1
    sender_component_id: int = -1
    sender_port_id: int = -1
    receiver_component_id: int = -1
    receiver_port_id: int = -1


@dataclass(frozen=True)
class BatchedNodeSpec:
    name: str
    component_type: str
    function_type: str
    input_width: int
    output_width: int
    combine: str = "sum"
    params: Mapping[str, str] = field(default_factory=dict)
    attrs: Mapping[str, Any] = field(default_factory=dict)
    # Deterministic lowering-local identity.  ``name`` remains the public PNL
    # lookup/display contract; code generation must use this numeric identity
    # so distinct names that sanitize to the same target identifier cannot
    # alias one another.
    component_id: int = -1


@dataclass(frozen=True)
class BatchedStateFunctionInitializer:
    """Initialize state by applying a registered elementwise function.

    ``input_value`` is the function input before lane-specific parameters are
    applied. ``params`` maps the function implementation's argument names to
    stable public parameter names in :class:`BatchedCompositionIR`.
    """

    spec_key: str
    input_value: tuple[float, ...]
    params: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchedStateSpec:
    name: str
    node: str
    width: int
    initial_value: tuple[float, ...]
    component_id: int = -1
    state_id: int = -1
    function_initializer: BatchedStateFunctionInitializer | None = None


@dataclass(frozen=True)
class BatchedRngStreamSpec:
    name: str
    node: str
    width: int
    step_extent: str
    component_id: int = -1
    stream_id: int = -1


@dataclass(frozen=True)
class BatchedSchedulerSpec:
    """One explicit scheduler predicate attached to a graph component.

    ``dependencies`` and their numeric IDs identify predicate operands such as
    the mechanism referenced by ``WhenFinished``.  ``finished_value_ids`` then
    names the corresponding boolean values declared in
    :class:`BatchedFinishedValueSpec`.  Condition-specific scalar data (for
    example an ``AtPass`` index and time scale) remains in ``attrs`` so this
    schema can grow without embedding PsyNeuLink condition objects in the IR.
    """

    node: str
    condition_type: str
    dependencies: tuple[str, ...] = ()
    attrs: Mapping[str, Any] = field(default_factory=dict)
    component_id: int = -1
    dependency_component_ids: tuple[int, ...] = ()
    finished_value_ids: tuple[int, ...] = ()
    region: str = "pass"
    consideration_set_id: int = -1


@dataclass(frozen=True)
class BatchedScheduleRegionSpec:
    """A semantic scheduler region independent of backend loop placement."""

    name: str
    kind: str
    time_scale: str
    parent: str = ""


@dataclass(frozen=True)
class BatchedConsiderationSetSpec:
    """One ordered scheduler consideration set within a pass.

    Members of a consideration set observe inputs frozen at the beginning of
    that set, while sets are considered in ascending ``consideration_set_id``
    order.  This distinction is required for predicates that can become true
    part-way through a pass: a later set may observe the transition in that
    pass, whereas an earlier set waits until the next pass.
    """

    consideration_set_id: int
    nodes: tuple[str, ...]
    component_ids: tuple[int, ...]
    region: str = "pass"
    inputs_frozen: bool = True


@dataclass(frozen=True)
class BatchedTerminationSpec:
    """One typed scheduler termination predicate.

    Termination is independent of node execution predicates: the scheduler
    reevaluates it between consideration-set executions.  Component operands
    are expanded to stable numeric IDs during semantic lowering so a host or
    backend planner never needs live PsyNeuLink objects to interpret the
    predicate.
    """

    time_scale: str
    condition_type: str
    dependency_component_ids: tuple[int, ...] = ()
    attrs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchedScheduleTraceStepSpec:
    """One nonempty consideration-set execution in a precomputed trace.

    Within the supported precomputed subset, all call-count dependencies must
    originate in an earlier consideration set.  Members can therefore be
    selected from the same beginning-of-set scheduler snapshot.
    ``component_ids`` is an unordered execution set with deterministic storage
    order, not a sequence whose earlier members may make later members
    eligible.
    """

    pass_index: int
    consideration_set_id: int
    component_ids: tuple[int, ...]


@dataclass(frozen=True)
class BatchedScheduleTraceSpec:
    """Finite, lane-invariant execution trace for one trial.

    Empty consideration-set visits are omitted because they have no execution
    effect.  Their timing is retained by the absolute ``pass_index`` on each
    nonempty step and by ``num_passes``.  ``component_execution_count`` is the
    expansion quantity bounded by the host planner before KernelIR duplicates
    any component bodies.
    """

    steps: tuple[BatchedScheduleTraceStepSpec, ...]
    num_passes: int
    component_execution_count: int


@dataclass(frozen=True)
class BatchedFinishedValueSpec:
    """Boolean ``Mechanism.is_finished`` value consumed by scheduler predicates."""

    name: str
    node: str
    component_id: int = -1
    value_id: int = -1
    width: int = 1
    dtype: str = "bool"
    storage: str = "combinational"
    producer_consideration_set_id: int = -1


@dataclass(frozen=True)
class BatchedResetSpec:
    """A retained-state reset policy owned by one component.

    ``Never`` is represented explicitly as well as reset events so a backend
    cannot accidentally turn persistent state into trial-local state.  Storage
    that has been semantically optimized away has no state ID and is omitted;
    emitter-private trial state will move into this schema in a later slice.
    """

    node: str
    condition_type: str
    state_ids: tuple[int, ...] = ()
    attrs: Mapping[str, Any] = field(default_factory=dict)
    component_id: int = -1
    region: str = "trial"


@dataclass(frozen=True)
class BatchedOp:
    kind: str
    target: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    attrs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchedGraphIR:
    nodes: tuple[BatchedNodeSpec, ...]
    inputs: tuple[BatchedInputSpec, ...]
    projections: tuple[BatchedProjectionSpec, ...]
    outputs: tuple[BatchedOutputSpec, ...]
    states: tuple[BatchedStateSpec, ...]
    scheduler: tuple[BatchedSchedulerSpec, ...]
    ops: tuple[BatchedOp, ...]
    execution_order: tuple[str, ...]
    fusion_kind: str | None = None
    executable: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    rng_streams: tuple[BatchedRngStreamSpec, ...] = ()
    schedule_regions: tuple[BatchedScheduleRegionSpec, ...] = ()
    consideration_sets: tuple[BatchedConsiderationSetSpec, ...] = ()
    finished_values: tuple[BatchedFinishedValueSpec, ...] = ()
    resets: tuple[BatchedResetSpec, ...] = ()
    termination: tuple[BatchedTerminationSpec, ...] = ()

    def node(self, name: str) -> BatchedNodeSpec:
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(name)


@dataclass(frozen=True)
class BatchedCompositionIR:
    model_kind: str
    node_names: tuple[str, ...]
    params: tuple[BatchedParamSpec, ...]
    output_names: tuple[str, ...] = ("decision", "response_time")
    max_steps: int = 3000
    graph: BatchedGraphIR | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def param_defaults(self) -> Mapping[str, float]:
        return {p.name: p.default for p in self.params}


@dataclass(frozen=True)
class BatchedSimulationResult:
    values: np.ndarray
    output_names: tuple[str, ...]
    backend: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
