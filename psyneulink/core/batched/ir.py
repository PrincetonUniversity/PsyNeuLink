from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Every integer through this bound is represented exactly by IEEE-754 fp32.
# Batched inputs and parameter buffers currently use fp32, so values that carry
# discrete execution semantics must stay within this range before conversion.
FP32_EXACT_INTEGER_LIMIT = 2 ** 24


@dataclass(frozen=True)
class BatchedTrialParameter:
    """Values for one batched model parameter that vary across trials.

    ``values`` may be a one-dimensional trial vector or a two-dimensional
    ``[subject, trial]`` array.  The runtime validates its shape against the
    prepared inputs.  This explicit wrapper keeps ordinary vector-valued
    mappings available as the existing shorthand for multiple parameter sets.
    """

    values: Any


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
    owner_component_id: int = -1
    owner_scope: str = ""


@dataclass(frozen=True)
class BatchedParameterBindingSpec:
    """Bind one registered implementation argument to one lowered parameter."""

    argument: str
    parameter: str
    parameter_id: int

    def __post_init__(self) -> None:
        if (
            type(self.argument) is not str
            or not self.argument
            or type(self.parameter) is not str
            or not self.parameter
            or type(self.parameter_id) is not int
            or self.parameter_id < 0
        ):
            raise ValueError(
                "Batched parameter bindings require nonempty labels and a "
                "non-negative non-bool parameter ID."
            )


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
class BatchedAbsorbedProjectionSpec:
    """A validated projection intentionally represented by another IR effect."""

    projection_id: int
    name: str
    kind: str
    sender: str
    sender_component_id: int
    sender_port: str
    sender_port_id: int
    receiver: str
    receiver_component_id: int
    receiver_port: str
    receiver_port_id: int
    width: int = 1
    reason: str = "typed_scalar_override"
    initial_value: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        ids = (
            self.projection_id,
            self.sender_component_id,
            self.sender_port_id,
            self.receiver_component_id,
            self.receiver_port_id,
        )
        if any(type(value) is not int or value < 0 for value in ids):
            raise ValueError(
                "Batched absorbed-projection identities must be non-negative "
                "non-bool integers."
            )
        if any(
            type(value) is not str or not value
            for value in (
                self.name,
                self.kind,
                self.sender,
                self.sender_port,
                self.receiver,
                self.receiver_port,
            )
        ):
            raise ValueError(
                "Batched absorbed-projection labels must be nonempty strings."
            )
        if (
            self.kind not in {"MappingProjection", "ControlProjection"}
            or self.width != 1
            or self.reason != "typed_scalar_override"
        ):
            raise ValueError(
                "Batched absorbed control projections currently require a "
                "scalar typed OVERRIDE identity chain."
            )
        if type(self.initial_value) is not tuple:
            raise ValueError(
                "Batched absorbed-projection initial values must be tuples."
            )
        if self.kind == "MappingProjection" and self.initial_value:
            raise ValueError(
                "Batched absorbed MappingProjections do not retain a held value."
            )
        if self.kind == "ControlProjection":
            array = np.asarray(self.initial_value)
            packed = array.astype(np.float32) if array.dtype.kind in "biuf" else array
            if (
                len(self.initial_value) != self.width
                or array.dtype.kind not in "biuf"
                or not bool(np.all(np.isfinite(array)))
                or not bool(np.all(np.isfinite(packed)))
            ):
                raise ValueError(
                    "Batched absorbed ControlProjection initial value must be "
                    "finite real scalar data representable in float32 range."
                )


@dataclass(frozen=True)
class BatchedPortSpec:
    """Object-free identity and ownership for one live PsyNeuLink Port."""

    port_id: int
    name: str
    owner: str
    owner_component_id: int
    kind: str
    width: int

    def __post_init__(self) -> None:
        if (
            type(self.port_id) is not int
            or self.port_id < 0
            or type(self.owner_component_id) is not int
            or self.owner_component_id < 0
        ):
            raise ValueError(
                "Batched port identities must be non-negative non-bool integers."
            )
        if any(
            type(value) is not str or not value
            for value in (self.name, self.owner, self.kind)
        ):
            raise ValueError("Batched port labels and kinds must be nonempty strings.")
        if type(self.width) is not int or self.width <= 0:
            raise ValueError("Batched port width must be a positive non-bool integer.")


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
    # Canonical lowering-time port ownership.  The graph-wide port inventory
    # describes each port, while these ordered IDs anchor those declarations to
    # the live component's InputPort, OutputPort, and named ParameterPort
    # collections.  Parameter names are semantic here because registered
    # mechanism capabilities select controllable parameters by their PNL name.
    input_port_ids: tuple[int, ...] = ()
    output_port_ids: tuple[int, ...] = ()
    parameter_port_ids: tuple[tuple[str, int], ...] = ()


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
    # Object-free semantic definition of the finished value.  ``dynamic`` is
    # declaration-only; the first executable form is
    # ``execution_count_at_least`` with ``attrs={"count": N}``, evaluated after
    # the owner executes and reset with the scheduler's per-trial counts.
    predicate_kind: str = "dynamic"
    attrs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchedEffectiveParameterSpec:
    """Held modulation value and target-parameter sampling semantics.

    PsyNeuLink modulation is not an ordinary data edge.  A ControlProjection
    retains its last value across trials, updates that value when its controller
    executes, and the target ParameterPort samples it when its owner executes.
    This declaration records those semantics before an executable KernelIR op
    is introduced.
    """

    effective_parameter_id: int
    target: str
    target_component_id: int
    target_parameter: str
    target_parameter_port_id: int
    base_value: tuple[float, ...]
    initial_modulation_value: tuple[float, ...]
    width: int = 1
    dtype: str = "float32"
    storage: str = "lane_persistent"
    reset: str = "Never"
    update_event: str = "after_controller_execution"
    sample_event: str = "at_target_parameter_update"

    def __post_init__(self) -> None:
        ids = (
            self.effective_parameter_id,
            self.target_component_id,
            self.target_parameter_port_id,
        )
        if any(type(value) is not int or value < 0 for value in ids):
            raise ValueError(
                "Batched effective-parameter identities must be non-negative "
                "non-bool integers."
            )
        if any(
            type(value) is not str or not value
            for value in (self.target, self.target_parameter)
        ):
            raise ValueError(
                "Batched effective-parameter labels must be nonempty strings."
            )
        if (
            self.width != 1
            or self.dtype != "float32"
            or self.storage != "lane_persistent"
            or self.reset != "Never"
            or self.update_event != "after_controller_execution"
            or self.sample_event != "at_target_parameter_update"
        ):
            raise ValueError(
                "Batched effective parameters currently require scalar float32 "
                "lane-persistent held OVERRIDE semantics."
            )
        for label, values in (
            ("base", self.base_value),
            ("initial modulation", self.initial_modulation_value),
        ):
            if type(values) is not tuple or len(values) != self.width:
                raise ValueError(
                    f"Batched effective-parameter {label} value must match its width."
                )
            array = np.asarray(values)
            if array.dtype.kind not in "biuf":
                raise ValueError(
                    f"Batched effective-parameter {label} value must be real numeric."
                )
            packed = array.astype(np.float32)
            if (
                not bool(np.all(np.isfinite(array)))
                or not bool(np.all(np.isfinite(packed)))
            ):
                raise ValueError(
                    f"Batched effective-parameter {label} value must be finite "
                    "and representable in float32 range."
                )


@dataclass(frozen=True)
class BatchedFoldedAffineControlSpec:
    """A scheduler-owned affine update for an absorbed control pathway.

    Some stateful control pathways have no ordinary processing edge after
    semantic lowering.  Their observable effect is nevertheless explicit: a
    scheduled controller publishes ``base + delta * execution_ordinal`` and
    commits that value to one lane-persistent effective parameter.  Numeric
    identities bind every participant without retaining the omitted source or
    a live PsyNeuLink object in GraphIR.
    """

    folded_control_id: int
    effective_parameter_id: int
    controller: str
    controller_component_id: int
    controller_output_port_id: int
    target: str
    target_component_id: int
    target_parameter: str
    target_parameter_port_id: int
    base_parameter: str
    base_parameter_id: int
    delta_parameter: str
    delta_parameter_id: int
    clock_component_id: int
    initial_value: tuple[float, ...]
    width: int = 1
    dtype: str = "float32"
    storage: str = "lane_persistent"
    reset: str = "Never"
    update_event: str = "after_controller_execution"
    update_expression: str = "base_plus_delta_times_controller_execution_ordinal"
    sample_event: str = "at_target_parameter_update"

    def __post_init__(self) -> None:
        ids = (
            self.folded_control_id,
            self.effective_parameter_id,
            self.controller_component_id,
            self.controller_output_port_id,
            self.target_component_id,
            self.target_parameter_port_id,
            self.base_parameter_id,
            self.delta_parameter_id,
            self.clock_component_id,
        )
        if any(type(value) is not int or value < 0 for value in ids):
            raise ValueError(
                "Batched folded-control identities must be non-negative "
                "non-bool integers."
            )
        if any(
            type(value) is not str or not value
            for value in (
                self.controller,
                self.target,
                self.target_parameter,
                self.base_parameter,
                self.delta_parameter,
            )
        ):
            raise ValueError(
                "Batched folded-control labels must be nonempty strings."
            )
        if (
            self.base_parameter_id == self.delta_parameter_id
            or self.clock_component_id != self.controller_component_id
            or self.width != 1
            or self.dtype != "float32"
            or self.storage != "lane_persistent"
            or self.reset != "Never"
            or self.update_event != "after_controller_execution"
            or self.update_expression
            != "base_plus_delta_times_controller_execution_ordinal"
            or self.sample_event != "at_target_parameter_update"
        ):
            raise ValueError(
                "Batched folded controls require the exact scalar affine "
                "scheduler-update policy."
            )
        if type(self.initial_value) is not tuple or len(self.initial_value) != 1:
            raise ValueError(
                "Batched folded-control initial value must be a scalar tuple."
            )
        array = np.asarray(self.initial_value)
        packed = array.astype(np.float32) if array.dtype.kind in "biuf" else array
        if (
            array.dtype.kind not in "biuf"
            or not bool(np.all(np.isfinite(array)))
            or not bool(np.all(np.isfinite(packed)))
        ):
            raise ValueError(
                "Batched folded-control initial value must be finite real "
                "scalar data representable in float32 range."
            )


@dataclass(frozen=True)
class BatchedModulationSpec:
    """One object-free effective-parameter edge.

    The first declared subset is a scalar ``OVERRIDE`` supplied by one
    ControlMechanism.  Numeric component, port, and parameter IDs are the
    semantic identity; names remain diagnostic labels.  The identity port IDs
    make the intentionally absorbed monitor/signal/projection chain auditable
    without retaining live PsyNeuLink objects in the IR.
    """

    modulation_id: int
    controller: str
    controller_component_id: int
    controller_input_port: str
    controller_input_port_id: int
    control_signal_port: str
    control_signal_port_id: int
    source: str
    source_component_id: int
    source_port: str
    source_port_id: int
    target: str
    target_component_id: int
    target_parameter: str
    target_parameter_port_id: int
    effective_parameter_id: int
    monitor_projection_id: int
    control_projection_id: int
    mode: str = "OVERRIDE"
    width: int = 1
    dtype: str = "float32"
    absorbed_identity_chain: bool = True
    controller_function_spec_key: str = ""
    controller_param_bindings: tuple[BatchedParameterBindingSpec, ...] = ()

    def __post_init__(self) -> None:
        ids = (
            self.modulation_id,
            self.controller_component_id,
            self.controller_input_port_id,
            self.control_signal_port_id,
            self.source_component_id,
            self.source_port_id,
            self.target_component_id,
            self.target_parameter_port_id,
            self.effective_parameter_id,
            self.monitor_projection_id,
            self.control_projection_id,
        )
        if any(type(value) is not int or value < 0 for value in ids):
            raise ValueError(
                "Batched modulation identities must be non-negative "
                "non-bool integers."
            )
        labels = (
            self.controller,
            self.controller_input_port,
            self.control_signal_port,
            self.source,
            self.source_port,
            self.target,
            self.target_parameter,
        )
        if any(type(value) is not str or not value for value in labels):
            raise ValueError("Batched modulation labels must be nonempty strings.")
        if self.mode != "OVERRIDE" or self.width != 1 or self.dtype != "float32":
            raise ValueError(
                "Batched modulation currently requires scalar float32 OVERRIDE."
            )
        if self.absorbed_identity_chain is not True:
            raise ValueError(
                "Batched modulation currently requires a validated absorbed "
                "identity projection chain."
            )
        if type(self.controller_param_bindings) is not tuple or any(
            type(binding) is not BatchedParameterBindingSpec
            for binding in self.controller_param_bindings
        ):
            raise ValueError(
                "Batched modulation controller bindings must be a tuple of "
                "BatchedParameterBindingSpec values."
            )
        arguments = tuple(
            binding.argument for binding in self.controller_param_bindings
        )
        parameter_ids = tuple(
            binding.parameter_id for binding in self.controller_param_bindings
        )
        if len(set(arguments)) != len(arguments) or len(set(parameter_ids)) != len(
            parameter_ids
        ):
            raise ValueError(
                "Batched modulation controller bindings must have unique "
                "arguments and parameter IDs."
            )
        if bool(self.controller_function_spec_key) != bool(
            self.controller_param_bindings
        ):
            raise ValueError(
                "Batched modulation controller implementation and parameter "
                "bindings must either both be declared or both be empty."
            )


@dataclass(frozen=True)
class BatchedResetSpec:
    """A retained-state reset policy owned by one component.

    ``Never`` is represented explicitly as well as reset events so a backend
    cannot accidentally turn persistent state into trial-local state.  Storage
    that has been semantically optimized away has no state ID and is omitted;
    registered trial-local mechanism state is represented by KernelIR loop
    carries rather than these retained GraphIR reset records.
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
    ports: tuple[BatchedPortSpec, ...] = ()
    absorbed_projections: tuple[BatchedAbsorbedProjectionSpec, ...] = ()
    rng_streams: tuple[BatchedRngStreamSpec, ...] = ()
    schedule_regions: tuple[BatchedScheduleRegionSpec, ...] = ()
    consideration_sets: tuple[BatchedConsiderationSetSpec, ...] = ()
    finished_values: tuple[BatchedFinishedValueSpec, ...] = ()
    effective_parameters: tuple[BatchedEffectiveParameterSpec, ...] = ()
    modulations: tuple[BatchedModulationSpec, ...] = ()
    folded_affine_controls: tuple[BatchedFoldedAffineControlSpec, ...] = ()
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
