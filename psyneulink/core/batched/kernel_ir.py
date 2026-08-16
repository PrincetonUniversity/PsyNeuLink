from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

from psyneulink.core.batched.graph import (
    COEVOLVING_GRAPH_FUSION,
    STATEFUL_GRAPH_FUSION,
    STATELESS_GRAPH_FUSION,
    projection_inputs,
)
from psyneulink.core.batched.ir import (
    BatchedAbsorbedProjectionSpec,
    BatchedCompositionIR,
    BatchedConsiderationSetSpec,
    BatchedEffectiveParameterSpec,
    BatchedFinishedValueSpec,
    BatchedGraphIR,
    BatchedInputSpec,
    BatchedModulationSpec,
    BatchedOutputSpec,
    BatchedParamSpec,
    BatchedPortSpec,
    BatchedResetSpec,
    BatchedScheduleTraceSpec,
    BatchedScheduleRegionSpec,
    BatchedSchedulerSpec,
    BatchedStateSpec,
    BatchedTerminationSpec,
)
from psyneulink.core.batched.schedule import (
    PRECOMPUTED_TRACE_COMPONENT_BUDGET,
    plan_precomputed_schedule_trace,
)
from psyneulink.core.batched.specs import (
    BatchedOpSpecError,
    BatchedOpSpecSnapshot,
    ElementwiseFunctionSpec,
    MechanismOpSpec,
    snapshot_batched_op_specs,
)


TRIAL_LANE_LAYOUT = "trial"
STATEFUL_LANE_LAYOUT = "stateful"
KernelConstant = Real | Iterable[Real]
_DEFAULT_TRACE_WEIGHTED_OP_BUDGET = 65536
_TRACE_COMPONENT_BUDGET_KEY = "schedule_trace_component_budget"
_TRACE_WEIGHTED_OP_BUDGET_KEY = "schedule_trace_weighted_op_budget"


@dataclass(frozen=True)
class KernelLaneLayout:
    """Lane indexing policy for batched execution.

    MLIR note: this maps to outer parallel loop dimensions.  Trial lanes use
    one lane per `(parameter, subject, trial, estimate)`.  Stateful lanes use
    one lane per `(parameter, subject, estimate)` and represent trials with an
    inner structured loop so lane-local state can persist across trials.
    """

    kind: str
    dimensions: tuple[str, ...]


@dataclass(frozen=True)
class KernelValue:
    """Typed symbolic value produced or consumed by KernelIR ops."""

    name: str
    width: int
    dtype: str = "float32"


@dataclass(frozen=True)
class KernelRngStream:
    """Lane-local random stream descriptor.

    MLIR note: this is intentionally a semantic stream id plus step extent, not
    a Triton offset expression.  Backends lower it to their target RNG ABI.
    """

    name: str
    node: str
    width: int
    step_extent: str
    component_id: int = -1
    stream_id: int = -1


@dataclass(frozen=True)
class KernelOp:
    """Backend-neutral execution op.

    The op set is deliberately close to MLIR structured lowering: explicit
    buffer reads/writes, arithmetic/math operations, state effects, RNG reads,
    and structured loop bodies.  `attrs` may hold Python values such as dense
    matrices or parameter names, but must not hold backend source fragments.
    """

    kind: str
    target: str
    inputs: tuple[KernelValue, ...] = ()
    outputs: tuple[KernelValue, ...] = ()
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind == "AddConstant":
            _validate_constant_elementwise_op(self, ("value",))
        elif self.kind == "Clamp":
            _validate_constant_elementwise_op(self, ("lower", "upper"))
            _validate_clamp_bounds(self)
        elif self.kind == "Concatenate":
            _validate_concatenate(self)
        elif self.kind == "ExtractSlice":
            _validate_extract_slice(self)
        elif self.kind == "ForPasses":
            _validate_for_passes(self)
        elif self.kind == "ExecuteConsiderationSet":
            _validate_execute_consideration_set(self)
        elif self.kind == "StepMechanism":
            _validate_step_mechanism(self)
        elif self.kind == "ResetState":
            _validate_reset_state(self)


def add_constant_op(
    *,
    target: str,
    input_value: KernelValue,
    output_value: KernelValue,
    value: KernelConstant,
) -> KernelOp:
    """Build an elementwise constant addition with scalar broadcast support."""

    return KernelOp(
        kind="AddConstant",
        target=target,
        inputs=(input_value,),
        outputs=(output_value,),
        attrs={"value": value},
    )


def clamp_op(
    *,
    target: str,
    input_value: KernelValue,
    output_value: KernelValue,
    lower: KernelConstant,
    upper: KernelConstant,
) -> KernelOp:
    """Build an elementwise clamp with scalar or exact-width vector bounds."""

    return KernelOp(
        kind="Clamp",
        target=target,
        inputs=(input_value,),
        outputs=(output_value,),
        attrs={"lower": lower, "upper": upper},
    )


def _validate_constant_elementwise_op(
    op: KernelOp,
    constant_attrs: tuple[str, ...],
) -> None:
    if len(op.inputs) != 1 or len(op.outputs) != 1:
        raise ValueError(
            f"KernelIR {op.kind} requires exactly one input and one output."
        )
    input_value = op.inputs[0]
    output_value = op.outputs[0]
    if input_value.width != output_value.width:
        raise ValueError(
            f"KernelIR {op.kind} input/output widths must match, got "
            f"{input_value.width} and {output_value.width}."
        )
    if input_value.dtype != output_value.dtype:
        raise ValueError(
            f"KernelIR {op.kind} input/output dtypes must match, got "
            f"'{input_value.dtype}' and '{output_value.dtype}'."
        )

    attrs = dict(op.attrs)
    for attr in constant_attrs:
        try:
            value = attrs[attr]
        except KeyError as error:
            raise ValueError(
                f"KernelIR {op.kind} requires a '{attr}' constant."
            ) from error
        attrs[attr] = _normalize_constant(
            value,
            width=input_value.width,
            op_kind=op.kind,
            attr=attr,
        )
    object.__setattr__(op, "attrs", attrs)


def _normalize_constant(value, *, width: int, op_kind: str, attr: str) -> tuple[float, ...]:
    if isinstance(value, Real):
        values = (float(value),)
    elif isinstance(value, (str, bytes)):
        raise ValueError(
            f"KernelIR {op_kind} '{attr}' must be a numeric scalar or vector."
        )
    else:
        try:
            values = tuple(float(component) for component in value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"KernelIR {op_kind} '{attr}' must be a numeric scalar or vector."
            ) from error

    if len(values) not in (1, width):
        raise ValueError(
            f"KernelIR {op_kind} '{attr}' must be scalar or have width {width}, "
            f"got width {len(values)}."
        )
    return values


def _validate_clamp_bounds(op: KernelOp) -> None:
    width = op.inputs[0].width
    lower = op.attrs["lower"]
    upper = op.attrs["upper"]
    for index in range(width):
        component_lower = lower[0] if len(lower) == 1 else lower[index]
        component_upper = upper[0] if len(upper) == 1 else upper[index]
        if component_lower > component_upper:
            raise ValueError(
                "KernelIR Clamp lower bound exceeds upper bound at component "
                f"{index}: {component_lower} > {component_upper}."
            )


def _validate_concatenate(op: KernelOp) -> None:
    if not op.inputs or len(op.outputs) != 1:
        raise ValueError(
            "KernelIR Concatenate requires at least one input and exactly one output."
        )
    output = op.outputs[0]
    if sum(value.width for value in op.inputs) != output.width:
        raise ValueError(
            "KernelIR Concatenate input widths must sum to its output width."
        )
    if any(value.dtype != output.dtype for value in op.inputs):
        raise ValueError("KernelIR Concatenate input/output dtypes must match.")


def _validate_extract_slice(op: KernelOp) -> None:
    if len(op.inputs) != 1 or len(op.outputs) != 1:
        raise ValueError(
            "KernelIR ExtractSlice requires exactly one input and one output."
        )
    start = int(op.attrs.get("start", -1))
    stop = int(op.attrs.get("stop", -1))
    if start < 0 or stop < start or stop > op.inputs[0].width:
        raise ValueError(
            f"KernelIR ExtractSlice has invalid bounds [{start}:{stop}] for "
            f"input width {op.inputs[0].width}."
        )
    if stop - start != op.outputs[0].width:
        raise ValueError(
            "KernelIR ExtractSlice bounds must match its output width."
        )
    if op.inputs[0].dtype != op.outputs[0].dtype:
        raise ValueError("KernelIR ExtractSlice input/output dtypes must match.")


def _validate_for_passes(op: KernelOp) -> None:
    if op.inputs or op.outputs:
        raise ValueError("KernelIR ForPasses cannot have value inputs or outputs.")
    declaration_only = op.attrs.get("declaration_only")
    if type(declaration_only) is not bool:
        raise ValueError(
            "KernelIR ForPasses requires a typed declaration_only flag."
        )
    body = op.attrs.get("body")
    if type(body) is not tuple or any(type(child) is not KernelOp for child in body):
        raise ValueError("KernelIR ForPasses body must be a tuple of KernelOps.")
    if declaration_only:
        return
    if op.attrs.get("trace_kind") != "precomputed":
        raise ValueError(
            "Executable KernelIR ForPasses requires trace_kind='precomputed'."
        )
    if not body or any(child.kind != "ExecuteConsiderationSet" for child in body):
        raise ValueError(
            "Executable KernelIR ForPasses must contain only nonempty "
            "ExecuteConsiderationSet ops."
        )


def _validate_execute_consideration_set(op: KernelOp) -> None:
    if op.inputs or op.outputs:
        raise ValueError(
            "KernelIR ExecuteConsiderationSet cannot have value inputs or outputs."
        )
    pass_index = op.attrs.get("pass_index")
    consideration_set_id = op.attrs.get("consideration_set_id")
    component_ids = op.attrs.get("component_ids")
    body = op.attrs.get("body")
    if type(pass_index) is not int or pass_index < 0:
        raise ValueError(
            "KernelIR ExecuteConsiderationSet pass_index must be a non-negative "
            "non-bool integer."
        )
    if type(consideration_set_id) is not int or consideration_set_id < 0:
        raise ValueError(
            "KernelIR ExecuteConsiderationSet consideration_set_id must be a "
            "non-negative non-bool integer."
        )
    if (
        type(component_ids) is not tuple
        or not component_ids
        or any(type(component_id) is not int or component_id < 0 for component_id in component_ids)
        or component_ids != tuple(sorted(set(component_ids)))
    ):
        raise ValueError(
            "KernelIR ExecuteConsiderationSet component_ids must be a nonempty "
            "tuple of unique, sorted, non-negative non-bool integers."
        )
    if type(body) is not tuple or not body or any(
        type(child) is not KernelOp for child in body
    ):
        raise ValueError(
            "KernelIR ExecuteConsiderationSet body must be a nonempty tuple of "
            "KernelOps."
        )
    if any(
        child.kind in {"ForPasses", "ExecuteConsiderationSet", "StoreOutput", "StoreFlag"}
        for child in body
    ):
        raise ValueError(
            "KernelIR ExecuteConsiderationSet body cannot contain nested schedule "
            "regions or host-buffer stores."
        )


def _validate_step_mechanism(op: KernelOp) -> None:
    if len(op.inputs) != 1 or not op.outputs:
        raise ValueError(
            "KernelIR StepMechanism requires exactly one input and at least one "
            "output."
        )
    component_id = op.attrs.get("component_id")
    execution_index = op.attrs.get("execution_index")
    state_ids = op.attrs.get("state_ids")
    if type(component_id) is not int or component_id < 0:
        raise ValueError(
            "KernelIR StepMechanism component_id must be a non-negative "
            "non-bool integer."
        )
    if type(execution_index) is not int or execution_index < 0:
        raise ValueError(
            "KernelIR StepMechanism execution_index must be a non-negative "
            "non-bool integer."
        )
    if (
        type(state_ids) is not tuple
        or not state_ids
        or any(type(state_id) is not int or state_id < 0 for state_id in state_ids)
        or state_ids != tuple(sorted(set(state_ids)))
    ):
        raise ValueError(
            "KernelIR StepMechanism state_ids must be a nonempty tuple of "
            "unique, sorted, non-negative non-bool integers."
        )
    if op.attrs.get("active_lanes") != "all":
        raise ValueError(
            "KernelIR precomputed StepMechanism currently requires "
            "active_lanes='all'."
        )
    if not isinstance(op.attrs.get("spec_key"), str) or not op.attrs["spec_key"]:
        raise ValueError("KernelIR StepMechanism requires a nonempty spec_key.")


def _validate_reset_state(op: KernelOp) -> None:
    if op.inputs or not op.outputs:
        raise ValueError(
            "KernelIR ResetState requires no inputs and at least one state output."
        )
    attrs = op.attrs
    if set(attrs) != {"component_id", "state_ids", "condition_type", "region"}:
        raise ValueError(
            "KernelIR ResetState requires exactly component_id, state_ids, "
            "condition_type, and region attributes."
        )
    component_id = attrs["component_id"]
    state_ids = attrs["state_ids"]
    if type(component_id) is not int or component_id < 0:
        raise ValueError(
            "KernelIR ResetState component_id must be a non-negative "
            "non-bool integer."
        )
    if (
        type(state_ids) is not tuple
        or not state_ids
        or any(type(state_id) is not int or state_id < 0 for state_id in state_ids)
        or state_ids != tuple(sorted(set(state_ids)))
        or len(op.outputs) != len(state_ids)
    ):
        raise ValueError(
            "KernelIR ResetState state_ids must be a nonempty tuple of unique, "
            "sorted, non-negative integers matching its state outputs."
        )
    if attrs["condition_type"] != "AtTrialStart" or attrs["region"] != "trial":
        raise ValueError(
            "KernelIR ResetState requires condition_type='AtTrialStart' and "
            "region='trial'."
        )
@dataclass(frozen=True)
class KernelIR:
    """Backend-neutral batched execution plan.

    Triton and the CPU debug executor both consume this IR.  A future MLIR
    backend should lower from this representation rather than re-discovering
    semantics from generated Triton source. ``op_specs`` is an immutable
    lowering-environment sidecar: it fixes the registered implementations used
    by this plan without putting implementation objects in individual op attrs.
    """

    model_kind: str
    fusion_kind: str | None
    lane_layout: KernelLaneLayout
    inputs: tuple[BatchedInputSpec, ...]
    params: tuple[BatchedParamSpec, ...]
    states: tuple[BatchedStateSpec, ...]
    outputs: tuple[BatchedOutputSpec, ...]
    rng_streams: tuple[KernelRngStream, ...]
    ops: tuple[KernelOp, ...]
    output_names: tuple[str, ...]
    max_steps: int
    graph: BatchedGraphIR
    op_specs: BatchedOpSpecSnapshot
    executable: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    ports: tuple[BatchedPortSpec, ...] = ()
    absorbed_projections: tuple[BatchedAbsorbedProjectionSpec, ...] = ()
    scheduler: tuple[BatchedSchedulerSpec, ...] = ()
    schedule_regions: tuple[BatchedScheduleRegionSpec, ...] = ()
    consideration_sets: tuple[BatchedConsiderationSetSpec, ...] = ()
    finished_values: tuple[BatchedFinishedValueSpec, ...] = ()
    effective_parameters: tuple[BatchedEffectiveParameterSpec, ...] = ()
    modulations: tuple[BatchedModulationSpec, ...] = ()
    resets: tuple[BatchedResetSpec, ...] = ()
    termination: tuple[BatchedTerminationSpec, ...] = ()
    schedule_trace: BatchedScheduleTraceSpec | None = None

    def __post_init__(self) -> None:
        validate_kernel_ir(self)


def validate_kernel_ir(kernel: KernelIR) -> None:
    """Validate cross-op identity and effect invariants in a complete KernelIR."""

    # Validate identity-bearing declaration sequences before constructing any
    # lookup dictionaries from them.  Otherwise duplicate or bool-valued IDs
    # can be silently collapsed by Python's mapping-key equality.
    _validate_kernel_parameters(kernel)
    _validate_kernel_ports(kernel)
    _validate_kernel_finished_scheduler(kernel)

    semantic_fields = (
        ("input", kernel.inputs, kernel.graph.inputs),
        ("retained-state", kernel.states, kernel.graph.states),
        ("output", kernel.outputs, kernel.graph.outputs),
        ("port", kernel.ports, kernel.graph.ports),
        (
            "absorbed-projection",
            kernel.absorbed_projections,
            kernel.graph.absorbed_projections,
        ),
        ("scheduler", kernel.scheduler, kernel.graph.scheduler),
        ("schedule-region", kernel.schedule_regions, kernel.graph.schedule_regions),
        ("consideration-set", kernel.consideration_sets, kernel.graph.consideration_sets),
        ("finished-value", kernel.finished_values, kernel.graph.finished_values),
        (
            "effective-parameter",
            kernel.effective_parameters,
            kernel.graph.effective_parameters,
        ),
        ("modulation", kernel.modulations, kernel.graph.modulations),
        ("reset", kernel.resets, kernel.graph.resets),
        ("termination", kernel.termination, kernel.graph.termination),
    )
    for label, kernel_declarations, graph_declarations in semantic_fields:
        if kernel_declarations != graph_declarations:
            raise ValueError(
                f"KernelIR {label} declarations must exactly match GraphIR."
            )
    _validate_kernel_modulations(kernel)
    _validate_kernel_reset_declarations(kernel)
    step_counts: dict[int, int] = {}
    precomputed_regions: list[KernelOp] = []
    output_stores: list[KernelOp] = []
    reset_ops: list[KernelOp] = []

    def visit(op: KernelOp, *, in_precomputed_region: bool = False) -> None:
        if op.kind == "StepMechanism":
            if not in_precomputed_region:
                raise ValueError(
                    "KernelIR StepMechanism must belong to an executable "
                    "precomputed ForPasses region."
                )
            try:
                node = kernel.graph.node(op.target)
            except KeyError as error:
                raise ValueError(
                    f"KernelIR StepMechanism target '{op.target}' is not declared."
                ) from error
            component_id = _component_id(kernel.graph, node)
            expected_state_ids = tuple(
                sorted(
                    state.state_id
                    for state in kernel.states
                    if _state_component_id(kernel.graph, state) == component_id
                )
            )
            if op.attrs["component_id"] != component_id:
                raise ValueError(
                    f"KernelIR StepMechanism target '{op.target}' has component "
                    f"id {op.attrs['component_id']}, expected {component_id}."
                )
            if op.attrs["state_ids"] != expected_state_ids:
                raise ValueError(
                    f"KernelIR StepMechanism target '{op.target}' has state IDs "
                    f"{op.attrs['state_ids']}, expected {expected_state_ids}."
                )
            if op.attrs["spec_key"] != node.attrs.get("spec_key"):
                raise ValueError(
                    f"KernelIR StepMechanism target '{op.target}' does not use "
                    "its frozen graph implementation key."
                )
            expected_execution_index = step_counts.get(component_id, 0)
            if op.attrs["execution_index"] != expected_execution_index:
                raise ValueError(
                    f"KernelIR StepMechanism target '{op.target}' has execution "
                    f"index {op.attrs['execution_index']}, expected "
                    f"{expected_execution_index}."
                )
            step_counts[component_id] = expected_execution_index + 1
        elif op.kind == "StoreOutput":
            output_stores.append(op)
        elif op.kind == "ResetState":
            # attrs is a Mapping and may have been mutated after construction;
            # source emission re-runs this complete-IR boundary.
            _validate_reset_state(op)
            reset_ops.append(op)

        child_precomputed = in_precomputed_region
        if op.kind == "ForPasses":
            child_precomputed = (
                op.attrs.get("declaration_only") is False
                and op.attrs.get("trace_kind") == "precomputed"
            )
            if child_precomputed:
                precomputed_regions.append(op)
        for child in op.attrs.get("body", ()):
            visit(child, in_precomputed_region=child_precomputed)

    for op in kernel.ops:
        visit(op)

    if kernel.schedule_trace is not None:
        # This first executable schedule tier has a compiler-owned epilogue.
        # Enforce it here so a hand-forged scheduled body cannot add or replace
        # a host-buffer write.  Ordinary KernelIR remains extensible: custom
        # lowering may legitimately feed a declared output from a different
        # SSA value (for example after AddConstant/Clamp transforms).
        if len(output_stores) != len(kernel.graph.outputs):
            raise ValueError(
                "KernelIR precomputed schedule must contain exactly one "
                "StoreOutput for each declared GraphIR output."
            )
        for store, output in zip(output_stores, kernel.graph.outputs):
            expected_input = KernelValue(
                node_output_value_name(kernel.graph, output.node, output.port),
                output.width,
            )
            expected_attrs = {
                "node": output.node,
                "port": output.port,
                "width": output.width,
                "component_id": output.component_id,
                "port_id": output.port_id,
                "flat_start": output.flat_start,
                "flat_stop": output.flat_stop,
            }
            if (
                store.target != output.name
                or store.inputs != (expected_input,)
                or store.outputs
                or dict(store.attrs) != expected_attrs
            ):
                raise ValueError(
                    "KernelIR StoreOutput does not exactly match its declared "
                    f"GraphIR output '{output.name}'."
                )

        if len(precomputed_regions) != 1:
            raise ValueError(
                "KernelIR schedule_trace requires exactly one executable "
                "precomputed ForPasses region."
            )
        region_steps = tuple(
            (
                child.attrs.get("pass_index"),
                child.attrs.get("consideration_set_id"),
                child.attrs.get("component_ids"),
            )
            for child in precomputed_regions[0].attrs["body"]
        )
        declared_steps = tuple(
            (
                step.pass_index,
                step.consideration_set_id,
                step.component_ids,
            )
            for step in kernel.schedule_trace.steps
        )
        if region_steps != declared_steps:
            raise ValueError(
                "KernelIR executable precomputed region does not match its "
                "typed schedule_trace."
            )
        for execution in precomputed_regions[0].attrs["body"]:
            declared_component_ids = set(execution.attrs["component_ids"])
            body_component_ids = set()
            for child in execution.attrs["body"]:
                try:
                    child_component_id = _component_id(
                        kernel.graph,
                        kernel.graph.node(child.target),
                    )
                except KeyError as error:
                    raise ValueError(
                        "KernelIR executable consideration-set body references "
                        f"undeclared target '{child.target}'."
                    ) from error
                if child_component_id not in declared_component_ids:
                    raise ValueError(
                        "KernelIR executable consideration-set body target "
                        f"'{child.target}' is not one of its declared component "
                        "IDs."
                    )
                body_component_ids.add(child_component_id)
            if body_component_ids != declared_component_ids:
                raise ValueError(
                    "KernelIR executable consideration-set body does not cover "
                    "exactly its declared component IDs."
                )
        declared_execution_count = sum(
            len(step.component_ids) for step in kernel.schedule_trace.steps
        )
        if (
            kernel.schedule_trace.component_execution_count
            != declared_execution_count
        ):
            raise ValueError(
                "KernelIR schedule_trace component_execution_count does not "
                "match its declared steps."
            )
    elif precomputed_regions:
        raise ValueError(
            "KernelIR executable precomputed ForPasses region requires a typed "
            "schedule_trace."
        )

    expected_reset_declarations = tuple(
        reset
        for reset in kernel.resets
        if reset.condition_type == "AtTrialStart"
    )
    stateful_layout = kernel.lane_layout.kind == STATEFUL_LANE_LAYOUT
    trial_regions = tuple(op for op in kernel.ops if op.kind == "ForTrials")
    if (
        stateful_layout
        and (
            kernel.schedule_trace is not None
            or expected_reset_declarations
            or reset_ops
        )
        and len(trial_regions) != 1
    ):
        raise ValueError(
            "KernelIR stateful scheduled/reset effects require exactly one "
            "ForTrials region."
        )
    if stateful_layout and (expected_reset_declarations or reset_ops):
        trial_body = tuple(trial_regions[0].attrs.get("body", ()))
        reset_prefix = tuple(
            op for op in trial_body[:len(reset_ops)] if op.kind == "ResetState"
        )
        if tuple(reset_ops) != reset_prefix:
            raise ValueError(
                "KernelIR ResetState operations must form an unconditional "
                "prefix of the ForTrials body."
            )
        if len(reset_prefix) != len(expected_reset_declarations):
            raise ValueError(
                "KernelIR must contain exactly one ResetState for each "
                "AtTrialStart reset declaration."
            )
        state_values = {
            state.state_id: value
            for state, value in zip(kernel.states, _state_kernel_values(kernel.graph))
        }
        states_by_id = {state.state_id: state for state in kernel.states}
        for op, reset in zip(reset_prefix, expected_reset_declarations):
            try:
                declared_states = tuple(
                    states_by_id[state_id]
                    for state_id in reset.state_ids
                )
                expected_outputs = tuple(
                    state_values[state_id]
                    for state_id in reset.state_ids
                )
            except KeyError as error:
                raise ValueError(
                    f"KernelIR reset for '{reset.node}' references an "
                    "undeclared state ID."
                ) from error
            if (
                reset.attrs
                or reset.region != "trial"
                or any(state.component_id != reset.component_id for state in declared_states)
                or op.target != reset.node
                or op.outputs != expected_outputs
                or dict(op.attrs) != {
                    "component_id": reset.component_id,
                    "state_ids": reset.state_ids,
                    "condition_type": "AtTrialStart",
                    "region": "trial",
                }
            ):
                raise ValueError(
                    "KernelIR ResetState does not exactly match its declared "
                    f"reset for '{reset.node}'."
                )
    elif reset_ops or expected_reset_declarations:
        raise ValueError(
            "KernelIR AtTrialStart reset effects require a stateful lane "
            "layout."
        )

    counted_finished = {
        value.component_id: value.attrs.get("count")
        for value in kernel.finished_values
        if value.predicate_kind == "execution_count_at_least"
    }
    if not step_counts:
        if kernel.executable and counted_finished:
            raise ValueError(
                "Executable KernelIR counted finished values require typed "
                "StepMechanism operations."
            )
        return
    if set(step_counts) != set(counted_finished):
        raise ValueError(
            "KernelIR StepMechanism owners must exactly match counted finished "
            "value owners."
        )
    trace_execution_counts: dict[int, int] = {}
    for step in kernel.schedule_trace.steps:
        for component_id in step.component_ids:
            trace_execution_counts[component_id] = (
                trace_execution_counts.get(component_id, 0) + 1
            )
    for component_id, count in counted_finished.items():
        if step_counts[component_id] != trace_execution_counts.get(component_id, 0):
            raise ValueError(
                "KernelIR counted finished component id "
                f"{component_id} StepMechanism count does not match its typed "
                "schedule trace."
            )
        if type(count) is not int or count <= 0 or step_counts[component_id] < count:
            raise ValueError(
                "KernelIR counted finished component id "
                f"{component_id} requires at least {count!r} scheduled steps, "
                f"found {step_counts[component_id]}."
            )


def _nodes_resolving_component_id(
    graph: BatchedGraphIR,
    component_id: int,
) -> tuple:
    """Return graph nodes resolving to one lowering-local component identity."""

    return tuple(
        node
        for node in graph.nodes
        if _component_id(graph, node) == component_id
    )


def _consideration_set_owns(
    consideration_set: BatchedConsiderationSetSpec,
    node: str,
    component_id: int,
) -> bool:
    """Whether one typed consideration set owns an exact node/ID pair."""

    return bool(
        type(consideration_set.consideration_set_id) is int
        and type(consideration_set.nodes) is tuple
        and type(consideration_set.component_ids) is tuple
        and len(consideration_set.nodes) == len(consideration_set.component_ids)
        and any(
            type(member_name) is str
            and type(member_id) is int
            and member_name == node
            and member_id == component_id
            for member_name, member_id in zip(
                consideration_set.nodes,
                consideration_set.component_ids,
            )
        )
    )


def _validate_kernel_parameters(kernel: KernelIR) -> None:
    """Validate the global parameter inventory without lossy ID lookups."""

    parameter_ids = tuple(parameter.parameter_id for parameter in kernel.params)
    if any(type(parameter_id) is not int for parameter_id in parameter_ids):
        raise ValueError(
            "KernelIR parameter IDs must be exact non-bool integers."
        )
    if parameter_ids != tuple(range(len(kernel.params))):
        raise ValueError(
            "KernelIR parameter IDs must be unique, contiguous, and in "
            "declaration order."
        )

    parameter_names = tuple(parameter.name for parameter in kernel.params)
    if any(type(name) is not str or not name for name in parameter_names):
        raise ValueError(
            "KernelIR parameter canonical names must be nonempty strings."
        )
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError(
            "KernelIR parameter canonical names must be unique."
        )

    for parameter in kernel.params:
        if type(parameter.owner_component_id) is not int:
            raise ValueError(
                "KernelIR parameter owner component IDs must be exact non-bool "
                "integers."
            )
        owners = _nodes_resolving_component_id(
            kernel.graph,
            parameter.owner_component_id,
        )
        if len(owners) != 1:
            raise ValueError(
                f"KernelIR parameter '{parameter.name}' owner component id "
                f"{parameter.owner_component_id} must resolve to exactly one "
                "GraphIR node."
            )
        if (
            type(parameter.owner_scope) is not str
            or parameter.owner_scope not in {"function", "mechanism"}
        ):
            raise ValueError(
                f"KernelIR parameter '{parameter.name}' owner_scope must be "
                "'function' or 'mechanism'."
            )


def _validate_kernel_ports(kernel: KernelIR) -> None:
    """Validate the complete typed port inventory for every KernelIR shape."""

    port_ids = tuple(port.port_id for port in kernel.ports)
    if any(type(port_id) is not int for port_id in port_ids):
        raise ValueError("KernelIR port IDs must be exact non-bool integers.")
    if port_ids != tuple(range(len(kernel.ports))):
        raise ValueError(
            "KernelIR port IDs must be unique, contiguous, and in declaration "
            "order."
        )

    identities = []
    supported_kinds = {"InputPort", "OutputPort", "ParameterPort", "ControlSignal"}
    for port in kernel.ports:
        if (
            type(port.name) is not str
            or not port.name
            or type(port.owner) is not str
            or not port.owner
            or type(port.owner_component_id) is not int
            or type(port.kind) is not str
            or port.kind not in supported_kinds
            or type(port.width) is not int
            or port.width <= 0
        ):
            raise ValueError(
                "KernelIR ports require nonempty labels, exact owner IDs, a "
                "supported port kind, and a positive non-bool width."
            )
        owners = tuple(
            node
            for node in _nodes_resolving_component_id(
                kernel.graph,
                port.owner_component_id,
            )
            if node.name == port.owner
        )
        if len(owners) != 1:
            raise ValueError(
                f"KernelIR port '{port.name}' owner name and component id must "
                "resolve to exactly one GraphIR node."
            )
        identities.append((port.owner_component_id, port.kind, port.name))
    if len(set(identities)) != len(identities):
        raise ValueError(
            "KernelIR ports must have unique (owner component, kind, name) "
            "identities."
        )

    ports_by_id = {port.port_id: port for port in kernel.ports}
    for node in kernel.graph.nodes:
        component_id = _component_id(kernel.graph, node)
        owned_ports = tuple(
            port
            for port in kernel.ports
            if port.owner_component_id == component_id
        )
        anchors_present = bool(
            node.input_port_ids
            or node.output_port_ids
            or node.parameter_port_ids
        )
        # Hand-built declaration-only legacy fixtures may omit the global port
        # inventory entirely.  Once either side declares ports, require an
        # exact component-local anchor rather than trusting free-form labels.
        if not owned_ports and not anchors_present:
            continue
        expected_input_ids = tuple(
            port.port_id for port in owned_ports if port.kind == "InputPort"
        )
        expected_output_ids = tuple(
            port.port_id
            for port in owned_ports
            if port.kind in {"OutputPort", "ControlSignal"}
        )
        expected_parameter_ids = tuple(
            (port.name, port.port_id)
            for port in owned_ports
            if port.kind == "ParameterPort"
        )
        if (
            type(node.input_port_ids) is not tuple
            or type(node.output_port_ids) is not tuple
            or type(node.parameter_port_ids) is not tuple
            or node.input_port_ids != expected_input_ids
            or node.output_port_ids != expected_output_ids
            or node.parameter_port_ids != expected_parameter_ids
            or any(port_id not in ports_by_id for port_id in node.input_port_ids)
            or any(port_id not in ports_by_id for port_id in node.output_port_ids)
        ):
            raise ValueError(
                f"KernelIR node '{node.name}' port anchors must exactly match "
                "its ordered typed port inventory."
            )


def _validate_kernel_finished_scheduler(kernel: KernelIR) -> None:
    """Validate scheduler-visible finished values and their typed references."""

    if kernel.scheduler:
        scheduler_component_ids = tuple(
            condition.component_id for condition in kernel.scheduler
        )
        graph_component_ids = tuple(
            _component_id(kernel.graph, node) for node in kernel.graph.nodes
        )
        if (
            len(set(scheduler_component_ids)) != len(scheduler_component_ids)
            or set(scheduler_component_ids) != set(graph_component_ids)
        ):
            raise ValueError(
                "KernelIR scheduler declarations must contain exactly one "
                "condition for every GraphIR node."
            )

    value_ids = tuple(value.value_id for value in kernel.finished_values)
    if any(type(value_id) is not int for value_id in value_ids):
        raise ValueError(
            "KernelIR finished-value IDs must be exact non-bool integers."
        )
    if value_ids != tuple(range(len(kernel.finished_values))):
        raise ValueError(
            "KernelIR finished-value IDs must be unique, contiguous, and in "
            "declaration order."
        )

    finished_components = []
    for value in kernel.finished_values:
        if (
            type(value.component_id) is not int
            or type(value.producer_consideration_set_id) is not int
            or type(value.name) is not str
            or type(value.node) is not str
            or value.name != f"{value.node}.is_finished"
            or value.width != 1
            or type(value.width) is not int
            or value.dtype != "bool"
            or value.storage != "combinational"
        ):
            raise ValueError(
                "KernelIR finished values require exact node/component/set "
                "identity and scalar bool combinational storage."
            )
        owners = tuple(
            node
            for node in _nodes_resolving_component_id(kernel.graph, value.component_id)
            if node.name == value.node
        )
        producer_sets = tuple(
            consideration_set
            for consideration_set in kernel.consideration_sets
            if consideration_set.consideration_set_id
            == value.producer_consideration_set_id
            and _consideration_set_owns(
                consideration_set,
                value.node,
                value.component_id,
            )
        )
        if len(owners) != 1 or len(producer_sets) != 1:
            raise ValueError(
                f"KernelIR finished value '{value.name}' must resolve to its "
                "exact GraphIR node and producer consideration set."
            )
        finished_components.append(value.component_id)
    if len(set(finished_components)) != len(finished_components):
        raise ValueError(
            "KernelIR finished values must declare at most one value per "
            "producer component."
        )

    referenced_value_ids = set()
    for condition in kernel.scheduler:
        if (
            type(condition.component_id) is not int
            or type(condition.consideration_set_id) is not int
        ):
            raise ValueError(
                "KernelIR scheduler component and consideration-set IDs must be "
                "exact non-bool integers."
            )
        targets = tuple(
            node
            for node in _nodes_resolving_component_id(
                kernel.graph,
                condition.component_id,
            )
            if node.name == condition.node
        )
        target_sets = tuple(
            consideration_set
            for consideration_set in kernel.consideration_sets
            if consideration_set.consideration_set_id
            == condition.consideration_set_id
            and _consideration_set_owns(
                consideration_set,
                condition.node,
                condition.component_id,
            )
        )
        if len(targets) != 1 or len(target_sets) != 1:
            raise ValueError(
                f"KernelIR scheduler condition for '{condition.node}' must "
                "resolve to its exact GraphIR node and consideration set."
            )
        if (
            type(condition.dependencies) is not tuple
            or type(condition.dependency_component_ids) is not tuple
            or len(condition.dependencies) != len(condition.dependency_component_ids)
            or any(
                type(component_id) is not int
                for component_id in condition.dependency_component_ids
            )
        ):
            raise ValueError(
                "KernelIR scheduler dependencies require parallel name and exact "
                "non-bool component-ID tuples."
            )
        for dependency, component_id in zip(
            condition.dependencies,
            condition.dependency_component_ids,
        ):
            matches = tuple(
                node
                for node in _nodes_resolving_component_id(kernel.graph, component_id)
                if node.name == dependency
            )
            if len(matches) != 1:
                raise ValueError(
                    f"KernelIR scheduler dependency '{dependency}' does not "
                    "match component id {component_id}."
                )

        if type(condition.finished_value_ids) is not tuple:
            raise ValueError(
                "KernelIR scheduler finished-value IDs must be a typed tuple."
            )
        if condition.condition_type != "WhenFinished":
            if condition.finished_value_ids:
                raise ValueError(
                    "KernelIR finished-value IDs are valid only on WhenFinished "
                    "scheduler conditions."
                )
            continue
        if any(
            type(value_id) is not int
            for value_id in condition.finished_value_ids
        ):
            raise ValueError(
                "KernelIR WhenFinished finished-value IDs must be exact "
                "non-bool integers."
            )
        if (
            not condition.dependencies
            or len(condition.finished_value_ids) != len(condition.dependencies)
            or len(set(condition.finished_value_ids))
            != len(condition.finished_value_ids)
        ):
            raise ValueError(
                "KernelIR WhenFinished dependencies and finished-value IDs must "
                "form parallel unique tuples."
            )
        for component_id, finished_value_id in zip(
            condition.dependency_component_ids,
            condition.finished_value_ids,
        ):
            matches = tuple(
                value
                for value in kernel.finished_values
                if value.value_id == finished_value_id
                and value.component_id == component_id
            )
            if len(matches) != 1:
                raise ValueError(
                    "KernelIR WhenFinished reference does not match its declared "
                    "dependency finished value."
                )
            referenced_value_ids.add(finished_value_id)

    if referenced_value_ids != set(value_ids):
        raise ValueError(
            "KernelIR WhenFinished references and finished-value declarations "
            "must form an exact component-wise bijection."
        )


def _validate_kernel_reset_declarations(kernel: KernelIR) -> None:
    """Require a canonical reset declaration for every retained-state owner."""

    if kernel.lane_layout.kind != STATEFUL_LANE_LAYOUT:
        if kernel.resets:
            raise ValueError(
                "KernelIR stateless lane layouts cannot retain reset "
                "declarations."
            )
        return

    states_by_component: dict[int, list[BatchedStateSpec]] = {}
    for state in kernel.states:
        component_id = _state_component_id(kernel.graph, state)
        states_by_component.setdefault(component_id, []).append(state)

    component_ids = tuple(states_by_component)
    reset_component_ids = tuple(reset.component_id for reset in kernel.resets)
    if reset_component_ids != component_ids:
        raise ValueError(
            "KernelIR stateful reset declarations must contain exactly one "
            "entry for every retained-state owner, in state declaration order."
        )

    for reset in kernel.resets:
        try:
            node = next(
                node
                for node in kernel.graph.nodes
                if _component_id(kernel.graph, node) == reset.component_id
            )
        except StopIteration as error:
            raise ValueError(
                "KernelIR reset declaration references an undeclared component "
                f"id {reset.component_id}."
            ) from error
        expected_state_ids = tuple(
            state.state_id
            for state in states_by_component[reset.component_id]
        )
        if (
            reset.node != node.name
            or reset.condition_type not in {"Never", "AtTrialStart"}
            or reset.state_ids != expected_state_ids
            or reset.attrs
            or reset.region != "trial"
        ):
            raise ValueError(
                "KernelIR reset declaration does not exactly match retained "
                f"state owner '{node.name}'."
            )


def _validate_kernel_modulations(kernel: KernelIR) -> None:
    """Validate the declaration-only scalar ``OVERRIDE`` identity boundary."""

    typed_controller_ids = {
        _component_id(kernel.graph, node)
        for node in kernel.graph.nodes
        if node.component_type == "ControlMechanism"
        and node.attrs.get("control_function") in {"identity", "registered"}
    }
    modulation_controller_ids = {
        modulation.controller_component_id for modulation in kernel.modulations
    }
    if typed_controller_ids != modulation_controller_ids:
        raise ValueError(
            "KernelIR typed ControlMechanism declarations and modulation "
            "effects must form an exact controller-component bijection."
        )

    dynamic_finished = tuple(
        value
        for value in kernel.finished_values
        if value.predicate_kind
        == "execution_count_at_least_effective_parameter"
    )
    if (
        not kernel.modulations
        and not kernel.effective_parameters
        and not dynamic_finished
        and not kernel.absorbed_projections
    ):
        return
    if kernel.graph.executable or kernel.executable:
        raise ValueError(
            "KernelIR effective-parameter modulation is declaration-only until "
            "ApplyModulation and lane-local scheduler effects are lowered."
        )

    modulation_ids = tuple(
        modulation.modulation_id for modulation in kernel.modulations
    )
    effective_parameter_ids = tuple(
        modulation.effective_parameter_id for modulation in kernel.modulations
    )
    declared_effective_ids = tuple(
        parameter.effective_parameter_id
        for parameter in kernel.effective_parameters
    )
    if modulation_ids != tuple(range(len(kernel.modulations))):
        raise ValueError(
            "KernelIR modulation IDs must be contiguous and in declaration order."
        )
    if len(set(effective_parameter_ids)) != len(effective_parameter_ids):
        raise ValueError(
            "KernelIR effective-parameter modulation IDs must be unique."
        )
    if (
        declared_effective_ids != tuple(range(len(kernel.effective_parameters)))
        or declared_effective_ids != effective_parameter_ids
        or len(dynamic_finished) != len(kernel.modulations)
        or len(kernel.absorbed_projections) != 2 * len(kernel.modulations)
    ):
        raise ValueError(
            "KernelIR modulation, held effective-parameter, and dynamic "
            "finished declarations must form an exact bijection."
        )

    ports_by_id = {port.port_id: port for port in kernel.ports}
    absorbed_by_id = {
        projection.projection_id: projection
        for projection in kernel.absorbed_projections
    }
    if tuple(sorted(absorbed_by_id)) != tuple(
        range(len(kernel.absorbed_projections))
    ):
        raise ValueError(
            "KernelIR absorbed-projection declarations must have unique "
            "contiguous IDs."
        )
    for projection in kernel.absorbed_projections:
        try:
            sender = kernel.graph.node(projection.sender)
            receiver = kernel.graph.node(projection.receiver)
        except KeyError as error:
            raise ValueError(
                "KernelIR absorbed projection references an undeclared component."
            ) from error
        sender_port = ports_by_id.get(projection.sender_port_id)
        receiver_port = ports_by_id.get(projection.receiver_port_id)
        if (
            _component_id(kernel.graph, sender) != projection.sender_component_id
            or _component_id(kernel.graph, receiver)
            != projection.receiver_component_id
            or sender_port is None
            or sender_port.owner_component_id != projection.sender_component_id
            or sender_port.name != projection.sender_port
            or sender_port.width != projection.width
            or receiver_port is None
            or receiver_port.owner_component_id
            != projection.receiver_component_id
            or receiver_port.name != projection.receiver_port
            or receiver_port.width != projection.width
        ):
            raise ValueError(
                "KernelIR absorbed projection endpoint identity does not match "
                "its typed port ownership."
            )

    params_by_id = {parameter.parameter_id: parameter for parameter in kernel.params}
    params_by_name = {parameter.name: parameter for parameter in kernel.params}
    scheduler_by_component_id = {
        condition.component_id: condition for condition in kernel.scheduler
    }
    effective_by_id = {
        parameter.effective_parameter_id: parameter
        for parameter in kernel.effective_parameters
    }
    target_port_ids = set()
    referenced_absorbed_projection_ids = []
    for modulation in kernel.modulations:
        try:
            controller = kernel.graph.node(modulation.controller)
            source = kernel.graph.node(modulation.source)
            target = kernel.graph.node(modulation.target)
        except KeyError as error:
            raise ValueError(
                "KernelIR modulation references an undeclared component."
            ) from error
        if (
            _component_id(kernel.graph, controller)
            != modulation.controller_component_id
            or _component_id(kernel.graph, source)
            != modulation.source_component_id
            or _component_id(kernel.graph, target)
            != modulation.target_component_id
            or controller.component_type != "ControlMechanism"
            or source.component_type != "TransferMechanism"
            or target.component_type != "LCAMechanism"
            or dict(controller.params)
            != {
                binding.argument: binding.parameter
                for binding in modulation.controller_param_bindings
            }
            or controller.attrs.get("spec_key", "")
            != modulation.controller_function_spec_key
        ):
            raise ValueError(
                "KernelIR modulation component identity or controller binding "
                "does not match GraphIR."
            )

        source_condition = scheduler_by_component_id.get(
            modulation.source_component_id
        )
        controller_condition = scheduler_by_component_id.get(
            modulation.controller_component_id
        )
        target_condition = scheduler_by_component_id.get(
            modulation.target_component_id
        )
        if (
            source_condition is None
            or controller_condition is None
            or target_condition is None
            or not (
                source_condition.consideration_set_id
                < controller_condition.consideration_set_id
                < target_condition.consideration_set_id
            )
            or target_condition.condition_type != "Always"
        ):
            raise ValueError(
                "KernelIR modulation requires strictly ordered source, "
                "controller, and Always-target consideration sets."
            )

        source_inputs = tuple(
            input_spec
            for input_spec in kernel.inputs
            if input_spec.component_id == modulation.source_component_id
        )
        source_spec_key = source.attrs.get("spec_key", "")
        try:
            source_implementation = kernel.op_specs.lookup_spec(source_spec_key)
        except BatchedOpSpecError as error:
            raise ValueError(
                "KernelIR modulation source must use its frozen registered "
                "identity Linear implementation."
            ) from error
        expected_source_defaults = {
            "slope": 1.0,
            "intercept": 0.0,
            "scale": 1.0,
            "offset": 0.0,
        }
        source_parameters = {
            argument: params_by_name.get(parameter_name)
            for argument, parameter_name in dict(source.params).items()
        }
        if (
            source.function_type != "Linear"
            or source.attrs.get("spec_kind") != "elementwise"
            or source.attrs.get("noise") is not None
            or source.attrs.get("clip") is not None
            or source.attrs.get("integrator_pre") is not None
            or source.input_width != 1
            or source.output_width != 1
            or len(source.input_port_ids) != 1
            or not source.output_port_ids
            or source.output_port_ids[0] != modulation.source_port_id
            or len(source_inputs) != 1
            or source_inputs[0].port_id != source.input_port_ids[0]
            or source_inputs[0].width != 1
            or any(
                projection.receiver_component_id
                == modulation.source_component_id
                for projection in kernel.graph.projections
            )
            or not isinstance(source_implementation, ElementwiseFunctionSpec)
            or source_implementation.function_class.__name__ != "Linear"
            or set(source_parameters) != set(expected_source_defaults)
            or any(parameter is None for parameter in source_parameters.values())
            or any(
                source_parameters[argument].default != expected_default
                or source_parameters[argument].runtime_mutable
                or source_parameters[argument].owner_component_id
                != modulation.source_component_id
                for argument, expected_default in expected_source_defaults.items()
            )
        ):
            raise ValueError(
                "KernelIR controlled-finished source must be the frozen scalar "
                "identity Linear origin declared by GraphIR."
            )
        endpoint_expectations = (
            (
                modulation.source_port_id,
                modulation.source_component_id,
                modulation.source_port,
                "OutputPort",
            ),
            (
                modulation.controller_input_port_id,
                modulation.controller_component_id,
                modulation.controller_input_port,
                "InputPort",
            ),
            (
                modulation.control_signal_port_id,
                modulation.controller_component_id,
                modulation.control_signal_port,
                "ControlSignal",
            ),
            (
                modulation.target_parameter_port_id,
                modulation.target_component_id,
                modulation.target_parameter,
                "ParameterPort",
            ),
        )
        for port_id, owner_id, port_name, port_kind in endpoint_expectations:
            port = ports_by_id.get(port_id)
            if (
                port is None
                or port.owner_component_id != owner_id
                or port.name != port_name
                or port.kind != port_kind
                or port.width != 1
            ):
                raise ValueError(
                    "KernelIR modulation endpoint port identity does not match "
                    "its declared owner, role, name, and width."
                )
        if modulation.target_parameter != "termination_threshold":
            raise ValueError(
                "KernelIR controlled finished modulation must target the LCA "
                "termination_threshold ParameterPort."
            )
        target_parameter_ports = dict(target.parameter_port_ids)
        if (
            target_parameter_ports.get("termination_threshold")
            != modulation.target_parameter_port_id
        ):
            raise ValueError(
                "KernelIR controlled-finished target must use the LCA node's "
                "canonical termination_threshold ParameterPort identity."
            )
        monitor_projection = absorbed_by_id.get(
            modulation.monitor_projection_id
        )
        control_projection = absorbed_by_id.get(
            modulation.control_projection_id
        )
        referenced_absorbed_projection_ids.extend(
            (
                modulation.monitor_projection_id,
                modulation.control_projection_id,
            )
        )
        if (
            monitor_projection is None
            or monitor_projection.kind != "MappingProjection"
            or monitor_projection.sender_component_id
            != modulation.source_component_id
            or monitor_projection.sender_port_id != modulation.source_port_id
            or monitor_projection.receiver_component_id
            != modulation.controller_component_id
            or monitor_projection.receiver_port_id
            != modulation.controller_input_port_id
            or control_projection is None
            or control_projection.kind != "ControlProjection"
            or control_projection.sender_component_id
            != modulation.controller_component_id
            or control_projection.sender_port_id
            != modulation.control_signal_port_id
            or control_projection.receiver_component_id
            != modulation.target_component_id
            or control_projection.receiver_port_id
            != modulation.target_parameter_port_id
        ):
            raise ValueError(
                "KernelIR modulation must exactly reference its typed absorbed "
                "monitor and ControlProjection routes."
            )
        parameter_bindings = modulation.controller_param_bindings
        implementation_bindings = {}
        if modulation.controller_function_spec_key:
            implementation = kernel.op_specs.lookup_spec(
                modulation.controller_function_spec_key
            )
            if (
                not isinstance(implementation, ElementwiseFunctionSpec)
                or controller.function_type
                != implementation.function_class.__name__
                or controller.attrs.get("spec_kind") != "control"
                or controller.attrs.get("control_function") != "registered"
            ):
                raise ValueError(
                    "KernelIR registered controller function identity must "
                    "match its frozen elementwise implementation."
                )
            implementation_bindings = {
                binding.arg: binding for binding in implementation.params
            }
            expected_arguments = tuple(
                binding.arg for binding in implementation.params
            )
            if tuple(binding.argument for binding in parameter_bindings) != (
                expected_arguments
            ):
                raise ValueError(
                    "KernelIR modulation controller bindings do not match its "
                    "frozen registered implementation signature."
                )
        elif (
            parameter_bindings
            or controller.function_type != "Identity"
            or controller.params
            or controller.attrs.get("spec_kind") != "control"
            or controller.attrs.get("control_function") != "identity"
            or controller.attrs.get("spec_key", "")
        ):
            raise ValueError(
                "KernelIR identity controller requires an exact Identity "
                "function with no registered implementation or parameters."
            )
        for binding in parameter_bindings:
            parameter = params_by_id.get(binding.parameter_id)
            if (
                parameter is None
                or parameter.name != binding.parameter
                or parameter.runtime_mutable
                or parameter.owner_component_id
                != modulation.controller_component_id
                or parameter.owner_scope
                != implementation_bindings[binding.argument].scope
            ):
                raise ValueError(
                    "KernelIR modulation controller parameter identity must "
                    "resolve to one frozen KernelIR parameter."
                )
        if modulation.target_parameter_port_id in target_port_ids:
            raise ValueError(
                "KernelIR supports at most one modulation for each target "
                "ParameterPort."
            )
        target_port_ids.add(modulation.target_parameter_port_id)

        held_parameter = effective_by_id[modulation.effective_parameter_id]
        expected_base_value = _normalize_constant(
            target.attrs.get("termination_threshold"),
            width=1,
            op_kind="EffectiveParameter",
            attr="base_value",
        )
        if (
            held_parameter.target != modulation.target
            or held_parameter.target_component_id
            != modulation.target_component_id
            or held_parameter.target_parameter != modulation.target_parameter
            or held_parameter.target_parameter_port_id
            != modulation.target_parameter_port_id
            or held_parameter.base_value != expected_base_value
            or control_projection.initial_value
            != held_parameter.initial_modulation_value
        ):
            raise ValueError(
                "KernelIR held effective-parameter target or initial values do "
                "not match its modulation edge and target declaration."
            )

        matches = tuple(
            value
            for value in dynamic_finished
            if value.component_id == modulation.target_component_id
        )
        expected_attrs = {
            "effective_parameter_id": modulation.effective_parameter_id,
            "target_parameter_port_id": modulation.target_parameter_port_id,
            "rounding": "ceil",
            "minimum": 1,
            "maximum": 2 ** 24,
        }
        if len(matches) != 1 or dict(matches[0].attrs) != expected_attrs:
            raise ValueError(
                "KernelIR modulation must own exactly one matching dynamic "
                "finished-value declaration."
            )
    if tuple(sorted(referenced_absorbed_projection_ids)) != tuple(
        range(len(kernel.absorbed_projections))
    ):
        raise ValueError(
            "KernelIR modulation routes must form an exact bijection with "
            "absorbed projection declarations."
        )


def lower_to_kernel_ir(
    ir: BatchedCompositionIR,
    *,
    op_specs: BatchedOpSpecSnapshot | None = None,
) -> KernelIR:
    """Lower semantic BatchedGraphIR and freeze its resolved op specs.

    Supplying ``op_specs`` lets a caller retain an earlier compilation
    snapshot. Direct IR callers remain supported and capture the registry as it
    exists at lowering time.
    """

    graph = ir.graph
    if graph is None:
        raise ValueError("KernelIR lowering requires a batched graph IR.")

    if op_specs is None:
        op_specs = snapshot_batched_op_specs(_graph_spec_keys(graph))

    lane_layout = _lane_layout_for(graph.fusion_kind)
    rng_streams = _rng_streams(graph)
    trial_ops = _trial_body_ops(graph)
    schedule_trace = None
    declaration_only_schedule = False
    scheduled_trial_ops = trial_ops
    trace_requested = (
        graph.metadata.get("schedule_kind") == "precomputed_trace"
        and bool(graph.termination)
    )
    if (
        _precomputed_trace_eligible(graph, lane_layout, rng_streams, op_specs)
        and trace_requested
    ):
        component_budget = _trace_budget(
            graph,
            _TRACE_COMPONENT_BUDGET_KEY,
            PRECOMPUTED_TRACE_COMPONENT_BUDGET,
        )
        weighted_op_budget = _trace_budget(
            graph,
            _TRACE_WEIGHTED_OP_BUDGET_KEY,
            _DEFAULT_TRACE_WEIGHTED_OP_BUDGET,
        )
        schedule_trace = plan_precomputed_schedule_trace(
            scheduler=graph.scheduler,
            consideration_sets=graph.consideration_sets,
            termination=graph.termination,
            expansion_budget=component_budget,
            projections=graph.projections,
            finished_values=graph.finished_values,
        )
        scheduled_trial_ops = _precomputed_trace_ops(
            graph,
            trial_ops,
            schedule_trace,
            op_specs=op_specs,
            component_budget=component_budget,
            weighted_op_budget=weighted_op_budget,
        )
    elif trace_requested or _requires_pass_region(graph):
        declaration_only_schedule = True
        pass_region = next(
            (
                region
                for region in graph.schedule_regions
                if region.kind == "pass"
            ),
            None,
        )
        scheduled_trial_ops = (
            KernelOp(
                kind="ForPasses",
                target="passes",
                attrs={
                    "region": pass_region,
                    "conditions": graph.scheduler,
                    "consideration_sets": graph.consideration_sets,
                    "finished_values": tuple(
                        KernelValue(value.name, value.width, value.dtype)
                        for value in graph.finished_values
                    ),
                    "body": trial_ops,
                    # No backend may treat this declaration as sequential
                    # execution.  Each additional scheduler tier replaces this
                    # marker only after its semantics are represented.
                    "declaration_only": True,
                },
            ),
        )
    if lane_layout.kind == STATEFUL_LANE_LAYOUT:
        initial_state_values = _state_kernel_values(graph)
        trial_reset_ops = _trial_reset_ops(graph, initial_state_values)
        ops = (
            KernelOp(
                kind="InitializeState",
                target="lane",
                outputs=initial_state_values,
            ),
            KernelOp(
                kind="ForTrials",
                target="trials",
                attrs={"body": (*trial_reset_ops, *scheduled_trial_ops)},
            ),
        )
    else:
        ops = scheduled_trial_ops

    return KernelIR(
        model_kind=ir.model_kind,
        fusion_kind=graph.fusion_kind,
        lane_layout=lane_layout,
        inputs=graph.inputs,
        params=ir.params,
        states=graph.states,
        outputs=graph.outputs,
        rng_streams=rng_streams,
        ops=ops,
        output_names=ir.output_names,
        max_steps=ir.max_steps,
        graph=graph,
        op_specs=op_specs,
        executable=graph.executable and not declaration_only_schedule,
        metadata={
            "composition_name": ir.metadata.get("composition_name"),
            "fusion_kind": graph.fusion_kind,
            **graph.metadata,
        },
        ports=graph.ports,
        absorbed_projections=graph.absorbed_projections,
        scheduler=graph.scheduler,
        schedule_regions=graph.schedule_regions,
        consideration_sets=graph.consideration_sets,
        finished_values=graph.finished_values,
        effective_parameters=graph.effective_parameters,
        modulations=graph.modulations,
        resets=graph.resets,
        termination=graph.termination,
        schedule_trace=schedule_trace,
    )


def _state_kernel_values(graph: BatchedGraphIR) -> tuple[KernelValue, ...]:
    state_slots: dict[int, int] = {}
    values = []
    for state in graph.states:
        component_id = _state_component_id(graph, state)
        state_slot = state_slots.get(component_id, 0)
        state_slots[component_id] = state_slot + 1
        values.append(
            KernelValue(
                f"n{component_id}:state:{state_slot}",
                state.width,
            )
        )
    return tuple(values)


def _trial_reset_ops(
    graph: BatchedGraphIR,
    state_values: tuple[KernelValue, ...],
) -> tuple[KernelOp, ...]:
    values_by_state_id = {
        state.state_id: value
        for state, value in zip(graph.states, state_values)
    }
    reset_ops = []
    for reset in graph.resets:
        if reset.condition_type != "AtTrialStart":
            continue
        try:
            outputs = tuple(
                values_by_state_id[state_id]
                for state_id in reset.state_ids
            )
        except KeyError as error:
            raise ValueError(
                f"Batched reset for '{reset.node}' references an undeclared "
                "state ID."
            ) from error
        reset_ops.append(
            KernelOp(
                kind="ResetState",
                target=reset.node,
                outputs=outputs,
                attrs={
                    "component_id": reset.component_id,
                    "state_ids": reset.state_ids,
                    "condition_type": reset.condition_type,
                    "region": reset.region,
                },
            )
        )
    return tuple(reset_ops)


def _trace_budget(graph: BatchedGraphIR, key: str, default: int) -> int:
    value = graph.metadata.get(key, default)
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"KernelIR metadata '{key}' must be a positive non-bool integer."
        )
    return min(value, default)


def _precomputed_trace_eligible(
    graph: BatchedGraphIR,
    lane_layout: KernelLaneLayout,
    rng_streams: tuple[KernelRngStream, ...],
    op_specs: BatchedOpSpecSnapshot,
) -> bool:
    """Defensive boundary for the first executable trace tier.

    Capability analysis remains the primary gate, but direct/public IR callers
    can construct inconsistent records.  Do not turn one of those records into
    executable nested bodies merely because its metadata requests a trace.
    """

    if not graph.executable or rng_streams:
        return False

    stateless = (
        graph.fusion_kind == STATELESS_GRAPH_FUSION
        and lane_layout.kind == TRIAL_LANE_LAYOUT
        and not graph.states
        and not graph.resets
        and not graph.finished_values
    )
    counted_stateful = _counted_stateful_trace_eligible(
        graph,
        lane_layout,
        op_specs,
    )
    if not stateless and not counted_stateful:
        return False

    try:
        node_ids = tuple(_component_id(graph, node) for node in graph.nodes)
        execution_ids = tuple(
            _component_id(graph, graph.node(node_name))
            for node_name in graph.execution_order
        )
    except (KeyError, TypeError, ValueError):
        return False
    scheduler_ids = tuple(condition.component_id for condition in graph.scheduler)
    consideration_ids = tuple(
        component_id
        for consideration_set in graph.consideration_sets
        for component_id in consideration_set.component_ids
    )
    expected_ids = tuple(sorted(node_ids))
    return (
        len(set(node_ids)) == len(node_ids)
        and tuple(sorted(execution_ids)) == expected_ids
        and tuple(sorted(scheduler_ids)) == expected_ids
        and tuple(sorted(consideration_ids)) == expected_ids
        and all(
            node.attrs.get("spec_kind") in {"elementwise", "mechanism"}
            and not node.attrs.get("diagnostics")
            for node in graph.nodes
        )
    )


def _counted_stateful_trace_eligible(
    graph: BatchedGraphIR,
    lane_layout: KernelLaneLayout,
    op_specs: BatchedOpSpecSnapshot,
) -> bool:
    """Defensive boundary for compile-time counted stateful schedules."""

    if (
        graph.fusion_kind != STATEFUL_GRAPH_FUSION
        or lane_layout.kind != STATEFUL_LANE_LAYOUT
        or not graph.states
        or not graph.finished_values
    ):
        return False

    state_component_ids = {state.component_id for state in graph.states}
    states_by_component_id = {
        component_id: tuple(
            state
            for state in graph.states
            if state.component_id == component_id
        )
        for component_id in state_component_ids
    }
    if len(graph.resets) != len(state_component_ids):
        return False
    reset_component_ids = []
    reset_state_ids = []
    for reset in graph.resets:
        try:
            node = next(
                node
                for node in graph.nodes
                if _component_id(graph, node) == reset.component_id
            )
            expected_states = states_by_component_id[reset.component_id]
        except (KeyError, StopIteration, TypeError, ValueError):
            return False
        expected_state_ids = tuple(state.state_id for state in expected_states)
        if (
            reset.node != node.name
            or reset.condition_type not in {"Never", "AtTrialStart"}
            or reset.state_ids != expected_state_ids
            or reset.attrs
            or reset.region != "trial"
        ):
            return False
        reset_component_ids.append(reset.component_id)
        reset_state_ids.extend(reset.state_ids)
    if (
        len(set(reset_component_ids)) != len(reset_component_ids)
        or tuple(sorted(reset_state_ids))
        != tuple(sorted(state.state_id for state in graph.states))
    ):
        return False

    finished_component_ids = {
        value.component_id
        for value in graph.finished_values
        if (
            value.predicate_kind == "execution_count_at_least"
            and value.storage == "combinational"
            and value.width == 1
            and value.dtype == "bool"
        )
    }
    if not (
        state_component_ids
        and state_component_ids == set(reset_component_ids) == finished_component_ids
    ):
        return False

    for component_id in finished_component_ids:
        try:
            node = next(
                node for node in graph.nodes if _component_id(graph, node) == component_id
            )
            spec = op_specs.lookup_spec(node.attrs["spec_key"])
        except (KeyError, StopIteration, TypeError, ValueError):
            return False
        if (
            not isinstance(spec, MechanismOpSpec)
            or not spec.can_step
            or spec.trial_states
            or node.attrs.get("diagnostics")
        ):
            return False
    return True


def _precomputed_trace_ops(
    graph: BatchedGraphIR,
    trial_ops: tuple[KernelOp, ...],
    trace: BatchedScheduleTraceSpec,
    *,
    op_specs: BatchedOpSpecSnapshot,
    component_budget: int,
    weighted_op_budget: int,
) -> tuple[KernelOp, ...]:
    component_bodies, epilogue = _partition_trace_trial_ops(graph, trial_ops)
    component_weights = {
        component_id: sum(_kernel_op_source_weight(op) for op in body)
        for component_id, body in component_bodies.items()
    }

    # Compute the source-expansion proxy before constructing repeated nested
    # bodies.  The component planner bounds scheduler work; this second bound
    # accounts for wide and projection-heavy component bodies whose emitted
    # Triton source can be much larger than their component count suggests.
    weighted_expansion = 1 + sum(
        _kernel_op_source_weight(op) for op in epilogue
    )
    for step in trace.steps:
        weighted_expansion += 1
        for component_id in step.component_ids:
            try:
                weighted_expansion += component_weights[component_id]
            except KeyError as error:
                raise ValueError(
                    "KernelIR precomputed trace references component id "
                    f"{component_id}, which has no lowered component body."
                ) from error
    if weighted_expansion > weighted_op_budget:
        raise ValueError(
            "KernelIR precomputed trace weighted op expansion "
            f"{weighted_expansion} exceeds budget {weighted_op_budget}."
        )

    step_component_ids = {
        value.component_id
        for value in graph.finished_values
        if value.predicate_kind == "execution_count_at_least"
    }
    _validate_scheduled_step_bodies(
        graph,
        component_bodies,
        step_component_ids,
        op_specs,
    )
    component_execution_counts = dict.fromkeys(component_bodies, 0)
    executions = []
    for step in trace.steps:
        body = []
        for component_id in step.component_ids:
            execution_index = component_execution_counts[component_id]
            body.extend(
                _scheduled_component_op(
                    graph,
                    op,
                    component_id=component_id,
                    execution_index=execution_index,
                    step_component_ids=step_component_ids,
                )
                for op in component_bodies[component_id]
            )
            component_execution_counts[component_id] += 1
        executions.append(
            KernelOp(
                kind="ExecuteConsiderationSet",
                target=(
                    f"pass-{step.pass_index}:consideration-set-"
                    f"{step.consideration_set_id}"
                ),
                attrs={
                    "pass_index": step.pass_index,
                    "consideration_set_id": step.consideration_set_id,
                    "component_ids": step.component_ids,
                    "body": tuple(body),
                },
            )
        )
    pass_regions = tuple(
        region for region in graph.schedule_regions if region.kind == "pass"
    )
    if len(pass_regions) != 1:
        raise ValueError(
            "KernelIR precomputed trace requires exactly one typed pass region, "
            f"found {len(pass_regions)}."
        )
    pass_region = pass_regions[0]
    return (
        KernelOp(
            kind="ForPasses",
            target="passes",
            attrs={
                "region": pass_region,
                "body": tuple(executions),
                "declaration_only": False,
                "trace_kind": "precomputed",
                "component_expansion_budget": component_budget,
                "weighted_op_expansion": weighted_expansion,
                "weighted_op_expansion_budget": weighted_op_budget,
            },
        ),
        *epilogue,
    )


def _validate_scheduled_step_bodies(
    graph: BatchedGraphIR,
    component_bodies: Mapping[int, tuple[KernelOp, ...]],
    step_component_ids: set[int],
    op_specs: BatchedOpSpecSnapshot,
) -> None:
    for component_id in step_component_ids:
        calls = tuple(
            op
            for op in component_bodies.get(component_id, ())
            if op.kind == "CallMechanism"
        )
        if len(calls) != 1:
            raise ValueError(
                "KernelIR counted finished component id "
                f"{component_id} requires exactly one CallMechanism body, "
                f"found {len(calls)}."
            )
        try:
            spec = op_specs.lookup_spec(calls[0].attrs["spec_key"])
        except KeyError as error:
            raise ValueError(
                "KernelIR counted finished component id "
                f"{component_id} has no frozen mechanism implementation."
            ) from error
        if not isinstance(spec, MechanismOpSpec) or not spec.can_step:
            raise ValueError(
                "KernelIR counted finished component id "
                f"{component_id} has no one-step mechanism implementation."
            )
        state_ids = tuple(
            sorted(
                state.state_id
                for state in graph.states
                if state.component_id == component_id
            )
        )
        if not state_ids:
            raise ValueError(
                "KernelIR counted finished component id "
                f"{component_id} has no retained state declarations."
            )


def _scheduled_component_op(
    graph: BatchedGraphIR,
    op: KernelOp,
    *,
    component_id: int,
    execution_index: int,
    step_component_ids: set[int],
) -> KernelOp:
    if component_id not in step_component_ids or op.kind != "CallMechanism":
        return op
    state_ids = tuple(
        sorted(
            state.state_id
            for state in graph.states
            if state.component_id == component_id
        )
    )
    return KernelOp(
        kind="StepMechanism",
        target=op.target,
        inputs=op.inputs,
        outputs=op.outputs,
        attrs={
            **op.attrs,
            "state_ids": state_ids,
            "execution_index": execution_index,
            "active_lanes": "all",
        },
    )


def _partition_trace_trial_ops(
    graph: BatchedGraphIR,
    trial_ops: tuple[KernelOp, ...],
):
    component_bodies = {
        _component_id(graph, node): []
        for node in graph.nodes
    }
    epilogue = []
    for op in trial_ops:
        if op.kind in {"StoreOutput", "StoreFlag"}:
            epilogue.append(op)
            continue
        try:
            node = graph.node(op.target)
        except KeyError as error:
            raise ValueError(
                "KernelIR cannot assign op "
                f"'{op.kind}' target '{op.target}' to a component body."
            ) from error
        component_id = _component_id(graph, node)
        component_bodies[component_id].append(_trace_component_op(op))

    empty = tuple(
        sorted(
            component_id
            for component_id, body in component_bodies.items()
            if not body
        )
    )
    if empty:
        raise ValueError(
            "KernelIR precomputed trace has no lowered body for component id(s) "
            f"{empty}."
        )
    return (
        {component_id: tuple(body) for component_id, body in component_bodies.items()},
        tuple(epilogue),
    )


def _trace_component_op(op: KernelOp) -> KernelOp:
    if op.kind != "CallFunction" or "onset_step" not in op.attrs:
        return op
    return KernelOp(
        kind=op.kind,
        target=op.target,
        inputs=op.inputs,
        outputs=op.outputs,
        attrs={
            key: value
            for key, value in op.attrs.items()
            if key != "onset_step"
        },
    )


def _kernel_op_source_weight(op: KernelOp) -> int:
    input_width = sum(value.width for value in op.inputs)
    output_width = sum(value.width for value in op.outputs)
    if op.kind == "CallProjection" and op.inputs and op.outputs:
        work = op.inputs[0].width * op.outputs[0].width
    elif op.kind in {"CombineSum", "CombineProduct"} and op.outputs:
        work = len(op.inputs) * op.outputs[0].width
    else:
        work = max(input_width, output_width)
    return 1 + max(1, work)


def _requires_pass_region(graph: BatchedGraphIR) -> bool:
    """Whether scheduler predicates require an explicit per-trial pass loop."""

    if not graph.executable:
        return True
    if graph.metadata.get("scheduler_requires_pass_region", False):
        return True
    if graph.fusion_kind == COEVOLVING_GRAPH_FUSION:
        return True

    consideration_set_ids = {
        component_id: consideration_set.consideration_set_id
        for consideration_set in graph.consideration_sets
        for component_id in consideration_set.component_ids
    }
    for condition in graph.scheduler:
        if condition.condition_type == "AtPass":
            if (
                condition.attrs.get("pass_index") != 0
                or condition.attrs.get("time_scale")
                != "ENVIRONMENT_STATE_UPDATE"
            ):
                return True
        elif condition.condition_type == "WhenFinished":
            target_set = condition.consideration_set_id
            if any(
                consideration_set_ids.get(component_id, target_set) >= target_set
                for component_id in condition.dependency_component_ids
            ):
                return True
        elif condition.condition_type in {"EveryNCalls", "AllEveryNCalls"}:
            target_set = condition.consideration_set_id
            if any(
                consideration_set_ids.get(component_id, target_set) >= target_set
                for component_id in condition.dependency_component_ids
            ):
                return True
        elif condition.condition_type not in {"Always", "AtTrialStart"}:
            return True
    return False


def _graph_spec_keys(graph: BatchedGraphIR) -> tuple[str, ...]:
    """Return each registry key referenced by ``graph``, in graph order."""

    keys = [
        graph.node(node_name).attrs["spec_key"]
        for node_name in graph.execution_order
    ]
    keys.extend(
        projection.spec_key
        for projection in graph.projections
        if projection.spec_key
    )
    keys.extend(
        state.function_initializer.spec_key
        for state in graph.states
        if state.function_initializer is not None
    )
    keys.extend(
        modulation.controller_function_spec_key
        for modulation in graph.modulations
        if modulation.controller_function_spec_key
    )
    return tuple(dict.fromkeys(keys))


def diag_slots(kernel: KernelIR) -> tuple[tuple[str, str], ...]:
    """Return the `(node, diagnostic_name)` for each diagnostic slot, by slot index.

    These are the per-lane flags the bounded-loop ops emit through `StoreFlag`
    (currently DDM truncation); the tuple's length is the diagnostic buffer's
    inner width and its order matches the slot indices written by the kernel.
    """

    slots: dict[int, tuple[str, str]] = {}
    for op in iter_kernel_ops(kernel):
        if op.kind == "StoreFlag":
            slots[int(op.attrs["slot"])] = (op.attrs["node"], op.attrs["name"])
    return tuple(slots[idx] for idx in range(len(slots)))


def iter_kernel_ops(kernel: KernelIR) -> tuple[KernelOp, ...]:
    """Return flattened KernelIR ops for tests and diagnostics."""

    flattened: list[KernelOp] = []

    def visit(op: KernelOp):
        flattened.append(op)
        for child in op.attrs.get("body", ()):
            visit(child)

    for op in kernel.ops:
        visit(op)
    return tuple(flattened)


def _lane_layout_for(fusion_kind: str | None) -> KernelLaneLayout:
    if fusion_kind in (STATEFUL_GRAPH_FUSION, COEVOLVING_GRAPH_FUSION):
        return KernelLaneLayout(
            kind=STATEFUL_LANE_LAYOUT,
            dimensions=("parameter_set", "subject", "estimate"),
        )
    return KernelLaneLayout(
        kind=TRIAL_LANE_LAYOUT,
        dimensions=("parameter_set", "subject", "trial", "estimate"),
    )


def _rng_streams(graph: BatchedGraphIR) -> tuple[KernelRngStream, ...]:
    if graph.rng_streams:
        return tuple(
            KernelRngStream(
                name=stream.name,
                node=stream.node,
                width=stream.width,
                step_extent=stream.step_extent,
                component_id=stream.component_id,
                stream_id=stream.stream_id,
            )
            for stream in graph.rng_streams
        )

    streams = []
    for node_name in graph.execution_order:
        node = graph.node(node_name)
        for stream_name, step_extent, width in node.attrs.get("rng_streams", ()):
            streams.append(
                KernelRngStream(
                    name=f"{node.name}.{stream_name}",
                    node=node.name,
                    width=int(width),
                    step_extent=step_extent,
                    component_id=_component_id(graph, node),
                    stream_id=len(streams),
                )
            )
    return tuple(streams)


def component_symbol(graph: BatchedGraphIR, node_or_name) -> str:
    """Backend-safe lowering-local symbol prefix for a graph component.

    Component display names remain the graph lookup and decorator contract.  A
    numeric prefix is used for generated symbols so target-language identifier
    sanitization can never merge distinct PNL components.
    """

    return f"n{_component_id(graph, node_or_name)}"


def node_input_value_name(graph: BatchedGraphIR, node_or_name) -> str:
    return f"{component_symbol(graph, node_or_name)}:input"


def _node_input_port_layout(graph: BatchedGraphIR, node):
    """Per-port semantic layout with a legacy fallback for direct IR callers."""

    layout = tuple(node.attrs.get("input_ports", ()))
    if layout:
        return layout

    input_spec = next(
        (
            candidate
            for candidate in graph.inputs
            if candidate.component_id == node.component_id
            or candidate.node == node.name
        ),
        None,
    )
    projection = next(
        (
            candidate
            for candidate in graph.projections
            if candidate.receiver_component_id == node.component_id
            or candidate.receiver == node.name
        ),
        None,
    )
    port_name = (
        input_spec.port
        if input_spec is not None and input_spec.port
        else projection.receiver_port
        if projection is not None
        else "InputPort-0"
    )
    port_id = (
        input_spec.port_id
        if input_spec is not None
        else projection.receiver_port_id
        if projection is not None
        else -1
    )
    return ((port_name, node.input_width, node.combine, port_id, 0, node.input_width),)


def _input_spec_for_port(graph: BatchedGraphIR, node, port_id: int):
    matches = tuple(
        input_spec
        for input_spec in graph.inputs
        if (
            port_id >= 0
            and input_spec.port_id == port_id
        )
        or (
            port_id < 0
            and (
                input_spec.component_id == node.component_id
                or input_spec.node == node.name
            )
        )
    )
    if len(matches) != 1:
        raise ValueError(
            f"Batched node '{node.name}' InputPort id {port_id} requires exactly "
            f"one external input spec, found {len(matches)}."
        )
    return matches[0]


def node_output_value_name(graph: BatchedGraphIR, node_or_name, port: str) -> str:
    node = graph.node(node_or_name) if isinstance(node_or_name, str) else node_or_name
    output_ports = tuple(name for name, _ in node.attrs.get("op_outputs", ()))
    if not output_ports:
        output_ports = tuple(node.attrs.get("output_ports", ())) or ("RESULT",)
    try:
        port_slot = output_ports.index(port)
    except ValueError as error:
        raise ValueError(
            f"Batched node '{node.name}' has no lowered output port '{port}'."
        ) from error
    return f"{component_symbol(graph, node)}:output:{port_slot}"


def node_diagnostic_value_name(
    graph: BatchedGraphIR,
    node_or_name,
    diagnostic_slot: int,
) -> str:
    return f"{component_symbol(graph, node_or_name)}:diagnostic:{diagnostic_slot}"


def _component_id(graph: BatchedGraphIR, node_or_name) -> int:
    node = graph.node(node_or_name) if isinstance(node_or_name, str) else node_or_name
    component_id = int(node.component_id)
    if component_id >= 0:
        return component_id
    # Preserve direct construction of the public experimental IR dataclasses:
    # old callers that omit ``component_id`` still receive a distinct numeric
    # identity, while normal Composition lowering always assigns one explicitly.
    for fallback_id, candidate in enumerate(graph.nodes):
        if candidate is node or candidate.name == node.name:
            return fallback_id
    raise KeyError(node.name)


def _state_component_id(graph: BatchedGraphIR, state: BatchedStateSpec) -> int:
    if state.component_id >= 0:
        return int(state.component_id)
    return _component_id(graph, state.node)


def _trial_body_ops(graph: BatchedGraphIR) -> tuple[KernelOp, ...]:
    ops: list[KernelOp] = []
    diag_slot = 0
    for node_name in graph.execution_order:
        node = graph.node(node_name)
        node_input = KernelValue(node_input_value_name(graph, node), node.input_width)
        input_ports = _node_input_port_layout(graph, node)
        port_values = []
        for port_slot, (
            port_name,
            port_width,
            combine,
            port_id,
            flat_start,
            flat_stop,
        ) in enumerate(input_ports):
            port_value = (
                node_input
                if len(input_ports) == 1
                else KernelValue(
                    f"{component_symbol(graph, node)}:input-port:{port_slot}",
                    port_width,
                )
            )
            port_values.append(port_value)
            projections = projection_inputs(
                graph,
                node.name,
                receiver_port_id=port_id,
            )
            if projections:
                projected_values = []
                for projection in projections:
                    projected = KernelValue(
                        f"{component_symbol(graph, node)}:projection:"
                        f"{projection.projection_id}",
                        projection.matrix.shape[1],
                    )
                    if projected.width != port_width:
                        raise ValueError(
                            f"Batched projection into '{node.name}.{port_name}' has "
                            f"width {projected.width}, expected {port_width}."
                        )
                    projected_values.append(projected)
                    ops.append(
                        KernelOp(
                            kind="CallProjection",
                            target=projection.receiver,
                            inputs=(
                                KernelValue(
                                    node_output_value_name(
                                        graph,
                                        projection.sender,
                                        projection.sender_port,
                                    ),
                                    projection.matrix.shape[0],
                                ),
                            ),
                            outputs=(projected,),
                            attrs={
                                "sender": projection.sender,
                                "sender_port": projection.sender_port,
                                "receiver": projection.receiver,
                                "receiver_port": projection.receiver_port,
                                "projection_id": projection.projection_id,
                                "sender_component_id": projection.sender_component_id,
                                "sender_port_id": projection.sender_port_id,
                                "receiver_component_id": projection.receiver_component_id,
                                "receiver_port_id": projection.receiver_port_id,
                                "matrix": projection.matrix,
                                "projection_type": "MappingProjection",
                                "spec_key": projection.spec_key,
                            },
                        )
                    )
                ops.append(
                    KernelOp(
                        kind=(
                            "CombineProduct"
                            if combine == "product"
                            else "CombineSum"
                        ),
                        target=node.name,
                        inputs=tuple(projected_values),
                        outputs=(port_value,),
                        attrs={
                            "component_id": node.component_id,
                            "receiver_port": port_name,
                            "receiver_port_id": port_id,
                            "flat_start": flat_start,
                            "flat_stop": flat_stop,
                        },
                    )
                )
            else:
                input_spec = _input_spec_for_port(graph, node, port_id)
                ops.append(
                    KernelOp(
                        kind="LoadInput",
                        target=node.name,
                        outputs=(port_value,),
                        attrs={
                            "node": node.name,
                            "input_name": input_spec.name,
                            "width": port_width,
                            "component_id": node.component_id,
                            "port": port_name,
                            "port_id": port_id,
                            "flat_start": flat_start,
                            "flat_stop": flat_stop,
                        },
                    )
                )

        if len(port_values) > 1:
            ops.append(
                KernelOp(
                    kind="Concatenate",
                    target=node.name,
                    inputs=tuple(port_values),
                    outputs=(node_input,),
                    attrs={
                        "component_id": node.component_id,
                        "port_ids": tuple(port[3] for port in input_ports),
                        "ports": tuple(port[0] for port in input_ports),
                    },
                )
            )

        spec_kind = node.attrs.get("spec_kind")
        if spec_kind == "elementwise":
            output_port_slices = _node_output_port_slices(node)
            output_port = output_port_slices[0][0]
            needs_port_extracts = not (
                len(output_port_slices) == 1
                and output_port_slices[0][3] == 0
                and output_port_slices[0][4] == node.output_width
            )
            output_value = (
                KernelValue(
                    f"{component_symbol(graph, node)}:mechanism-value",
                    node.output_width,
                )
                if needs_port_extracts
                else KernelValue(
                    node_output_value_name(graph, node, output_port),
                    node.output_width,
                )
            )
            function_input = node_input
            if "noise" in node.attrs:
                noisy_input = KernelValue(
                    f"{component_symbol(graph, node)}:noise",
                    node.input_width,
                )
                ops.append(
                    add_constant_op(
                        target=node.name,
                        input_value=function_input,
                        output_value=noisy_input,
                        value=node.attrs["noise"],
                    )
                )
                function_input = noisy_input
            function_output = (
                KernelValue(
                    f"{component_symbol(graph, node)}:function",
                    node.output_width,
                )
                if "clip" in node.attrs
                else output_value
            )
            attrs = {
                "component_type": node.component_type,
                "function_type": node.function_type,
                "component_id": node.component_id,
                "params": dict(node.params),
                "output_port": output_port,
                "spec_key": node.attrs["spec_key"],
            }
            if "integrator_pre" in node.attrs:
                attrs["integrator_pre"] = node.attrs["integrator_pre"]
            if "onset_step" in node.attrs:
                attrs["onset_step"] = node.attrs["onset_step"]
            ops.append(
                KernelOp(
                    kind="CallFunction",
                    target=node.name,
                    inputs=(function_input,),
                    outputs=(function_output,),
                    attrs=attrs,
                )
            )
            if "clip" in node.attrs:
                lower, upper = node.attrs["clip"]
                ops.append(
                    clamp_op(
                        target=node.name,
                        input_value=function_output,
                        output_value=output_value,
                        lower=lower,
                        upper=upper,
                    )
                )
            if needs_port_extracts:
                for (
                    port_name,
                    port_width,
                    port_id,
                    flat_start,
                    flat_stop,
                ) in output_port_slices:
                    ops.append(
                        KernelOp(
                            kind="ExtractSlice",
                            target=node.name,
                            inputs=(output_value,),
                            outputs=(
                                KernelValue(
                                    node_output_value_name(graph, node, port_name),
                                    port_width,
                                ),
                            ),
                            attrs={
                                "component_id": node.component_id,
                                "port": port_name,
                                "port_id": port_id,
                                "start": flat_start,
                                "stop": flat_stop,
                            },
                        )
                    )
        elif spec_kind == "mechanism":
            op_outputs = tuple(node.attrs.get("op_outputs", ()))
            rng_streams = tuple(node.attrs.get("rng_streams", ()))
            attrs = {
                "component_type": node.component_type,
                "function_type": node.function_type,
                "component_id": node.component_id,
                "params": dict(node.params),
                "spec_key": node.attrs["spec_key"],
            }
            if rng_streams:
                attrs["step_extent"] = rng_streams[0][1]
            diagnostics = tuple(node.attrs.get("diagnostics", ()))
            if diagnostics:
                attrs["diagnostics"] = diagnostics
                attrs["diagnostic_values"] = tuple(
                    node_diagnostic_value_name(graph, node, index)
                    for index, _ in enumerate(diagnostics)
                )
            ops.append(
                KernelOp(
                    kind="CallMechanism",
                    target=node.name,
                    inputs=(node_input,),
                    outputs=tuple(
                        KernelValue(
                            node_output_value_name(graph, node, port),
                            int(width),
                        )
                        for port, width in op_outputs
                    ),
                    attrs=attrs,
                )
            )
            for diagnostic_index, name in enumerate(diagnostics):
                ops.append(
                    KernelOp(
                        kind="StoreFlag",
                        target=node.name,
                        inputs=(
                            KernelValue(
                                node_diagnostic_value_name(
                                    graph,
                                    node,
                                    diagnostic_index,
                                ),
                                1,
                            ),
                        ),
                        attrs={"node": node.name, "name": name, "slot": diag_slot},
                    )
                )
                diag_slot += 1
        else:
            raise ValueError(
                f"Batched graph node '{node.name}' has no registered batched op spec."
            )

    for output in graph.outputs:
        ops.append(
            KernelOp(
                kind="StoreOutput",
                target=output.name,
                inputs=(
                    KernelValue(
                        node_output_value_name(graph, output.node, output.port),
                        output.width,
                    ),
                ),
                attrs={
                    "node": output.node,
                    "port": output.port,
                    "width": output.width,
                    "component_id": output.component_id,
                    "port_id": output.port_id,
                    "flat_start": output.flat_start,
                    "flat_stop": output.flat_stop,
                },
            )
        )
    return tuple(ops)


def _primary_output_port_name(node) -> str:
    output_ports = tuple(node.attrs.get("output_ports", ()))
    if output_ports:
        return output_ports[0]
    return "RESULT"


def _node_output_port_slices(node):
    slices = tuple(node.attrs.get("output_port_slices", ()))
    if slices:
        return slices
    return ((
        _primary_output_port_name(node),
        node.output_width,
        -1,
        0,
        node.output_width,
    ),)
