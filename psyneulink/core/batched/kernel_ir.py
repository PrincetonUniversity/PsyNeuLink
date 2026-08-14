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
    BatchedCompositionIR,
    BatchedConsiderationSetSpec,
    BatchedFinishedValueSpec,
    BatchedGraphIR,
    BatchedInputSpec,
    BatchedOutputSpec,
    BatchedParamSpec,
    BatchedResetSpec,
    BatchedScheduleTraceSpec,
    BatchedScheduleRegionSpec,
    BatchedSchedulerSpec,
    BatchedStateSpec,
    BatchedTerminationSpec,
)
from psyneulink.core.batched.schedule import plan_precomputed_schedule_trace
from psyneulink.core.batched.specs import (
    BatchedOpSpecSnapshot,
    snapshot_batched_op_specs,
)


TRIAL_LANE_LAYOUT = "trial"
STATEFUL_LANE_LAYOUT = "stateful"
KernelConstant = Real | Iterable[Real]
_DEFAULT_TRACE_COMPONENT_BUDGET = 4096
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
    scheduler: tuple[BatchedSchedulerSpec, ...] = ()
    schedule_regions: tuple[BatchedScheduleRegionSpec, ...] = ()
    consideration_sets: tuple[BatchedConsiderationSetSpec, ...] = ()
    finished_values: tuple[BatchedFinishedValueSpec, ...] = ()
    resets: tuple[BatchedResetSpec, ...] = ()
    termination: tuple[BatchedTerminationSpec, ...] = ()
    schedule_trace: BatchedScheduleTraceSpec | None = None


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
        _precomputed_trace_eligible(graph, lane_layout, rng_streams)
        and trace_requested
    ):
        component_budget = _trace_budget(
            graph,
            _TRACE_COMPONENT_BUDGET_KEY,
            _DEFAULT_TRACE_COMPONENT_BUDGET,
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
        )
        scheduled_trial_ops = _precomputed_trace_ops(
            graph,
            trial_ops,
            schedule_trace,
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
                    # execution.  The next checkpoint replaces this marker with
                    # executable predicate/conditional-region lowering.
                    "declaration_only": True,
                },
            ),
        )
    if lane_layout.kind == STATEFUL_LANE_LAYOUT:
        state_slots: dict[int, int] = {}
        initial_state_values = []
        for state in graph.states:
            component_id = _state_component_id(graph, state)
            state_slot = state_slots.get(component_id, 0)
            state_slots[component_id] = state_slot + 1
            initial_state_values.append(
                KernelValue(
                    f"n{component_id}:state:{state_slot}",
                    state.width,
                )
            )
        ops = (
            KernelOp(
                kind="InitializeState",
                target="lane",
                outputs=tuple(initial_state_values),
            ),
            KernelOp(
                kind="ForTrials",
                target="trials",
                attrs={"body": scheduled_trial_ops},
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
        scheduler=graph.scheduler,
        schedule_regions=graph.schedule_regions,
        consideration_sets=graph.consideration_sets,
        finished_values=graph.finished_values,
        resets=graph.resets,
        termination=graph.termination,
        schedule_trace=schedule_trace,
    )


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
) -> bool:
    """Defensive boundary for the first executable trace tier.

    Capability analysis remains the primary gate, but direct/public IR callers
    can construct inconsistent records.  Do not turn one of those records into
    executable nested bodies merely because its metadata requests a trace.
    """

    if (
        not graph.executable
        or graph.fusion_kind != STATELESS_GRAPH_FUSION
        or lane_layout.kind != TRIAL_LANE_LAYOUT
        or graph.states
        or rng_streams
        or graph.resets
        or graph.finished_values
    ):
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


def _precomputed_trace_ops(
    graph: BatchedGraphIR,
    trial_ops: tuple[KernelOp, ...],
    trace: BatchedScheduleTraceSpec,
    *,
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

    executions = tuple(
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
                "body": tuple(
                    op
                    for component_id in step.component_ids
                    for op in component_bodies[component_id]
                ),
            },
        )
        for step in trace.steps
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
                "body": executions,
                "declaration_only": False,
                "trace_kind": "precomputed",
                "component_expansion_budget": component_budget,
                "weighted_op_expansion": weighted_expansion,
                "weighted_op_expansion_budget": weighted_op_budget,
            },
        ),
        *epilogue,
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
