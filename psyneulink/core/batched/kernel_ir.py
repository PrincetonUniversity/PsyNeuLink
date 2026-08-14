from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

from psyneulink.core.batched.graph import (
    COEVOLVING_GRAPH_FUSION,
    STATEFUL_GRAPH_FUSION,
    projection_inputs,
)
from psyneulink.core.batched.ir import (
    BatchedCompositionIR,
    BatchedGraphIR,
    BatchedInputSpec,
    BatchedOutputSpec,
    BatchedParamSpec,
    BatchedStateSpec,
)
from psyneulink.core.batched.specs import (
    BatchedOpSpecSnapshot,
    snapshot_batched_op_specs,
)


TRIAL_LANE_LAYOUT = "trial"
STATEFUL_LANE_LAYOUT = "stateful"
KernelConstant = Real | Iterable[Real]


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
    metadata: Mapping[str, Any] = field(default_factory=dict)


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
                attrs={"body": trial_ops},
            ),
        )
    else:
        ops = trial_ops

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
        metadata={
            "composition_name": ir.metadata.get("composition_name"),
            "fusion_kind": graph.fusion_kind,
            **graph.metadata,
        },
    )


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
        projections = projection_inputs(graph, node.name)
        if projections:
            projected_values = []
            for idx, projection in enumerate(projections):
                projected = KernelValue(
                    f"{component_symbol(graph, node)}:projection:{idx}",
                    projection.matrix.shape[1],
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
                    kind="CombineProduct" if node.combine == "product" else "CombineSum",
                    target=node.name,
                    inputs=tuple(projected_values),
                    outputs=(node_input,),
                    attrs={"component_id": node.component_id},
                )
            )
        else:
            ops.append(
                KernelOp(
                    kind="LoadInput",
                    target=node.name,
                    outputs=(node_input,),
                    attrs={
                        "node": node.name,
                        "width": node.input_width,
                        "component_id": node.component_id,
                        "port_id": next(
                            (
                                input_spec.port_id
                                for input_spec in graph.inputs
                                if (
                                    input_spec.component_id == node.component_id
                                    or input_spec.node == node.name
                                )
                            ),
                            -1,
                        ),
                    },
                )
            )

        spec_kind = node.attrs.get("spec_kind")
        if spec_kind == "elementwise":
            output_port = _primary_output_port_name(node)
            output_value = KernelValue(
                node_output_value_name(graph, node, output_port),
                node.output_width,
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
