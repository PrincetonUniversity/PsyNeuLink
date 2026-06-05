from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from psyneulink.core.batched.graph import (
    DDM_GRAPH_FUSION,
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


TRIAL_LANE_LAYOUT = "trial"
STATEFUL_LANE_LAYOUT = "stateful"


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


@dataclass(frozen=True)
class KernelIR:
    """Backend-neutral batched execution plan.

    Triton and the CPU debug executor both consume this IR.  A future MLIR
    backend should lower from this representation rather than re-discovering
    semantics from generated Triton source.
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
    metadata: Mapping[str, Any] = field(default_factory=dict)


def lower_to_kernel_ir(ir: BatchedCompositionIR) -> KernelIR:
    """Lower semantic BatchedGraphIR to backend-neutral KernelIR."""

    graph = ir.graph
    if graph is None:
        raise ValueError("KernelIR lowering requires a batched graph IR.")

    lane_layout = _lane_layout_for(graph.fusion_kind)
    rng_streams = _rng_streams(graph)
    trial_ops = _trial_body_ops(graph)
    if lane_layout.kind == STATEFUL_LANE_LAYOUT:
        ops = (
            KernelOp(
                kind="InitializeState",
                target="lane",
                outputs=tuple(
                    KernelValue(f"state:{state.name}", state.width)
                    for state in graph.states
                ),
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
        metadata={
            "composition_name": ir.metadata.get("composition_name"),
            "fusion_kind": graph.fusion_kind,
            **graph.metadata,
        },
    )


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
    if fusion_kind == STATEFUL_GRAPH_FUSION:
        return KernelLaneLayout(
            kind=STATEFUL_LANE_LAYOUT,
            dimensions=("parameter_set", "subject", "estimate"),
        )
    return KernelLaneLayout(
        kind=TRIAL_LANE_LAYOUT,
        dimensions=("parameter_set", "subject", "trial", "estimate"),
    )


def _rng_streams(graph: BatchedGraphIR) -> tuple[KernelRngStream, ...]:
    streams = []
    for node_name in graph.execution_order:
        node = graph.node(node_name)
        if node.component_type == "LCAMechanism":
            streams.append(
                KernelRngStream(
                    name=f"{node.name}.lca",
                    node=node.name,
                    width=node.output_width,
                    step_extent="LCA_MAX_STEPS",
                )
            )
        elif node.component_type == "DDM":
            streams.append(
                KernelRngStream(
                    name=f"{node.name}.ddm",
                    node=node.name,
                    width=1,
                    step_extent="MAX_STEPS",
                )
            )
    return tuple(streams)


def _trial_body_ops(graph: BatchedGraphIR) -> tuple[KernelOp, ...]:
    ops: list[KernelOp] = []
    for node_name in graph.execution_order:
        node = graph.node(node_name)
        node_input = KernelValue(f"{node.name}:input", node.input_width)
        projections = projection_inputs(graph, node.name)
        if projections:
            projected_values = []
            for idx, projection in enumerate(projections):
                projected = KernelValue(
                    f"{projection.receiver}:projection:{idx}",
                    projection.matrix.shape[1],
                )
                projected_values.append(projected)
                ops.append(
                    KernelOp(
                        kind="DenseMatVec",
                        target=projection.receiver,
                        inputs=(
                            KernelValue(
                                f"{projection.sender}:{projection.sender_port}",
                                projection.matrix.shape[0],
                            ),
                        ),
                        outputs=(projected,),
                        attrs={
                            "sender": projection.sender,
                            "sender_port": projection.sender_port,
                            "receiver": projection.receiver,
                            "receiver_port": projection.receiver_port,
                            "matrix": projection.matrix,
                        },
                    )
                )
            ops.append(
                KernelOp(
                    kind="CombineProduct" if node.combine == "product" else "CombineSum",
                    target=node.name,
                    inputs=tuple(projected_values),
                    outputs=(node_input,),
                )
            )
        else:
            ops.append(
                KernelOp(
                    kind="LoadInput",
                    target=node.name,
                    outputs=(node_input,),
                    attrs={"node": node.name, "width": node.input_width},
                )
            )

        if node.component_type in {"TransferMechanism", "ProcessingMechanism"}:
            output_port = _primary_output_port_name(node)
            output_value = KernelValue(f"{node.name}:{output_port}", node.output_width)
            kind = {
                "Linear": "ElementwiseLinear",
                "Logistic": "ElementwiseLogistic",
            }.get(node.function_type)
            if kind is None:
                kind = f"UnsupportedFunction:{node.function_type}"
            ops.append(
                KernelOp(
                    kind=kind,
                    target=node.name,
                    inputs=(node_input,),
                    outputs=(output_value,),
                    attrs={
                        "params": dict(node.params),
                        "output_port": output_port,
                    },
                )
            )
        elif node.component_type == "LCAMechanism":
            output_port = _primary_output_port_name(node)
            ops.append(
                KernelOp(
                    kind="LCAIntegrateUntilFinished",
                    target=node.name,
                    inputs=(node_input,),
                    outputs=(KernelValue(f"{node.name}:{output_port}", node.output_width),),
                    attrs={
                        "params": dict(node.params),
                        "output_port": output_port,
                        "pre_state": f"{node.name}.pre",
                        "act_state": f"{node.name}.act",
                        "termination_input_node": node.attrs.get("termination_input_node"),
                        "step_extent": "LCA_MAX_STEPS",
                    },
                )
            )
        elif node.component_type == "DDM":
            ops.append(
                KernelOp(
                    kind="DDMIntegrateUntilFinished",
                    target=node.name,
                    inputs=(node_input,),
                    outputs=(
                        KernelValue(f"{node.name}:DECISION_OUTCOME", 1),
                        KernelValue(f"{node.name}:RESPONSE_TIME", 1),
                    ),
                    attrs={
                        "params": dict(node.params),
                        "step_extent": "MAX_STEPS",
                    },
                )
            )

    for output in graph.outputs:
        ops.append(
            KernelOp(
                kind="StoreOutput",
                target=output.name,
                inputs=(KernelValue(f"{output.node}:{output.port}", output.width),),
                attrs={
                    "node": output.node,
                    "port": output.port,
                    "width": output.width,
                },
            )
        )
    return tuple(ops)


def _primary_output_port_name(node) -> str:
    output_ports = tuple(node.attrs.get("output_ports", ()))
    if output_ports:
        return output_ports[0]
    return "RESULT"
