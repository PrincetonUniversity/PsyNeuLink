from __future__ import annotations

import importlib
import importlib.util
from dataclasses import replace

from psyneulink.core.batched import specs
from psyneulink.core.batched.diagnostics import (
    BatchedCapabilityReport,
    BatchedDiagnostic,
    BatchedDiagnosticCode,
)
from psyneulink.core.batched.graph import lower_composition
from psyneulink.core.batched.ir import BatchedCompositionIR


def analyze_composition(composition, backend: str = "triton_cpu", outputs=None, max_steps: int | None = None):
    lowering = lower_composition(composition, outputs=outputs)
    backend_available, backend_diagnostics = _backend_availability(backend)

    rejected_nodes = [_normalize_model_diagnostic(d) for d in lowering.rejected_nodes]
    rejected_conditions = [
        _normalize_model_diagnostic(d, component_kind="node")
        for d in lowering.rejected_conditions
    ]

    if lowering.graph is None and not rejected_nodes:
        rejected_nodes.append(
            BatchedDiagnostic(
                component=getattr(composition, "name", type(composition).__name__),
                reason="unsupported composition topology",
                detail="composition could not be lowered into the batched graph IR",
                code=BatchedDiagnosticCode.MODEL_TOPOLOGY_UNSUPPORTED,
                component_id=(
                    f"composition:{getattr(composition, 'name', type(composition).__name__)}"
                ),
            )
        )

    model_supported = (
        lowering.model_kind is not None
        and not rejected_nodes
        and not rejected_conditions
    )
    codegen_ready = None
    codegen_diagnostics: list[BatchedDiagnostic] = []
    if model_supported and lowering.graph is not None:
        if backend in ("triton", "triton_cpu"):
            codegen_diagnostics.extend(_triton_spec_diagnostics(lowering.graph))
        if lowering.graph.fusion_kind is None:
            name = getattr(composition, "name", type(composition).__name__)
            codegen_diagnostics.append(
                BatchedDiagnostic(
                    component=name,
                    reason="backend does not support this lowered batched graph shape",
                    detail=backend,
                    code=BatchedDiagnosticCode.CODEGEN_FUSION_UNSUPPORTED,
                    component_id=f"composition:{name}",
                )
            )
        codegen_ready = len(codegen_diagnostics) == 0

    report = BatchedCapabilityReport(
        backend=backend,
        model_kind=lowering.model_kind,
        supported_nodes=lowering.supported_nodes,
        rejected_nodes=tuple(rejected_nodes),
        supported_conditions=lowering.supported_conditions,
        rejected_conditions=tuple(rejected_conditions),
        codegen_ready=codegen_ready,
        codegen_diagnostics=tuple(codegen_diagnostics),
        backend_available=backend_available,
        backend_diagnostics=tuple(backend_diagnostics),
        messages=tuple(d.reason for d in backend_diagnostics),
        metadata={
            "num_nodes": len(getattr(composition, "nodes", [])),
            "fusion_kind": None if lowering.graph is None else lowering.graph.fusion_kind,
            "schedule_kind": lowering.schedule_kind,
        },
    )

    ir = None
    if report.is_supported and lowering.graph is not None:
        output_names = tuple(output.name for output in lowering.graph.outputs)
        ir = BatchedCompositionIR(
            model_kind=lowering.model_kind,
            node_names=tuple(node.name for node in lowering.graph.nodes),
            params=lowering.params,
            output_names=output_names,
            max_steps=256 if max_steps is None else int(max_steps),
            graph=lowering.graph,
            metadata={
                "composition_name": getattr(composition, "name", None),
                "fusion_kind": lowering.graph.fusion_kind,
                **lowering.graph.metadata,
            },
        )

    return report, ir, lowering.bindings


def _triton_spec_diagnostics(graph) -> list[BatchedDiagnostic]:
    """Report nodes/projections whose batched op specs lack a Triton implementation."""

    diagnostics: list[BatchedDiagnostic] = []

    for node_name in graph.execution_order:
        node = graph.node(node_name)
        spec_key = node.attrs.get("spec_key")
        if not spec_key:
            continue
        spec = specs.lookup_spec(spec_key)
        if isinstance(spec, specs.ElementwiseFunctionSpec):
            if spec.triton_template is None:
                diagnostics.append(
                    BatchedDiagnostic(
                        component=node.name,
                        reason="missing Triton implementation for batched op",
                        detail=node.function_type,
                        code=BatchedDiagnosticCode.CODEGEN_OP_MISSING,
                        component_id=f"node:{node.name}",
                    )
                )
        elif isinstance(spec, specs.MechanismOpSpec):
            if not spec.has_triton:
                diagnostics.append(
                    BatchedDiagnostic(
                        component=node.name,
                        reason="missing Triton implementation for batched op",
                        detail=node.component_type,
                        code=BatchedDiagnosticCode.CODEGEN_OP_MISSING,
                        component_id=f"node:{node.name}",
                    )
                )

    for projection in graph.projections:
        spec = specs.lookup_spec(projection.spec_key) if projection.spec_key else None
        if spec is None or spec.triton_emit is None:
            component = (
                f"{projection.sender}.{projection.sender_port}->"
                f"{projection.receiver}.{projection.receiver_port}"
            )
            diagnostics.append(
                BatchedDiagnostic(
                    component=component,
                    reason="missing Triton implementation for batched op",
                    detail="projection",
                    code=BatchedDiagnosticCode.CODEGEN_OP_MISSING,
                    component_id=f"projection:{component}",
                )
            )

    return diagnostics


def _backend_availability(backend: str) -> tuple[bool, list[BatchedDiagnostic]]:
    # "triton" compiles to GPU; "triton_cpu" runs the same kernels through Triton's
    # interpreter on CPU.  Both need torch + triton importable; CUDA availability for
    # the GPU path additionally requires a usable CUDA device.
    if backend not in ("triton", "triton_cpu"):
        return False, [
            BatchedDiagnostic(
                component=backend,
                reason=f"Unknown batched backend '{backend}'.",
                code=BatchedDiagnosticCode.BACKEND_UNKNOWN,
                component_id=f"backend:{backend}",
            )
        ]

    diagnostics = []
    if importlib.util.find_spec("triton") is None:
        diagnostics.append(
            BatchedDiagnostic(
                component=backend,
                reason="Triton is not installed; install psyneulink[triton] to execute this backend.",
                code=BatchedDiagnosticCode.BACKEND_TRITON_MISSING,
                component_id=f"backend:{backend}",
            )
        )

    torch = None
    if importlib.util.find_spec("torch") is None:
        diagnostics.append(
            BatchedDiagnostic(
                component=backend,
                reason="Torch is not installed; Triton execution uses torch tensors for launch buffers.",
                code=BatchedDiagnosticCode.BACKEND_TORCH_MISSING,
                component_id=f"backend:{backend}",
            )
        )
    else:
        try:
            torch = importlib.import_module("torch")
        except Exception as error:
            diagnostics.append(
                BatchedDiagnostic(
                    component=backend,
                    reason="Torch is installed but could not be imported.",
                    detail=str(error),
                    code=BatchedDiagnosticCode.BACKEND_TORCH_UNUSABLE,
                    component_id=f"backend:{backend}",
                )
            )

    if backend == "triton" and torch is not None and not torch.cuda.is_available():
        diagnostics.append(
            BatchedDiagnostic(
                component=backend,
                reason="CUDA is not available for Triton GPU execution.",
                code=BatchedDiagnosticCode.BACKEND_CUDA_UNAVAILABLE,
                component_id=f"backend:{backend}",
            )
        )

    return len(diagnostics) == 0, diagnostics


def _normalize_model_diagnostic(
    diagnostic: BatchedDiagnostic,
    *,
    component_kind: str | None = None,
) -> BatchedDiagnostic:
    """Attach stable metadata to diagnostics produced by the model lowerer."""

    reason = diagnostic.reason
    if reason == "unsupported composition topology":
        code = BatchedDiagnosticCode.MODEL_TOPOLOGY_UNSUPPORTED
        kind = "composition"
    elif reason == "unsupported stateful transfer (integrator_mode) for batched v2":
        code = BatchedDiagnosticCode.MODEL_STATEFUL_TRANSFER_UNSUPPORTED
        kind = "node"
    elif reason == "unsupported function for batched v2" or (
        reason.startswith("unsupported ") and " function for batched v2" in reason
    ):
        code = BatchedDiagnosticCode.MODEL_FUNCTION_UNSUPPORTED
        kind = "node"
    elif reason == "unsupported input combine for batched v2":
        code = BatchedDiagnosticCode.MODEL_INPUT_COMBINE_UNSUPPORTED
        kind = "node"
    elif reason == "unsupported projection for batched v2":
        code = BatchedDiagnosticCode.MODEL_PROJECTION_UNSUPPORTED
        kind = "projection"
    elif reason == "unsupported LCA width for batched v2":
        code = BatchedDiagnosticCode.MODEL_LCA_WIDTH_UNSUPPORTED
        kind = "node"
    elif reason == "unsupported scheduler condition for static batched graph":
        code = BatchedDiagnosticCode.MODEL_SCHEDULER_CONDITION_UNSUPPORTED
        kind = "node"
    elif reason == "batched schedule kind is not executable yet":
        code = BatchedDiagnosticCode.MODEL_SCHEDULE_NOT_EXECUTABLE
        kind = "node"
    elif reason == "unsupported node for batched v2":
        code = BatchedDiagnosticCode.MODEL_NODE_UNSUPPORTED
        kind = "node"
    else:
        code = BatchedDiagnosticCode.MODEL_UNSUPPORTED
        kind = "component"

    kind = component_kind or kind
    return replace(
        diagnostic,
        code=code,
        component_id=f"{kind}:{diagnostic.component}",
    )
