from __future__ import annotations

import importlib.util

from psyneulink.core.batched.diagnostics import BatchedCapabilityReport, BatchedDiagnostic
from psyneulink.core.batched.graph import (
    DDM_MODEL,
    GRAPH_MODEL,
    STABILITY_FLEXIBILITY_MODEL,
    lower_composition,
)
from psyneulink.core.batched.ir import BatchedCompositionIR


def analyze_composition(composition, backend: str = "ir_debug", outputs=None, max_steps: int | None = None):
    lowering = lower_composition(composition, outputs=outputs)
    backend_available, backend_messages = _backend_availability(backend, lowering.model_kind, lowering.graph)

    rejected_nodes = list(lowering.rejected_nodes)
    if backend == "triton" and lowering.graph is not None and not rejected_nodes:
        from psyneulink.core.batched.backend.triton.component_hooks import (
            triton_hook_diagnostics,
        )

        rejected_nodes.extend(
            triton_hook_diagnostics(lowering.graph, lowering.bindings)
        )

    if lowering.graph is None and not rejected_nodes:
        rejected_nodes.append(
            BatchedDiagnostic(
                component=getattr(composition, "name", type(composition).__name__),
                reason="unsupported composition topology",
                detail="composition could not be lowered into the batched graph IR",
            )
        )

    report = BatchedCapabilityReport(
        backend=backend,
        model_kind=lowering.model_kind,
        supported_nodes=lowering.supported_nodes,
        rejected_nodes=tuple(rejected_nodes),
        supported_conditions=lowering.supported_conditions,
        rejected_conditions=lowering.rejected_conditions,
        messages=tuple(backend_messages),
        backend_available=backend_available,
        metadata={
            "num_nodes": len(getattr(composition, "nodes", [])),
            "fusion_kind": None if lowering.graph is None else lowering.graph.fusion_kind,
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


def _backend_availability(backend: str, model_kind: str | None, graph) -> tuple[bool, list[str]]:
    if backend == "ir_debug":
        return True, []
    if backend != "triton":
        return False, [f"Unknown batched backend '{backend}'."]

    messages = []
    if importlib.util.find_spec("triton") is None:
        messages.append("Triton is not installed; install psyneulink[triton] to execute this backend.")
    if importlib.util.find_spec("torch") is None:
        messages.append("Torch is not installed; Triton execution uses torch tensors for launch buffers.")
    if messages:
        return False, messages

    if graph is not None and graph.fusion_kind is None:
        return False, ["The Triton backend does not yet support this lowered batched graph shape."]
    if model_kind is None:
        return False, ["Composition could not be lowered for Triton execution."]
    return True, []
