from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class BatchedDiagnosticCode:
    """Stable machine-readable codes emitted by batched capability analysis."""

    MODEL_UNSUPPORTED = "model.unsupported"
    MODEL_TOPOLOGY_UNSUPPORTED = "model.topology_unsupported"
    MODEL_NODE_UNSUPPORTED = "model.node_unsupported"
    MODEL_FUNCTION_UNSUPPORTED = "model.function_unsupported"
    MODEL_STATEFUL_TRANSFER_UNSUPPORTED = "model.stateful_transfer_unsupported"
    MODEL_INPUT_COMBINE_UNSUPPORTED = "model.input_combine_unsupported"
    MODEL_PROJECTION_UNSUPPORTED = "model.projection_unsupported"
    MODEL_LCA_WIDTH_UNSUPPORTED = "model.lca_width_unsupported"
    MODEL_SCHEDULER_CONDITION_UNSUPPORTED = "model.scheduler_condition_unsupported"
    MODEL_SCHEDULE_NOT_EXECUTABLE = "model.schedule_not_executable"

    CODEGEN_OP_MISSING = "codegen.op_missing"
    CODEGEN_FUSION_UNSUPPORTED = "codegen.fusion_unsupported"

    BACKEND_UNKNOWN = "backend.unknown"
    BACKEND_TRITON_MISSING = "backend.triton_missing"
    BACKEND_TORCH_MISSING = "backend.torch_missing"
    BACKEND_TORCH_UNUSABLE = "backend.torch_unusable"
    BACKEND_CUDA_UNAVAILABLE = "backend.cuda_unavailable"


@dataclass(frozen=True)
class BatchedDiagnostic:
    component: str
    reason: str
    detail: str = ""
    code: str = BatchedDiagnosticCode.MODEL_UNSUPPORTED
    component_id: str | None = None

    def __post_init__(self):
        if self.component_id is None:
            object.__setattr__(self, "component_id", f"component:{self.component}")

    @property
    def formatted_reason(self) -> str:
        if self.detail:
            return f"{self.component}: {self.reason} ({self.detail})"
        return f"{self.component}: {self.reason}"

    def to_dict(self) -> Mapping[str, str]:
        return {
            "code": self.code,
            "component": self.component,
            "component_id": self.component_id,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class BatchedCapabilityReport:
    """Independent model, code-generation, and runtime availability results.

    ``codegen_ready=None`` means code generation was not checked, normally
    because model lowering was rejected before a complete graph was available.
    """

    backend: str
    model_kind: str | None
    supported_nodes: tuple[str, ...] = ()
    rejected_nodes: tuple[BatchedDiagnostic, ...] = ()
    supported_conditions: tuple[str, ...] = ()
    rejected_conditions: tuple[BatchedDiagnostic, ...] = ()
    codegen_ready: bool | None = None
    codegen_diagnostics: tuple[BatchedDiagnostic, ...] = ()
    backend_available: bool = True
    backend_diagnostics: tuple[BatchedDiagnostic, ...] = ()
    messages: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def model_supported(self) -> bool:
        """Whether the composition itself is in the supported model subset."""

        return (
            self.model_kind is not None
            and len(self.rejected_nodes) == 0
            and len(self.rejected_conditions) == 0
        )

    @property
    def codegen_status(self) -> str:
        if self.codegen_ready is None:
            return "not_checked"
        return "ready" if self.codegen_ready else "not_ready"

    @property
    def can_execute(self) -> bool:
        """Whether this model can be compiled and executed in the current environment."""

        return self.model_supported and self.codegen_ready is True and self.backend_available

    @property
    def is_supported(self) -> bool:
        """Compatibility view of compiler support, excluding backend availability.

        New execution-routing code should use :attr:`can_execute`.  This property
        preserves the previous meaning used by callers that ask whether the model
        and its generated operations are supported independent of the machine on
        which diagnosis occurs.
        """

        return self.model_supported and self.codegen_ready is True

    @property
    def model_diagnostics(self) -> tuple[BatchedDiagnostic, ...]:
        return self.rejected_nodes + self.rejected_conditions

    @property
    def diagnostics(self) -> tuple[BatchedDiagnostic, ...]:
        return self.model_diagnostics + self.codegen_diagnostics + self.backend_diagnostics

    @property
    def unsupported_reasons(self) -> tuple[str, ...]:
        """Model/codegen blockers, retaining the pre-existing compatibility view."""

        reasons = [diagnostic.formatted_reason for diagnostic in self.model_diagnostics]
        reasons.extend(diagnostic.formatted_reason for diagnostic in self.codegen_diagnostics)
        if self.model_kind is None and not reasons:
            reasons.append("composition does not match a supported batched model family")
        return tuple(reasons)

    @property
    def backend_unavailable_reasons(self) -> tuple[str, ...]:
        return tuple(diagnostic.formatted_reason for diagnostic in self.backend_diagnostics)

    @property
    def execution_blockers(self) -> tuple[str, ...]:
        reasons = list(self.unsupported_reasons)
        reasons.extend(self.backend_unavailable_reasons)
        if self.model_supported and self.codegen_ready is None:
            reasons.append("code generation readiness was not checked")
        return tuple(reasons)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "backend": self.backend,
            "backend_available": self.backend_available,
            "backend_diagnostics": tuple(d.to_dict() for d in self.backend_diagnostics),
            "model_kind": self.model_kind,
            "model_supported": self.model_supported,
            "codegen_ready": self.codegen_ready,
            "codegen_status": self.codegen_status,
            "can_execute": self.can_execute,
            "is_supported": self.is_supported,
            "supported_nodes": self.supported_nodes,
            "rejected_nodes": tuple(d.to_dict() for d in self.rejected_nodes),
            "supported_conditions": self.supported_conditions,
            "rejected_conditions": tuple(d.to_dict() for d in self.rejected_conditions),
            "codegen_diagnostics": tuple(d.to_dict() for d in self.codegen_diagnostics),
            "messages": self.messages,
            "metadata": dict(self.metadata),
        }

    def __bool__(self) -> bool:
        return self.is_supported
