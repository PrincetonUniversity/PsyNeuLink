from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class BatchedDiagnostic:
    component: str
    reason: str
    detail: str = ""

    def to_dict(self) -> Mapping[str, str]:
        return {
            "component": self.component,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class BatchedCapabilityReport:
    backend: str
    model_kind: str | None
    supported_nodes: tuple[str, ...] = ()
    rejected_nodes: tuple[BatchedDiagnostic, ...] = ()
    supported_conditions: tuple[str, ...] = ()
    rejected_conditions: tuple[BatchedDiagnostic, ...] = ()
    messages: tuple[str, ...] = ()
    backend_available: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_supported(self) -> bool:
        return (
            self.model_kind is not None
            and len(self.rejected_nodes) == 0
            and len(self.rejected_conditions) == 0
        )

    @property
    def unsupported_reasons(self) -> tuple[str, ...]:
        reasons = []
        for diagnostic in self.rejected_nodes + self.rejected_conditions:
            if diagnostic.detail:
                reasons.append(f"{diagnostic.component}: {diagnostic.reason} ({diagnostic.detail})")
            else:
                reasons.append(f"{diagnostic.component}: {diagnostic.reason}")
        if self.model_kind is None and not reasons:
            reasons.append("composition does not match a supported batched model family")
        return tuple(reasons)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "backend": self.backend,
            "backend_available": self.backend_available,
            "model_kind": self.model_kind,
            "is_supported": self.is_supported,
            "supported_nodes": self.supported_nodes,
            "rejected_nodes": tuple(d.to_dict() for d in self.rejected_nodes),
            "supported_conditions": self.supported_conditions,
            "rejected_conditions": tuple(d.to_dict() for d in self.rejected_conditions),
            "messages": self.messages,
            "metadata": dict(self.metadata),
        }

    def __bool__(self) -> bool:
        return self.is_supported
