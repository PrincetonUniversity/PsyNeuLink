from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def projection_binding_key(
    sender: str,
    sender_port: str,
    receiver: str,
    receiver_port: str,
) -> str:
    return f"{sender}.{sender_port}->{receiver}.{receiver_port}"


@dataclass(frozen=True)
class BatchedComponentBindings:
    """Live PsyNeuLink objects associated with semantic batched IR specs.

    BatchedGraphIR and KernelIR intentionally stay serializable and
    backend-neutral.  Backend code that needs component-owned implementation
    hooks uses this sidecar binding map instead of storing live objects in IR.
    """

    nodes: Mapping[str, Any] = field(default_factory=dict)
    functions: Mapping[str, Any] = field(default_factory=dict)
    projections: Mapping[str, Any] = field(default_factory=dict)

    def node(self, name: str):
        return self.nodes[name]

    def function(self, node: str):
        return self.functions[node]

    def projection(
        self,
        sender: str,
        sender_port: str,
        receiver: str,
        receiver_port: str,
    ):
        return self.projections[
            projection_binding_key(sender, sender_port, receiver, receiver_port)
        ]


EMPTY_COMPONENT_BINDINGS = BatchedComponentBindings()
