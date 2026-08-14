from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


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
    nodes_by_id: Mapping[int, Any] = field(default_factory=dict)
    functions_by_id: Mapping[int, Any] = field(default_factory=dict)
    parameters_by_id: Mapping[int, Any] = field(default_factory=dict)
    ports_by_id: Mapping[int, Any] = field(default_factory=dict)
    projections_by_id: Mapping[int, Any] = field(default_factory=dict)

    def node(self, name: str):
        return self.nodes[name]

    def function(self, node: str):
        return self.functions[node]

    def node_by_id(self, component_id: int):
        return self.nodes_by_id[component_id]

    def function_by_id(self, component_id: int):
        return self.functions_by_id[component_id]

    def parameter_by_id(self, parameter_id: int):
        return self.parameters_by_id[parameter_id]

    def port_by_id(self, port_id: int):
        return self.ports_by_id[port_id]

    def projection_by_id(self, projection_id: int):
        return self.projections_by_id[projection_id]

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
