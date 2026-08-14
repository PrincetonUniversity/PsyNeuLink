from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BatchedParamSpec:
    name: str
    default: float
    aliases: tuple[str, ...] = ()
    parameter_id: int = -1


@dataclass(frozen=True)
class BatchedInputSpec:
    name: str
    node: str
    width: int
    component_id: int = -1
    port_id: int = -1
    port: str = ""


@dataclass(frozen=True)
class BatchedOutputSpec:
    name: str
    node: str
    port: str
    width: int
    component_id: int = -1
    port_id: int = -1
    flat_start: int = -1
    flat_stop: int = -1

    def __post_init__(self) -> None:
        if self.flat_start == -1 and self.flat_stop == -1:
            return
        if self.flat_start < 0 or self.flat_stop < 0:
            raise ValueError(
                f"Batched output '{self.name}' requires both flattened bounds."
            )
        if self.flat_stop < self.flat_start:
            raise ValueError(
                f"Batched output '{self.name}' has flat_stop {self.flat_stop} "
                f"before flat_start {self.flat_start}."
            )
        if self.flat_stop - self.flat_start != self.width:
            raise ValueError(
                f"Batched output '{self.name}' flattened slice width "
                f"{self.flat_stop - self.flat_start} does not match output width "
                f"{self.width}."
            )

    @property
    def flat_slice(self) -> slice:
        if self.flat_start < 0 or self.flat_stop < 0:
            raise ValueError(
                f"Batched output '{self.name}' has no flattened slice assignment."
            )
        return slice(self.flat_start, self.flat_stop)


@dataclass(frozen=True)
class BatchedProjectionSpec:
    sender: str
    sender_port: str
    receiver: str
    receiver_port: str
    matrix: np.ndarray
    spec_key: str = ""
    projection_id: int = -1
    sender_component_id: int = -1
    sender_port_id: int = -1
    receiver_component_id: int = -1
    receiver_port_id: int = -1


@dataclass(frozen=True)
class BatchedNodeSpec:
    name: str
    component_type: str
    function_type: str
    input_width: int
    output_width: int
    combine: str = "sum"
    params: Mapping[str, str] = field(default_factory=dict)
    attrs: Mapping[str, Any] = field(default_factory=dict)
    # Deterministic lowering-local identity.  ``name`` remains the public PNL
    # lookup/display contract; code generation must use this numeric identity
    # so distinct names that sanitize to the same target identifier cannot
    # alias one another.
    component_id: int = -1


@dataclass(frozen=True)
class BatchedStateSpec:
    name: str
    node: str
    width: int
    initial_value: tuple[float, ...]
    component_id: int = -1
    state_id: int = -1


@dataclass(frozen=True)
class BatchedRngStreamSpec:
    name: str
    node: str
    width: int
    step_extent: str
    component_id: int = -1
    stream_id: int = -1


@dataclass(frozen=True)
class BatchedSchedulerSpec:
    node: str
    condition_type: str
    dependencies: tuple[str, ...] = ()
    attrs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchedOp:
    kind: str
    target: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    attrs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchedGraphIR:
    nodes: tuple[BatchedNodeSpec, ...]
    inputs: tuple[BatchedInputSpec, ...]
    projections: tuple[BatchedProjectionSpec, ...]
    outputs: tuple[BatchedOutputSpec, ...]
    states: tuple[BatchedStateSpec, ...]
    scheduler: tuple[BatchedSchedulerSpec, ...]
    ops: tuple[BatchedOp, ...]
    execution_order: tuple[str, ...]
    fusion_kind: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    rng_streams: tuple[BatchedRngStreamSpec, ...] = ()

    def node(self, name: str) -> BatchedNodeSpec:
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(name)


@dataclass(frozen=True)
class BatchedCompositionIR:
    model_kind: str
    node_names: tuple[str, ...]
    params: tuple[BatchedParamSpec, ...]
    output_names: tuple[str, ...] = ("decision", "response_time")
    max_steps: int = 3000
    graph: BatchedGraphIR | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def param_defaults(self) -> Mapping[str, float]:
        return {p.name: p.default for p in self.params}


@dataclass(frozen=True)
class BatchedSimulationResult:
    values: np.ndarray
    output_names: tuple[str, ...]
    backend: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
