from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class BatchedParamSpec:
    name: str
    default: float
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class BatchedCompositionIR:
    model_kind: str
    node_names: tuple[str, ...]
    params: tuple[BatchedParamSpec, ...]
    output_names: tuple[str, ...] = ("decision", "response_time")
    max_steps: int = 3000
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
