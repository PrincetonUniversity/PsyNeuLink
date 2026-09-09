"""Observation declarations and lowering to object-free port identities.

Recording precision belongs to the experiment, not the histogram bandwidth.
An event-time role requests endpoint analysis; it does not assert that the
output uniquely determines a scheduler event or its persistent-state effects.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ObservationField:
    """One observed OutputPort (live object accepted only at the front end).

    Columns follow field order and each port's flattened value order.
    ``condition_history`` and ``score`` are distinct: a known but unscored
    trial/field may still be needed to reconstruct subsequent state. Per-row
    masks will be validated by a future likelihood execution API.
    """

    output_port: object
    measure: str
    role: str = "value"
    recording: str = "exact"
    precision: float | None = None
    availability: str = "complete"
    condition_history: bool = True
    score: bool = True

    def __post_init__(self):
        if self.measure not in ("counting", "lebesgue"):
            raise ValueError("Observation measure must be 'counting' or 'lebesgue'.")
        if self.role not in ("value", "event_time"):
            raise ValueError("Observation role must be 'value' or 'event_time'.")
        if self.recording not in ("exact", "rounded", "noisy", "censored"):
            raise ValueError("Unknown observation recording policy.")
        if self.availability not in ("complete", "may_be_missing"):
            raise ValueError("Unknown observation availability policy.")
        if type(self.score) is not bool or type(self.condition_history) is not bool:
            raise ValueError("Observation score and condition_history must be booleans.")
        if not self.score and not self.condition_history:
            raise ValueError("An observation must be scored or used for conditioning.")
        if self.recording == "rounded":
            if (
                isinstance(self.precision, bool)
                or not isinstance(self.precision, (int, float))
                or not math.isfinite(self.precision)
                or self.precision <= 0
            ):
                raise ValueError("Rounded observations require finite positive precision.")
            if self.measure != "counting":
                raise ValueError("Rounded observations use counting measure.")
        elif self.precision is not None:
            raise ValueError("Precision is only defined for rounded observations.")


@dataclass(frozen=True)
class ObservationSpec:
    fields: tuple[ObservationField, ...]
    initial_state: str = "model_defaults"
    input_policy: str = "conditioned"
    sequence_policy: str = "explicit_subject_boundaries"

    def __post_init__(self):
        object.__setattr__(self, "fields", tuple(self.fields))
        if not self.fields or any(type(f) is not ObservationField for f in self.fields):
            raise ValueError("ObservationSpec requires nonempty ObservationField declarations.")
        if self.initial_state not in ("model_defaults", "latent"):
            raise ValueError("Initial state must be 'model_defaults' or 'latent'.")
        if self.input_policy != "conditioned":
            raise ValueError("Only likelihood conditional on supplied inputs is supported.")
        if self.sequence_policy != "explicit_subject_boundaries":
            raise ValueError("Sequence boundaries must be supplied explicitly.")

    @property
    def output_ports(self) -> tuple:
        return tuple(field.output_port for field in self.fields)


@dataclass(frozen=True)
class ResolvedObservationField:
    component_id: int
    port_id: int
    width: int
    column_start: int
    measure: str
    role: str
    recording: str
    precision: float | None
    availability: str
    condition_history: bool
    score: bool


def resolve_observations(spec, graph, bindings) -> tuple[ResolvedObservationField, ...]:
    """Authenticate exact live ports against the frozen graph's binding map."""
    if type(spec) is not ObservationSpec:
        raise TypeError("observations must be an ObservationSpec.")
    result = []
    seen = set()
    column = 0
    for field in spec.fields:
        matches = [
            output for output in graph.outputs
            if output.port_id in bindings.ports_by_id
            and bindings.ports_by_id[output.port_id] is field.output_port
        ]
        if len(matches) != 1:
            raise ValueError("Observed port must be an exact output of this compiled graph.")
        output = matches[0]
        if output.port_id in seen:
            raise ValueError("An output port cannot be observed twice.")
        if field.role == "event_time" and output.width != 1:
            raise ValueError("An event-time observation must be scalar.")
        seen.add(output.port_id)
        result.append(ResolvedObservationField(
            output.component_id, output.port_id, output.width, column,
            field.measure, field.role, field.recording, field.precision,
            field.availability, field.condition_history, field.score,
        ))
        column += output.width
    return tuple(result)
