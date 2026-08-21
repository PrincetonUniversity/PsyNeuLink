from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from numbers import Real
import re
from typing import Any

import numpy as np

from psyneulink.core.batched.graph import (
    COEVOLVING_GRAPH_FUSION,
    STATEFUL_GRAPH_FUSION,
    STATELESS_GRAPH_FUSION,
    _dynamic_controlled_coevolving_graph_eligible,
    _node_param_aliases,
    projection_inputs,
)
from psyneulink.core.batched.ir import (
    FP32_EXACT_INTEGER_LIMIT,
    BatchedAbsorbedProjectionSpec,
    BatchedCompositionIR,
    BatchedConsiderationSetSpec,
    BatchedEffectiveParameterSpec,
    BatchedFinishedValueSpec,
    BatchedGraphIR,
    BatchedInputSpec,
    BatchedModulationSpec,
    BatchedOutputSpec,
    BatchedParamSpec,
    BatchedPortSpec,
    BatchedResetSpec,
    BatchedScheduleTraceSpec,
    BatchedScheduleRegionSpec,
    BatchedSchedulerSpec,
    BatchedStateFunctionInitializer,
    BatchedStateSpec,
    BatchedTerminationSpec,
)
from psyneulink.core.batched.schedule import (
    PRECOMPUTED_TRACE_COMPONENT_BUDGET,
    plan_precomputed_schedule_trace,
)
from psyneulink.core.batched.specs import (
    BatchedOpSpecError,
    BatchedOpSpecSnapshot,
    DenseProjectionSpec,
    ElementwiseFunctionSpec,
    MechanismOpSpec,
    snapshot_batched_op_specs,
)


TRIAL_LANE_LAYOUT = "trial"
STATEFUL_LANE_LAYOUT = "stateful"
KernelConstant = Real | Iterable[Real]
_DEFAULT_TRACE_WEIGHTED_OP_BUDGET = 65536
_TRACE_COMPONENT_BUDGET_KEY = "schedule_trace_component_budget"
_TRACE_WEIGHTED_OP_BUDGET_KEY = "schedule_trace_weighted_op_budget"


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
    component_id: int = -1
    stream_id: int = -1


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

    def __post_init__(self) -> None:
        if self.kind == "AddConstant":
            _validate_constant_elementwise_op(self, ("value",))
        elif self.kind == "Clamp":
            _validate_constant_elementwise_op(self, ("lower", "upper"))
            _validate_clamp_bounds(self)
        elif self.kind == "Concatenate":
            _validate_concatenate(self)
        elif self.kind == "ExtractSlice":
            _validate_extract_slice(self)
        elif self.kind == "InitializeEffectiveParameter":
            _validate_initialize_effective_parameter(self)
        elif self.kind == "ApplyModulation":
            _validate_apply_modulation(self)
        elif self.kind == "ForPasses":
            _validate_for_passes(self)
        elif self.kind == "ExecuteConsiderationSet":
            _validate_execute_consideration_set(self)
        elif self.kind == "StepMechanism":
            _validate_step_mechanism(self)
        elif self.kind == "ResetState":
            _validate_reset_state(self)


_DYNAMIC_MEMBER_PREDICATES = frozenset({
    "Always",
    "AtPass",
    "AtTrialStart",
    "EveryNCalls",
    "AllEveryNCalls",
    "WhenFinished",
})
_DYNAMIC_SCHEDULE_PREDICATES = _DYNAMIC_MEMBER_PREDICATES | {"AllHaveRun"}
_DYNAMIC_SCHEDULE_SLOT_KINDS = frozenset({
    "pass_index",
    "execution_count",
    "has_run",
    "usable_call",
    "finished",
    "rng_clock",
})
_DYNAMIC_SCHEDULE_CARRY_KINDS = frozenset(
    {"state", "effective_parameter", "output", "diagnostic"}
)
_DYNAMIC_SCHEDULE_BODY_OP_KINDS = frozenset({
    "AddConstant",
    "CallFunction",
    "CallMechanism",
    "CallProjection",
    "Clamp",
    "CombineProduct",
    "CombineSum",
    "Concatenate",
    "ExtractSlice",
    "LoadInput",
    "StepMechanism",
})
_DYNAMIC_SCHEDULE_EFFECT_OP_KINDS = frozenset({"ApplyModulation"})


@dataclass(frozen=True, eq=False)
class KernelSchedulePredicate:
    """One object-free predicate over lane-local scheduler state."""

    kind: str
    dependency_component_ids: tuple[int, ...] = ()
    finished_value_ids: tuple[int, ...] = ()
    pass_index: int | None = None
    call_count: int | None = None

    def __post_init__(self) -> None:
        ids = (self.dependency_component_ids, self.finished_value_ids)
        indices = (self.pass_index, self.call_count)
        valid_ids = all(_valid_dynamic_ids(values) for values in ids)
        shape = (
            (len(ids[0]), len(ids[1]), *(value is not None for value in indices))
            if valid_ids
            else None
        )
        fixed_shapes = {
            "Always": (0, 0, False, False),
            "AtPass": (0, 0, True, False),
            "AtTrialStart": (0, 0, True, False),
            "EveryNCalls": (1, 0, False, True),
            "WhenFinished": (1, 1, False, False),
        }
        valid_shape = (
            shape == fixed_shapes.get(self.kind)
            if type(self.kind) is str
            else False
        ) or (
            self.kind == "AllEveryNCalls"
            and shape is not None
            and shape[0] >= 2
            and shape[1:] == (0, False, True)
        ) or (
            self.kind == "AllHaveRun"
            and shape is not None
            and shape[0] >= 1
            and shape[1:] == (0, False, False)
        )
        if (
            type(self.kind) is not str
            or self.kind not in _DYNAMIC_SCHEDULE_PREDICATES
            or not valid_ids
            or any(
                value is not None and not _valid_dynamic_index(value)
                for value in indices
            )
            or not valid_shape
            or self.kind == "AtTrialStart" and self.pass_index != 0
            # GraphIR currently authenticates only the scheduler's implicit
            # one-call dependency predicate.  Keep this declaration equally
            # narrow until N-call credit consumption is implemented end to end.
            or self.call_count is not None and self.call_count != 1
        ):
            raise ValueError(f"KernelIR {self.kind!r} predicate operands are invalid.")


@dataclass(frozen=True, eq=False)
class KernelPublication:
    """Publish one member-local candidate into an identified loop carry."""

    source: KernelValue
    kind: str
    owner_component_id: int
    value_id: int

    def __post_init__(self) -> None:
        if (
            not _valid_dynamic_value(self.source)
            or type(self.kind) is not str
            or self.kind not in _DYNAMIC_SCHEDULE_CARRY_KINDS
            or not _valid_dynamic_id(self.owner_component_id)
            or not _valid_dynamic_id(self.value_id)
        ):
            raise ValueError("KernelIR publication has invalid typed identity.")


@dataclass(frozen=True, eq=False)
class KernelScheduledComponent:
    """One consideration-set member and its explicit publication boundary."""

    component_id: int
    predicate: KernelSchedulePredicate
    body: tuple[KernelOp, ...]
    publications: tuple[KernelPublication, ...]
    effects: tuple[KernelOp, ...] = ()

    def __post_init__(self) -> None:
        if (
            not _valid_dynamic_id(self.component_id)
            or type(self.predicate) is not KernelSchedulePredicate
            or self.predicate.kind not in _DYNAMIC_MEMBER_PREDICATES
            or not _valid_dynamic_ops(
                self.body,
                allowed_kinds=_DYNAMIC_SCHEDULE_BODY_OP_KINDS,
                nonempty=True,
            )
            or not _valid_dynamic_ops(
                self.effects,
                allowed_kinds=_DYNAMIC_SCHEDULE_EFFECT_OP_KINDS,
            )
            or not _exact_tuple(
                self.publications,
                KernelPublication,
                nonempty=True,
            )
        ):
            raise ValueError("KernelIR scheduled component fields are invalid.")
        published = tuple(
            _kernel_value_key(publication.source)
            for publication in self.publications
        )
        defined = {
            _kernel_value_key(value) for op in self.body for value in op.outputs
        }
        if len(set(published)) != len(published) or not set(published) <= defined:
            raise ValueError(
                "KernelIR published values must be unique outputs of their member body."
            )


@dataclass(frozen=True, eq=False)
class KernelConsiderationSetProgram:
    """One ordered, beginning-of-set-frozen dynamic schedule unit."""

    consideration_set_id: int
    members: tuple[KernelScheduledComponent, ...]
    inputs_frozen: bool = True

    def __post_init__(self) -> None:
        valid_members = _exact_tuple(
            self.members,
            KernelScheduledComponent,
            nonempty=True,
        )
        member_ids = (
            tuple(member.component_id for member in self.members)
            if valid_members
            else ()
        )
        if (
            not _valid_dynamic_id(self.consideration_set_id)
            or not valid_members
            or member_ids != tuple(sorted(set(member_ids)))
            or self.inputs_frozen is not True
        ):
            raise ValueError("KernelIR consideration set is not ordered and frozen.")


@dataclass(frozen=True, eq=False)
class KernelSchedulerStateSlot:
    """One typed lane-local scheduler slot with role-specific GraphIR IDs."""

    kind: str
    value: KernelValue
    owner_component_id: int | None = None
    producer_component_id: int | None = None
    consumer_component_id: int | None = None
    finished_value_id: int | None = None
    rng_stream_id: int | None = None

    def __post_init__(self) -> None:
        owner = self.owner_component_id
        producer = self.producer_component_id
        consumer = self.consumer_component_id
        finished = self.finished_value_id
        rng_stream = self.rng_stream_id
        values = (owner, producer, consumer, finished, rng_stream)
        identity_shape = tuple(value is not None for value in values)
        expected_shapes = {
            "pass_index": (False, False, False, False, False),
            "execution_count": (True, False, False, False, False),
            "has_run": (True, False, False, False, False),
            "usable_call": (False, True, True, False, False),
            "finished": (True, False, False, True, False),
            "rng_clock": (True, False, False, False, True),
        }
        expected_shape = (
            expected_shapes.get(self.kind) if type(self.kind) is str else None
        )
        expected_dtype = (
            "bool"
            if type(self.kind) is str and self.kind in {"has_run", "finished"}
            else "int32"
        )
        if (
            expected_shape is None
            or not _valid_dynamic_value(self.value)
            or self.value.width != 1
            or self.value.dtype != expected_dtype
            or any(
                value is not None and not _valid_dynamic_id(value)
                for value in values
            )
            or identity_shape != expected_shape
            or self.kind == "usable_call" and producer == consumer
        ):
            raise ValueError("KernelIR scheduler slot has invalid typed identity.")


@dataclass(frozen=True, eq=False)
class KernelLoopCarry:
    """One explicitly owned value carried by the future dynamic region."""

    kind: str
    owner_component_id: int
    value_id: int
    value: KernelValue

    def __post_init__(self) -> None:
        if (
            type(self.kind) is not str
            or self.kind not in _DYNAMIC_SCHEDULE_CARRY_KINDS
            or not _valid_dynamic_id(self.owner_component_id)
            or not _valid_dynamic_id(self.value_id)
            or not _valid_dynamic_value(self.value)
        ):
            raise ValueError("KernelIR loop carry has invalid typed identity.")


@dataclass(frozen=True, eq=False)
class KernelComponentExecutionBudget:
    """A component-local allowance, not a whole-trial termination proof."""

    component_id: int
    maximum: int

    def __post_init__(self) -> None:
        if (
            not _valid_dynamic_id(self.component_id)
            or type(self.maximum) is not int
            or not 0 < self.maximum <= FP32_EXACT_INTEGER_LIMIT
        ):
            raise ValueError("KernelIR component execution budget is invalid.")


@dataclass(frozen=True, eq=False)
class KernelDynamicScheduleProgram:
    """An inert, typed lane-local dynamic schedule declaration.

    Before this can authorize execution, the enclosing KernelIR must supply an
    authenticated trial-global pass/exhaustion budget.  Component-local budgets
    cannot stop a lane whose next required component never becomes eligible.
    """

    consideration_sets: tuple[KernelConsiderationSetProgram, ...]
    scheduler_state_slots: tuple[KernelSchedulerStateSlot, ...]
    loop_carries: tuple[KernelLoopCarry, ...]
    execution_budgets: tuple[KernelComponentExecutionBudget, ...]
    trial_termination: KernelSchedulePredicate

    def __post_init__(self) -> None:
        _validate_dynamic_schedule_program(self)


def _valid_dynamic_id(value) -> bool:
    return type(value) is int and value >= 0


def _valid_dynamic_ids(values) -> bool:
    return (
        type(values) is tuple
        and all(_valid_dynamic_id(value) for value in values)
        and values == tuple(sorted(set(values)))
    )


def _valid_dynamic_index(value) -> bool:
    return _valid_dynamic_id(value) and value <= FP32_EXACT_INTEGER_LIMIT


def _exact_tuple(values, expected_type, *, nonempty=False) -> bool:
    return bool(
        type(values) is tuple
        and (values or not nonempty)
        and all(type(value) is expected_type for value in values)
    )


def _valid_dynamic_value(value) -> bool:
    return bool(
        type(value) is KernelValue
        and type(value.name) is str
        and value.name
        and type(value.width) is int
        and value.width > 0
        and type(value.dtype) is str
        and value.dtype
    )


def _valid_dynamic_values(values, *, nonempty=False) -> bool:
    return _exact_tuple(values, KernelValue, nonempty=nonempty) and all(
        _valid_dynamic_value(value) for value in values
    )


def _valid_dynamic_ops(values, *, allowed_kinds, nonempty=False) -> bool:
    return _exact_tuple(values, KernelOp, nonempty=nonempty) and all(
        type(op.kind) is str
        and op.kind in allowed_kinds
        and type(op.target) is str
        and op.target
        and _valid_dynamic_values(op.inputs)
        and _valid_dynamic_values(op.outputs)
        and isinstance(op.attrs, Mapping)
        for op in values
    )


def _dynamic_slot_key(slot: KernelSchedulerStateSlot):
    return (
        slot.kind,
        slot.owner_component_id,
        slot.producer_component_id,
        slot.consumer_component_id,
        slot.finished_value_id,
        slot.rng_stream_id,
    )


def _dynamic_carry_key(record) -> tuple[str, int, int]:
    return (record.kind, record.owner_component_id, record.value_id)


def _dynamic_value_type(value: KernelValue) -> tuple[int, str]:
    return (value.width, value.dtype)


def _validate_dynamic_member_dataflow(
    members: tuple[KernelScheduledComponent, ...],
    carries: tuple[KernelLoopCarry, ...],
    slots: tuple[KernelSchedulerStateSlot, ...],
) -> None:
    """Authenticate frozen snapshot reads and deferred carry publication."""

    carries_by_key = {_dynamic_carry_key(carry): carry for carry in carries}
    snapshot_by_name = {carry.value.name: carry for carry in carries}
    if len(snapshot_by_name) != len(carries):
        raise ValueError("KernelIR loop-carry value names must be globally unique.")

    slot_names = {slot.value.name for slot in slots}
    if len(slot_names) != len(slots) or slot_names.intersection(snapshot_by_name):
        raise ValueError(
            "KernelIR scheduler and carry value names must be globally distinct."
        )

    all_body_output_names: set[str] = set()
    publication_source_names: set[str] = set()
    destination_writers: set[tuple[str, int, int]] = set()
    effect_destination_names: set[str] = set()

    for member in members:
        local_by_name: dict[str, KernelValue] = {}
        for op in member.body:
            for value in op.inputs:
                available = local_by_name.get(value.name)
                if available is None:
                    carry = snapshot_by_name.get(value.name)
                    available = None if carry is None else carry.value
                if available is None or _kernel_value_key(available) != (
                    _kernel_value_key(value)
                ):
                    raise ValueError(
                        "KernelIR dynamic member inputs must read a snapshot "
                        "carry or an earlier member-local output."
                    )
            for value in op.outputs:
                if (
                    value.name in snapshot_by_name
                    or value.name in slot_names
                    or value.name in local_by_name
                    or value.name in all_body_output_names
                ):
                    raise ValueError(
                        "KernelIR dynamic member outputs require unique local "
                        "candidate names distinct from scheduler and carry values."
                    )
                local_by_name[value.name] = value
                all_body_output_names.add(value.name)

        for publication in member.publications:
            source = local_by_name.get(publication.source.name)
            destination_key = _dynamic_carry_key(publication)
            destination = carries_by_key.get(destination_key)
            if (
                publication.owner_component_id != member.component_id
                or source is None
                or _kernel_value_key(source) != _kernel_value_key(publication.source)
                or destination is None
                or publication.source.name == destination.value.name
                or _dynamic_value_type(publication.source)
                != _dynamic_value_type(destination.value)
                or publication.source.name in publication_source_names
                or destination_key in destination_writers
            ):
                raise ValueError(
                    "KernelIR publications require unique member-local sources "
                    "and distinct, type-matched owned carry destinations."
                )
            publication_source_names.add(publication.source.name)
            destination_writers.add(destination_key)

        for effect in member.effects:
            _validate_apply_modulation(effect)
            for value in effect.inputs:
                available = local_by_name.get(value.name)
                if available is None:
                    carry = snapshot_by_name.get(value.name)
                    available = None if carry is None else carry.value
                if available is None or _kernel_value_key(available) != (
                    _kernel_value_key(value)
                ):
                    raise ValueError(
                        "KernelIR dynamic effects may read only snapshot carries "
                        "or outputs local to their scheduled member."
                    )
            if len(effect.outputs) != 1:
                raise ValueError(
                    "KernelIR dynamic effects require one carried destination."
                )
            output = effect.outputs[0]
            destination = snapshot_by_name.get(output.name)
            if (
                destination is None
                or _kernel_value_key(destination.value) != _kernel_value_key(output)
                or destination.kind != "effective_parameter"
                or effect.attrs.get("effective_parameter_id")
                != destination.value_id
                or effect.attrs.get("target_component_id")
                != destination.owner_component_id
                or effect.attrs.get("controller_component_id")
                != member.component_id
                or output.name in effect_destination_names
                or _dynamic_carry_key(destination) in destination_writers
            ):
                raise ValueError(
                    "KernelIR modulation effects require a unique authenticated "
                    "effective-parameter carry destination."
                )
            effect_destination_names.add(output.name)
            destination_writers.add(_dynamic_carry_key(destination))


def _validate_dynamic_schedule_program(program: KernelDynamicScheduleProgram) -> None:
    if not (
        _exact_tuple(
            program.consideration_sets,
            KernelConsiderationSetProgram,
            nonempty=True,
        )
        and _exact_tuple(program.scheduler_state_slots, KernelSchedulerStateSlot)
        and _exact_tuple(program.loop_carries, KernelLoopCarry)
        and _exact_tuple(program.execution_budgets, KernelComponentExecutionBudget)
        and type(program.trial_termination) is KernelSchedulePredicate
    ):
        raise ValueError("KernelIR dynamic schedule fields require exact typed tuples.")

    set_ids = tuple(item.consideration_set_id for item in program.consideration_sets)
    members = tuple(
        member for item in program.consideration_sets for member in item.members
    )
    member_ids = tuple(member.component_id for member in members)
    member_id_set = set(member_ids)
    if (
        set_ids != tuple(range(len(set_ids)))
        or len(member_id_set) != len(member_ids)
        or program.trial_termination.kind != "AllHaveRun"
        or program.trial_termination.dependency_component_ids
        != tuple(sorted(member_ids))
        or any(
            dependency not in member_id_set
            for member in members
            for dependency in member.predicate.dependency_component_ids
        )
    ):
        raise ValueError(
            "KernelIR dynamic schedule requires ordered sets, unique members, "
            "valid predicate references, and exact AllHaveRun termination."
        )

    slot_keys = tuple(
        _dynamic_slot_key(slot) for slot in program.scheduler_state_slots
    )
    slot_values = tuple(
        slot.value.name for slot in program.scheduler_state_slots
    )
    expected_slots = {("pass_index", None, None, None, None, None)} | {
        (kind, component_id, None, None, None, None)
        for component_id in member_ids
        for kind in ("execution_count", "has_run")
    } | {
        (
            "usable_call",
            None,
            dependency_id,
            member.component_id,
            None,
            None,
        )
        for member in members
        if member.predicate.kind in {"EveryNCalls", "AllEveryNCalls"}
        for dependency_id in member.predicate.dependency_component_ids
    } | {
        (
            "finished",
            member.predicate.dependency_component_ids[0],
            None,
            None,
            member.predicate.finished_value_ids[0],
            None,
        )
        for member in members
        if member.predicate.kind == "WhenFinished"
    }
    carry_keys = tuple(
        _dynamic_carry_key(carry) for carry in program.loop_carries
    )
    carry_values = tuple(
        carry.value.name for carry in program.loop_carries
    )
    budget_ids = tuple(budget.component_id for budget in program.execution_budgets)
    rng_stream_ids = tuple(
        slot.rng_stream_id
        for slot in program.scheduler_state_slots
        if slot.kind == "rng_clock"
    )
    if (
        len(set(slot_keys)) != len(slot_keys)
        or len(set(slot_values)) != len(slot_values)
        or {key for key in slot_keys if key[0] != "rng_clock"} != expected_slots
        or len(set(carry_keys)) != len(carry_keys)
        or len(set(carry_values)) != len(carry_values)
        or len(set(budget_ids)) != len(budget_ids)
        or len(set(rng_stream_ids)) != len(rng_stream_ids)
        or any(
            component_id is not None and component_id not in member_id_set
            for slot in program.scheduler_state_slots
            for component_id in (
                slot.owner_component_id,
                slot.producer_component_id,
                slot.consumer_component_id,
            )
        )
        or any(
            carry.owner_component_id not in member_id_set
            for carry in program.loop_carries
        )
        or any(component_id not in member_id_set for component_id in budget_ids)
    ):
        raise ValueError(
            "KernelIR dynamic state, carry, and budget identities must be "
            "unique and owned by scheduled components."
        )
    _validate_dynamic_member_dataflow(
        members,
        program.loop_carries,
        program.scheduler_state_slots,
    )


def add_constant_op(
    *,
    target: str,
    input_value: KernelValue,
    output_value: KernelValue,
    value: KernelConstant,
) -> KernelOp:
    """Build an elementwise constant addition with scalar broadcast support."""

    return KernelOp(
        kind="AddConstant",
        target=target,
        inputs=(input_value,),
        outputs=(output_value,),
        attrs={"value": value},
    )


def clamp_op(
    *,
    target: str,
    input_value: KernelValue,
    output_value: KernelValue,
    lower: KernelConstant,
    upper: KernelConstant,
) -> KernelOp:
    """Build an elementwise clamp with scalar or exact-width vector bounds."""

    return KernelOp(
        kind="Clamp",
        target=target,
        inputs=(input_value,),
        outputs=(output_value,),
        attrs={"lower": lower, "upper": upper},
    )


def effective_parameter_value(
    parameter: BatchedEffectiveParameterSpec,
) -> KernelValue:
    """Return the canonical held value for one typed effective parameter."""

    return KernelValue(
        f"effective:{parameter.effective_parameter_id}",
        parameter.width,
        parameter.dtype,
    )


def dynamic_truncation_value(
    finished_value: BatchedFinishedValueSpec,
) -> KernelValue:
    """Return the canonical lane-local pass-cap diagnostic value."""

    return KernelValue(
        f"dynamic-truncated:{finished_value.value_id}",
        1,
        "float32",
    )


def initialize_effective_parameter_op(
    parameter: BatchedEffectiveParameterSpec,
) -> KernelOp:
    """Initialize one lane-persistent effective-parameter value."""

    return KernelOp(
        kind="InitializeEffectiveParameter",
        target=parameter.target,
        outputs=(effective_parameter_value(parameter),),
        attrs=_effective_parameter_initializer_attrs(parameter),
    )


def apply_modulation_op(
    modulation: BatchedModulationSpec,
    *,
    held_effective: KernelValue,
    controller_value: KernelValue,
) -> KernelOp:
    """Apply a typed scalar ``OVERRIDE`` to a held effective parameter."""

    return KernelOp(
        kind="ApplyModulation",
        target=modulation.target,
        inputs=(held_effective, controller_value),
        outputs=(held_effective,),
        attrs={
            "modulation_id": modulation.modulation_id,
            "controller_component_id": modulation.controller_component_id,
            "control_signal_port_id": modulation.control_signal_port_id,
            "target_component_id": modulation.target_component_id,
            "target_parameter_port_id": modulation.target_parameter_port_id,
            "effective_parameter_id": modulation.effective_parameter_id,
            "mode": modulation.mode,
            "update_event": "after_controller_execution",
        },
    )


def _effective_parameter_initializer_attrs(
    parameter: BatchedEffectiveParameterSpec,
) -> dict[str, Any]:
    return {
        "effective_parameter_id": parameter.effective_parameter_id,
        "target": parameter.target,
        "target_component_id": parameter.target_component_id,
        "target_parameter": parameter.target_parameter,
        "target_parameter_port_id": parameter.target_parameter_port_id,
        "base_value": parameter.base_value,
        "initial_modulation_value": parameter.initial_modulation_value,
        "storage": parameter.storage,
        "reset": parameter.reset,
        "update_event": parameter.update_event,
        "sample_event": parameter.sample_event,
    }


def _validate_initialize_effective_parameter(op: KernelOp) -> None:
    if op.inputs or len(op.outputs) != 1:
        raise ValueError(
            "KernelIR InitializeEffectiveParameter requires no inputs and "
            "exactly one output."
        )
    effective_parameter_id = op.attrs.get("effective_parameter_id")
    expected_keys = {
        "effective_parameter_id",
        "target",
        "target_component_id",
        "target_parameter",
        "target_parameter_port_id",
        "base_value",
        "initial_modulation_value",
        "storage",
        "reset",
        "update_event",
        "sample_event",
    }
    output = op.outputs[0]
    if (
        set(op.attrs) != expected_keys
        or type(effective_parameter_id) is not int
        or effective_parameter_id < 0
        or not _kernel_value_matches(
            output,
            KernelValue(f"effective:{effective_parameter_id}", 1, "float32"),
        )
    ):
        raise ValueError(
            "KernelIR InitializeEffectiveParameter requires its exact scalar "
            "float32 effective-parameter identity and attributes."
        )


def _validate_apply_modulation(op: KernelOp) -> None:
    expected_keys = {
        "modulation_id",
        "controller_component_id",
        "control_signal_port_id",
        "target_component_id",
        "target_parameter_port_id",
        "effective_parameter_id",
        "mode",
        "update_event",
    }
    integer_keys = expected_keys - {"mode", "update_event"}
    effective_parameter_id = op.attrs.get("effective_parameter_id")
    expected_held = KernelValue(
        f"effective:{effective_parameter_id}",
        1,
        "float32",
    )
    controller_value = op.inputs[1] if len(op.inputs) == 2 else None
    if (
        set(op.attrs) != expected_keys
        or any(
            type(op.attrs.get(key)) is not int or op.attrs[key] < 0
            for key in integer_keys
        )
        or op.attrs.get("mode") != "OVERRIDE"
        or op.attrs.get("update_event") != "after_controller_execution"
        or len(op.inputs) != 2
        or len(op.outputs) != 1
        or not _kernel_value_matches(op.inputs[0], expected_held)
        or not _kernel_value_matches(op.outputs[0], expected_held)
        or type(controller_value) is not KernelValue
        or type(controller_value.name) is not str
        or not controller_value.name
        or type(controller_value.width) is not int
        or controller_value.width != 1
        or type(controller_value.dtype) is not str
        or controller_value.dtype != "float32"
    ):
        raise ValueError(
            "KernelIR ApplyModulation requires exact scalar float32 held-value "
            "rebinding and typed OVERRIDE identities."
        )


def _validate_constant_elementwise_op(
    op: KernelOp,
    constant_attrs: tuple[str, ...],
) -> None:
    if len(op.inputs) != 1 or len(op.outputs) != 1:
        raise ValueError(
            f"KernelIR {op.kind} requires exactly one input and one output."
        )
    input_value = op.inputs[0]
    output_value = op.outputs[0]
    if input_value.width != output_value.width:
        raise ValueError(
            f"KernelIR {op.kind} input/output widths must match, got "
            f"{input_value.width} and {output_value.width}."
        )
    if input_value.dtype != output_value.dtype:
        raise ValueError(
            f"KernelIR {op.kind} input/output dtypes must match, got "
            f"'{input_value.dtype}' and '{output_value.dtype}'."
        )

    attrs = dict(op.attrs)
    for attr in constant_attrs:
        try:
            value = attrs[attr]
        except KeyError as error:
            raise ValueError(
                f"KernelIR {op.kind} requires a '{attr}' constant."
            ) from error
        attrs[attr] = _normalize_constant(
            value,
            width=input_value.width,
            op_kind=op.kind,
            attr=attr,
        )
    object.__setattr__(op, "attrs", attrs)


def _normalize_constant(value, *, width: int, op_kind: str, attr: str) -> tuple[float, ...]:
    if isinstance(value, Real):
        values = (float(value),)
    elif isinstance(value, (str, bytes)):
        raise ValueError(
            f"KernelIR {op_kind} '{attr}' must be a numeric scalar or vector."
        )
    else:
        try:
            values = tuple(float(component) for component in value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"KernelIR {op_kind} '{attr}' must be a numeric scalar or vector."
            ) from error

    if len(values) not in (1, width):
        raise ValueError(
            f"KernelIR {op_kind} '{attr}' must be scalar or have width {width}, "
            f"got width {len(values)}."
        )
    return values


def _validate_clamp_bounds(op: KernelOp) -> None:
    width = op.inputs[0].width
    lower = op.attrs["lower"]
    upper = op.attrs["upper"]
    for index in range(width):
        component_lower = lower[0] if len(lower) == 1 else lower[index]
        component_upper = upper[0] if len(upper) == 1 else upper[index]
        if component_lower > component_upper:
            raise ValueError(
                "KernelIR Clamp lower bound exceeds upper bound at component "
                f"{index}: {component_lower} > {component_upper}."
            )


def _validate_concatenate(op: KernelOp) -> None:
    if not op.inputs or len(op.outputs) != 1:
        raise ValueError(
            "KernelIR Concatenate requires at least one input and exactly one output."
        )
    output = op.outputs[0]
    if sum(value.width for value in op.inputs) != output.width:
        raise ValueError(
            "KernelIR Concatenate input widths must sum to its output width."
        )
    if any(value.dtype != output.dtype for value in op.inputs):
        raise ValueError("KernelIR Concatenate input/output dtypes must match.")


def _validate_extract_slice(op: KernelOp) -> None:
    if len(op.inputs) != 1 or len(op.outputs) != 1:
        raise ValueError(
            "KernelIR ExtractSlice requires exactly one input and one output."
        )
    start = int(op.attrs.get("start", -1))
    stop = int(op.attrs.get("stop", -1))
    if start < 0 or stop < start or stop > op.inputs[0].width:
        raise ValueError(
            f"KernelIR ExtractSlice has invalid bounds [{start}:{stop}] for "
            f"input width {op.inputs[0].width}."
        )
    if stop - start != op.outputs[0].width:
        raise ValueError(
            "KernelIR ExtractSlice bounds must match its output width."
        )
    if op.inputs[0].dtype != op.outputs[0].dtype:
        raise ValueError("KernelIR ExtractSlice input/output dtypes must match.")


def _validate_for_passes(op: KernelOp) -> None:
    declaration_only = op.attrs.get("declaration_only")
    if type(declaration_only) is not bool:
        raise ValueError(
            "KernelIR ForPasses requires a typed declaration_only flag."
        )
    body = op.attrs.get("body")
    if type(body) is not tuple or any(type(child) is not KernelOp for child in body):
        raise ValueError("KernelIR ForPasses body must be a tuple of KernelOps.")
    if declaration_only:
        if op.inputs or op.outputs:
            raise ValueError(
                "Declaration-only KernelIR ForPasses cannot have value inputs "
                "or outputs."
            )
        return
    trace_kind = op.attrs.get("trace_kind")
    if trace_kind == "lane_local_counted":
        expected_keys = {
            "region",
            "body",
            "declaration_only",
            "trace_kind",
            "finished_value_id",
            "effective_parameter_id",
            "target_component_id",
            "target_parameter_port_id",
            "producer_consideration_set_id",
            "rounding",
            "minimum",
            "maximum",
            "max_steps",
        }
        integer_keys = {
            "finished_value_id",
            "effective_parameter_id",
            "target_component_id",
            "target_parameter_port_id",
            "producer_consideration_set_id",
            "minimum",
            "maximum",
            "max_steps",
        }
        if (
            set(op.attrs) != expected_keys
            or op.target != "passes"
            or len(op.inputs) != 1
            or not body
            or len(op.outputs) < 2
            or len(op.outputs[:-1]) != len(body[-1].outputs)
            or any(
                not _kernel_value_matches(region_output, step_output)
                for region_output, step_output in zip(
                    op.outputs[:-1],
                    body[-1].outputs,
                )
            )
            or not _kernel_value_matches(
                op.inputs[0],
                KernelValue(
                    f"effective:{op.attrs.get('effective_parameter_id')}",
                    1,
                    "float32",
                ),
            )
            or not _kernel_value_matches(
                op.outputs[-1],
                KernelValue(
                    f"dynamic-truncated:{op.attrs.get('finished_value_id')}",
                    1,
                    "float32",
                ),
            )
            or any(
                type(op.attrs.get(key)) is not int or op.attrs[key] < 0
                for key in integer_keys
            )
            or op.attrs["minimum"] != 1
            or op.attrs["maximum"] != 2 ** 24
            or op.attrs["max_steps"] <= 0
            or op.attrs.get("rounding") != "ceil"
            or len(tuple(child for child in body if child.kind == "StepMechanism"))
            != 1
            or body[-1].kind != "StepMechanism"
            or any(
                child.kind
                in {
                    "InitializeEffectiveParameter",
                    "ApplyModulation",
                    "ForPasses",
                    "ExecuteConsiderationSet",
                    "StoreOutput",
                    "StoreFlag",
                }
                for child in body
            )
        ):
            raise ValueError(
                "Executable lane-local counted KernelIR ForPasses requires "
                "exact held-value, truncation, bound, and one-step structure."
            )
        return
    if trace_kind == "lane_local_coevolving":
        expected_keys = {
            "region",
            "body",
            "declaration_only",
            "trace_kind",
            "stepper_component_id",
            "stepper_finished_value_id",
            "terminator_component_id",
            "terminator_finished_value_id",
            "effective_parameter_id",
            "target_parameter_port_id",
            "producer_consideration_set_id",
            "rounding",
            "minimum",
            "maximum",
            "max_steps",
            "max_steps_scope",
            "consideration_set_ids",
            "terminator_trial_states",
            "terminator_finished_output",
            "terminator_initial_control_value",
            "terminator_control_storage",
            "terminator_control_update",
            "completion_cleanup",
        }
        integer_keys = {
            "stepper_component_id",
            "stepper_finished_value_id",
            "terminator_component_id",
            "terminator_finished_value_id",
            "effective_parameter_id",
            "target_parameter_port_id",
            "producer_consideration_set_id",
            "minimum",
            "maximum",
            "max_steps",
        }
        trial_states = op.attrs.get("terminator_trial_states")
        if (
            set(op.attrs) != expected_keys
            or op.target != "passes"
            or len(op.inputs) != 1
            or op.outputs
            or not body
            or not _kernel_value_matches(
                op.inputs[0],
                KernelValue(
                    f"effective:{op.attrs.get('effective_parameter_id')}",
                    1,
                    "float32",
                ),
            )
            or any(
                type(op.attrs.get(key)) is not int or op.attrs[key] < 0
                for key in integer_keys
            )
            or op.attrs["stepper_component_id"]
            == op.attrs["terminator_component_id"]
            or op.attrs["stepper_finished_value_id"]
            == op.attrs["terminator_finished_value_id"]
            or op.attrs["minimum"] != 1
            or op.attrs["maximum"] != 2 ** 24
            or op.attrs["max_steps"] <= 0
            or op.attrs.get("max_steps_scope") != "terminator_local"
            or op.attrs.get("rounding") != "ceil"
            or type(op.attrs.get("consideration_set_ids")) is not tuple
            or op.attrs["consideration_set_ids"]
            != tuple(range(len(op.attrs["consideration_set_ids"])))
            or not op.attrs["consideration_set_ids"]
            or type(trial_states) is not tuple
            or not trial_states
            or any(
                type(item) is not tuple
                or len(item) != 3
                or type(item[0]) is not str
                or not item[0]
                or type(item[1]) is not int
                or item[1] <= 0
                or type(item[2]) is not float
                for item in trial_states
            )
            or type(op.attrs.get("terminator_finished_output")) is not str
            or not op.attrs["terminator_finished_output"]
            or type(op.attrs.get("terminator_initial_control_value")) is not float
            or op.attrs["terminator_initial_control_value"] != 1.0
            or op.attrs.get("terminator_control_storage") != "lane_persistent"
            or op.attrs.get("terminator_control_update")
            != "ordered_threshold_override"
            or op.attrs.get("completion_cleanup")
            != "fold_terminator_control_state"
            or any(
                child.kind in {"ForPasses", "ExecuteConsiderationSet"}
                for child in body
            )
            or not any(child.kind == "CallMechanism" for child in body)
            or not any(child.kind == "StoreOutput" for child in body)
        ):
            raise ValueError(
                "Executable lane-local coevolving KernelIR ForPasses requires "
                "exact stepper, terminator, held-value, trial-state, and bounded "
                "region structure."
            )
        return
    if op.inputs or op.outputs:
        raise ValueError(
            "Precomputed KernelIR ForPasses cannot have value inputs or outputs."
        )
    if trace_kind != "precomputed":
        raise ValueError(
            "Executable KernelIR ForPasses requires trace_kind='precomputed' "
            "'lane_local_counted', or 'lane_local_coevolving'."
        )
    if not body or any(child.kind != "ExecuteConsiderationSet" for child in body):
        raise ValueError(
            "Executable KernelIR ForPasses must contain only nonempty "
            "ExecuteConsiderationSet ops."
        )


def _validate_execute_consideration_set(op: KernelOp) -> None:
    if op.inputs or op.outputs:
        raise ValueError(
            "KernelIR ExecuteConsiderationSet cannot have value inputs or outputs."
        )
    pass_index = op.attrs.get("pass_index")
    consideration_set_id = op.attrs.get("consideration_set_id")
    component_ids = op.attrs.get("component_ids")
    body = op.attrs.get("body")
    if type(pass_index) is not int or pass_index < 0:
        raise ValueError(
            "KernelIR ExecuteConsiderationSet pass_index must be a non-negative "
            "non-bool integer."
        )
    if type(consideration_set_id) is not int or consideration_set_id < 0:
        raise ValueError(
            "KernelIR ExecuteConsiderationSet consideration_set_id must be a "
            "non-negative non-bool integer."
        )
    if (
        type(component_ids) is not tuple
        or not component_ids
        or any(type(component_id) is not int or component_id < 0 for component_id in component_ids)
        or component_ids != tuple(sorted(set(component_ids)))
    ):
        raise ValueError(
            "KernelIR ExecuteConsiderationSet component_ids must be a nonempty "
            "tuple of unique, sorted, non-negative non-bool integers."
        )
    if type(body) is not tuple or not body or any(
        type(child) is not KernelOp for child in body
    ):
        raise ValueError(
            "KernelIR ExecuteConsiderationSet body must be a nonempty tuple of "
            "KernelOps."
        )
    if any(
        child.kind in {"ForPasses", "ExecuteConsiderationSet", "StoreOutput", "StoreFlag"}
        for child in body
    ):
        raise ValueError(
            "KernelIR ExecuteConsiderationSet body cannot contain nested schedule "
            "regions or host-buffer stores."
        )


def _validate_step_mechanism(op: KernelOp) -> None:
    if len(op.inputs) != 1 or not op.outputs:
        raise ValueError(
            "KernelIR StepMechanism requires exactly one input and at least one "
            "output."
        )
    component_id = op.attrs.get("component_id")
    state_ids = op.attrs.get("state_ids")
    if type(component_id) is not int or component_id < 0:
        raise ValueError(
            "KernelIR StepMechanism component_id must be a non-negative "
            "non-bool integer."
        )
    if (
        type(state_ids) is not tuple
        or not state_ids
        or any(type(state_id) is not int or state_id < 0 for state_id in state_ids)
        or state_ids != tuple(sorted(set(state_ids)))
    ):
        raise ValueError(
            "KernelIR StepMechanism state_ids must be a nonempty tuple of "
            "unique, sorted, non-negative non-bool integers."
        )
    active_lanes = op.attrs.get("active_lanes")
    if active_lanes == "all":
        execution_index = op.attrs.get("execution_index")
        if type(execution_index) is not int or execution_index < 0:
            raise ValueError(
                "KernelIR precomputed StepMechanism execution_index must be a "
                "non-negative non-bool integer."
            )
        if any(
            key in op.attrs
            for key in (
                "loop_counter",
                "finished_value_id",
                "effective_parameter_id",
                "target_parameter_port_id",
            )
        ):
            raise ValueError(
                "KernelIR precomputed StepMechanism cannot carry lane-local "
                "scheduler identities."
            )
    elif active_lanes == "parent_finished_predicate":
        identity_keys = (
            "finished_value_id",
            "effective_parameter_id",
            "target_parameter_port_id",
        )
        if (
            op.attrs.get("loop_counter") != "parent_pass_index"
            or "execution_index" in op.attrs
            or any(
                type(op.attrs.get(key)) is not int or op.attrs[key] < 0
                for key in identity_keys
            )
        ):
            raise ValueError(
                "KernelIR lane-local StepMechanism requires exact parent pass "
                "counter and finished/effective-parameter identities."
            )
    else:
        raise ValueError(
            "KernelIR StepMechanism active_lanes must be 'all' or "
            "'parent_finished_predicate'."
        )
    if not isinstance(op.attrs.get("spec_key"), str) or not op.attrs["spec_key"]:
        raise ValueError("KernelIR StepMechanism requires a nonempty spec_key.")


def _validate_reset_state(op: KernelOp) -> None:
    if op.inputs or not op.outputs:
        raise ValueError(
            "KernelIR ResetState requires no inputs and at least one state output."
        )
    attrs = op.attrs
    if set(attrs) != {"component_id", "state_ids", "condition_type", "region"}:
        raise ValueError(
            "KernelIR ResetState requires exactly component_id, state_ids, "
            "condition_type, and region attributes."
        )
    component_id = attrs["component_id"]
    state_ids = attrs["state_ids"]
    if type(component_id) is not int or component_id < 0:
        raise ValueError(
            "KernelIR ResetState component_id must be a non-negative "
            "non-bool integer."
        )
    if (
        type(state_ids) is not tuple
        or not state_ids
        or any(type(state_id) is not int or state_id < 0 for state_id in state_ids)
        or state_ids != tuple(sorted(set(state_ids)))
        or len(op.outputs) != len(state_ids)
    ):
        raise ValueError(
            "KernelIR ResetState state_ids must be a nonempty tuple of unique, "
            "sorted, non-negative integers matching its state outputs."
        )
    if attrs["condition_type"] != "AtTrialStart" or attrs["region"] != "trial":
        raise ValueError(
            "KernelIR ResetState requires condition_type='AtTrialStart' and "
            "region='trial'."
        )
@dataclass(frozen=True)
class KernelIR:
    """Backend-neutral batched execution plan.

    Triton and the CPU debug executor both consume this IR.  A future MLIR
    backend should lower from this representation rather than re-discovering
    semantics from generated Triton source. ``op_specs`` is an immutable
    lowering-environment sidecar: it fixes the registered implementations used
    by this plan without putting implementation objects in individual op attrs.
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
    op_specs: BatchedOpSpecSnapshot
    executable: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    ports: tuple[BatchedPortSpec, ...] = ()
    absorbed_projections: tuple[BatchedAbsorbedProjectionSpec, ...] = ()
    scheduler: tuple[BatchedSchedulerSpec, ...] = ()
    schedule_regions: tuple[BatchedScheduleRegionSpec, ...] = ()
    consideration_sets: tuple[BatchedConsiderationSetSpec, ...] = ()
    finished_values: tuple[BatchedFinishedValueSpec, ...] = ()
    effective_parameters: tuple[BatchedEffectiveParameterSpec, ...] = ()
    modulations: tuple[BatchedModulationSpec, ...] = ()
    resets: tuple[BatchedResetSpec, ...] = ()
    termination: tuple[BatchedTerminationSpec, ...] = ()
    schedule_trace: BatchedScheduleTraceSpec | None = None

    def __post_init__(self) -> None:
        validate_kernel_ir(self)


def validate_kernel_ir(kernel: KernelIR) -> None:
    """Validate cross-op identity and effect invariants in a complete KernelIR."""

    if (
        type(kernel.executable) is not bool
        or type(kernel.graph.executable) is not bool
    ):
        raise ValueError(
            "KernelIR and GraphIR executable flags must be exact booleans."
        )
    if kernel.executable and not kernel.graph.executable:
        raise ValueError(
            "Executable KernelIR requires executable GraphIR capability authority."
        )

    # Validate identity-bearing declaration sequences before constructing any
    # lookup dictionaries from them.  Otherwise duplicate or bool-valued IDs
    # can be silently collapsed by Python's mapping-key equality.
    _validate_kernel_parameters(kernel)
    _validate_kernel_ports(kernel)
    _validate_kernel_node_implementations(kernel)
    _validate_kernel_states(kernel)
    _validate_kernel_projections(kernel)
    _validate_kernel_io_declarations(kernel)
    _validate_kernel_finished_scheduler(kernel)

    semantic_fields = (
        ("input", kernel.inputs, kernel.graph.inputs),
        ("retained-state", kernel.states, kernel.graph.states),
        ("output", kernel.outputs, kernel.graph.outputs),
        ("port", kernel.ports, kernel.graph.ports),
        (
            "absorbed-projection",
            kernel.absorbed_projections,
            kernel.graph.absorbed_projections,
        ),
        ("scheduler", kernel.scheduler, kernel.graph.scheduler),
        ("schedule-region", kernel.schedule_regions, kernel.graph.schedule_regions),
        ("consideration-set", kernel.consideration_sets, kernel.graph.consideration_sets),
        ("finished-value", kernel.finished_values, kernel.graph.finished_values),
        (
            "effective-parameter",
            kernel.effective_parameters,
            kernel.graph.effective_parameters,
        ),
        ("modulation", kernel.modulations, kernel.graph.modulations),
        ("reset", kernel.resets, kernel.graph.resets),
        ("termination", kernel.termination, kernel.graph.termination),
    )
    for label, kernel_declarations, graph_declarations in semantic_fields:
        if kernel_declarations != graph_declarations:
            raise ValueError(
                f"KernelIR {label} declarations must exactly match GraphIR."
            )
    _validate_kernel_modulations(kernel)
    _validate_kernel_reset_declarations(kernel)
    step_counts: dict[int, int] = {}
    precomputed_regions: list[KernelOp] = []
    input_loads: list[KernelOp] = []
    output_stores: list[KernelOp] = []
    value_lineage: dict[
        tuple[str, int, str],
        tuple[frozenset[int], frozenset[int]],
    ] = {}
    lineage_at_store: dict[
        int,
        tuple[frozenset[int], frozenset[int]],
    ] = {}
    reset_ops: list[KernelOp] = []

    def visit(
        op: KernelOp,
        *,
        in_precomputed_region: bool = False,
        in_lane_local_region: bool = False,
        in_trials: bool = False,
    ) -> None:
        for value in (*op.inputs, *op.outputs):
            if (
                type(value) is not KernelValue
                or type(value.name) is not str
                or not value.name
                or type(value.width) is not int
                or value.width <= 0
                or type(value.dtype) is not str
                or not value.dtype
            ):
                raise ValueError(
                    f"KernelIR {op.kind} values require nonempty names and "
                    "dtypes plus positive non-bool widths."
                )
        if op.kind == "InitializeEffectiveParameter":
            _validate_initialize_effective_parameter(op)
        elif op.kind == "ApplyModulation":
            _validate_apply_modulation(op)
        elif op.kind == "ForPasses":
            _validate_for_passes(op)
        elif op.kind == "StepMechanism":
            _validate_step_mechanism(op)
        if op.kind != "StoreOutput":
            missing_inputs = tuple(
                value.name
                for value in op.inputs
                if _kernel_value_key(value) not in value_lineage
            )
            if missing_inputs:
                raise ValueError(
                    f"KernelIR {op.kind} input value(s) {missing_inputs} must "
                    "be defined by a dominating operation."
                )
        if op.kind == "StepMechanism":
            if not in_precomputed_region and not in_lane_local_region:
                raise ValueError(
                    "KernelIR StepMechanism must belong to an executable "
                    "precomputed or lane-local counted ForPasses region."
                )
            if (
                in_precomputed_region
                and op.attrs.get("active_lanes") != "all"
            ) or (
                in_lane_local_region
                and op.attrs.get("active_lanes")
                != "parent_finished_predicate"
            ):
                raise ValueError(
                    "KernelIR StepMechanism active-lane policy must match its "
                    "parent ForPasses trace kind."
                )
            try:
                node = kernel.graph.node(op.target)
            except KeyError as error:
                raise ValueError(
                    f"KernelIR StepMechanism target '{op.target}' is not declared."
                ) from error
            component_id = _component_id(kernel.graph, node)
            expected_state_ids = tuple(
                sorted(
                    state.state_id
                    for state in kernel.states
                    if _state_component_id(kernel.graph, state) == component_id
                )
            )
            if op.attrs["component_id"] != component_id:
                raise ValueError(
                    f"KernelIR StepMechanism target '{op.target}' has component "
                    f"id {op.attrs['component_id']}, expected {component_id}."
                )
            if op.attrs["state_ids"] != expected_state_ids:
                raise ValueError(
                    f"KernelIR StepMechanism target '{op.target}' has state IDs "
                    f"{op.attrs['state_ids']}, expected {expected_state_ids}."
                )
            if op.attrs["spec_key"] != node.attrs.get("spec_key"):
                raise ValueError(
                    f"KernelIR StepMechanism target '{op.target}' does not use "
                    "its frozen graph implementation key."
                )
            if in_precomputed_region:
                expected_execution_index = step_counts.get(component_id, 0)
                if op.attrs["execution_index"] != expected_execution_index:
                    raise ValueError(
                        f"KernelIR StepMechanism target '{op.target}' has execution "
                        f"index {op.attrs['execution_index']}, expected "
                        f"{expected_execution_index}."
                    )
                step_counts[component_id] = expected_execution_index + 1
        elif op.kind == "LoadInput":
            input_loads.append(op)
        elif op.kind == "StoreOutput":
            if (
                kernel.lane_layout.kind == STATEFUL_LANE_LAYOUT
                and not in_trials
            ):
                raise ValueError(
                    "Stateful KernelIR StoreOutput operations must be inside "
                    "the ForTrials body."
                )
            if kernel.lane_layout.kind == TRIAL_LANE_LAYOUT and in_trials:
                raise ValueError(
                    "Trial-lane KernelIR StoreOutput operations cannot be "
                    "inside a ForTrials body."
                )
            output_stores.append(op)
            lineage_at_store[id(op)] = (
                value_lineage.get(
                    _kernel_value_key(op.inputs[0]),
                    (frozenset(), frozenset()),
                )
                if len(op.inputs) == 1 and type(op.inputs[0]) is KernelValue
                else (frozenset(), frozenset())
            )
        elif op.kind == "ResetState":
            # attrs is a Mapping and may have been mutated after construction;
            # source emission re-runs this complete-IR boundary.
            _validate_reset_state(op)
            reset_ops.append(op)

        lane_local_region = bool(
            op.kind == "ForPasses"
            and op.attrs.get("declaration_only") is False
            and op.attrs.get("trace_kind") == "lane_local_counted"
        )
        outer_lineage = dict(value_lineage) if lane_local_region else None
        region_output_lineage = _kernel_op_output_lineage(
            kernel,
            op,
            value_lineage,
        )
        if not lane_local_region:
            value_lineage.update(region_output_lineage)

        child_precomputed = in_precomputed_region
        child_lane_local = in_lane_local_region
        if op.kind == "ForPasses":
            child_precomputed = (
                op.attrs.get("declaration_only") is False
                and op.attrs.get("trace_kind") == "precomputed"
            )
            child_lane_local = (
                op.attrs.get("declaration_only") is False
                and op.attrs.get("trace_kind") == "lane_local_counted"
            )
            if child_precomputed:
                precomputed_regions.append(op)
        for child in op.attrs.get("body", ()):
            visit(
                child,
                in_precomputed_region=child_precomputed,
                in_lane_local_region=child_lane_local,
                in_trials=in_trials or op.kind == "ForTrials",
            )
        if lane_local_region:
            assert outer_lineage is not None
            yielded_lineage = {}
            for output in op.outputs[:-1]:
                output_key = _kernel_value_key(output)
                try:
                    yielded_lineage[output_key] = value_lineage[output_key]
                except KeyError as error:
                    raise ValueError(
                        "KernelIR lane-local ForPasses result values must be "
                        "defined by its body."
                    ) from error
            value_lineage.clear()
            value_lineage.update(outer_lineage)
            value_lineage.update(region_output_lineage)
            value_lineage.update(yielded_lineage)

    for op in kernel.ops:
        visit(op)

    if kernel.executable:
        _validate_executable_kernel_io_ops(
            kernel,
            input_loads=input_loads,
            output_stores=output_stores,
            lineage_at_store=lineage_at_store,
        )

    if kernel.schedule_trace is not None:
        if len(precomputed_regions) != 1:
            raise ValueError(
                "KernelIR schedule_trace requires exactly one executable "
                "precomputed ForPasses region."
            )
        region_steps = tuple(
            (
                child.attrs.get("pass_index"),
                child.attrs.get("consideration_set_id"),
                child.attrs.get("component_ids"),
            )
            for child in precomputed_regions[0].attrs["body"]
        )
        declared_steps = tuple(
            (
                step.pass_index,
                step.consideration_set_id,
                step.component_ids,
            )
            for step in kernel.schedule_trace.steps
        )
        if region_steps != declared_steps:
            raise ValueError(
                "KernelIR executable precomputed region does not match its "
                "typed schedule_trace."
            )
        for execution in precomputed_regions[0].attrs["body"]:
            declared_component_ids = set(execution.attrs["component_ids"])
            body_component_ids = set()
            for child in execution.attrs["body"]:
                try:
                    child_component_id = _component_id(
                        kernel.graph,
                        kernel.graph.node(child.target),
                    )
                except KeyError as error:
                    raise ValueError(
                        "KernelIR executable consideration-set body references "
                        f"undeclared target '{child.target}'."
                    ) from error
                if child_component_id not in declared_component_ids:
                    raise ValueError(
                        "KernelIR executable consideration-set body target "
                        f"'{child.target}' is not one of its declared component "
                        "IDs."
                    )
                body_component_ids.add(child_component_id)
            if body_component_ids != declared_component_ids:
                raise ValueError(
                    "KernelIR executable consideration-set body does not cover "
                    "exactly its declared component IDs."
                )
        declared_execution_count = sum(
            len(step.component_ids) for step in kernel.schedule_trace.steps
        )
        if (
            kernel.schedule_trace.component_execution_count
            != declared_execution_count
        ):
            raise ValueError(
                "KernelIR schedule_trace component_execution_count does not "
                "match its declared steps."
            )
    elif precomputed_regions:
        raise ValueError(
            "KernelIR executable precomputed ForPasses region requires a typed "
            "schedule_trace."
        )

    expected_reset_declarations = tuple(
        reset
        for reset in kernel.resets
        if reset.condition_type == "AtTrialStart"
    )
    stateful_layout = kernel.lane_layout.kind == STATEFUL_LANE_LAYOUT
    trial_regions = tuple(op for op in kernel.ops if op.kind == "ForTrials")
    if (
        stateful_layout
        and (
            kernel.schedule_trace is not None
            or expected_reset_declarations
            or reset_ops
        )
        and len(trial_regions) != 1
    ):
        raise ValueError(
            "KernelIR stateful scheduled/reset effects require exactly one "
            "ForTrials region."
        )
    if stateful_layout and (expected_reset_declarations or reset_ops):
        trial_body = tuple(trial_regions[0].attrs.get("body", ()))
        reset_prefix = tuple(
            op for op in trial_body[:len(reset_ops)] if op.kind == "ResetState"
        )
        if tuple(reset_ops) != reset_prefix:
            raise ValueError(
                "KernelIR ResetState operations must form an unconditional "
                "prefix of the ForTrials body."
            )
        if len(reset_prefix) != len(expected_reset_declarations):
            raise ValueError(
                "KernelIR must contain exactly one ResetState for each "
                "AtTrialStart reset declaration."
            )
        state_values = {
            state.state_id: value
            for state, value in zip(kernel.states, _state_kernel_values(kernel.graph))
        }
        states_by_id = {state.state_id: state for state in kernel.states}
        for op, reset in zip(reset_prefix, expected_reset_declarations):
            try:
                declared_states = tuple(
                    states_by_id[state_id]
                    for state_id in reset.state_ids
                )
                expected_outputs = tuple(
                    state_values[state_id]
                    for state_id in reset.state_ids
                )
            except KeyError as error:
                raise ValueError(
                    f"KernelIR reset for '{reset.node}' references an "
                    "undeclared state ID."
                ) from error
            if (
                reset.attrs
                or reset.region != "trial"
                or any(state.component_id != reset.component_id for state in declared_states)
                or op.target != reset.node
                or op.outputs != expected_outputs
                or dict(op.attrs) != {
                    "component_id": reset.component_id,
                    "state_ids": reset.state_ids,
                    "condition_type": "AtTrialStart",
                    "region": "trial",
                }
            ):
                raise ValueError(
                    "KernelIR ResetState does not exactly match its declared "
                    f"reset for '{reset.node}'."
                )
    elif reset_ops or expected_reset_declarations:
        raise ValueError(
            "KernelIR AtTrialStart reset effects require a stateful lane "
            "layout."
        )

    _validate_dynamic_modulation_ops(kernel)

    counted_finished = {
        value.component_id: value.attrs.get("count")
        for value in kernel.finished_values
        if value.predicate_kind == "execution_count_at_least"
    }
    if not step_counts:
        if kernel.executable and counted_finished:
            raise ValueError(
                "Executable KernelIR counted finished values require typed "
                "StepMechanism operations."
            )
        return
    if set(step_counts) != set(counted_finished):
        raise ValueError(
            "KernelIR StepMechanism owners must exactly match counted finished "
            "value owners."
        )
    trace_execution_counts: dict[int, int] = {}
    for step in kernel.schedule_trace.steps:
        for component_id in step.component_ids:
            trace_execution_counts[component_id] = (
                trace_execution_counts.get(component_id, 0) + 1
            )
    for component_id, count in counted_finished.items():
        if step_counts[component_id] != trace_execution_counts.get(component_id, 0):
            raise ValueError(
                "KernelIR counted finished component id "
                f"{component_id} StepMechanism count does not match its typed "
                "schedule trace."
            )
        if type(count) is not int or count <= 0 or step_counts[component_id] < count:
            raise ValueError(
                "KernelIR counted finished component id "
                f"{component_id} requires at least {count!r} scheduled steps, "
                f"found {step_counts[component_id]}."
            )


def _nodes_resolving_component_id(
    graph: BatchedGraphIR,
    component_id: int,
) -> tuple:
    """Return graph nodes resolving to one lowering-local component identity."""

    return tuple(
        node
        for node in graph.nodes
        if _component_id(graph, node) == component_id
    )


def _consideration_set_owns(
    consideration_set: BatchedConsiderationSetSpec,
    node: str,
    component_id: int,
) -> bool:
    """Whether one typed consideration set owns an exact node/ID pair."""

    return bool(
        type(consideration_set.consideration_set_id) is int
        and type(consideration_set.nodes) is tuple
        and type(consideration_set.component_ids) is tuple
        and len(consideration_set.nodes) == len(consideration_set.component_ids)
        and any(
            type(member_name) is str
            and type(member_id) is int
            and member_name == node
            and member_id == component_id
            for member_name, member_id in zip(
                consideration_set.nodes,
                consideration_set.component_ids,
            )
        )
    )


def _validate_kernel_parameters(kernel: KernelIR) -> None:
    """Validate the global parameter inventory without lossy ID lookups."""

    parameter_ids = tuple(parameter.parameter_id for parameter in kernel.params)
    if any(type(parameter_id) is not int for parameter_id in parameter_ids):
        raise ValueError(
            "KernelIR parameter IDs must be exact non-bool integers."
        )
    if parameter_ids != tuple(range(len(kernel.params))):
        raise ValueError(
            "KernelIR parameter IDs must be unique, contiguous, and in "
            "declaration order."
        )

    parameter_names = tuple(parameter.name for parameter in kernel.params)
    if any(type(name) is not str or not name for name in parameter_names):
        raise ValueError(
            "KernelIR parameter canonical names must be nonempty strings."
        )
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError(
            "KernelIR parameter canonical names must be unique."
        )

    for parameter in kernel.params:
        if type(parameter.owner_component_id) is not int:
            raise ValueError(
                "KernelIR parameter owner component IDs must be exact non-bool "
                "integers."
            )
        owners = _nodes_resolving_component_id(
            kernel.graph,
            parameter.owner_component_id,
        )
        if len(owners) != 1:
            raise ValueError(
                f"KernelIR parameter '{parameter.name}' owner component id "
                f"{parameter.owner_component_id} must resolve to exactly one "
                "GraphIR node."
            )
        if (
            type(parameter.owner_scope) is not str
            or parameter.owner_scope not in {"function", "mechanism"}
        ):
            raise ValueError(
                f"KernelIR parameter '{parameter.name}' owner_scope must be "
                "'function' or 'mechanism'."
            )


def _validate_kernel_node_implementations(kernel: KernelIR) -> None:
    """Tie executable node declarations to frozen implementations and params."""

    if (
        not kernel.graph.executable
        or kernel.graph.metadata.get("schedule_kind") != "dynamic_lane_local"
    ):
        return

    parameters_by_name = {parameter.name: parameter for parameter in kernel.params}
    ports_by_id = {port.port_id: port for port in kernel.ports}
    for node in kernel.graph.nodes:
        spec_kind = node.attrs.get("spec_kind")
        spec_key = node.attrs.get("spec_key", "")
        if node.component_type == "ControlMechanism":
            # Absorbed controllers have a separate identity/binding validator
            # because the exact Identity form deliberately has no registry key.
            # Executable coevolution additionally authenticates its folded
            # threshold controller as part of the complete schedule boundary.
            continue
        if (
            spec_kind not in {"elementwise", "mechanism"}
            or type(spec_key) is not str
            or not spec_key
            or not isinstance(node.params, Mapping)
        ):
            raise ValueError(
                f"KernelIR executable node '{node.name}' requires a typed frozen "
                "implementation declaration."
            )
        try:
            implementation = kernel.op_specs.lookup_spec(spec_key)
        except BatchedOpSpecError as error:
            raise ValueError(
                f"KernelIR executable node '{node.name}' has no frozen "
                "implementation."
            ) from error

        if spec_kind == "elementwise":
            implementation_matches = bool(
                isinstance(implementation, ElementwiseFunctionSpec)
                and implementation.function_class.__name__ == node.function_type
            )
        else:
            unsuffixed_name = re.sub(r"-\d+$", "", node.name)
            instance_implementation = bool(
                isinstance(implementation, MechanismOpSpec)
                and implementation.mechanism_class is None
                and implementation.function_class is None
                and implementation.display_name == unsuffixed_name
                and spec_key == f"instance:{unsuffixed_name}"
            )
            class_implementation = bool(
                isinstance(implementation, MechanismOpSpec)
                and implementation.mechanism_class is not None
                and implementation.mechanism_class.__name__
                == node.component_type
                and (
                    implementation.function_class is None
                    or implementation.function_class.__name__
                    == node.function_type
                )
            )
            implementation_matches = bool(
                instance_implementation or class_implementation
            )
        bindings = implementation.params if implementation_matches else ()
        expected_arguments = tuple(binding.arg for binding in bindings)
        input_ports = tuple(
            ports_by_id.get(port_id) for port_id in node.input_port_ids
        )
        output_ports = tuple(
            ports_by_id.get(port_id) for port_id in node.output_port_ids
        )
        if isinstance(implementation, MechanismOpSpec) and implementation.outputs:
            expected_op_outputs = tuple(
                (declaration.port, declaration.width)
                for declaration in implementation.outputs
            )
            output_shape_matches = bool(
                node.output_width == implementation.outputs[0].width
                and _kernel_attribute_matches_exactly(
                    node.attrs.get("op_outputs"),
                    expected_op_outputs,
                )
            )
        elif output_ports:
            output_shape_matches = bool(
                node.output_width == output_ports[0].width
                if isinstance(implementation, MechanismOpSpec)
                else node.output_width == node.input_width
            )
        else:
            output_shape_matches = False
        if (
            not implementation_matches
            or tuple(node.params) != expected_arguments
            or any(
                type(parameter_name) is not str or not parameter_name
                for parameter_name in node.params.values()
            )
            or type(node.input_width) is not int
            or node.input_width <= 0
            or any(port is None for port in input_ports)
            or node.input_width != sum(port.width for port in input_ports)
            or type(node.output_width) is not int
            or node.output_width <= 0
            or any(port is None for port in output_ports)
            or not output_shape_matches
        ):
            raise ValueError(
                f"KernelIR executable node '{node.name}' implementation and "
                "parameter/shape signature must exactly match its frozen spec."
            )
        for binding in bindings:
            parameter = parameters_by_name.get(node.params[binding.arg])
            if (
                parameter is None
                or parameter.owner_component_id != node.component_id
                or parameter.owner_scope != binding.scope
            ):
                raise ValueError(
                    f"KernelIR executable node '{node.name}' parameter "
                    f"'{binding.arg}' must resolve to its exact owned binding."
                )


def _validate_kernel_ports(kernel: KernelIR) -> None:
    """Validate the complete typed port inventory for every KernelIR shape."""

    port_ids = tuple(port.port_id for port in kernel.ports)
    if any(type(port_id) is not int for port_id in port_ids):
        raise ValueError("KernelIR port IDs must be exact non-bool integers.")
    if port_ids != tuple(range(len(kernel.ports))):
        raise ValueError(
            "KernelIR port IDs must be unique, contiguous, and in declaration "
            "order."
        )

    identities = []
    supported_kinds = {"InputPort", "OutputPort", "ParameterPort", "ControlSignal"}
    for port in kernel.ports:
        if (
            type(port.name) is not str
            or not port.name
            or type(port.owner) is not str
            or not port.owner
            or type(port.owner_component_id) is not int
            or type(port.kind) is not str
            or port.kind not in supported_kinds
            or type(port.width) is not int
            or port.width <= 0
        ):
            raise ValueError(
                "KernelIR ports require nonempty labels, exact owner IDs, a "
                "supported port kind, and a positive non-bool width."
            )
        owners = tuple(
            node
            for node in _nodes_resolving_component_id(
                kernel.graph,
                port.owner_component_id,
            )
            if node.name == port.owner
        )
        if len(owners) != 1:
            raise ValueError(
                f"KernelIR port '{port.name}' owner name and component id must "
                "resolve to exactly one GraphIR node."
            )
        identities.append((port.owner_component_id, port.kind, port.name))
    if len(set(identities)) != len(identities):
        raise ValueError(
            "KernelIR ports must have unique (owner component, kind, name) "
            "identities."
        )

    ports_by_id = {port.port_id: port for port in kernel.ports}
    for node in kernel.graph.nodes:
        component_id = _component_id(kernel.graph, node)
        owned_ports = tuple(
            port
            for port in kernel.ports
            if port.owner_component_id == component_id
        )
        anchors_present = bool(
            node.input_port_ids
            or node.output_port_ids
            or node.parameter_port_ids
        )
        # Hand-built declaration-only legacy fixtures may omit the global port
        # inventory entirely.  Once either side declares ports, require an
        # exact component-local anchor rather than trusting free-form labels.
        if not owned_ports and not anchors_present:
            continue
        expected_input_ids = tuple(
            port.port_id for port in owned_ports if port.kind == "InputPort"
        )
        expected_output_ids = tuple(
            port.port_id
            for port in owned_ports
            if port.kind in {"OutputPort", "ControlSignal"}
        )
        expected_parameter_ids = tuple(
            (port.name, port.port_id)
            for port in owned_ports
            if port.kind == "ParameterPort"
        )
        if (
            type(node.input_port_ids) is not tuple
            or type(node.output_port_ids) is not tuple
            or type(node.parameter_port_ids) is not tuple
            or node.input_port_ids != expected_input_ids
            or node.output_port_ids != expected_output_ids
            or node.parameter_port_ids != expected_parameter_ids
            or any(port_id not in ports_by_id for port_id in node.input_port_ids)
            or any(port_id not in ports_by_id for port_id in node.output_port_ids)
        ):
            raise ValueError(
                f"KernelIR node '{node.name}' port anchors must exactly match "
                "its ordered typed port inventory."
            )


def _validate_kernel_projections(kernel: KernelIR) -> None:
    """Authenticate ordinary dense projection endpoint identities and shapes."""

    if not _has_typed_port_inventory(kernel):
        return

    projection_ids = tuple(
        projection.projection_id for projection in kernel.graph.projections
    )
    if (
        any(type(projection_id) is not int for projection_id in projection_ids)
        or projection_ids != tuple(range(len(kernel.graph.projections)))
    ):
        raise ValueError(
            "KernelIR ordinary projection IDs must be exact, unique, contiguous "
            "non-bool integers in declaration order."
        )

    ports_by_id = {port.port_id: port for port in kernel.ports}
    for projection in kernel.graph.projections:
        ids = (
            projection.sender_component_id,
            projection.sender_port_id,
            projection.receiver_component_id,
            projection.receiver_port_id,
        )
        sender = _exact_graph_node(
            kernel,
            node=projection.sender,
            component_id=projection.sender_component_id,
        )
        receiver = _exact_graph_node(
            kernel,
            node=projection.receiver,
            component_id=projection.receiver_component_id,
        )
        sender_port = ports_by_id.get(projection.sender_port_id)
        receiver_port = ports_by_id.get(projection.receiver_port_id)
        matrix = projection.matrix
        try:
            implementation = kernel.op_specs.lookup_spec(projection.spec_key)
        except BatchedOpSpecError:
            implementation = None
        if (
            any(type(value) is not int or value < 0 for value in ids)
            or sender is None
            or receiver is None
            or sender_port is None
            or sender_port.kind != "OutputPort"
            or sender_port.owner != projection.sender
            or sender_port.owner_component_id
            != projection.sender_component_id
            or sender_port.name != projection.sender_port
            or receiver_port is None
            or receiver_port.kind != "InputPort"
            or receiver_port.owner != projection.receiver
            or receiver_port.owner_component_id
            != projection.receiver_component_id
            or receiver_port.name != projection.receiver_port
            or type(projection.spec_key) is not str
            or not projection.spec_key
            or not isinstance(implementation, DenseProjectionSpec)
            or implementation.projection_class.__name__ != "MappingProjection"
            or not callable(implementation.triton_emit)
            or type(matrix) is not np.ndarray
            or matrix.dtype != np.dtype(np.float32)
            or matrix.shape != (sender_port.width, receiver_port.width)
            or not bool(np.all(np.isfinite(matrix)))
        ):
            raise ValueError(
                "KernelIR ordinary projection must match exact GraphIR node/port "
                "ownership and a finite dense float32 matrix shape."
            )


def _validate_kernel_states(kernel: KernelIR) -> None:
    """Authenticate retained state against frozen mechanism declarations."""

    if not _has_typed_port_inventory(kernel):
        return

    state_ids = tuple(state.state_id for state in kernel.states)
    if (
        any(type(state_id) is not int for state_id in state_ids)
        or state_ids != tuple(range(len(kernel.states)))
    ):
        raise ValueError(
            "KernelIR retained-state IDs must be exact, unique, contiguous "
            "non-bool integers in declaration order."
        )

    # Numeric-noise LCA construction has one semantic bit that is not implied
    # by its numeric state initializer: a Never-reset LCA exposes
    # Logistic(noise * sqrt(dt)) as its initial recurrent sender, whereas an
    # AtTrialStart reset replaces that sender with Logistic(initializer).
    # Authenticate the bit for every executable LCA fusion, including ordinary
    # stateful graphs that do not use the dynamic-pass validator below.
    for node in kernel.graph.nodes:
        spec_key = node.attrs.get("spec_key")
        if type(spec_key) is not str or not spec_key:
            continue
        try:
            implementation = kernel.op_specs.lookup_spec(spec_key)
        except BatchedOpSpecError:
            continue
        if not (
            isinstance(implementation, MechanismOpSpec)
            and implementation.mechanism_class is not None
            and implementation.mechanism_class.__name__ == "LCAMechanism"
            and tuple(state.name for state in implementation.states)
            == ("pre", "act", "initialized")
        ):
            continue
        resets = tuple(
            reset
            for reset in kernel.resets
            if reset.component_id == node.component_id
        )
        if (
            len(resets) != 1
            or resets[0].node != node.name
            or resets[0].condition_type not in {"Never", "AtTrialStart"}
        ):
            raise ValueError(
                f"KernelIR numeric-noise LCA '{node.name}' requires one exact "
                "Never or AtTrialStart reset declaration."
            )
        expected = resets[0].condition_type == "Never"
        if node.attrs.get("initialize_noise_sender") is not expected:
            raise ValueError(
                f"KernelIR numeric-noise LCA '{node.name}' initialization "
                "policy must exactly match its reset declaration."
            )

    # Deep mechanism-declaration authentication is currently part of the
    # executable controlled-pass boundary.  Legacy hand-built KernelIR
    # fixtures may intentionally omit frozen mechanism keys; their state
    # identity/effect invariants remain covered by the complete validator.
    if kernel.graph.metadata.get("schedule_kind") != "dynamic_lane_local":
        return

    expected = []
    for node in kernel.graph.nodes:
        spec_key = node.attrs.get("spec_key")
        if type(spec_key) is not str or not spec_key:
            continue
        try:
            spec = kernel.op_specs.lookup_spec(spec_key)
        except BatchedOpSpecError:
            continue
        if not isinstance(spec, MechanismOpSpec):
            continue
        expected.extend((node, declaration) for declaration in spec.states)
    if len(kernel.states) != len(expected):
        raise ValueError(
            "KernelIR retained-state declarations must form an exact bijection "
            "with frozen mechanism state declarations."
        )

    for state, (node, declaration) in zip(kernel.states, expected):
        width = declaration.width or node.output_width
        initial_value = tuple(declaration.initial for _ in range(width))
        if (
            type(state.component_id) is not int
            or state.component_id != node.component_id
            or type(state.node) is not str
            or state.node != node.name
            or type(state.name) is not str
            or state.name != f"{node.name}.{declaration.name}"
            or type(state.width) is not int
            or state.width != width
            or not _kernel_attribute_matches_exactly(
                state.initial_value,
                initial_value,
            )
        ):
            raise ValueError(
                "KernelIR retained state does not exactly match its frozen "
                f"mechanism declaration for '{node.name}.{declaration.name}'."
            )

        initializer = state.function_initializer
        if not declaration.initialize_with_function:
            if initializer is not None:
                raise ValueError(
                    "KernelIR retained state has an undeclared function "
                    f"initializer for '{state.name}'."
                )
            continue
        if type(initializer) is not BatchedStateFunctionInitializer:
            raise ValueError(
                "KernelIR retained state requires its typed function "
                f"initializer for '{state.name}'."
            )
        try:
            initializer_spec = kernel.op_specs.lookup_spec(initializer.spec_key)
        except BatchedOpSpecError as error:
            raise ValueError(
                "KernelIR retained-state initializer has no frozen "
                f"implementation for '{state.name}'."
            ) from error
        expected_params = (
            {
                binding.arg: node.params[binding.arg]
                for binding in initializer_spec.params
            }
            if isinstance(initializer_spec, ElementwiseFunctionSpec)
            and all(binding.arg in node.params for binding in initializer_spec.params)
            else None
        )
        if (
            not isinstance(initializer_spec, ElementwiseFunctionSpec)
            or initializer_spec.function_class.__name__ != node.function_type
            or not _kernel_attribute_matches_exactly(
                initializer.input_value,
                initial_value,
            )
            or expected_params is None
            or not _kernel_attribute_matches_exactly(
                initializer.params,
                expected_params,
            )
        ):
            raise ValueError(
                "KernelIR retained-state function initializer does not exactly "
                f"match '{state.name}'."
            )


def _has_typed_port_inventory(kernel: KernelIR) -> bool:
    """Whether this KernelIR uses the evolved, identity-bearing Port schema."""

    return bool(
        kernel.ports
        or any(
            node.input_port_ids
            or node.output_port_ids
            or node.parameter_port_ids
            for node in kernel.graph.nodes
        )
        or (
            kernel.executable
            and bool(kernel.inputs or kernel.outputs)
        )
    )


def _exact_graph_node(kernel: KernelIR, *, node: str, component_id: int):
    if type(node) is not str or not node or type(component_id) is not int:
        return None
    matches = tuple(
        candidate
        for candidate in _nodes_resolving_component_id(
            kernel.graph,
            component_id,
        )
        if candidate.name == node
    )
    return matches[0] if len(matches) == 1 else None


def _validate_kernel_io_declarations(kernel: KernelIR) -> None:
    """Tie public input/output records to the complete typed Port inventory."""

    # Preserve direct construction of the original experimental dataclasses,
    # whose legacy records have no numeric Port inventory.  Composition
    # lowering always emits that inventory, and any partially typed graph is
    # rejected by ``_validate_kernel_ports`` before reaching this boundary.
    if not _has_typed_port_inventory(kernel):
        return

    ports_by_id = {port.port_id: port for port in kernel.ports}
    input_port_ids = []
    input_component_ids = []
    input_names = []
    projected_receiver_port_ids = {
        projection.receiver_port_id for projection in kernel.graph.projections
    }
    for input_spec in kernel.inputs:
        node = _exact_graph_node(
            kernel,
            node=input_spec.node,
            component_id=input_spec.component_id,
        )
        port = ports_by_id.get(input_spec.port_id)
        expected_name = (
            input_spec.node
            if node is not None and len(node.input_port_ids) == 1
            else f"{input_spec.node}.{input_spec.port}"
        )
        if (
            node is None
            or type(input_spec.port_id) is not int
            or type(input_spec.width) is not int
            or input_spec.width <= 0
            or type(input_spec.port) is not str
            or not input_spec.port
            or type(input_spec.name) is not str
            or not input_spec.name
            or port is None
            or port.kind != "InputPort"
            or port.owner != input_spec.node
            or port.owner_component_id != input_spec.component_id
            or port.name != input_spec.port
            or port.width != input_spec.width
            or input_spec.port_id not in node.input_port_ids
            or input_spec.port_id in projected_receiver_port_ids
            or input_spec.name != expected_name
        ):
            raise ValueError(
                f"KernelIR input '{input_spec.name}' must match its exact typed "
                "GraphIR node and external InputPort identity, width, and label."
            )
        input_port_ids.append(input_spec.port_id)
        input_component_ids.append(input_spec.component_id)
        input_names.append(input_spec.name)
    if (
        len(set(input_port_ids)) != len(input_port_ids)
        or len(set(input_component_ids)) != len(input_component_ids)
        or len(set(input_names)) != len(input_names)
    ):
        raise ValueError(
            "KernelIR inputs must have unique external Port, component, and "
            "public-name identities."
        )
    expected_external_port_ids = tuple(
        port_id
        for node in kernel.graph.nodes
        if node.component_type != "ControlMechanism"
        for port_id in node.input_port_ids
        if port_id not in projected_receiver_port_ids
    )
    if kernel.executable and tuple(input_port_ids) != expected_external_port_ids:
        raise ValueError(
            "KernelIR inputs must cover every external typed InputPort exactly "
            "once and in GraphIR declaration order."
        )

    expected_output_names = []
    flat_cursor = 0
    for output in kernel.outputs:
        node = _exact_graph_node(
            kernel,
            node=output.node,
            component_id=output.component_id,
        )
        port = ports_by_id.get(output.port_id)
        expected_name = f"{output.node}.{output.port}"
        if (
            node is None
            or type(output.port_id) is not int
            or type(output.width) is not int
            or output.width <= 0
            or type(output.port) is not str
            or not output.port
            or type(output.name) is not str
            or not output.name
            or type(output.flat_start) is not int
            or type(output.flat_stop) is not int
            or port is None
            or port.kind != "OutputPort"
            or port.owner != output.node
            or port.owner_component_id != output.component_id
            or port.name != output.port
            or port.width != output.width
            or output.port_id not in node.output_port_ids
            or output.name != expected_name
        ):
            raise ValueError(
                f"KernelIR output '{output.name}' must match its exact typed "
                "GraphIR node and OutputPort identity, width, and label."
            )
        if (
            output.flat_start != flat_cursor
            or output.flat_stop != flat_cursor + output.width
        ):
            raise ValueError(
                "KernelIR output flattened slices must be contiguous, ordered, "
                "non-overlapping, and match each declared width."
            )
        flat_cursor = output.flat_stop
        expected_output_names.append(output.name)

    if (
        type(kernel.output_names) is not tuple
        or kernel.output_names != tuple(expected_output_names)
        or any(type(name) is not str for name in kernel.output_names)
    ):
        raise ValueError(
            "KernelIR output_names must exactly match declared outputs in "
            "flattened buffer order."
        )


def _expected_input_load(kernel: KernelIR, input_spec: BatchedInputSpec):
    node = _exact_graph_node(
        kernel,
        node=input_spec.node,
        component_id=input_spec.component_id,
    )
    assert node is not None
    ports_by_id = {port.port_id: port for port in kernel.ports}
    port_slot = node.input_port_ids.index(input_spec.port_id)
    flat_start = sum(
        ports_by_id[port_id].width
        for port_id in node.input_port_ids[:port_slot]
    )
    flat_stop = flat_start + input_spec.width
    value_name = (
        node_input_value_name(kernel.graph, node)
        if len(node.input_port_ids) == 1
        else f"{component_symbol(kernel.graph, node)}:input-port:{port_slot}"
    )
    return (
        node,
        KernelValue(value_name, input_spec.width),
        {
            "node": input_spec.node,
            "input_name": input_spec.name,
            "width": input_spec.width,
            "component_id": input_spec.component_id,
            "port": input_spec.port,
            "port_id": input_spec.port_id,
            "flat_start": flat_start,
            "flat_stop": flat_stop,
        },
    )


def _attrs_match_exactly(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Mapping equality that does not let bool values alias integer IDs."""

    if set(actual) != set(expected):
        return False
    return all(
        type(actual[key]) is type(value) and actual[key] == value
        for key, value in expected.items()
    )


def _kernel_value_matches(actual: KernelValue, expected: KernelValue) -> bool:
    return bool(
        type(actual) is KernelValue
        and type(actual.name) is str
        and actual.name == expected.name
        and type(actual.width) is int
        and actual.width == expected.width
        and type(actual.dtype) is str
        and actual.dtype == expected.dtype
    )


def _kernel_value_key(value: KernelValue) -> tuple[str, int, str]:
    return (value.name, value.width, value.dtype)


def _kernel_output_port(kernel: KernelIR, node, *, name: str, port_id: int | None = None):
    matches = tuple(
        port
        for port in kernel.ports
        if port.owner_component_id == _component_id(kernel.graph, node)
        and port.kind == "OutputPort"
        and port.name == name
        and (port_id is None or port.port_id == port_id)
    )
    return matches[0] if len(matches) == 1 else None


def _kernel_op_output_lineage(
    kernel: KernelIR,
    op: KernelOp,
    value_lineage: Mapping[
        tuple[str, int, str],
        tuple[frozenset[int], frozenset[int]],
    ],
) -> dict[tuple[str, int, str], tuple[frozenset[int], frozenset[int]]]:
    """Track exact component and OutputPort provenance through typed ops."""

    empty = (frozenset(), frozenset())
    output_lineage = {_kernel_value_key(value): empty for value in op.outputs}
    try:
        node = kernel.graph.node(op.target)
    except KeyError:
        return output_lineage
    component_id = _component_id(kernel.graph, node)

    if op.kind in {"CallFunction", "CallMechanism", "StepMechanism"}:
        for index, value in enumerate(op.outputs):
            port = None
            if op.kind == "CallFunction" and len(op.outputs) == 1:
                port_name = op.attrs.get("output_port")
                if type(port_name) is str:
                    port = _kernel_output_port(kernel, node, name=port_name)
            else:
                op_outputs = tuple(node.attrs.get("op_outputs", ()))
                if index < len(op_outputs):
                    port_name, port_width = op_outputs[index]
                    if value.width == port_width:
                        port = _kernel_output_port(kernel, node, name=port_name)
            port_ids = (
                frozenset((port.port_id,))
                if port is not None and value.width == port.width
                else frozenset()
            )
            output_lineage[_kernel_value_key(value)] = (
                frozenset((component_id,)),
                port_ids,
            )
        if op.kind == "CallMechanism":
            for diagnostic_value in op.attrs.get("diagnostic_values", ()):
                if type(diagnostic_value) is str and diagnostic_value:
                    output_lineage[(diagnostic_value, 1, "float32")] = (
                        frozenset((component_id,)),
                        frozenset(),
                    )
        return output_lineage

    if op.kind not in {"AddConstant", "Clamp", "ExtractSlice"}:
        return output_lineage

    input_components = frozenset().union(
        *(value_lineage.get(_kernel_value_key(value), empty)[0] for value in op.inputs)
    )
    input_ports = frozenset().union(
        *(value_lineage.get(_kernel_value_key(value), empty)[1] for value in op.inputs)
    )
    if component_id not in input_components:
        return output_lineage

    if op.kind == "ExtractSlice":
        port_id = op.attrs.get("port_id")
        port_name = op.attrs.get("port")
        port = (
            _kernel_output_port(
                kernel,
                node,
                name=port_name,
                port_id=port_id,
            )
            if type(port_name) is str and type(port_id) is int
            else None
        )
        matching_slice = next(
            (
                item
                for item in _node_output_port_slices(node)
                if item
                == (
                    port_name,
                    port.width if port is not None else -1,
                    port_id,
                    op.attrs.get("start"),
                    op.attrs.get("stop"),
                )
            ),
            None,
        )
        propagated_ports = (
            frozenset((port.port_id,))
            if port is not None and matching_slice is not None
            else frozenset()
        )
    else:
        propagated_ports = input_ports

    propagated = (input_components, propagated_ports)
    return {_kernel_value_key(value): propagated for value in op.outputs}


def _validate_executable_kernel_io_ops(
    kernel: KernelIR,
    *,
    input_loads: list[KernelOp],
    output_stores: list[KernelOp],
    lineage_at_store: Mapping[
        int,
        tuple[frozenset[int], frozenset[int]],
    ],
) -> None:
    """Require executable host-buffer effects to implement typed IO exactly."""

    if not _has_typed_port_inventory(kernel):
        return

    inputs_by_port_id = {input_spec.port_id: input_spec for input_spec in kernel.inputs}
    actual_load_counts = {port_id: 0 for port_id in inputs_by_port_id}
    for load in input_loads:
        port_id = load.attrs.get("port_id")
        input_spec = inputs_by_port_id.get(port_id)
        if input_spec is None:
            raise ValueError(
                "Executable KernelIR LoadInput references no declared external "
                "InputPort."
            )
        node, expected_output, expected_attrs = _expected_input_load(
            kernel,
            input_spec,
        )
        if (
            load.target != node.name
            or load.inputs
            or len(load.outputs) != 1
            or not _kernel_value_matches(load.outputs[0], expected_output)
            or not _attrs_match_exactly(load.attrs, expected_attrs)
        ):
            raise ValueError(
                f"Executable KernelIR LoadInput does not exactly match declared "
                f"input '{input_spec.name}'."
            )
        actual_load_counts[port_id] += 1

    expected_load_counts = {
        input_spec.port_id: (
            sum(
                input_spec.component_id in step.component_ids
                for step in kernel.schedule_trace.steps
            )
            if kernel.schedule_trace is not None
            else 1
        )
        for input_spec in kernel.inputs
    }
    if actual_load_counts != expected_load_counts:
        raise ValueError(
            "Executable KernelIR must contain the exact scheduled LoadInput "
            "count for every declared external InputPort."
        )

    if len(output_stores) != len(kernel.outputs):
        raise ValueError(
            "Executable KernelIR must contain exactly one StoreOutput for each "
            "declared output."
        )
    for store, output in zip(output_stores, kernel.outputs):
        expected_attrs = {
            "node": output.node,
            "port": output.port,
            "width": output.width,
            "component_id": output.component_id,
            "port_id": output.port_id,
            "flat_start": output.flat_start,
            "flat_stop": output.flat_stop,
        }
        stored_value = store.inputs[0] if len(store.inputs) == 1 else None
        stored_value_is_typed = bool(
            type(stored_value) is KernelValue
            and type(stored_value.name) is str
            and stored_value.name
            and type(stored_value.width) is int
            and stored_value.width == output.width
            and type(stored_value.dtype) is str
            and stored_value.dtype == "float32"
        )
        component_lineage, port_lineage = lineage_at_store.get(
            id(store),
            (frozenset(), frozenset()),
        )
        if (
            store.target != output.name
            or not stored_value_is_typed
            or output.component_id not in component_lineage
            or output.port_id not in port_lineage
            or store.outputs
            or not _attrs_match_exactly(store.attrs, expected_attrs)
        ):
            raise ValueError(
                "Executable KernelIR StoreOutput does not exactly match its "
                f"declared output '{output.name}'."
            )


def _validate_kernel_finished_scheduler(kernel: KernelIR) -> None:
    """Validate scheduler-visible finished values and their typed references."""

    if kernel.scheduler:
        scheduler_component_ids = tuple(
            condition.component_id for condition in kernel.scheduler
        )
        graph_component_ids = tuple(
            _component_id(kernel.graph, node) for node in kernel.graph.nodes
        )
        if (
            len(set(scheduler_component_ids)) != len(scheduler_component_ids)
            or set(scheduler_component_ids) != set(graph_component_ids)
        ):
            raise ValueError(
                "KernelIR scheduler declarations must contain exactly one "
                "condition for every GraphIR node."
            )

    value_ids = tuple(value.value_id for value in kernel.finished_values)
    if any(type(value_id) is not int for value_id in value_ids):
        raise ValueError(
            "KernelIR finished-value IDs must be exact non-bool integers."
        )
    if value_ids != tuple(range(len(kernel.finished_values))):
        raise ValueError(
            "KernelIR finished-value IDs must be unique, contiguous, and in "
            "declaration order."
        )

    finished_components = []
    for value in kernel.finished_values:
        if (
            type(value.component_id) is not int
            or type(value.producer_consideration_set_id) is not int
            or type(value.name) is not str
            or type(value.node) is not str
            or value.name != f"{value.node}.is_finished"
            or value.width != 1
            or type(value.width) is not int
            or value.dtype != "bool"
            or value.storage != "combinational"
        ):
            raise ValueError(
                "KernelIR finished values require exact node/component/set "
                "identity and scalar bool combinational storage."
            )
        owners = tuple(
            node
            for node in _nodes_resolving_component_id(kernel.graph, value.component_id)
            if node.name == value.node
        )
        producer_sets = tuple(
            consideration_set
            for consideration_set in kernel.consideration_sets
            if consideration_set.consideration_set_id
            == value.producer_consideration_set_id
            and _consideration_set_owns(
                consideration_set,
                value.node,
                value.component_id,
            )
        )
        if len(owners) != 1 or len(producer_sets) != 1:
            raise ValueError(
                f"KernelIR finished value '{value.name}' must resolve to its "
                "exact GraphIR node and producer consideration set."
            )
        finished_components.append(value.component_id)
    if len(set(finished_components)) != len(finished_components):
        raise ValueError(
            "KernelIR finished values must declare at most one value per "
            "producer component."
        )

    referenced_value_ids = set()
    for condition in kernel.scheduler:
        if (
            type(condition.component_id) is not int
            or type(condition.consideration_set_id) is not int
        ):
            raise ValueError(
                "KernelIR scheduler component and consideration-set IDs must be "
                "exact non-bool integers."
            )
        targets = tuple(
            node
            for node in _nodes_resolving_component_id(
                kernel.graph,
                condition.component_id,
            )
            if node.name == condition.node
        )
        target_sets = tuple(
            consideration_set
            for consideration_set in kernel.consideration_sets
            if consideration_set.consideration_set_id
            == condition.consideration_set_id
            and _consideration_set_owns(
                consideration_set,
                condition.node,
                condition.component_id,
            )
        )
        if len(targets) != 1 or len(target_sets) != 1:
            raise ValueError(
                f"KernelIR scheduler condition for '{condition.node}' must "
                "resolve to its exact GraphIR node and consideration set."
            )
        if (
            type(condition.dependencies) is not tuple
            or type(condition.dependency_component_ids) is not tuple
            or len(condition.dependencies) != len(condition.dependency_component_ids)
            or any(
                type(component_id) is not int
                for component_id in condition.dependency_component_ids
            )
        ):
            raise ValueError(
                "KernelIR scheduler dependencies require parallel name and exact "
                "non-bool component-ID tuples."
            )
        for dependency, component_id in zip(
            condition.dependencies,
            condition.dependency_component_ids,
        ):
            matches = tuple(
                node
                for node in _nodes_resolving_component_id(kernel.graph, component_id)
                if node.name == dependency
            )
            if len(matches) != 1:
                raise ValueError(
                    f"KernelIR scheduler dependency '{dependency}' does not "
                    "match component id {component_id}."
                )

        if type(condition.finished_value_ids) is not tuple:
            raise ValueError(
                "KernelIR scheduler finished-value IDs must be a typed tuple."
            )
        if condition.condition_type != "WhenFinished":
            if condition.finished_value_ids:
                raise ValueError(
                    "KernelIR finished-value IDs are valid only on WhenFinished "
                    "scheduler conditions."
                )
            continue
        if any(
            type(value_id) is not int
            for value_id in condition.finished_value_ids
        ):
            raise ValueError(
                "KernelIR WhenFinished finished-value IDs must be exact "
                "non-bool integers."
            )
        if (
            not condition.dependencies
            or len(condition.finished_value_ids) != len(condition.dependencies)
            or len(set(condition.finished_value_ids))
            != len(condition.finished_value_ids)
        ):
            raise ValueError(
                "KernelIR WhenFinished dependencies and finished-value IDs must "
                "form parallel unique tuples."
            )
        for component_id, finished_value_id in zip(
            condition.dependency_component_ids,
            condition.finished_value_ids,
        ):
            matches = tuple(
                value
                for value in kernel.finished_values
                if value.value_id == finished_value_id
                and value.component_id == component_id
            )
            if len(matches) != 1:
                raise ValueError(
                    "KernelIR WhenFinished reference does not match its declared "
                    "dependency finished value."
                )
            referenced_value_ids.add(finished_value_id)

    if referenced_value_ids != set(value_ids):
        raise ValueError(
            "KernelIR WhenFinished references and finished-value declarations "
            "must form an exact component-wise bijection."
        )


def _validate_kernel_reset_declarations(kernel: KernelIR) -> None:
    """Require a canonical reset declaration for every retained-state owner."""

    if kernel.lane_layout.kind != STATEFUL_LANE_LAYOUT:
        if kernel.resets:
            raise ValueError(
                "KernelIR stateless lane layouts cannot retain reset "
                "declarations."
            )
        return

    states_by_component: dict[int, list[BatchedStateSpec]] = {}
    for state in kernel.states:
        component_id = _state_component_id(kernel.graph, state)
        states_by_component.setdefault(component_id, []).append(state)

    component_ids = tuple(states_by_component)
    reset_component_ids = tuple(reset.component_id for reset in kernel.resets)
    if reset_component_ids != component_ids:
        raise ValueError(
            "KernelIR stateful reset declarations must contain exactly one "
            "entry for every retained-state owner, in state declaration order."
        )

    for reset in kernel.resets:
        try:
            node = next(
                node
                for node in kernel.graph.nodes
                if _component_id(kernel.graph, node) == reset.component_id
            )
        except StopIteration as error:
            raise ValueError(
                "KernelIR reset declaration references an undeclared component "
                f"id {reset.component_id}."
            ) from error
        expected_state_ids = tuple(
            state.state_id
            for state in states_by_component[reset.component_id]
        )
        if (
            reset.node != node.name
            or reset.condition_type not in {"Never", "AtTrialStart"}
            or reset.state_ids != expected_state_ids
            or reset.attrs
            or reset.region != "trial"
        ):
            raise ValueError(
                "KernelIR reset declaration does not exactly match retained "
                f"state owner '{node.name}'."
            )


def _validate_kernel_modulations(kernel: KernelIR) -> None:
    """Validate the scalar ``OVERRIDE`` identity and execution boundary."""

    typed_controller_ids = {
        _component_id(kernel.graph, node)
        for node in kernel.graph.nodes
        if node.component_type == "ControlMechanism"
        and node.attrs.get("control_function") in {"identity", "registered"}
    }
    modulation_controller_ids = {
        modulation.controller_component_id for modulation in kernel.modulations
    }
    if typed_controller_ids != modulation_controller_ids:
        raise ValueError(
            "KernelIR typed ControlMechanism declarations and modulation "
            "effects must form an exact controller-component bijection."
        )

    dynamic_finished = tuple(
        value
        for value in kernel.finished_values
        if value.predicate_kind
        == "execution_count_at_least_effective_parameter"
    )
    if (
        not kernel.modulations
        and not kernel.effective_parameters
        and not dynamic_finished
        and not kernel.absorbed_projections
    ):
        return
    if kernel.executable:
        records = []

        def collect(ops: tuple[KernelOp, ...]) -> None:
            for op in ops:
                records.append(op)
                body = op.attrs.get("body", ())
                if type(body) is tuple:
                    collect(body)

        collect(kernel.ops)
        counted_regions = sum(
            op.kind == "ForPasses"
            and op.attrs.get("trace_kind") == "lane_local_counted"
            for op in records
        )
        coevolving_regions = sum(
            op.kind == "ForPasses"
            and op.attrs.get("trace_kind") == "lane_local_coevolving"
            for op in records
        )
        has_complete_schedule_effects = bool(
            (
                counted_regions == len(kernel.modulations)
                and coevolving_regions == 0
            )
            or (
                counted_regions == 0
                and coevolving_regions == 1
                and len(kernel.modulations) == 1
            )
        )
        if not (
            sum(op.kind == "InitializeEffectiveParameter" for op in records)
            == len(kernel.effective_parameters)
            and sum(op.kind == "ApplyModulation" for op in records)
            == len(kernel.modulations)
            and has_complete_schedule_effects
        ):
            raise ValueError(
                "Executable KernelIR modulation requires the complete typed "
                "effective-parameter and lane-local scheduler effect inventory."
            )

    modulation_ids = tuple(
        modulation.modulation_id for modulation in kernel.modulations
    )
    effective_parameter_ids = tuple(
        modulation.effective_parameter_id for modulation in kernel.modulations
    )
    declared_effective_ids = tuple(
        parameter.effective_parameter_id
        for parameter in kernel.effective_parameters
    )
    if modulation_ids != tuple(range(len(kernel.modulations))):
        raise ValueError(
            "KernelIR modulation IDs must be contiguous and in declaration order."
        )
    if len(set(effective_parameter_ids)) != len(effective_parameter_ids):
        raise ValueError(
            "KernelIR effective-parameter modulation IDs must be unique."
        )
    if (
        declared_effective_ids != tuple(range(len(kernel.effective_parameters)))
        or declared_effective_ids != effective_parameter_ids
        or len(dynamic_finished) != len(kernel.modulations)
        or len(kernel.absorbed_projections) != 2 * len(kernel.modulations)
    ):
        raise ValueError(
            "KernelIR modulation, held effective-parameter, and dynamic "
            "finished declarations must form an exact bijection."
        )

    ports_by_id = {port.port_id: port for port in kernel.ports}
    absorbed_by_id = {
        projection.projection_id: projection
        for projection in kernel.absorbed_projections
    }
    if tuple(sorted(absorbed_by_id)) != tuple(
        range(len(kernel.absorbed_projections))
    ):
        raise ValueError(
            "KernelIR absorbed-projection declarations must have unique "
            "contiguous IDs."
        )
    for projection in kernel.absorbed_projections:
        try:
            sender = kernel.graph.node(projection.sender)
            receiver = kernel.graph.node(projection.receiver)
        except KeyError as error:
            raise ValueError(
                "KernelIR absorbed projection references an undeclared component."
            ) from error
        sender_port = ports_by_id.get(projection.sender_port_id)
        receiver_port = ports_by_id.get(projection.receiver_port_id)
        if (
            _component_id(kernel.graph, sender) != projection.sender_component_id
            or _component_id(kernel.graph, receiver)
            != projection.receiver_component_id
            or sender_port is None
            or sender_port.owner_component_id != projection.sender_component_id
            or sender_port.name != projection.sender_port
            or sender_port.width != projection.width
            or receiver_port is None
            or receiver_port.owner_component_id
            != projection.receiver_component_id
            or receiver_port.name != projection.receiver_port
            or receiver_port.width != projection.width
        ):
            raise ValueError(
                "KernelIR absorbed projection endpoint identity does not match "
                "its typed port ownership."
            )

    params_by_id = {parameter.parameter_id: parameter for parameter in kernel.params}
    params_by_name = {parameter.name: parameter for parameter in kernel.params}
    scheduler_by_component_id = {
        condition.component_id: condition for condition in kernel.scheduler
    }
    effective_by_id = {
        parameter.effective_parameter_id: parameter
        for parameter in kernel.effective_parameters
    }
    target_port_ids = set()
    referenced_absorbed_projection_ids = []
    for modulation in kernel.modulations:
        try:
            controller = kernel.graph.node(modulation.controller)
            source = kernel.graph.node(modulation.source)
            target = kernel.graph.node(modulation.target)
        except KeyError as error:
            raise ValueError(
                "KernelIR modulation references an undeclared component."
            ) from error
        if (
            _component_id(kernel.graph, controller)
            != modulation.controller_component_id
            or _component_id(kernel.graph, source)
            != modulation.source_component_id
            or _component_id(kernel.graph, target)
            != modulation.target_component_id
            or controller.component_type != "ControlMechanism"
            or source.component_type
            not in {"TransferMechanism", "ProcessingMechanism"}
            or target.component_type != "LCAMechanism"
            or dict(controller.params)
            != {
                binding.argument: binding.parameter
                for binding in modulation.controller_param_bindings
            }
            or controller.attrs.get("spec_key", "")
            != modulation.controller_function_spec_key
        ):
            raise ValueError(
                "KernelIR modulation component identity or controller binding "
                "does not match GraphIR."
            )

        source_condition = scheduler_by_component_id.get(
            modulation.source_component_id
        )
        controller_condition = scheduler_by_component_id.get(
            modulation.controller_component_id
        )
        target_condition = scheduler_by_component_id.get(
            modulation.target_component_id
        )
        if (
            source_condition is None
            or controller_condition is None
            or target_condition is None
            or not (
                source_condition.consideration_set_id
                < controller_condition.consideration_set_id
                < target_condition.consideration_set_id
            )
            or target_condition.condition_type != "Always"
        ):
            raise ValueError(
                "KernelIR modulation requires strictly ordered source, "
                "controller, and Always-target consideration sets."
            )

        source_inputs = tuple(
            input_spec
            for input_spec in kernel.inputs
            if input_spec.component_id == modulation.source_component_id
        )
        source_spec_key = source.attrs.get("spec_key", "")
        try:
            source_implementation = kernel.op_specs.lookup_spec(source_spec_key)
        except BatchedOpSpecError as error:
            raise ValueError(
                "KernelIR modulation source must use its frozen registered "
                "identity Linear implementation."
            ) from error
        expected_source_defaults = {
            "slope": 1.0,
            "intercept": 0.0,
            "scale": 1.0,
            "offset": 0.0,
        }
        source_parameters = {
            argument: params_by_name.get(parameter_name)
            for argument, parameter_name in dict(source.params).items()
        }
        # The source schema is local typed data, not a whole-program capability
        # decision.  A declaration-only co-evolving graph may legitimately
        # carry this affine source even when another forged/unsupported edge
        # keeps the graph non-executable; final materialization separately
        # reauthenticates the complete graph boundary.
        coevolving_affine_source = (
            kernel.graph.fusion_kind == COEVOLVING_GRAPH_FUSION
        )
        if coevolving_affine_source:
            source_parameter_semantics_match = all(
                parameter is not None
                and type(parameter.default) is float
                and np.isfinite(parameter.default)
                and 0.0 <= parameter.default <= FP32_EXACT_INTEGER_LIMIT
                and parameter.default.is_integer()
                and parameter.runtime_mutable
                is (argument in {"slope", "intercept"})
                and parameter.owner_component_id
                == modulation.source_component_id
                for argument, parameter in source_parameters.items()
            )
        else:
            source_parameter_semantics_match = all(
                source_parameters[argument].default == expected_default
                and not source_parameters[argument].runtime_mutable
                and source_parameters[argument].owner_component_id
                == modulation.source_component_id
                for argument, expected_default in expected_source_defaults.items()
            ) if all(
                parameter is not None for parameter in source_parameters.values()
            ) else False
        if (
            source.function_type != "Linear"
            or source.attrs.get("spec_kind") != "elementwise"
            or source.attrs.get("noise") is not None
            or source.attrs.get("clip") is not None
            or source.attrs.get("integrator_pre") is not None
            or source.input_width != 1
            or source.output_width != 1
            or len(source.input_port_ids) != 1
            or not source.output_port_ids
            or source.output_port_ids[0] != modulation.source_port_id
            or len(source_inputs) != 1
            or source_inputs[0].port_id != source.input_port_ids[0]
            or source_inputs[0].width != 1
            or any(
                projection.receiver_component_id
                == modulation.source_component_id
                for projection in kernel.graph.projections
            )
            or not isinstance(source_implementation, ElementwiseFunctionSpec)
            or source_implementation.function_class.__name__ != "Linear"
            or set(source_parameters) != set(expected_source_defaults)
            or any(parameter is None for parameter in source_parameters.values())
            or not source_parameter_semantics_match
        ):
            raise ValueError(
                "KernelIR controlled-finished source must be the authenticated "
                "scalar Linear count origin declared by GraphIR."
            )
        endpoint_expectations = (
            (
                modulation.source_port_id,
                modulation.source_component_id,
                modulation.source_port,
                "OutputPort",
            ),
            (
                modulation.controller_input_port_id,
                modulation.controller_component_id,
                modulation.controller_input_port,
                "InputPort",
            ),
            (
                modulation.control_signal_port_id,
                modulation.controller_component_id,
                modulation.control_signal_port,
                "ControlSignal",
            ),
            (
                modulation.target_parameter_port_id,
                modulation.target_component_id,
                modulation.target_parameter,
                "ParameterPort",
            ),
        )
        for port_id, owner_id, port_name, port_kind in endpoint_expectations:
            port = ports_by_id.get(port_id)
            if (
                port is None
                or port.owner_component_id != owner_id
                or port.name != port_name
                or port.kind != port_kind
                or port.width != 1
            ):
                raise ValueError(
                    "KernelIR modulation endpoint port identity does not match "
                    "its declared owner, role, name, and width."
                )
        if modulation.target_parameter != "termination_threshold":
            raise ValueError(
                "KernelIR controlled finished modulation must target the LCA "
                "termination_threshold ParameterPort."
            )
        target_parameter_ports = dict(target.parameter_port_ids)
        if (
            target_parameter_ports.get("termination_threshold")
            != modulation.target_parameter_port_id
        ):
            raise ValueError(
                "KernelIR controlled-finished target must use the LCA node's "
                "canonical termination_threshold ParameterPort identity."
            )
        monitor_projection = absorbed_by_id.get(
            modulation.monitor_projection_id
        )
        control_projection = absorbed_by_id.get(
            modulation.control_projection_id
        )
        referenced_absorbed_projection_ids.extend(
            (
                modulation.monitor_projection_id,
                modulation.control_projection_id,
            )
        )
        if (
            monitor_projection is None
            or monitor_projection.kind != "MappingProjection"
            or monitor_projection.sender_component_id
            != modulation.source_component_id
            or monitor_projection.sender_port_id != modulation.source_port_id
            or monitor_projection.receiver_component_id
            != modulation.controller_component_id
            or monitor_projection.receiver_port_id
            != modulation.controller_input_port_id
            or control_projection is None
            or control_projection.kind != "ControlProjection"
            or control_projection.sender_component_id
            != modulation.controller_component_id
            or control_projection.sender_port_id
            != modulation.control_signal_port_id
            or control_projection.receiver_component_id
            != modulation.target_component_id
            or control_projection.receiver_port_id
            != modulation.target_parameter_port_id
        ):
            raise ValueError(
                "KernelIR modulation must exactly reference its typed absorbed "
                "monitor and ControlProjection routes."
            )
        parameter_bindings = modulation.controller_param_bindings
        implementation_bindings = {}
        if modulation.controller_function_spec_key:
            implementation = kernel.op_specs.lookup_spec(
                modulation.controller_function_spec_key
            )
            if (
                not isinstance(implementation, ElementwiseFunctionSpec)
                or controller.function_type
                != implementation.function_class.__name__
                or controller.attrs.get("spec_kind") != "control"
                or controller.attrs.get("control_function") != "registered"
            ):
                raise ValueError(
                    "KernelIR registered controller function identity must "
                    "match its frozen elementwise implementation."
                )
            implementation_bindings = {
                binding.arg: binding for binding in implementation.params
            }
            expected_arguments = tuple(
                binding.arg for binding in implementation.params
            )
            if tuple(binding.argument for binding in parameter_bindings) != (
                expected_arguments
            ):
                raise ValueError(
                    "KernelIR modulation controller bindings do not match its "
                    "frozen registered implementation signature."
                )
        elif (
            parameter_bindings
            or controller.function_type != "Identity"
            or controller.params
            or controller.attrs.get("spec_kind") != "control"
            or controller.attrs.get("control_function") != "identity"
            or controller.attrs.get("spec_key", "")
        ):
            raise ValueError(
                "KernelIR identity controller requires an exact Identity "
                "function with no registered implementation or parameters."
            )
        for binding in parameter_bindings:
            parameter = params_by_id.get(binding.parameter_id)
            if (
                parameter is None
                or parameter.name != binding.parameter
                or parameter.runtime_mutable
                or parameter.owner_component_id
                != modulation.controller_component_id
                or parameter.owner_scope
                != implementation_bindings[binding.argument].scope
            ):
                raise ValueError(
                    "KernelIR modulation controller parameter identity must "
                    "resolve to one frozen KernelIR parameter."
                )
        if modulation.target_parameter_port_id in target_port_ids:
            raise ValueError(
                "KernelIR supports at most one modulation for each target "
                "ParameterPort."
            )
        target_port_ids.add(modulation.target_parameter_port_id)

        held_parameter = effective_by_id[modulation.effective_parameter_id]
        expected_base_value = _normalize_constant(
            target.attrs.get("termination_threshold"),
            width=1,
            op_kind="EffectiveParameter",
            attr="base_value",
        )
        if (
            held_parameter.target != modulation.target
            or held_parameter.target_component_id
            != modulation.target_component_id
            or held_parameter.target_parameter != modulation.target_parameter
            or held_parameter.target_parameter_port_id
            != modulation.target_parameter_port_id
            or held_parameter.base_value != expected_base_value
            or control_projection.initial_value
            != held_parameter.initial_modulation_value
        ):
            raise ValueError(
                "KernelIR held effective-parameter target or initial values do "
                "not match its modulation edge and target declaration."
            )

        matches = tuple(
            value
            for value in dynamic_finished
            if value.component_id == modulation.target_component_id
        )
        expected_attrs = {
            "effective_parameter_id": modulation.effective_parameter_id,
            "target_parameter_port_id": modulation.target_parameter_port_id,
            "rounding": "ceil",
            "minimum": 1,
            "maximum": 2 ** 24,
        }
        if len(matches) != 1 or not _kernel_attribute_matches_exactly(
            dict(matches[0].attrs),
            expected_attrs,
        ):
            raise ValueError(
                "KernelIR modulation must own exactly one matching dynamic "
                "finished-value declaration."
            )
    if tuple(sorted(referenced_absorbed_projection_ids)) != tuple(
        range(len(kernel.absorbed_projections))
    ):
        raise ValueError(
            "KernelIR modulation routes must form an exact bijection with "
            "absorbed projection declarations."
        )


def _validate_dynamic_modulation_ops(kernel: KernelIR) -> None:
    """Authenticate the exact declaration or executable lane-local op suite.

    Partial plans remain non-executable and carry no inferred effects.  A
    complete canonical suite may execute only after the whole graph boundary
    and every placement/identity invariant below have also been authenticated.
    """

    records: list[tuple[KernelOp, KernelOp | None]] = []

    def collect(ops: tuple[KernelOp, ...], parent: KernelOp | None = None) -> None:
        for op in ops:
            records.append((op, parent))
            body = op.attrs.get("body", ())
            if type(body) is tuple:
                collect(body, op)

    collect(kernel.ops)
    coevolving_regions = tuple(
        op
        for op, _ in records
        if op.kind == "ForPasses"
        and op.attrs.get("trace_kind") == "lane_local_coevolving"
    )
    if coevolving_regions:
        _validate_dynamic_coevolving_modulation_ops(
            kernel,
            coevolving_regions=coevolving_regions,
            records=records,
        )
        return

    effect_kinds = {
        "InitializeEffectiveParameter",
        "ApplyModulation",
    }
    has_dynamic_ops = any(
        op.kind in effect_kinds
        or (
            op.kind == "ForPasses"
            and op.attrs.get("trace_kind") == "lane_local_counted"
        )
        or (
            op.kind == "StepMechanism"
            and op.attrs.get("active_lanes") == "parent_finished_predicate"
        )
        for op, _ in records
    )
    if not has_dynamic_ops:
        return

    dynamic_finished = tuple(
        value
        for value in kernel.finished_values
        if value.predicate_kind
        == "execution_count_at_least_effective_parameter"
    )
    if (
        kernel.lane_layout.kind != STATEFUL_LANE_LAYOUT
        or len(kernel.modulations) != 1
        or len(kernel.modulations) != len(kernel.effective_parameters)
        or len(kernel.modulations) != len(dynamic_finished)
    ):
        raise ValueError(
            "KernelIR lane-local modulation ops require exactly one stateful "
            "modulation/effective-parameter/finished-value declaration suite."
        )
    if not _dynamic_modulation_lowering_eligible(kernel):
        raise ValueError(
            "KernelIR lane-local modulation operations fall outside the exact "
            "first controlled-finished lowering boundary."
        )

    expected_top_kinds = (
        "InitializeState",
        *("InitializeEffectiveParameter" for _ in kernel.effective_parameters),
        "ForTrials",
    )
    if tuple(op.kind for op in kernel.ops) != expected_top_kinds:
        raise ValueError(
            "KernelIR lane-local effective parameters must initialize exactly "
            "once after state initialization and before one trial region."
        )

    initializer_ops = tuple(
        op for op in kernel.ops if op.kind == "InitializeEffectiveParameter"
    )
    all_initializer_ops = tuple(
        op for op, _ in records if op.kind == "InitializeEffectiveParameter"
    )
    if initializer_ops != all_initializer_ops:
        raise ValueError(
            "KernelIR InitializeEffectiveParameter operations must be top-level."
        )
    for op, parameter in zip(initializer_ops, kernel.effective_parameters):
        expected_value = effective_parameter_value(parameter)
        if (
            op.target != parameter.target
            or op.inputs
            or len(op.outputs) != 1
            or not _kernel_value_matches(op.outputs[0], expected_value)
            or not _attrs_match_exactly(
                op.attrs,
                _effective_parameter_initializer_attrs(parameter),
            )
        ):
            raise ValueError(
                "KernelIR InitializeEffectiveParameter does not exactly match "
                f"effective parameter {parameter.effective_parameter_id}."
            )

    trials = kernel.ops[-1]
    trial_body = trials.attrs.get("body")
    if type(trial_body) is not tuple or any(
        type(op) is not KernelOp for op in trial_body
    ):
        raise ValueError(
            "KernelIR lane-local modulation requires one typed ForTrials body."
        )

    direct_apply_ops = tuple(
        op for op in trial_body if op.kind == "ApplyModulation"
    )
    all_apply_ops = tuple(op for op, _ in records if op.kind == "ApplyModulation")
    direct_pass_regions = tuple(
        op
        for op in trial_body
        if op.kind == "ForPasses"
        and op.attrs.get("trace_kind") == "lane_local_counted"
    )
    all_pass_regions = tuple(
        op
        for op, _ in records
        if op.kind == "ForPasses"
        and op.attrs.get("trace_kind") == "lane_local_counted"
    )
    if (
        len(direct_apply_ops) != len(kernel.modulations)
        or direct_apply_ops != all_apply_ops
        or len(direct_pass_regions) != len(kernel.modulations)
        or direct_pass_regions != all_pass_regions
    ):
        raise ValueError(
            "KernelIR must place exactly one ApplyModulation and one lane-local "
            "ForPasses region per modulation directly in the trial body."
        )

    pass_regions = tuple(
        region for region in kernel.schedule_regions if region.kind == "pass"
    )
    if len(pass_regions) != 1:
        raise ValueError(
            "KernelIR lane-local counted execution requires exactly one typed "
            "pass schedule region."
        )
    pass_region = pass_regions[0]
    effective_by_id = {
        parameter.effective_parameter_id: parameter
        for parameter in kernel.effective_parameters
    }
    finished_by_effective_id = {
        value.attrs["effective_parameter_id"]: value
        for value in dynamic_finished
    }
    body_indices = {id(op): index for index, op in enumerate(trial_body)}

    for modulation in kernel.modulations:
        parameter = effective_by_id[modulation.effective_parameter_id]
        finished = finished_by_effective_id[modulation.effective_parameter_id]
        apply_matches = tuple(
            op
            for op in direct_apply_ops
            if op.attrs.get("modulation_id") == modulation.modulation_id
        )
        region_matches = tuple(
            op
            for op in direct_pass_regions
            if op.attrs.get("effective_parameter_id")
            == modulation.effective_parameter_id
        )
        if len(apply_matches) != 1 or len(region_matches) != 1:
            raise ValueError(
                "KernelIR lane-local modulation operations must form an exact "
                "declaration-ID bijection."
            )
        apply = apply_matches[0]
        region = region_matches[0]
        apply_index = body_indices[id(apply)]
        if (
            apply_index == 0
            or apply_index + 1 >= len(trial_body)
            or trial_body[apply_index + 1] is not region
        ):
            raise ValueError(
                "KernelIR ApplyModulation must immediately follow its controller "
                "call and immediately precede its lane-local pass region."
            )

        controller = kernel.graph.node(modulation.controller)
        source = kernel.graph.node(modulation.source)
        controller_value = KernelValue(
            node_output_value_name(
                kernel.graph,
                controller,
                modulation.control_signal_port,
            ),
            modulation.width,
            modulation.dtype,
        )
        source_value = KernelValue(
            node_output_value_name(
                kernel.graph,
                source,
                modulation.source_port,
            ),
            modulation.width,
            modulation.dtype,
        )
        controller_call = trial_body[apply_index - 1]
        expected_controller_attrs = {
            "component_type": controller.component_type,
            "function_type": controller.function_type,
            "component_id": modulation.controller_component_id,
            "params": dict(controller.params),
            "output_port": modulation.control_signal_port,
            "spec_key": modulation.controller_function_spec_key,
        }
        if (
            controller_call.kind != "CallFunction"
            or controller_call.target != modulation.controller
            or len(controller_call.inputs) != 1
            or not _kernel_value_matches(controller_call.inputs[0], source_value)
            or len(controller_call.outputs) != 1
            or not _kernel_value_matches(
                controller_call.outputs[0],
                controller_value,
            )
            or not _attrs_match_exactly(
                controller_call.attrs,
                expected_controller_attrs,
            )
        ):
            raise ValueError(
                "KernelIR ApplyModulation must consume its exact typed "
                "controller CallFunction output."
            )

        held_value = effective_parameter_value(parameter)
        expected_apply = apply_modulation_op(
            modulation,
            held_effective=held_value,
            controller_value=controller_value,
        )
        if (
            apply.target != expected_apply.target
            or len(apply.inputs) != 2
            or any(
                not _kernel_value_matches(actual, expected)
                for actual, expected in zip(apply.inputs, expected_apply.inputs)
            )
            or len(apply.outputs) != 1
            or not _kernel_value_matches(
                apply.outputs[0],
                expected_apply.outputs[0],
            )
            or not _attrs_match_exactly(apply.attrs, expected_apply.attrs)
        ):
            raise ValueError(
                "KernelIR ApplyModulation does not exactly match modulation "
                f"{modulation.modulation_id}."
            )

        target = kernel.graph.node(modulation.target)
        output_slices = _node_output_port_slices(target)
        expected_outputs = tuple(
            KernelValue(
                node_output_value_name(kernel.graph, target, port_name),
                port_width,
            )
            for port_name, port_width, _, _, _ in output_slices
        )
        expected_region_attrs = {
            "region": pass_region,
            "body": region.attrs["body"],
            "declaration_only": False,
            "trace_kind": "lane_local_counted",
            "finished_value_id": finished.value_id,
            "effective_parameter_id": parameter.effective_parameter_id,
            "target_component_id": modulation.target_component_id,
            "target_parameter_port_id": modulation.target_parameter_port_id,
            "producer_consideration_set_id": (
                finished.producer_consideration_set_id
            ),
            "rounding": finished.attrs["rounding"],
            "minimum": finished.attrs["minimum"],
            "maximum": finished.attrs["maximum"],
            "max_steps": kernel.max_steps,
        }
        if (
            region.target != "passes"
            or len(region.inputs) != 1
            or not _kernel_value_matches(region.inputs[0], held_value)
            or len(region.outputs) != len(expected_outputs) + 1
            or any(
                not _kernel_value_matches(actual, expected)
                for actual, expected in zip(
                    region.outputs[:-1],
                    expected_outputs,
                )
            )
            or not _kernel_value_matches(
                region.outputs[-1],
                dynamic_truncation_value(finished),
            )
            or not _attrs_match_exactly(region.attrs, expected_region_attrs)
        ):
            raise ValueError(
                "KernelIR lane-local ForPasses does not exactly match its typed "
                "effective parameter and finished value."
            )

        region_body = region.attrs["body"]
        steps = tuple(op for op in region_body if op.kind == "StepMechanism")
        expected_state_ids = tuple(
            sorted(
                state.state_id
                for state in kernel.states
                if _state_component_id(kernel.graph, state)
                == modulation.target_component_id
            )
        )
        expected_step_attrs = {
            "component_type": target.component_type,
            "function_type": target.function_type,
            "component_id": modulation.target_component_id,
            "params": dict(target.params),
            "spec_key": target.attrs["spec_key"],
            "state_ids": expected_state_ids,
            "active_lanes": "parent_finished_predicate",
            "loop_counter": "parent_pass_index",
            "finished_value_id": finished.value_id,
            "effective_parameter_id": parameter.effective_parameter_id,
            "target_parameter_port_id": modulation.target_parameter_port_id,
        }
        supported_body_kinds = {
            "CallProjection",
            "CombineSum",
            "CombineProduct",
            "Concatenate",
            "StepMechanism",
        }
        if (
            len(steps) != 1
            or region_body[-1] is not steps[0]
            or any(op.target != target.name for op in region_body)
            or any(op.kind not in supported_body_kinds for op in region_body)
            or len(steps[0].inputs) != 1
            or not _kernel_value_matches(
                steps[0].inputs[0],
                KernelValue(
                    node_input_value_name(kernel.graph, target),
                    target.input_width,
                ),
            )
            or len(steps[0].outputs) != len(expected_outputs)
            or any(
                not _kernel_value_matches(actual, expected)
                for actual, expected in zip(steps[0].outputs, expected_outputs)
            )
            or not _attrs_match_exactly(
                steps[0].attrs,
                expected_step_attrs,
            )
        ):
            raise ValueError(
                "KernelIR lane-local ForPasses must end in exactly one dynamic "
                "StepMechanism matching its finished-value owner."
            )

    expected_ops = _canonical_dynamic_modulation_kernel_ops(kernel)
    if not _kernel_op_sequences_match_exactly(kernel.ops, expected_ops):
        raise ValueError(
            "KernelIR lane-local modulation operations must exactly match the "
            "complete compiler-derived trial program."
        )


def _validate_dynamic_coevolving_modulation_ops(
    kernel: KernelIR,
    *,
    coevolving_regions: tuple[KernelOp, ...],
    records: list[tuple[KernelOp, KernelOp | None]],
) -> None:
    """Authenticate the first complete controlled coevolution program."""

    if (
        len(coevolving_regions) != 1
        or not _dynamic_coevolving_lowering_eligible(kernel)
    ):
        raise ValueError(
            "KernelIR lane-local coevolving operations fall outside the exact "
            "controlled LCA/DDM lowering boundary."
        )
    expected_top_kinds = (
        "InitializeState",
        *("InitializeEffectiveParameter" for _ in kernel.effective_parameters),
        "ForTrials",
    )
    top_initializers = tuple(
        op for op in kernel.ops if op.kind == "InitializeEffectiveParameter"
    )
    all_initializers = tuple(
        op for op, _ in records if op.kind == "InitializeEffectiveParameter"
    )
    all_apply_ops = tuple(
        op for op, _ in records if op.kind == "ApplyModulation"
    )
    if (
        tuple(op.kind for op in kernel.ops) != expected_top_kinds
        or top_initializers != all_initializers
        or len(all_apply_ops) != 1
    ):
        raise ValueError(
            "KernelIR controlled coevolution requires top-level state/effective "
            "initialization and exactly one nested modulation effect."
        )

    expected_ops = _canonical_dynamic_coevolving_kernel_ops(kernel)
    if not _kernel_op_sequences_match_exactly(kernel.ops, expected_ops):
        raise ValueError(
            "KernelIR controlled coevolution operations must exactly match the "
            "complete compiler-derived trial program."
        )


def _canonical_dynamic_coevolving_kernel_ops(
    kernel: KernelIR,
) -> tuple[KernelOp, ...]:
    """Build the first typed controlled LCA/DDM coevolution program."""

    if not _dynamic_coevolving_lowering_eligible(kernel):
        raise ValueError(
            "KernelIR graph is not eligible for controlled coevolving lowering."
        )
    graph = kernel.graph
    modulation = kernel.modulations[0]
    parameter = kernel.effective_parameters[0]
    dynamic_finished = next(
        value
        for value in kernel.finished_values
        if value.predicate_kind
        == "execution_count_at_least_effective_parameter"
    )
    terminator_finished = next(
        value
        for value in kernel.finished_values
        if value.predicate_kind == "dynamic"
    )
    controller = graph.node(modulation.controller)
    source = graph.node(modulation.source)
    terminator = graph.node(terminator_finished.node)
    threshold_controller = next(
        node
        for node in graph.nodes
        if node.component_type == "ControlMechanism"
        and node.component_id != controller.component_id
    )
    terminator_spec = kernel.op_specs.lookup_spec(
        terminator.attrs["spec_key"]
    )
    assert isinstance(terminator_spec, MechanismOpSpec)

    canonical_trial_ops = _trial_body_ops(graph)
    target_indices = tuple(
        index
        for index, op in enumerate(canonical_trial_ops)
        if op.target == modulation.target
    )
    if not target_indices:
        raise ValueError(
            "KernelIR controlled coevolution target has no compiler-derived body."
        )

    controller_value = KernelValue(
        node_output_value_name(
            graph,
            controller,
            modulation.control_signal_port,
        ),
        modulation.width,
        modulation.dtype,
    )
    controller_call = KernelOp(
        kind="CallFunction",
        target=controller.name,
        inputs=(
            KernelValue(
                node_output_value_name(
                    graph,
                    source,
                    modulation.source_port,
                ),
                modulation.width,
                modulation.dtype,
            ),
        ),
        outputs=(controller_value,),
        attrs={
            "component_type": controller.component_type,
            "function_type": controller.function_type,
            "component_id": modulation.controller_component_id,
            "params": dict(controller.params),
            "output_port": modulation.control_signal_port,
            "spec_key": modulation.controller_function_spec_key,
        },
    )
    held_value = effective_parameter_value(parameter)
    apply = apply_modulation_op(
        modulation,
        held_effective=held_value,
        controller_value=controller_value,
    )
    insert_at = target_indices[0]
    coevolving_body = (
        *canonical_trial_ops[:insert_at],
        controller_call,
        apply,
        *canonical_trial_ops[insert_at:],
    )
    pass_regions = tuple(
        region for region in kernel.schedule_regions if region.kind == "pass"
    )
    assert len(pass_regions) == 1
    coevolving_region = KernelOp(
        kind="ForPasses",
        target="passes",
        inputs=(held_value,),
        attrs={
            "region": pass_regions[0],
            "body": coevolving_body,
            "declaration_only": False,
            "trace_kind": "lane_local_coevolving",
            "stepper_component_id": modulation.target_component_id,
            "stepper_finished_value_id": dynamic_finished.value_id,
            "terminator_component_id": terminator_finished.component_id,
            "terminator_finished_value_id": terminator_finished.value_id,
            "effective_parameter_id": parameter.effective_parameter_id,
            "target_parameter_port_id": modulation.target_parameter_port_id,
            "producer_consideration_set_id": (
                dynamic_finished.producer_consideration_set_id
            ),
            "rounding": dynamic_finished.attrs["rounding"],
            "minimum": dynamic_finished.attrs["minimum"],
            "maximum": dynamic_finished.attrs["maximum"],
            "max_steps": kernel.max_steps,
            "max_steps_scope": "terminator_local",
            "consideration_set_ids": tuple(
                item.consideration_set_id
                for item in sorted(
                    kernel.consideration_sets,
                    key=lambda item: item.consideration_set_id,
                )
            ),
            "terminator_trial_states": tuple(
                (
                    declaration.name,
                    int(declaration.width or 1),
                    float(declaration.initial),
                )
                for declaration in terminator_spec.trial_states
            ),
            "terminator_finished_output": terminator_spec.finished_output,
            "terminator_initial_control_value": threshold_controller.attrs[
                "absorbed_control_initial_value"
            ],
            "terminator_control_storage": "lane_persistent",
            "terminator_control_update": "ordered_threshold_override",
            # A one-step DDM finishes before the earlier threshold-controller
            # set has run.  PNL's AllHaveRun cleanup visits that set after the
            # gates; fold its only observable effect into the held control
            # value carried to the next trial.
            "completion_cleanup": "fold_terminator_control_state",
        },
    )
    state_values = _state_kernel_values(graph)
    return (
        KernelOp(
            kind="InitializeState",
            target="lane",
            outputs=state_values,
        ),
        *(
            initialize_effective_parameter_op(item)
            for item in kernel.effective_parameters
        ),
        KernelOp(
            kind="ForTrials",
            target="trials",
            attrs={
                "body": (
                    *_trial_reset_ops(graph, state_values),
                    coevolving_region,
                ),
            },
        ),
    )


def _canonical_dynamic_modulation_kernel_ops(
    kernel: KernelIR,
) -> tuple[KernelOp, ...]:
    """Build the only complete lane-local modulation program accepted so far.

    The declaration checkpoint is intentionally hand-constructible, but that
    must not let a caller erase or rewrite ordinary dataflow around the new
    effects.  Rebuild every non-control operation from GraphIR, replace each
    controlled target's atomic mechanism call with its authenticated dynamic
    region, and insert the absorbed controller immediately before that region.
    The same builder is also the lowering path for the later executable
    checkpoint, so validation and construction cannot drift apart.
    """

    graph = kernel.graph
    pass_regions = tuple(
        region for region in kernel.schedule_regions if region.kind == "pass"
    )
    if len(pass_regions) != 1:
        raise ValueError(
            "KernelIR lane-local counted execution requires exactly one typed "
            "pass schedule region."
        )
    pass_region = pass_regions[0]

    canonical_trial_ops = _trial_body_ops(graph)
    replacements: dict[int, tuple[KernelOp, ...]] = {}
    replaced_indices: set[int] = set()
    effective_by_id = {
        parameter.effective_parameter_id: parameter
        for parameter in kernel.effective_parameters
    }
    finished_by_effective_id = {
        value.attrs["effective_parameter_id"]: value
        for value in kernel.finished_values
        if value.predicate_kind
        == "execution_count_at_least_effective_parameter"
    }

    for modulation in kernel.modulations:
        parameter = effective_by_id[modulation.effective_parameter_id]
        finished = finished_by_effective_id[modulation.effective_parameter_id]
        target = graph.node(modulation.target)
        target_indices = tuple(
            index
            for index, op in enumerate(canonical_trial_ops)
            if op.target == target.name
        )
        if (
            not target_indices
            or target_indices
            != tuple(range(target_indices[0], target_indices[-1] + 1))
            or replaced_indices.intersection(target_indices)
        ):
            raise ValueError(
                "KernelIR lane-local target operations must form one unique "
                "contiguous compiler-derived component body."
            )
        target_ops = tuple(canonical_trial_ops[index] for index in target_indices)
        target_calls = tuple(
            op for op in target_ops if op.kind == "CallMechanism"
        )
        if len(target_calls) != 1:
            raise ValueError(
                "KernelIR lane-local target requires exactly one compiler-derived "
                "CallMechanism operation."
            )
        target_call = target_calls[0]
        state_ids = tuple(
            sorted(
                state.state_id
                for state in kernel.states
                if _state_component_id(graph, state)
                == modulation.target_component_id
            )
        )
        dynamic_step = KernelOp(
            kind="StepMechanism",
            target=target_call.target,
            inputs=target_call.inputs,
            outputs=target_call.outputs,
            attrs={
                **target_call.attrs,
                "state_ids": state_ids,
                "active_lanes": "parent_finished_predicate",
                "loop_counter": "parent_pass_index",
                "finished_value_id": finished.value_id,
                "effective_parameter_id": parameter.effective_parameter_id,
                "target_parameter_port_id": (
                    modulation.target_parameter_port_id
                ),
            },
        )
        dynamic_body = tuple(
            dynamic_step if op is target_call else op for op in target_ops
        )

        controller = graph.node(modulation.controller)
        source = graph.node(modulation.source)
        controller_value = KernelValue(
            node_output_value_name(
                graph,
                controller,
                modulation.control_signal_port,
            ),
            modulation.width,
            modulation.dtype,
        )
        controller_call = KernelOp(
            kind="CallFunction",
            target=controller.name,
            inputs=(
                KernelValue(
                    node_output_value_name(
                        graph,
                        source,
                        modulation.source_port,
                    ),
                    modulation.width,
                    modulation.dtype,
                ),
            ),
            outputs=(controller_value,),
            attrs={
                "component_type": controller.component_type,
                "function_type": controller.function_type,
                "component_id": modulation.controller_component_id,
                "params": dict(controller.params),
                "output_port": modulation.control_signal_port,
                "spec_key": modulation.controller_function_spec_key,
            },
        )
        held_value = effective_parameter_value(parameter)
        apply = apply_modulation_op(
            modulation,
            held_effective=held_value,
            controller_value=controller_value,
        )
        dynamic_region = KernelOp(
            kind="ForPasses",
            target="passes",
            inputs=(held_value,),
            outputs=(*dynamic_step.outputs, dynamic_truncation_value(finished)),
            attrs={
                "region": pass_region,
                "body": dynamic_body,
                "declaration_only": False,
                "trace_kind": "lane_local_counted",
                "finished_value_id": finished.value_id,
                "effective_parameter_id": parameter.effective_parameter_id,
                "target_component_id": modulation.target_component_id,
                "target_parameter_port_id": (
                    modulation.target_parameter_port_id
                ),
                "producer_consideration_set_id": (
                    finished.producer_consideration_set_id
                ),
                "rounding": finished.attrs["rounding"],
                "minimum": finished.attrs["minimum"],
                "maximum": finished.attrs["maximum"],
                "max_steps": kernel.max_steps,
            },
        )
        truncation_store = KernelOp(
            kind="StoreFlag",
            target=target.name,
            inputs=(dynamic_region.outputs[-1],),
            attrs={
                "node": target.name,
                "name": "truncated",
                "slot": sum(
                    op.kind == "StoreFlag" for op in canonical_trial_ops
                )
                + modulation.modulation_id,
            },
        )
        replacements[target_indices[0]] = (
            controller_call,
            apply,
            dynamic_region,
            truncation_store,
        )
        replaced_indices.update(target_indices)

    scheduled_trial_ops = []
    for index, op in enumerate(canonical_trial_ops):
        replacement = replacements.get(index)
        if replacement is not None:
            scheduled_trial_ops.extend(replacement)
        if index not in replaced_indices:
            scheduled_trial_ops.append(op)

    state_values = _state_kernel_values(graph)
    trial_body = (
        *_trial_reset_ops(graph, state_values),
        *scheduled_trial_ops,
    )
    return (
        KernelOp(
            kind="InitializeState",
            target="lane",
            outputs=state_values,
        ),
        *(
            initialize_effective_parameter_op(parameter)
            for parameter in kernel.effective_parameters
        ),
        KernelOp(
            kind="ForTrials",
            target="trials",
            attrs={"body": trial_body},
        ),
    )


def _kernel_op_sequences_match_exactly(
    actual: tuple[KernelOp, ...],
    expected: tuple[KernelOp, ...],
) -> bool:
    """Type-strict recursive equality for complete compiler-owned programs."""

    if type(actual) is not tuple or len(actual) != len(expected):
        return False
    for actual_op, expected_op in zip(actual, expected):
        if (
            type(actual_op) is not KernelOp
            or type(actual_op.kind) is not str
            or actual_op.kind != expected_op.kind
            or type(actual_op.target) is not str
            or actual_op.target != expected_op.target
            or type(actual_op.inputs) is not tuple
            or len(actual_op.inputs) != len(expected_op.inputs)
            or any(
                not _kernel_value_matches(actual_value, expected_value)
                for actual_value, expected_value in zip(
                    actual_op.inputs,
                    expected_op.inputs,
                )
            )
            or type(actual_op.outputs) is not tuple
            or len(actual_op.outputs) != len(expected_op.outputs)
            or any(
                not _kernel_value_matches(actual_value, expected_value)
                for actual_value, expected_value in zip(
                    actual_op.outputs,
                    expected_op.outputs,
                )
            )
            or set(actual_op.attrs) != set(expected_op.attrs)
        ):
            return False
        for key, expected_value in expected_op.attrs.items():
            actual_value = actual_op.attrs[key]
            if key == "body":
                if not _kernel_op_sequences_match_exactly(
                    actual_value,
                    expected_value,
                ):
                    return False
            elif not _kernel_attribute_matches_exactly(
                actual_value,
                expected_value,
            ):
                return False
    return True


def _kernel_attribute_matches_exactly(actual: Any, expected: Any) -> bool:
    """Compare nested attrs without bool/int aliasing or ndarray ambiguity."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        return bool(
            set(actual) == set(expected)
            and all(
                _kernel_attribute_matches_exactly(actual[key], value)
                for key, value in expected.items()
            )
        )
    if isinstance(expected, tuple):
        return bool(
            len(actual) == len(expected)
            and all(
                _kernel_attribute_matches_exactly(actual_value, expected_value)
                for actual_value, expected_value in zip(actual, expected)
            )
        )
    if isinstance(expected, np.ndarray):
        return bool(
            actual.dtype == expected.dtype
            and actual.shape == expected.shape
            and np.array_equal(actual, expected)
        )
    comparison = actual == expected
    if type(comparison) is bool:
        return comparison
    try:
        return bool(comparison.all())
    except AttributeError:
        return bool(comparison)


def lower_to_kernel_ir(
    ir: BatchedCompositionIR,
    *,
    op_specs: BatchedOpSpecSnapshot | None = None,
) -> KernelIR:
    """Lower semantic BatchedGraphIR and freeze its resolved op specs.

    Supplying ``op_specs`` lets a caller retain an earlier compilation
    snapshot. Direct IR callers remain supported and capture the registry as it
    exists at lowering time.
    """

    graph = ir.graph
    if graph is None:
        raise ValueError("KernelIR lowering requires a batched graph IR.")

    if op_specs is None:
        op_specs = snapshot_batched_op_specs(_graph_spec_keys(graph))

    lane_layout = _lane_layout_for(graph.fusion_kind)
    rng_streams = _rng_streams(graph)
    trial_ops = _trial_body_ops(graph)
    schedule_trace = None
    declaration_only_schedule = False
    scheduled_trial_ops = trial_ops
    trace_requested = (
        graph.metadata.get("schedule_kind") == "precomputed_trace"
        and bool(graph.termination)
    )
    if (
        _precomputed_trace_eligible(graph, lane_layout, rng_streams, op_specs)
        and trace_requested
    ):
        component_budget = _trace_budget(
            graph,
            _TRACE_COMPONENT_BUDGET_KEY,
            PRECOMPUTED_TRACE_COMPONENT_BUDGET,
        )
        weighted_op_budget = _trace_budget(
            graph,
            _TRACE_WEIGHTED_OP_BUDGET_KEY,
            _DEFAULT_TRACE_WEIGHTED_OP_BUDGET,
        )
        schedule_trace = plan_precomputed_schedule_trace(
            scheduler=graph.scheduler,
            consideration_sets=graph.consideration_sets,
            termination=graph.termination,
            expansion_budget=component_budget,
            projections=graph.projections,
            finished_values=graph.finished_values,
        )
        scheduled_trial_ops = _precomputed_trace_ops(
            graph,
            trial_ops,
            schedule_trace,
            op_specs=op_specs,
            component_budget=component_budget,
            weighted_op_budget=weighted_op_budget,
        )
    elif trace_requested or _requires_pass_region(graph):
        declaration_only_schedule = True
        pass_region = next(
            (
                region
                for region in graph.schedule_regions
                if region.kind == "pass"
            ),
            None,
        )
        scheduled_trial_ops = (
            KernelOp(
                kind="ForPasses",
                target="passes",
                attrs={
                    "region": pass_region,
                    "conditions": graph.scheduler,
                    "consideration_sets": graph.consideration_sets,
                    "finished_values": tuple(
                        KernelValue(value.name, value.width, value.dtype)
                        for value in graph.finished_values
                    ),
                    "body": trial_ops,
                    # No backend may treat this declaration as sequential
                    # execution.  Each additional scheduler tier replaces this
                    # marker only after its semantics are represented.
                    "declaration_only": True,
                },
            ),
        )
    if lane_layout.kind == STATEFUL_LANE_LAYOUT:
        initial_state_values = _state_kernel_values(graph)
        trial_reset_ops = _trial_reset_ops(graph, initial_state_values)
        ops = (
            KernelOp(
                kind="InitializeState",
                target="lane",
                outputs=initial_state_values,
            ),
            KernelOp(
                kind="ForTrials",
                target="trials",
                attrs={"body": (*trial_reset_ops, *scheduled_trial_ops)},
            ),
        )
    else:
        ops = scheduled_trial_ops

    kernel = KernelIR(
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
        op_specs=op_specs,
        executable=graph.executable and not declaration_only_schedule,
        metadata={
            "composition_name": ir.metadata.get("composition_name"),
            "fusion_kind": graph.fusion_kind,
            **graph.metadata,
        },
        ports=graph.ports,
        absorbed_projections=graph.absorbed_projections,
        scheduler=graph.scheduler,
        schedule_regions=graph.schedule_regions,
        consideration_sets=graph.consideration_sets,
        finished_values=graph.finished_values,
        effective_parameters=graph.effective_parameters,
        modulations=graph.modulations,
        resets=graph.resets,
        termination=graph.termination,
        schedule_trace=schedule_trace,
    )
    if _dynamic_modulation_lowering_eligible(kernel):
        return replace(
            kernel,
            ops=_canonical_dynamic_modulation_kernel_ops(kernel),
            executable=graph.executable,
        )
    if _dynamic_coevolving_lowering_eligible(kernel):
        return replace(
            kernel,
            ops=_canonical_dynamic_coevolving_kernel_ops(kernel),
            executable=graph.executable,
        )
    return kernel


def _dynamic_coevolving_lowering_eligible(kernel: KernelIR) -> bool:
    """Whether the first exact controlled LCA/DDM region can materialize."""

    graph = kernel.graph
    dynamic_finished = tuple(
        value
        for value in kernel.finished_values
        if value.predicate_kind
        == "execution_count_at_least_effective_parameter"
    )
    terminator_finished = tuple(
        value
        for value in kernel.finished_values
        if value.predicate_kind == "dynamic"
    )
    node_component_ids = tuple(node.component_id for node in graph.nodes)
    if not (
        _dynamic_controlled_coevolving_graph_eligible(
            graph,
            kernel.params,
            op_specs=kernel.op_specs,
        )
        and graph.metadata.get("schedule_kind") == "dynamic_lane_local"
        and graph.fusion_kind == COEVOLVING_GRAPH_FUSION
        and kernel.lane_layout.kind == STATEFUL_LANE_LAYOUT
        and kernel.schedule_trace is None
        and len(kernel.modulations) == 1
        and len(kernel.effective_parameters) == 1
        and len(dynamic_finished) == 1
        and len(terminator_finished) == 1
        and len(graph.nodes) == 11
        and len(kernel.consideration_sets) == 6
        and len(kernel.schedule_regions) == 2
        and len(kernel.termination) == 2
        and len(kernel.states) == 3
        and len(kernel.resets) == 1
        and len(kernel.inputs) == 4
        and len(kernel.outputs) == 2
        and all(type(component_id) is int for component_id in node_component_ids)
        and node_component_ids == tuple(range(len(graph.nodes)))
    ):
        return False

    modulation = kernel.modulations[0]
    parameter = kernel.effective_parameters[0]
    stepper_finished = dynamic_finished[0]
    ddm_finished = terminator_finished[0]
    try:
        source = graph.node(modulation.source)
        controller = graph.node(modulation.controller)
        stepper = graph.node(modulation.target)
        terminator = graph.node(ddm_finished.node)
        stepper_spec = kernel.op_specs.lookup_spec(stepper.attrs["spec_key"])
        terminator_spec = kernel.op_specs.lookup_spec(
            terminator.attrs["spec_key"]
        )
    except (BatchedOpSpecError, KeyError):
        return False
    if not (
        modulation.effective_parameter_id == parameter.effective_parameter_id
        and stepper_finished.component_id == modulation.target_component_id
        and stepper_finished.producer_consideration_set_id == 2
        and ddm_finished.producer_consideration_set_id == 4
        and _kernel_attribute_matches_exactly(ddm_finished.attrs, {})
        and isinstance(stepper_spec, MechanismOpSpec)
        and stepper_spec.can_step
        and stepper_spec.persistent_state
        and not stepper_spec.is_terminator
        and isinstance(terminator_spec, MechanismOpSpec)
        and terminator_spec.can_step
        and terminator_spec.is_terminator
        and terminator_spec.readout_emit is not None
        and terminator_spec.finished_output == "finished"
        and tuple(
            (
                declaration.name,
                declaration.width,
                declaration.initial,
            )
            for declaration in terminator_spec.trial_states
        )
        == (
            ("value", 1, 0.0),
            ("steps", 1, 0.0),
            ("finished", 1, 0.0),
        )
        and _dynamic_count_controller_eligible(kernel, controller)
        and _dynamic_controller_shape_eligible(
            kernel,
            controller,
            modulation,
        )
        and stepper.attrs.get("termination_input_node") == source.name
        and source in graph.nodes
        and controller in graph.nodes
        and stepper in graph.nodes
        and terminator in graph.nodes
    ):
        return False

    consideration_sets = tuple(
        sorted(
            kernel.consideration_sets,
            key=lambda item: item.consideration_set_id,
        )
    )
    if (
        tuple(item.consideration_set_id for item in consideration_sets)
        != tuple(range(6))
        or tuple(len(item.nodes) for item in consideration_sets)
        != (4, 2, 1, 1, 1, 2)
        or any(
            item.region != "pass"
            or item.inputs_frozen is not True
            or item.component_ids
            != tuple(graph.node(name).component_id for name in item.nodes)
            or len(set(item.nodes)) != len(item.nodes)
            or len(set(item.component_ids)) != len(item.component_ids)
            for item in consideration_sets
        )
        or source.name not in consideration_sets[0].nodes
        or controller.name not in consideration_sets[1].nodes
        or consideration_sets[2].nodes != (stepper.name,)
        or consideration_sets[2].component_ids
        != (modulation.target_component_id,)
        or consideration_sets[4].nodes != (terminator.name,)
        or consideration_sets[4].component_ids != (ddm_finished.component_id,)
    ):
        return False

    extra_controller_name = next(
        (
            name
            for name in consideration_sets[1].nodes
            if name != controller.name
        ),
        None,
    )
    if extra_controller_name is None:
        return False
    extra_controller = graph.node(extra_controller_name)
    middle = graph.node(consideration_sets[3].nodes[0])
    followers = tuple(graph.node(name) for name in consideration_sets[5].nodes)
    try:
        middle_spec = kernel.op_specs.lookup_spec(middle.attrs["spec_key"])
    except (BatchedOpSpecError, KeyError):
        return False
    parameters_by_name = {parameter.name: parameter for parameter in kernel.params}
    runtime_ddm_arguments = (
        "threshold",
        "threshold_collapse",
    )
    frozen_ddm_arguments = (
        "noise",
        "starting_value",
        "offset",
    )
    runtime_ddm_parameters = tuple(
        parameters_by_name.get(terminator.params.get(argument))
        for argument in runtime_ddm_arguments
    )
    frozen_ddm_parameters = tuple(
        parameters_by_name.get(terminator.params.get(argument))
        for argument in frozen_ddm_arguments
    )
    ports_by_id = {port.port_id: port for port in kernel.ports}
    extra_input = (
        ports_by_id.get(extra_controller.input_port_ids[0])
        if len(extra_controller.input_port_ids) == 1
        else None
    )
    extra_output = (
        ports_by_id.get(extra_controller.output_port_ids[0])
        if len(extra_controller.output_port_ids) == 1
        else None
    )
    absorbed_control = extra_controller.attrs.get("absorbed_control")
    if not (
        extra_controller.component_type == "ControlMechanism"
        and extra_controller.function_type == "Identity"
        and type(extra_controller.input_width) is int
        and extra_controller.input_width == 1
        and type(extra_controller.output_width) is int
        and extra_controller.output_width == 1
        and extra_controller.combine == "sum"
        and not extra_controller.params
        and not extra_controller.parameter_port_ids
        and set(extra_controller.attrs)
        == {
            "input_ports",
            "output_ports",
            "absorbed_control",
            "absorbed_control_initial_value",
        }
        and type(
            extra_controller.attrs["absorbed_control_initial_value"]
        ) is float
        and extra_controller.attrs["absorbed_control_initial_value"] == 1.0
        and extra_input is not None
        and extra_input.kind == "InputPort"
        and extra_input.width == 1
        and extra_output is not None
        and extra_output.kind == "ControlSignal"
        and extra_output.width == 1
        and _kernel_attribute_matches_exactly(
            extra_controller.attrs["input_ports"],
            ((extra_input.name, 1, "sum", extra_input.port_id, 0, 1),),
        )
        and _kernel_attribute_matches_exactly(
            extra_controller.attrs["output_ports"],
            (extra_output.name,),
        )
        and isinstance(absorbed_control, Mapping)
        and set(absorbed_control)
        == {"source", "target", "parameter", "modulation"}
        and type(absorbed_control["source"]) is str
        and absorbed_control["source"]
        and absorbed_control["source"] not in {node.name for node in graph.nodes}
        and absorbed_control["target"] == terminator.name
        and absorbed_control["parameter"] == "threshold"
        and absorbed_control["modulation"] == "OVERRIDE"
        and isinstance(middle_spec, MechanismOpSpec)
        and middle_spec.has_triton
        and not middle_spec.can_step
        and not middle_spec.states
        and not middle_spec.trial_states
        and not middle_spec.rng
        and middle.input_width == 7
        and middle.output_width == 1
        and all(parameter is not None for parameter in runtime_ddm_parameters)
        and runtime_ddm_parameters[0].name
        == f"{terminator.name}.threshold"
        and runtime_ddm_parameters[0].aliases
        == (
            "ddm.threshold",
            "DDM.threshold",
            *_node_param_aliases(terminator.name, "threshold"),
            *_node_param_aliases(absorbed_control["source"], "intercept"),
        )
        and type(runtime_ddm_parameters[0].default) is float
        and np.isfinite(runtime_ddm_parameters[0].default)
        and runtime_ddm_parameters[0].default >= 0.0
        and runtime_ddm_parameters[0].minimum == 0.0
        and runtime_ddm_parameters[0].minimum_inclusive is True
        and runtime_ddm_parameters[0].maximum is None
        and runtime_ddm_parameters[0].maximum_inclusive is True
        and runtime_ddm_parameters[1].name
        == f"{terminator.name}.threshold_collapse"
        and runtime_ddm_parameters[1].aliases
        == (
            "ddm.threshold_collapse",
            "DDM.threshold_collapse",
            *_node_param_aliases(
                terminator.name,
                "threshold_collapse",
            ),
            *_node_param_aliases(
                absorbed_control["source"],
                "offset-integrator_function",
            ),
        )
        and type(runtime_ddm_parameters[1].default) is float
        and np.isfinite(runtime_ddm_parameters[1].default)
        and runtime_ddm_parameters[1].default <= 0.0
        and runtime_ddm_parameters[1].minimum is None
        and runtime_ddm_parameters[1].minimum_inclusive is True
        and runtime_ddm_parameters[1].maximum == 0.0
        and runtime_ddm_parameters[1].maximum_inclusive is True
        and all(
            parameter.owner_component_id == terminator.component_id
            and parameter.owner_scope == "function"
            and parameter.runtime_mutable is True
            and parameter.runtime_constraint == ""
            for parameter in runtime_ddm_parameters
        )
        and all(parameter is not None for parameter in frozen_ddm_parameters)
        and all(
            parameter.owner_component_id == terminator.component_id
            and parameter.runtime_mutable is False
            and parameter.runtime_constraint
            == "first coevolving DDM boundary is frozen in KernelIR"
            for parameter in frozen_ddm_parameters
        )
        and stepper.input_width == 2
        and stepper.output_width == 2
        and terminator.input_width == 1
        and len(terminator.output_port_ids) == 2
        and all(
            follower.input_width == 1 and follower.output_width == 1
            for follower in followers
        )
        and sorted(graph.node(name).input_width for name in consideration_sets[0].nodes)
        == [1, 1, 2, 4]
    ):
        return False

    scheduler_by_id = {
        condition.component_id: condition for condition in kernel.scheduler
    }

    def simple_condition_matches(condition, kind, attrs) -> bool:
        return bool(
            condition is not None
            and condition.condition_type == kind
            and condition.region == "pass"
            and _kernel_attribute_matches_exactly(condition.dependencies, ())
            and _kernel_attribute_matches_exactly(
                condition.dependency_component_ids,
                (),
            )
            and _kernel_attribute_matches_exactly(
                condition.finished_value_ids,
                (),
            )
            and _kernel_attribute_matches_exactly(condition.attrs, attrs)
        )

    def when_finished_matches(condition, dependency, finished) -> bool:
        return bool(
            condition is not None
            and condition.condition_type == "WhenFinished"
            and condition.region == "pass"
            and _kernel_attribute_matches_exactly(
                condition.dependencies,
                (dependency.name,),
            )
            and _kernel_attribute_matches_exactly(
                condition.dependency_component_ids,
                (dependency.component_id,),
            )
            and _kernel_attribute_matches_exactly(
                condition.finished_value_ids,
                (finished.value_id,),
            )
            and _kernel_attribute_matches_exactly(
                condition.attrs,
                {"predicate": "is_finished"},
            )
        )

    at_pass_zero = {
        "pass_index": 0,
        "time_scale": "ENVIRONMENT_STATE_UPDATE",
    }
    stepper_senders = tuple(
        projection.sender_component_id
        for projection in graph.projections
        if projection.receiver_component_id == stepper.component_id
    )
    if len(stepper_senders) != 1:
        return False
    task_origin_id = stepper_senders[0]
    warmup = kernel.metadata.get("coevolve_warmup")
    if type(warmup) is not int or warmup < 0:
        return False

    def origin_condition_matches(component_id: int) -> bool:
        pass_index = warmup if component_id == task_origin_id else 0
        return simple_condition_matches(
            scheduler_by_id.get(component_id),
            "AtPass",
            {
                "pass_index": pass_index,
                "time_scale": "ENVIRONMENT_STATE_UPDATE",
            },
        )

    if not (
        all(
            origin_condition_matches(component_id)
            for component_id in consideration_sets[0].component_ids
        )
        and simple_condition_matches(
            scheduler_by_id.get(modulation.controller_component_id),
            "AtPass",
            at_pass_zero,
        )
        and simple_condition_matches(
            scheduler_by_id.get(modulation.target_component_id),
            "Always",
            {},
        )
        and when_finished_matches(
            scheduler_by_id.get(extra_controller.component_id),
            stepper,
            stepper_finished,
        )
        and when_finished_matches(
            scheduler_by_id.get(middle.component_id),
            stepper,
            stepper_finished,
        )
        and when_finished_matches(
            scheduler_by_id.get(terminator.component_id),
            stepper,
            stepper_finished,
        )
        and all(
            when_finished_matches(
                scheduler_by_id.get(follower.component_id),
                terminator,
                ddm_finished,
            )
            for follower in followers
        )
        and graph.execution_order
        == (
            *consideration_sets[0].nodes,
            stepper.name,
            middle.name,
            terminator.name,
            *consideration_sets[5].nodes,
        )
    ):
        return False

    if tuple(
        (region.name, region.kind, region.time_scale, region.parent)
        for region in kernel.schedule_regions
    ) != (
        ("trial", "trial", "ENVIRONMENT_STATE_UPDATE", ""),
        ("pass", "pass", "PASS", "trial"),
    ):
        return False
    termination_by_scale = {
        item.time_scale: item for item in kernel.termination
    }
    trial_termination = termination_by_scale.get("ENVIRONMENT_STATE_UPDATE")
    run_termination = termination_by_scale.get("ENVIRONMENT_SEQUENCE")
    if not (
        len(termination_by_scale) == 2
        and trial_termination is not None
        and trial_termination.condition_type == "AllHaveRun"
        and _kernel_attribute_matches_exactly(
            trial_termination.dependency_component_ids,
            node_component_ids,
        )
        and _kernel_attribute_matches_exactly(trial_termination.attrs, {})
        and run_termination is not None
        and run_termination.condition_type == "Never"
        and _kernel_attribute_matches_exactly(
            run_termination.dependency_component_ids,
            (),
        )
        and _kernel_attribute_matches_exactly(run_termination.attrs, {})
    ):
        return False

    state_component_ids = tuple(state.component_id for state in kernel.states)
    reset = kernel.resets[0]
    stream = kernel.rng_streams[0] if len(kernel.rng_streams) == 1 else None
    if not (
        state_component_ids
        == (
            modulation.target_component_id,
            modulation.target_component_id,
            modulation.target_component_id,
        )
        and reset.component_id == modulation.target_component_id
        and reset.state_ids == tuple(state.state_id for state in kernel.states)
        and reset.condition_type == "Never"
        and reset.region == "trial"
        and _kernel_attribute_matches_exactly(reset.attrs, {})
        and stream is not None
        and stream.name == f"{terminator.name}.rng"
        and stream.node == terminator.name
        and type(stream.width) is int
        and stream.width == 1
        and stream.step_extent == "MAX_STEPS"
        and type(stream.component_id) is int
        and stream.component_id == terminator.component_id
        and type(stream.stream_id) is int
        and stream.stream_id == 0
        and tuple(item.component_id for item in kernel.inputs)
        == consideration_sets[0].component_ids
        and tuple(item.component_id for item in kernel.outputs)
        == consideration_sets[5].component_ids
    ):
        return False

    edges = tuple(
        (projection.sender_component_id, projection.receiver_component_id)
        for projection in graph.projections
    )
    target_senders = tuple(
        sender for sender, receiver in edges if receiver == stepper.component_id
    )
    if len(target_senders) != 1:
        return False
    origin_ids = set(consideration_sets[0].component_ids)
    target_origin_id = target_senders[0]
    middle_senders = tuple(
        sender for sender, receiver in edges if receiver == middle.component_id
    )
    follower_sender_sets = sorted(
        (
            tuple(sorted(sender for sender, receiver in edges if receiver == follower.component_id))
            for follower in followers
        ),
        key=lambda item: (len(item), item),
    )
    return bool(
        len(edges) == 8
        and target_origin_id in origin_ids
        and target_origin_id != source.component_id
        and set(middle_senders)
        == {
            stepper.component_id,
            *(origin_ids - {source.component_id, target_origin_id}),
        }
        and len(middle_senders) == 3
        and tuple(
            sender
            for sender, receiver in edges
            if receiver == terminator.component_id
        )
        == (middle.component_id,)
        and follower_sender_sets
        == [
            (terminator.component_id,),
            tuple(sorted((source.component_id, terminator.component_id))),
        ]
    )


def _dynamic_modulation_lowering_eligible(kernel: KernelIR) -> bool:
    """Whether the first exact controlled-finished program can be materialized.

    Structural eligibility is independent of executable flags so the same
    predicate can authenticate the provisional declaration and its final
    executable replacement.
    """

    dynamic_finished = tuple(
        value
        for value in kernel.finished_values
        if value.predicate_kind
        == "execution_count_at_least_effective_parameter"
    )
    node_component_ids = tuple(
        node.component_id for node in kernel.graph.nodes
    )
    if not (
        kernel.graph.metadata.get("schedule_kind") == "dynamic_lane_local"
        and kernel.graph.fusion_kind == STATEFUL_GRAPH_FUSION
        and kernel.lane_layout.kind == STATEFUL_LANE_LAYOUT
        and not kernel.rng_streams
        and kernel.schedule_trace is None
        and len(kernel.modulations) == 1
        and len(kernel.effective_parameters) == 1
        and len(dynamic_finished) == 1
        and all(type(component_id) is int for component_id in node_component_ids)
        and node_component_ids == tuple(range(len(kernel.graph.nodes)))
    ):
        return False

    modulation = kernel.modulations[0]
    try:
        source = kernel.graph.node(modulation.source)
        controller = kernel.graph.node(modulation.controller)
        target = kernel.graph.node(modulation.target)
        target_spec = kernel.op_specs.lookup_spec(target.attrs["spec_key"])
    except (BatchedOpSpecError, KeyError):
        return False
    if not (
        isinstance(target_spec, MechanismOpSpec)
        and target_spec.can_step
        and _dynamic_count_controller_eligible(kernel, controller)
        and _dynamic_controller_shape_eligible(
            kernel,
            controller,
            modulation,
        )
        and target.attrs.get("termination_input_node") == source.name
        and not target.attrs.get("diagnostics")
        and modulation.controller not in kernel.graph.execution_order
        and modulation.target in kernel.graph.execution_order
        and modulation.source in kernel.graph.execution_order
    ):
        return False

    scheduler_by_id = {
        condition.component_id: condition for condition in kernel.scheduler
    }
    source_condition = scheduler_by_id.get(modulation.source_component_id)
    controller_condition = scheduler_by_id.get(
        modulation.controller_component_id
    )
    target_condition = scheduler_by_id.get(modulation.target_component_id)
    finished = dynamic_finished[0]
    followers = tuple(
        condition
        for condition in kernel.scheduler
        if condition.condition_type == "WhenFinished"
        and condition.dependency_component_ids
        == (modulation.target_component_id,)
        and condition.finished_value_ids == (finished.value_id,)
    )
    if (
        source_condition is None
        or controller_condition is None
        or target_condition is None
        or len(followers) != 1
    ):
        return False
    follower_condition = followers[0]
    follower = kernel.graph.node(follower_condition.node)

    role_ids = {
        modulation.source_component_id,
        modulation.controller_component_id,
        modulation.target_component_id,
        follower_condition.component_id,
    }
    prelude_ids = tuple(
        node.component_id
        for node in kernel.graph.nodes
        if node.component_id not in role_ids
    )
    if len(prelude_ids) != 1:
        return False
    prelude_id = prelude_ids[0]
    prelude = kernel.graph.node(
        scheduler_by_id[prelude_id].node
    ) if prelude_id in scheduler_by_id else None
    if prelude is None:
        return False

    def condition_matches(condition, condition_type, attrs) -> bool:
        return bool(
            condition.condition_type == condition_type
            and condition.region == "pass"
            and not condition.dependencies
            and not condition.dependency_component_ids
            and not condition.finished_value_ids
            and _kernel_attribute_matches_exactly(
                condition.attrs,
                attrs,
            )
        )

    at_pass_zero = {
        "pass_index": 0,
        "time_scale": "ENVIRONMENT_STATE_UPDATE",
    }
    prelude_condition = scheduler_by_id[prelude_id]
    if not (
        condition_matches(source_condition, "AtPass", at_pass_zero)
        and condition_matches(controller_condition, "AtPass", at_pass_zero)
        and condition_matches(prelude_condition, "AtPass", at_pass_zero)
        and condition_matches(target_condition, "Always", {})
        and follower_condition.region == "pass"
        and follower_condition.dependencies == (target.name,)
        and follower_condition.dependency_component_ids
        == (modulation.target_component_id,)
        and follower_condition.finished_value_ids == (finished.value_id,)
        and _kernel_attribute_matches_exactly(
            follower_condition.attrs,
            {"predicate": "is_finished"},
        )
    ):
        return False

    consideration_sets = tuple(
        sorted(
            kernel.consideration_sets,
            key=lambda item: item.consideration_set_id,
        )
    )
    if (
        len(consideration_sets) != 4
        or tuple(item.consideration_set_id for item in consideration_sets)
        != (0, 1, 2, 3)
        or any(
            item.region != "pass" or item.inputs_frozen is not True
            for item in consideration_sets
        )
        or any(
            item.component_ids
            != tuple(
                kernel.graph.node(name).component_id
                for name in item.nodes
            )
            for item in consideration_sets
        )
        or set(consideration_sets[0].component_ids)
        != {modulation.source_component_id, prelude_id}
        or len(consideration_sets[0].nodes) != 2
        or len(set(consideration_sets[0].nodes)) != 2
        or len(consideration_sets[0].component_ids) != 2
        or len(set(consideration_sets[0].component_ids)) != 2
        or consideration_sets[1].nodes != (controller.name,)
        or consideration_sets[1].component_ids
        != (modulation.controller_component_id,)
        or consideration_sets[2].nodes != (target.name,)
        or consideration_sets[2].component_ids
        != (modulation.target_component_id,)
        or consideration_sets[3].nodes != (follower.name,)
        or consideration_sets[3].component_ids
        != (follower_condition.component_id,)
    ):
        return False
    expected_execution_order = (
        *consideration_sets[0].nodes,
        target.name,
        follower.name,
    )
    if kernel.graph.execution_order != expected_execution_order:
        return False

    if (
        len(kernel.schedule_regions) != 2
        or tuple(
            (region.name, region.kind, region.time_scale, region.parent)
            for region in kernel.schedule_regions
        )
        != (
            ("trial", "trial", "ENVIRONMENT_STATE_UPDATE", ""),
            ("pass", "pass", "PASS", "trial"),
        )
    ):
        return False
    if len(kernel.termination) != 2:
        return False
    termination_by_scale = {
        termination.time_scale: termination
        for termination in kernel.termination
    }
    trial_termination = termination_by_scale.get("ENVIRONMENT_STATE_UPDATE")
    run_termination = termination_by_scale.get("ENVIRONMENT_SEQUENCE")
    all_component_ids = tuple(
        node.component_id for node in kernel.graph.nodes
    )
    if not (
        len(termination_by_scale) == len(kernel.termination)
        and trial_termination is not None
        and trial_termination.condition_type == "AllHaveRun"
        and _kernel_attribute_matches_exactly(
            trial_termination.dependency_component_ids,
            all_component_ids,
        )
        and _kernel_attribute_matches_exactly(
            trial_termination.attrs,
            {},
        )
        and run_termination is not None
        and run_termination.condition_type == "Never"
        and _kernel_attribute_matches_exactly(
            run_termination.dependency_component_ids,
            (),
        )
        and _kernel_attribute_matches_exactly(
            run_termination.attrs,
            {},
        )
    ):
        return False

    target_state_ids = tuple(
        state.state_id
        for state in kernel.states
        if state.component_id == modulation.target_component_id
    )
    if (
        not target_state_ids
        or any(
            state.component_id != modulation.target_component_id
            for state in kernel.states
        )
        or len(kernel.resets) != 1
        or kernel.resets[0].component_id != modulation.target_component_id
        or kernel.resets[0].state_ids != target_state_ids
        or kernel.resets[0].condition_type not in {"AtTrialStart", "Never"}
        or kernel.resets[0].region != "trial"
        or not _kernel_attribute_matches_exactly(kernel.resets[0].attrs, {})
    ):
        return False

    projection_pairs = tuple(
        (projection.sender_component_id, projection.receiver_component_id)
        for projection in kernel.graph.projections
    )
    return bool(
        len(kernel.graph.nodes) == 5
        and prelude.component_type == "TransferMechanism"
        and follower.component_type == "TransferMechanism"
        and source.component_id == modulation.source_component_id
        and controller.component_id == modulation.controller_component_id
        and target.component_id == modulation.target_component_id
        and _kernel_attribute_matches_exactly(
            projection_pairs,
            (
                (prelude_id, modulation.target_component_id),
                (modulation.target_component_id, follower_condition.component_id),
            ),
        )
        and {input_spec.component_id for input_spec in kernel.inputs}
        == {modulation.source_component_id, prelude_id}
        and len(kernel.outputs) == 1
        and kernel.outputs[0].component_id == follower_condition.component_id
    )


def _dynamic_count_controller_eligible(kernel: KernelIR, controller) -> bool:
    """Require a count transform whose fp32 ceil is semantically stable."""

    spec_key = controller.attrs.get("spec_key", "")
    if not spec_key:
        return bool(
            controller.function_type == "Identity"
            and not controller.params
            and controller.attrs.get("spec_kind") == "control"
            and controller.attrs.get("control_function") == "identity"
        )
    try:
        implementation = kernel.op_specs.lookup_spec(spec_key)
    except BatchedOpSpecError:
        return False
    if (
        not isinstance(implementation, ElementwiseFunctionSpec)
        or implementation.function_class.__name__ != "Linear"
        or controller.function_type != "Linear"
        or controller.attrs.get("spec_kind") != "control"
        or controller.attrs.get("control_function") != "registered"
    ):
        return False

    expected_arguments = tuple(binding.arg for binding in implementation.params)
    if (
        expected_arguments != ("slope", "intercept", "scale", "offset")
        or tuple(controller.params) != expected_arguments
    ):
        return False
    parameters_by_name = {parameter.name: parameter for parameter in kernel.params}
    try:
        bound = {
            argument: parameters_by_name[controller.params[argument]]
            for argument in expected_arguments
        }
    except KeyError:
        return False
    if any(
        parameter.owner_component_id != controller.component_id
        or parameter.owner_scope != binding.scope
        or parameter.runtime_mutable
        for parameter, binding in zip(bound.values(), implementation.params)
    ):
        return False

    intercept = bound["intercept"].default
    return bool(
        all(type(parameter.default) is float for parameter in bound.values())
        and bound["slope"].default == 1.0
        and bound["scale"].default == 1.0
        and bound["offset"].default == 0.0
        and np.isfinite(intercept)
        and 0.0 <= intercept <= FP32_EXACT_INTEGER_LIMIT
        and float(intercept).is_integer()
        and (
            intercept == 0.0
            or kernel.max_steps < FP32_EXACT_INTEGER_LIMIT
        )
    )


def _dynamic_controller_shape_eligible(
    kernel: KernelIR,
    controller,
    modulation: BatchedModulationSpec,
) -> bool:
    """Authenticate the complete scalar absorbed-controller port surface."""

    ports_by_id = {port.port_id: port for port in kernel.ports}
    input_port = ports_by_id.get(modulation.controller_input_port_id)
    signal_port = ports_by_id.get(modulation.control_signal_port_id)
    if input_port is None or signal_port is None:
        return False
    expected_absorbed_control = {
        "source": modulation.source,
        "target": modulation.target,
        "parameter": modulation.target_parameter,
        "modulation": modulation.mode,
    }
    return bool(
        type(controller.input_width) is int
        and controller.input_width == 1
        and type(controller.output_width) is int
        and controller.output_width == 1
        and controller.combine == "sum"
        and controller.input_port_ids
        == (modulation.controller_input_port_id,)
        and controller.output_port_ids == (modulation.control_signal_port_id,)
        and len(controller.parameter_port_ids) == len(controller.params)
        and {name for name, _ in controller.parameter_port_ids}
        == set(controller.params)
        and _kernel_attribute_matches_exactly(
            controller.attrs.get("input_ports"),
            ((input_port.name, 1, "sum", input_port.port_id, 0, 1),),
        )
        and _kernel_attribute_matches_exactly(
            controller.attrs.get("output_ports"),
            (signal_port.name,),
        )
        and _kernel_attribute_matches_exactly(
            controller.attrs.get("absorbed_control"),
            expected_absorbed_control,
        )
    )


def _state_kernel_values(graph: BatchedGraphIR) -> tuple[KernelValue, ...]:
    state_slots: dict[int, int] = {}
    values = []
    for state in graph.states:
        component_id = _state_component_id(graph, state)
        state_slot = state_slots.get(component_id, 0)
        state_slots[component_id] = state_slot + 1
        values.append(
            KernelValue(
                f"n{component_id}:state:{state_slot}",
                state.width,
            )
        )
    return tuple(values)


def _trial_reset_ops(
    graph: BatchedGraphIR,
    state_values: tuple[KernelValue, ...],
) -> tuple[KernelOp, ...]:
    values_by_state_id = {
        state.state_id: value
        for state, value in zip(graph.states, state_values)
    }
    reset_ops = []
    for reset in graph.resets:
        if reset.condition_type != "AtTrialStart":
            continue
        try:
            outputs = tuple(
                values_by_state_id[state_id]
                for state_id in reset.state_ids
            )
        except KeyError as error:
            raise ValueError(
                f"Batched reset for '{reset.node}' references an undeclared "
                "state ID."
            ) from error
        reset_ops.append(
            KernelOp(
                kind="ResetState",
                target=reset.node,
                outputs=outputs,
                attrs={
                    "component_id": reset.component_id,
                    "state_ids": reset.state_ids,
                    "condition_type": reset.condition_type,
                    "region": reset.region,
                },
            )
        )
    return tuple(reset_ops)


def _trace_budget(graph: BatchedGraphIR, key: str, default: int) -> int:
    value = graph.metadata.get(key, default)
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"KernelIR metadata '{key}' must be a positive non-bool integer."
        )
    return min(value, default)


def _precomputed_trace_eligible(
    graph: BatchedGraphIR,
    lane_layout: KernelLaneLayout,
    rng_streams: tuple[KernelRngStream, ...],
    op_specs: BatchedOpSpecSnapshot,
) -> bool:
    """Defensive boundary for the first executable trace tier.

    Capability analysis remains the primary gate, but direct/public IR callers
    can construct inconsistent records.  Do not turn one of those records into
    executable nested bodies merely because its metadata requests a trace.
    """

    if not graph.executable or rng_streams:
        return False

    stateless = (
        graph.fusion_kind == STATELESS_GRAPH_FUSION
        and lane_layout.kind == TRIAL_LANE_LAYOUT
        and not graph.states
        and not graph.resets
        and not graph.finished_values
    )
    counted_stateful = _counted_stateful_trace_eligible(
        graph,
        lane_layout,
        op_specs,
    )
    if not stateless and not counted_stateful:
        return False

    try:
        node_ids = tuple(_component_id(graph, node) for node in graph.nodes)
        execution_ids = tuple(
            _component_id(graph, graph.node(node_name))
            for node_name in graph.execution_order
        )
    except (KeyError, TypeError, ValueError):
        return False
    scheduler_ids = tuple(condition.component_id for condition in graph.scheduler)
    consideration_ids = tuple(
        component_id
        for consideration_set in graph.consideration_sets
        for component_id in consideration_set.component_ids
    )
    expected_ids = tuple(sorted(node_ids))
    return (
        len(set(node_ids)) == len(node_ids)
        and tuple(sorted(execution_ids)) == expected_ids
        and tuple(sorted(scheduler_ids)) == expected_ids
        and tuple(sorted(consideration_ids)) == expected_ids
        and all(
            node.attrs.get("spec_kind") in {"elementwise", "mechanism"}
            and not node.attrs.get("diagnostics")
            for node in graph.nodes
        )
    )


def _counted_stateful_trace_eligible(
    graph: BatchedGraphIR,
    lane_layout: KernelLaneLayout,
    op_specs: BatchedOpSpecSnapshot,
) -> bool:
    """Defensive boundary for compile-time counted stateful schedules."""

    if (
        graph.fusion_kind != STATEFUL_GRAPH_FUSION
        or lane_layout.kind != STATEFUL_LANE_LAYOUT
        or not graph.states
        or not graph.finished_values
    ):
        return False

    state_component_ids = {state.component_id for state in graph.states}
    states_by_component_id = {
        component_id: tuple(
            state
            for state in graph.states
            if state.component_id == component_id
        )
        for component_id in state_component_ids
    }
    if len(graph.resets) != len(state_component_ids):
        return False
    reset_component_ids = []
    reset_state_ids = []
    for reset in graph.resets:
        try:
            node = next(
                node
                for node in graph.nodes
                if _component_id(graph, node) == reset.component_id
            )
            expected_states = states_by_component_id[reset.component_id]
        except (KeyError, StopIteration, TypeError, ValueError):
            return False
        expected_state_ids = tuple(state.state_id for state in expected_states)
        if (
            reset.node != node.name
            or reset.condition_type not in {"Never", "AtTrialStart"}
            or reset.state_ids != expected_state_ids
            or reset.attrs
            or reset.region != "trial"
        ):
            return False
        reset_component_ids.append(reset.component_id)
        reset_state_ids.extend(reset.state_ids)
    if (
        len(set(reset_component_ids)) != len(reset_component_ids)
        or tuple(sorted(reset_state_ids))
        != tuple(sorted(state.state_id for state in graph.states))
    ):
        return False

    finished_component_ids = {
        value.component_id
        for value in graph.finished_values
        if (
            value.predicate_kind == "execution_count_at_least"
            and value.storage == "combinational"
            and value.width == 1
            and value.dtype == "bool"
        )
    }
    if not (
        state_component_ids
        and state_component_ids == set(reset_component_ids) == finished_component_ids
    ):
        return False

    for component_id in finished_component_ids:
        try:
            node = next(
                node for node in graph.nodes if _component_id(graph, node) == component_id
            )
            spec = op_specs.lookup_spec(node.attrs["spec_key"])
        except (KeyError, StopIteration, TypeError, ValueError):
            return False
        if (
            not isinstance(spec, MechanismOpSpec)
            or not spec.can_step
            or spec.trial_states
            or node.attrs.get("diagnostics")
        ):
            return False
    return True


def _precomputed_trace_ops(
    graph: BatchedGraphIR,
    trial_ops: tuple[KernelOp, ...],
    trace: BatchedScheduleTraceSpec,
    *,
    op_specs: BatchedOpSpecSnapshot,
    component_budget: int,
    weighted_op_budget: int,
) -> tuple[KernelOp, ...]:
    component_bodies, epilogue = _partition_trace_trial_ops(graph, trial_ops)
    component_weights = {
        component_id: sum(_kernel_op_source_weight(op) for op in body)
        for component_id, body in component_bodies.items()
    }

    # Compute the source-expansion proxy before constructing repeated nested
    # bodies.  The component planner bounds scheduler work; this second bound
    # accounts for wide and projection-heavy component bodies whose emitted
    # Triton source can be much larger than their component count suggests.
    weighted_expansion = 1 + sum(
        _kernel_op_source_weight(op) for op in epilogue
    )
    for step in trace.steps:
        weighted_expansion += 1
        for component_id in step.component_ids:
            try:
                weighted_expansion += component_weights[component_id]
            except KeyError as error:
                raise ValueError(
                    "KernelIR precomputed trace references component id "
                    f"{component_id}, which has no lowered component body."
                ) from error
    if weighted_expansion > weighted_op_budget:
        raise ValueError(
            "KernelIR precomputed trace weighted op expansion "
            f"{weighted_expansion} exceeds budget {weighted_op_budget}."
        )

    step_component_ids = {
        value.component_id
        for value in graph.finished_values
        if value.predicate_kind == "execution_count_at_least"
    }
    _validate_scheduled_step_bodies(
        graph,
        component_bodies,
        step_component_ids,
        op_specs,
    )
    component_execution_counts = dict.fromkeys(component_bodies, 0)
    executions = []
    for step in trace.steps:
        body = []
        for component_id in step.component_ids:
            execution_index = component_execution_counts[component_id]
            body.extend(
                _scheduled_component_op(
                    graph,
                    op,
                    component_id=component_id,
                    execution_index=execution_index,
                    step_component_ids=step_component_ids,
                )
                for op in component_bodies[component_id]
            )
            component_execution_counts[component_id] += 1
        executions.append(
            KernelOp(
                kind="ExecuteConsiderationSet",
                target=(
                    f"pass-{step.pass_index}:consideration-set-"
                    f"{step.consideration_set_id}"
                ),
                attrs={
                    "pass_index": step.pass_index,
                    "consideration_set_id": step.consideration_set_id,
                    "component_ids": step.component_ids,
                    "body": tuple(body),
                },
            )
        )
    pass_regions = tuple(
        region for region in graph.schedule_regions if region.kind == "pass"
    )
    if len(pass_regions) != 1:
        raise ValueError(
            "KernelIR precomputed trace requires exactly one typed pass region, "
            f"found {len(pass_regions)}."
        )
    pass_region = pass_regions[0]
    return (
        KernelOp(
            kind="ForPasses",
            target="passes",
            attrs={
                "region": pass_region,
                "body": tuple(executions),
                "declaration_only": False,
                "trace_kind": "precomputed",
                "component_expansion_budget": component_budget,
                "weighted_op_expansion": weighted_expansion,
                "weighted_op_expansion_budget": weighted_op_budget,
            },
        ),
        *epilogue,
    )


def _validate_scheduled_step_bodies(
    graph: BatchedGraphIR,
    component_bodies: Mapping[int, tuple[KernelOp, ...]],
    step_component_ids: set[int],
    op_specs: BatchedOpSpecSnapshot,
) -> None:
    for component_id in step_component_ids:
        calls = tuple(
            op
            for op in component_bodies.get(component_id, ())
            if op.kind == "CallMechanism"
        )
        if len(calls) != 1:
            raise ValueError(
                "KernelIR counted finished component id "
                f"{component_id} requires exactly one CallMechanism body, "
                f"found {len(calls)}."
            )
        try:
            spec = op_specs.lookup_spec(calls[0].attrs["spec_key"])
        except KeyError as error:
            raise ValueError(
                "KernelIR counted finished component id "
                f"{component_id} has no frozen mechanism implementation."
            ) from error
        if not isinstance(spec, MechanismOpSpec) or not spec.can_step:
            raise ValueError(
                "KernelIR counted finished component id "
                f"{component_id} has no one-step mechanism implementation."
            )
        state_ids = tuple(
            sorted(
                state.state_id
                for state in graph.states
                if state.component_id == component_id
            )
        )
        if not state_ids:
            raise ValueError(
                "KernelIR counted finished component id "
                f"{component_id} has no retained state declarations."
            )


def _scheduled_component_op(
    graph: BatchedGraphIR,
    op: KernelOp,
    *,
    component_id: int,
    execution_index: int,
    step_component_ids: set[int],
) -> KernelOp:
    if component_id not in step_component_ids or op.kind != "CallMechanism":
        return op
    state_ids = tuple(
        sorted(
            state.state_id
            for state in graph.states
            if state.component_id == component_id
        )
    )
    return KernelOp(
        kind="StepMechanism",
        target=op.target,
        inputs=op.inputs,
        outputs=op.outputs,
        attrs={
            **op.attrs,
            "state_ids": state_ids,
            "execution_index": execution_index,
            "active_lanes": "all",
        },
    )


def _partition_trace_trial_ops(
    graph: BatchedGraphIR,
    trial_ops: tuple[KernelOp, ...],
):
    component_bodies = {
        _component_id(graph, node): []
        for node in graph.nodes
    }
    epilogue = []
    for op in trial_ops:
        if op.kind in {"StoreOutput", "StoreFlag"}:
            epilogue.append(op)
            continue
        try:
            node = graph.node(op.target)
        except KeyError as error:
            raise ValueError(
                "KernelIR cannot assign op "
                f"'{op.kind}' target '{op.target}' to a component body."
            ) from error
        component_id = _component_id(graph, node)
        component_bodies[component_id].append(_trace_component_op(op))

    empty = tuple(
        sorted(
            component_id
            for component_id, body in component_bodies.items()
            if not body
        )
    )
    if empty:
        raise ValueError(
            "KernelIR precomputed trace has no lowered body for component id(s) "
            f"{empty}."
        )
    return (
        {component_id: tuple(body) for component_id, body in component_bodies.items()},
        tuple(epilogue),
    )


def _trace_component_op(op: KernelOp) -> KernelOp:
    if op.kind != "CallFunction" or "onset_step" not in op.attrs:
        return op
    return KernelOp(
        kind=op.kind,
        target=op.target,
        inputs=op.inputs,
        outputs=op.outputs,
        attrs={
            key: value
            for key, value in op.attrs.items()
            if key != "onset_step"
        },
    )


def _kernel_op_source_weight(op: KernelOp) -> int:
    input_width = sum(value.width for value in op.inputs)
    output_width = sum(value.width for value in op.outputs)
    if op.kind == "CallProjection" and op.inputs and op.outputs:
        work = op.inputs[0].width * op.outputs[0].width
    elif op.kind in {"CombineSum", "CombineProduct"} and op.outputs:
        work = len(op.inputs) * op.outputs[0].width
    else:
        work = max(input_width, output_width)
    return 1 + max(1, work)


def _requires_pass_region(graph: BatchedGraphIR) -> bool:
    """Whether scheduler predicates require an explicit per-trial pass loop."""

    if not graph.executable:
        return True
    if graph.metadata.get("scheduler_requires_pass_region", False):
        return True
    if graph.fusion_kind == COEVOLVING_GRAPH_FUSION:
        return True

    consideration_set_ids = {
        component_id: consideration_set.consideration_set_id
        for consideration_set in graph.consideration_sets
        for component_id in consideration_set.component_ids
    }
    for condition in graph.scheduler:
        if condition.condition_type == "AtPass":
            if (
                condition.attrs.get("pass_index") != 0
                or condition.attrs.get("time_scale")
                != "ENVIRONMENT_STATE_UPDATE"
            ):
                return True
        elif condition.condition_type == "WhenFinished":
            target_set = condition.consideration_set_id
            if any(
                consideration_set_ids.get(component_id, target_set) >= target_set
                for component_id in condition.dependency_component_ids
            ):
                return True
        elif condition.condition_type in {"EveryNCalls", "AllEveryNCalls"}:
            target_set = condition.consideration_set_id
            if any(
                consideration_set_ids.get(component_id, target_set) >= target_set
                for component_id in condition.dependency_component_ids
            ):
                return True
        elif condition.condition_type not in {"Always", "AtTrialStart"}:
            return True
    return False


def _graph_spec_keys(graph: BatchedGraphIR) -> tuple[str, ...]:
    """Return each registry key referenced by ``graph``, in graph order."""

    keys = [
        graph.node(node_name).attrs["spec_key"]
        for node_name in graph.execution_order
    ]
    keys.extend(
        projection.spec_key
        for projection in graph.projections
        if projection.spec_key
    )
    keys.extend(
        state.function_initializer.spec_key
        for state in graph.states
        if state.function_initializer is not None
    )
    keys.extend(
        modulation.controller_function_spec_key
        for modulation in graph.modulations
        if modulation.controller_function_spec_key
    )
    return tuple(dict.fromkeys(keys))


def diag_slots(kernel: KernelIR) -> tuple[tuple[str, str], ...]:
    """Return the `(node, diagnostic_name)` for each diagnostic slot, by slot index.

    These are the per-lane flags the bounded-loop ops emit through `StoreFlag`
    (currently DDM truncation); the tuple's length is the diagnostic buffer's
    inner width and its order matches the slot indices written by the kernel.
    """

    slots: dict[int, tuple[str, str]] = {}
    for op in iter_kernel_ops(kernel):
        if op.kind == "StoreFlag":
            slots[int(op.attrs["slot"])] = (op.attrs["node"], op.attrs["name"])
    return tuple(slots[idx] for idx in range(len(slots)))


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
    if fusion_kind in (STATEFUL_GRAPH_FUSION, COEVOLVING_GRAPH_FUSION):
        return KernelLaneLayout(
            kind=STATEFUL_LANE_LAYOUT,
            dimensions=("parameter_set", "subject", "estimate"),
        )
    return KernelLaneLayout(
        kind=TRIAL_LANE_LAYOUT,
        dimensions=("parameter_set", "subject", "trial", "estimate"),
    )


def _rng_streams(graph: BatchedGraphIR) -> tuple[KernelRngStream, ...]:
    if graph.rng_streams:
        return tuple(
            KernelRngStream(
                name=stream.name,
                node=stream.node,
                width=stream.width,
                step_extent=stream.step_extent,
                component_id=stream.component_id,
                stream_id=stream.stream_id,
            )
            for stream in graph.rng_streams
        )

    streams = []
    for node_name in graph.execution_order:
        node = graph.node(node_name)
        for stream_name, step_extent, width in node.attrs.get("rng_streams", ()):
            streams.append(
                KernelRngStream(
                    name=f"{node.name}.{stream_name}",
                    node=node.name,
                    width=int(width),
                    step_extent=step_extent,
                    component_id=_component_id(graph, node),
                    stream_id=len(streams),
                )
            )
    return tuple(streams)


def component_symbol(graph: BatchedGraphIR, node_or_name) -> str:
    """Backend-safe lowering-local symbol prefix for a graph component.

    Component display names remain the graph lookup and decorator contract.  A
    numeric prefix is used for generated symbols so target-language identifier
    sanitization can never merge distinct PNL components.
    """

    return f"n{_component_id(graph, node_or_name)}"


def node_input_value_name(graph: BatchedGraphIR, node_or_name) -> str:
    return f"{component_symbol(graph, node_or_name)}:input"


def _node_input_port_layout(graph: BatchedGraphIR, node):
    """Per-port semantic layout with a legacy fallback for direct IR callers."""

    layout = tuple(node.attrs.get("input_ports", ()))
    if layout:
        return layout

    input_spec = next(
        (
            candidate
            for candidate in graph.inputs
            if candidate.component_id == node.component_id
            or candidate.node == node.name
        ),
        None,
    )
    projection = next(
        (
            candidate
            for candidate in graph.projections
            if candidate.receiver_component_id == node.component_id
            or candidate.receiver == node.name
        ),
        None,
    )
    port_name = (
        input_spec.port
        if input_spec is not None and input_spec.port
        else projection.receiver_port
        if projection is not None
        else "InputPort-0"
    )
    port_id = (
        input_spec.port_id
        if input_spec is not None
        else projection.receiver_port_id
        if projection is not None
        else -1
    )
    return ((port_name, node.input_width, node.combine, port_id, 0, node.input_width),)


def _input_spec_for_port(graph: BatchedGraphIR, node, port_id: int):
    matches = tuple(
        input_spec
        for input_spec in graph.inputs
        if (
            port_id >= 0
            and input_spec.port_id == port_id
        )
        or (
            port_id < 0
            and (
                input_spec.component_id == node.component_id
                or input_spec.node == node.name
            )
        )
    )
    if len(matches) != 1:
        raise ValueError(
            f"Batched node '{node.name}' InputPort id {port_id} requires exactly "
            f"one external input spec, found {len(matches)}."
        )
    return matches[0]


def node_output_value_name(graph: BatchedGraphIR, node_or_name, port: str) -> str:
    node = graph.node(node_or_name) if isinstance(node_or_name, str) else node_or_name
    output_ports = tuple(name for name, _ in node.attrs.get("op_outputs", ()))
    if not output_ports:
        output_ports = tuple(node.attrs.get("output_ports", ())) or ("RESULT",)
    try:
        port_slot = output_ports.index(port)
    except ValueError as error:
        raise ValueError(
            f"Batched node '{node.name}' has no lowered output port '{port}'."
        ) from error
    return f"{component_symbol(graph, node)}:output:{port_slot}"


def node_diagnostic_value_name(
    graph: BatchedGraphIR,
    node_or_name,
    diagnostic_slot: int,
) -> str:
    return f"{component_symbol(graph, node_or_name)}:diagnostic:{diagnostic_slot}"


def _component_id(graph: BatchedGraphIR, node_or_name) -> int:
    node = graph.node(node_or_name) if isinstance(node_or_name, str) else node_or_name
    component_id = int(node.component_id)
    if component_id >= 0:
        return component_id
    # Preserve direct construction of the public experimental IR dataclasses:
    # old callers that omit ``component_id`` still receive a distinct numeric
    # identity, while normal Composition lowering always assigns one explicitly.
    for fallback_id, candidate in enumerate(graph.nodes):
        if candidate is node or candidate.name == node.name:
            return fallback_id
    raise KeyError(node.name)


def _state_component_id(graph: BatchedGraphIR, state: BatchedStateSpec) -> int:
    if state.component_id >= 0:
        return int(state.component_id)
    return _component_id(graph, state.node)


def _compose_component_trial_body_ops(
    graph: BatchedGraphIR,
    component_names: Iterable[str],
    *,
    diagnostic_slot: int = 0,
) -> tuple[KernelOp, ...]:
    ops: list[KernelOp] = []
    diag_slot = diagnostic_slot
    for node_name in component_names:
        node = graph.node(node_name)
        node_input = KernelValue(node_input_value_name(graph, node), node.input_width)
        input_ports = _node_input_port_layout(graph, node)
        port_values = []
        for port_slot, (
            port_name,
            port_width,
            combine,
            port_id,
            flat_start,
            flat_stop,
        ) in enumerate(input_ports):
            port_value = (
                node_input
                if len(input_ports) == 1
                else KernelValue(
                    f"{component_symbol(graph, node)}:input-port:{port_slot}",
                    port_width,
                )
            )
            port_values.append(port_value)
            projections = projection_inputs(
                graph,
                node.name,
                receiver_port_id=port_id,
            )
            if projections:
                projected_values = []
                for projection in projections:
                    projected = KernelValue(
                        f"{component_symbol(graph, node)}:projection:"
                        f"{projection.projection_id}",
                        projection.matrix.shape[1],
                    )
                    if projected.width != port_width:
                        raise ValueError(
                            f"Batched projection into '{node.name}.{port_name}' has "
                            f"width {projected.width}, expected {port_width}."
                        )
                    projected_values.append(projected)
                    ops.append(
                        KernelOp(
                            kind="CallProjection",
                            target=projection.receiver,
                            inputs=(
                                KernelValue(
                                    node_output_value_name(
                                        graph,
                                        projection.sender,
                                        projection.sender_port,
                                    ),
                                    projection.matrix.shape[0],
                                ),
                            ),
                            outputs=(projected,),
                            attrs={
                                "sender": projection.sender,
                                "sender_port": projection.sender_port,
                                "receiver": projection.receiver,
                                "receiver_port": projection.receiver_port,
                                "projection_id": projection.projection_id,
                                "sender_component_id": projection.sender_component_id,
                                "sender_port_id": projection.sender_port_id,
                                "receiver_component_id": projection.receiver_component_id,
                                "receiver_port_id": projection.receiver_port_id,
                                # Snapshot independently from GraphIR so an op
                                # mutation cannot rewrite its own validation
                                # authority through ndarray aliasing.
                                "matrix": np.array(projection.matrix, copy=True),
                                "projection_type": "MappingProjection",
                                "spec_key": projection.spec_key,
                            },
                        )
                    )
                ops.append(
                    KernelOp(
                        kind=(
                            "CombineProduct"
                            if combine == "product"
                            else "CombineSum"
                        ),
                        target=node.name,
                        inputs=tuple(projected_values),
                        outputs=(port_value,),
                        attrs={
                            "component_id": node.component_id,
                            "receiver_port": port_name,
                            "receiver_port_id": port_id,
                            "flat_start": flat_start,
                            "flat_stop": flat_stop,
                        },
                    )
                )
            else:
                input_spec = _input_spec_for_port(graph, node, port_id)
                ops.append(
                    KernelOp(
                        kind="LoadInput",
                        target=node.name,
                        outputs=(port_value,),
                        attrs={
                            "node": node.name,
                            "input_name": input_spec.name,
                            "width": port_width,
                            "component_id": node.component_id,
                            "port": port_name,
                            "port_id": port_id,
                            "flat_start": flat_start,
                            "flat_stop": flat_stop,
                        },
                    )
                )

        if len(port_values) > 1:
            ops.append(
                KernelOp(
                    kind="Concatenate",
                    target=node.name,
                    inputs=tuple(port_values),
                    outputs=(node_input,),
                    attrs={
                        "component_id": node.component_id,
                        "port_ids": tuple(port[3] for port in input_ports),
                        "ports": tuple(port[0] for port in input_ports),
                    },
                )
            )

        spec_kind = node.attrs.get("spec_kind")
        if spec_kind == "elementwise":
            output_port_slices = _node_output_port_slices(node)
            output_port = output_port_slices[0][0]
            needs_port_extracts = not (
                len(output_port_slices) == 1
                and output_port_slices[0][3] == 0
                and output_port_slices[0][4] == node.output_width
            )
            output_value = (
                KernelValue(
                    f"{component_symbol(graph, node)}:mechanism-value",
                    node.output_width,
                )
                if needs_port_extracts
                else KernelValue(
                    node_output_value_name(graph, node, output_port),
                    node.output_width,
                )
            )
            function_input = node_input
            if "noise" in node.attrs:
                noisy_input = KernelValue(
                    f"{component_symbol(graph, node)}:noise",
                    node.input_width,
                )
                ops.append(
                    add_constant_op(
                        target=node.name,
                        input_value=function_input,
                        output_value=noisy_input,
                        value=node.attrs["noise"],
                    )
                )
                function_input = noisy_input
            function_output = (
                KernelValue(
                    f"{component_symbol(graph, node)}:function",
                    node.output_width,
                )
                if "clip" in node.attrs
                else output_value
            )
            attrs = {
                "component_type": node.component_type,
                "function_type": node.function_type,
                "component_id": node.component_id,
                "params": dict(node.params),
                "output_port": output_port,
                "spec_key": node.attrs["spec_key"],
            }
            if "integrator_pre" in node.attrs:
                attrs["integrator_pre"] = node.attrs["integrator_pre"]
            if "onset_step" in node.attrs:
                attrs["onset_step"] = node.attrs["onset_step"]
            ops.append(
                KernelOp(
                    kind="CallFunction",
                    target=node.name,
                    inputs=(function_input,),
                    outputs=(function_output,),
                    attrs=attrs,
                )
            )
            if "clip" in node.attrs:
                lower, upper = node.attrs["clip"]
                ops.append(
                    clamp_op(
                        target=node.name,
                        input_value=function_output,
                        output_value=output_value,
                        lower=lower,
                        upper=upper,
                    )
                )
            if needs_port_extracts:
                for (
                    port_name,
                    port_width,
                    port_id,
                    flat_start,
                    flat_stop,
                ) in output_port_slices:
                    ops.append(
                        KernelOp(
                            kind="ExtractSlice",
                            target=node.name,
                            inputs=(output_value,),
                            outputs=(
                                KernelValue(
                                    node_output_value_name(graph, node, port_name),
                                    port_width,
                                ),
                            ),
                            attrs={
                                "component_id": node.component_id,
                                "port": port_name,
                                "port_id": port_id,
                                "start": flat_start,
                                "stop": flat_stop,
                            },
                        )
                    )
        elif spec_kind == "mechanism":
            op_outputs = tuple(node.attrs.get("op_outputs", ()))
            rng_streams = tuple(node.attrs.get("rng_streams", ()))
            attrs = {
                "component_type": node.component_type,
                "function_type": node.function_type,
                "component_id": node.component_id,
                "params": dict(node.params),
                "spec_key": node.attrs["spec_key"],
            }
            if rng_streams:
                attrs["step_extent"] = rng_streams[0][1]
            diagnostics = tuple(node.attrs.get("diagnostics", ()))
            if diagnostics:
                attrs["diagnostics"] = diagnostics
                attrs["diagnostic_values"] = tuple(
                    node_diagnostic_value_name(graph, node, index)
                    for index, _ in enumerate(diagnostics)
                )
            ops.append(
                KernelOp(
                    kind="CallMechanism",
                    target=node.name,
                    inputs=(node_input,),
                    outputs=tuple(
                        KernelValue(
                            node_output_value_name(graph, node, port),
                            int(width),
                        )
                        for port, width in op_outputs
                    ),
                    attrs=attrs,
                )
            )
            for diagnostic_index, name in enumerate(diagnostics):
                ops.append(
                    KernelOp(
                        kind="StoreFlag",
                        target=node.name,
                        inputs=(
                            KernelValue(
                                node_diagnostic_value_name(
                                    graph,
                                    node,
                                    diagnostic_index,
                                ),
                                1,
                            ),
                        ),
                        attrs={"node": node.name, "name": name, "slot": diag_slot},
                    )
                )
                diag_slot += 1
        else:
            raise ValueError(
                f"Batched graph node '{node.name}' has no registered batched op spec."
            )

    return tuple(ops)


def _component_trial_body_ops(
    graph: BatchedGraphIR,
    node_name: str,
    *,
    diagnostic_slot: int = 0,
) -> tuple[KernelOp, ...]:
    """Build one component body with an explicit first diagnostic slot."""

    return _compose_component_trial_body_ops(
        graph,
        (node_name,),
        diagnostic_slot=diagnostic_slot,
    )


def _trial_output_ops(graph: BatchedGraphIR) -> tuple[KernelOp, ...]:
    ops = []

    for output in graph.outputs:
        ops.append(
            KernelOp(
                kind="StoreOutput",
                target=output.name,
                inputs=(
                    KernelValue(
                        node_output_value_name(graph, output.node, output.port),
                        output.width,
                    ),
                ),
                attrs={
                    "node": output.node,
                    "port": output.port,
                    "width": output.width,
                    "component_id": output.component_id,
                    "port_id": output.port_id,
                    "flat_start": output.flat_start,
                    "flat_stop": output.flat_stop,
                },
            )
        )
    return tuple(ops)


def _trial_body_ops(graph: BatchedGraphIR) -> tuple[KernelOp, ...]:
    return (
        *_compose_component_trial_body_ops(graph, graph.execution_order),
        *_trial_output_ops(graph),
    )


def _primary_output_port_name(node) -> str:
    output_ports = tuple(node.attrs.get("output_ports", ()))
    if output_ports:
        return output_ports[0]
    return "RESULT"


def _node_output_port_slices(node):
    slices = tuple(node.attrs.get("output_port_slices", ()))
    if slices:
        return slices
    return ((
        _primary_output_port_name(node),
        node.output_width,
        -1,
        0,
        node.output_width,
    ),)
