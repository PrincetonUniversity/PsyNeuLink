"""Backend-neutral planning for lane-invariant batched schedules.

This module deliberately depends only on typed batched IR declarations.  It
does not import PsyNeuLink scheduler or Component classes and never evaluates a
live Condition object.  Capability analysis will consume this planner in a
later checkpoint; defining it alone does not make any new model executable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from psyneulink.core.batched.ir import (
    BatchedConsiderationSetSpec,
    BatchedProjectionSpec,
    BatchedScheduleTraceSpec,
    BatchedScheduleTraceStepSpec,
    BatchedSchedulerSpec,
    BatchedTerminationSpec,
)


_TRIAL_TIME_SCALE = "ENVIRONMENT_STATE_UPDATE"
_SEQUENCE_TIME_SCALE = "ENVIRONMENT_SEQUENCE"
_SUPPORTED_CONDITIONS = {
    "Always",
    "AtPass",
    "AtTrialStart",
    "EveryNCalls",
    "AllEveryNCalls",
}


class BatchedScheduleTraceError(ValueError):
    """A fail-closed precomputed-schedule planning error.

    ``code`` is intentionally backend-neutral so capability analysis can map it
    to a public diagnostic without parsing prose.  Numeric component operands
    are retained for an actionable diagnostic without retaining live objects.
    """

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        component_ids: tuple[int, ...] = (),
    ) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.component_ids = component_ids


def plan_precomputed_schedule_trace(
    *,
    scheduler: Iterable[BatchedSchedulerSpec],
    consideration_sets: Iterable[BatchedConsiderationSetSpec],
    termination: Iterable[BatchedTerminationSpec],
    expansion_budget: int,
    projections: Iterable[BatchedProjectionSpec],
) -> BatchedScheduleTraceSpec:
    """Plan one trial of an exact, lane-invariant scheduler subset.

    Supported execution predicates are ``Always``, ``AtPass(n)`` and
    ``AtTrialStart``, plus the implicit one-call dependency predicates emitted
    for ordinary processing edges.  Only the default ``AllHaveRun`` trial and
    ``Never`` sequence termination contract is accepted.

    Conditions for every member of a consideration set are evaluated against a
    single beginning-of-set usable-count snapshot.  Counts are consumed when
    their owner executes, and selected members update counts only after the
    complete set has been chosen.  The returned trace omits empty set visits but
    retains absolute pass and consideration-set indices.

    The initial executable tier uses one independent lane per trial.  It cannot
    reproduce a projection receiver reading a producer's output from a previous
    trial, so this planner rejects any receiver execution before its producer's
    first execution in the current trace.  Same-set data edges are rejected
    separately because they require frozen current/next value banks.
    """

    if type(expansion_budget) is not int or expansion_budget <= 0:
        _raise(
            "schedule.invalid_expansion_budget",
            "expansion_budget must be a positive non-bool integer",
        )

    scheduler = _typed_tuple(scheduler, BatchedSchedulerSpec, "scheduler")
    consideration_sets = _typed_tuple(
        consideration_sets,
        BatchedConsiderationSetSpec,
        "consideration_sets",
    )
    termination = _typed_tuple(
        termination,
        BatchedTerminationSpec,
        "termination",
    )
    projections = _typed_tuple(
        projections,
        BatchedProjectionSpec,
        "projections",
    )

    (
        ordered_sets,
        component_set_ids,
        component_names,
    ) = _validate_consideration_sets(consideration_sets)
    conditions = _validate_scheduler(
        scheduler,
        component_set_ids,
        component_names,
    )
    terminating_components = _validate_termination(
        termination,
        tuple(sorted(conditions)),
    )
    incoming_senders = _validate_projection_edges(
        projections,
        component_set_ids,
    )

    # Only dependency pairs consumed by a supported condition need storage.
    # Their counts begin at zero at the start of every trial.
    usable_counts = {
        (dependency_id, component_id): 0
        for component_id, condition in conditions.items()
        for dependency_id in condition.dependency_component_ids
    }
    executed_this_trial: set[int] = set()
    trace_steps: list[BatchedScheduleTraceStepSpec] = []
    component_execution_count = 0
    pass_index = 0

    while True:
        any_execution_this_pass = False
        for consideration_set in ordered_sets:
            # Default AllHaveRun is reevaluated between consideration-set
            # executions.  Returning here, rather than after the whole pass,
            # preserves PsyNeuLink's mid-pass termination behavior.
            if terminating_components.issubset(executed_this_trial):
                return _trace(
                    trace_steps,
                    pass_index,
                    component_execution_count,
                )

            # Select the entire set against one snapshot.  Do not let an
            # earlier member's count update make a later member eligible.
            usable_snapshot = dict(usable_counts)
            selected = tuple(
                component_id
                for component_id in consideration_set.component_ids
                if _condition_is_satisfied(
                    conditions[component_id],
                    pass_index,
                    usable_snapshot,
                )
            )
            if not selected:
                continue

            _validate_fresh_reads(
                selected,
                incoming_senders,
                executed_this_trial,
                pass_index,
                consideration_set.consideration_set_id,
            )
            if component_execution_count + len(selected) > expansion_budget:
                _raise(
                    "schedule.expansion_budget_exceeded",
                    "precomputed schedule requires more than "
                    f"{expansion_budget} component executions",
                    component_ids=selected,
                )

            trace_steps.append(
                BatchedScheduleTraceStepSpec(
                    pass_index=pass_index,
                    consideration_set_id=(
                        consideration_set.consideration_set_id
                    ),
                    component_ids=selected,
                )
            )
            component_execution_count += len(selected)
            any_execution_this_pass = True

            selected_set = set(selected)
            # Executing an owner spends every usable dependency count it could
            # observe.  New producer credits are then published for later
            # consideration sets/passes.
            for producer_id, consumer_id in usable_counts:
                if consumer_id in selected_set:
                    usable_counts[(producer_id, consumer_id)] = 0
            for producer_id, consumer_id in usable_counts:
                if producer_id in selected_set:
                    usable_counts[(producer_id, consumer_id)] += 1
            executed_this_trial.update(selected_set)

        if terminating_components.issubset(executed_this_trial):
            return _trace(
                trace_steps,
                pass_index,
                component_execution_count,
            )

        if any_execution_this_pass:
            pass_index += 1
            continue

        # An entirely empty pass cannot change a usable-call predicate.  Jump
        # directly to the next exact AtPass event instead of materializing an
        # arbitrary number of empty visits.
        future_passes = tuple(
            condition.attrs["pass_index"]
            for condition in conditions.values()
            if condition.condition_type == "AtPass"
            and condition.attrs["pass_index"] > pass_index
        )
        if not future_passes:
            _raise(
                "schedule.nonterminating",
                "default AllHaveRun cannot become satisfied from the declared "
                "scheduler predicates",
                component_ids=tuple(
                    sorted(terminating_components - executed_this_trial)
                ),
            )
        pass_index = min(future_passes)


def _typed_tuple(values, expected_type, label: str):
    try:
        values = tuple(values)
    except TypeError:
        _raise(
            "schedule.invalid_declaration",
            f"{label} must be an iterable of {expected_type.__name__}",
        )
    for value in values:
        if type(value) is not expected_type:
            _raise(
                "schedule.invalid_declaration",
                f"{label} contains {type(value).__name__}, expected "
                f"{expected_type.__name__}",
            )
    return values


def _validate_consideration_sets(consideration_sets):
    if not consideration_sets:
        _raise(
            "schedule.invalid_declaration",
            "precomputed scheduling requires at least one consideration set",
        )

    component_set_ids: dict[int, int] = {}
    component_names: dict[int, str] = {}
    for expected_set_id, consideration_set in enumerate(consideration_sets):
        if type(consideration_set.consideration_set_id) is not int:
            _raise(
                "schedule.invalid_declaration",
                "consideration-set IDs must be non-bool integers",
            )
        if consideration_set.consideration_set_id != expected_set_id:
            _raise(
                "schedule.invalid_declaration",
                "consideration-set IDs must be unique, contiguous, and stored "
                "in execution order",
            )
        if consideration_set.region != "pass":
            _raise(
                "schedule.invalid_declaration",
                "precomputed consideration sets must belong to the pass region",
            )
        if consideration_set.inputs_frozen is not True:
            _raise(
                "schedule.invalid_declaration",
                "precomputed consideration sets require frozen start-of-set inputs",
            )
        if not consideration_set.component_ids:
            _raise(
                "schedule.invalid_declaration",
                f"consideration set {expected_set_id} is empty",
            )
        if type(consideration_set.nodes) is not tuple or type(
            consideration_set.component_ids
        ) is not tuple:
            _raise(
                "schedule.invalid_declaration",
                f"consideration set {expected_set_id} nodes and component IDs "
                "must be tuples",
            )
        if len(consideration_set.nodes) != len(consideration_set.component_ids):
            _raise(
                "schedule.invalid_declaration",
                f"consideration set {expected_set_id} has mismatched node and "
                "component-ID counts",
            )
        if consideration_set.component_ids != tuple(
            sorted(consideration_set.component_ids)
        ):
            _raise(
                "schedule.invalid_declaration",
                f"consideration set {expected_set_id} component IDs must be in "
                "deterministic numeric order",
                component_ids=consideration_set.component_ids,
            )

        for node_name, component_id in zip(
            consideration_set.nodes,
            consideration_set.component_ids,
        ):
            if type(node_name) is not str:
                _raise(
                    "schedule.invalid_declaration",
                    f"consideration set {expected_set_id} has a non-string node name",
                )
            _validate_component_id(component_id, "consideration set member")
            if component_id in component_set_ids:
                _raise(
                    "schedule.invalid_declaration",
                    f"component {component_id} belongs to more than one "
                    "consideration set",
                    component_ids=(component_id,),
                )
            component_set_ids[component_id] = expected_set_id
            component_names[component_id] = node_name

    return consideration_sets, component_set_ids, component_names


def _validate_scheduler(scheduler, component_set_ids, component_names):
    conditions: dict[int, BatchedSchedulerSpec] = {}
    for condition in scheduler:
        component_id = condition.component_id
        _validate_component_id(component_id, "scheduler component")
        if component_id not in component_set_ids:
            _raise(
                "schedule.invalid_declaration",
                f"scheduler component {component_id} has no consideration set",
                component_ids=(component_id,),
            )
        if component_id in conditions:
            _raise(
                "schedule.invalid_declaration",
                f"component {component_id} has more than one scheduler predicate",
                component_ids=(component_id,),
            )
        if type(condition.node) is not str or condition.node != component_names[component_id]:
            _raise(
                "schedule.invalid_declaration",
                f"scheduler component {component_id} does not match its "
                "consideration-set node name",
                component_ids=(component_id,),
            )
        if condition.consideration_set_id != component_set_ids[component_id]:
            _raise(
                "schedule.invalid_declaration",
                f"scheduler component {component_id} has the wrong "
                "consideration-set ID",
                component_ids=(component_id,),
            )
        if type(condition.consideration_set_id) is not int:
            _raise(
                "schedule.invalid_declaration",
                f"scheduler component {component_id} has a non-integer "
                "consideration-set ID",
                component_ids=(component_id,),
            )
        if condition.region != "pass":
            _raise(
                "schedule.invalid_declaration",
                f"scheduler component {component_id} is not pass-scoped",
                component_ids=(component_id,),
            )
        _validate_condition(condition, component_set_ids, component_names)
        conditions[component_id] = condition

    missing = tuple(sorted(set(component_set_ids) - set(conditions)))
    extra = tuple(sorted(set(conditions) - set(component_set_ids)))
    if missing or extra:
        _raise(
            "schedule.invalid_declaration",
            f"scheduler predicates do not exactly cover consideration-set "
            f"members; missing={missing}, extra={extra}",
            component_ids=missing + extra,
        )
    return conditions


def _validate_condition(condition, component_set_ids, component_names):
    component_id = condition.component_id
    condition_type = condition.condition_type
    if type(condition_type) is not str:
        _raise(
            "schedule.invalid_declaration",
            f"scheduler predicate type for component {component_id} must be a string",
            component_ids=(component_id,),
        )
    if condition_type not in _SUPPORTED_CONDITIONS:
        _raise(
            "schedule.unsupported_condition",
            f"unsupported precomputed scheduler predicate {condition_type!r}",
            component_ids=(component_id,),
        )
    if condition.finished_value_ids:
        _raise(
            "schedule.unsupported_condition",
            "precomputed scheduler predicates cannot consume finished values",
            component_ids=(component_id,),
        )
    if type(condition.attrs) is not dict and not isinstance(condition.attrs, Mapping):
        _raise(
            "schedule.invalid_declaration",
            f"scheduler attrs for component {component_id} must be a mapping",
            component_ids=(component_id,),
        )
    _validate_object_free(condition.attrs, f"scheduler attrs for component {component_id}")
    attrs = dict(condition.attrs)

    if (
        type(condition.dependencies) is not tuple
        or type(condition.dependency_component_ids) is not tuple
        or type(condition.finished_value_ids) is not tuple
    ):
        _raise(
            "schedule.invalid_declaration",
            f"scheduler component {component_id} dependencies and finished-value "
            "IDs must be tuples",
            component_ids=(component_id,),
        )
    if len(condition.dependencies) != len(condition.dependency_component_ids):
        _raise(
            "schedule.invalid_declaration",
            f"scheduler component {component_id} has mismatched dependency names "
            "and IDs",
            component_ids=(component_id,),
        )
    for dependency_name, dependency_id in zip(
        condition.dependencies,
        condition.dependency_component_ids,
    ):
        if type(dependency_name) is not str:
            _raise(
                "schedule.invalid_declaration",
                f"scheduler dependency names for component {component_id} must "
                "be strings",
                component_ids=(component_id,),
            )
        _validate_component_id(dependency_id, "scheduler dependency")
        if dependency_id not in component_set_ids:
            _raise(
                "schedule.invalid_declaration",
                f"scheduler dependency {dependency_id} is not declared",
                component_ids=(component_id, dependency_id),
            )
        if dependency_name != component_names[dependency_id]:
            _raise(
                "schedule.invalid_declaration",
                f"scheduler dependency name for component {dependency_id} does "
                "not match its consideration-set declaration",
                component_ids=(component_id, dependency_id),
            )

    if condition_type == "Always":
        _require_no_dependencies(condition)
        if set(attrs) - {"implicit"} or (
            "implicit" in attrs and attrs["implicit"] is not True
        ):
            _invalid_attrs(condition)
        return

    if condition_type in {"AtPass", "AtTrialStart"}:
        _require_no_dependencies(condition)
        if set(attrs) != {"pass_index", "time_scale"}:
            _invalid_attrs(condition)
        pass_index = attrs.get("pass_index")
        if type(pass_index) is not int or pass_index < 0:
            _invalid_attrs(condition)
        if attrs.get("time_scale") != _TRIAL_TIME_SCALE:
            _invalid_attrs(condition)
        if condition_type == "AtTrialStart" and pass_index != 0:
            _invalid_attrs(condition)
        return

    if set(attrs) != {"implicit", "calls", "time_scale"}:
        _invalid_attrs(condition)
    if (
        attrs.get("implicit") is not True
        or type(attrs.get("calls")) is not int
        or attrs.get("calls") != 1
        or attrs.get("time_scale") != _TRIAL_TIME_SCALE
    ):
        _invalid_attrs(condition)
    expected_dependencies = 1 if condition_type == "EveryNCalls" else None
    if expected_dependencies is not None and len(condition.dependency_component_ids) != 1:
        _invalid_attrs(condition)
    if condition_type == "AllEveryNCalls" and len(condition.dependency_component_ids) < 2:
        _invalid_attrs(condition)
    if len(set(condition.dependency_component_ids)) != len(
        condition.dependency_component_ids
    ):
        _raise(
            "schedule.invalid_declaration",
            f"scheduler component {component_id} repeats a dependency",
            component_ids=(component_id,),
        )
    if condition.dependency_component_ids != tuple(
        sorted(condition.dependency_component_ids)
    ):
        _raise(
            "schedule.invalid_declaration",
            f"scheduler component {component_id} dependency IDs must be in "
            "deterministic numeric order",
            component_ids=(component_id,),
        )
    for dependency_id in condition.dependency_component_ids:
        if component_set_ids[dependency_id] >= condition.consideration_set_id:
            _raise(
                "schedule.unsupported_dependency_order",
                "implicit call-count dependencies must originate in an earlier "
                "consideration set",
                component_ids=(dependency_id, component_id),
            )


def _validate_termination(termination, component_ids):
    if len(termination) != 2:
        _raise(
            "schedule.unsupported_termination",
            "precomputed scheduling requires exactly the default trial and "
            "environment-sequence termination predicates",
        )
    by_scale = {}
    for item in termination:
        if type(item.time_scale) is not str or item.time_scale in by_scale:
            _raise(
                "schedule.unsupported_termination",
                "termination time scales must be unique typed strings",
            )
        if not isinstance(item.attrs, Mapping):
            _raise(
                "schedule.invalid_declaration",
                "termination attrs must be a mapping",
            )
        _validate_object_free(item.attrs, "termination attrs")
        if item.attrs:
            _raise(
                "schedule.unsupported_termination",
                "default termination predicates cannot carry extra attrs",
            )
        if type(item.condition_type) is not str or type(
            item.dependency_component_ids
        ) is not tuple:
            _raise(
                "schedule.invalid_declaration",
                "termination condition types must be strings and component IDs "
                "must be tuples",
            )
        for component_id in item.dependency_component_ids:
            _validate_component_id(component_id, "termination dependency")
        by_scale[item.time_scale] = item

    trial = by_scale.get(_TRIAL_TIME_SCALE)
    sequence = by_scale.get(_SEQUENCE_TIME_SCALE)
    if trial is None or sequence is None:
        _raise(
            "schedule.unsupported_termination",
            "termination must declare ENVIRONMENT_STATE_UPDATE and "
            "ENVIRONMENT_SEQUENCE",
        )
    if (
        trial.condition_type != "AllHaveRun"
        or trial.dependency_component_ids != component_ids
    ):
        _raise(
            "schedule.unsupported_termination",
            "trial termination must be AllHaveRun expanded to every scheduler "
            "component ID",
            component_ids=trial.dependency_component_ids,
        )
    if sequence.condition_type != "Never" or sequence.dependency_component_ids:
        _raise(
            "schedule.unsupported_termination",
            "environment-sequence termination must be the default Never predicate",
            component_ids=sequence.dependency_component_ids,
        )

    return set(component_ids)


def _validate_projection_edges(projections, component_set_ids):
    incoming_senders: dict[int, set[int]] = {}
    for projection in projections:
        sender_id = projection.sender_component_id
        receiver_id = projection.receiver_component_id
        _validate_component_id(sender_id, "projection sender")
        _validate_component_id(receiver_id, "projection receiver")
        if sender_id not in component_set_ids or receiver_id not in component_set_ids:
            _raise(
                "schedule.invalid_declaration",
                "projection endpoint is absent from the scheduler declarations",
                component_ids=(sender_id, receiver_id),
            )
        if component_set_ids[sender_id] == component_set_ids[receiver_id]:
            _raise(
                "schedule.same_set_edge",
                "a data edge within one consideration set requires frozen "
                "current/next output storage",
                component_ids=(sender_id, receiver_id),
            )
        incoming_senders.setdefault(receiver_id, set()).add(sender_id)
    return incoming_senders


def _validate_fresh_reads(
    selected,
    incoming_senders,
    executed_this_trial,
    pass_index,
    consideration_set_id,
):
    for receiver_id in selected:
        missing = tuple(sorted(
            incoming_senders.get(receiver_id, set()) - executed_this_trial
        ))
        if missing:
            _raise(
                "schedule.freshness_hazard",
                f"component {receiver_id} executes at pass {pass_index}, "
                f"consideration set {consideration_set_id} before sender(s) "
                f"{missing} have executed in this trial",
                component_ids=missing + (receiver_id,),
            )


def _condition_is_satisfied(condition, pass_index, usable_counts):
    if condition.condition_type == "Always":
        return True
    if condition.condition_type in {"AtPass", "AtTrialStart"}:
        return pass_index == condition.attrs["pass_index"]
    return all(
        usable_counts[(dependency_id, condition.component_id)] >= 1
        for dependency_id in condition.dependency_component_ids
    )


def _trace(steps, final_pass_index, component_execution_count):
    return BatchedScheduleTraceSpec(
        steps=tuple(steps),
        num_passes=final_pass_index + 1,
        component_execution_count=component_execution_count,
    )


def _require_no_dependencies(condition):
    if condition.dependencies or condition.dependency_component_ids:
        _raise(
            "schedule.invalid_declaration",
            f"{condition.condition_type} cannot carry component dependencies",
            component_ids=(condition.component_id,),
        )


def _invalid_attrs(condition):
    _raise(
        "schedule.invalid_condition_attrs",
        f"invalid {condition.condition_type} attrs for component "
        f"{condition.component_id}: {dict(condition.attrs)!r}",
        component_ids=(condition.component_id,),
    )


def _validate_component_id(value, label):
    if type(value) is not int or value < 0:
        _raise(
            "schedule.invalid_declaration",
            f"{label} must be a non-negative non-bool integer, got {value!r}",
        )


def _validate_object_free(value, label):
    if value is None or type(value) in {str, int, float, bool}:
        return
    if isinstance(value, tuple):
        for item in value:
            _validate_object_free(item, label)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) not in {str, int}:
                _raise(
                    "schedule.live_object",
                    f"{label} contains unsupported key {key!r}",
                )
            _validate_object_free(item, label)
        return
    _raise(
        "schedule.live_object",
        f"{label} contains unsupported value of type {type(value).__name__}",
    )


def _raise(code, detail, *, component_ids=()):
    raise BatchedScheduleTraceError(
        code,
        detail,
        component_ids=tuple(component_ids),
    )
