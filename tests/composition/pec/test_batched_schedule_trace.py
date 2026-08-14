"""Backend-neutral contract for precomputed batched schedule traces."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import numpy as np
import pytest

from psyneulink.core.batched.ir import (
    BatchedConsiderationSetSpec,
    BatchedFinishedValueSpec,
    BatchedProjectionSpec,
    BatchedSchedulerSpec,
    BatchedTerminationSpec,
)
from psyneulink.core.batched.schedule import (
    BatchedScheduleTraceError,
    plan_precomputed_schedule_trace,
)


pytestmark = pytest.mark.batched


def _name(component_id):
    return f"node {component_id}"


def _sets(*members):
    return tuple(
        BatchedConsiderationSetSpec(
            consideration_set_id=set_id,
            nodes=tuple(_name(component_id) for component_id in component_ids),
            component_ids=tuple(component_ids),
        )
        for set_id, component_ids in enumerate(members)
    )


def _always(component_id, set_id):
    return BatchedSchedulerSpec(
        node=_name(component_id),
        condition_type="Always",
        component_id=component_id,
        consideration_set_id=set_id,
    )


def _at_pass(component_id, set_id, pass_index):
    return BatchedSchedulerSpec(
        node=_name(component_id),
        condition_type="AtPass",
        attrs={
            "pass_index": pass_index,
            "time_scale": "ENVIRONMENT_STATE_UPDATE",
        },
        component_id=component_id,
        consideration_set_id=set_id,
    )


def _at_trial_start(component_id, set_id):
    return BatchedSchedulerSpec(
        node=_name(component_id),
        condition_type="AtTrialStart",
        attrs={
            "pass_index": 0,
            "time_scale": "ENVIRONMENT_STATE_UPDATE",
        },
        component_id=component_id,
        consideration_set_id=set_id,
    )


def _implicit_calls(component_id, set_id, *dependencies):
    condition_type = "EveryNCalls" if len(dependencies) == 1 else "AllEveryNCalls"
    return BatchedSchedulerSpec(
        node=_name(component_id),
        condition_type=condition_type,
        dependencies=tuple(_name(dependency) for dependency in dependencies),
        attrs={
            "implicit": True,
            "calls": 1,
            "time_scale": "ENVIRONMENT_STATE_UPDATE",
        },
        component_id=component_id,
        dependency_component_ids=tuple(dependencies),
        consideration_set_id=set_id,
    )


def _when_finished(component_id, set_id, dependency, finished_value_id=0):
    return BatchedSchedulerSpec(
        node=_name(component_id),
        condition_type="WhenFinished",
        dependencies=(_name(dependency),),
        attrs={"predicate": "is_finished"},
        component_id=component_id,
        dependency_component_ids=(dependency,),
        finished_value_ids=(finished_value_id,),
        consideration_set_id=set_id,
    )


def _count_finished(
    component_id,
    set_id,
    count,
    *,
    value_id=0,
    **overrides,
):
    values = {
        "name": f"{_name(component_id)}.is_finished",
        "node": _name(component_id),
        "component_id": component_id,
        "value_id": value_id,
        "width": 1,
        "dtype": "bool",
        "storage": "combinational",
        "producer_consideration_set_id": set_id,
        "predicate_kind": "execution_count_at_least",
        "attrs": {"count": count},
    }
    values.update(overrides)
    return BatchedFinishedValueSpec(**values)


def _termination(*component_ids):
    return (
        BatchedTerminationSpec(
            time_scale="ENVIRONMENT_STATE_UPDATE",
            condition_type="AllHaveRun",
            dependency_component_ids=tuple(sorted(component_ids)),
        ),
        BatchedTerminationSpec(
            time_scale="ENVIRONMENT_SEQUENCE",
            condition_type="Never",
        ),
    )


def _edge(sender, receiver):
    return BatchedProjectionSpec(
        sender=_name(sender),
        sender_port="RESULT",
        receiver=_name(receiver),
        receiver_port="InputPort-0",
        matrix=np.ones((1, 1), dtype=np.float32),
        sender_component_id=sender,
        receiver_component_id=receiver,
    )


def _plan(
    scheduler,
    consideration_sets,
    *,
    projections=(),
    finished_values=(),
    budget=64,
):
    component_ids = tuple(
        component_id
        for consideration_set in consideration_sets
        for component_id in consideration_set.component_ids
    )
    return plan_precomputed_schedule_trace(
        scheduler=scheduler,
        consideration_sets=consideration_sets,
        termination=_termination(*component_ids),
        projections=projections,
        finished_values=finished_values,
        expansion_budget=budget,
    )


def _steps(trace):
    return tuple(
        (step.pass_index, step.consideration_set_id, step.component_ids)
        for step in trace.steps
    )


def test_single_delayed_node_jumps_empty_passes():
    consideration_sets = _sets((0,))
    trace = _plan((_at_pass(0, 0, 3),), consideration_sets)

    assert _steps(trace) == ((3, 0, (0,)),)
    assert trace.num_passes == 4
    assert trace.component_execution_count == 1


def test_at_trial_start_has_the_same_trace_as_at_pass_zero():
    consideration_sets = _sets((0,))

    at_trial_start = _plan(
        (_at_trial_start(0, 0),),
        consideration_sets,
    )
    at_pass_zero = _plan(
        (_at_pass(0, 0, 0),),
        consideration_sets,
    )

    assert at_trial_start == at_pass_zero
    assert _steps(at_trial_start) == ((0, 0, (0,)),)


def test_sparse_delayed_node_does_not_expand_empty_passes():
    consideration_sets = _sets((0,))
    trace = _plan(
        (_at_pass(0, 0, 10**12),),
        consideration_sets,
        budget=1,
    )

    assert _steps(trace) == ((10**12, 0, (0,)),)
    assert trace.num_passes == 10**12 + 1
    assert trace.component_execution_count == 1


def test_same_set_members_form_one_snapshot_execution_step():
    consideration_sets = _sets((0, 1))
    trace = _plan(
        (_at_pass(0, 0, 2), _at_pass(1, 0, 2)),
        consideration_sets,
    )

    assert _steps(trace) == ((2, 0, (0, 1)),)


def test_delayed_origin_enables_implicit_child_in_later_set():
    consideration_sets = _sets((0,), (1,))
    trace = _plan(
        (
            _at_pass(0, 0, 3),
            _implicit_calls(1, 1, 0),
        ),
        consideration_sets,
        projections=(_edge(0, 1),),
    )

    assert _steps(trace) == (
        (3, 0, (0,)),
        (3, 1, (1,)),
    )


def test_always_source_repeats_until_delayed_child_executes():
    consideration_sets = _sets((0,), (1,))
    trace = _plan(
        (
            _always(0, 0),
            _at_pass(1, 1, 3),
        ),
        consideration_sets,
        projections=(_edge(0, 1),),
    )

    assert _steps(trace) == (
        (0, 0, (0,)),
        (1, 0, (0,)),
        (2, 0, (0,)),
        (3, 0, (0,)),
        (3, 1, (1,)),
    )
    assert trace.component_execution_count == 5


def test_multi_parent_implicit_child_retains_usable_counts_across_passes():
    consideration_sets = _sets((0, 1), (2,))
    trace = _plan(
        (
            _at_pass(0, 0, 1),
            _at_pass(1, 0, 3),
            _implicit_calls(2, 1, 0, 1),
        ),
        consideration_sets,
        projections=(_edge(0, 2), _edge(1, 2)),
    )

    assert _steps(trace) == (
        (1, 0, (0,)),
        (3, 0, (1,)),
        (3, 1, (2,)),
    )


def test_owner_execution_consumes_usable_dependency_count():
    # node 0 produces one credit at pass 0.  Node 1 consumes it in the later
    # set.  While node 2 keeps the trial open until pass 3, node 1 must not run
    # again without a new node-0 execution.
    consideration_sets = _sets((0, 2), (1,))
    trace = _plan(
        (
            _at_pass(0, 0, 0),
            _at_pass(2, 0, 3),
            _implicit_calls(1, 1, 0),
        ),
        consideration_sets,
        projections=(_edge(0, 1),),
    )

    assert _steps(trace) == (
        (0, 0, (0,)),
        (0, 1, (1,)),
        (3, 0, (2,)),
    )


@pytest.mark.parametrize(
    "count, expected_steps",
    [
        (1, ((0, 0, (0,)), (0, 1, (1,)))),
        (
            3,
            (
                (0, 0, (0,)),
                (1, 0, (0,)),
                (2, 0, (0,)),
                (2, 1, (1,)),
            ),
        ),
    ],
)
def test_when_finished_uses_trial_local_owner_execution_count(
    count,
    expected_steps,
):
    consideration_sets = _sets((0,), (1,))
    trace = _plan(
        (_always(0, 0), _when_finished(1, 1, 0)),
        consideration_sets,
        finished_values=(_count_finished(0, 0, count),),
    )

    assert _steps(trace) == expected_steps
    assert trace.num_passes == count
    assert trace.component_execution_count == count + 1


def test_when_finished_later_set_observes_owner_finishing_in_same_pass():
    consideration_sets = _sets((0,), (1,))
    trace = _plan(
        (_always(0, 0), _when_finished(1, 1, 0)),
        consideration_sets,
        finished_values=(_count_finished(0, 0, 1),),
    )

    assert _steps(trace) == (
        (0, 0, (0,)),
        (0, 1, (1,)),
    )


def test_when_finished_same_set_remains_fail_closed():
    consideration_sets = _sets((0, 1))

    with pytest.raises(BatchedScheduleTraceError) as error:
        _plan(
            (_always(0, 0), _when_finished(1, 0, 0)),
            consideration_sets,
            finished_values=(_count_finished(0, 0, 1),),
        )

    assert error.value.code == "schedule.unsupported_dependency_order"
    assert error.value.component_ids == (0, 1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"predicate_kind": "dynamic"},
        {"attrs": {"count": 0}},
        {"attrs": {"count": True}},
        {"attrs": {"count": 1, "extra": 0}},
        {"storage": "state"},
        {"width": 2},
        {"dtype": "float32"},
    ],
)
def test_when_finished_rejects_unsupported_finished_declaration(overrides):
    consideration_sets = _sets((0,), (1,))

    with pytest.raises(BatchedScheduleTraceError) as error:
        _plan(
            (_always(0, 0), _when_finished(1, 1, 0)),
            consideration_sets,
            finished_values=(_count_finished(0, 0, 1, **overrides),),
        )

    assert error.value.code == "schedule.unsupported_finished_predicate"
    assert error.value.component_ids == (0,)


def test_when_finished_value_must_be_owned_by_condition_dependency():
    consideration_sets = _sets((0,), (1,), (2,))

    with pytest.raises(BatchedScheduleTraceError) as error:
        _plan(
            (
                _always(0, 0),
                _always(1, 1),
                _when_finished(2, 2, 0),
            ),
            consideration_sets,
            finished_values=(_count_finished(1, 1, 1),),
        )

    assert error.value.code == "schedule.invalid_declaration"
    assert error.value.component_ids == (0, 2)


def test_when_finished_rejects_undeclared_finished_value_id():
    consideration_sets = _sets((0,), (1,))

    with pytest.raises(BatchedScheduleTraceError) as error:
        _plan(
            (_always(0, 0), _when_finished(1, 1, 0, finished_value_id=7)),
            consideration_sets,
        )

    assert error.value.code == "schedule.invalid_declaration"
    assert error.value.component_ids == (1,)


def test_when_finished_trace_expansion_budget_remains_fail_closed():
    consideration_sets = _sets((0,), (1,))
    scheduler = (_always(0, 0), _when_finished(1, 1, 0))
    finished_values = (_count_finished(0, 0, 3),)

    with pytest.raises(BatchedScheduleTraceError) as error:
        _plan(
            scheduler,
            consideration_sets,
            finished_values=finished_values,
            budget=3,
        )

    assert error.value.code == "schedule.expansion_budget_exceeded"
    assert error.value.component_ids == (1,)

    trace = _plan(
        scheduler,
        consideration_sets,
        finished_values=finished_values,
        budget=4,
    )
    assert trace.component_execution_count == 4


def test_all_have_run_stops_before_later_set_in_same_pass():
    # The always follower has already run on earlier passes.  Once the delayed
    # blocker executes in set 0, AllHaveRun is true and set 1 is not considered.
    consideration_sets = _sets((0, 2), (1,))
    trace = _plan(
        (
            _at_pass(0, 0, 0),
            _at_pass(2, 0, 2),
            _always(1, 1),
        ),
        consideration_sets,
        projections=(_edge(0, 1),),
    )

    assert _steps(trace) == (
        (0, 0, (0,)),
        (0, 1, (1,)),
        (1, 1, (1,)),
        (2, 0, (2,)),
    )
    assert trace.num_passes == 3


def test_trace_records_are_typed_and_object_free():
    consideration_sets = _sets((0,), (1,))
    trace = _plan(
        (_at_pass(0, 0, 2), _implicit_calls(1, 1, 0)),
        consideration_sets,
        projections=(_edge(0, 1),),
    )

    def assert_object_free(value):
        if is_dataclass(value) and not isinstance(value, type):
            assert type(value).__module__.startswith("psyneulink.core.batched")
            for item in fields(value):
                assert_object_free(getattr(value, item.name))
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                assert_object_free(key)
                assert_object_free(item)
            return
        if isinstance(value, tuple):
            for item in value:
                assert_object_free(item)
            return
        assert value is None or type(value) in {str, int, float, bool}

    assert_object_free(trace)


def test_receiver_before_sender_is_a_freshness_hazard():
    consideration_sets = _sets((0,), (1,))

    with pytest.raises(BatchedScheduleTraceError) as error:
        _plan(
            (_at_pass(0, 0, 3), _always(1, 1)),
            consideration_sets,
            projections=(_edge(0, 1),),
        )

    assert error.value.code == "schedule.freshness_hazard"
    assert error.value.component_ids == (0, 1)


def test_same_consideration_set_data_edge_is_rejected():
    consideration_sets = _sets((0, 1))

    with pytest.raises(BatchedScheduleTraceError) as error:
        _plan(
            (_always(0, 0), _always(1, 0)),
            consideration_sets,
            projections=(_edge(0, 1),),
        )

    assert error.value.code == "schedule.same_set_edge"
    assert error.value.component_ids == (0, 1)


def test_trace_expansion_budget_is_explicit_and_fail_closed():
    consideration_sets = _sets((0, 1))

    with pytest.raises(BatchedScheduleTraceError) as error:
        _plan(
            (_always(0, 0), _at_pass(1, 0, 5)),
            consideration_sets,
            budget=3,
        )

    assert error.value.code == "schedule.expansion_budget_exceeded"
    assert error.value.component_ids == (0,)


def test_trace_expansion_budget_exact_boundary_succeeds():
    consideration_sets = _sets((0, 1))
    trace = _plan(
        (_always(0, 0), _at_pass(1, 0, 3)),
        consideration_sets,
        budget=5,
    )

    assert trace.component_execution_count == 5
    assert _steps(trace)[-1] == (3, 0, (0, 1))


def test_implicit_dependency_must_come_from_earlier_set():
    consideration_sets = _sets((1,), (0,))

    with pytest.raises(BatchedScheduleTraceError) as error:
        _plan(
            (_always(0, 1), _implicit_calls(1, 0, 0)),
            consideration_sets,
        )

    assert error.value.code == "schedule.unsupported_dependency_order"
    assert error.value.component_ids == (0, 1)


def test_multi_parent_dependency_ids_require_deterministic_order():
    consideration_sets = _sets((0, 1), (2,))
    unordered = BatchedSchedulerSpec(
        node=_name(2),
        condition_type="AllEveryNCalls",
        dependencies=(_name(1), _name(0)),
        attrs={
            "implicit": True,
            "calls": 1,
            "time_scale": "ENVIRONMENT_STATE_UPDATE",
        },
        component_id=2,
        dependency_component_ids=(1, 0),
        consideration_set_id=1,
    )

    with pytest.raises(BatchedScheduleTraceError) as error:
        _plan(
            (_always(0, 0), _always(1, 0), unordered),
            consideration_sets,
        )

    assert error.value.code == "schedule.invalid_declaration"


def test_nondefault_termination_is_rejected():
    consideration_sets = _sets((0,))
    termination = (
        BatchedTerminationSpec(
            time_scale="ENVIRONMENT_STATE_UPDATE",
            condition_type="AtPass",
            attrs={"pass_index": 3},
        ),
        BatchedTerminationSpec(
            time_scale="ENVIRONMENT_SEQUENCE",
            condition_type="Never",
        ),
    )

    with pytest.raises(BatchedScheduleTraceError) as error:
        plan_precomputed_schedule_trace(
            scheduler=(_always(0, 0),),
            consideration_sets=consideration_sets,
            termination=termination,
            projections=(),
            expansion_budget=8,
        )

    assert error.value.code == "schedule.unsupported_termination"


def test_nonterminating_execution_predicate_is_rejected():
    consideration_sets = _sets((0,))
    never = BatchedSchedulerSpec(
        node=_name(0),
        condition_type="Never",
        component_id=0,
        consideration_set_id=0,
    )

    with pytest.raises(BatchedScheduleTraceError) as error:
        _plan((never,), consideration_sets)

    assert error.value.code == "schedule.unsupported_condition"
    assert error.value.component_ids == (0,)
