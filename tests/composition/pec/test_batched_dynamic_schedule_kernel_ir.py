"""Typed, non-executable KernelIR scaffold for lane-local scheduling."""

from dataclasses import FrozenInstanceError, fields, replace
from types import SimpleNamespace

import pytest

from psyneulink.core.batched.backend.triton.emit.ops import OpEmitMixin
from psyneulink.core.batched.backend.triton.source_builder import SourceBuilder
from psyneulink.core.batched.kernel_ir import (
    KernelComponentExecutionBudget,
    KernelConsiderationSetProgram,
    KernelDynamicScheduleProgram,
    KernelLoopCarry,
    KernelOp,
    KernelPublication,
    KernelScheduledComponent,
    KernelSchedulePredicate,
    KernelSchedulerStateSlot,
    KernelValue,
)


pytestmark = pytest.mark.batched


class _SchedulerEmitHarness(OpEmitMixin):
    def __init__(self):
        self.builder = SourceBuilder()


def _value(name, dtype="float32"):
    return KernelValue(name, 1, dtype)


def _member(component_id, predicate):
    output = _value(f"component:{component_id}:candidate")
    return KernelScheduledComponent(
        component_id=component_id,
        predicate=predicate,
        body=(
            KernelOp(
                kind="CallFunction",
                target=f"component {component_id}",
                outputs=(output,),
                attrs={"component_id": component_id},
            ),
        ),
        publications=(
            KernelPublication(output, "output", component_id, 10 + component_id),
        ),
    )


def _slot(
    kind,
    *,
    owner=None,
    producer=None,
    consumer=None,
    finished=None,
    rng_stream=None,
    dtype="int32",
):
    suffix = ":".join(
        str(value)
        for value in (owner, producer, consumer, finished, rng_stream)
        if value is not None
    )
    return KernelSchedulerStateSlot(
        kind=kind,
        value=_value(f"schedule:{kind}:{suffix}", dtype),
        owner_component_id=owner,
        producer_component_id=producer,
        consumer_component_id=consumer,
        finished_value_id=finished,
        rng_stream_id=rng_stream,
    )


def _carry(member):
    publication = member.publications[0]
    return KernelLoopCarry(
        publication.kind,
        publication.owner_component_id,
        publication.value_id,
        _value(f"component:{member.component_id}:snapshot"),
    )


def _program():
    members = (
        _member(0, KernelSchedulePredicate("Always")),
        _member(
            1,
            KernelSchedulePredicate(
                "EveryNCalls",
                dependency_component_ids=(0,),
                call_count=1,
            ),
        ),
        _member(
            2,
            KernelSchedulePredicate(
                "WhenFinished",
                dependency_component_ids=(1,),
                finished_value_ids=(7,),
            ),
        ),
    )
    slots = (
        _slot("pass_index"),
        *(
            slot
            for component_id in range(3)
            for slot in (
                _slot("execution_count", owner=component_id),
                _slot("has_run", owner=component_id, dtype="bool"),
            )
        ),
        _slot("usable_call", producer=0, consumer=1),
        _slot("finished", owner=1, finished=7, dtype="bool"),
        _slot("rng_clock", owner=0, rng_stream=0),
    )
    return KernelDynamicScheduleProgram(
        consideration_sets=tuple(
            KernelConsiderationSetProgram(index, (member,))
            for index, member in enumerate(members)
        ),
        scheduler_state_slots=slots,
        loop_carries=tuple(_carry(member) for member in members),
        execution_budgets=tuple(
            KernelComponentExecutionBudget(
                component_id,
                100,
                **(
                    {
                        "finished_value_id": 7,
                        "unfinished_maximum": 100,
                        "post_finish": "continue",
                    }
                    if component_id == 1
                    else {}
                ),
            )
            for component_id in range(3)
        ),
        trial_termination=KernelSchedulePredicate(
            "AllHaveRun",
            dependency_component_ids=(0, 1, 2),
        ),
        schedule_fuel=100,
    )


def test_triton_has_run_slots_pack_across_multiple_words():
    slots = tuple(
        _slot("has_run", owner=component_id, dtype="bool")
        for component_id in range(32)
    )
    emitter = _SchedulerEmitHarness()

    layout = emitter._emit_dynamic_has_run_initializers(
        SimpleNamespace(scheduler_state_slots=slots)
    )
    source = emitter.builder.render()

    assert layout[0] == ("dynamic_has_run_word_0", 1)
    assert layout[30] == ("dynamic_has_run_word_0", 1 << 30)
    assert layout[31] == ("dynamic_has_run_word_1", 1)
    assert source.count("tl.zeros((BLOCK,), dtype=tl.int32)") == 2


def test_triton_elides_only_proven_redundant_execution_counts():
    program = _program()
    emitter = _SchedulerEmitHarness()

    # Components 0 and 2 can run at most once per outer schedule round, and
    # their budget is exactly the outer fuel.  Component 1 has a distinct
    # post-finish budget and must retain its explicit count.
    assert emitter._dynamic_fuel_bounded_component_ids(program) == frozenset(
        {0, 2}
    )

    single_execution_program = replace(
        program,
        execution_budgets=tuple(
            replace(budget, maximum=1)
            if budget.component_id == 0
            else budget
            for budget in program.execution_budgets
        ),
    )
    assert emitter._dynamic_single_execution_component_ids(
        single_execution_program
    ) == frozenset({0})

    # A body that observes its own execution ordinal makes the count semantic,
    # even when its maximum is otherwise redundant with schedule fuel.
    count_value = next(
        slot.value
        for slot in program.scheduler_state_slots
        if slot.kind == "execution_count" and slot.owner_component_id == 0
    )
    member = program.consideration_sets[0].members[0]
    count_reading_member = replace(
        member,
        body=(
            replace(
                member.body[0],
                kind="AffineSchedulerValue",
                inputs=(count_value,),
                attrs={
                    "folded_control_id": 0,
                    "base_parameter_id": 0,
                    "delta_parameter_id": 1,
                },
            ),
        ),
    )
    count_reading_program = replace(
        program,
        consideration_sets=(
            replace(
                program.consideration_sets[0],
                members=(count_reading_member,),
            ),
            *program.consideration_sets[1:],
        ),
    )
    assert emitter._dynamic_fuel_bounded_component_ids(
        count_reading_program
    ) == frozenset({2})


def test_dynamic_schedule_records_are_frozen_typed_and_attr_free():
    program = _program()
    records = (
        program,
        *program.consideration_sets,
        *(item.members[0] for item in program.consideration_sets),
        *(
            publication
            for item in program.consideration_sets
            for publication in item.members[0].publications
        ),
        *program.scheduler_state_slots,
        *program.loop_carries,
        *program.execution_budgets,
        program.trial_termination,
    )

    assert program.consideration_sets[0].inputs_frozen is True
    assert all(
        "attrs" not in {field.name for field in fields(record)}
        for record in records
    )
    assert program.scheduler_state_slots[-1].kind == "rng_clock"
    assert program.scheduler_state_slots[-1].rng_stream_id == 0
    parameter_initialized = KernelLoopCarry(
        "trial_state",
        0,
        0,
        _value("trial state"),
        initial_parameter_id=3,
    )
    assert parameter_initialized.initial_value is None
    with pytest.raises(FrozenInstanceError):
        program.loop_carries = ()


@pytest.mark.parametrize(
    "predicate",
    [
        KernelSchedulePredicate("Always"),
        KernelSchedulePredicate("AtPass", pass_index=3),
        KernelSchedulePredicate("AtTrialStart", pass_index=0),
        KernelSchedulePredicate(
            "AllEveryNCalls", dependency_component_ids=(0, 1), call_count=1
        ),
        KernelSchedulePredicate(
            "WhenFinished", dependency_component_ids=(0,), finished_value_ids=(4,)
        ),
        KernelSchedulePredicate("AllHaveRun", dependency_component_ids=(0, 1)),
    ],
)
def test_predicate_forms_have_only_typed_operands(predicate):
    assert not hasattr(predicate, "attrs")


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "Any"},
        {"kind": []},
        {"kind": "Always", "dependency_component_ids": [0]},
        {"kind": "Always", "dependency_component_ids": None},
        {"kind": "Always", "dependency_component_ids": (True,)},
        {"kind": "Always", "dependency_component_ids": (1, 1)},
        {"kind": "AtPass", "pass_index": True},
        {"kind": "AtTrialStart", "pass_index": 1},
        {"kind": "EveryNCalls", "dependency_component_ids": (0,), "call_count": 0},
        {"kind": "EveryNCalls", "dependency_component_ids": (0,), "call_count": 2},
        {"kind": "WhenFinished", "dependency_component_ids": (0,)},
    ],
)
def test_predicate_constructor_fails_closed(changes):
    with pytest.raises(ValueError):
        KernelSchedulePredicate(**changes)


def test_member_requires_exact_ops_and_body_defined_publications():
    member = _member(0, KernelSchedulePredicate("Always"))

    with pytest.raises(ValueError, match="published values"):
        replace(
            member,
            publications=(
                replace(member.publications[0], source=_value("forged")),
            ),
        )

    with pytest.raises(ValueError, match="typed identity"):
        KernelPublication(_value("finished candidate"), "finished", 0, 0)
    finished_candidate = _value("finished candidate", "bool")
    finished_member = replace(
        member,
        body=(
            replace(
                member.body[0],
                outputs=(*member.body[0].outputs, finished_candidate),
            ),
        ),
        publications=(
            *member.publications,
            KernelPublication(finished_candidate, "finished", 0, 0),
        ),
    )
    assert finished_member.publications[-1].kind == "finished"
    with pytest.raises(ValueError, match="fields"):
        replace(member, body=list(member.body))
    with pytest.raises(ValueError, match="fields"):
        replace(
            member,
            body=(
                KernelOp(
                    "ForPasses",
                    "passes",
                    attrs={"declaration_only": True, "body": ()},
                ),
            ),
        )
    with pytest.raises(ValueError, match="fields"):
        replace(member, effects=member.body)
    with pytest.raises(ValueError, match="fields"):
        replace(
            member,
            effects=(
                KernelOp(
                    "StoreOutput",
                    "result",
                    inputs=(member.publications[0].source,),
                ),
            ),
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: _slot("execution_count"),
        lambda: _slot([], owner=0),
        lambda: _slot("has_run", owner=0, dtype="float32"),
        lambda: KernelSchedulerStateSlot(
            "execution_count",
            KernelValue("wide count", 2, "int32"),
            owner_component_id=0,
        ),
        lambda: _slot("pass_index", owner=0),
        lambda: _slot("usable_call", producer=0, consumer=0),
        lambda: _slot("finished", owner=0),
        lambda: KernelSchedulerStateSlot(
            "finished",
            _value("schedule:finished:0:0", "bool"),
            owner_component_id=0,
            finished_value_id=0,
            initialization="count_zero_vs_effective_parameter",
        ),
        lambda: KernelSchedulerStateSlot(
            "execution_count",
            _value("schedule:execution-count:0", "int32"),
            owner_component_id=0,
            initialization="count_zero_vs_effective_parameter",
            initial_effective_parameter_id=0,
        ),
        lambda: _slot("rng_clock", owner=0),
        lambda: _slot("rng_clock", owner=0, rng_stream=0, dtype="bool"),
        lambda: KernelLoopCarry("output", True, 0, _value("carry")),
        lambda: KernelLoopCarry("topology", 0, 0, _value("carry")),
        lambda: KernelLoopCarry("output", 0, 0, object()),
        lambda: KernelLoopCarry("trial_state", 0, 0, _value("carry")),
        lambda: KernelLoopCarry(
            "trial_state", 0, 0, _value("carry"), initial_value=(float("nan"),)
        ),
        lambda: KernelLoopCarry(
            "trial_state",
            0,
            0,
            _value("carry"),
            initial_value=(0.0,),
            initial_parameter_id=1,
        ),
        lambda: KernelLoopCarry(
            "trial_state", 0, 0, _value("carry"), initial_parameter_id=True
        ),
        lambda: KernelLoopCarry(
            "output", 0, 0, _value("carry"), initial_value=(0.0,)
        ),
        lambda: KernelComponentExecutionBudget(0, True),
        lambda: KernelComponentExecutionBudget(
            0, 2, finished_value_id=1, post_finish="continue"
        ),
        lambda: KernelComponentExecutionBudget(
            0,
            2,
            finished_value_id=1,
            unfinished_maximum=3,
            post_finish="stop",
        ),
        lambda: KernelComponentExecutionBudget(0, 2, post_finish="forged"),
    ],
)
def test_state_carry_and_budget_identities_fail_closed(factory):
    with pytest.raises(ValueError):
        factory()


def test_program_authenticates_coverage_references_and_ownership():
    program = _program()

    missing_slot = tuple(
        slot for slot in program.scheduler_state_slots if slot.kind != "usable_call"
    )
    with pytest.raises(ValueError, match="identities"):
        replace(program, scheduler_state_slots=missing_slot)
    aliased_slots = (
        program.scheduler_state_slots[0],
        replace(
            program.scheduler_state_slots[1],
            value=program.scheduler_state_slots[0].value,
        ),
        *program.scheduler_state_slots[2:],
    )
    with pytest.raises(ValueError, match="identities"):
        replace(program, scheduler_state_slots=aliased_slots)
    with pytest.raises(ValueError, match="ordered sets"):
        replace(
            program,
            trial_termination=KernelSchedulePredicate(
                "AllHaveRun", dependency_component_ids=(0, 1)
            ),
        )
    with pytest.raises(ValueError, match="owned"):
        replace(program, execution_budgets=(KernelComponentExecutionBudget(9, 1),))
    with pytest.raises(ValueError, match="fuel"):
        replace(program, schedule_fuel=99)
    with pytest.raises(ValueError, match="owned"):
        replace(
            program,
            loop_carries=(KernelLoopCarry("state", 9, 0, _value("carry")),),
        )
    first_carry = program.loop_carries[0]
    with pytest.raises(ValueError, match="identities"):
        replace(
            program,
            loop_carries=(
                first_carry,
                KernelLoopCarry("state", 1, 11, first_carry.value),
            ),
        )
    duplicate = replace(program.consideration_sets[1], consideration_set_id=0)
    with pytest.raises(ValueError, match="ordered sets"):
        replace(program, consideration_sets=(program.consideration_sets[0], duplicate))


def test_member_inputs_require_snapshot_or_earlier_local_definition():
    program = _program()
    first_set = program.consideration_sets[0]
    member = first_set.members[0]
    forged = replace(
        member,
        body=(replace(member.body[0], inputs=(_value("undefined"),)),),
    )

    with pytest.raises(ValueError, match="snapshot carry or an earlier"):
        replace(
            program,
            consideration_sets=(
                replace(first_set, members=(forged,)),
                *program.consideration_sets[1:],
            ),
        )


def test_same_set_member_cannot_read_another_members_candidate():
    program = _program()
    first = program.consideration_sets[0].members[0]
    second = program.consideration_sets[1].members[0]
    forged_second = replace(
        second,
        body=(
            replace(second.body[0], inputs=(first.publications[0].source,)),
        ),
    )
    combined = KernelConsiderationSetProgram(0, (first, forged_second))
    final = replace(program.consideration_sets[2], consideration_set_id=1)

    with pytest.raises(ValueError, match="snapshot carry or an earlier"):
        replace(program, consideration_sets=(combined, final))


def test_publications_require_owned_type_matched_nonconflicting_carries():
    program = _program()
    first_set = program.consideration_sets[0]
    member = first_set.members[0]
    publication = member.publications[0]

    cross_owner = replace(
        member,
        publications=(
            replace(publication, owner_component_id=1, value_id=11),
        ),
    )
    with pytest.raises(ValueError, match="owned carry destinations"):
        replace(
            program,
            consideration_sets=(
                replace(first_set, members=(cross_owner,)),
                *program.consideration_sets[1:],
            ),
        )

    mismatched_carry = replace(program.loop_carries[0], value=_value("wrong", "bool"))
    with pytest.raises(ValueError, match="type-matched"):
        replace(
            program,
            loop_carries=(mismatched_carry, *program.loop_carries[1:]),
        )

    second_source = _value("component:0:second-candidate")
    conflicting = replace(
        member,
        body=(
            *member.body,
            KernelOp("CallFunction", "component 0", outputs=(second_source,)),
        ),
        publications=(
            publication,
            replace(publication, source=second_source),
        ),
    )
    with pytest.raises(ValueError, match="carry destinations"):
        replace(
            program,
            consideration_sets=(
                replace(first_set, members=(conflicting,)),
                *program.consideration_sets[1:],
            ),
        )


def test_rng_clocks_have_unique_explicit_stream_identities():
    program = _program()
    second_stream = _slot("rng_clock", owner=0, rng_stream=1)
    extended = replace(
        program,
        scheduler_state_slots=(*program.scheduler_state_slots, second_stream),
    )
    assert extended.scheduler_state_slots[-1].rng_stream_id == 1

    with pytest.raises(ValueError, match="identities"):
        replace(
            program,
            scheduler_state_slots=(
                *program.scheduler_state_slots,
                _slot("rng_clock", owner=1, rng_stream=0),
            ),
        )


def test_schema_equality_does_not_recurse_into_mutable_op_attrs():
    member = _member(0, KernelSchedulePredicate("Always"))
    clone = replace(
        member,
        body=(replace(member.body[0], attrs=dict(member.body[0].attrs)),),
    )

    assert member != clone
    assert type(hash(member)) is int
    member.body[0].attrs["component_id"] = 999
    assert member != clone
