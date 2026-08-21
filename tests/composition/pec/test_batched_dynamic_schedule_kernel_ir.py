"""Typed, non-executable KernelIR scaffold for lane-local scheduling."""

from dataclasses import FrozenInstanceError, fields, replace

import pytest

from psyneulink.core.batched.kernel_ir import (
    KernelComponentExecutionBudget,
    KernelConsiderationSetProgram,
    KernelDynamicScheduleProgram,
    KernelLoopCarry,
    KernelOp,
    KernelScheduledComponent,
    KernelSchedulePredicate,
    KernelSchedulerStateSlot,
    KernelValue,
)


pytestmark = pytest.mark.batched


def _value(name, dtype="float32"):
    return KernelValue(name, 1, dtype)


def _member(component_id, predicate):
    output = _value(f"component:{component_id}:output")
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
        published_values=(output,),
    )


def _slot(kind, *, owner=None, dependency=None, finished=None, dtype="int32"):
    suffix = ":".join(
        str(value) for value in (owner, dependency, finished) if value is not None
    )
    return KernelSchedulerStateSlot(
        kind=kind,
        value=_value(f"schedule:{kind}:{suffix}", dtype),
        owner_component_id=owner,
        dependency_component_id=dependency,
        finished_value_id=finished,
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
        _slot("usable_call", owner=1, dependency=0),
        _slot("finished", owner=1, finished=7, dtype="bool"),
        _slot("rng_clock", owner=0),
    )
    return KernelDynamicScheduleProgram(
        consideration_sets=tuple(
            KernelConsiderationSetProgram(index, (member,))
            for index, member in enumerate(members)
        ),
        scheduler_state_slots=slots,
        loop_carries=(
            KernelLoopCarry("output", 0, 10, members[0].published_values[0]),
        ),
        execution_budgets=(KernelComponentExecutionBudget(0, 100),),
        trial_termination=KernelSchedulePredicate(
            "AllHaveRun",
            dependency_component_ids=(0, 1, 2),
        ),
    )


def test_dynamic_schedule_records_are_frozen_typed_and_attr_free():
    program = _program()
    records = (
        program,
        *program.consideration_sets,
        *(item.members[0] for item in program.consideration_sets),
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
    with pytest.raises(FrozenInstanceError):
        program.loop_carries = ()


@pytest.mark.parametrize(
    "predicate",
    [
        KernelSchedulePredicate("Always"),
        KernelSchedulePredicate("AtPass", pass_index=3),
        KernelSchedulePredicate("AtTrialStart", pass_index=0),
        KernelSchedulePredicate(
            "AllEveryNCalls", dependency_component_ids=(0, 1), call_count=2
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
        {"kind": "WhenFinished", "dependency_component_ids": (0,)},
    ],
)
def test_predicate_constructor_fails_closed(changes):
    with pytest.raises(ValueError):
        KernelSchedulePredicate(**changes)


def test_member_requires_exact_ops_and_body_defined_publications():
    member = _member(0, KernelSchedulePredicate("Always"))

    with pytest.raises(ValueError, match="published values"):
        replace(member, published_values=(_value("forged"),))
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
                KernelOp("StoreOutput", "result", inputs=member.published_values),
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
        lambda: _slot("usable_call", owner=0, dependency=0),
        lambda: _slot("finished", owner=0),
        lambda: KernelLoopCarry("output", True, 0, _value("carry")),
        lambda: KernelLoopCarry("topology", 0, 0, _value("carry")),
        lambda: KernelLoopCarry("output", 0, 0, object()),
        lambda: KernelComponentExecutionBudget(0, True),
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
