"""Fail-closed snapshotting of mutable PsyNeuLink conditions."""

from dataclasses import dataclass

import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompileError,
    BatchedCompositionCompiler,
    BatchedDiagnosticCode,
)
from psyneulink.core.batched.condition_validation import is_canonical_condition
from psyneulink.core.scheduling.condition import All, EveryNCalls


pytestmark = [pytest.mark.batched, pytest.mark.composition]


@dataclass(frozen=True)
class _ConditionCase:
    name: str
    composition: pnl.Composition
    output: object
    component: object
    code: str
    reason: str


def _scheduler_case(condition_kind):
    source = pnl.TransferMechanism(input_shapes=1, name="condition source")
    target = pnl.TransferMechanism(input_shapes=1, name="condition target")
    composition = pnl.Composition(pathways=[source, target])
    if condition_kind == "Always":
        component = source
        condition = pnl.Always()
    elif condition_kind == "AtPass":
        component = source
        condition = pnl.AtPass(0)
    elif condition_kind == "AtTrialStart":
        component = source
        condition = pnl.AtTrialStart()
    else:
        component = target
        condition = pnl.WhenFinished(source)
    condition.func = lambda *args, **kwargs: False
    composition.scheduler.add_condition(component, condition)
    return _ConditionCase(
        name=f"mutated {condition_kind}",
        composition=composition,
        output=target.output_port,
        component=component,
        code=BatchedDiagnosticCode.MODEL_SCHEDULER_CONDITION_UNSUPPORTED,
        reason="unsupported scheduler condition for static batched graph",
    )


def _ddm_reset_case(mutation):
    reset = pnl.AtTrialStart()
    if mutation == "args":
        reset.args = (1,)
    else:
        reset.func = lambda *args, **kwargs: False
    mechanism = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            rate=1.0,
            noise=0.0,
            threshold=1.0,
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        reset_stateful_function_when=reset,
        name=f"mutated AtTrialStart {mutation}",
    )
    return _ConditionCase(
        name=f"mutated AtTrialStart {mutation}",
        composition=pnl.Composition(pathways=mechanism),
        output=mechanism.output_ports[pnl.DECISION_OUTCOME],
        component=mechanism,
        code=BatchedDiagnosticCode.MODEL_UNSUPPORTED,
        reason="unsupported DDM reset policy for batched v2",
    )


def _lca_never_case():
    reset = pnl.Never()
    reset.func = lambda *args, **kwargs: True
    mechanism = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(),
        noise=0.0,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=2,
        reset_stateful_function_when=reset,
        name="mutated Never reset",
    )
    return _ConditionCase(
        name="mutated Never",
        composition=pnl.Composition(pathways=mechanism),
        output=mechanism.output_port,
        component=mechanism,
        code=BatchedDiagnosticCode.MODEL_UNSUPPORTED,
        reason="unsupported LCA reset policy for batched v2",
    )


def test_unmodified_builtin_conditions_are_canonical():
    dependency = pnl.TransferMechanism(input_shapes=1)
    every = EveryNCalls(dependency, 1)
    conditions = (
        pnl.Always(),
        pnl.AtPass(0),
        pnl.AtTrialStart(),
        pnl.WhenFinished(dependency),
        pnl.Never(),
        every,
        All(every),
    )
    assert all(is_canonical_condition(condition) for condition in conditions)


@pytest.mark.parametrize(
    "case_factory",
    (
        pytest.param(lambda: _scheduler_case("Always"), id="Always"),
        pytest.param(lambda: _scheduler_case("AtPass"), id="AtPass"),
        pytest.param(
            lambda: _scheduler_case("AtTrialStart"),
            id="AtTrialStart",
        ),
        pytest.param(
            lambda: _scheduler_case("WhenFinished"),
            id="WhenFinished",
        ),
        pytest.param(lambda: _ddm_reset_case("args"), id="reset-args"),
        pytest.param(lambda: _ddm_reset_case("func"), id="reset-func"),
        pytest.param(_lca_never_case, id="Never"),
    ),
)
def test_mutated_builtin_condition_rejects_before_snapshot(case_factory):
    case = case_factory()
    report = BatchedCompositionCompiler.diagnose(
        case.composition,
        backend="triton_cpu",
        outputs=(case.output,),
    )

    assert not report.model_supported
    matches = [
        diagnostic
        for diagnostic in report.model_diagnostics
        if diagnostic.code == case.code
        and diagnostic.component == case.component.name
        and diagnostic.reason == case.reason
    ]
    assert len(matches) == 1
    with pytest.raises(BatchedCompileError) as error:
        BatchedCompositionCompiler.compile(
            case.composition,
            backend="triton_cpu",
            outputs=(case.output,),
        )
    assert error.value.capability_report == report
