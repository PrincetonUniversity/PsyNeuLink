"""Backend-neutral capability contract for precomputed schedule traces."""

import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompileError,
    BatchedCompositionCompiler,
    BatchedDiagnosticCode,
)
from psyneulink.core.batched import graph as batched_graph
from psyneulink.core.batched import kernel_ir as batched_kernel_ir
from psyneulink.core.batched import registry as batched_registry


pytestmark = [pytest.mark.batched, pytest.mark.composition]


@pytest.fixture(autouse=True)
def _backend_capability_is_available(monkeypatch):
    """Keep semantic capability checks independent of optional Triton installs."""

    monkeypatch.setattr(
        batched_registry,
        "_backend_availability",
        lambda _backend: (True, ()),
    )


def _scheduled_graph(*, freshness_hazard=False):
    source = pnl.TransferMechanism(input_shapes=1, name="delayed source")
    receiver = pnl.TransferMechanism(input_shapes=1, name="scheduled receiver")
    composition = pnl.Composition(pathways=[[source, receiver]])
    if freshness_hazard:
        composition.scheduler.add_condition(source, pnl.AtPass(3))
        composition.scheduler.add_condition(receiver, pnl.AtPass(0))
    else:
        composition.scheduler.add_condition(source, pnl.Always())
        composition.scheduler.add_condition(receiver, pnl.AtPass(3))
    return composition, receiver


def _schedule_diagnostics(report):
    return tuple(
        diagnostic
        for diagnostic in report.rejected_conditions
        if diagnostic.code == BatchedDiagnosticCode.MODEL_SCHEDULE_NOT_EXECUTABLE
    )


def test_exact_stateless_precomputed_trace_is_executable():
    composition, receiver = _scheduled_graph()

    report = BatchedCompositionCompiler.diagnose(
        composition,
        backend="triton_cpu",
        outputs=(receiver.output_port,),
    )
    plan = BatchedCompositionCompiler.compile(
        composition,
        backend="triton_cpu",
        outputs=(receiver.output_port,),
    )

    assert report.model_supported
    assert report.codegen_ready
    assert report.can_execute
    assert report.metadata["schedule_kind"] == "precomputed_trace"
    assert not report.rejected_conditions
    assert plan.kernel_ir.executable
    assert plan.kernel_ir.schedule_trace is not None
    assert plan.kernel_ir.schedule_trace.component_execution_count == 5


def test_freshness_failure_is_one_structured_schedule_diagnostic():
    composition, receiver = _scheduled_graph(freshness_hazard=True)

    report = BatchedCompositionCompiler.diagnose(
        composition,
        backend="triton_cpu",
        outputs=(receiver.output_port,),
    )

    diagnostics = _schedule_diagnostics(report)
    assert len(diagnostics) == 1
    assert diagnostics[0].component_id.startswith("composition:")
    assert "schedule.freshness_hazard" in diagnostics[0].detail
    assert not report.model_supported
    with pytest.raises(BatchedCompileError) as error:
        BatchedCompositionCompiler.compile(
            composition,
            backend="triton_cpu",
            outputs=(receiver.output_port,),
        )
    assert error.value.capability_report is not None
    assert len(_schedule_diagnostics(error.value.capability_report)) == 1


def test_component_expansion_cap_is_a_model_diagnostic(monkeypatch):
    monkeypatch.setattr(
        batched_graph,
        "PRECOMPUTED_TRACE_COMPONENT_BUDGET",
        3,
    )
    composition, receiver = _scheduled_graph()

    report = BatchedCompositionCompiler.diagnose(
        composition,
        backend="triton_cpu",
        outputs=(receiver.output_port,),
    )

    diagnostics = _schedule_diagnostics(report)
    assert len(diagnostics) == 1
    assert "schedule.expansion_budget_exceeded" in diagnostics[0].detail
    assert not report.can_execute
    with pytest.raises(BatchedCompileError):
        BatchedCompositionCompiler.compile(
            composition,
            backend="triton_cpu",
            outputs=(receiver.output_port,),
        )


def test_weighted_kernel_expansion_cap_is_a_codegen_diagnostic(monkeypatch):
    monkeypatch.setattr(
        batched_kernel_ir,
        "_DEFAULT_TRACE_WEIGHTED_OP_BUDGET",
        1,
    )
    composition, receiver = _scheduled_graph()

    report = BatchedCompositionCompiler.diagnose(
        composition,
        backend="triton_cpu",
        outputs=(receiver.output_port,),
    )

    assert report.model_supported
    assert report.codegen_ready is False
    assert not report.can_execute
    assert len(report.codegen_diagnostics) == 1
    diagnostic = report.codegen_diagnostics[0]
    assert (
        diagnostic.code
        == BatchedDiagnosticCode.CODEGEN_KERNEL_IR_LOWERING_FAILED
    )
    assert "weighted op expansion" in diagnostic.detail
    with pytest.raises(BatchedCompileError) as error:
        BatchedCompositionCompiler.compile(
            composition,
            backend="triton_cpu",
            outputs=(receiver.output_port,),
        )
    assert error.value.capability_report is not None
    assert (
        error.value.capability_report.codegen_diagnostics[0].code
        == BatchedDiagnosticCode.CODEGEN_KERNEL_IR_LOWERING_FAILED
    )


def test_stateful_precomputed_schedule_remains_fail_closed():
    mechanism = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(),
        leak=0.0,
        competition=0.0,
        self_excitation=0.0,
        noise=0.0,
        time_step_size=0.4,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=1,
        reset_stateful_function_when=pnl.Never(),
        name="stateful delayed mechanism",
    )
    composition = pnl.Composition(pathways=mechanism)
    composition.scheduler.add_condition(mechanism, pnl.AtPass(3))

    report = BatchedCompositionCompiler.diagnose(
        composition,
        backend="triton_cpu",
    )

    diagnostics = _schedule_diagnostics(report)
    assert len(diagnostics) == 1
    assert "retained state" in diagnostics[0].detail
    assert not report.can_execute


def test_untyped_explicit_call_condition_keeps_scheduler_diagnostic():
    source = pnl.TransferMechanism(input_shapes=1, name="call source")
    receiver = pnl.TransferMechanism(input_shapes=1, name="explicit call receiver")
    composition = pnl.Composition(pathways=[[source, receiver]])
    composition.scheduler.add_condition(receiver, pnl.EveryNCalls(source, 2))

    report = BatchedCompositionCompiler.diagnose(
        composition,
        backend="triton_cpu",
    )

    assert not report.model_supported
    assert any(
        diagnostic.code == BatchedDiagnosticCode.MODEL_SCHEDULE_NOT_EXECUTABLE
        and "EveryNCalls" in diagnostic.detail
        for diagnostic in report.rejected_conditions
    )
