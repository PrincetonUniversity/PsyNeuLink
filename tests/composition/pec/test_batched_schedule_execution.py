"""Semantic contract for executable stateless batched schedules.

The runtime cases are ordinary Python-oracle cases rather than source goldens:
each backend must execute the typed pass trace, predicates, and consideration
sets with the same trial-local semantics as PsyNeuLink.  Rejection cases pin
nearby semantics that are outside this first executable subset.
"""

from collections.abc import Callable
from dataclasses import dataclass
import itertools

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompileError,
    BatchedCompositionCompiler,
    BatchedDiagnosticCode,
)
from psyneulink.core.batched.ir import (
    BatchedScheduleTraceSpec,
    BatchedScheduleTraceStepSpec,
)

from batched_semantic_test_support import (
    SemanticCase,
    SemanticModel,
    assert_matches_python,
)


pytestmark = [pytest.mark.batched, pytest.mark.composition]


_AT_PASS_LLVM_PROVENANCE = (
    "tests/scheduling/test_scheduler.py::"
    "TestFeedback::test_scheduler_conditions[AtPass]"
)
_MULTI_PARENT_LLVM_PROVENANCE = (
    "tests/composition/test_composition.py::"
    "TestRun::test_3_mechanisms_2_origins_1_terminal"
)
_MULTI_TRIAL_LLVM_PROVENANCE = (
    "tests/composition/test_composition.py::"
    "TestRun::test_run_2_mechanisms_with_multiple_trials_of_input_values"
)
_TERMINATION_LLVM_PROVENANCE = (
    "tests/scheduling/test_scheduler.py::"
    "TestFeedback::test_run_term_conditions"
)
_WHEN_FINISHED_LLVM_PROVENANCE = (
    "tests/scheduling/test_scheduler.py::"
    "TestFeedback::test_time_termination_measures"
)
_CONTROL_LLVM_PROVENANCE = (
    "tests/composition/test_control.py::"
    "TestControlMechanisms::test_control_of_mech_port[OVERRIDE]"
)


_THREE_TRIAL_INPUTS = np.array([[1.0], [-2.0], [0.5]])


def _delayed_sender_implicit_receiver_case():
    build_number = itertools.count()

    def build():
        suffix = next(build_number)
        sender = pnl.TransferMechanism(
            input_shapes=1,
            function=pnl.Linear(slope=2.0, intercept=0.5),
            name=f"delayed sender {suffix}",
        )
        receiver = pnl.TransferMechanism(
            input_shapes=1,
            function=pnl.Linear(slope=-1.5, intercept=0.25),
            name=f"implicit receiver {suffix}",
        )
        composition = pnl.Composition(pathways=[[sender, receiver]])
        composition.scheduler.add_condition(sender, pnl.AtPass(3))
        return SemanticModel(
            composition=composition,
            inputs={sender: _THREE_TRIAL_INPUTS.copy()},
            outputs=(receiver.output_port,),
        )

    return SemanticCase(
        name="delayed_sender_implicit_receiver",
        build=build,
        provenance=f"{_AT_PASS_LLVM_PROVENANCE}; {_MULTI_TRIAL_LLVM_PROVENANCE}",
        max_steps=8,
    )


def _always_sender_delayed_receiver_case():
    build_number = itertools.count()

    def build():
        suffix = next(build_number)
        sender = pnl.TransferMechanism(
            input_shapes=1,
            function=pnl.Linear(slope=-0.75, intercept=1.0),
            name=f"always sender {suffix}",
        )
        receiver = pnl.TransferMechanism(
            input_shapes=1,
            function=pnl.Linear(slope=2.5, intercept=-0.5),
            name=f"delayed receiver {suffix}",
        )
        composition = pnl.Composition(pathways=[[sender, receiver]])
        composition.scheduler.add_condition(sender, pnl.Always())
        composition.scheduler.add_condition(receiver, pnl.AtPass(3))
        return SemanticModel(
            composition=composition,
            inputs={sender: _THREE_TRIAL_INPUTS.copy()},
            outputs=(receiver.output_port,),
        )

    return SemanticCase(
        name="always_sender_delayed_receiver",
        build=build,
        provenance=f"{_AT_PASS_LLVM_PROVENANCE}; {_MULTI_TRIAL_LLVM_PROVENANCE}",
        max_steps=8,
    )


def _delayed_multi_parent_sum_case():
    build_number = itertools.count()
    right_inputs = np.array([[4.0], [-3.0], [2.0]])

    def build():
        suffix = next(build_number)
        left = pnl.TransferMechanism(
            input_shapes=1,
            function=pnl.Linear(slope=2.0),
            name=f"left delayed origin {suffix}",
        )
        right = pnl.TransferMechanism(
            input_shapes=1,
            function=pnl.Linear(slope=-1.0),
            name=f"right delayed origin {suffix}",
        )
        receiver = pnl.TransferMechanism(
            input_shapes=1,
            function=pnl.Linear(slope=0.5),
            name=f"implicit sum receiver {suffix}",
        )
        composition = pnl.Composition()
        # Insertion order is intentionally unrelated to dependency order.
        composition.add_nodes([receiver, right, left])
        composition.add_projection(sender=left, receiver=receiver)
        composition.add_projection(sender=right, receiver=receiver)
        composition.scheduler.add_condition(left, pnl.AtPass(1))
        composition.scheduler.add_condition(right, pnl.AtPass(3))
        return SemanticModel(
            composition=composition,
            inputs={
                left: _THREE_TRIAL_INPUTS.copy(),
                right: right_inputs.copy(),
            },
            outputs=(receiver.output_port,),
        )

    return SemanticCase(
        name="delayed_multi_parent_implicit_sum",
        build=build,
        provenance=(
            f"{_AT_PASS_LLVM_PROVENANCE}; {_MULTI_PARENT_LLVM_PROVENANCE}; "
            f"{_MULTI_TRIAL_LLVM_PROVENANCE}"
        ),
        max_steps=8,
    )


EXECUTABLE_CASES = (
    _delayed_sender_implicit_receiver_case(),
    _always_sender_delayed_receiver_case(),
    _delayed_multi_parent_sum_case(),
)


_EXPECTED_TRACES = {
    "delayed_sender_implicit_receiver": (
        (
            (3, 0, (0,)),
            (3, 1, (1,)),
        ),
        4,
        2,
    ),
    "always_sender_delayed_receiver": (
        (
            (0, 0, (0,)),
            (1, 0, (0,)),
            (2, 0, (0,)),
            (3, 0, (0,)),
            (3, 1, (1,)),
        ),
        4,
        5,
    ),
    "delayed_multi_parent_implicit_sum": (
        (
            (1, 0, (0,)),
            (3, 0, (1,)),
            (3, 1, (2,)),
        ),
        4,
        3,
    ),
}


def _assert_executable_trace_contract(case, backend):
    model = case.build()
    plan = BatchedCompositionCompiler.compile(
        model.composition,
        backend=backend,
        outputs=tuple(model.outputs),
        max_steps=case.max_steps,
    )
    report = plan.capability_report

    assert report.model_supported
    assert report.codegen_ready is True
    assert report.backend_available
    assert report.can_execute
    assert not report.model_diagnostics
    assert not report.codegen_diagnostics
    assert not report.backend_diagnostics
    assert report.metadata["fusion_kind"] == "stateless_graph"
    assert report.metadata["schedule_kind"] == "precomputed_trace"

    graph = plan.ir.graph
    assert graph is not None
    assert graph.executable
    assert graph.metadata["schedule_kind"] == "precomputed_trace"
    assert plan.kernel_ir.executable
    assert plan.kernel_ir.termination == graph.termination

    trace = plan.kernel_ir.schedule_trace
    assert type(trace) is BatchedScheduleTraceSpec
    assert all(type(step) is BatchedScheduleTraceStepSpec for step in trace.steps)
    expected_steps, expected_passes, expected_executions = _EXPECTED_TRACES[
        case.name
    ]
    assert tuple(
        (
            step.pass_index,
            step.consideration_set_id,
            step.component_ids,
        )
        for step in trace.steps
    ) == expected_steps
    assert trace.num_passes == expected_passes
    assert trace.component_execution_count == expected_executions
    assert sum(len(step.component_ids) for step in trace.steps) == (
        trace.component_execution_count
    )


@pytest.mark.parametrize("case", EXECUTABLE_CASES, ids=lambda case: case.name)
def test_stateless_schedule_matches_python(case, batched_backend):
    """The typed precomputed trace must execute with Python trial semantics."""

    _assert_executable_trace_contract(case, batched_backend)

    comparison = assert_matches_python(case, backend=batched_backend)

    # Every builder uses multiple trials so pass/call counts must reset at each
    # trial boundary without introducing mechanism state.
    assert comparison.python_values.shape == (len(_THREE_TRIAL_INPUTS), 1)


@dataclass(frozen=True)
class _RejectionCase:
    name: str
    build: Callable[[], pnl.Composition]
    provenance: str
    expected_code: str


def _freshness_hazard_composition():
    sender = pnl.TransferMechanism(input_shapes=1, name="late producer")
    receiver = pnl.TransferMechanism(input_shapes=1, name="early consumer")
    composition = pnl.Composition(pathways=[[sender, receiver]])
    composition.scheduler.add_condition(sender, pnl.AtPass(3))
    composition.scheduler.add_condition(receiver, pnl.AtPass(0))
    return composition


def _custom_termination_composition():
    mechanism = pnl.TransferMechanism(input_shapes=1, name="custom termination node")
    return pnl.Composition(
        pathways=mechanism,
        termination_processing={pnl.TimeScale.TRIAL: pnl.AtPass(2)},
    )


def _stateful_composition():
    mechanism = pnl.TransferMechanism(
        input_shapes=1,
        integrator_mode=True,
        name="stateful schedule node",
    )
    return pnl.Composition(pathways=mechanism)


def _same_set_when_finished_composition():
    producer = pnl.TransferMechanism(input_shapes=1, name="finished producer")
    consumer = pnl.TransferMechanism(input_shapes=1, name="finished consumer")
    composition = pnl.Composition()
    composition.add_nodes([producer, consumer])
    composition.scheduler.add_condition(consumer, pnl.WhenFinished(producer))
    return composition


def _when_finished_control_composition():
    source = pnl.TransferMechanism(input_shapes=1, name="control monitor")
    target = pnl.TransferMechanism(input_shapes=1, name="controlled target")
    controller = pnl.ControlMechanism(
        function=pnl.Identity(),
        monitor_for_control=source,
        control_signals=[(pnl.SLOPE, target)],
        modulation=pnl.OVERRIDE,
        name="scheduled controller",
    )
    composition = pnl.Composition()
    composition.add_nodes([target, controller, source])
    composition.scheduler.add_condition(controller, pnl.WhenFinished(source))
    return composition


REJECTION_CASES = (
    _RejectionCase(
        name="receiver_before_sender_freshness_hazard",
        build=_freshness_hazard_composition,
        provenance=_AT_PASS_LLVM_PROVENANCE,
        expected_code=BatchedDiagnosticCode.MODEL_SCHEDULE_NOT_EXECUTABLE,
    ),
    _RejectionCase(
        name="custom_trial_termination",
        build=_custom_termination_composition,
        provenance=_TERMINATION_LLVM_PROVENANCE,
        expected_code=(
            BatchedDiagnosticCode.MODEL_SCHEDULER_TERMINATION_UNSUPPORTED
        ),
    ),
    _RejectionCase(
        name="stateful_node",
        build=_stateful_composition,
        provenance=_WHEN_FINISHED_LLVM_PROVENANCE,
        expected_code=BatchedDiagnosticCode.MODEL_STATEFUL_TRANSFER_UNSUPPORTED,
    ),
    _RejectionCase(
        name="same_set_when_finished",
        build=_same_set_when_finished_composition,
        provenance=_WHEN_FINISHED_LLVM_PROVENANCE,
        expected_code=BatchedDiagnosticCode.MODEL_SCHEDULE_NOT_EXECUTABLE,
    ),
    _RejectionCase(
        name="when_finished_generic_control",
        build=_when_finished_control_composition,
        provenance=f"{_WHEN_FINISHED_LLVM_PROVENANCE}; {_CONTROL_LLVM_PROVENANCE}",
        expected_code=BatchedDiagnosticCode.MODEL_UNSUPPORTED,
    ),
)


@pytest.mark.parametrize("case", REJECTION_CASES, ids=lambda case: case.name)
def test_schedule_outside_first_executable_subset_fails_closed(case):
    composition = case.build()
    report = BatchedCompositionCompiler.diagnose(
        composition,
        backend="triton_cpu",
        max_steps=8,
    )

    assert not report.model_supported, (
        f"{case.name} must remain fail-closed; provenance: {case.provenance}"
    )
    assert report.model_diagnostics
    assert case.expected_code in {
        diagnostic.code for diagnostic in report.model_diagnostics
    }
    assert all(
        diagnostic.component_id is not None
        for diagnostic in report.model_diagnostics
    )
    with pytest.raises(BatchedCompileError) as error:
        BatchedCompositionCompiler.compile(
            composition,
            backend="triton_cpu",
            max_steps=8,
        )
    assert error.value.capability_report == report
