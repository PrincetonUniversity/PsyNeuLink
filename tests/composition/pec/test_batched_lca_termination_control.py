"""Acceptance boundary for LCA-controlled generic dynamic schedules.

The CSI surrogate in ``Scripts/Debug/pec_batch_compile/csi_model_surrogate.py``
motivates these cases.  This module isolates its first control chain:

``AtPass(n) input -> Always LCA -> WhenFinished(LCA) DDM``

The generic executor accepts affine controls whose static onset relationship
is sound and matches their multi-trial Python semantics.  Deliberately
mismatched controller thresholds remain structured, fail-closed rejections.
"""

from dataclasses import dataclass

import numpy as np
import pytest

import psyneulink as pnl
from tests.composition.pec.batched_semantic_test_support import (
    SemanticCase,
    SemanticModel,
    assert_matches_python,
)
from psyneulink.core.batched import (
    BatchedCompileError,
    BatchedCompositionCompiler,
    BatchedDiagnosticCode,
)
from psyneulink.core.batched.kernel_ir import iter_kernel_ops

pytestmark = [
    pytest.mark.batched,
    pytest.mark.composition,
]


_UNSUPPORTED_SCHEDULE_DETAIL = (
    "coevolving Always/WhenFinished execution requires explicit finished "
    "predicates and conditional pass regions in KernelIR"
)

_VECTOR_INPUTS = np.asarray(
    [[1.0, -0.5], [-0.25, 1.25], [0.75, -1.0]],
    dtype=float,
)
_TIMING_INPUTS = np.asarray([[0.0], [1.0], [2.0]], dtype=float)


@dataclass(frozen=True)
class _TerminationTiming:
    onset: int
    monitor_slope: float = 0.0
    monitor_intercept: float = 0.0
    controller_intercept: float | None = None

    @property
    def static_controller_intercept(self):
        if self.controller_intercept is None:
            return float(self.onset)
        return float(self.controller_intercept)


def _make_scheduled_model(timing: _TerminationTiming) -> SemanticModel:
    """Build the target graph shape without relying on any CSI-specific name."""

    delayed_input = pnl.TransferMechanism(
        input_shapes=2,
        function=pnl.Linear,
        integrator_mode=True,
        integration_rate=1.0,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=0,
        execute_until_finished=False,
        reset_stateful_function_when=pnl.AtTrialStart(),
        name="delayed vector origin",
    )
    timing_source = pnl.ProcessingMechanism(
        input_shapes=1,
        function=pnl.Linear(
            slope=timing.monitor_slope,
            intercept=timing.monitor_intercept,
        ),
        name="arbitrary scalar origin",
    )
    stepper = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(gain=3.0),
        leak=1.0,
        competition=1.0,
        self_excitation=0.0,
        noise=0.0,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=1,
        time_step_size=0.01,
        execute_until_finished=False,
        name="persistent upstream stepper",
    )
    termination_override = pnl.ControlMechanism(
        function=pnl.Linear(
            slope=1.0,
            intercept=timing.static_controller_intercept,
        ),
        monitor_for_control=timing_source,
        control_signals=[(pnl.TERMINATION_THRESHOLD, stepper)],
        modulation=pnl.OVERRIDE,
        name="generic parameter override",
    )
    terminator = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0,
            rate=1.0,
            noise=0.0,
            threshold=0.02,
            non_decision_time=0.0,
            time_step_size=0.01,
        ),
        reset_stateful_function_when=pnl.AtTrialStart(),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        execute_until_finished=False,
        name="lane-local terminator",
    )
    decision_readout = pnl.ProcessingMechanism(
        input_shapes=1,
        name="decision readout",
    )
    response_time_readout = pnl.ProcessingMechanism(
        input_shapes=1,
        name="response-time readout",
    )

    composition = pnl.Composition()
    composition.add_nodes(
        [
            delayed_input,
            timing_source,
            stepper,
            termination_override,
            terminator,
            decision_readout,
            response_time_readout,
        ]
    )
    composition.add_projection(sender=delayed_input, receiver=stepper)
    composition.add_projection(
        sender=stepper,
        receiver=terminator,
        projection=pnl.MappingProjection(matrix=[[1.0], [-1.0]]),
    )
    composition.add_projection(
        sender=terminator.output_ports[pnl.DECISION_OUTCOME],
        receiver=decision_readout,
    )
    composition.add_projection(
        sender=terminator.output_ports[pnl.RESPONSE_TIME],
        receiver=response_time_readout,
    )

    composition.scheduler.add_condition(delayed_input, pnl.AtPass(timing.onset))
    composition.scheduler.add_condition(timing_source, pnl.AtPass(0))
    composition.scheduler.add_condition(termination_override, pnl.AtPass(0))
    composition.scheduler.add_condition(stepper, pnl.Always())
    composition.scheduler.add_condition(terminator, pnl.WhenFinished(stepper))
    composition.scheduler.add_condition(
        decision_readout,
        pnl.WhenFinished(terminator),
    )
    composition.scheduler.add_condition(
        response_time_readout,
        pnl.WhenFinished(terminator),
    )

    return SemanticModel(
        composition=composition,
        inputs={
            delayed_input: _VECTOR_INPUTS,
            timing_source: _TIMING_INPUTS,
        },
        outputs=(decision_readout.output_port, response_time_readout.output_port),
    )


TERMINATION_TIMING_CASES = (
    pytest.param(
        _TerminationTiming(onset=0),
        id="zero_monitor_zero_onset",
    ),
    pytest.param(
        _TerminationTiming(onset=3),
        id="zero_monitor_matching_static_onset",
    ),
    pytest.param(
        _TerminationTiming(onset=0, monitor_slope=2.0),
        id="cue_dependent_monitor",
    ),
    pytest.param(
        _TerminationTiming(onset=0, monitor_intercept=2.0),
        id="nonzero_repeat_monitor",
    ),
    pytest.param(
        _TerminationTiming(onset=3, controller_intercept=2.0),
        id="controller_threshold_before_onset",
    ),
    pytest.param(
        _TerminationTiming(onset=3, controller_intercept=4.0),
        id="controller_threshold_after_onset",
    ),
)

SUPPORTED_TIMING_CASES = TERMINATION_TIMING_CASES[:4]
MISMATCHED_TIMING_CASES = TERMINATION_TIMING_CASES[4:]


def _make_uncontrolled_dynamic_model():
    stepper = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(gain=2.0),
        leak=1.0,
        competition=0.5,
        self_excitation=0.0,
        noise=0.0,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=2,
        time_step_size=0.01,
        execute_until_finished=False,
        name="uncontrolled persistent stepper",
    )
    terminator = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0,
            rate=1.0,
            noise=0.0,
            threshold=0.02,
            non_decision_time=0.0,
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        execute_until_finished=False,
        name="uncontrolled lane-local terminator",
    )
    composition = pnl.Composition()
    composition.add_nodes([stepper, terminator])
    composition.add_projection(
        sender=stepper,
        receiver=terminator,
        projection=pnl.MappingProjection(matrix=[[1.0], [-1.0]]),
    )
    composition.scheduler.add_condition(stepper, pnl.Always())
    composition.scheduler.add_condition(terminator, pnl.WhenFinished(stepper))
    return composition, stepper, tuple(terminator.output_ports)


def _dynamic_regions(kernel):
    return tuple(
        op
        for op in iter_kernel_ops(kernel)
        if op.kind == "ForPasses"
        and op.attrs.get("trace_kind") == "lane_local_dynamic"
    )


def _assert_mismatched_control_diagnostic(report, composition):
    steppers = tuple(
        node for node in composition.nodes if isinstance(node, pnl.LCAMechanism)
    )
    assert len(steppers) == 1
    stepper = steppers[0]
    assert not report.model_supported
    assert len(report.model_diagnostics) == 1
    diagnostic = report.model_diagnostics[0]
    assert diagnostic.component == stepper.name
    assert diagnostic.code == BatchedDiagnosticCode.MODEL_SCHEDULE_NOT_EXECUTABLE
    assert diagnostic.component_id == f"node:{stepper.name}"
    assert diagnostic.reason == "batched schedule kind is not executable yet"
    assert diagnostic.detail == _UNSUPPORTED_SCHEDULE_DETAIL
    return diagnostic


@pytest.mark.parametrize("timing", SUPPORTED_TIMING_CASES)
def test_controlled_lca_schedule_matches_fresh_python(timing, batched_backend):
    compiled_model = _make_scheduled_model(timing)
    report = BatchedCompositionCompiler.diagnose(
        compiled_model.composition,
        backend=batched_backend,
        outputs=compiled_model.outputs,
        max_steps=128,
    )
    assert report.can_execute
    assert report.metadata["schedule_kind"] == "dynamic_lane_local"
    assert not report.model_diagnostics
    assert not report.codegen_diagnostics

    plan = BatchedCompositionCompiler.compile(
        compiled_model.composition,
        backend=batched_backend,
        outputs=compiled_model.outputs,
        max_steps=128,
    )
    assert len(_dynamic_regions(plan.kernel_ir)) == 1
    comparison = assert_matches_python(
        SemanticCase(
            name=repr(timing),
            build=lambda: _make_scheduled_model(timing),
            provenance=__file__,
            max_steps=128,
        ),
        backend=batched_backend,
    )

    assert comparison.python_values.shape == (
        len(_VECTOR_INPUTS),
        len(compiled_model.outputs),
    )


def test_controlled_lca_compile_reports_generic_schedule_capability():
    model = _make_scheduled_model(_TerminationTiming(onset=0))
    diagnosed = BatchedCompositionCompiler.diagnose(
        model.composition,
        outputs=model.outputs,
        max_steps=128,
    )
    plan = BatchedCompositionCompiler.compile(
        model.composition,
        outputs=model.outputs,
        max_steps=128,
    )

    assert plan.capability_report == diagnosed
    assert diagnosed.can_execute
    assert diagnosed.metadata["schedule_kind"] == "dynamic_lane_local"
    assert len(_dynamic_regions(plan.kernel_ir)) == 1


@pytest.mark.parametrize("timing", MISMATCHED_TIMING_CASES)
def test_mismatched_lca_control_remains_structured_rejection(timing):
    model = _make_scheduled_model(timing)
    report = BatchedCompositionCompiler.diagnose(
        model.composition,
        outputs=model.outputs,
        max_steps=128,
    )

    _assert_mismatched_control_diagnostic(report, model.composition)


def test_uncontrolled_lca_when_finished_remains_structured_rejection():
    composition, stepper, outputs = _make_uncontrolled_dynamic_model()
    report = BatchedCompositionCompiler.diagnose(
        composition,
        outputs=outputs,
        max_steps=128,
    )

    matches = [
        diagnostic
        for diagnostic in report.model_diagnostics
        if diagnostic.component == stepper.name
        and diagnostic.reason == "batched schedule kind is not executable yet"
    ]
    assert not report.model_supported
    assert len(matches) == 1
    diagnostic = matches[0]
    assert diagnostic.code == BatchedDiagnosticCode.MODEL_SCHEDULE_NOT_EXECUTABLE
    assert diagnostic.component_id == f"node:{stepper.name}"
    assert diagnostic.detail == _UNSUPPORTED_SCHEDULE_DETAIL

    with pytest.raises(BatchedCompileError) as error:
        BatchedCompositionCompiler.compile(
            composition,
            outputs=outputs,
            max_steps=128,
        )
    assert error.value.capability_report == report
    assert diagnostic.formatted_reason in str(error.value)
