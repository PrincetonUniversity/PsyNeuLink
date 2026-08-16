"""Fail-closed boundary for LCA-controlled termination in co-evolving graphs.

The CSI surrogate in ``Scripts/Debug/pec_batch_compile/csi_model_surrogate.py``
motivates these cases, but it also contains a UDF and a second control chain.
This module keeps only the semantic pattern relevant to the batched compiler:

``AtPass(n) input -> Always LCA -> WhenFinished(LCA) DDM``

The current KernelIR records the static onset ``n``, but this LCA-to-DDM
topology is outside the first typed controlled-finished subset and still lacks
an executable conditional pass region.  Even constant-zero and statically
matching controls must therefore fail closed until generic scheduler/control
lowering can preserve those semantics.
"""

from dataclasses import dataclass

import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompileError,
    BatchedCompositionCompiler,
    BatchedDiagnosticCode,
)

pytestmark = [
    pytest.mark.batched,
    pytest.mark.composition,
]


_UNMODELED_CONTROL_DETAIL = (
    "coevolving Always/WhenFinished execution falls outside the typed "
    "controlled-finished subset and requires executable conditional pass regions"
)
_UNMODELED_COEVOLUTION_DETAIL = (
    "coevolving Always/WhenFinished execution requires explicit finished "
    "predicates and conditional pass regions in KernelIR"
)


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


@dataclass(frozen=True)
class _CoevolvingModel:
    composition: pnl.Composition
    outputs: tuple


def _make_coevolving_model(timing: _TerminationTiming) -> _CoevolvingModel:
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

    return _CoevolvingModel(
        composition=composition,
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


def _termination_override(composition):
    """Find the relevant control edge by its receiver semantics, not its name."""

    matches = []
    for node in composition.nodes:
        if not isinstance(node, pnl.ControlMechanism):
            continue
        for signal in node.control_signals:
            for projection in signal.efferents:
                receiver = projection.receiver
                if (
                    isinstance(receiver.owner, pnl.LCAMechanism)
                    and receiver.name == pnl.TERMINATION_THRESHOLD
                ):
                    matches.append(node)
    assert len(matches) == 1
    return matches[0]


def _make_uncontrolled_coevolving_model():
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


def _assert_unmodeled_control_diagnostic(report, controller):
    assert not report.model_supported
    assert len(report.model_diagnostics) == 1
    diagnostic = report.model_diagnostics[0]
    assert diagnostic.component == controller.name
    assert diagnostic.code == BatchedDiagnosticCode.MODEL_SCHEDULE_NOT_EXECUTABLE
    assert diagnostic.component_id == f"node:{controller.name}"
    assert diagnostic.reason == "batched schedule kind is not executable yet"
    assert diagnostic.detail == _UNMODELED_CONTROL_DETAIL
    return diagnostic


@pytest.mark.parametrize("timing", TERMINATION_TIMING_CASES)
def test_controlled_lca_when_finished_has_structured_rejection(timing):
    model = _make_coevolving_model(timing)
    controller = _termination_override(model.composition)
    report = BatchedCompositionCompiler.diagnose(
        model.composition,
        outputs=model.outputs,
        max_steps=128,
    )

    _assert_unmodeled_control_diagnostic(report, controller)


def test_controlled_lca_compile_error_carries_capability_report():
    # Use the most tempting special case (zero monitor, zero onset): it must not
    # bypass capability analysis merely because a sampled execution can appear
    # equivalent while this coevolving predicate remains non-executable.
    model = _make_coevolving_model(_TerminationTiming(onset=0))
    controller = _termination_override(model.composition)
    diagnosed = BatchedCompositionCompiler.diagnose(
        model.composition,
        outputs=model.outputs,
        max_steps=128,
    )

    with pytest.raises(BatchedCompileError) as error:
        BatchedCompositionCompiler.compile(
            model.composition,
            outputs=model.outputs,
            max_steps=128,
        )

    report = error.value.capability_report
    assert report == diagnosed
    diagnostic = _assert_unmodeled_control_diagnostic(report, controller)
    assert diagnostic.formatted_reason in str(error.value)


def test_uncontrolled_lca_when_finished_rejects_missing_generic_schedule_ir():
    composition, stepper, outputs = _make_uncontrolled_coevolving_model()
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
    assert diagnostic.detail == _UNMODELED_COEVOLUTION_DETAIL

    with pytest.raises(BatchedCompileError) as error:
        BatchedCompositionCompiler.compile(
            composition,
            outputs=outputs,
            max_steps=128,
        )
    assert error.value.capability_report == report
    assert diagnostic.formatted_reason in str(error.value)
