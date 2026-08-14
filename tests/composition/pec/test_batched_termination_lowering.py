"""Backend-neutral contract for Composition termination lowering.

The Python traces below retain the stopping behavior exercised by the broader
scheduler corpus, notably
``tests/scheduling/test_condition.py::TestCondition::test_AllHaveRun`` and
``TestCondition::test_AtTrial``.  Batched compilation may precompute only the
typed default termination contract at this checkpoint; every custom condition
must fail closed until its semantics are executable.
"""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompileError,
    BatchedCompositionCompiler,
    BatchedDiagnosticCode,
)
from psyneulink.core.batched.graph import lower_composition
from psyneulink.core.batched.ir import BatchedTerminationSpec


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _chain(name, termination_factory=None):
    source = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Linear(slope=2.0),
        name=f"{name} source",
    )
    receiver = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Linear(slope=3.0),
        name=f"{name} receiver",
    )
    kwargs = {}
    if termination_factory is not None:
        kwargs["termination_processing"] = termination_factory(source)
    composition = pnl.Composition(
        pathways=[source, receiver],
        name=f"{name} composition",
        **kwargs,
    )
    return composition, source, receiver


def _execution_trace(composition):
    return tuple(
        frozenset(execution_set)
        for execution_set in composition.scheduler.execution_list[
            composition.default_execution_id
        ]
    )


def _assert_data_only(value):
    """Reject live PsyNeuLink conditions and components anywhere in the IR."""

    if is_dataclass(value) and not isinstance(value, type):
        assert type(value).__module__.startswith("psyneulink.core.batched")
        for field in fields(value):
            _assert_data_only(getattr(value, field.name))
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_data_only(key)
            _assert_data_only(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _assert_data_only(item)
        return
    assert value is None or isinstance(value, (str, int, float, bool))


def _assert_structured_termination_rejection(
    composition,
    output_port,
    *,
    expected_detail,
):
    lowering = lower_composition(composition, outputs=(output_port,))
    termination_diagnostics = [
        diagnostic
        for diagnostic in lowering.rejected_conditions
        if "termination" in diagnostic.reason.lower()
    ]
    assert len(termination_diagnostics) == 1, lowering.rejected_conditions
    raw_diagnostic = termination_diagnostics[0]
    assert raw_diagnostic.component == composition.name
    for text in expected_detail:
        assert text in raw_diagnostic.formatted_reason

    report = BatchedCompositionCompiler.diagnose(
        composition,
        backend="triton_cpu",
        outputs=(output_port,),
    )
    assert not report.model_supported
    assert report.codegen_ready is None
    assert len(report.rejected_conditions) == 1, report.to_dict()
    diagnostic = report.rejected_conditions[0]
    assert diagnostic.code == (
        BatchedDiagnosticCode.MODEL_SCHEDULER_TERMINATION_UNSUPPORTED
    )
    assert diagnostic.component == composition.name
    assert diagnostic.component_id == f"composition:{composition.name}"
    for text in expected_detail:
        assert text in diagnostic.formatted_reason

    with pytest.raises(BatchedCompileError) as error:
        BatchedCompositionCompiler.compile(
            composition,
            backend="triton_cpu",
            outputs=(output_port,),
        )
    assert error.value.capability_report == report


def test_default_termination_is_typed_expanded_and_data_only():
    python_composition, python_source, python_receiver = _chain("default python")
    result = python_composition.run(inputs={python_source: [[2.0]]})

    np.testing.assert_array_equal(result, [[12.0]])
    assert _execution_trace(python_composition) == (
        frozenset((python_source,)),
        frozenset((python_receiver,)),
    )

    composition, _, receiver = _chain("default lowering")
    lowering = lower_composition(composition, outputs=(receiver.output_port,))

    assert not lowering.rejected_conditions
    graph = lowering.graph
    assert graph is not None
    component_ids = tuple(sorted(node.component_id for node in graph.nodes))
    assert graph.termination == (
        BatchedTerminationSpec(
            time_scale="ENVIRONMENT_STATE_UPDATE",
            condition_type="AllHaveRun",
            dependency_component_ids=component_ids,
        ),
        BatchedTerminationSpec(
            time_scale="ENVIRONMENT_SEQUENCE",
            condition_type="Never",
        ),
    )
    _assert_data_only(graph.termination)


def test_custom_all_have_run_subset_is_rejected_after_python_midpass_stop():
    def subset(source):
        return {pnl.TimeScale.TRIAL: pnl.AllHaveRun(source)}

    python_composition, python_source, python_receiver = _chain(
        "subset python",
        subset,
    )
    result = python_composition.run(inputs={python_source: [[2.0]]})

    # AllHaveRun(source) is checked between consideration sets, so the terminal
    # receiver never executes and its initial value is the Composition result.
    np.testing.assert_array_equal(result, [[0.0]])
    assert _execution_trace(python_composition) == (frozenset((python_source,)),)
    np.testing.assert_array_equal(python_receiver.value, [[0.0]])

    composition, _, receiver = _chain("subset lowering", subset)
    _assert_structured_termination_rejection(
        composition,
        receiver.output_port,
        expected_detail=("AllHaveRun", "every scheduler component ID"),
    )


def test_all_have_run_internal_nondefault_time_scale_is_rejected():
    def all_have_run_per_pass(_source):
        return {
            pnl.TimeScale.TRIAL: pnl.AllHaveRun(time_scale=pnl.TimeScale.PASS),
        }

    python_composition, python_source, python_receiver = _chain(
        "per-pass termination python",
        all_have_run_per_pass,
    )
    trial_termination = python_composition.termination_processing[pnl.TimeScale.TRIAL]
    assert trial_termination.time_scale is pnl.TimeScale.PASS
    result = python_composition.run(inputs={python_source: [[2.0]]})

    np.testing.assert_array_equal(result, [[12.0]])
    assert _execution_trace(python_composition) == (
        frozenset((python_source,)),
        frozenset((python_receiver,)),
    )

    composition, _, receiver = _chain(
        "per-pass termination lowering",
        all_have_run_per_pass,
    )
    _assert_structured_termination_rejection(
        composition,
        receiver.output_port,
        expected_detail=("AllHaveRun", "PASS"),
    )


def test_all_have_run_string_time_scale_is_not_treated_as_the_enum_member():
    def all_have_run_string_clock(_source):
        return {
            pnl.TimeScale.TRIAL: pnl.AllHaveRun(
                time_scale="ENVIRONMENT_STATE_UPDATE",
            ),
        }

    composition, _, receiver = _chain(
        "string-clock termination lowering",
        all_have_run_string_clock,
    )
    _assert_structured_termination_rejection(
        composition,
        receiver.output_port,
        expected_detail=("AllHaveRun", "ENVIRONMENT_STATE_UPDATE"),
    )


def test_name_equivalent_fake_termination_map_key_is_rejected():
    class _FakeTrialScale:
        name = "ENVIRONMENT_STATE_UPDATE"

    composition, _, receiver = _chain("fake-key termination lowering")
    conditions = composition.scheduler.termination_conds
    trial = conditions.pop(pnl.TimeScale.TRIAL)
    conditions[_FakeTrialScale()] = trial

    _assert_structured_termination_rejection(
        composition,
        receiver.output_port,
        expected_detail=("canonical PsyNeuLink", "TimeScale"),
    )


@pytest.mark.parametrize(
    "time_scale",
    (pnl.TimeScale.TRIAL, pnl.TimeScale.RUN),
)
def test_malformed_termination_args_reject_without_diagnose_crash(time_scale):
    composition, _, receiver = _chain("malformed-args termination lowering")
    composition.scheduler.termination_conds[time_scale].args = None

    _assert_structured_termination_rejection(
        composition,
        receiver.output_port,
        expected_detail=("args=None",),
    )


def test_custom_trial_at_pass_is_rejected_after_python_two_pass_trace():
    def at_pass_two(_source):
        return {pnl.TimeScale.TRIAL: pnl.AtPass(2)}

    python_composition, python_source, python_receiver = _chain(
        "trial AtPass python",
        at_pass_two,
    )
    result = python_composition.run(inputs={python_source: [[2.0]]})

    np.testing.assert_array_equal(result, [[12.0]])
    assert _execution_trace(python_composition) == (
        frozenset((python_source,)),
        frozenset((python_receiver,)),
        frozenset((python_source,)),
        frozenset((python_receiver,)),
    )

    composition, _, receiver = _chain("trial AtPass lowering", at_pass_two)
    _assert_structured_termination_rejection(
        composition,
        receiver.output_port,
        expected_detail=("AllHaveRun",),
    )


def test_custom_sequence_termination_is_rejected_after_python_run_stop():
    def at_trial_one(_source):
        return {pnl.TimeScale.RUN: pnl.AtTrial(1)}

    python_composition, python_source, python_receiver = _chain(
        "sequence AtTrial python",
        at_trial_one,
    )
    result = python_composition.run(
        inputs={python_source: [[1.0], [2.0], [3.0]]},
        num_trials=3,
    )

    # The requested three-trial run stops after trial zero.  The output 6.0 is
    # 1.0 transformed by source slope 2 and receiver slope 3.
    np.testing.assert_array_equal(result, [[6.0]])
    np.testing.assert_array_equal(python_composition.results, [[[6.0]]])
    assert _execution_trace(python_composition) == (
        frozenset((python_source,)),
        frozenset((python_receiver,)),
    )

    composition, _, receiver = _chain("sequence AtTrial lowering", at_trial_one)
    _assert_structured_termination_rejection(
        composition,
        receiver.output_port,
        expected_detail=("Never",),
    )
