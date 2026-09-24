"""The original CSI composition, rather than a likelihood, defines semantics."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompositionCompiler, LikelihoodEffectContract,
    ObservationField, ObservationSpec, batched_node_op, unregister_batched_instance_op,
)
from psyneulink.core.batched.graph import lower_composition
from test_batched_csi_coevolving_acceptance import _csi_drift_rate, _node, _selected_python_results


pytestmark = [pytest.mark.batched, pytest.mark.composition, pytest.mark.llvm]


def _original_model():
    path = (Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile/csi"
            / "csi_fit/data fitting/expectation_model_study2_study3.py")
    spec = importlib.util.spec_from_file_location("_original_csi_threshold_regression", path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model


@pytest.mark.triton_gpu
@pytest.mark.parametrize("legacy_schedule", [False, True])
def test_generated_csi_matches_llvm_after_threshold_collapse(legacy_schedule):
    """A fresh threshold and fresh DDM output remove the cross-trial artifact.

    A zero-drift first trial reaches boundary collapse. The next trial has
    positive drift, but LLVM still terminates it on its first step using the
    previous negative threshold under the legacy schedule. The fixed model
    must integrate normally, and the compiler must reproduce both schedules.
    """
    model = _original_model()

    def build():
        composition = model.make_stab_flex(
            gain=10., leak=12., competition=3., iti=2, csi_switch=0,
            threshold=.05005, threshold_collapse=-.0003,
            non_decision_time=.2, ddm_noise=0., lca_noise=0.,
            lca_time_step_size=.001, ddm_time_step_size=.001,
        )
        inputs = {
            _node(composition, "Task Input"): [[1., 0.]] * 3,
            _node(composition, "Stimulus Input"): [[1., 0., 1., 0.]] * 3,
            _node(composition, "Correct Response"): [[0.], [1.], [1.]],
            _node(composition, "Cue Stimulus Interval"): [[0.]] * 3,
            _node(composition, "Threshold Mechanism"): [[0.]] * 3,
        }
        outputs = tuple(_node(composition, name).output_port
                        for name in ("DECISION_GATE", "RESPONSE_GATE"))
        return composition, inputs, outputs

    composition, inputs, outputs = build()
    if legacy_schedule:
        for conditions in list(composition.scheduler.conditions.conditions_structural.values()):
            for condition in list(conditions):
                composition.scheduler.remove_condition(condition)
        for name in ("DECISION_GATE", "RESPONSE_GATE"):
            composition.scheduler.add_condition(
                _node(composition, name), pnl.WhenFinished(_node(composition, "DDM")),
            )
    drift = _node(composition, "Drift Rate Value")
    batched_node_op(drift.name, likelihood_contract=LikelihoodEffectContract())(_csi_drift_rate)
    try:
        plan = BatchedCompositionCompiler.compile(
            composition, backend="triton", outputs=outputs, max_steps=256,
        )
        composition.run(inputs=inputs, execution_mode=pnl.ExecutionMode.LLVMRun)
        expected = _selected_python_results(composition, outputs)
        # This is actual LLVM model output, not its likelihood. The compiled
        # plan was frozen before this run and starts from the same initial state.
        np.testing.assert_allclose(expected[1], [0., .201] if legacy_schedule else [1., .308],
                                   rtol=0, atol=1e-12)
        assert expected[0, 1] > .35
        assert expected[2, 1] > .21
        actual = plan.run(inputs, [{}], num_estimates=1, seed=101,
                          strict_truncation=True).values[0, 0, :, 0]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
        observations = ObservationSpec((
            ObservationField(outputs[0], "counting"),
            ObservationField(outputs[1], "lebesgue", role="event_time", history_timing="ceil_fp32_8ulp"),
        ))
        history = plan.compile_history_replay(observations)
        paths = history.compile_boundary_trajectories().generate(inputs, expected, horizon=2)
        threshold = next(field.column_start for field in paths.fields if field.kind == "held_modulation")
        if legacy_schedule:
            assert paths.values[0, 1, 0, threshold] < 0
            np.testing.assert_array_equal(paths.history.event_counts[0, 1], 1)
        else:
            np.testing.assert_allclose(paths.values[0, :, 0, threshold], .05005 - .0003,
                                       rtol=0, atol=1e-7)
            np.testing.assert_array_equal(paths.history.event_counts[0, 1], 108)
        assert paths.values[0, 1, 1, threshold] > 0
    finally:
        unregister_batched_instance_op(drift.name)


@pytest.mark.parametrize("mutation", ["reversed", "different_dependency", "two_calls", "custom_callable", "split_threshold_chain"])
def test_fixed_csi_scheduler_admission_is_exact(mutation):
    composition = _original_model().make_stab_flex()
    ddm = _node(composition, "DDM")
    lca = _node(composition, "Task Activations [C1, C2]")
    gate = _node(composition, "RESPONSE_GATE")
    if mutation == "split_threshold_chain":
        for condition in list(composition.scheduler.conditions.conditions_structural[lca]):
            composition.scheduler.remove_condition(condition)
        controller = next(node for node in composition.nodes if node.name.startswith("ControlMechanism"))
        composition.scheduler.add_condition(lca, pnl.AddEdgeTo(controller))
    else:
        call = pnl.EveryNCalls(lca if mutation == "different_dependency" else ddm,
                              2 if mutation == "two_calls" else 1)
        if mutation == "custom_callable":
            call.func = lambda *args, **kwargs: True
        composition.scheduler.add_condition(gate, pnl.All(call, pnl.WhenFinished(ddm)))
    drift = _node(composition, "Drift Rate Value")
    batched_node_op(drift.name, likelihood_contract=LikelihoodEffectContract())(_csi_drift_rate)
    try:
        lowering = lower_composition(composition)
        if mutation == "reversed":
            assert lowering.graph is not None and lowering.graph.executable
            predicate = next(item for item in lowering.graph.scheduler if item.node == gate.name)
            assert predicate.condition_type == "WhenFinishedAndEveryNCalls"
        else:
            assert lowering.graph is None or not lowering.graph.executable
            assert lowering.rejected_conditions or lowering.rejected_nodes
    finally:
        unregister_batched_instance_op(drift.name)
