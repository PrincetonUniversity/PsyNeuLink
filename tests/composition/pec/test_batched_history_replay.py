"""Checked event replay versus unmodified coupled execution."""

from dataclasses import replace

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompileError, BatchedCompositionCompiler, BatchedTrialParameter, HistoryReplayError, LikelihoodEffectContract,
    batched_node_op, unregister_batched_instance_op,
)
from psyneulink.core.batched.history import validate_history_witness
from psyneulink.core.batched.specs import register_batched_op
from test_batched_endpoints import _model, _node, _csi_drift_rate, _spec, coupled
from test_batched_likelihood_analysis import _ddm


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _plan(composition, outputs):
    return BatchedCompositionCompiler.compile_history_replay(composition, _spec(outputs), max_steps=128)


@pytest.mark.parametrize("mutation", [
    lambda w: replace(w, state_ids=()),
    lambda w: replace(w, effective_parameter_ids=()),
    lambda w: replace(w, resolved_termination_edges=()),
    lambda w: replace(w, endpoint=replace(w.endpoint, minimum_count=0)),
    lambda w: replace(w, guarantee="proved"),
])
def test_forged_history_witness_is_rejected(coupled, mutation):
    composition, _, outputs = coupled
    plan = _plan(composition, outputs)
    with pytest.raises(HistoryReplayError, match="does not match"):
        validate_history_witness(plan.simulation_plan, plan.observations, mutation(plan.witness))
    with pytest.raises(HistoryReplayError):
        replace(plan, witness=mutation(plan.witness)).source()


def test_replay_requires_known_initial_state(coupled):
    composition, _, outputs = coupled
    observations = replace(_spec(outputs), initial_state="latent")
    with pytest.raises(HistoryReplayError) as error:
        BatchedCompositionCompiler.compile_history_replay(composition, observations, max_steps=128)
    assert error.value.code == "history.dependencies_unresolved"


def test_replay_source_uses_no_random_draws(coupled):
    composition, _, outputs = coupled
    plan = _plan(composition, outputs)
    source = plan.source()
    assert "history_target" in source
    assert "_pnl_triton_ddm_update" not in source
    assert "tl.rand" not in source
    assert "_pnl_triton_logistic" in source
    assert "_pnl_triton_ddm_update" in plan.source(replay=False)


def test_algebraic_readout_alone_does_not_authorize_clock_substitution(coupled):
    composition, _, outputs = coupled
    plan = _plan(composition, outputs)
    original = plan.simulation_plan.kernel_ir.op_specs.lookup_spec(plan.witness.endpoint.clock_spec_key)
    contract = original.likelihood_contract
    try:
        register_batched_op(replace(original, likelihood_contract=replace(
            contract, event_readout=replace(contract.event_readout, execution_rule=None),
            # Strip the continuous reset-process assertion as well: this test
            # deliberately leaves only an algebraic counter readout.
            wiener_readout=None,
        )))
        # Old snapshots remain valid; new plans require the stronger rule.
        assert "history_target" in plan.source()
        with pytest.raises(HistoryReplayError) as error:
            _plan(composition, outputs)
        assert error.value.code == "history.execution_rule_missing"
    finally:
        register_batched_op(original)


def test_cyclic_stochastic_feedback_keeps_source_lowering_rejection(coupled):
    composition, _, outputs = coupled
    decision = _node(composition, "DDM")
    lca = _node(composition, "Task Activations [C1, C2]")
    composition.add_projection(
        sender=decision.output_ports[0], receiver=lca,
        projection=pnl.MappingProjection(matrix=[[1.0, -1.0]]),
    )
    with pytest.raises(BatchedCompileError, match="cyclic processing dependencies"):
        _plan(composition, outputs)


def test_supported_stochastic_state_dependency_is_rejected_by_history_analysis():
    composition, decision = _ddm()
    state = pnl.LCAMechanism(
        input_shapes=2, noise=0.0, function=pnl.Logistic(),
        termination_measure=pnl.TimeScale.TRIAL, termination_threshold=3,
        reset_stateful_function_when=pnl.Never(),
    )
    composition.add_node(state)
    composition.add_projection(sender=decision.output_ports[0], receiver=state,
                               projection=pnl.MappingProjection(matrix=[[1.0, -1.0]]))
    outputs = (state.output_port, decision.output_ports[1])
    simulation = BatchedCompositionCompiler.compile(composition, outputs=outputs, max_steps=128)
    assert simulation.capability_report.is_supported
    with pytest.raises(HistoryReplayError) as error:
        simulation.compile_history_replay(_spec(outputs))
    assert error.value.code == "history.dependencies_unresolved"


@pytest.mark.triton_interpreter
@pytest.mark.parametrize("noise", [0.0, 0.15])
def test_replay_recovers_coupled_states_controls_and_schedule(noise):
    composition, inputs, outputs = _model(ddm_noise=noise)
    drift = _node(composition, "Drift Rate Value")
    batched_node_op(drift.name, likelihood_contract=LikelihoodEffectContract())(_csi_drift_rate)
    try:
        plan = _plan(composition, outputs)
        forward = plan.simulate_reference(inputs, seed=12)
        ordinary = plan.simulation_plan.run(inputs, [{}], num_estimates=1, seed=12,
                                            strict_truncation=True, return_final_states=True)
        np.testing.assert_array_equal(forward.observations, ordinary.values[:, 0, :, 0, :])
        np.testing.assert_array_equal(forward.end_states[:, -1], ordinary.metadata["final_states"][:, 0, 0])
        reconstructed = plan.reconstruct(inputs, forward.observations[0])
        assert reconstructed.observations is None
        for name in (
            "start_states", "end_states", "start_effective_parameters", "end_effective_parameters",
            "execution_counts", "scheduler_rounds", "event_counts",
        ):
            np.testing.assert_array_equal(getattr(reconstructed, name), getattr(forward, name), err_msg=name)
            assert not getattr(reconstructed, name).flags.writeable
        np.testing.assert_array_equal(reconstructed.start_states[:, 1:], reconstructed.end_states[:, :-1])
        # CSI's LCA starts before the DDM and still updates on its final pass.
        lca_id = next(node.component_id for node in plan.simulation_plan.ir.graph.nodes
                      if node.name == _node(composition, "Task Activations [C1, C2]").name)
        lca_calls = reconstructed.execution_counts[..., plan.witness.component_ids.index(lca_id)]
        cue = np.asarray(inputs[_node(composition, "Cue Stimulus Interval")]).reshape(-1)
        np.testing.assert_array_equal(lca_calls, reconstructed.event_counts + cue - 1)
    finally:
        unregister_batched_instance_op(drift.name)


@pytest.mark.triton_interpreter
def test_candidate_and_trial_parameters_recompute_history(coupled):
    composition, inputs, outputs = coupled
    plan = _plan(composition, outputs)
    reference = plan.simulate_reference(inputs)
    data = reference.observations[0]
    parameter = f"{_node(composition, 'DDM').name}.non_decision_time"
    rows = [{}, {parameter: 0.31}, {
        parameter: BatchedTrialParameter([0.31, 0.32]),
    }]
    batched = plan.reconstruct(inputs, data, rows)
    assert not np.array_equal(batched.end_states[0], batched.end_states[1])
    np.testing.assert_array_equal(batched.event_counts[1], batched.event_counts[0] - 1)
    np.testing.assert_array_equal(batched.event_counts[2], batched.event_counts[0] - [1, 2])
    for index, parameters in enumerate(rows):
        single = plan.reconstruct(inputs, data, parameters)
        np.testing.assert_array_equal(batched.end_states[index], single.end_states[0])
        np.testing.assert_array_equal(batched.end_effective_parameters[index], single.end_effective_parameters[0])


@pytest.mark.triton_interpreter
def test_renamed_model_changed_gates_and_longer_prelude(coupled):
    composition, inputs, outputs = coupled
    response = outputs[1].owner
    response.function.parameters.slope.set(2.0)
    response.function.parameters.intercept.set(0.1)
    decision = _node(composition, "DDM")
    for projection in response.path_afferents:
        if projection.sender is decision.output_ports[1]:
            projection.parameters.matrix.set([[3.0]])
    cue = _node(composition, "Cue Stimulus Interval")
    inputs[cue] = [[5.0], [8.0]]
    drift = _node(composition, "Drift Rate Value")
    original_name = drift.name
    for index, node in enumerate(composition.nodes):
        node.name = f"history node {index}"
    batched_node_op(drift.name, likelihood_contract=LikelihoodEffectContract())(_csi_drift_rate)
    try:
        plan = _plan(composition, outputs)
        forward = plan.simulate_reference(inputs)
        reconstructed = plan.reconstruct(inputs, forward.observations[0])
        np.testing.assert_array_equal(reconstructed.end_states, forward.end_states)
        np.testing.assert_array_equal(reconstructed.execution_counts, forward.execution_counts)
        np.testing.assert_array_equal(reconstructed.end_effective_parameters, forward.end_effective_parameters)
    finally:
        unregister_batched_instance_op(drift.name)
        drift.name = original_name


@pytest.mark.triton_interpreter
def test_trial_resets_are_replayed_by_the_original_scheduler(coupled):
    composition, inputs, outputs = coupled
    lca = _node(composition, "Task Activations [C1, C2]")
    lca.reset_stateful_function_when = pnl.AtTrialStart()
    plan = _plan(composition, outputs)
    assert all(reset.condition_type == "AtTrialStart" for reset in plan.simulation_plan.ir.graph.resets)
    forward = plan.simulate_reference(inputs)
    replayed = plan.reconstruct(inputs, forward.observations[0])
    np.testing.assert_array_equal(replayed.end_states, forward.end_states)
    np.testing.assert_array_equal(replayed.start_states, forward.start_states)
    np.testing.assert_array_equal(replayed.end_effective_parameters, forward.end_effective_parameters)


@pytest.mark.triton_interpreter
def test_one_step_and_exact_cap_still_execute_the_terminating_pass(coupled):
    composition, inputs, outputs = coupled
    plan = _plan(composition, outputs)
    cue = np.asarray(inputs[_node(composition, "Cue Stimulus Interval")]).reshape(-1)
    counts = np.array([1, 128])
    data = np.column_stack((np.ones(2), 0.3 + (counts + cue) * 0.01))
    replayed = plan.reconstruct(inputs, data)
    np.testing.assert_array_equal(replayed.event_counts, counts[None, :])
    lca_id = next(node.component_id for node in plan.simulation_plan.ir.graph.nodes
                  if node.name == _node(composition, "Task Activations [C1, C2]").name)
    np.testing.assert_array_equal(
        replayed.execution_counts[..., plan.witness.component_ids.index(lca_id)],
        (counts + cue - 1)[None, :],
    )


@pytest.mark.triton_interpreter
@pytest.mark.parametrize("options", [
    {"iti": 10, "csi_repeat": 0, "csi_switch": 0, "cue_values": [[0.0], [1.0]]},
    {"iti": 2, "csi_repeat": 3, "csi_switch": 4, "cue_values": [[0.0], [1.0], [0.0]], "lca_noise": 0.1},
])
def test_delayed_and_zero_onset_replay_preserves_held_controls(options):
    composition, inputs, outputs = _model(**options)
    drift = _node(composition, "Drift Rate Value")
    batched_node_op(drift.name, likelihood_contract=LikelihoodEffectContract())(_csi_drift_rate)
    try:
        plan = _plan(composition, outputs)
        forward = plan.simulate_reference(inputs)
        replayed = plan.reconstruct(inputs, forward.observations[0])
        np.testing.assert_array_equal(replayed.end_states, forward.end_states)
        np.testing.assert_array_equal(replayed.execution_counts, forward.execution_counts)
        np.testing.assert_array_equal(replayed.end_effective_parameters, forward.end_effective_parameters)
        np.testing.assert_array_equal(replayed.start_effective_parameters[:, 1:], replayed.end_effective_parameters[:, :-1])
    finally:
        unregister_batched_instance_op(drift.name)
