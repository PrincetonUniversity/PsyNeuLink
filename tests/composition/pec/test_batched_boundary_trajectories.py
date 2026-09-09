"""Deterministic boundary prefixes versus actual coupled step inputs."""

from dataclasses import replace

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    AxisDependencyEdge, BatchedTrialParameter, BoundaryTrajectoryError, LikelihoodEffectContract,
    batched_node_op, unregister_batched_instance_op,
)
from test_batched_history_replay import _plan, _model, _node, _csi_drift_rate, coupled


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def test_boundary_fields_are_typed_from_actual_step_inputs(coupled):
    composition, _, outputs = coupled
    plan = _plan(composition, outputs).compile_boundary_trajectories()
    assert [field.kind for field in plan.witness.fields] == ["input", "held_modulation"]
    assert [field.column_start for field in plan.witness.fields] == [0, 1]
    assert plan.witness.source_component_ids
    assert plan.witness.static_parameter_ids
    source = plan.source()
    assert "while trial_idx < num_trials" not in source
    assert "path_trial = offsets % num_trials" in source
    assert "tl.rand" not in source
    assert "path_starts" in source
    assert "while trial_idx < num_trials" in plan.source(parallel_trials=False)


def test_boundary_dependency_check_is_separate_from_history_check(coupled, monkeypatch):
    """Synthetic dependency evidence isolates the boundary rejection rule."""
    from psyneulink.core.batched import trajectories

    composition, _, outputs = coupled
    history = _plan(composition, outputs)
    kernel = history.simulation_plan.kernel_ir
    clock = history.witness.endpoint.clock_component_id
    sender = next(p.sender_component_id for p in kernel.graph.projections if p.receiver_component_id == clock)
    axis = trajectories.analyze_axis_dependencies(kernel.graph, kernel.params)
    # Only the boundary pass receives this synthetic random-input edge. The
    # history checker still revalidates its real source dependencies normally.
    altered = replace(axis, edges=(*axis.edges, AxisDependencyEdge(clock, sender, "projection")))
    monkeypatch.setattr(trajectories, "analyze_axis_dependencies", lambda *args: altered)
    with pytest.raises(BoundaryTrajectoryError) as error:
        history.compile_boundary_trajectories()
    assert error.value.code == "boundary.stochastic_dependency"


@pytest.mark.parametrize("mutation", [
    lambda w: replace(w, fields=()),
    lambda w: replace(w, source_component_ids=()),
    lambda w: replace(w, static_parameter_ids=()),
    lambda w: replace(w, consideration_set_id=w.consideration_set_id + 1),
    lambda w: replace(w, history=replace(w.history, state_ids=())),
    lambda w: replace(w, fields=(replace(w.fields[0], value_name="unrelated"), *w.fields[1:])),
])
def test_forged_boundary_witness_cannot_generate_source(coupled, mutation):
    composition, _, outputs = coupled
    plan = _plan(composition, outputs).compile_boundary_trajectories()
    with pytest.raises(BoundaryTrajectoryError, match="does not match"):
        replace(plan, witness=mutation(plan.witness)).source()


@pytest.mark.parametrize("options, code", [
    ({"horizon": 0}, "boundary.horizon"),
    ({"horizon": 129}, "boundary.horizon"),
    ({"horizon": True}, "boundary.horizon"),
    ({"max_buffer_bytes": 1}, "boundary.memory_budget"),
])
def test_resource_guards_run_before_history_execution(coupled, options, code):
    composition, inputs, outputs = coupled
    plan = _plan(composition, outputs).compile_boundary_trajectories()
    with pytest.raises(BoundaryTrajectoryError) as error:
        plan.generate(inputs, None, **options)
    assert error.value.code == code


@pytest.mark.triton_interpreter
@pytest.mark.parametrize("noise", [0.0, 0.15])
def test_parallel_paths_match_coupled_pre_step_boundary(noise):
    composition, inputs, outputs = _model(ddm_noise=noise)
    drift = _node(composition, "Drift Rate Value")
    batched_node_op(drift.name, likelihood_contract=LikelihoodEffectContract())(_csi_drift_rate)
    try:
        plan = _plan(composition, outputs).compile_boundary_trajectories()
        reference = plan.simulate_reference(inputs, seed=12)
        ordinary = plan.history_plan.simulate_reference(inputs, seed=12)
        np.testing.assert_array_equal(reference.history.observations, ordinary.observations)
        generated = plan.generate(inputs, reference.history.observations[0])
        assert generated.mode == "observed_history_paths"
        assert generated.history.observations is None
        assert np.all(generated.valid)
        np.testing.assert_array_equal(generated.values[reference.valid], reference.values[reference.valid])
        np.testing.assert_array_equal(generated.pass_indices[reference.valid], reference.pass_indices[reference.valid])
        np.testing.assert_array_equal(generated.history.end_states, reference.history.end_states)
        assert np.all(np.isnan(reference.values[~reference.valid]))
        assert np.all(reference.pass_indices[~reference.valid] == -1)
        assert not generated.values.flags.writeable
        assert not generated.valid.flags.writeable
        assert not generated.pass_indices.flags.writeable
    finally:
        unregister_batched_instance_op(drift.name)


@pytest.mark.triton_interpreter
def test_hypothetical_tails_never_become_next_trial_history(coupled):
    composition, inputs, outputs = coupled
    plan = _plan(composition, outputs).compile_boundary_trajectories()
    reference = plan.simulate_reference(inputs, horizon=8)
    data = reference.history.observations[0]
    short = plan.generate(inputs, data, horizon=4)
    longer = plan.generate(inputs, data, horizon=8)
    np.testing.assert_array_equal(short.values, longer.values[:, :, :4])
    np.testing.assert_array_equal(short.history.end_states, longer.history.end_states)
    # Alter only trial 1's observed endpoint. Trial 1's path starts at the same
    # state; trial 2 must start at the newly reconstructed carried state.
    changed_data = data.copy()
    changed_data[0, 1] += 0.05
    changed = plan.generate(inputs, changed_data, horizon=4)
    np.testing.assert_array_equal(short.values[:, 0], changed.values[:, 0])
    assert not np.array_equal(short.values[:, 1], changed.values[:, 1])
    np.testing.assert_array_equal(short.history.event_counts + [[5, 0]], changed.history.event_counts)


@pytest.mark.triton_interpreter
def test_candidate_and_trial_parameter_lanes_match_separate_generation(coupled):
    composition, inputs, outputs = coupled
    plan = _plan(composition, outputs).compile_boundary_trajectories()
    data = plan.simulate_reference(inputs, horizon=4).history.observations[0]
    ndt = f"{_node(composition, 'DDM').name}.non_decision_time"
    gain = f"{_node(composition, 'Task Activations [C1, C2]').name}.gain"
    rows = [{}, {ndt: 0.31}, {ndt: BatchedTrialParameter([0.31, 0.32]), gain: BatchedTrialParameter([2.0, 1.5])}]
    batched = plan.generate(inputs, data, rows, horizon=4)
    for index, parameters in enumerate(rows):
        single = plan.generate(inputs, data, parameters, horizon=4)
        np.testing.assert_array_equal(batched.values[index], single.values[0])
        np.testing.assert_array_equal(batched.pass_indices[index], single.pass_indices[0])
        np.testing.assert_array_equal(batched.history.end_states[index], single.history.end_states[0])


def _alternative_drift(x0, x1, x2, x3, x4, x5, x6):
    return 0.25 * (x0 - x1 + x2 - x3) + 0.1 * (x4 - x5) + 0.05 * x6


@pytest.mark.triton_interpreter
def test_paths_follow_renamed_model_new_drift_and_projection(coupled):
    composition, inputs, outputs = coupled
    drift = _node(composition, "Drift Rate Value")
    decision = _node(composition, "DDM")
    for projection in decision.path_afferents:
        projection.parameters.matrix.set([[1.5]])
    original_name = drift.name
    for index, node in enumerate(composition.nodes):
        node.name = f"path node {index}"
    batched_node_op(drift.name, likelihood_contract=LikelihoodEffectContract())(_alternative_drift)
    try:
        plan = _plan(composition, outputs).compile_boundary_trajectories()
        reference = plan.simulate_reference(inputs, horizon=32)
        generated = plan.generate(inputs, reference.history.observations[0], horizon=32)
        np.testing.assert_array_equal(generated.values[reference.valid], reference.values[reference.valid])
        np.testing.assert_array_equal(generated.pass_indices[reference.valid], reference.pass_indices[reference.valid])
        assert "0.25" in plan.source()
    finally:
        unregister_batched_instance_op(drift.name)
        drift.name = original_name


@pytest.mark.triton_interpreter
@pytest.mark.parametrize("reset", [False, True])
def test_delayed_onset_paths_preserve_resets_and_held_control_starts(reset):
    composition, inputs, outputs = _model(iti=2, csi_repeat=3, csi_switch=4, cue_values=[[0.0], [1.0]])
    drift = _node(composition, "Drift Rate Value")
    if reset:
        _node(composition, "Task Activations [C1, C2]").reset_stateful_function_when = pnl.AtTrialStart()
    batched_node_op(drift.name, likelihood_contract=LikelihoodEffectContract())(_csi_drift_rate)
    try:
        plan = _plan(composition, outputs).compile_boundary_trajectories()
        reference = plan.simulate_reference(inputs, horizon=16)
        generated = plan.generate(inputs, reference.history.observations[0], horizon=16)
        np.testing.assert_array_equal(generated.values[reference.valid], reference.values[reference.valid])
        np.testing.assert_array_equal(generated.pass_indices[reference.valid], reference.pass_indices[reference.valid])
        np.testing.assert_array_equal(generated.history.start_effective_parameters, reference.history.start_effective_parameters)
    finally:
        unregister_batched_instance_op(drift.name)
