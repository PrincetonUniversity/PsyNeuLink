"""Generated stochastic region versus full coupled conditional trials."""

from dataclasses import replace

import numpy as np
import pytest

from psyneulink.core.batched import (
    BatchedCompositionCompiler, BatchedTrialParameter, LikelihoodEffectContract, StochasticSamplingError,
    batched_node_op, unregister_batched_instance_op,
)
from test_batched_history_replay import _plan, _model, _node, _csi_drift_rate, coupled


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def test_sampler_uses_source_primitive_without_coupled_scheduler(coupled):
    composition, _, outputs = coupled
    sampler = _plan(composition, outputs).compile_boundary_trajectories().compile_stochastic_sampler()
    source = sampler.source()
    assert "_pnl_triton_ddm_step(" in source
    assert "path_values + path_row" in source
    assert "dynamic_pass" not in source
    assert "while trial_idx < num_trials" not in source
    assert len(sampler.witness.outputs) == 2
    assert sum(field.width for field in sampler.witness.outputs) == 2


@pytest.mark.parametrize("mutation", [
    lambda w: replace(w, outputs=()),
    lambda w: replace(w, rng_stream_ids=()),
    lambda w: replace(w, spec_key="unregistered"),
])
def test_forged_sampler_witness_rejected(coupled, mutation):
    composition, _, outputs = coupled
    sampler = _plan(composition, outputs).compile_boundary_trajectories().compile_stochastic_sampler()
    with pytest.raises(StochasticSamplingError, match="does not match"):
        replace(sampler, witness=mutation(sampler.witness)).source()


@pytest.mark.parametrize("options, code", [
    ({"horizon": 0}, "sampling.horizon"),
    ({"horizon": 129}, "sampling.horizon"),
    ({"num_estimates": True}, "sampling.estimates"),
    ({"num_estimates": 0}, "sampling.estimates"),
    ({"max_buffer_bytes": 1}, "sampling.memory_budget"),
])
def test_sampler_guards_before_history_execution(coupled, options, code):
    composition, inputs, outputs = coupled
    sampler = _plan(composition, outputs).compile_boundary_trajectories().compile_stochastic_sampler()
    with pytest.raises(StochasticSamplingError) as error:
        sampler.sample(inputs, None, **options)
    assert error.value.code == code


@pytest.mark.triton_interpreter
@pytest.mark.parametrize("noise", [0.0, 0.15])
@pytest.mark.parametrize("common_random", [True, False])
def test_samples_match_full_coupled_conditional_reference(noise, common_random):
    composition, inputs, outputs = _model(ddm_noise=noise)
    drift = _node(composition, "Drift Rate Value")
    batched_node_op(drift.name, likelihood_contract=LikelihoodEffectContract())(_csi_drift_rate)
    try:
        history = _plan(composition, outputs)
        data = history.simulate_reference(inputs, seed=12).observations[0]
        sampler = history.compile_boundary_trajectories().compile_stochastic_sampler()
        ndt = f"{_node(composition, 'DDM').name}.non_decision_time"
        rows = [{}, {ndt: 0.31}]
        options = dict(num_estimates=8, seed=21, common_random_numbers=common_random)
        sampled = sampler.sample(inputs, data, rows, **options)
        reference = sampler.simulate_reference(inputs, data, rows, **options)
        np.testing.assert_array_equal(sampled.values, reference.values)
        np.testing.assert_array_equal(sampled.event_counts, reference.event_counts)
        np.testing.assert_array_equal(sampled.truncated, reference.truncated)
        np.testing.assert_array_equal(sampled.history.end_states, reference.history.end_states)
        assert not np.any(sampled.truncated)
        assert sampled.values.shape == (2, 2, 8, 2)
        assert not sampled.values.flags.writeable
    finally:
        unregister_batched_instance_op(drift.name)


@pytest.mark.triton_interpreter
def test_raw_samples_match_ordinary_forward_and_candidate_rng():
    composition, inputs, outputs = _model(ddm_noise=0.3)
    drift = _node(composition, "Drift Rate Value")
    batched_node_op(drift.name, likelihood_contract=LikelihoodEffectContract())(_csi_drift_rate)
    try:
        _check_forward_and_candidate_rng(composition, inputs, outputs)
    finally:
        unregister_batched_instance_op(drift.name)


def _check_forward_and_candidate_rng(composition, inputs, outputs):
    decision = _node(composition, "DDM")
    history = _plan(composition, outputs)
    sampler = history.compile_boundary_trajectories().compile_stochastic_sampler()
    forward_plan = BatchedCompositionCompiler.compile(
        composition, outputs=(*outputs, *decision.output_ports), max_steps=128,
    )
    forward = forward_plan.run(inputs, [{}], num_estimates=1, seed=31, strict_truncation=True)
    data = forward.values[0, 0, :, 0, :2]
    sampled = sampler.sample(inputs, data, num_estimates=1, seed=31)
    np.testing.assert_array_equal(sampled.values[0, :, 0], forward.values[0, 0, :, 0, 2:])
    # Identical candidates share draws under CRN, and batching estimates does
    # not duplicate state/history updates. Noise is frozen in the source plan.
    row = {}
    shared = sampler.sample(inputs, data, [row, row], num_estimates=8, seed=31)
    np.testing.assert_array_equal(shared.values[0], shared.values[1])
    independent = sampler.sample(inputs, data, [row, row], num_estimates=8, seed=31,
                                 common_random_numbers=False)
    assert not np.array_equal(independent.values[0], independent.values[1])
    assert np.unique(shared.event_counts).size > 1


@pytest.mark.triton_interpreter
def test_delayed_onset_and_trial_parameters_preserve_source_execution():
    composition, inputs, outputs = _model(ddm_noise=0.15, iti=2, csi_repeat=3, csi_switch=4,
                                         cue_values=[[0.0], [1.0]])
    drift = _node(composition, "Drift Rate Value")
    batched_node_op(drift.name, likelihood_contract=LikelihoodEffectContract())(_csi_drift_rate)
    try:
        history = _plan(composition, outputs)
        data = history.simulate_reference(inputs, seed=12).observations[0]
        sampler = history.compile_boundary_trajectories().compile_stochastic_sampler()
        decision = _node(composition, "DDM")
        rows = [{f"{decision.name}.non_decision_time": BatchedTrialParameter([0.31, 0.32]),
                 f"{_node(composition, 'Task Activations [C1, C2]').name}.gain": BatchedTrialParameter([2.0, 1.5])}]
        options = dict(num_estimates=8, seed=7)
        sampled = sampler.sample(inputs, data, rows, **options)
        reference = sampler.simulate_reference(inputs, data, rows, **options)
        np.testing.assert_array_equal(sampled.values, reference.values)
        np.testing.assert_array_equal(sampled.event_counts, reference.event_counts)
    finally:
        unregister_batched_instance_op(drift.name)


@pytest.mark.triton_interpreter
def test_truncation_is_explicit_and_horizon_does_not_change_history(coupled):
    composition, inputs, outputs = coupled
    history = _plan(composition, outputs)
    data = history.simulate_reference(inputs).observations[0]
    sampler = history.compile_boundary_trajectories().compile_stochastic_sampler()
    with pytest.raises(StochasticSamplingError) as error:
        sampler.sample(inputs, data, horizon=1, num_estimates=4)
    assert error.value.code == "sampling.truncated"
    options = dict(horizon=1, num_estimates=4, strict_truncation=False)
    sampled = sampler.sample(inputs, data, **options)
    reference = sampler.simulate_reference(inputs, data, **options)
    assert np.all(sampled.truncated)
    np.testing.assert_array_equal(sampled.truncated, reference.truncated)
    np.testing.assert_array_equal(sampled.values, reference.values)
    np.testing.assert_array_equal(sampled.event_counts, np.ones((1, 2, 4), dtype=np.int32))
    completed = sampler.sample(inputs, data, num_estimates=4)
    np.testing.assert_array_equal(sampled.history.end_states, completed.history.end_states)
    assert not np.any(completed.truncated)
