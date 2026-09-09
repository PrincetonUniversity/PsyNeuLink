"""Declared observation outputs and explicit count-domain empirical masses."""

from dataclasses import replace

import numpy as np
import pytest

from psyneulink.core.batched import (
    BatchedCompositionCompiler, LikelihoodEffectContract, StochasticSamplingError,
    batched_node_op, unregister_batched_instance_op,
)
from test_batched_endpoints import _model, _node, _csi_drift_rate, _spec, coupled


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _observed(history):
    return history.compile_boundary_trajectories().compile_stochastic_sampler().compile_observation_sampler()


@pytest.mark.parametrize("mutation", [
    lambda w: replace(w, readouts=()),
    lambda w: replace(w, readouts=tuple(reversed(w.readouts))),
    lambda w: replace(w, readouts=(replace(w.readouts[0], expression=w.readouts[1].expression), *w.readouts[1:])),
])
def test_observation_witness_is_rederived(coupled, mutation):
    composition, _, outputs = coupled
    plan = _observed(BatchedCompositionCompiler.compile_history_replay(composition, _spec(outputs), max_steps=128))
    with pytest.raises(StochasticSamplingError, match="does not match"):
        replace(plan, witness=mutation(plan.witness)).source()


def test_mass_does_not_infer_a_density(coupled):
    composition, _, outputs = coupled
    observations = _spec(outputs)
    observations = replace(observations, fields=(observations.fields[0], replace(observations.fields[1], measure="lebesgue")))
    with pytest.raises(StochasticSamplingError) as error:
        BatchedCompositionCompiler.compile_empirical_mass(composition, observations, max_steps=128)
    assert error.value.code == "mass.measure"


@pytest.mark.parametrize("noise", [0.0, 0.15])
def test_observations_and_mass_match_coupled_execution(batched_backend, noise):
    composition, inputs, outputs = _model(ddm_noise=noise)
    drift = _node(composition, "Drift Rate Value")
    batched_node_op(drift.name, likelihood_contract=LikelihoodEffectContract())(_csi_drift_rate)
    try:
        mass = BatchedCompositionCompiler.compile_empirical_mass(composition, _spec(outputs), backend=batched_backend, max_steps=128)
        plan = mass.observation_plan
        history = plan.sampler.path_plan.history_plan
        forward = history.simulate_reference(inputs, seed=12)
        data = forward.observations[0]
        # Event-time gate and choice gate have their actual source readouts.
        once = plan.sample(inputs, data, seed=12)
        np.testing.assert_allclose(once.values[:, :, 0], forward.observations, rtol=1e-6, atol=1e-7)
        rows = [{}, {f"{_node(composition, 'DDM').name}.non_decision_time": 0.31}]
        options = dict(num_estimates=17, seed=21, common_random_numbers=False)
        sampled = plan.sample(inputs, data, rows, **options)
        reference = plan.simulate_reference(inputs, data, rows, **options)
        np.testing.assert_allclose(sampled.values, reference.values, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(sampled.event_counts, reference.event_counts)
        score = mass.score(inputs, data, rows, **options)
        reference_score = mass.score(inputs, data, rows, reference=True, **options)
        manual = ((sampled.event_counts == sampled.history.event_counts[..., None])
                  & (sampled.values[..., 0] == data[None, :, None, 0])).sum(axis=-1)
        np.testing.assert_array_equal(score.successes, manual)
        np.testing.assert_array_equal(score.successes, reference_score.successes)
        np.testing.assert_array_equal(score.probabilities, manual / options["num_estimates"])
        assert score.backend == batched_backend
    finally:
        unregister_batched_instance_op(drift.name)


def test_mass_zero_hits_and_unscored_time(batched_backend):
    composition, inputs, outputs = _model(ddm_noise=0.0)
    drift = _node(composition, "Drift Rate Value")
    batched_node_op(drift.name, likelihood_contract=LikelihoodEffectContract())(_csi_drift_rate)
    try:
        observations = _spec(outputs)
        observations = replace(observations, fields=(observations.fields[0], replace(observations.fields[1], score=False)))
        mass = BatchedCompositionCompiler.compile_empirical_mass(composition, observations, backend=batched_backend, max_steps=128)
        history = mass.observation_plan.sampler.path_plan.history_plan
        data = history.simulate_reference(inputs).observations[0]
        normal = mass.score(inputs, data, num_estimates=4)
        np.testing.assert_array_equal(normal.probabilities, [[1.0, 1.0]])
        impossible = data.copy()
        impossible[:, 0] = 999
        absent = mass.score(inputs, impossible, num_estimates=4)
        assert np.all(absent.zero_hits)
        assert np.all(np.isneginf(absent.log_likelihood))
        np.testing.assert_array_equal(absent.probabilities, [[0.0, 0.0]])
        with pytest.raises(StochasticSamplingError) as error:
            mass.score(inputs, data, num_estimates=4, horizon=1)
        assert error.value.code == "sampling.truncated"
    finally:
        unregister_batched_instance_op(drift.name)
