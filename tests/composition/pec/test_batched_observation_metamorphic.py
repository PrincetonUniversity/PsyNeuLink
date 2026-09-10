"""Observation lowering follows graph identities and source affine parameters."""

import numpy as np
import pytest

from psyneulink.core.batched import (
    BatchedCompositionCompiler, LikelihoodEffectContract, ObservationField, ObservationSpec,
    StochasticSamplingError, batched_node_op, unregister_batched_instance_op,
)
from test_batched_endpoints import _model, _node, _csi_drift_rate, _spec, coupled


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def test_second_scored_event_time_cannot_reuse_first_count_unchecked(coupled):
    composition, _, outputs = coupled
    original = _spec(outputs)
    observations = ObservationSpec((*original.fields, ObservationField(
        _node(composition, "DDM").output_ports[1], "counting", role="event_time", condition_history=False,
    )))
    with pytest.raises(StochasticSamplingError) as error:
        BatchedCompositionCompiler.compile_empirical_mass(composition, observations, max_steps=128)
    assert error.value.code == "mass.event_field"


@pytest.mark.parametrize("time_sign", [1., -1.])
def test_renamed_gates_projection_and_reordered_observations(batched_backend, time_sign):
    composition, inputs, _ = _model(ddm_noise=0.15, iti=2, csi_repeat=3, csi_switch=4,
                                    cue_values=[[0.0], [1.0]])
    choice = _node(composition, "DECISION_GATE")
    timing = _node(composition, "RESPONSE_GATE")
    drift = _node(composition, "Drift Rate Value")
    choice.function.parameters.slope.set(-2.0)
    choice.function.parameters.intercept.set(1.0)
    choice.function.parameters.scale.set(4.0)
    choice.function.parameters.offset.set(0.25)
    timing.function.parameters.slope.set(2.0 * time_sign)
    timing.function.parameters.intercept.set(0.125)
    timing.function.parameters.scale.set(0.5)
    timing.function.parameters.offset.set(0.25)
    for projection in choice.path_afferents:
        projection.parameters.matrix.set([[0.5]])
    for index, node in enumerate(composition.nodes):
        node.name = f"observed component {index}"
    batched_node_op(drift.name, likelihood_contract=LikelihoodEffectContract())(_csi_drift_rate)
    try:
        observations = ObservationSpec((
            ObservationField(timing.output_port, "counting", role="event_time"),
            ObservationField(choice.output_port, "counting"),
        ))
        mass = BatchedCompositionCompiler.compile_empirical_mass(composition, observations, backend=batched_backend, max_steps=128)
        plan = mass.observation_plan
        data = plan.sampler.path_plan.history_plan.simulate_reference(inputs, seed=17).observations[0]
        sampled = plan.sample(inputs, data, num_estimates=33, seed=29)
        reference = plan.simulate_reference(inputs, data, num_estimates=33, seed=29)
        np.testing.assert_allclose(sampled.values, reference.values, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(sampled.event_counts, reference.event_counts)
        score = mass.score(inputs, data, num_estimates=33, seed=29)
        manual = ((sampled.event_counts == sampled.history.event_counts[..., None])
                  & (sampled.values[..., 1] == data[None, :, None, 1])).sum(axis=-1)
        np.testing.assert_array_equal(score.successes, manual)
        histogram = BatchedCompositionCompiler.compile_histogram_score(
            composition, observations, backend=batched_backend, max_steps=128,
            categorical_dims=[1], bins=17, smoothing_sigma=.5, pseudocount=.1,
        )
        fused = histogram.score(inputs, data, num_estimates=33, seed=29)
        histogram_reference = histogram.score(inputs, data, num_estimates=33, seed=29, reference=True)
        np.testing.assert_array_equal(fused.bin_counts, histogram_reference.bin_counts)
        window = histogram.score(inputs, data, num_estimates=33, seed=29, execution="window")
        np.testing.assert_array_equal(window.bin_counts, histogram_reference.bin_counts)
        np.testing.assert_array_equal(window.log_likelihood, histogram_reference.log_likelihood)
    finally:
        unregister_batched_instance_op(drift.name)
