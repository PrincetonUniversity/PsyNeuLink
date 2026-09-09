"""Real-device acceptance for the generic conditioned execution split."""

import numpy as np
import pytest

from psyneulink.core.batched import BatchedCompositionCompiler, LikelihoodEffectContract, batched_node_op, unregister_batched_instance_op
from test_batched_endpoints import _model, _node, _csi_drift_rate, _spec


pytestmark = [pytest.mark.batched, pytest.mark.composition, pytest.mark.triton_gpu]


@pytest.mark.parametrize("noise", [0.0, 0.15])
def test_gpu_generated_sampling_matches_coupled_reference(noise):
    composition, inputs, outputs = _model(ddm_noise=noise)
    drift = _node(composition, "Drift Rate Value")
    batched_node_op(drift.name, likelihood_contract=LikelihoodEffectContract())(_csi_drift_rate)
    try:
        history = BatchedCompositionCompiler.compile_history_replay(composition, _spec(outputs), backend="triton", max_steps=128)
        forward = history.simulate_reference(inputs, seed=12)
        data = forward.observations[0]
        replayed = history.reconstruct(inputs, data)
        np.testing.assert_allclose(replayed.end_states, forward.end_states, rtol=1e-6, atol=1e-7)
        paths = history.compile_boundary_trajectories()
        sampler = paths.compile_stochastic_sampler()
        options = dict(num_estimates=257, seed=21)
        sampled = sampler.sample(inputs, data, **options)
        reference = sampler.simulate_reference(inputs, data, **options)
        np.testing.assert_array_equal(sampled.event_counts, reference.event_counts)
        np.testing.assert_allclose(sampled.values, reference.values, rtol=1e-6, atol=1e-7)
        assert not np.any(sampled.truncated)
    finally:
        unregister_batched_instance_op(drift.name)
