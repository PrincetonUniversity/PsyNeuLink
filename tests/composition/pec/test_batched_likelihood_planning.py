"""The common facade preserves existing generated CSI estimator semantics."""

import numpy as np
import pytest

from psyneulink.core.batched import (
    BatchedCompositionCompiler as Compiler, HistogramEstimatorSpec,
    LikelihoodEffectContract, LikelihoodPlanningError,
    batched_node_op, unregister_batched_instance_op,
)
from test_batched_endpoints import _model, _node, _csi_drift_rate, _spec


pytestmark = [pytest.mark.batched, pytest.mark.composition]


@pytest.mark.parametrize("estimator", [HistogramEstimatorSpec((0,), bins=13, smoothing_sigma=.5,
                                                            pseudocount=.1, categorical_cardinalities=(2,)),
                                       "empirical_mass"])
def test_common_sampling_route_matches_existing_csi_score(batched_backend, estimator):
    composition, inputs, outputs = _model(ddm_noise=.15)
    drift = _node(composition, "Drift Rate Value")
    batched_node_op(drift.name, likelihood_contract=LikelihoodEffectContract())(_csi_drift_rate)
    try:
        observations = _spec(outputs)
        plan = Compiler.compile_likelihood(composition, observations, estimator=estimator,
                                           backend=batched_backend, max_steps=128)
        assert plan.description.method == "sampling"
        assert plan.description.process == "source"
        history = plan.evaluator.observation_plan.sampler.path_plan.history_plan
        data = history.simulate_reference(inputs, seed=12).observations[0]
        if isinstance(estimator, HistogramEstimatorSpec):
            old = Compiler.compile_histogram_score(composition, observations, backend=batched_backend,
                                                    max_steps=128, categorical_dims=(0,), bins=13,
                                                    smoothing_sigma=.5, pseudocount=.1, categorical_cardinalities=(2,))
        else:
            old = Compiler.compile_empirical_mass(composition, observations, backend=batched_backend, max_steps=128)
        kwargs = dict(num_estimates=17, seed=42)
        result, reference = plan.score(inputs, data, **kwargs), old.score(inputs, data, **kwargs)
        np.testing.assert_array_equal(result.log_factors, reference.log_factors)
        np.testing.assert_array_equal(result.log_likelihood, reference.log_likelihood)
        with pytest.raises(LikelihoodPlanningError, match="no registered gradient"):
            plan.value_and_grad(inputs, data)
    finally:
        unregister_batched_instance_op(drift.name)
