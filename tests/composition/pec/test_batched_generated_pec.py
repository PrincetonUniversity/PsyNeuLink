"""Opt-in PEC routing uses checked generic likelihood compilation, never CSI fallback."""

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    LikelihoodEffectContract, ObservationField, ObservationSpec,
    BatchedSimulationPlan, batched_node_op, unregister_batched_instance_op,
)
from test_batched_csi_coevolving_acceptance import _recovery_surface_model, _recovery_pec, _csi_drift_rate, _DRIFT_NODE_NAME


pytestmark = [pytest.mark.batched, pytest.mark.composition]


@pytest.fixture
def generated_pec(batched_backend):
    batched_node_op(_DRIFT_NODE_NAME, likelihood_contract=LikelihoodEffectContract())(_csi_drift_rate)
    try:
        composition, inputs, _, outputs = _recovery_surface_model()
        spec = ObservationSpec((ObservationField(outputs[0], "counting"),
                                ObservationField(outputs[1], "lebesgue", role="event_time", history_timing="ceil_fp32_8ulp")))
        pec = _recovery_pec(composition, inputs, outputs, backend=batched_backend,
                            include_historical_threshold_parameters=False, batched_observations=spec)
        yield pec
    finally:
        unregister_batched_instance_op(_DRIFT_NODE_NAME)


def test_pec_generated_objective_batches_arbitrary_ndt_and_never_calls_custom(generated_pec, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Generated likelihood routed through the handwritten CSI implementation")

    monkeypatch.setattr(BatchedSimulationPlan, "deterministic_history_log_likelihood", forbidden)
    opt = generated_pec.controller.function
    objective = opt._make_objective_func()
    candidates = [[10., 10., .2037], [12., 10., .2113]]
    actual = objective._batched_parameter_sets(candidates)
    assert np.all(np.isfinite(actual))
    cached = opt._batched_likelihood_plan
    np.testing.assert_array_equal(actual, [objective(*row) for row in candidates])
    assert opt._batched_likelihood_plan is cached
    plan = opt._compile_batched_plan()
    histogram = opt._compile_batched_histogram_plan(plan)
    expected = histogram.score(opt._batched_stimulus_inputs(), generated_pec._data_numpy,
                               [opt._batched_parameter_set(row) for row in candidates],
                               num_estimates=8, seed=29, execution="window")
    np.testing.assert_array_equal(actual, expected.log_likelihood)
    assert np.isfinite(objective(10., 10., .6))


def test_pec_rejects_reordered_observation_binding(generated_pec):
    from dataclasses import replace
    from psyneulink.core.components.functions.nonstateful.fitfunctions import OptimizationFunctionError

    opt = generated_pec.controller.function
    opt._make_objective_func()(10., 10., .2037)  # An existing cache must not bypass binding checks.
    spec = opt.batched_observations
    opt.batched_observations = replace(spec, fields=tuple(reversed(spec.fields)))
    with pytest.raises(OptimizationFunctionError, match="data-column order"):
        opt._make_objective_func()(10., 10., .2037)


def test_generated_mode_requires_explicit_valid_configuration():
    spec = ObservationSpec((ObservationField(object(), "counting"),))
    for extra in ({}, {"batched_backend": "triton", "conditioned_likelihood": True},
                  {"batched_backend": "triton", "deterministic_history_likelihood": True}):
        with pytest.raises(ValueError, match="exclusive"):
            pnl.PECOptimizationFunction(method="differential_evolution", batched_observations=spec, **extra)
    with pytest.raises(TypeError, match="ObservationSpec"):
        pnl.PECOptimizationFunction(method="differential_evolution", batched_backend="triton", batched_observations={})
