"""Registered analytic rules, semantic guards, covariance and gradient oracles."""

from dataclasses import replace
import ast
import json

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm
import torch

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompositionCompiler as Compiler, BatchedTrialParameter,
    GaussianReadout, LikelihoodEffectContract, LikelihoodPlanningError,
    ObservationField, ObservationSpec,
)
from psyneulink.core.batched import registry, specs


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _model():
    normal = pnl.ProcessingMechanism(function=pnl.NormalDist(mean=.3, standard_deviation=.7))
    affine = pnl.ProcessingMechanism(function=pnl.Linear(slope=-1.8, intercept=.4, scale=1.2, offset=-.1))
    composition = pnl.Composition(pathways=[normal, affine])
    return composition, normal, affine


def _compile(composition, node, **kwargs):
    return Compiler.compile_likelihood(composition, ObservationSpec((ObservationField(node.output_port, "lebesgue"),)),
                                       process="ideal_real", **kwargs)


def test_automatic_selection_and_finite_difference_all_parameters():
    c, normal, affine = _model()
    plan = _compile(c, affine)
    assert plan.description.method == "analytic"
    assert plan.description.backend == "torch_cpu"
    json.dumps(plan.explain())
    y = np.array([-3., -.1, .76, 2.4])
    inputs = {normal: np.zeros(len(y))}
    result = plan.value_and_grad(inputs, y)
    gain = -1.8 * 1.2
    mean = gain * .3 + 1.2 * .4 - .1
    sd = abs(gain) * .7
    np.testing.assert_allclose(result.log_factors[0], norm.logpdf(y, mean, sd), atol=1e-13)
    baseline = dict(plan.evaluator.ir.param_defaults)
    for j, name in enumerate(result.parameter_names):
        h = 1e-5
        plus, minus = baseline.copy(), baseline.copy()
        plus[name] += h
        minus[name] -= h
        finite_difference = (plan.score(inputs, y, plus).log_likelihood - plan.score(inputs, y, minus).log_likelihood) / (2 * h)
        np.testing.assert_allclose(result.gradient[:, j], finite_difference, rtol=2e-8, atol=2e-8)
    tensor = torch.tensor(list(baseline.values()), dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda p: plan.evaluator.log_prob(inputs, y, p), (tensor,))
    # Distinct nearby candidates must not be quantized through FP32 packing.
    a = {f"{normal.name}.mean": .300000001}
    b = {f"{normal.name}.mean": .300000002}
    scores = plan.score(inputs, y, [a, b]).log_likelihood
    assert scores[0] != scores[1]


def test_density_normalization_and_change_of_variables():
    c, normal, affine = _model()
    plan = _compile(c, affine)
    p = plan.evaluator
    # Independent quadrature checks normalization including a negative Jacobian.
    mass, _ = quad(lambda y: np.exp(plan.score({normal: [0.]}, [y]).log_likelihood[0]), -15, 15)
    assert mass == pytest.approx(1., abs=1e-12)
    source_plan = _compile(c, normal)
    x = np.array([-2., .3, 1.1])
    gain, offset = -1.8 * 1.2, 1.2 * .4 - .1
    np.testing.assert_allclose(
        p.score({normal: [0.] * len(x)}, gain * x + offset).log_factors,
        source_plan.score({normal: [0.] * len(x)}, x).log_factors - np.log(abs(gain)), atol=1e-13,
    )


def _reconvergent_model():
    z = pnl.ProcessingMechanism(function=pnl.NormalDist(mean=.2, standard_deviation=.6))
    w = pnl.ProcessingMechanism(function=pnl.NormalDist(mean=-.3, standard_deviation=.8))
    left = pnl.ProcessingMechanism(function=pnl.Linear(slope=2.))
    right = pnl.ProcessingMechanism(function=pnl.Linear(slope=-.5))
    out = pnl.ProcessingMechanism()
    c = pnl.Composition(pathways=[[z, left, out], [z, right, out], [w, out]])
    return c, z, w, right, out


def test_reconvergent_covariance_and_independent_roots():
    c, z, w, right, out = _reconvergent_model()
    plan = _compile(c, out)
    from psyneulink.core.batched.backend.triton.graph_emit import triton_graph_kernel_source

    # The general multi-mechanism launch tier must emit valid source even
    # though these stochastic mechanisms have no retained final state.
    ast.parse(triton_graph_kernel_source(plan.evaluator.kernel))
    y = np.array([-.9, .1, 1.7])
    expected_sd = np.sqrt((1.5 * .6) ** 2 + .8 ** 2)
    np.testing.assert_allclose(plan.score({z: np.zeros(3), w: np.zeros(3)}, y).log_factors[0],
                               norm.logpdf(y, 1.5 * .2 - .3, expected_sd), atol=1e-13)
    # Cancellation of a shared draw must be singular, not two independent variances.
    cancellation = {f"{right.name}.slope": -2., f"{w.name}.standard_deviation": 0.}
    with pytest.raises(LikelihoodPlanningError, match="variance is zero"):
        plan.score({z: np.zeros(3), w: np.zeros(3)}, y, cancellation)


def test_conditioned_input_mask_and_candidate_axes():
    c, normal, out = _model()
    predictor = pnl.ProcessingMechanism(function=pnl.Linear(slope=.4))
    c.add_linear_processing_pathway([predictor, out])
    plan = _compile(c, out)
    x, y = np.array([-.4, .8, 1.2]), np.array([.4, -.6, .2])
    inputs = {normal: [0., 0., 0.], predictor: x}
    result = plan.value_and_grad(inputs, y, [{}, {f"{normal.name}.mean": .8}], include_mask=[True, False, True])
    assert result.log_factors.shape == (2, 3)
    np.testing.assert_allclose(result.log_likelihood, result.log_factors[:, [0, 2]].sum(-1))
    for i, mu in enumerate([.3, .8]):
        expected_mean = 1.2 * (-1.8 * (mu + .4 * x) + .4) - .1
        np.testing.assert_allclose(result.log_factors[i], norm.logpdf(y, expected_mean, 1.8 * 1.2 * .7))
    empty = plan.value_and_grad(inputs, y, include_mask=[False] * 3)
    np.testing.assert_array_equal(empty.log_likelihood, 0.)
    np.testing.assert_array_equal(empty.gradient, 0.)
    with pytest.raises(ValueError, match="boolean"):
        plan.score(inputs, y, include_mask=[1, 0, 1])
    with pytest.raises(ValueError, match="trial-varying"):
        plan.score(inputs, y, {f"{normal.name}.mean": BatchedTrialParameter([0., 1., 2.])})


@pytest.mark.parametrize("options,code", [
    ({"method": "analytic"}, "likelihood.process_mismatch"),
    ({}, "likelihood.estimator_required"),
    ({"process": "continuous_time"}, "wiener.structure"),
    ({"method": "numerical"}, "likelihood.numerical_not_registered"),
    ({"process": "ideal_real", "backend": "triton"}, "likelihood.backend_unsupported"),
    ({"process": "ideal_real", "method": "sampling"}, "likelihood.target_estimator_mismatch"),
])
def test_no_silent_target_or_estimator_fallback(options, code):
    c, _, out = _model()
    with pytest.raises(LikelihoodPlanningError) as exc:
        Compiler.compile_likelihood(c, ObservationSpec((ObservationField(out.output_port, "lebesgue"),)), **options)
    assert exc.value.code == code


@pytest.mark.parametrize("options", [
    {"measure": "counting"}, {"recording": "noisy"}, {"availability": "may_be_missing"},
    {"score": False}, {"role": "event_time"},
])
def test_observation_operators_are_not_inferred(options):
    c, _, out = _model()
    obs = ObservationSpec((ObservationField(out.output_port, **{"measure": "lebesgue", **options}),))
    with pytest.raises(LikelihoodPlanningError):
        Compiler.compile_likelihood(c, obs, process="ideal_real")


def test_snapshot_survives_live_and_registry_mutation(monkeypatch):
    c, normal, out = _model()
    plan = _compile(c, out)
    original = plan.score({normal: [0.]}, [.3]).log_likelihood.copy()
    normal.function.parameters.mean.set(100.)
    spec = specs.mechanism_spec_for(normal)
    pair = (spec.mechanism_class, spec.function_class)
    monkeypatch.setitem(specs._FUNCTION_MECHANISM_SPECS, pair, replace(spec, likelihood_contract=None))
    monkeypatch.setitem(specs._SPECS_BY_KEY, spec.key, replace(spec, likelihood_contract=None))
    np.testing.assert_array_equal(plan.score({normal: [0.]}, [.3]).log_likelihood, original)
    with pytest.raises(LikelihoodPlanningError):
        _compile(c, out)
    forged = replace(plan.evaluator, witness=replace(plan.evaluator.witness, output_component_id=-1))
    with pytest.raises(LikelihoodPlanningError, match="frozen source"):
        forged.score({normal: [0.]}, [.3])


def test_analytic_does_not_require_gpu_or_triton_runtime(monkeypatch):
    c, normal, out = _model()
    monkeypatch.setattr(registry, "_backend_availability", lambda backend: (False, []))
    plan = _compile(c, out)
    assert np.isfinite(plan.score({normal: [0.]}, [.2]).log_likelihood).all()


def test_unsupported_rules_and_singular_proposals():
    c, normal, out = _model()
    plan = _compile(c, out)
    for row in ({f"{normal.name}.standard_deviation": 0.}, {f"{out.name}.slope": 0.}):
        with pytest.raises(LikelihoodPlanningError, match="variance is zero"):
            plan.score({normal: [0.]}, [.2], row)
    with pytest.raises(ValueError):
        plan.score({normal: [0.]}, [.2], {f"{normal.name}.standard_deviation": -1.})
    nonlinear = pnl.ProcessingMechanism(function=pnl.Logistic())
    c.add_linear_processing_pathway([out, nonlinear])
    with pytest.raises(LikelihoodPlanningError, match="neither"):
        _compile(c, nonlinear)
    obs = ObservationSpec((ObservationField(out.output_port, "lebesgue"), ObservationField(normal.output_port, "lebesgue")))
    with pytest.raises(LikelihoodPlanningError, match="exactly one"):
        Compiler.compile_likelihood(c, obs, process="ideal_real")


def test_reject_dynamic_schedule_and_latent_initial_state():
    c, normal, out = _model()
    c.scheduler.add_condition(out, pnl.EveryNCalls(normal, 2))
    with pytest.raises(LikelihoodPlanningError):
        _compile(c, out)
    c, _, out = _model()
    obs = ObservationSpec((ObservationField(out.output_port, "lebesgue"),), initial_state="latent")
    with pytest.raises(LikelihoodPlanningError, match="Independent trials"):
        Compiler.compile_likelihood(c, obs, process="ideal_real")


def test_registration_does_not_shadow_other_processing_functions():
    c, normal, out = _model()
    assert specs.mechanism_spec_for(normal).likelihood_contract.gaussian_readout is not None
    assert specs.mechanism_spec_for(out) is None
    assert Compiler.compile(c).capability_report.can_execute
    with pytest.raises(ValueError):
        LikelihoodEffectContract(gaussian_readout=GaussianReadout(None, "mean", "sd"))
    with pytest.raises(ValueError):
        specs.register_batched_op(specs.function_spec_for(out.function), function_specific=True)


def test_original_pnl_forward_distribution():
    c, normal, out = _model()
    # Python PNL execution is independent of both our analytic rules and Triton
    # emission. Seeds need not give paired draws across the two RNG algorithms.
    c.run(inputs={normal: [[0.]]}, num_trials=1024)
    draws = np.asarray(c.results).reshape(-1)
    mean, sd = 1.2 * (-1.8 * .3 + .4) - .1, 1.2 * 1.8 * .7
    assert abs(draws.mean() - mean) < 6 * sd / np.sqrt(len(draws))
    assert abs(draws.std() - sd) < 6 * sd / np.sqrt(2 * len(draws))


@pytest.mark.parametrize("data", [[[np.nan]], [[np.inf]], [], [[1., 2.]], [[[1.]]]])
def test_invalid_observed_data_is_not_coerced_or_masked(data):
    c, normal, out = _model()
    with pytest.raises(ValueError, match="data must be"):
        _compile(c, out).score({normal: [0.]}, data)


def test_input_precision_and_foreign_binding():
    c, normal, out = _model()
    x = pnl.ProcessingMechanism()
    c.add_linear_processing_pathway([x, out])
    plan = _compile(c, out)
    a = plan.score({normal: [0.], x: [.300000001]}, [.2]).log_likelihood
    b = plan.score({normal: [0.], x: [.300000002]}, [.2]).log_likelihood
    assert a[0] != b[0]
    foreign = pnl.ProcessingMechanism()
    with pytest.raises(KeyError):
        plan.score({foreign: [0.], x: [.3]}, [.2])


@pytest.mark.triton_gpu
def test_gpu_forward_gaussian_distribution_matches_analytic():
    c, normal, out = _model()
    plan = _compile(c, out)
    simulation = Compiler.compile(c, outputs=[out.output_port], backend="triton")
    draws = simulation.run({normal: [0., 0.]}, None, num_estimates=50000, seed=127).values
    draws = np.asarray(draws).reshape(2, 50000)
    mean, sd = 1.2 * (-1.8 * .3 + .4) - .1, 1.2 * 1.8 * .7
    for trial in draws:
        assert abs(trial.mean() - mean) < 6 * sd / np.sqrt(len(trial))
        assert abs(trial.std() - sd) < 6 * sd / np.sqrt(2 * len(trial))
        # Interval probabilities test the distribution, not a KDE at selected points.
        for q in [-1., 0., 1.]:
            assert abs(np.mean(trial <= mean + q * sd) - norm.cdf(q)) < .015
    assert abs(np.corrcoef(draws)[0, 1]) < .03
    np.testing.assert_allclose(plan.score({normal: [0., 0.]}, [mean, mean + sd]).log_factors[0],
                               norm.logpdf([mean, mean + sd], mean, sd), atol=1e-13)


@pytest.mark.triton_gpu
def test_gpu_multiple_gaussian_sources_and_shared_paths():
    c, z, w, _, out = _reconvergent_model()
    inputs = {z: [0.], w: [0.]}
    simulation = Compiler.compile(c, outputs=[out.output_port], backend="triton")
    draws = np.asarray(simulation.run(inputs, None, num_estimates=50000, seed=219).values).reshape(-1)
    mean, sd = 1.5 * .2 - .3, np.sqrt((1.5 * .6) ** 2 + .8 ** 2)
    assert abs(draws.mean() - mean) < 6 * sd / np.sqrt(len(draws))
    assert abs(draws.std() - sd) < 6 * sd / np.sqrt(2 * len(draws))
    assert abs(np.mean(draws <= mean + sd) - norm.cdf(1.)) < .015
    np.testing.assert_allclose(_compile(c, out).score(inputs, [mean + sd]).log_factors[0],
                               norm.logpdf([mean + sd], mean, sd), atol=1e-13)
