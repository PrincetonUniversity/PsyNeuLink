"""Continuous Wiener provider: independent series, support, derivatives, GPU refinement."""

from dataclasses import replace
import json
import math

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import quad
import torch

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompositionCompiler as Compiler, LikelihoodPlanningError,
    ObservationField, ObservationSpec,
)
from psyneulink.core.batched import registry, specs
from psyneulink.core.batched.wiener import (
    _small_time_log_density, _large_time_log_density, wiener_log_density,
)


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _t(value):
    return torch.as_tensor(value, dtype=torch.float64)


def _model(**kwargs):
    parameters = dict(rate=1.2, threshold=.4, noise=.5, initializer=.06,
                      non_decision_time=.2, time_step_size=.01)
    parameters.update(kwargs)
    decision = pnl.DDM(function=pnl.DriftDiffusionIntegrator(**parameters),
                       output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME])
    composition = pnl.Composition(pathways=decision)
    observations = ObservationSpec((ObservationField(decision.output_ports[0], "counting"),
                                    ObservationField(decision.output_ports[1], "lebesgue", role="event_time")))
    return composition, decision, observations


def _compile(**kwargs):
    c, node, obs = _model(**kwargs)
    return Compiler.compile_likelihood(c, obs, process="continuous_time"), node


def _mp_log_density(t, drift, a, sigma, start, choice):
    """Independent unpaired image/spectral formulas, at 90 decimal digits."""
    with mp.workdps(90):
        t, drift, a, sigma, start = map(mp.mpf, (t, drift, a, sigma, start))
        width = 2 * a
        w = (a - start if choice else a + start) / width
        rho = (-drift if choice else drift) * width / sigma**2
        tau = t * sigma**2 / width**2
        if tau <= mp.mpf('.25'):
            total = mp.fsum((w + 2 * k) * mp.exp(-(w + 2 * k)**2 / (2 * tau)) for k in range(-24, 25))
            log_standard = mp.log(total) - mp.log(2 * mp.pi) / 2 - mp.log(tau) * mp.mpf('1.5')
        else:
            total = mp.fsum(k * mp.sin(k * mp.pi * w) * mp.exp(-k * k * mp.pi**2 * tau / 2) for k in range(1, 100))
            log_standard = mp.log(mp.pi * total)
        return float(mp.log(sigma**2 / width**2) - rho * w - rho * rho * tau / 2 + log_standard)


@pytest.mark.parametrize("start", [-.5 + 1e-10, -.49, -.2, 0., .3, .49, .5 - 1e-10])
def test_density_matches_high_precision_reference(start):
    times = [1e-5, .001, .05, .199999999, .2, .200000001, .5, 2., 10., 100.]
    for drift in (-3., 0., .7):
        for choice in (0., 1.):
            actual = wiener_log_density(_t(times), _t(drift), _t(.5), _t(1.), _t(start), _t(choice))
            expected = [_mp_log_density(t, drift, .5, 1., start, choice) for t in times]
            np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=2e-9)


def test_series_switch_values_and_derivatives_agree():
    tau = _t([.2, .2, .2, .2]).requires_grad_()
    w = _t([.00001, .3, .5, .99999]).requires_grad_()
    short, long = _small_time_log_density(tau, w), _large_time_log_density(tau, w)
    torch.testing.assert_close(short, long, rtol=1e-12, atol=1e-12)
    ga = torch.autograd.grad(short.sum(), (tau, w))
    gb = torch.autograd.grad(long.sum(), (tau, w))
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("drift", [-1e6, 1e6])
def test_large_drift_peak_avoids_log_cancellation(drift):
    choice = float(drift > 0)
    args = (5e-7, drift, .5, .7, 0., choice)
    actual = wiener_log_density(*map(_t, args)).item()
    assert actual == pytest.approx(_mp_log_density(*args), abs=1e-12)


@pytest.mark.parametrize("drift,start", [(0., 0.), (.3, .08), (-.4, -.1)])
def test_normalization_choice_probability_and_mean_time(drift, start):
    a, sigma = .4, .5
    probability, moment = [], []
    for choice in (0., 1.):
        def density(t):
            return wiener_log_density(_t(t), _t(drift), _t(a), _t(sigma), _t(start), _t(choice)).exp().item()
        probability.append(quad(density, 0, np.inf, epsabs=2e-10)[0])
        moment.append(quad(lambda t: t * density(t), 0, np.inf, epsabs=2e-10)[0])
    upper = (start + a) / (2 * a) if drift == 0 else np.expm1(-2 * drift * (start + a) / sigma**2) / np.expm1(-4 * drift * a / sigma**2)
    mean = (a * a - start * start) / sigma**2 if drift == 0 else (a * (2 * upper - 1) - start) / drift
    np.testing.assert_allclose(probability, [1 - upper, upper], atol=3e-9)
    assert sum(moment) == pytest.approx(mean, abs=3e-9)


@pytest.mark.parametrize("initializer", [0., .06])
def test_compiled_gradient_fd_gradcheck_and_dt_invariance(initializer):
    plan, node = _compile(initializer=initializer)
    assert plan.description.method == "analytic"
    assert plan.description.process == "continuous_time"
    json.dumps(plan.explain())
    inputs = {node: [.2, -.1, .35, .15]}
    data = [[1., .21], [0., .35], [1., .8], [0., 1.9]]
    result = plan.value_and_grad(inputs, data)
    defaults = dict(plan.evaluator.ir.param_defaults)
    for name in plan.evaluator.active_parameter_names:
        plus, minus = defaults.copy(), defaults.copy()
        plus[name] += 1e-6
        minus[name] -= 1e-6
        fd = (plan.score(inputs, data, plus).log_likelihood - plan.score(inputs, data, minus).log_likelihood) / 2e-6
        i = result.parameter_names.index(name)
        np.testing.assert_allclose(result.gradient[:, i], fd, rtol=1e-6, atol=1e-6)
    active_indices = [result.parameter_names.index(name) for name in plan.evaluator.active_parameter_names]
    full = _t(list(defaults.values()))
    active = full[active_indices].clone().requires_grad_()

    def log_prob(x):
        p = full.index_put((_t(active_indices).to(torch.int64),), x)
        return plan.evaluator.log_prob(inputs, data, p)
    assert torch.autograd.gradcheck(log_prob, (active,))
    different_dt = plan.score(inputs, data, [{"time_step_size": .001}, {"time_step_size": .1}])
    np.testing.assert_array_equal(different_dt.log_factors[0], different_dt.log_factors[1])
    assert result.gradient[0, result.parameter_names.index("time_step_size")] == 0


def test_reordered_fields_masks_candidate_axes_and_support():
    c, node, obs = _model()
    first = Compiler.compile_likelihood(c, obs, process="continuous_time")
    reverse = Compiler.compile_likelihood(c, replace(obs, fields=obs.fields[::-1]), process="continuous_time")
    data = np.array([[0., .1], [1., .7], [0., 1.2]])
    inputs = {node: [.1, .3, -.2]}
    rows = [{}, {"rate": .8}]
    result = first.score(inputs, data, rows)
    assert np.isneginf(result.log_factors[:, 0]).all()
    assert np.isneginf(result.log_likelihood).all()
    np.testing.assert_array_equal(result.log_factors, reverse.score(inputs, data[:, ::-1], rows).log_factors)
    with pytest.raises(LikelihoodPlanningError, match="gradient is undefined"):
        first.value_and_grad(inputs, data)
    selected = first.value_and_grad(inputs, data, rows, include_mask=[False, True, True])
    np.testing.assert_allclose(selected.log_likelihood, result.log_factors[:, 1:].sum(-1))
    assert np.isfinite(selected.gradient).all()
    empty = first.value_and_grad(inputs, data, include_mask=[False] * 3)
    np.testing.assert_array_equal(empty.gradient, 0.)
    # Even if no times have support, the likelihood remains -inf, not NaN.
    all_outside = first.score(inputs, data, {"non_decision_time": 2.})
    assert np.isneginf(all_outside.log_factors).all()


@pytest.mark.parametrize("row", [{"offset": .001}, {"threshold_collapse": -.001},
                                  {"noise": 0.}, {"noise": -.1}, {"threshold": 0.},
                                  {"starting_value": .4}, {"starting_value": -.4},
                                  {"non_decision_time": -.1}, {"time_step_size": 0.}])
def test_invalid_proposals_do_not_change_model_or_add_floors(row):
    plan, node = _compile()
    with pytest.raises(ValueError):
        plan.score({node: [.1]}, [[1., .7]], row)


def test_source_defaults_and_unregistered_models_fail_closed():
    c, _, obs = _model(offset=.01)
    with pytest.raises(LikelihoodPlanningError, match="zero source offset"):
        Compiler.compile_likelihood(c, obs, process="continuous_time")
    c, node, obs = _model()
    for kwargs in ({"method": "analytic"}, {"process": "ideal_real"},
                   {"process": "continuous_time", "method": "sampling"},
                   {"process": "continuous_time", "max_steps": 10}):
        with pytest.raises(LikelihoodPlanningError):
            Compiler.compile_likelihood(c, obs, **kwargs)
    node.reset_stateful_function_when = pnl.Never()
    with pytest.raises(LikelihoodPlanningError):
        Compiler.compile_likelihood(c, obs, process="continuous_time")


@pytest.mark.parametrize("change", [dict(measure="counting"), dict(recording="noisy"),
                                    dict(availability="may_be_missing"), dict(history_timing="ceil_fp32_8ulp")])
def test_rt_observation_semantics_are_explicit(change):
    c, _, obs = _model()
    obs = replace(obs, fields=(obs.fields[0], replace(obs.fields[1], **change)))
    with pytest.raises(LikelihoodPlanningError):
        Compiler.compile_likelihood(c, obs, process="continuous_time")


def test_frozen_contract_and_backend_independence(monkeypatch):
    c, node, obs = _model()
    monkeypatch.setattr(registry, "_backend_availability", lambda backend: (False, []))
    plan = Compiler.compile_likelihood(c, obs, process="continuous_time")
    reference = plan.score({node: [.2]}, [[1., .5]]).log_likelihood.copy()
    spec = specs.mechanism_spec_for(node)
    stripped = replace(spec, likelihood_contract=replace(spec.likelihood_contract, wiener_readout=None))
    monkeypatch.setitem(specs._MECHANISM_SPECS, type(node), stripped)
    monkeypatch.setitem(specs._SPECS_BY_KEY, spec.key, stripped)
    np.testing.assert_array_equal(plan.score({node: [.2]}, [[1., .5]]).log_likelihood, reference)
    with pytest.raises(LikelihoodPlanningError, match="no registered"):
        Compiler.compile_likelihood(c, obs, process="continuous_time")
    forged = replace(plan.evaluator, witness=replace(plan.evaluator.witness, choice_column=1))
    with pytest.raises(LikelihoodPlanningError, match="witness"):
        forged.score({node: [.2]}, [[1., .5]])


def test_wiener_contract_registration_checks_required_bindings():
    c, node, obs = _model()
    Compiler.compile_likelihood(c, obs, process="continuous_time")
    original = specs.mechanism_spec_for(node)
    contract = original.likelihood_contract
    bad_parameter = replace(contract.wiener_readout, rate_parameter="missing")
    bad_clock = replace(contract.event_readout, execution_rule=None)
    for invalid in (replace(contract, wiener_readout=bad_parameter), replace(contract, event_readout=bad_clock)):
        with pytest.raises(specs.BatchedOpSpecError, match="Wiener interpretation"):
            specs.register_batched_op(replace(original, likelihood_contract=invalid))
    assert specs.mechanism_spec_for(node) is original


def test_against_existing_csi_pde_reference():
    from test_csi_direct_likelihood import MovingBoundaryDDMSolver

    solver = MovingBoundaryDDMSolver(time_step=.001, spatial_points=65, noise=.1)
    result = solver.solve_observation_batch(
        drift=torch.full((2, 403), .03, dtype=torch.float64), threshold=_t([.12, .12]),
        collapse_rate=_t([0., 0.]), interval_low=_t([.398, .398]),
        interval_high=_t([.402, .402]), choice=_t([1., 0.]),
    )
    expected = [quad(lambda t: wiener_log_density(_t(t), _t(.03), _t(.12), _t(.1), _t(0.), _t(c)).exp().item(), .398, .402)[0]
                for c in (1., 0.)]
    np.testing.assert_allclose(result.probability, expected, rtol=.002)


@pytest.mark.triton_gpu
def test_gpu_endpoint_refinement_toward_continuous_density():
    c, node, obs = _model(rate=1., initializer=0.)
    direct = Compiler.compile_likelihood(c, obs, process="continuous_time")
    inputs = {node: [.25]}
    direct_scores = direct.score(inputs, [[1., .6]], {"time_step_size": .001})
    assert np.isfinite(direct_scores.log_likelihood).all()
    # Endpoint-tested simulation is only a convergence check, not exact equality.
    simulation = Compiler.compile(c, outputs=obs.output_ports, backend="triton", max_steps=16000)
    estimates = 100000
    errors = []
    expected = quad(lambda t: wiener_log_density(_t(t), _t(.25), _t(.4), _t(.5), _t(0.), _t(1.)).exp().item(), 0, .4)[0]
    upper = 1 / (1 + math.exp(-2 * .25 * .4 / .5**2))
    mean_time = .4 * math.tanh(.25 * .4 / .5**2) / .25
    for dt in (.01, .001):
        result = simulation.run(inputs, [{"time_step_size": dt}], estimates, seed=219, strict_truncation=True)
        draws = np.asarray(result.values).reshape(estimates, 2)
        joint = np.mean((draws[:, 0] == 1) & (draws[:, 1] <= .6 + 1e-7))
        errors.append(abs(joint - expected))
        print(dict(dt=dt, estimates=estimates, joint_by_rt=joint, continuous_joint=expected,
                   upper_probability=float(np.mean(draws[:, 0])), continuous_upper=upper,
                   mean_decision_time=float(np.mean(draws[:, 1] - .2)), continuous_mean=mean_time))
        if dt == .001:
            assert abs(np.mean(draws[:, 0]) - upper) < .015
            assert abs(np.mean(draws[:, 1] - .2) - mean_time) < .035
    assert errors[1] < errors[0]
    assert errors[1] < .015
