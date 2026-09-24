"""Source refinement, optional native adjoints, and recovery observations."""

from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

DIRECTORY = Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile"
sys.path.insert(0, str(DIRECTORY))
from dawa_likelihood.continuous_solver import ContinuousConfig, ContinuousDistribution, ContinuousResponseSolver, _heun  # noqa: E402
from dawa_likelihood.continuous_model import continuous_path, advance_control  # noqa: E402
from dawa_likelihood.continuous_likelihood import continuous_sequence_likelihood  # noqa: E402
from dawa_likelihood.model import response_path  # noqa: E402
from dawa_likelihood.solver import ResponseSolver  # noqa: E402
from dawa_likelihood.validation import native_replay_error  # noqa: E402
from dawa_likelihood.fit import minimize_bounded_bfgs, _box_quadratic_step  # noqa: E402
from dawa_continuous_recovery import bin_probabilities, observation_counts  # noqa: E402
from dawa_continuous_recovery_report import paired_score_difference  # noqa: E402
from dawa_continuous_study import empirical_cdf  # noqa: E402

pytestmark = [pytest.mark.composition]


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    yield
    torch.set_num_threads(old)


def test_refined_source_replay_preserves_schedule_and_trial_resets():
    assert native_replay_error(time_step=.00125, trials=2, steps=4) < 1.e-12
    p = torch.tensor([.3, .2, -.45, 10., .9, 1., 5.], dtype=torch.float64)
    path = response_path(p, [1, 0], [0, 1, 0, 1], 2, time_step=.001)
    with pytest.raises(ValueError, match="10 ms"):
        ResponseSolver().solve(path, p[0], p[2])


def test_generated_equations_match_torch_states_and_parameter_and_initial_state_gradients():
    from dawa_likelihood.continuous_native import continuous_path_native, advance_control_native
    p = torch.tensor([.3, .2, -.45, 10., .9, 1., 5.], dtype=torch.float64, requires_grad=True)
    c = p.new_tensor([-.02, -.06], requires_grad=True)
    outputs, gradients = [], []
    for fn in (continuous_path, continuous_path_native):
        path = fn(p, [1, 0], [0, 1, 0, 1], steps=100, control=c)
        outputs.append(path.states)
        gradients.append(torch.autograd.grad(path.inputs.square().mean() + path.gain.mean() + path.gain_rate.mean(), (p, c)))
    torch.testing.assert_close(outputs[0], outputs[1], atol=2.e-7, rtol=2.e-7)
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, atol=1.e-6, rtol=1.e-6)
    duration = p.new_tensor(.3457, requires_grad=True)
    expected = advance_control(c, p, [1, 0], duration)
    actual = advance_control_native(c, p, [1, 0], duration)
    torch.testing.assert_close(actual, expected, atol=1.e-10, rtol=1.e-10)
    for a, b in zip(torch.autograd.grad(expected.sum(), (p, c, duration)), torch.autograd.grad(actual.sum(), (p, c, duration))):
        torch.testing.assert_close(a, b, atol=1.e-9, rtol=1.e-9)


def test_native_flux_adjoint_matches_torch_for_mass_rates_and_all_exit_categories():
    from dawa_likelihood.continuous_flux import native_flux_block
    generator = torch.Generator().manual_seed(31)
    mass = torch.rand(9, 9, dtype=torch.float64, generator=generator, requires_grad=True)
    rates = torch.rand(4, 4, 9, 9, dtype=torch.float64, generator=generator, requires_grad=True)
    weight = torch.randn(mass.shape, dtype=mass.dtype, generator=generator)
    exit_weight = torch.randn(3, 3, dtype=mass.dtype, generator=generator)
    m, fluxes = mass, []
    for k in range(3):
        flux = mass.new_zeros(3)
        for j in range(3):
            r0 = rates[k] + j / 3. * (rates[k + 1] - rates[k])
            r1 = rates[k] + (j + 1) / 3. * (rates[k + 1] - rates[k])
            m, increment = _heun(m, r0, r1, .01)
            flux = flux + increment
        fluxes.append(flux)
    flux = torch.stack(fluxes)
    native, native_flux, minimum = native_flux_block(mass, rates, 3, .03)
    torch.testing.assert_close(native, m, atol=1.e-13, rtol=1.e-13)
    torch.testing.assert_close(native_flux, flux, atol=1.e-13, rtol=1.e-13)
    assert float(minimum) >= 0.
    actual = torch.autograd.grad((native * weight).sum() + (native_flux * exit_weight).sum(), (mass, rates))
    expected = torch.autograd.grad((m * weight).sum() + (flux * exit_weight).sum(), (mass, rates))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1.e-12, rtol=1.e-12)


def test_native_coefficients_and_all_coefficient_gradients_match_reference():
    generator = torch.Generator().manual_seed(19)
    cfg = ContinuousConfig(points=11)
    values = [torch.randn(5, 2, dtype=torch.float64, generator=generator) * 3.,
              torch.tensor([.001, 2., 5., 20., 150.], dtype=torch.float64),
              torch.tensor(-.4, dtype=torch.float64),
              torch.linspace(.04, .7, 5, dtype=torch.float64),
              torch.tensor([0., -10., 5., .1, 1.], dtype=torch.float64)]
    values = [v.requires_grad_() for v in values]
    expected = ContinuousResponseSolver(cfg)._rates(*values)
    actual = ContinuousResponseSolver(replace(cfg, flux_backend="native"))._rates(*values)
    torch.testing.assert_close(actual, expected, atol=1.e-10, rtol=1.e-12)
    weights = torch.randn(expected.shape, dtype=torch.float64, generator=generator)
    gradients = [torch.autograd.grad((r * weights).sum(), values) for r in (expected, actual)]
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, atol=1.e-9, rtol=1.e-11)


def test_native_sequential_likelihood_matches_torch_and_ndt_finite_difference():
    p = torch.tensor([[.18, .193, -.43, 11., .75, 1.3, 5.2]] * 2, dtype=torch.float64, requires_grad=True)
    cfg = ContinuousConfig(points=25, time_step=.002)
    native = replace(cfg, ode_backend="generated", flux_backend="native")

    def evaluate(vector, config):
        return continuous_sequence_likelihood(vector, [[1, 0]] * 2, [[0, 1, 0, 1], [1, 0, 0, 1]],
                                              [1, 0], [.3437, .4117], include=[False, True], config=config).log_likelihood

    value = evaluate(p, native)
    gradient = torch.autograd.grad(value, p)[0]
    retained = evaluate(p, replace(native, recompute_rates=False))
    torch.testing.assert_close(value, retained, atol=1.e-12, rtol=1.e-12)
    torch.testing.assert_close(gradient, torch.autograd.grad(retained, p)[0], atol=1.e-10, rtol=1.e-10)
    reference = evaluate(p, cfg)
    torch.testing.assert_close(value, reference, atol=2.e-7, rtol=2.e-7)
    torch.testing.assert_close(gradient, torch.autograd.grad(reference, p)[0], atol=2.e-5, rtol=2.e-5)
    assert float(gradient[0, 1].abs()) > 1.e-5
    with torch.no_grad():
        plus, minus = p.clone(), p.clone()
        plus[0, 1] += 1.e-6
        minus[0, 1] -= 1.e-6
        finite = (evaluate(plus, native) - evaluate(minus, native)) / 2.e-6
        torch.testing.assert_close(gradient[0, 1], finite, atol=1.e-6, rtol=1.e-6)


def test_binned_recovery_probabilities_preserve_mass_censoring_and_shift_gradient():
    mass = torch.tensor([[.1, .2], [.3, .15], [.05, .1]], dtype=torch.float64)
    d = ContinuousDistribution(mass, mass.new_tensor(.1), mass.new_zeros(()), mass.new_zeros(()), mass.new_zeros(()), .01, 1, .5)
    ndt = mass.new_tensor(.002, requires_grad=True)
    edges = mass.new_tensor([0., .007, .019, .027])
    probabilities = bin_probabilities(d, edges, ndt)
    scalar = torch.stack([d.interval_probability(c, low - ndt, high - ndt)
                          for low, high in zip(edges[:-1], edges[1:]) for c in (0, 1)])
    torch.testing.assert_close(probabilities[:-1], scalar)
    torch.testing.assert_close(probabilities.sum(), mass.new_tensor(1.))
    a = torch.autograd.grad(probabilities[2], ndt, retain_graph=True)[0]
    b = torch.autograd.grad(scalar[2], ndt)[0]
    torch.testing.assert_close(a, b)
    samples = np.array([[0, .004], [1, .01], [0, .03], [-1, .1]])
    counts = observation_counts(samples, .002, edges.numpy())
    assert counts.sum() == 4
    assert counts[-1] == 2
    cdf = empirical_cdf(samples, [.02])
    np.testing.assert_allclose(cdf, [[.25, .25]])  # censored samples remain in denominator


def test_bfgs_backtracks_on_infeasible_steps_without_false_convergence():
    def objective(x):
        if x[0] >= .6:
            return np.inf, np.zeros(1)
        return float((x[0] - .8)**2), np.array([2. * (x[0] - .8)])

    result = minimize_bounded_bfgs(objective, [.2], iterations=10, step_limit=.5)
    assert result["rejected_proposals"] > 0
    assert .59 < result["x"][0] < .6
    assert not result["success"]
    assert np.all(np.diff(result["accepted_losses"]) < 0)
    target = np.array([.55, .35])
    matrix = np.array([[4., 1.], [1., 2.]])

    def feasible(x):
        residual = x - target
        return float(.5 * residual @ matrix @ residual), matrix @ residual

    fitted = minimize_bounded_bfgs(feasible, [.2, .8], iterations=25)
    assert fitted["success"]
    np.testing.assert_allclose(fitted["x"], target, atol=1.e-6)


def test_bfgs_resumes_curvature_and_converges_at_an_active_bound():
    matrix = np.array([[10., 3.], [3., 2.]])

    def objective(x):
        residual = x - [1.4, .2]
        return float(.5 * residual @ matrix @ residual), matrix @ residual

    saved = []
    first = minimize_bounded_bfgs(objective, [.2, .3], iterations=4, callback=saved.append)
    assert not first["success"]
    resumed = minimize_bounded_bfgs(objective, [.2, .3], iterations=100, state=saved[-1])
    full = minimize_bounded_bfgs(objective, [.2, .3], iterations=104)
    assert resumed["success"] and full["success"]
    np.testing.assert_allclose(resumed["x"], [1., .8], atol=1.e-6)
    np.testing.assert_allclose(resumed["accepted_losses"], full["accepted_losses"], atol=1.e-13)
    assert resumed["projected_gradient_norm"] < 1.e-6
    preconditioned = minimize_bounded_bfgs(objective, [.2, .3], iterations=100, initial_inverse=np.linalg.inv(matrix))
    assert preconditioned["success"]
    np.testing.assert_allclose(preconditioned["x"], [1., .8], atol=1.e-6)


def test_bfgs_quadratic_step_satisfies_constrained_stationarity():
    random = np.random.default_rng(42)
    for _ in range(30):
        a = random.normal(size=(7, 7))
        hessian = a.T @ a + .01 * np.eye(7)
        x = random.uniform(size=7)
        x[:2] = [0., 1.]
        gradient = random.normal(size=7)
        step = _box_quadratic_step(np.linalg.inv(hessian), gradient, x, .1)
        low, high = np.maximum(-x, -.1), np.minimum(1. - x, .1)
        residual = gradient + hessian @ step
        projected = np.clip(step - residual, low, high) - step
        assert np.max(np.abs(projected)) < 1.e-9
        assert gradient @ step <= 0.


def test_recovery_cli_resume_preserves_unstarted_fits(tmp_path, monkeypatch):
    import dawa_continuous_recovery as recovery

    for condition in (1, 3):
        np.savez(tmp_path / f"continuous_1_{condition}.npz", samples=np.array([[0., .1], [1., .15]] * 8))

    def cheap_probabilities(p, conditions, edges, config):
        index = torch.arange(2 * (len(edges) - 1) + 1, dtype=p.dtype)
        row = torch.softmax(-p[0] * index + p[1] * torch.cos(index), dim=0)
        return row.expand(len(conditions), -1)

    monkeypatch.setattr(recovery, "probabilities", cheap_probabilities)
    monkeypatch.setattr(recovery, "plot_report", lambda *args: None)
    output = tmp_path / "recovery.json"
    arguments = ["recovery", "--study", str(tmp_path), "--output", str(output), "--estimates", "8", "--heldout", "8",
                 "--points", "8", "--verify-points", "8", "--starts", "2", "--iterations", "2",
                 "--free", "threshold", "non_decision_time"]

    def interrupted(*args, **kwargs):
        minimize_bounded_bfgs(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(recovery, "minimize_bounded_bfgs", interrupted)
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(KeyboardInterrupt):
        recovery.main()
    pending = json.loads(output.read_text())
    assert len(pending["fits"]) == 1
    assert len(pending["requested_initials"]) == 2
    assert pending["fits"][0]["optimizer_state"]["accepted_losses"]
    monkeypatch.setattr(recovery, "minimize_bounded_bfgs", minimize_bounded_bfgs)
    monkeypatch.setattr(sys, "argv", arguments + ["--resume"])
    recovery.main()
    complete = json.loads(output.read_text())
    assert len(complete["fits"]) == 2
    assert all("refined_check" in fit for fit in complete["fits"])
    stale = {"parameter_sets": [{}, dict(complete["truth"], threshold=.5)], "lc_clock_ratio": 20.}
    (tmp_path / "results.json").write_text(json.dumps(stale))
    with pytest.raises(SystemExit):
        recovery.main()
    assert json.loads(output.read_text()) == complete


def test_paired_heldout_summary_matches_expanded_stratified_observations():
    counts = np.array([[2, 3, 0], [1, 3, 3]])
    p = np.array([[.2, .3, .5], [.1, .5, .4]])
    q = np.array([[.3, .3, .4], [.15, .45, .4]])
    mean, se = paired_score_difference(p, q, counts)
    samples = [np.repeat(np.log(a / b), c) for a, b, c in zip(p, q, counts)]
    expected_mean = np.concatenate(samples).mean()
    expected_se = np.sqrt(sum(len(s) * s.var(ddof=1) for s in samples)) / counts.sum()
    np.testing.assert_allclose([mean, se], [expected_mean, expected_se], atol=1.e-15)
