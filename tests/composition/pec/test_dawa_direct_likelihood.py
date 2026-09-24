"""Numerical and gradient contracts for the research-local DAWA likelihood."""

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr
import torch


DIRECTORY = Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile"
sys.path.insert(0, str(DIRECTORY))
from dawa_likelihood import ResponseSolver, SolverConfig, sequence_likelihood, step_sequence_likelihood  # noqa: E402
from dawa_likelihood.fit import fit_projected_gradient  # noqa: E402
from dawa_likelihood.model import DT  # noqa: E402
from dawa_likelihood.validation import native_replay_error  # noqa: E402


pytestmark = [pytest.mark.composition]


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def first_step(means=(0., 0.), threshold=.5):
    p = torch.tensor(threshold, dtype=torch.float64, requires_grad=True)
    path = SimpleNamespace(inputs=torch.tensor([means], dtype=p.dtype) / DT + 4.,
                           gain=p.new_tensor([5.]), initial_activity=p.new_tensor([.5, .5]))
    return p, path


def test_first_step_symmetry_and_analytic_threshold_derivative():
    threshold, path = first_step()
    result = ResponseSolver(SolverConfig(points=33, lower_bound=-.1)).solve(path, threshold, threshold.new_zeros(()))
    torch.testing.assert_close(result.choice_step, threshold.new_tensor([[.375, .375]]), atol=1.e-14, rtol=1.e-14)
    torch.testing.assert_close(result.survival, threshold.new_tensor(.25), atol=1.e-14, rtol=1.e-14)
    expected = .5 * (1. - torch.special.ndtr(torch.logit(threshold) / (5. * .01)).square())
    actual_gradient = torch.autograd.grad(result.choice_step[0, 0], threshold)[0]
    expected_gradient = torch.autograd.grad(expected, threshold)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1.e-12, rtol=1.e-12)


def test_both_crossing_choice_rule_matches_independent_quadrature():
    threshold, path = first_step((.004, -.003))
    result = ResponseSolver(SolverConfig(points=33, lower_bound=-.1, winner_points=96)).solve(path, threshold, threshold.new_zeros(()))
    expected = []
    for a, b in ((.004, -.003), (-.003, .004)):
        value = quad(lambda x: np.exp(-.5 * ((x - a) / .01)**2) / (.01 * np.sqrt(2. * np.pi))
                     * ndtr((x - b) / .01), 0., .2, epsabs=1.e-12)[0]
        expected.append(value)
    np.testing.assert_allclose(result.choice_step.detach().numpy()[0], expected, atol=2.e-7, rtol=2.e-7)


def test_lower_domain_escape_is_reported_without_renormalizing_survivors():
    threshold, path = first_step()
    result = ResponseSolver(SolverConfig(points=33, lower_bound=-.01)).solve(path, threshold, threshold.new_zeros(()))
    expected_survival = (ndtr(0.) - ndtr(-1.))**2
    assert float(result.survival.detach()) == pytest.approx(expected_survival, abs=1.e-14)
    assert float(result.lower_loss.detach()) == pytest.approx(.25 - expected_survival, abs=1.e-14)
    assert float(result.mass_error.detach()) < 1.e-14


def short_sequence(parameters, *, config=None, include=None):
    return sequence_likelihood(parameters, [[1., 0.], [1., 0.]], [[0., 1., 0., 1.], [1., 0., 0., 1.]],
                               [1, 0], [.263, .281], [4, 7], max_steps=12,
                               config=config or SolverConfig(points=33, lower_bound=-.15, checkpoint_steps=3), include=include)


def test_all_parameter_gradients_and_checkpoint_equivalence():
    p = torch.tensor((.15, .2, -.45, 11., .65, 1.5, 5.2), dtype=torch.float64, requires_grad=True)
    result = short_sequence(p)
    gradient = torch.autograd.grad(result.log_likelihood, p)[0]
    finite = []
    with torch.no_grad():
        for i in range(7):
            h = 1.e-5 * max(1., float(abs(p[i])))
            plus, minus = p.clone(), p.clone()
            plus[i] += h
            minus[i] -= h
            finite.append(float((short_sequence(plus).log_likelihood - short_sequence(minus).log_likelihood) / (2. * h)))
    np.testing.assert_allclose(gradient.numpy(), finite, atol=2.e-5, rtol=2.e-5)
    assert bool((gradient.abs() > 1.e-8).all())
    uncheckpointed = short_sequence(p, config=SolverConfig(points=33, lower_bound=-.15, checkpoint_steps=0))
    plain = torch.autograd.grad(uncheckpointed.log_likelihood, p)[0]
    torch.testing.assert_close(plain, gradient, atol=1.e-12, rtol=1.e-12)


def test_excluded_trial_still_carries_differentiable_history():
    p = torch.tensor([[.15, .2, -.45, 11., .65, 1.5, 5.2]] * 2, dtype=torch.float64, requires_grad=True)
    result = short_sequence(p, include=[False, True])
    gradient = torch.autograd.grad(result.log_likelihood, p)[0]
    assert float(gradient[0, 3].abs()) > 1.e-6  # previous control gain affects current trial
    assert float(gradient[0, 0]) == 0.  # supplied history durations are fixed observations
    assert float(gradient[0, 1]) == 0.
    assert float(gradient[1, 1].abs()) > 1.e-6  # RT observation model supplies the NDT derivative
    torch.testing.assert_close(result.log_likelihood, result.probability[1].log())


def test_deterministic_replay_matches_native_across_resets():
    assert native_replay_error((.4, .2, -.3, 13., .5, 2., 6.), trials=2, steps=3) < 1.e-12


def test_stopping_step_likelihood_needs_no_rt_measurement_model():
    p = torch.tensor((.15, .2, -.45, 11., .65, 1.5, 5.2), dtype=torch.float64, requires_grad=True)
    result = step_sequence_likelihood(p, [[1., 0.], [1., 0.]], [[0., 1., 0., 1.], [1., 0., 0., 1.]],
                                      [1, 0], [4, 7], config=SolverConfig(points=33, lower_bound=-.15))
    gradient = torch.autograd.grad(result.log_likelihood, p)[0]
    assert float(gradient[1]) == 0.
    assert bool(torch.isfinite(gradient).all())
    torch.testing.assert_close(result.probability[0], result.distributions[0].choice_step[3, 1])
    torch.testing.assert_close(result.probability[1], result.distributions[1].choice_step[6, 0])


def test_gradient_optimizer_backtracks_on_invalid_proposals():
    def objective(x):
        if x[0] >= .6:
            raise FloatingPointError("Outside supported likelihood region")
        return -float((x[0] - .8)**2), np.array([-2. * (x[0] - .8)])

    result = fit_projected_gradient(objective, [.2], [[0., 1.]], iterations=3, step_size=1.)
    assert result["rejected_proposals"] > 0
    assert np.all(np.diff(result["accepted_log_likelihoods"]) > 0)
    assert .2 < result["parameters"][0] < .6


def test_invalid_histories_and_observations_are_rejected():
    p = torch.tensor((.15, .2, -.45, 11., .65, 1.5, 5.2), dtype=torch.float64)
    with pytest.raises(ValueError, match="integers"):
        sequence_likelihood(p, [[1, 0]], [[0, 1, 0, 1]], [1], [.263], [4.5], max_steps=12)
    with pytest.raises(ValueError, match="at least one"):
        short_sequence(p, include=[False, False])
    with pytest.raises(ValueError, match="Choices"):
        sequence_likelihood(p, [[1, 0]], [[0, 1, 0, 1]], [1.2], [.263], [4], max_steps=12,
                            config=SolverConfig(points=16, lower_bound=-.15))
    threshold, path = first_step()
    with pytest.raises(ValueError, match="lower grid"):
        ResponseSolver(replace(SolverConfig(), lower_bound=.1)).solve(path, threshold, threshold.new_zeros(()))
