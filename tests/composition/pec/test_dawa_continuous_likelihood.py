"""Continuous absorption, moving-domain, and sequential-gradient contracts."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch


DIRECTORY = Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile/dawa"
sys.path.insert(0, str(DIRECTORY))
from dawa_likelihood import ContinuousConfig, ContinuousResponseSolver, continuous_sequence_likelihood  # noqa: E402
from dawa_likelihood.continuous_model import advance_control, continuous_path  # noqa: E402
from dawa_likelihood.continuous_solver import ContinuousDistribution  # noqa: E402
from dawa_likelihood.continuous_validation import gradient_check, independent_race_cdf, ode_reference_check  # noqa: E402


pytestmark = [pytest.mark.composition]


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_coupled_ode_converges_to_independent_adaptive_reference():
    records = ode_reference_check()
    assert records[-1]["maximum_state_error"] < 3.e-9


def test_moving_boundary_first_passage_matches_analytic_race():
    # A linearly receding absorbing boundary is equivalent to a fixed boundary
    # and reduced physical drift. This tests the mesh-velocity term and flux.
    t = torch.arange(401, dtype=torch.float64) * .001
    reference = independent_race_cdf(t.numpy())[1:]
    errors = []
    with torch.no_grad():
        for n in (33, 65):
            solver = ContinuousResponseSolver(ContinuousConfig(points=n, noise=.2, lower_bound=-.6, leak=0., competition=0.))
            d = solver.solve_coefficients(t.new_tensor([.4, .3]).expand(len(t), 2), torch.ones_like(t), t.new_zeros(()),
                                          .15 + .05 * t, t * 0 + .05)
            errors.append(np.abs(d.choice_mass.cumsum(0).numpy() - reference).max())
            assert float(d.mass_error) < 1.e-12
            assert float(d.minimum_mass) >= 0.
            assert float(d.lower_loss) < 1.e-6
            assert d.maximum_cfl <= .8
    assert errors[1] < .002
    assert errors[1] < errors[0] * .5


def test_symmetric_race_and_lower_escape_are_conservative():
    t = torch.arange(101, dtype=torch.float64) * .001
    solver = ContinuousResponseSolver(ContinuousConfig(points=33, noise=.2, lower_bound=-.02, leak=0., competition=0.))
    d = solver.solve_coefficients(t.new_zeros((len(t), 2)), torch.ones_like(t), t.new_zeros(()),
                                  t * 0 + .05, t * 0)
    torch.testing.assert_close(d.choice_mass[:, 0], d.choice_mass[:, 1], atol=1.e-15, rtol=1.e-13)
    assert float(d.lower_loss) > .5
    assert float(d.minimum_mass) >= 0.
    torch.testing.assert_close(d.choice_mass.sum() + d.survival + d.lower_loss, t.new_tensor(1.))
    assert float(d.mass_error) < 1.e-12


def test_rt_intervals_integrate_flux_and_differentiate_continuous_shift():
    m = torch.tensor([[.1, .2], [.3, .15], [.05, .1]], dtype=torch.float64)
    d = ContinuousDistribution(m, m.new_tensor(.1), m.new_zeros(()), m.new_zeros(()), m.new_zeros(()), .01, 1, .5)
    shift = m.new_tensor(.001, requires_grad=True)
    probability = d.interval_probability(0, .004 + shift, .016 + shift)
    assert float(probability.detach()) == pytest.approx(.15 * .5 + .30375 * .7)
    assert float(torch.autograd.grad(probability, shift)[0]) == pytest.approx((.295 - .1) / .01)
    assert float(d.interval_probability(0, 0., .03)) == pytest.approx(.45)
    narrow = d.interval_probability(0, .010 + shift, .013 + shift)
    assert float(torch.autograd.grad(narrow, shift)[0]) == pytest.approx(-.75)


def test_all_seven_gradients_and_checkpoint_equivalence():
    result = gradient_check()
    assert result["max_scaled_error"] < 2.e-7
    assert min(abs(v) for v in result["autograd"]) > 1.e-5


def test_excluded_history_uses_candidate_ndt_and_resets_other_states():
    p = torch.tensor([[.18, .193, -.43, 11., .75, 1.3, 5.2]] * 2, dtype=torch.float64, requires_grad=True)

    def run(vector):
        return continuous_sequence_likelihood(vector, [[1, 0]] * 2, [[0, 1, 0, 1], [1, 0, 0, 1]], [1, 0],
                                              [.3437, .4117], include=[False, True],
                                              config=ContinuousConfig(points=25, time_step=.002))

    result = run(p)
    gradient = torch.autograd.grad(result.log_likelihood, p)[0]
    assert result.distributions[0] is None
    assert bool(torch.isnan(result.probability[0]))
    assert abs(float(gradient[0, 1])) > 1.e-5  # NDT affects next trial via time spent in control.
    assert abs(float(gradient[0, 3])) > 1.e-5
    torch.testing.assert_close(gradient[0, [0, 2, 4, 5, 6]], p.new_zeros(5), atol=0., rtol=0.)
    with torch.no_grad():
        plus, minus = p.clone(), p.clone()
        plus[0, 1] += 1.e-6
        minus[0, 1] -= 1.e-6
        finite = (run(plus).log_likelihood - run(minus).log_likelihood) / 2.e-6
        torch.testing.assert_close(gradient[0, 1], finite, rtol=1.e-6, atol=1.e-6)


def test_control_replay_agrees_with_full_network_and_resets():
    p = torch.tensor([.18, .193, -.43, 11., .75, 1.3, 5.2], dtype=torch.float64)
    with torch.no_grad():
        first = continuous_path(p, [1, 0], [0, 1, 0, 1], steps=100)
        carried = advance_control(p.new_zeros(2), p, [1, 0], p.new_tensor(.1))
        torch.testing.assert_close(carried, first.states[-1, :2], atol=1.e-9, rtol=1.e-8)
        second = continuous_path(p, [0, 1], [1, 0, 1, 0], steps=10, control=carried)
    torch.testing.assert_close(second.states[0, :2], carried)
    torch.testing.assert_close(second.states[0, 2:], p.new_zeros(8))
    torch.testing.assert_close(second.gain[0], p[6])


def test_invalid_observations_and_solver_settings_are_rejected():
    p = torch.tensor([.18, .193, -.43, 11., .75, 1.3, 5.2], dtype=torch.float64)
    with pytest.raises(FloatingPointError, match="nondecision"):
        continuous_sequence_likelihood(p, [[1, 0]], [[0, 1, 0, 1]], [1], [.18])
    with pytest.raises(ValueError, match="Choices"):
        continuous_sequence_likelihood(p, [[1, 0]], [[0, 1, 0, 1]], [1.5], [.35])
    with pytest.raises(ValueError, match="at least one"):
        continuous_sequence_likelihood(p, [[1, 0]], [[0, 1, 0, 1]], [1], [.35], include=[False])
    for kwargs in ({"cfl": 1.1}, {"lower_bound": 0.}, {"noise": 0.}, {"time_step": float("nan")}):
        with pytest.raises(ValueError):
            ContinuousConfig(**kwargs)
    solver = ContinuousResponseSolver(ContinuousConfig(points=8))
    with pytest.raises(FloatingPointError, match="boundary above"):
        solver.solve_coefficients(p.new_zeros((2, 2)), p.new_ones(2), p[2], p.new_zeros(2), p.new_zeros(2))
