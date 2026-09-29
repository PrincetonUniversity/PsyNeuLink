"""Independent finite-state references for the production particle filter.

The source simulator draws both latent transitions and categorical emissions.
Only the reference integrates those emissions analytically. Thus these tests
exercise the same simulation/observation/resampling split as a compiled model,
without sharing its state propagation or likelihood implementation.
"""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from psyneulink.core.batched.compiler import BatchedSimulationPlan
from psyneulink.core.batched.graph import COEVOLVING_GRAPH_FUSION

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.batched, pytest.mark.composition]

INITIAL = np.array([.70, .20, .10])
TRANSITION = np.array([[.92, .06, .02], [.04, .92, .04], [.06, .08, .86]])
EMISSION = np.array([[.72, .22, .055, .005], [.18, .70, .11, .01], [.05, .20, .65, .10]])
CONTAMINATION = .02
CATEGORIES = EMISSION.shape[1]


def _forward(observed, contamination=CONTAMINATION):
    """Exact matrix forward recursion, independent of Torch and filter helpers."""
    emission = (1. - contamination) * EMISSION + contamination / CATEGORIES
    prediction = INITIAL.copy()
    densities, posteriors = [], []
    for observation in observed:
        unnormalized = prediction * emission[:, observation]
        density = unnormalized.sum()
        posterior = unnormalized / density
        densities.append(density)
        posteriors.append(posterior)
        prediction = posterior @ TRANSITION
    return np.array(densities), np.array(posteriors)


def _observations(length=40):
    rng = np.random.default_rng(219)
    state = rng.choice(3, p=INITIAL)
    observations = []
    emission = (1. - CONTAMINATION) * EMISSION + CONTAMINATION / CATEGORIES
    for _ in range(length):
        observations.append(rng.choice(CATEGORIES, p=emission[state]))
        state = rng.choice(3, p=TRANSITION[state])
    # Deliberately include low-probability observations, which can substantially
    # revise the state posterior and reveal problems hidden by easy sequences.
    observations[4] = observations[15] = observations[27] = 3
    return np.asarray(observations)


class _StochasticHMM:
    backend = "triton_cpu"  # Use the real conditioning loop, without Triton.
    ir = SimpleNamespace(param_defaults={}, params=())
    kernel_ir = SimpleNamespace(fusion_kind=COEVOLVING_GRAPH_FUSION)

    def __init__(self):
        self.resampled_posteriors = []

    def run(self, inputs, parameter_sets, num_estimates, *, initial_states,
            seed, rng_trial_offset, **kwargs):
        assert len(parameter_sets) == 1
        rng = np.random.default_rng(np.random.SeedSequence([seed, rng_trial_offset, 591]))
        if initial_states is None:
            states = rng.choice(3, size=num_estimates, p=INITIAL)
        else:
            parents = initial_states.numpy().reshape(-1).astype(int)
            self.resampled_posteriors.append(np.bincount(parents, minlength=3) / num_estimates)
            states = (rng.random((num_estimates, 1)) > TRANSITION[parents].cumsum(axis=1)).sum(axis=1)
        emissions = (rng.random((num_estimates, 1)) > EMISSION[states].cumsum(axis=1)).sum(axis=1)
        return SimpleNamespace(
            values=torch.tensor(emissions, dtype=torch.float32).reshape(1, 1, 1, num_estimates, 1),
            metadata={"final_states": torch.tensor(states, dtype=torch.float32).reshape(1, 1, num_estimates, 1)},
        )


def _filter(observed, estimates, seed, include_mask=None):
    simulator = _StochasticHMM()
    # Keep the observation model fixed while changing the Monte Carlo budget.
    alpha = CONTAMINATION * estimates / (CATEGORIES * (1. - CONTAMINATION))
    score, diagnostics = BatchedSimulationPlan.conditioned_log_likelihood(
        simulator, {"unused_input": np.zeros((len(observed), 1))}, [{}], estimates,
        np.asarray(observed)[:, None], [0], pseudocount=alpha,
        categorical_cardinalities=[CATEGORIES], seed=seed, include_mask=include_mask,
        return_diagnostics=True,
    )
    assert diagnostics["observation_contamination_probability"] == pytest.approx(CONTAMINATION)
    assert not diagnostics["zero_support"].any()
    return score, diagnostics["per_trial_densities"][0, 0], np.array(simulator.resampled_posteriors)


@pytest.fixture
def one_torch_thread():
    """Small CPU filters should not repeatedly start a large worker team."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_hmm_forward_reference_matches_exhaustive_joint_enumeration():
    emission = (1. - CONTAMINATION) * EMISSION + CONTAMINATION / CATEGORIES
    all_observation_probability = 0.
    for observed in product(range(CATEGORIES), repeat=3):
        joint = 0.
        terminal_joint = np.zeros(3)
        for x0, x1, x2 in product(range(3), repeat=3):
            probability = (INITIAL[x0] * TRANSITION[x0, x1] * TRANSITION[x1, x2]
                           * emission[x0, observed[0]] * emission[x1, observed[1]] * emission[x2, observed[2]])
            joint += probability
            terminal_joint[x2] += probability
        density, posterior = _forward(observed)
        assert density.prod() == pytest.approx(joint, rel=1e-13)
        np.testing.assert_allclose(posterior[-1], terminal_joint / joint, rtol=1e-13, atol=0.)
        all_observation_probability += joint
    assert all_observation_probability == pytest.approx(1., abs=1e-13)


def test_stochastic_hmm_likelihood_and_posterior_converge_to_exact_reference(one_torch_thread):
    observed = _observations()
    exact_density, exact_posterior = _forward(observed)
    exact_log_likelihood = np.log(exact_density).sum()
    ensembles = {}
    for estimates in (512, 8192):
        results = [_filter(observed, estimates, seed) for seed in range(16)]
        ensembles[estimates] = tuple(np.array([row[column] for row in results]) for column in range(3))

    scores, densities, posteriors = ensembles[8192]
    # Test the *likelihood*, not its biased logarithm. Six replicate standard
    # errors allow finite-seed fluctuation; the precision bound ensures this
    # does not become an uninformative test with huge Monte Carlo variance.
    likelihood_ratios = np.exp(scores - exact_log_likelihood)
    standard_error = likelihood_ratios.std(ddof=1) / np.sqrt(len(scores))
    assert standard_error < .12
    assert abs(likelihood_ratios.mean() - 1.) < 6 * standard_error + .01

    for samples, exact in [(densities, exact_density), (posteriors, exact_posterior[:-1])]:
        standard_errors = samples.std(axis=0, ddof=1) / np.sqrt(len(samples))
        # Posterior estimates are ratios and can have O(1/N) bias. A small
        # absolute allowance prevents asserting finite-N unbiasedness here.
        assert np.all(np.abs(samples.mean(axis=0) - exact) < 6 * standard_errors + .003)

    low_error = np.mean((ensembles[512][1] - exact_density) ** 2)
    high_error = np.mean((densities - exact_density) ** 2)
    # A 16-fold budget increase should substantially reduce ensemble error.
    # We deliberately do not require every seed or trial to improve.
    assert high_error < low_error / 3
    assert np.sqrt(high_error) < .02


def test_hmm_masks_select_factors_but_all_observations_update_history(one_torch_thread):
    observed = np.array([0, 2, 2, 1, 3, 2, 2, 0, 1, 0, 3, 1])
    changed = observed.copy()
    changed[0] = 3
    mask = np.arange(len(observed)) % 3 != 0
    reference, _ = _forward(observed)
    changed_reference, _ = _forward(changed)
    assert abs(reference[1] - changed_reference[1]) > .04

    results = [_filter(observed, 8192, seed, mask) for seed in range(8)]
    changed_results = [_filter(changed, 8192, seed, mask) for seed in range(8)]
    for observations, reference_density, runs in [(observed, reference, results),
                                                 (changed, changed_reference, changed_results)]:
        scores = np.array([row[0] for row in runs])
        exact_selected_score = np.log(reference_density[mask]).sum()
        standard_error = scores.std(ddof=1) / np.sqrt(len(scores))
        assert abs(scores.mean() - exact_selected_score) < 6 * standard_error + .03
        # For a fixed seed, a scoring mask must not change the filtering path.
        full_score, full_density, full_posterior = _filter(observations, 8192, 0)
        np.testing.assert_array_equal(runs[0][1], full_density)
        np.testing.assert_array_equal(runs[0][2], full_posterior)
        assert full_score == pytest.approx(np.log(full_density.astype(float)).sum(), abs=3e-6)

    changes = np.array([new[1][1] - old[1][1] for old, new in zip(results, changed_results)])
    assert abs(changes.mean() - (changed_reference[1] - reference[1])) < .02
