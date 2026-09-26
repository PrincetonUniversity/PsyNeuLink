"""Audit experimental scoring shortcuts against the materialized PEC estimator."""

from pathlib import Path
import sys

import numpy as np
import optuna
import pytest
import torch
from optuna.distributions import FloatDistribution
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from psyneulink.core.batched import likelihood
from psyneulink.core.components.functions.nonstateful.fitfunctions import _run_batched_ask_tell_rounds

DIRECTORY = Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile/dawa"
sys.path.insert(0, str(DIRECTORY))
from dawa_fitting_budget_study import decision_time_counts, rescale_prior, score_ndt_grid, total_scores  # noqa: E402


@pytest.mark.parametrize("alpha", [.02, .1, 1.])
def test_cached_ndt_matches_materialized_histogram_at_edges_and_arbitrary_shifts(alpha):
    observed = np.array([[0., .01], [1., 1.2], [1., 3.], [0., .7]])
    exp = torch.tensor(observed, dtype=torch.float32)
    edges = likelihood._bin_edges(exp[:, 1:], exp[:, 1:], 100, [(0., 3.)], torch)[0]
    samples = torch.tensor([[[0., 0.], [0., .01], [1., .3], [0., 1.], [1., 2.99], [0., 3.1]]] * 4)
    # Exercise exact interior-edge membership and both ends of the finite range.
    samples[1, 2, 1] = edges[40] - .2
    samples[2, 4, 1] = edges[-1] - .2
    mask = np.array([True, True, True, False])
    times, counts = decision_time_counts(samples)
    shifts = np.array([0., .1, .1234, .2, .3001])
    actual = score_ndt_grid(times, counts, shifts, observed, mask, edges, 6, pseudocount=alpha, chunk=2)
    shifted = samples[None].repeat(len(shifts), 1, 1, 1)
    shifted[..., 1] += torch.tensor(shifts, dtype=torch.float32)[:, None, None]
    expected = likelihood.histogram_log_likelihood(
        shifted, observed, [0], bins=100, bin_range=[(0., 3.)], smoothing_sigma=.5,
        pseudocount=alpha, categorical_cardinalities=[2], include_mask=mask,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)


def test_scaled_prior_conversion_and_pooling_equal_combined_samples():
    generator = np.random.default_rng(728)
    samples = generator.uniform(0., 1., (2, 4, 20, 2)).astype(np.float32)
    samples[..., 0] = np.round(samples[..., 0])
    observed = np.array([[0, .3], [1, .5], [0, .8], [1, .6]])
    options = dict(categorical_dims=[0], bins=100, bin_range=[(0., 3.)], smoothing_sigma=.5,
                   categorical_cardinalities=[2])
    exp = torch.tensor(observed, dtype=torch.float32)
    edges = likelihood._bin_edges(exp[:, 1:], exp[:, 1:], 100, [(0., 3.)], torch)[0]
    block = []
    for part in (samples[:, :, :10], samples[:, :, 10:]):
        fixed = likelihood.histogram_likelihood(part, observed, pseudocount=1., **options)
        scaled = rescale_prior(fixed, 10, .1, float(edges[1] - edges[0]))
        oracle = likelihood.histogram_likelihood(part, observed, pseudocount=.1, **options)
        np.testing.assert_allclose(scaled, oracle, rtol=1e-6, atol=1e-7)
        block.append(scaled)
    combined = likelihood.histogram_likelihood(samples, observed, pseudocount=.2, **options)
    np.testing.assert_allclose(np.mean(block, axis=0), combined, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(total_scores(np.mean(block, axis=0), [True] * 4),
                               total_scores(combined, [True] * 4), rtol=1e-6, atol=1e-6)


def test_memory_and_journal_storage_preserve_cma_candidate_sequence(tmp_path):
    distributions = {f"x{i}": FloatDistribution(-1., 1., step=.01) for i in range(8)}
    initial = {name: 0. for name in distributions}
    sequences = []
    for storage in (None, JournalStorage(JournalFileBackend(str(tmp_path / "optimizer.journal")))):
        study = optuna.create_study(direction="maximize", storage=storage,
                                    sampler=optuna.samplers.CmaEsSampler(x0=initial, sigma0=.2,
                                                                        lr_adapt=True, popsize=4, seed=17))
        study.enqueue_trial(initial)
        sequence = []

        def evaluate(candidates):
            sequence.extend(candidates)
            return -np.square(np.asarray(candidates) - .3).sum(-1)

        _run_batched_ask_tell_rounds(study, distributions, list(distributions), 4, 21,
                                   evaluate, startup_trials=1)
        sequences.append(sequence)
    np.testing.assert_array_equal(sequences[0], sequences[1])
