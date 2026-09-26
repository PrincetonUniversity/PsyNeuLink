"""Exact support compression and shifted scores must match materialized histories."""

import numpy as np
import pytest

from psyneulink.core.batched import likelihood
from psyneulink.core.batched.errors import BatchedNumericalError
from psyneulink.core.batched.backend.triton.runtime import BatchedTruncationError
from psyneulink.core.batched.shifted_histogram import ShiftedHistogramScorer
from tests.composition.pec.test_batched_fused_histogram import _model

pytestmark = [pytest.mark.batched, pytest.mark.composition]


@pytest.mark.parametrize("schedule", ["synchronized", "independent"])
@pytest.mark.parametrize("common_random", [True, False])
def test_seed_blocks_preserve_addresses_and_complete_histories(batched_backend, schedule, common_random):
    plan, lca = _model(batched_backend)
    inputs = {lca: [[3., 1.], [1., 3.], [2., 1.], [1., 3.], [3., 1.], [1., 2.]]}
    parameters = [{}, {f"{lca.name}.gain": 1.2}]
    support = np.arange(97, dtype=np.float32) * np.float32(.1)
    data = np.array([[0, .3], [1, .6], [0, -.1]])
    options = dict(subject_slices=[slice(0, 3), slice(3, 6)], common_random_numbers=common_random,
                   triton_launch_options={"trial_schedule": schedule}, outcome_indices=[2, 3], support=support)
    seeds = [0, 37, 2**63 + 19, 37]
    blocks = plan.discrete_output_count_blocks(inputs, parameters, 37, data, [0], seeds=seeds, **options)
    assert len(blocks) == len(seeds)
    for seed, block in zip(seeds, blocks, strict=True):
        single = plan.discrete_output_counts(inputs, parameters, 37, data, [0], seed=seed, **options)
        np.testing.assert_array_equal(block.counts.cpu(), single.counts.cpu())
        np.testing.assert_array_equal(block.valid_candidates, single.valid_candidates)
        # The materialized simulator is also an independent seed-address oracle.
        raw_options = {key: value for key, value in options.items() if key not in ("support", "outcome_indices")}
        raw = plan.run(inputs, parameters, 37, seed=seed, **raw_options).values[..., [2, 3]].reshape(4, 3, 37, 2)
        expected = np.zeros((4, 3, len(support)), dtype=np.int32)
        for c in range(4):
            for t in range(3):
                values = raw[c, t, raw[c, t, :, 0] == data[t, 0], 1]
                np.add.at(expected[c, t], np.searchsorted(support, values), 1)
        np.testing.assert_array_equal(block.counts.cpu().numpy().reshape(expected.shape), expected)
    assert not np.array_equal(blocks[0].counts.cpu(), blocks[1].counts.cpu())


@pytest.mark.parametrize("schedule", ["synchronized", "independent"])
@pytest.mark.parametrize("common_random", [True, False])
def test_counts_and_shifted_scores_match_materialized_histories(batched_backend, schedule, common_random, monkeypatch):
    plan, lca = _model(batched_backend)
    inputs = {lca: [[3., 1.], [1., 3.], [2., 1.], [1., 3.], [3., 1.], [1., 2.]]}
    parameters = [{}, {f"{lca.name}.gain": 1.2}]
    options = dict(seed=29, subject_slices=[slice(0, 3), slice(3, 6)], common_random_numbers=common_random,
                   triton_launch_options={"trial_schedule": schedule}, strict_truncation=True)
    samples = plan.run(inputs, parameters, 37, **options).values[..., [2, 3]].reshape(4, 3, 37, 2)
    support = np.arange(32 * 3 + 1, dtype=np.float32) * np.float32(.1)
    data = np.array([[0, .3], [1, .6], [0, -.1]])
    del options["strict_truncation"]
    reduced = plan.discrete_output_counts(inputs, parameters, 37, data, [0], support=support,
                                          outcome_indices=[2, 3], **options)
    expected = np.zeros((4, 3, len(support)), dtype=np.int32)
    for c in range(4):
        for t in range(3):
            values = samples[c, t, samples[c, t, :, 0] == data[t, 0], 1]
            np.add.at(expected[c, t], np.searchsorted(support, values), 1)
    np.testing.assert_array_equal(reduced.counts.cpu().numpy().reshape(expected.shape), expected)
    assert reduced.valid_candidates.all()
    monkeypatch.setattr(type(plan), "run", lambda *a, **k: pytest.fail("Counts must not materialize trajectories"))
    plan.discrete_output_counts(inputs, parameters, 37, data, [0], support=support, outcome_indices=[2, 3], **options)
    for sigma, alpha in [(0., 0.), (.5, .37), (1., 1.)]:
        scorer = ShiftedHistogramScorer(support, [-.1, 0., .1234, .2, .3], data, [0], bins=5,
                                        bin_range=[(0., .7)], smoothing_sigma=sigma, categorical_cardinalities=[2],
                                        device=str(reduced.counts.device))
        actual = scorer.densities(reduced, pseudocount=alpha).reshape(4, len(scorer.shifts), 3)
        for k, shift in enumerate(scorer.shifts):
            shifted = samples.copy()
            shifted[..., 1] += np.float32(shift)
            reference = likelihood.histogram_likelihood(shifted, data, [0], bins=5, bin_range=[(0., .7)],
                                                        smoothing_sigma=sigma, pseudocount=alpha,
                                                        categorical_cardinalities=[2])
            np.testing.assert_allclose(actual[:, k], reference, rtol=2e-6, atol=1e-7)


def test_counts_reject_missing_support_and_truncated_candidates(batched_backend):
    plan, lca = _model(batched_backend, max_steps=8)
    inputs = {lca: [[3., 1.], [1., 3.]]}
    data = np.array([[0., .1], [1., .2]])
    good = {f"{lca.name}.termination_threshold": .2}
    bad = {f"{lca.name}.termination_threshold": 1.1}
    support = np.arange(17, dtype=np.float32) * np.float32(.1)
    options = dict(outcome_indices=[2, 3], support=support, seed=29,
                   triton_launch_options={"trial_schedule": "independent"})
    with pytest.raises(BatchedTruncationError):
        plan.discrete_output_counts(inputs, [good, bad], 5, data, [0], **options)
    result = plan.discrete_output_counts(inputs, [good, bad, good], 5, data, [0], invalid_candidates="nan", **options)
    np.testing.assert_array_equal(result.valid_candidates, [True, False, True])
    block_options = {key: value for key, value in options.items() if key != "seed"}
    blocks = plan.discrete_output_count_blocks(inputs, [good, bad, good], 5, data, [0], seeds=[29, 31],
                                               invalid_candidates="nan", **block_options)
    for block in blocks:
        np.testing.assert_array_equal(block.valid_candidates, [True, False, True])
    for invalid_seeds in ([], [-1], [2**64], [True], [1.5]):
        with pytest.raises(ValueError, match="seeds"):
            plan.discrete_output_count_blocks(inputs, [good], 5, data, [0], seeds=invalid_seeds, **block_options)
    scorer = ShiftedHistogramScorer(support, [.1, .2], data, [0], bins=5, bin_range=[(0., 2.)],
                                    device=str(result.counts.device))
    assert np.isnan(scorer.densities(result)[1]).all()
    options["support"] = [.123456]
    with pytest.raises(BatchedNumericalError, match="exact support"):
        plan.discrete_output_counts(inputs, [good], 5, data, [0], **options)
    options['support'] = np.array([0., 3e38], dtype=np.float32)
    with pytest.raises(BatchedNumericalError, match="nonfinite"):
        plan.discrete_output_counts(inputs, [{f"{lca.name}.time_step_size": 3e38}], 5, data, [0],
                                    invalid_candidates="nan", **options)


def test_shift_map_deduplication_preserves_all_grid_scores():
    import torch
    from psyneulink.core.batched.backend.triton.discrete_counts import DiscreteOutputCounts

    support = np.arange(301, dtype=np.float32) * np.float32(.01)
    shifts = np.linspace(.1, .3, 2001)
    data = np.array([[0., .22], [1., .3]])
    scorer = ShiftedHistogramScorer(support, shifts, data, [0], bins=100, bin_range=[(0., 3.)],
                                    smoothing_sigma=.5, categorical_cardinalities=[2], device="cpu")
    assert len(scorer.shifts) < 100
    rng = np.random.default_rng(125)
    samples = np.zeros((2, 100, 2), dtype=np.float32)
    samples[..., 0] = rng.integers(0, 2, (2, 100))
    samples[..., 1] = rng.choice(support, (2, 100))
    counts = np.zeros((1, 1, 2, len(support)), dtype=np.int32)
    for t in range(2):
        times = samples[t, samples[t, :, 0] == data[t, 0], 1]
        np.add.at(counts[0, 0, t], np.searchsorted(support, times), 1)
    reduced = DiscreteOutputCounts(torch.tensor(counts), torch.tensor(support), 100, np.array([True]))
    actual = scorer.densities(reduced, pseudocount=1.)[0, 0]
    scores = []
    for shift in shifts:
        shifted = samples.copy()
        shifted[..., 1] += np.float32(shift)
        density = likelihood.histogram_likelihood(shifted, data, [0], bins=100, bin_range=[(0., 3.)],
                                                  smoothing_sigma=.5, pseudocount=1., categorical_cardinalities=[2])
        scores.append(np.log(density).sum())
    assert np.log(actual).sum(-1).max() == pytest.approx(max(scores), abs=2e-6)
