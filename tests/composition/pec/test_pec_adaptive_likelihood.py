"""Adaptive likelihood budgets, independent final scores, and paired precision."""

import numpy as np
import pytest

from psyneulink.core.components.functions.nonstateful.adaptivelikelihood import (
    adaptive_log_likelihood,
)


def test_budget_growth_keeps_candidates_together_and_uses_only_final_scores():
    calls = []
    residuals = np.array([-3.0, -1.0, 1.0, 3.0])

    def evaluate(rows, count, seed):
        index = len(calls)
        calls.append((rows.copy(), count, seed))
        # Deliberate N-dependent bias catches accidental pooling across counts.
        return (
            np.array([-10.0, -12.0])
            + 100 / count
            + residuals[index % 4] / np.sqrt(count)
        )

    result = adaptive_log_likelihood(
        evaluate,
        [[0.1], [0.2]],
        max_estimates=40,
        seed=7,
        options=dict(min_estimates=4, repeats=4, target_se=0.4),
    )
    assert [stage["num_estimates"] for stage in result.history] == [4, 8, 16]
    assert [count for _, count, _ in calls] == [4] * 4 + [8] * 4 + [16] * 8
    assert result.num_estimates == 16
    assert result.converged and result.stop_reason == "target_met"
    np.testing.assert_allclose(result.log_likelihood, [-3.75, -5.75])
    np.testing.assert_allclose(result.standard_error, np.sqrt(5 / 48))
    np.testing.assert_array_equal(
        result.precision_standard_error, result.standard_error
    )
    assert result.log_likelihood_difference is None
    assert result.difference_standard_error is None
    assert result.sampling_work == {
        "batch_calls": 16,
        "candidate_runs": 32,
        "candidate_particles": 352,
    }
    assert len({seed for _, _, seed in calls}) == 16
    assert 7 not in {seed for _, _, seed in calls}
    assert tuple(seed for _, _, seed in calls[-4:]) == result.seeds


def test_final_check_can_fail_without_selecting_quieter_replicates():
    calls = []

    def evaluate(rows, count, seed):
        calls.append(seed)
        return [0.0 if len(calls) <= 4 else 10.0 * len(calls)]

    result = adaptive_log_likelihood(
        evaluate,
        [[0.1]],
        max_estimates=100,
        seed=29,
        options=dict(min_estimates=4, repeats=4, target_se=0.2),
    )
    assert result.history[0]["target_met"].all()
    assert not result.converged
    assert result.stop_reason == "precision_not_confirmed"
    assert result.num_estimates == 4
    assert len(calls) == 8
    np.testing.assert_array_equal(result.log_likelihood, [65.0])
    # Arithmetic mean of complete log scores, not log(mean(exp(log scores))).
    assert result.log_likelihood[0] != pytest.approx(
        np.logaddexp.reduce([50, 60, 70, 80]) - np.log(4)
    )


def test_cap_is_respected_even_when_target_cannot_be_met():
    calls = []

    def evaluate(rows, count, seed):
        calls.append(count)
        return [100.0 * (len(calls) % 2)]

    result = adaptive_log_likelihood(
        evaluate,
        [[0.1]],
        max_estimates=13,
        seed=3,
        options=dict(min_estimates=4, repeats=2, target_se=0.01),
    )
    assert calls == [4, 4, 8, 8, 13, 13]
    assert result.num_estimates == 13
    assert not result.converged and result.stop_reason == "max_estimates"


def test_fixed_minimum_equal_to_cap_does_not_waste_work_on_pilots():
    result = adaptive_log_likelihood(
        lambda rows, count, seed: [-3],
        [[1]],
        max_estimates=4,
        seed=2,
        options=dict(min_estimates=4, repeats=3),
    )
    assert result.history == ()
    assert result.converged and result.num_estimates == 4
    assert result.sampling_work == {
        "batch_calls": 3,
        "candidate_runs": 3,
        "candidate_particles": 12,
    }


def test_paired_precision_retains_covariance_instead_of_combining_marginal_errors():
    def evaluate(rows, count, seed):
        return np.array([-2.0, -3.0, -4.0]) + np.random.default_rng(seed).normal() * 100

    result = adaptive_log_likelihood(
        evaluate,
        [[1], [2], [3]],
        max_estimates=100,
        seed=23,
        options=dict(min_estimates=4, target_se=0.01, reference_index=1),
    )
    assert result.num_estimates == 4 and result.converged
    assert np.all(result.standard_error > 1)
    np.testing.assert_allclose(result.log_likelihood_difference, [1, 0, -1])
    np.testing.assert_allclose(result.difference_standard_error, 0, atol=1e-13)
    np.testing.assert_array_equal(
        result.precision_standard_error, result.difference_standard_error
    )


def test_seed_replay_and_reservations():
    def evaluate(rows, count, seed):
        return np.random.default_rng(seed).normal(size=len(rows))

    kwargs = dict(max_estimates=32, seed=81, options=dict(min_estimates=8, repeats=2))
    first = adaptive_log_likelihood(evaluate, [[1], [2]], **kwargs)
    second = adaptive_log_likelihood(evaluate, [[1], [2]], **kwargs)
    np.testing.assert_array_equal(
        first.replicate_log_likelihoods, second.replicate_log_likelihoods
    )
    assert first.seeds == second.seeds
    reserved = [first.history[0]["seeds"][0], first.seeds[0], 81]
    kwargs["options"]["reserved_seeds"] = reserved
    third = adaptive_log_likelihood(evaluate, [[1], [2]], **kwargs)
    used = set(third.seeds).union(*(stage["seeds"] for stage in third.history))
    assert used.isdisjoint(reserved)


@pytest.mark.parametrize(
    "options",
    [
        dict(min_estimates=0),
        dict(min_estimates=33),
        dict(min_estimates=True),
        dict(repeats=1),
        dict(repeats=2.5),
        dict(growth_factor=1),
        dict(target_se=0),
        dict(target_se=np.nan),
        dict(target_se=True),
        dict(reference_index=2),
        dict(reference_index=-1),
        dict(reference_index=True),
        dict(reserved_seeds=None),
        dict(reserved_seeds=[-1]),
        dict(reserved_seeds=[True]),
        dict(unknown_option=4),
        [],
    ],
)
def test_invalid_configuration_fails_before_sampling(options):
    def unexpected(*args):
        pytest.fail("Invalid configuration reached the evaluator")

    with pytest.raises((ValueError, TypeError)):
        adaptive_log_likelihood(
            unexpected, [[1], [2]], max_estimates=32, seed=3, options=options
        )


@pytest.mark.parametrize(
    "rows,seed",
    [
        ([], 3),
        ([1], 3),
        ([[np.nan]], 3),
        ([[1]], None),
        ([[1]], -1),
        ([[1]], True),
    ],
)
def test_invalid_rows_and_seed_fail_before_sampling(rows, seed):
    with pytest.raises(ValueError):
        adaptive_log_likelihood(
            lambda *args: pytest.fail("Unexpected evaluation"),
            rows,
            max_estimates=32,
            seed=seed,
        )


def test_paired_precision_cannot_compare_a_single_row_to_itself():
    with pytest.raises(ValueError, match="at least two"):
        adaptive_log_likelihood(
            lambda *args: [1],
            [[1]],
            max_estimates=32,
            seed=2,
            options=dict(reference_index=0),
        )


@pytest.mark.parametrize("scores", [[np.nan], [np.inf], [-np.inf], [[1.0]], [1.0, 2.0]])
def test_nonfinite_scores_and_trial_factors_are_not_precision_evidence(scores):
    with pytest.raises(FloatingPointError, match="finite complete log score"):
        adaptive_log_likelihood(lambda *args: scores, [[1]], max_estimates=32, seed=2)


def test_truncation_propagates_without_penalties():
    from psyneulink.core.batched.backend.triton.runtime import BatchedTruncationError

    def truncate(*args):
        raise BatchedTruncationError("execution cap")

    with pytest.raises(BatchedTruncationError, match="execution cap"):
        adaptive_log_likelihood(truncate, [[1]], max_estimates=32, seed=2)


def test_reported_standard_error_matches_sampling_variability_on_gaussian_oracle():
    # The pilot can select a noisy budget; final scores must retain their
    # conditional sampling distribution instead of reusing a lucky pilot block.
    standardized_errors = []
    for seed in range(200):
        result = adaptive_log_likelihood(
            lambda rows, count, seed: [
                -3 + np.random.default_rng(seed).normal() / np.sqrt(count)
            ],
            [[1]],
            max_estimates=128,
            seed=seed,
            options=dict(min_estimates=2, repeats=8, target_se=0.1),
        )
        standardized_errors.append(
            (result.log_likelihood[0] + 3) * np.sqrt(8 * result.num_estimates)
        )
    # Use the known oracle variance, not the noisy estimated SE, for this check.
    assert abs(np.mean(standardized_errors)) < 0.2
    assert 0.7 < np.std(standardized_errors, ddof=1) < 1.3
