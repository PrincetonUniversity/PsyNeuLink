"""Adaptive precision must pool probabilities and keep final selection separate."""

from pathlib import Path
import sys

import numpy as np
import optuna
import pytest

DIRECTORY = Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile/dawa"
sys.path.insert(0, str(DIRECTORY))
from dawa_adaptive_fit import (  # noqa: E402
    AdaptiveConfig, CovarianceCmaEsSampler, PopulationRacer,
    fit_adaptive, learned_covariance, pooled_scores, ranking_uncertainty,
)


def test_unequal_blocks_pool_before_logs_and_ignore_masked_trials():
    blocks = [np.array([[1., 100., 3.], [2., 1., 1.]]), np.array([[3., .001, 1.], [1., 999., 3.]])]
    scores, se, valid = pooled_scores(blocks, [2, 6], [True, False, True])
    expected = np.log((blocks[0] + 3 * blocks[1]) / 4)[:, [0, 2]].sum(-1)
    np.testing.assert_allclose(scores, expected)
    assert valid.all() and np.all(np.diag(se) == 0) and se[0, 1] > 0
    assert not np.allclose(scores, (np.log(blocks[0]) + 3 * np.log(blocks[1]))[:, [0, 2]].sum(-1) / 4)


@pytest.mark.parametrize("noisy", [False, True])
def test_race_preserves_old_blocks_and_uses_unique_seeds(noisy):
    calls = []

    def sample(rows, size, seed):
        calls.append((size, seed))
        if noisy:
            return np.array([[2., 1.], [1., 2.]]) if len(calls) % 2 else np.array([[1., 2.], [2., 1.]])
        return np.array([[4., 4.], [1., 1.]])

    config = AdaptiveConfig(min_estimates=4, max_estimates=16, rank_tolerance=0.)
    racer = PopulationRacer(sample, [True, False], config, 29, reserved_seeds=[8101, 8102])
    scores, detail = racer.evaluate([[0], [1]])
    assert detail["block_sizes"] == ([1, 1, 1, 1, 2, 2, 4, 4] if noisy else [1, 1, 1, 1])
    assert sum(size for size, _ in calls) == detail["estimates"]
    assert len(set(seed for _, seed in calls)) == len(calls)
    assert not ({29, 8101, 8102} & set(seed for _, seed in calls))
    assert np.isfinite(scores).all()


def test_checked_final_selection_ignores_biased_low_budget_scores():
    bounds = {"x": (0., 1., .01), "y": (0., 1., .01)}
    initial = {"x": .2, "y": .2}
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.CmaEsSampler(
        x0=initial, sigma0=.2, lr_adapt=True, popsize=4, seed=10))
    study.enqueue_trial(initial)
    calls, records = [], []

    def sample(rows, estimates, seed):
        calls.append((estimates, seed))
        x = np.asarray(rows)
        truth = -np.square(x - .6).sum(-1)
        return np.exp(truth[:, None] + (10. if estimates < 32 else 0.))

    def log(rows, scores, elapsed, metadata):
        records.extend((row, score, metadata) for row, score in zip(rows, scores))

    config = AdaptiveConfig(min_estimates=4, max_estimates=32, check_every=4, min_evaluations=5,
                            patience=1, progress_tolerance=100., refine_evaluations=5)
    fit, detail, refinement = fit_adaptive(study, bounds, initial, sample, [True], config,
                                          evaluations=30, population=4, simulation_seed=29,
                                          optimizer_seed=10, reserved_seeds=[8101], log_batch=log)
    fitted = np.array(list(fit["fitted_params"].values()))
    assert fit["optimal_value"] == pytest.approx(-np.square(fitted - .6).sum())
    assert detail["search_evaluations"] == 5
    assert len(records) == 10 and len(refinement.trials) == 5
    assert max(score for _, score, _ in records) > 9
    assert all(seed != 8101 for _, seed in calls)
    assert fit["optimal_value"] >= detail["initial_reference_score"]
    assert detail["refinement_evaluations"] == config.refine_evaluations
    assert "convergence not asserted" in detail["search_stop_reason"]
    selection_seeds = set(detail["final_selection"]["seeds"])
    racing_seeds = {seed for _, _, metadata in records if metadata["phase"] == "adaptive_search"
                    for seed in metadata["block_seeds"]}
    assert len(selection_seeds) == config.selection_blocks
    assert not selection_seeds & (racing_seeds | {29, 8101})


def test_rank_uncertainty_includes_swaps_within_elite():
    scores = np.array([20., 19.9, 10., 0.])
    pair_se = np.zeros((4, 4))
    pair_se[0, 1] = pair_se[1, 0] = 3.
    uncertainty, promote = ranking_uncertainty(scores, pair_se, np.ones(4, dtype=bool), 1.)
    assert promote and uncertainty > 1.
    # Top-half membership is certain; the uncertainty is wholly within that half.
    assert np.all(pair_se[:2, 2:] == 0)


def test_covariance_restart_keeps_normalized_parameter_order_and_correlations():
    from optuna._transform import _SearchSpaceTransform
    from optuna.distributions import FloatDistribution

    # Deliberately reverse insertion order, with different physical scales.
    distributions = {"z": FloatDistribution(10., 20.), "a": FloatDistribution(0., 1.)}
    initial = {"z": 15., "a": .5}
    covariance = np.array([[2., .7], [.7, .4]])
    sampler = CovarianceCmaEsSampler(covariance=covariance, parameter_order=["a", "z"],
                                    x0=initial, sigma0=.03, seed=11, popsize=4)
    transformed = _SearchSpaceTransform(dict(sorted(distributions.items())), transform_0_1=True)
    optimizer = sampler._init_optimizer(transformed, optuna.study.StudyDirection.MAXIMIZE)
    np.testing.assert_allclose(optimizer._C, covariance)
    draws = np.array([optimizer.ask() for _ in range(1000)])
    assert np.corrcoef(draws.T)[0, 1] > .7
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial(initial)
    for _ in range(6):
        study.tell(study.ask(distributions), 0.)
    restored = learned_covariance(study, list(distributions))
    assert restored is not None and np.linalg.eigvalsh(restored).min() > 0
    assert restored[0, 1] > 0
    covariance[0, 0] = 99.
    assert sampler._initial_covariance[0, 0] == 2.

    incorrect = optuna.create_study(direction="maximize", sampler=CovarianceCmaEsSampler(
        covariance=restored, parameter_order=["z", "a"], x0=initial, sigma0=.03, seed=11, popsize=4))
    incorrect.enqueue_trial(initial)
    incorrect.tell(incorrect.ask(distributions), 0.)
    with pytest.raises(RuntimeError, match="parameter order"):
        incorrect.ask(distributions)


def test_fresh_selection_can_reject_the_reference_seed_winner():
    bounds = {"x": (0., 1., .01), "y": (0., 1., .01)}
    initial = {"x": .4, "y": .4}
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.CmaEsSampler(
        x0=initial, sigma0=.2, lr_adapt=True, popsize=4, seed=10))
    study.enqueue_trial(initial)

    def sample(rows, estimates, seed):
        x = np.asarray(rows)
        # Opposite rankings on the training reference seed and fresh seeds.
        return np.exp((x[:, :1] + x[:, 1:]) * (1 if seed == 29 else -1))

    config = AdaptiveConfig(min_estimates=4, max_estimates=32, check_every=4,
                            min_evaluations=5, refine_evaluations=5, max_search_evaluations=9)
    fit, detail, _ = fit_adaptive(study, bounds, initial, sample, [True], config,
                                 evaluations=30, population=4, simulation_seed=29,
                                 optimizer_seed=10, reserved_seeds=[8101], log_batch=lambda *args: None)
    selection = detail["final_selection"]
    assert selection["winner"] != 0
    assert fit["optimal_value"] < detail["best_reference_score"]
    assert list(fit["fitted_params"].values()) == selection["candidates"][selection["winner"]]
    assert detail["search_evaluations"] == 9


def test_invalid_block_never_becomes_a_valid_pooled_candidate():
    scores, _, valid = pooled_scores([np.array([[np.nan], [1.]]), np.ones((2, 1))], [2, 2], [True])
    assert scores[0] == -1e10 and not valid[0] and valid[1]
