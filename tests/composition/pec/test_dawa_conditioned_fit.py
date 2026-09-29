"""Staged fitting must rerun whole filters and separate search and validation."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import optuna
import pytest

DIRECTORY = Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile/dawa"
sys.path.insert(0, str(DIRECTORY))
from dawa_conditioned_fit import StagedConfig, conditioned_scores, fit_staged  # noqa: E402
from dawa_adaptive_fit import PENALTY  # noqa: E402


def run_policy(sample, *, reserved=(), **options):
    bounds = {"y": (0.0, 1.0, 0.01), "x": (0.0, 1.0, 0.01)}
    initial = {"y": 0.2, "x": 0.2}
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.CmaEsSampler(
            x0=initial, sigma0=0.2, lr_adapt=True, popsize=4, seed=10
        ),
    )
    study.enqueue_trial(initial)
    records = []

    def log(rows, scores, elapsed, metadata):
        records.extend(
            (row, score, metadata) for row, score in zip(rows, scores, strict=True)
        )

    config = StagedConfig(
        search_estimates=4,
        reference_estimates=32,
        refine_evaluations=9,
        check_every=4,
        min_evaluations=9,
        patience=1,
        progress_tolerance=100.0,
        **options,
    )
    result = fit_staged(
        study,
        bounds,
        initial,
        sample,
        config,
        evaluations=30,
        population=4,
        simulation_seed=29,
        optimizer_seed=10,
        reserved_seeds=reserved,
        log_batch=log,
    )
    return *result, records


def test_reference_checks_ignore_low_count_bias_and_reuse_covariance():
    calls = []

    def sample(rows, count, seed):
        calls.append((np.asarray(rows), count, seed))
        return -np.square(np.asarray(rows) - 0.6).sum(-1) + (
            100.0 if count == 4 else 0.0
        )

    # Deliberately reserve the policy RNG's first draw, as well as validation seeds.
    reserved = {
        8101,
        8102,
        int(
            np.random.default_rng(np.random.SeedSequence([29, 69471])).integers(
                0, 2**31 - 1
            )
        ),
    }
    fit, detail, refinement, records = run_policy(sample, reserved=reserved)
    fitted = np.array(list(fit["fitted_params"].values()))
    assert fit["optimal_value"] == pytest.approx(-np.square(fitted - 0.6).sum())
    assert detail["search_evaluations"] == 9
    assert detail["refinement_evaluations"] == len(refinement.trials) == 9
    assert len(records) == 18
    assert detail["refinement_covariance"]["reused"]
    assert detail["refinement_covariance"]["parameter_order"] == ["x", "y"]
    assert max(score for _, score, _ in records) > 99.0
    assert {count for _, count, _ in calls} == {4, 32}
    assert all(seed not in reserved for _, _, seed in calls)
    assert all(count == 32 for _, count, seed in calls if seed != 29)
    assert {
        metadata["estimates"]
        for _, _, metadata in records
        if metadata["phase"] == "staged_search"
    } == {4}
    assert {
        metadata["estimates"]
        for _, _, metadata in records
        if metadata["phase"] == "refinement"
    } == {32}
    assert all(check["reference_score"] <= 0 for check in detail["checkpoints"])
    assert "convergence not asserted" in detail["search_stop_reason"]
    selection = detail["final_selection"]
    assert len(set(selection["seeds"])) == 3
    assert set(selection["seeds"]).isdisjoint(reserved | {29})
    np.testing.assert_allclose(
        selection["mean_log_scores"], np.mean(selection["replicate_log_scores"], axis=0)
    )
    assert fit["optimal_value"] == selection["reference_scores"][selection["winner"]]


def test_selection_averages_whole_log_scores_and_excludes_any_truncation():
    selection_order = []

    def sample(rows, count, seed):
        if seed == 29:
            return -np.square(np.asarray(rows) - 0.6).sum(-1)
        if seed not in selection_order:
            selection_order.append(seed)
        # Candidate 0 wins a log-mean-exp but loses the mean log score;
        # candidate 2 must be rejected despite large scores on other seeds.
        return np.array(
            [[10.0, 2.0, PENALTY], [-10.0, 2.0, 100.0], [-10.0, 2.0, 100.0]]
        )[selection_order.index(seed)]

    fit, detail, _, _ = run_policy(sample, selection_candidates=3)
    selection = detail["final_selection"]
    assert selection["winner"] == 1
    assert selection["valid"] == [True, True, False]
    assert selection["mean_log_scores"] == pytest.approx([-10 / 3, 2.0, PENALTY])
    assert list(fit["fitted_params"].values()) == selection["candidates"][1]


@pytest.mark.parametrize("bad", [np.ones((1, 2)), np.array([np.nan])])
def test_policy_rejects_trial_factors_and_nonfinite_scores(bad):
    with pytest.raises(FloatingPointError, match="one finite complete-filter"):
        run_policy(lambda *args: bad)


@pytest.mark.parametrize(
    "setting",
    [
        dict(search_estimates=0),
        dict(search_estimates=33),
        dict(selection_repeats=0),
        dict(progress_tolerance=np.nan),
        dict(refine_evaluations=29),
    ],
)
def test_invalid_policy_configuration(setting):
    options = dict(search_estimates=4, reference_estimates=32, refine_evaluations=9)
    options.update(setting)
    with pytest.raises(ValueError):
        StagedConfig(**options).validate(30, 4)


@pytest.mark.parametrize("failure", [False, True])
def test_callback_scales_contamination_and_restores_settings_on_errors(failure):
    seen = []
    function = SimpleNamespace(
        conditioned_likelihood=True, batched_seed=29, batched_pseudocount=1.0
    )
    controller = SimpleNamespace(function=function, num_estimates=100)
    pec = SimpleNamespace(controller=controller)

    def make_objective():
        def score(rows):
            seen.append(
                (
                    controller.num_estimates,
                    function.batched_seed,
                    function.batched_pseudocount,
                )
            )
            if failure:
                raise RuntimeError("unexpected error")
            return np.array([3.0])

        return SimpleNamespace(_batched_parameter_sets=score)

    function._make_objective_func = make_objective
    work = {}
    kwargs = dict(reference_estimates=100, pseudocount=1.0, invalid=[], work=work)
    if failure:
        with pytest.raises(RuntimeError, match="unexpected error"):
            conditioned_scores(pec, [[0.1]], 25, 77, **kwargs)
    else:
        np.testing.assert_array_equal(
            conditioned_scores(pec, [[0.1]], 25, 77, **kwargs), [3.0]
        )
    assert seen == [(25, 77, 0.25)]
    assert (
        controller.num_estimates,
        function.batched_seed,
        function.batched_pseudocount,
    ) == (100, 29, 1.0)
    assert work == {
        "filter_batch_calls": 1,
        "candidate_filter_runs": 1,
        "candidate_particles": 25,
    }


def test_callback_retries_truncating_batches_without_accepting_partial_paths():
    from psyneulink.core.batched.backend.triton.runtime import BatchedTruncationError

    seen = []
    function = SimpleNamespace(
        conditioned_likelihood=True, batched_seed=29, batched_pseudocount=1.0
    )
    controller = SimpleNamespace(function=function, num_estimates=100)

    def score(rows):
        seen.append(rows)
        if len(rows) > 1 or rows[0][0] == 0.9:
            raise BatchedTruncationError("execution cap")
        return [2.0]

    function._make_objective_func = lambda: SimpleNamespace(
        _batched_parameter_sets=score
    )
    invalid, work = [], {}
    scores = conditioned_scores(
        SimpleNamespace(controller=controller),
        [[0.1], [0.9]],
        25,
        77,
        reference_estimates=100,
        pseudocount=1.0,
        invalid=invalid,
        work=work,
    )
    np.testing.assert_array_equal(scores, [2.0, PENALTY])
    assert seen == [[[0.1], [0.9]], [[0.1]], [[0.9]]]
    assert invalid == [
        {"parameters": [0.9], "estimates": 25, "seed": 77, "reason": "execution cap"}
    ]
    assert work == {
        "filter_batch_calls": 3,
        "candidate_filter_runs": 4,
        "candidate_particles": 100,
    }


def test_all_truncated_finalists_are_an_error():
    def sample(rows, count, seed):
        return (
            -np.square(np.asarray(rows) - 0.6).sum(-1)
            if seed == 29
            else np.full(len(rows), PENALTY)
        )

    with pytest.raises(RuntimeError, match="All final candidates truncated"):
        run_policy(sample)
