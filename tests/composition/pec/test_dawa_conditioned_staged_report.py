"""Check that staged comparisons use a common finite reference and observations."""

import copy
from pathlib import Path
import sys

import pytest

DIRECTORY = Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile/dawa"
sys.path.insert(0, str(DIRECTORY))
from dawa_conditioned_staged_report import fit_comparison, ranking_summary  # noqa: E402


def test_rank_loss_uses_largest_count_mean_and_preserves_log_scores():
    labels = ["near_rank_0", "near_rank_1", "initial"]
    # Deliberately list the reference count first, and add a large low-count offset.
    report = {
        "status": "complete",
        "config": {
            "estimates": [16, 4],
            "seeds": [71, 72],
            "candidates": {"candidates": [{"label": label} for label in labels]},
        },
        "environment": {},
        "implementation_sha256": {},
        "git_commit": "test",
        "records": [],
    }
    for count, scores in ((16, [1.0, 3.0, 2.0]), (4, [110.0, 101.0, 115.0])):
        for seed in (71, 72):
            for label, score in zip(labels, scores, strict=True):
                report["records"].append(
                    dict(
                        estimates=count, seed=seed, label=label, device_log_score=score
                    )
                )
    result = ranking_summary(report)
    assert result["reference_mean_log_scores"] == [1.0, 3.0, 2.0]
    budgets = result["groups"]["all"]["budgets"]
    assert budgets[0]["mean_winner_reference_loss"] == 0.0
    assert budgets[1]["winner_reference_loss_per_seed"] == [1.0, 1.0]
    assert (
        result["groups"]["near_optimum_saved_candidates"]["budgets"][1][
            "mean_winner_reference_loss"
        ]
        == 2.0
    )
    assert result["complete_log_scores"][1][0] == [110.0, 101.0, 115.0]
    report["records"].append(report["records"][0])
    with pytest.raises(ValueError, match="exactly one"):
        ranking_summary(report)


@pytest.mark.parametrize("policy_key", ["staged", "adaptive"])
def test_fit_comparison_pairs_seeds_and_refuses_mismatched_observations(policy_key):
    fixed = dict(
        status="complete",
        validation_estimates=100,
        validation_pseudocount=1.0,
        fitted={},
        fit_seconds=120.0,
        total_seconds=150.0,
        evaluations=20,
        independent_seed_rescoring=[
            dict(seed=1, fitted=20.0),
            dict(seed=2, fitted=30.0),
        ],
    )
    staged = {
        **fixed,
        policy_key: {"policy": "staged"},
        "fit_seconds": 60.0,
        "independent_seed_rescoring": [
            dict(seed=2, fitted=32.0),
            dict(seed=1, fitted=21.0),
        ],
    }
    manifest = {
        key: {}
        for key in (
            "initial",
            "bounds",
            "noise",
            "time_steps",
            "estimator",
            "arguments",
        )
    }
    manifest.update(
        observations_sha256="observations",
        source_model_sha256="model",
        trials=10,
        scored_trials=9,
        git_commit="revision",
        driver_sha256="driver",
        gpu="gpu",
        staged_driver_sha256="staged",
    )
    if policy_key == "adaptive":
        manifest["adaptive_driver_sha256"] = manifest.pop("staged_driver_sha256")
    result = fit_comparison(fixed, staged, manifest, manifest)
    assert result["staged"]["staged"] == {"policy": "staged"}
    assert result["staged_provenance"]["staged_driver_sha256"] == "staged"
    assert result["fit_seconds_ratio_fixed_over_staged"] == 2.0
    assert result["staged_minus_fixed_validation"] == {
        "per_seed": [1.0, 2.0],
        "mean": 1.5,
        "mc_standard_error": 0.5,
    }
    other = copy.deepcopy(manifest)
    other["observations_sha256"] = "different"
    with pytest.raises(ValueError, match="observations_sha256"):
        fit_comparison(fixed, staged, manifest, other)
