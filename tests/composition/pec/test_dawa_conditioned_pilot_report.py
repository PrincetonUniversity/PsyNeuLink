import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


DIRECTORY = Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile/dawa"
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location(
    "dawa_conditioned_pilot_report", DIRECTORY / "dawa_conditioned_pilot_report.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def example():
    # Large seed-specific shifts cancel in paired comparisons. Deliberately put
    # records in a different order than the declared seeds and candidates.
    rows = []
    for budget, offset in [(100, 0.5), (1000, 0.0)]:
        for seed, shift in [(3, 30.0), (1, 10.0), (2, -10.0)]:
            for label, delta in [("profile", 2.0), ("fit_0", 0.0)]:
                rows.append(
                    {
                        "estimates": budget,
                        "seed": seed,
                        "label": label,
                        "selected_log_score": shift + offset + delta,
                        "minimum_ess": budget / 10,
                        "median_ess": budget / 2,
                        "contamination_above_half": 1,
                    }
                )
    return {
        "status": "complete",
        "config": {
            "validation_seeds": [1, 2, 3],
            "validation_estimates": [100, 1000],
            "anchor": {"label": "fit_0"},
        },
        "candidates": [{"label": "fit_0"}, {"label": "profile"}],
        "validation": rows,
        "environment": {},
        "points": [],
        "invalid_proposals": [],
        "total_seconds": 1.0,
        "source_sha256": {},
    }


def test_profile_report_pairs_complete_scores_by_seed():
    result = report.profile_summary(example())
    assert result["seed_score_matrices"]["1000"] == [
        [10.0, 12.0],
        [-10.0, -8.0],
        [30.0, 32.0],
    ]
    for row in result["summary"]:
        if row["label"] == "profile":
            assert row["difference_from_anchor"]["mean"] == 2.0
            assert row["difference_from_anchor"]["sd"] == 0.0
            assert row["score"]["sd"] > 10.0
    for row in result["particle_budget_drift"]:
        assert row["absolute_score_difference"]["mean"] == 0.5
        assert row["anchor_comparison_difference"]["mean"] == 0.0


@pytest.mark.parametrize("damage", ["missing", "duplicate", "nonfinite"])
def test_profile_report_rejects_unusable_replication(damage):
    raw = example()
    if damage == "missing":
        raw["validation"].pop()
    elif damage == "duplicate":
        raw["validation"].append(raw["validation"][0])
    else:
        raw["validation"][0]["selected_log_score"] = np.nan
    with pytest.raises(ValueError):
        report.profile_summary(raw)


def test_timing_comparison_requires_same_workload_and_sources():
    raw = {
        "config": {
            key: 1
            for key in (
                "data_sha256",
                "candidates",
                "histogram",
                "max_steps",
                "batch_size",
                "launch",
                "execution",
            )
        },
        "implementation_sha256": {"solver.py": "abc"},
        "environment": {},
        "runtime": {
            "budgets": [{"estimates": 100, "median_seconds_per_candidate": 2.0}],
            "note": "",
            "batches": [],
        },
    }
    raw["config"]["seeds"] = [1, 2, 3]
    other = copy.deepcopy(raw)
    other["runtime"]["budgets"][0]["median_seconds_per_candidate"] = 0.5
    assert report.timing_comparison(raw, other)["budgets"][0]["speedup"] == 4.0
    other["config"]["histogram"] = 2
    with pytest.raises(ValueError, match="histogram"):
        report.timing_comparison(raw, other)
    other["config"]["histogram"] = 1
    other["implementation_sha256"]["solver.py"] = "changed"
    with pytest.raises(ValueError, match="sources"):
        report.timing_comparison(raw, other)


def test_recovery_comparison_pairs_seeds_and_rejects_incompatible_runs():
    left = {
        "label": "left",
        "validation_estimates": 1000,
        "independent_seed_rescoring": [
            {"seed": s, "fitted": v} for s, v in [(1, 100.0), (2, 110.0), (3, 140.0)]
        ],
    }
    right = {
        "label": "right",
        "validation_estimates": 1000,
        "independent_seed_rescoring": [
            {"seed": s, "fitted": v} for s, v in [(3, 137.0), (1, 99.0), (2, 108.0)]
        ],
    }
    result = report.recovery_comparisons([left, right])[0]["left_minus_right"]
    assert result["mean"] == 2.0
    assert result["sd"] == 1.0
    right["independent_seed_rescoring"].append(right["independent_seed_rescoring"][0])
    with pytest.raises(ValueError, match="unique seeds"):
        report.recovery_comparisons([left, right])
    right["independent_seed_rescoring"].pop()
    right["validation_estimates"] = 100
    with pytest.raises(ValueError, match="budgets differ"):
        report.recovery_comparisons([left, right])
