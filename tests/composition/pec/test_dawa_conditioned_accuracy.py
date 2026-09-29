"""Independent-seed analysis and resume integrity for the DAWA accuracy study."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

DIRECTORY = Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile/dawa"
sys.path.insert(0, str(DIRECTORY))
from dawa_conditioned_accuracy import (  # noqa: E402
    SCHEMA_VERSION, analyze_records, diagnostic_groups, load_candidates, main, mean_statistics, validate_resume,
)


def test_mean_interval_uses_seed_sd_and_student_t():
    result = mean_statistics([1., 2., 3.])
    assert result["count"] == 3
    assert result["mean"] == 2.
    assert result["sd"] == 1.
    assert result["se"] == pytest.approx(1 / np.sqrt(3))
    # 0.975 quantile of Student-t with two degrees of freedom.
    half_width = 4.302652729749462 / np.sqrt(3)
    assert result["mean_ci95"] == pytest.approx([2. - half_width, 2. + half_width])
    assert mean_statistics([3.])["mean_ci95"] is None


def test_candidate_and_budget_differences_pair_seeds_and_cancel_common_variation():
    candidates = {"reference": "a", "candidates": [{"label": "a"}, {"label": "b"}]}
    records = []
    for estimates in (100, 400):
        for seed, base in zip((11, 22, 33), (10., 30., -20.), strict=True):
            if estimates == 400 and seed == 33:
                continue
            for label in ("a", "b"):
                value = base + (1. if estimates == 400 else 0.)
                if label == "b":
                    value += .1 if estimates == 100 else .2
                records.append({"estimates": estimates, "seed": seed, "label": label,
                                "selected_log_score": value, "full_log_score": value + 5.})
    # Pair by identifiers, never insertion order or independent variances.
    result = analyze_records(list(reversed(records)), candidates, [100, 400])
    small = next(row for row in result["candidates"] if row["label"] == "b" and row["estimates"] == 100)
    assert small["selected_log_score"]["sd"] > 20.
    paired = small["difference_from_reference_candidate"]["selected_log_score"]
    assert paired["count"] == 3 and paired["seeds"] == [11, 22, 33]
    assert paired["mean"] == pytest.approx(.1)
    assert paired["sd"] < 1.e-12
    absolute_drift = small["difference_from_largest_budget"]["selected_log_score"]
    assert absolute_drift["seeds"] == [11, 22]
    assert absolute_drift["mean"] == pytest.approx(-1.1)
    assert absolute_drift["mean_ci_within_tolerance"] is False
    contrast_drift = small["reference_candidate_contrast_drift"]["selected_log_score"]
    assert contrast_drift["mean"] == pytest.approx(-.1)
    assert contrast_drift["mean_ci_within_tolerance"] is True
    assert "not ground truth" in result["budget_note"]
    assert small["selected_log_score"]["log_mean_exp"] > small["selected_log_score"]["mean"]
    with pytest.raises(ValueError, match="Duplicate"):
        analyze_records(records + records[:1], candidates, [100, 400])


def test_mask_diagnostics_separate_history_updates_from_scored_trials():
    groups = diagnostic_groups(np.array([5., 150., 20.]), np.array([.9, .2, .7]),
                               np.array([False, True, True]), 200)
    assert groups["all"]["trials_with_ess_below_100"] == 2
    assert groups["history_only"]["trials_with_contamination_above_half"] == 1
    assert groups["scored"]["trials_with_contamination_above_half"] == 1
    assert groups["history_only"]["ess_fraction"]["minimum"] == .025
    assert diagnostic_groups(np.ones(2), np.zeros(2), np.ones(2, dtype=bool), 1)["history_only"] == {"trials": 0}


def test_resume_validates_provenance_and_unique_finite_records():
    config = {"estimates": [100], "seeds": [11], "candidates": {"candidates": [{"label": "a"}]}}
    hashes, environment = {"model.py": "abc"}, {"gpu": "test"}
    report = {"schema_version": SCHEMA_VERSION, "config": config, "implementation_sha256": hashes,
              "environment": environment, "records": [{"estimates": 100, "seed": 11, "label": "a",
                                                        "selected_log_score": 1., "full_log_score": 2., "device_log_score": 1.}]}
    assert validate_resume(report, config, hashes, environment) == {(100, 11, "a")}
    for key in ("config", "implementation_sha256", "environment"):
        changed = deepcopy(report)
        changed[key]["changed"] = True
        with pytest.raises(ValueError, match=f"{key} differs"):
            validate_resume(changed, config, hashes, environment)
    duplicated = deepcopy(report)
    duplicated["records"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        validate_resume(duplicated, config, hashes, environment)


def test_candidate_manifest_preserves_labels_parameter_order_and_selection_note(tmp_path):
    payload = {"reference": "anchor", "parameter_order": list("abcdefgh"), "selection_note": "not an optimum",
               "candidates": [{"label": "anchor", "parameters": list(range(8))}]}
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(payload))
    assert load_candidates(path) == payload
    payload["candidates"].append(payload["candidates"][0])
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unique"):
        load_candidates(path)


@pytest.mark.parametrize("extra,error", [
    (["--seeds", "1", "--repeats", "2"], "not both"),
    (["--repeats", "0"], "Repeats must be positive"),
    (["--estimates", "100", "100"], "duplicates"),
])
def test_invalid_study_arguments_fail_before_gpu_setup(tmp_path, capsys, extra, error):
    with pytest.raises(SystemExit, match="2"):
        main(["--output", str(tmp_path / "result.json"), *extra])
    assert error in capsys.readouterr().err
    assert not (tmp_path / "result.json").exists()
