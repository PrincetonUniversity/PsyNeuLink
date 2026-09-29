"""Statistical and provenance checks for the compact accuracy-study report."""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pytest

DIRECTORY = Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile/dawa"
sys.path.insert(0, str(DIRECTORY))
from dawa_conditioned_accuracy import diagnostic_groups  # noqa: E402
from dawa_conditioned_accuracy_report import (  # noqa: E402
    accuracy_analysis, compact_report, main, runtime_summary, trial_variability_summary,
)


def test_runtime_excludes_each_budget_sessions_first_call_and_handles_short_batches():
    batches = []
    for index, (session, budget, count, seconds) in enumerate([
        (0, 100, 4, 100.), (0, 100, 2, 6.), (0, 200, 4, 200.),
        (0, 200, 4, 20.), (1, 100, 4, 80.), (1, 100, 2, 10.),
    ]):
        batches.append({"id": str(index), "session": session, "estimates": budget, "seed": index,
                        "labels": list("abcd")[:count], "seconds_including_diagnostics": seconds})
    result = runtime_summary(batches, [100, 200, 400])
    first, second, empty = result["budgets"]
    assert first["excluded_batches"] == ["0", "4"]
    assert first["included_batches"] == 2
    assert first["median_seconds_per_candidate"] == 4.
    assert first["aggregate_seconds_per_candidate"] == 4.
    assert second["median_seconds_per_candidate"] == 5.
    assert empty["included_batches"] == 0


def test_compact_report_recomputes_paired_statistics_and_preserves_missing_matrix_cells():
    raw = {
        "status": "running", "git_commit": "commit", "environment": {"gpu": "test"},
        "implementation_sha256": {"model.py": "sourcehash"},
        "config": {"estimates": [100, 200], "seeds": [22, 11], "agreement_tolerance": .2,
                   "diagnostics_directory": "/private/raw/arrays", "candidates": {
                       "reference": "a", "candidates": [{"label": label} for label in ("a", "b", "stress_case")],
                   }}, "records": [], "batches": [], "analysis": {"intentionally": "stale"},
    }
    for budget in (100, 200):
        for seed, value in ((11, 1.), (22, 10.)):
            for label, delta in (("a", 0.), ("b", .1), ("stress_case", -5.)):
                if budget == 200 and seed == 22 and label == "stress_case":
                    continue
                raw["records"].append({
                    "estimates": budget, "seed": seed, "label": label,
                    "selected_log_score": value + delta, "full_log_score": value + delta - 2.,
                    "device_log_score": value + delta + 1.e-5,
                    "diagnostic_groups": diagnostic_groups(np.array([5., 80.]), np.array([.9, .2]),
                                                           np.array([False, True]), budget),
                })
    result = compact_report(raw)
    assert result["complete"] is False
    assert result["completed_records"] == 11 and result["expected_records"] == 12
    assert "diagnostics_directory" not in result["config"]
    assert result["figure"]["contrast_labels"] == ["b"]
    assert result["implementation_sha256"] == raw["implementation_sha256"]
    analysis_source = Path(accuracy_analysis.__file__).resolve()
    assert result["analysis_provenance"] == {
        "module": accuracy_analysis.__name__, "path": str(analysis_source),
        "sha256": hashlib.sha256(analysis_source.read_bytes()).hexdigest(),
    }
    matrix = result["seed_score_matrices"][1]
    assert matrix["seeds"] == [22, 11]
    assert matrix["selected_log_score"][0] == [10., 10.1, None]
    comparison = next(row for row in result["analysis"]["candidates"]
                      if row["estimates"] == 100 and row["label"] == "b")
    paired = comparison["difference_from_reference_candidate"]["selected_log_score"]
    assert paired["mean"] == pytest.approx(.1)
    assert paired["sd"] < 1.e-12
    assert comparison["selected_log_score"]["sd"] > 6.
    support = result["support_diagnostics"][0]
    assert support["groups"]["scored"]["trials_with_contamination_above_half"]["maximum"] == 0
    assert support["groups"]["history_only"]["trials_with_contamination_above_half"]["maximum"] == 1
    assert support["maximum_abs_device_selected_score_difference"] == pytest.approx(1.e-5)
    with pytest.raises(ValueError, match="non-reference"):
        compact_report(raw, contrast_labels=["a"])


def test_existing_render_outputs_require_explicit_overwrite(tmp_path, capsys):
    existing = tmp_path / "result.png"
    existing.write_bytes(b"original")
    with pytest.raises(SystemExit, match="2"):
        main(["--input", str(tmp_path / "missing.json"), "--output", str(tmp_path / "result")])
    assert "--overwrite" in capsys.readouterr().err
    assert existing.read_bytes() == b"original"
    assert not (tmp_path / "result.json").exists()


@pytest.fixture
def saved_trial_diagnostics(tmp_path):
    mask = np.array([True, True, False, False])
    raw = {"config": {"candidates": {"reference": "anchor", "candidates": [{"label": "anchor"}]},
                      "seeds": [11, 22, 33], "estimates": [200, 400], "retained_trials": 4, "scored_trials": 2},
           "records": [], "batches": []}
    for seed, value in zip((11, 22, 33), (-1., 0., 1.), strict=True):
        path = tmp_path / f"{seed}.npz"
        densities = np.exp(np.array([value, -value, 2 * value, 2 * value]))[None, None]
        np.savez_compressed(path, labels=np.array(["anchor"]), include_mask=mask,
                            observations=np.arange(8).reshape(4, 2), per_trial_densities=densities,
                            effective_sample_size=np.array([[[50., 150., 30., 180.]]]),
                            prior_mixture_fraction=np.array([[[.1, .7, .999, .4]]]),
                            zero_support=np.zeros((1, 1, 4), dtype=bool))
        logs = np.log(densities[0, 0])
        raw["records"].append({"estimates": 200, "seed": seed, "label": "anchor",
                               "selected_log_score": float(logs[mask].sum()), "full_log_score": float(logs.sum())})
        raw["batches"].append({"id": f"seed{seed}", "estimates": 200, "seed": seed,
                               "labels": ["anchor"], "diagnostics_file": str(path)})
    # An incomplete budget must be skipped without trying to load its missing artifact.
    raw["records"].append({**raw["records"][0], "estimates": 400})
    raw["batches"].append({**raw["batches"][0], "estimates": 400, "diagnostics_file": None})
    return raw


def test_trial_variances_distinguish_covariance_from_concentration(saved_trial_diagnostics, tmp_path):
    summary = trial_variability_summary(saved_trial_diagnostics)
    assert summary["skipped"] == [{"estimates": 400, "label": "anchor", "reason": "incomplete_seed_set",
                                   "missing_seeds": [22, 33]}]
    assert len(summary["summaries"]) == 1
    row = summary["summaries"][0]
    assert row["replicates"] == 3
    scored, history = row["groups"]["scored"], row["groups"]["history_only"]
    assert scored["sum_per_trial_variances"] == pytest.approx(2.)
    assert scored["total_score_variance"] == pytest.approx(0.)
    assert scored["covariance_contribution_to_score_variance"] == pytest.approx(-2.)
    assert scored["median_per_trial_log_density_sd"] == pytest.approx(1.)
    assert scored["top_trial_variance_share"] == pytest.approx({"1": .5, "5": 1., "10": 1.})
    assert history["sum_per_trial_variances"] == pytest.approx(8.)
    assert history["total_score_variance"] == pytest.approx(16.)
    assert history["covariance_contribution_to_score_variance"] == pytest.approx(8.)
    assert history["trials_with_median_contamination_above_99pct"] == 1
    assert scored["trials_with_median_contamination_above_99pct"] == 0
    assert scored["trials_with_median_ess_below_100"] == history["trials_with_median_ess_below_100"] == 1
    for entry, batch in zip(summary["diagnostic_file_sha256"], saved_trial_diagnostics["batches"][:3], strict=True):
        assert entry == {"batch_id": batch["id"], "sha256": hashlib.sha256(Path(batch["diagnostics_file"]).read_bytes()).hexdigest()}
    serialized = json.dumps(summary)
    assert str(tmp_path) not in serialized
    assert '"observations"' not in serialized
    assert '"include_mask"' not in serialized


@pytest.mark.parametrize("changed,error", [("labels", "labels"), ("mask", "masks differ"),
                                           ("shape", "shapes"), ("scores", "recorded scores")])
def test_trial_summary_rejects_mismatched_saved_arrays(saved_trial_diagnostics, changed, error):
    path = Path(saved_trial_diagnostics["batches"][1]["diagnostics_file"])
    with np.load(path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if changed == "labels":
        arrays["labels"] = np.array(["wrong"])
    elif changed == "mask":
        arrays["include_mask"] = np.array([True, False, True, False])
    elif changed == "shape":
        arrays["effective_sample_size"] = arrays["effective_sample_size"][:, 0]
    else:
        arrays["per_trial_densities"] *= 2.
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match=error):
        trial_variability_summary(saved_trial_diagnostics)
