"""Data selection and real-data/recovery separation in the DAWA fitting handoff."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

DIRECTORY = Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile/dawa"
sys.path.insert(0, str(DIRECTORY))
from dawa_pec_fit import load_subject  # noqa: E402


@pytest.fixture
def design():
    return pd.DataFrame({
        "subject_nr": [7, 42, 42, 42, 42, 42],
        "row_id": [99, 20, 8, 3, 16, 1],
        "PrevCongruency": [0., np.nan, 0., 1., 0., 1.],
        "likelihood_include_mask": [1, 1, 0, 1, 1, 1],
        "T1": [1.] * 6, "T2": [0.] * 6,
        "S1": [1.] * 6, "S2": [0.] * 6, "S3": [0.] * 6, "S4": [1.] * 6,
        "decision": [1., 0., 1., 0., 1., 0.],
        "response_time": [.51, .62, .73, .84, .95, 1.06],
    })


def test_empirical_selection_keeps_recorded_responses_order_and_masked_history(tmp_path, design):
    path = tmp_path / "data.csv"
    design.to_csv(path, index=False)
    actual = load_subject(path, 42)
    assert actual.row_id.tolist() == [8, 3, 16, 1]
    assert actual.likelihood_include_mask.tolist() == [False, True, True, True]
    np.testing.assert_array_equal(actual[["decision", "response_time"]], design.iloc[2:][["decision", "response_time"]])


def test_recovery_only_needs_design_and_drops_empirical_outcomes(tmp_path, design):
    path = tmp_path / "design.csv"
    design.to_csv(path, index=False)
    actual = load_subject(path, 42, recovery=True)
    assert "decision" not in actual and "response_time" not in actual
    design.drop(columns=["decision", "response_time"]).to_csv(path, index=False)
    pd.testing.assert_frame_equal(actual, load_subject(path, 42, recovery=True))


@pytest.mark.parametrize("column,value,error", [
    ("likelihood_include_mask", 2, "only 0/1"),
    ("likelihood_include_mask", np.nan, "finite values"),
    ("response_time", 600., "0–3 s"),
    ("decision", -1., "decision must be 0 or 1"),
    ("S1", np.nan, "finite values"),
])
def test_bad_data_rejected_before_fitting(tmp_path, design, column, value, error):
    design.loc[3, column] = value
    path = tmp_path / "bad.csv"
    design.to_csv(path, index=False)
    with pytest.raises(ValueError, match=error):
        load_subject(path, 42)


def test_masked_rt_outside_histogram_retains_history(tmp_path, design):
    design.loc[2, "response_time"] = 4.
    path = tmp_path / "data.csv"
    design.to_csv(path, index=False)
    assert load_subject(path, 42).response_time.iloc[0] == 4.


def test_missing_subject_and_unscored_condition_are_explicit_errors(tmp_path, design):
    path = tmp_path / "data.csv"
    design.to_csv(path, index=False)
    with pytest.raises(ValueError, match="subject_nr=8"):
        load_subject(path, 8)
    with pytest.raises(ValueError, match="scored trial.*each"):
        load_subject(path, 42, trials=2)


@pytest.mark.triton
@pytest.mark.triton_gpu
@pytest.mark.batched
@pytest.mark.parametrize("recovery", [False, True])
def test_gpu_cli_fits_empirical_or_generated_observations(tmp_path, design, recovery):
    path = tmp_path / "data.csv"
    design.to_csv(path, index=False)
    output = tmp_path / "result"
    script = "dawa_pec_recovery.py" if recovery else "dawa_pec_fit.py"
    command = [sys.executable, str(DIRECTORY / script), "--data", str(path), "--subject", "42",
               "--estimates", "64", "--evaluations", "11", "--predictive-estimates", "16",
               "--validation-seeds", "8101", "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180, cwd=DIRECTORY.parents[3])
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    report = json.loads((output / ("recovery.json" if recovery else "fit.json")).read_text())
    observations = pd.read_csv(output / ("synthetic_subject.csv" if recovery else "observed_subject.csv"))
    assert report["status"] == manifest["status"] == "complete"
    assert report["evaluations"] == 11
    assert manifest["trials"] == 4 and manifest["scored_trials"] == 3
    assert observations.row_id.tolist() == [8, 3, 16, 1]
    assert len(report["fitted"]) == 8
    assert set(report["predictive_summaries"]["fitted"]) == {"0", "1"}
    if recovery:
        assert manifest["generator_matches_pec_exactly"]
        assert "truth" in report and "errors" in report
        assert not np.array_equal(observations.response_time, design.iloc[2:].response_time)
    else:
        assert report["mode"] == "empirical_fit"
        assert "truth" not in report and "errors" not in report
        assert "generator_matches_pec_exactly" not in manifest
        np.testing.assert_array_equal(observations[["decision", "response_time"]],
                                      design.iloc[2:][["decision", "response_time"]])
        assert "fitted_minus_initial" in report["independent_seed_rescoring"][0]
