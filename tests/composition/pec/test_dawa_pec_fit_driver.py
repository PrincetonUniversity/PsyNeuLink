"""Data selection and real-data/recovery separation in the DAWA fitting handoff."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

DIRECTORY = Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile/dawa"
sys.path.insert(0, str(DIRECTORY))
from dawa_pec_fit import (  # noqa: E402
    LAUNCH, STARTS, build_model, histogram_settings, load_subject, main, make_fit_pec, node,
)


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
    frame = load_subject(path, 42)
    assert frame.response_time.iloc[0] == 4.
    bins, rt_range = histogram_settings(frame, "conditioned")
    assert bins == 134
    assert rt_range == pytest.approx((0., 4.02))
    assert (rt_range[1] - rt_range[0]) / bins == pytest.approx(.03)
    assert histogram_settings(frame, "marginal") == (100, (0., 3.))


@pytest.mark.parametrize("extra", [["--fit-strategy", "adaptive"], ["--profile-ndt"]])
def test_conditioned_rejects_unsupported_shortcuts_before_creating_output(tmp_path, capsys, extra):
    output = tmp_path / "must_not_exist"
    with pytest.raises(SystemExit, match="2"):
        main(["--data", str(tmp_path / "missing.csv"), "--output", str(output), *extra])
    assert "changes particle history" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("likelihood", ["conditioned", "marginal"])
def test_pec_objective_routes_training_and_rescoring_with_complete_history(
    tmp_path, design, monkeypatch, likelihood,
):
    """Exercise the real PEC closure without compiling/running a GPU kernel."""
    path = tmp_path / "data.csv"
    design.loc[2, "response_time"] = 4.
    design.to_csv(path, index=False)
    frame = load_subject(path, 42)
    model, inputs, outputs = build_model(trials=len(frame), c_noise=.1, s_noise=.1, d_noise=.1, r_noise=.1)
    inputs[node(model, "Task Input")] = frame[["T1", "T2"]].to_numpy()
    inputs[node(model, "Stimulus Input")] = frame[["S1", "S2", "S3", "S4"]].to_numpy()
    args = SimpleNamespace(likelihood=likelihood, evaluations=11, max_steps=2000,
                           simulation_seed=29, pseudocount=.5, population=2, estimates=64)
    pec = make_fit_pec(model, inputs, outputs, frame, args)
    function = pec.controller.function
    assert function.batched_parameter_batch_size == 2
    calls = []

    def record(method):
        def score(received_inputs, parameters, **kwargs):
            calls.append((method, received_inputs, parameters, kwargs))
            return np.arange(len(parameters), dtype=float) + 1.
        return score

    plan = SimpleNamespace(conditioned_log_likelihood=record("conditioned"),
                           log_likelihood=record("marginal"),
                           ir=SimpleNamespace(graph=SimpleNamespace(inputs=[
                               SimpleNamespace(node=mechanism.name) for mechanism in inputs
                           ])))
    monkeypatch.setattr(function, "_compile_batched_plan", lambda: plan)
    monkeypatch.setattr(function, "_batched_outcome_indices", lambda _: [0, 1])
    training = function._make_objective_func()._batched_parameter_sets(STARTS)
    np.testing.assert_array_equal(training, [1., 2.])
    # Rescoring must build the same conditioned closure at the fresh budget and
    # preserve the contamination fraction by scaling alpha with particle count.
    pec.controller.num_estimates = 128
    function.batched_pseudocount = 1.
    function.batched_seed = 8101
    validation = function._make_objective_func()._batched_parameter_sets(STARTS)
    np.testing.assert_array_equal(validation, training)
    for index, (method, received_inputs, parameters, kwargs) in enumerate(calls):
        assert method == likelihood
        assert len(parameters) == 2
        np.testing.assert_array_equal(received_inputs[node(model, "Task Input").name], frame[["T1", "T2"]])
        np.testing.assert_array_equal(kwargs["data"], frame[["decision", "response_time"]])
        np.testing.assert_array_equal(kwargs["include_mask"], [False, True, True, True])
        assert kwargs["strict_truncation"] is True
        assert kwargs["triton_launch_options"] == LAUNCH
        assert kwargs["num_estimates"] == (64, 128)[index]
        assert kwargs["pseudocount"] == (.5, 1.)[index]
        assert kwargs["seed"] == (29, 8101)[index]
        assert kwargs["bins"] == (134 if likelihood == "conditioned" else 100)
        for row, start in zip(parameters, STARTS, strict=True):
            mode = next(value for key, value in row.items() if key.endswith(".mode"))
            np.testing.assert_array_equal(mode.values, [start[4], start[5], start[4], start[5]])


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
    if not recovery:
        command.extend(["--pseudocount", ".5", "--validation-estimates", "128", "--optimizer-storage", "memory"])
    result = subprocess.run(command, capture_output=True, text=True, timeout=180, cwd=DIRECTORY.parents[3])
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    report = json.loads((output / ("recovery.json" if recovery else "fit.json")).read_text())
    observations = pd.read_csv(output / ("synthetic_subject.csv" if recovery else "observed_subject.csv"))
    assert report["status"] == manifest["status"] == "complete"
    assert report["evaluations"] == 11
    assert report["validation_estimates"] == (64 if recovery else 128)
    assert report["validation_pseudocount"] == 1.
    assert manifest["estimator"]["pseudocount"] == (1. if recovery else .5)
    assert manifest["arguments"]["likelihood"] == "conditioned"
    assert manifest["estimator"]["kind"] == "observation_conditioned_particle_histogram"
    assert manifest["estimator"]["masked_observations_condition_history"] is True
    assert report["estimator"] == manifest["estimator"]
    assert (output / "optimizer.journal").exists() == recovery
    assert (output / "optimizer_trials.csv").exists()
    assert (output / "evaluations.jsonl").exists()
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


@pytest.mark.triton
@pytest.mark.triton_gpu
@pytest.mark.batched
@pytest.mark.parametrize("profile_ndt,recovery", [(False, True), (True, True), (True, False)])
def test_gpu_adaptive_recovery_checks_and_refines_on_separate_budget(tmp_path, design, profile_ndt, recovery):
    path = tmp_path / "data.csv"
    design.to_csv(path, index=False)
    output = tmp_path / "adaptive"
    entry = "dawa_pec_recovery.py" if recovery else "dawa_pec_fit.py"
    command = [sys.executable, str(DIRECTORY / entry), "--data", str(path),
               "--subject", "42", "--likelihood", "marginal", "--fit-strategy", "adaptive", "--estimates", "64",
               "--adaptive-min-estimates", "16", "--evaluations", "41", "--adaptive-check-every", "10",
               "--adaptive-min-evaluations", "21", "--adaptive-patience", "1",
               "--adaptive-progress-tolerance", "100000", "--adaptive-refine-evaluations", "10",
               "--validation-estimates", "128", "--validation-seeds", "8101",
               "--predictive-estimates", "16", "--output", str(output)]
    if profile_ndt:
        command.append("--profile-ndt")
    result = subprocess.run(command, capture_output=True, text=True, timeout=180, cwd=DIRECTORY.parents[3])
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / ("recovery.json" if recovery else "fit.json")).read_text())
    records = [json.loads(line) for line in (output / "evaluations.jsonl").read_text().splitlines()]
    assert report["status"] == "complete"
    assert report["evaluations"] == 31 < report["requested_evaluations"]
    assert report["adaptive"]["search_evaluations"] == 21
    assert report["adaptive"]["refinement_evaluations"] == 10
    assert report["adaptive"]["best_reference_score"] >= report["adaptive"]["checkpoints"][-1]["reference_score"]
    selection = report["adaptive"]["final_selection"]
    assert report["best_training_log_likelihood"] == selection["selected_reference_score"]
    assert report["adaptive"]["policy_version"] == 2
    assert report["adaptive"]["refinement_covariance"]["reused"]
    assert report["validation_estimates"] == 128 and report["validation_pseudocount"] == 2.
    assert not (output / "optimizer.journal").exists()
    assert (output / "optimizer_refinement_trials.csv").exists()
    seeds = [seed for row in records for seed in row["block_seeds"]]
    assert not {8101, 20260925, 21260925} & set(seeds)
    assert {row["phase"] for row in records} == {"adaptive_search", "refinement"}
    if profile_ndt:
        assert len(report["ndt_profile"]["optimizer_parameters"]) == 7
        assert report["ndt_profile"]["distinct_bin_maps"] < report["ndt_profile"]["grid_values"]
        assert all(len(row["parameters"]) == 8 for row in records)
        profile = report["adaptive"]["profile"]
        assert report["fitted"][profile["parameter"]] == profile["values"][profile["selection_indices"][selection["winner"]]]
