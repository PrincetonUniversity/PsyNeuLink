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
import dawa_pec_fit as driver  # noqa: E402
from dawa_pec_fit import (  # noqa: E402
    LAUNCH,
    STARTS,
    build_model,
    histogram_settings,
    load_subject,
    main,
    make_fit_pec,
    node,
    recovery_observations,
    summarize_validation,
)


@pytest.fixture
def design():
    return pd.DataFrame(
        {
            "subject_nr": [7, 42, 42, 42, 42, 42],
            "row_id": [99, 20, 8, 3, 16, 1],
            "PrevCongruency": [0.0, np.nan, 0.0, 1.0, 0.0, 1.0],
            "likelihood_include_mask": [1, 1, 0, 1, 1, 1],
            "T1": [1.0] * 6,
            "T2": [0.0] * 6,
            "S1": [1.0] * 6,
            "S2": [0.0] * 6,
            "S3": [0.0] * 6,
            "S4": [1.0] * 6,
            "decision": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "response_time": [0.51, 0.62, 0.73, 0.84, 0.95, 1.06],
        }
    )


def test_empirical_selection_keeps_recorded_responses_order_and_masked_history(
    tmp_path, design
):
    path = tmp_path / "data.csv"
    design.to_csv(path, index=False)
    actual = load_subject(path, 42)
    assert actual.row_id.tolist() == [8, 3, 16, 1]
    assert actual.likelihood_include_mask.tolist() == [False, True, True, True]
    np.testing.assert_array_equal(
        actual[["decision", "response_time"]],
        design.iloc[2:][["decision", "response_time"]],
    )


def test_recovery_only_needs_design_and_drops_empirical_outcomes(tmp_path, design):
    path = tmp_path / "design.csv"
    design.to_csv(path, index=False)
    actual = load_subject(path, 42, recovery=True)
    assert "decision" not in actual and "response_time" not in actual
    design.drop(columns=["decision", "response_time"]).to_csv(path, index=False)
    pd.testing.assert_frame_equal(actual, load_subject(path, 42, recovery=True))


def test_recovery_measurement_law_matches_scoring_and_preserves_latent_history():
    from dawa_conditioned_reference import numpy_observation_density

    latent = np.tile([0.0, 0.005], (50000, 1))
    original = latent.copy()
    observed, metadata = recovery_observations(
        latent,
        observation_model="conditioned",
        seed=77,
        estimates=1000,
        pseudocount=1.0,
        device="cpu",
    )
    np.testing.assert_array_equal(latent, original)
    assert metadata["contamination_fraction"] == pytest.approx(1 / 6)
    edges = np.array(metadata["edges"], dtype=np.float32)
    centers = (edges[:-1].astype(float) + edges[1:]) / 2.0
    # Joint frequencies exercise source-edge normalization and choice contamination.
    for choice, index in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 99)]:
        target = [choice, centers[index]]
        expected = numpy_observation_density(
            original[:1], target, edges=edges, alpha_per_estimate=0.001
        )[0] * float(edges[1] - edges[0])
        count = ((observed[:, 0] == choice) & (observed[:, 1] == centers[index])).sum()
        assert (
            abs(count / len(observed) - expected)
            < 6 * np.sqrt(expected * (1 - expected) / len(observed)) + 1e-5
        )
    # Rescaling alpha and N together must leave generated observations identical.
    other, _ = recovery_observations(
        latent,
        observation_model="conditioned",
        seed=77,
        estimates=2000,
        pseudocount=2.0,
        device="cpu",
    )
    np.testing.assert_array_equal(observed, other)
    raw, metadata = recovery_observations(
        latent,
        observation_model="latent",
        seed=77,
        estimates=1000,
        pseudocount=1.0,
        device="cpu",
    )
    np.testing.assert_array_equal(raw, latent)
    assert metadata["kind"] == "latent_choices_rt"


def test_recovery_observation_overflow_is_not_silently_discarded():
    with pytest.raises(ValueError, match="outside the observation domain"):
        recovery_observations(
            [[0.0, 4.0]],
            observation_model="conditioned",
            seed=77,
            estimates=100000,
            pseudocount=1.0,
            device="cpu",
        )


def test_recovery_rejects_reused_observation_seed_before_setup(tmp_path, capsys):
    with pytest.raises(SystemExit, match="2"):
        main(
            ["--output", str(tmp_path / "result"), "--observation-seed", "29"],
            recovery=True,
        )
    assert "seeds must be independent" in capsys.readouterr().err
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize(
    "options,recovery",
    [
        (["--validation-seeds", "44", "44"], False),
        (["--validation-seeds", "21260925"], True),
        (["--simulation-seed", "21260925"], True),
    ],
)
def test_invalid_validation_or_predictive_seeds_rejected_before_setup(
    tmp_path, capsys, options, recovery
):
    output = tmp_path / "result"
    with pytest.raises(SystemExit, match="2"):
        main(["--output", str(output), *options], recovery=recovery)
    assert "seeds must be" in capsys.readouterr().err
    assert not output.exists()


def test_setup_failure_records_phase_and_preserves_output(
    tmp_path, design, monkeypatch
):
    path = tmp_path / "data.csv"
    design.to_csv(path, index=False)
    output = tmp_path / "failed"
    monkeypatch.setattr(driver.torch.cuda, "is_available", lambda: True)

    def broken_model(**kwargs):
        raise RuntimeError("setup failure for test")

    monkeypatch.setattr(driver, "build_model", broken_model)
    with pytest.raises(RuntimeError, match="setup failure for test"):
        main(["--data", str(path), "--subject", "42", "--output", str(output)])
    saved = (output / "manifest.json").read_bytes()
    manifest = json.loads(saved)
    assert manifest["status"] == "failed" and manifest["failed_phase"] == "preparing"
    assert "setup failure for test" in manifest["error"]
    assert manifest["arguments"]["optimizer_storage"] == "memory"
    with pytest.raises(FileExistsError):
        main(["--data", str(path), "--subject", "42", "--output", str(output)])
    assert (output / "manifest.json").read_bytes() == saved


def test_validation_summary_uses_paired_scores_and_independent_repetitions():
    rows = [
        {
            "seed": i,
            "initial": base,
            "fitted": base + difference,
            "fitted_minus_initial": difference,
        }
        for i, (base, difference) in enumerate(
            [(100.0, 1.0), (200.0, 2.0), (300.0, 3.0)]
        )
    ]
    summary = summarize_validation(rows)
    assert summary["replicates"] == 3
    paired = summary["statistics"]["fitted_minus_initial"]
    assert paired["mean"] == 2.0
    assert paired["sd"] == 1.0
    assert paired["mc_standard_error"] == pytest.approx(1 / np.sqrt(3))
    assert summary["statistics"]["fitted"]["sd"] == 101.0
    single = summarize_validation(rows[:1])["statistics"]["fitted"]
    assert single["sd"] is None and single["mc_standard_error"] is None
    with pytest.raises(ValueError, match="distinct seeds"):
        summarize_validation([rows[0], rows[0]])
    with pytest.raises(ValueError, match="match across seeds"):
        summarize_validation([rows[0], {"seed": 8, "fitted": 101.0}])
    with pytest.raises(ValueError, match="finite"):
        summarize_validation([{"seed": 8, "fitted": float("nan")}])


@pytest.mark.parametrize(
    "column,value,error",
    [
        ("likelihood_include_mask", 2, "only 0/1"),
        ("likelihood_include_mask", np.nan, "finite values"),
        ("response_time", 600.0, "0–3 s"),
        ("decision", -1.0, "decision must be 0 or 1"),
        ("S1", np.nan, "finite values"),
    ],
)
def test_bad_data_rejected_before_fitting(tmp_path, design, column, value, error):
    design.loc[3, column] = value
    path = tmp_path / "bad.csv"
    design.to_csv(path, index=False)
    with pytest.raises(ValueError, match=error):
        load_subject(path, 42)


def test_masked_rt_outside_histogram_retains_history(tmp_path, design):
    design.loc[2, "response_time"] = 4.0
    path = tmp_path / "data.csv"
    design.to_csv(path, index=False)
    frame = load_subject(path, 42)
    assert frame.response_time.iloc[0] == 4.0
    bins, rt_range = histogram_settings(frame, "conditioned")
    assert bins == 134
    assert rt_range == pytest.approx((0.0, 4.02))
    assert (rt_range[1] - rt_range[0]) / bins == pytest.approx(0.03)
    assert histogram_settings(frame, "marginal") == (100, (0.0, 3.0))


@pytest.mark.parametrize(
    "extra", [["--profile-ndt"], ["--fit-strategy", "adaptive", "--profile-ndt"]]
)
def test_conditioned_rejects_unsupported_shortcuts_before_creating_output(
    tmp_path, capsys, extra
):
    output = tmp_path / "must_not_exist"
    with pytest.raises(SystemExit, match="2"):
        main(["--data", str(tmp_path / "missing.csv"), "--output", str(output), *extra])
    assert "changes particle history" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("likelihood", ["conditioned", "marginal"])
def test_pec_objective_routes_training_and_rescoring_with_complete_history(
    tmp_path,
    design,
    monkeypatch,
    likelihood,
):
    """Exercise the real PEC closure without compiling/running a GPU kernel."""
    path = tmp_path / "data.csv"
    design.loc[2, "response_time"] = 4.0
    design.to_csv(path, index=False)
    frame = load_subject(path, 42)
    model, inputs, outputs = build_model(
        trials=len(frame), c_noise=0.1, s_noise=0.1, d_noise=0.1, r_noise=0.1
    )
    inputs[node(model, "Task Input")] = frame[["T1", "T2"]].to_numpy()
    inputs[node(model, "Stimulus Input")] = frame[["S1", "S2", "S3", "S4"]].to_numpy()
    args = SimpleNamespace(
        likelihood=likelihood,
        evaluations=11,
        max_steps=2000,
        simulation_seed=29,
        pseudocount=0.5,
        population=2,
        estimates=64,
    )
    pec = make_fit_pec(model, inputs, outputs, frame, args)
    function = pec.controller.function
    assert function.batched_parameter_batch_size == 2
    calls = []

    def record(method):
        def score(received_inputs, parameters, **kwargs):
            calls.append((method, received_inputs, parameters, kwargs))
            return np.arange(len(parameters), dtype=float) + 1.0

        return score

    plan = SimpleNamespace(
        conditioned_log_likelihood=record("conditioned"),
        log_likelihood=record("marginal"),
        ir=SimpleNamespace(
            graph=SimpleNamespace(
                inputs=[SimpleNamespace(node=mechanism.name) for mechanism in inputs]
            )
        ),
    )
    monkeypatch.setattr(function, "_compile_batched_plan", lambda: plan)
    monkeypatch.setattr(function, "_batched_outcome_indices", lambda _: [0, 1])
    training = pec.log_likelihood_batch(STARTS, inputs=inputs)
    np.testing.assert_array_equal(training, [1.0, 2.0])
    # Public rescoring changes precision while preserving the observation law.
    validation = pec.log_likelihood_batch(
        STARTS, inputs=inputs, num_estimates=128, seed=8101
    )
    np.testing.assert_array_equal(validation, training)
    if likelihood == "conditioned":
        scores = pec.log_likelihood_batch(
            STARTS, inputs=inputs, num_estimates=32, seed=42
        )
        np.testing.assert_array_equal(scores, training)
    assert (
        pec.controller.num_estimates,
        function.batched_seed,
        function.batched_pseudocount,
    ) == (64, 29, 0.5)
    for index, (method, received_inputs, parameters, kwargs) in enumerate(calls):
        assert method == likelihood
        assert len(parameters) == 2
        np.testing.assert_array_equal(
            np.asarray(received_inputs[node(model, "Task Input").name]).reshape(
                len(frame), 2
            ),
            frame[["T1", "T2"]],
        )
        np.testing.assert_array_equal(
            kwargs["data"], frame[["decision", "response_time"]]
        )
        np.testing.assert_array_equal(kwargs["include_mask"], [False, True, True, True])
        assert kwargs["strict_truncation"] is True
        assert kwargs["triton_launch_options"] == LAUNCH
        assert kwargs["num_estimates"] == (64, 128, 32)[index]
        assert kwargs["pseudocount"] == (0.5, 1.0, 0.25)[index]
        assert kwargs["seed"] == (29, 8101, 42)[index]
        assert kwargs["bins"] == (134 if likelihood == "conditioned" else 100)
        for row, start in zip(parameters, STARTS, strict=True):
            mode = next(value for key, value in row.items() if key.endswith(".mode"))
            np.testing.assert_array_equal(
                mode.values, [start[4], start[5], start[4], start[5]]
            )

    calls.clear()
    adaptive = pec.log_likelihood_batch(
        STARTS,
        inputs=inputs,
        adaptive=True,
        seed=8103,
        adaptive_options=dict(min_estimates=16, repeats=2, reference_index=0),
    )
    np.testing.assert_array_equal(adaptive.log_likelihood, training)
    assert adaptive.num_estimates == 16 and adaptive.converged
    assert len(calls) == 4
    for method, _, parameters, kwargs in calls:
        assert method == likelihood
        assert kwargs["num_estimates"] == 16
        assert kwargs["pseudocount"] == 0.125
        np.testing.assert_array_equal(kwargs["include_mask"], [False, True, True, True])
        for row, start in zip(parameters, STARTS, strict=True):
            mode = next(value for key, value in row.items() if key.endswith(".mode"))
            np.testing.assert_array_equal(
                mode.values, [start[4], start[5], start[4], start[5]]
            )


@pytest.mark.triton_gpu
@pytest.mark.batched
def test_gpu_adaptive_likelihood_replays_dawa_with_noise_in_every_layer(
    tmp_path, design
):
    path = tmp_path / "subject.csv"
    design.to_csv(path, index=False)
    frame = load_subject(path, 42)
    model, inputs, outputs = build_model(
        trials=len(frame), c_noise=0.1, s_noise=0.1, d_noise=0.1, r_noise=0.1
    )
    inputs[node(model, "Task Input")] = frame[["T1", "T2"]].to_numpy()
    inputs[node(model, "Stimulus Input")] = frame[["S1", "S2", "S3", "S4"]].to_numpy()
    args = SimpleNamespace(
        likelihood="conditioned",
        evaluations=11,
        max_steps=2000,
        simulation_seed=29,
        pseudocount=0.5,
        population=2,
        estimates=64,
    )
    pec = make_fit_pec(model, inputs, outputs, frame, args)
    result = pec.log_likelihood_batch(
        STARTS,
        inputs=inputs,
        adaptive=True,
        seed=8103,
        adaptive_options=dict(
            min_estimates=16, repeats=2, target_se=1e6, reference_index=0
        ),
    )
    replay = np.asarray(
        [
            pec.log_likelihood_batch(
                STARTS, inputs=inputs, num_estimates=result.num_estimates, seed=seed
            )
            for seed in result.seeds
        ]
    )
    np.testing.assert_array_equal(result.replicate_log_likelihoods, replay)
    np.testing.assert_array_equal(result.log_likelihood, replay.mean(axis=0))
    differences = replay - replay[:, 0, None]
    np.testing.assert_array_equal(
        result.difference_standard_error, differences.std(axis=0, ddof=1) / np.sqrt(2)
    )
    assert pec.controller.num_estimates == 64
    assert pec.controller.function.batched_seed == 29


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
    command = [
        sys.executable,
        str(DIRECTORY / script),
        "--data",
        str(path),
        "--subject",
        "42",
        "--estimates",
        "64",
        "--evaluations",
        "11",
        "--predictive-estimates",
        "16",
        "--validation-seeds",
        "8101",
        "--output",
        str(output),
    ]
    if not recovery:
        command.extend(
            [
                "--pseudocount",
                ".5",
                "--validation-estimates",
                "128",
                "--optimizer-storage",
                "memory",
            ]
        )
    else:
        command.extend(
            ["--optimizer-storage", "journal", "--validation-estimates", "64"]
        )
    # Source provenance must refer to this checkout even when launched elsewhere.
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=180, cwd=tmp_path
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    report = json.loads(
        (output / ("recovery.json" if recovery else "fit.json")).read_text()
    )
    observations = pd.read_csv(
        output / ("synthetic_subject.csv" if recovery else "observed_subject.csv")
    )
    assert report["status"] == manifest["status"] == "complete"
    assert report["evaluations"] == 11
    assert report["validation_estimates"] == (64 if recovery else 128)
    assert report["validation_pseudocount"] == 1.0
    assert manifest["estimator"]["pseudocount"] == (1.0 if recovery else 0.5)
    assert report["fit_strategy"] == manifest["fit_strategy"] == "fixed"
    assert report["fit_policy"] == manifest["fit_policy"] == "fixed"
    assert manifest["arguments"]["likelihood"] == "conditioned"
    assert manifest["estimator"]["kind"] == "observation_conditioned_particle_histogram"
    assert manifest["estimator"]["masked_observations_condition_history"] is True
    assert report["estimator"] == manifest["estimator"]
    assert (output / "optimizer.journal").exists() == recovery
    assert (output / "optimizer_trials.csv").exists()
    assert (output / "evaluations.jsonl").exists()
    checkpoint = json.loads((output / "fit_checkpoint.json").read_text())
    assert (
        checkpoint["status"] == "search_complete"
        and checkpoint["fitted"] == report["fitted"]
    )
    validation = json.loads((output / "validation.json").read_text())
    assert validation["status"] == "complete"
    assert (
        validation["independent_seed_rescoring"] == report["independent_seed_rescoring"]
    )
    assert report["validation_summary"] == validation["summary"]
    assert validation["summary"]["replicates"] == 1
    assert validation["summary"]["statistics"]["fitted"]["sd"] is None
    assert manifest["trials"] == 4 and manifest["scored_trials"] == 3
    assert observations.row_id.tolist() == [8, 3, 16, 1]
    assert len(report["fitted"]) == 8
    assert set(report["predictive_summaries"]["fitted"]) == {"0", "1"}
    if recovery:
        assert manifest["generator_matches_pec_exactly"]
        assert (
            manifest["synthetic_observation_model"]["kind"]
            == "binned_smoothed_uniform_contamination"
        )
        assert (output / "latent_subject.csv").exists()
        assert "truth" in report and "errors" in report
        assert not np.array_equal(
            observations.response_time, design.iloc[2:].response_time
        )
    else:
        assert report["mode"] == "empirical_fit"
        assert "truth" not in report and "errors" not in report
        assert "generator_matches_pec_exactly" not in manifest
        np.testing.assert_array_equal(
            observations[["decision", "response_time"]],
            design.iloc[2:][["decision", "response_time"]],
        )
        assert "fitted_minus_initial" in report["independent_seed_rescoring"][0]


@pytest.mark.triton
@pytest.mark.triton_gpu
@pytest.mark.batched
@pytest.mark.parametrize("failed_phase", ["validating", "predicting"])
def test_gpu_late_failure_preserves_search_and_completed_validation(
    tmp_path, design, monkeypatch, failed_phase
):
    path = tmp_path / "data.csv"
    design.to_csv(path, index=False)
    output = tmp_path / "result"

    def fail(*args, **kwargs):
        raise RuntimeError("late failure for test")

    if failed_phase == "validating":
        monkeypatch.setattr(driver, "summarize_validation", fail)
    else:
        original_summary = driver.summarize_samples
        summaries = []

        def fail_on_predictions(*args, **kwargs):
            if summaries:
                fail()
            summaries.append(True)
            return original_summary(*args, **kwargs)

        monkeypatch.setattr(driver, "summarize_samples", fail_on_predictions)
    with pytest.raises(RuntimeError, match="late failure for test"):
        main(
            [
                "--data",
                str(path),
                "--subject",
                "42",
                "--estimates",
                "16",
                "--evaluations",
                "1",
                "--predictive-estimates",
                "4",
                "--validation-estimates",
                "16",
                "--validation-seeds",
                "91001",
                "--output",
                str(output),
            ]
        )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["failed_phase"] == failed_phase
    checkpoint = json.loads((output / "fit_checkpoint.json").read_text())
    assert checkpoint["status"] == "search_complete" and checkpoint["evaluations"] == 1
    np.testing.assert_allclose(
        list(checkpoint["fitted"].values()), STARTS[0], rtol=0.0, atol=1e-12
    )
    validation = json.loads((output / "validation.json").read_text())
    assert len(validation["independent_seed_rescoring"]) == 1
    assert validation["status"] == (
        "complete" if failed_phase == "predicting" else "validating"
    )
    assert not (output / "fit.json").exists()


@pytest.mark.triton
@pytest.mark.triton_gpu
@pytest.mark.batched
@pytest.mark.parametrize(
    "profile_ndt,recovery", [(False, True), (True, True), (True, False)]
)
def test_gpu_adaptive_recovery_checks_and_refines_on_separate_budget(
    tmp_path, design, profile_ndt, recovery
):
    path = tmp_path / "data.csv"
    design.to_csv(path, index=False)
    output = tmp_path / "adaptive"
    entry = "dawa_pec_recovery.py" if recovery else "dawa_pec_fit.py"
    command = [
        sys.executable,
        str(DIRECTORY / entry),
        "--data",
        str(path),
        "--subject",
        "42",
        "--likelihood",
        "marginal",
        "--fit-strategy",
        "adaptive",
        "--estimates",
        "64",
        "--adaptive-min-estimates",
        "16",
        "--evaluations",
        "41",
        "--adaptive-check-every",
        "10",
        "--adaptive-min-evaluations",
        "21",
        "--adaptive-patience",
        "1",
        "--adaptive-progress-tolerance",
        "100000",
        "--adaptive-refine-evaluations",
        "10",
        "--validation-estimates",
        "128",
        "--validation-seeds",
        "8101",
        "--predictive-estimates",
        "16",
        "--output",
        str(output),
    ]
    if profile_ndt:
        command.append("--profile-ndt")
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=180, cwd=DIRECTORY.parents[3]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(
        (output / ("recovery.json" if recovery else "fit.json")).read_text()
    )
    records = [
        json.loads(line)
        for line in (output / "evaluations.jsonl").read_text().splitlines()
    ]
    assert report["status"] == "complete"
    manifest = json.loads((output / "manifest.json").read_text())
    assert report["fit_strategy"] == manifest["fit_strategy"] == "adaptive"
    assert report["fit_policy"] == manifest["fit_policy"] == "block_racing"
    assert report["adaptive"]["policy"] == "block_racing"
    assert manifest["adaptive_config"]["min_estimates"] == 16
    assert report["evaluations"] == 31 < report["requested_evaluations"]
    assert report["adaptive"]["search_evaluations"] == 21
    assert report["adaptive"]["refinement_evaluations"] == 10
    assert (
        report["adaptive"]["best_reference_score"]
        >= report["adaptive"]["checkpoints"][-1]["reference_score"]
    )
    selection = report["adaptive"]["final_selection"]
    assert (
        report["best_training_log_likelihood"] == selection["selected_reference_score"]
    )
    assert report["adaptive"]["policy_version"] == 2
    assert report["adaptive"]["refinement_covariance"]["reused"]
    assert (
        report["validation_estimates"] == 128
        and report["validation_pseudocount"] == 2.0
    )
    assert not (output / "optimizer.journal").exists()
    assert (output / "optimizer_refinement_trials.csv").exists()
    seeds = [seed for row in records for seed in row["block_seeds"]]
    assert not {8101, 20260925, 21260925} & set(seeds)
    assert {row["phase"] for row in records} == {"adaptive_search", "refinement"}
    if profile_ndt:
        assert len(report["ndt_profile"]["optimizer_parameters"]) == 7
        assert (
            report["ndt_profile"]["distinct_bin_maps"]
            < report["ndt_profile"]["grid_values"]
        )
        assert all(len(row["parameters"]) == 8 for row in records)
        profile = report["adaptive"]["profile"]
        assert (
            report["fitted"][profile["parameter"]]
            == profile["values"][profile["selection_indices"][selection["winner"]]]
        )


@pytest.mark.triton
@pytest.mark.triton_gpu
@pytest.mark.batched
@pytest.mark.parametrize("recovery", [False, True])
def test_gpu_adaptive_conditioned_fitting_preserves_observation_law(
    tmp_path, design, recovery
):
    path = tmp_path / "data.csv"
    design.to_csv(path, index=False)
    output = tmp_path / "staged"
    script = "dawa_pec_recovery.py" if recovery else "dawa_pec_fit.py"
    command = [
        sys.executable,
        str(DIRECTORY / script),
        "--data",
        str(path),
        "--subject",
        "42",
        "--fit-strategy",
        "adaptive",
        "--estimates",
        "64",
        "--adaptive-min-estimates",
        "16",
        "--evaluations",
        "31",
        "--population",
        "4",
        "--adaptive-check-every",
        "4",
        "--adaptive-min-evaluations",
        "9",
        "--adaptive-patience",
        "1",
        "--adaptive-progress-tolerance",
        "100000",
        "--adaptive-refine-evaluations",
        "10",
        "--pseudocount",
        ".5",
        "--validation-estimates",
        "128",
        "--predictive-estimates",
        "16",
        "--validation-seeds",
        "8101",
        "--output",
        str(output),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=180, cwd=tmp_path
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(
        (output / ("recovery.json" if recovery else "fit.json")).read_text()
    )
    manifest = json.loads((output / "manifest.json").read_text())
    records = [
        json.loads(line)
        for line in (output / "evaluations.jsonl").read_text().splitlines()
    ]
    assert report["status"] == manifest["status"] == "complete"
    assert report["fit_strategy"] == manifest["fit_strategy"] == "adaptive"
    assert report["fit_policy"] == manifest["fit_policy"] == "staged"
    assert report["adaptive"]["policy"] == "staged"
    assert manifest["adaptive_config"]["search_estimates"] == 16
    assert report["evaluations"] == 19
    assert report["adaptive"]["search_evaluations"] == 9
    assert report["adaptive"]["refinement_evaluations"] == 10
    assert report["adaptive"]["refinement_covariance"]["reused"]
    assert {row["estimates"] for row in records if row["phase"] == "staged_search"} == {
        16
    }
    assert {row["estimates"] for row in records if row["phase"] == "refinement"} == {64}
    assert report["adaptive"] == json.loads((output / "adaptive.json").read_text())
    assert (
        report["validation_estimates"] == 128
        and report["validation_pseudocount"] == 1.0
    )
    assert report["estimator"]["pseudocount"] == 0.5
    selection = report["adaptive"]["final_selection"]
    assert len(set(selection["seeds"])) == 3
    assert set(selection["seeds"]).isdisjoint({29, 8101, 20260925, 20260926, 21260925})
    assert selection["estimates_per_seed"] == 64
    np.testing.assert_allclose(
        selection["mean_log_scores"], np.mean(selection["replicate_log_scores"], axis=0)
    )
    assert (
        report["best_training_log_likelihood"]
        == selection["reference_scores"][selection["winner"]]
    )
    assert (
        list(report["fitted"].values()) == selection["candidates"][selection["winner"]]
    )
    assert manifest["trials"] == 4 and manifest["scored_trials"] == 3
    assert report["estimator"]["masked_observations_condition_history"]


@pytest.mark.parametrize("likelihood", ["conditioned", "marginal"])
def test_staged_is_not_a_public_fit_strategy(tmp_path, capsys, likelihood):
    output = tmp_path / "must_not_exist"
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--fit-strategy",
                "staged",
                "--likelihood",
                likelihood,
                "--output",
                str(output),
            ]
        )
    assert "invalid choice: 'staged'" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("likelihood", ["conditioned", "marginal"])
@pytest.mark.parametrize("override", [False, True])
def test_adaptive_cli_resolves_policy_defaults_and_shared_controls(
    tmp_path, design, monkeypatch, likelihood, override
):
    path = tmp_path / "data.csv"
    design.to_csv(path, index=False)
    received = []
    monkeypatch.setattr(driver.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        driver,
        "_run",
        lambda args, frame, recovery, seeds, config, manifest, started: received.append(
            (args, config)
        ),
    )
    options = (
        [
            "--adaptive-min-estimates",
            "2000",
            "--adaptive-refine-evaluations",
            "500",
            "--adaptive-selection-repeats",
            "2",
            "--validation-estimates",
            "400000",
            "--validation-seeds",
            "92001",
            "92002",
        ]
        if override
        else []
    )
    main(
        [
            "--data",
            str(path),
            "--subject",
            "42",
            "--likelihood",
            likelihood,
            "--fit-strategy",
            "adaptive",
            "--output",
            str(tmp_path / "result"),
            *options,
        ]
    )
    args, config = received[0]
    expected_count = (
        2000 if override else (10000 if likelihood == "conditioned" else 5000)
    )
    assert args.fit_strategy == "adaptive"
    assert args.adaptive_min_estimates == expected_count
    assert args.validation_estimates == (400000 if override else 1000000)
    assert args.validation_seeds == (
        [92001, 92002] if override else [91001, 91002, 91003, 91004, 91005]
    )
    assert config.refine_evaluations == (500 if override else 600)
    if likelihood == "conditioned":
        assert args.fit_policy == "staged" and isinstance(config, driver.StagedConfig)
        assert config.search_estimates == expected_count
        assert config.selection_repeats == (2 if override else 3)
    else:
        assert args.fit_policy == "block_racing" and isinstance(
            config, driver.AdaptiveConfig
        )
        assert config.min_estimates == expected_count
        assert config.selection_blocks == (2 if override else 3)
