"""Fit one subject's recorded choices and RTs with the compiled PEC GPU sampler.

CMA-ES fits the trial-marginal histogram objective with retained control state.
The recovery entry point uses this same pipeline with synthetic observations.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import pandas as pd
import psyneulink as pnl
import torch

from psyneulink.core.batched import BatchedCompositionCompiler, BatchedTrialParameter
from psyneulink.core.batched.backend.triton.runtime import BatchedTruncationError
from psyneulink.core.globals.utilities import set_global_seed
from dawa_batched_simulation import SOURCE, build_model, fit_surface, node


TRUTH = (.40, .22, -.40, 12., .65, .80, 1.5, 5.5)
STARTS = (
    (.475, .20, -.25, 12.5, .50, .50, 2.5, 6.5),
    (.35, .18, -.35, 9., .35, .65, 2.0, 6.0),
)
GRID_STEPS = (.001, .0001, .001, .01, .001, .001, .001)
LAUNCH = dict(block_size=32, num_warps=1, trial_schedule="independent", normal_rng="philox4x_fast_v1")
RT_RANGE = (0., 3.)


def load_subject(path, subject, *, trials=None, recovery=False):
    """Preserve the selected design's order and masked trials; validate before fitting."""
    frame = pd.read_csv(path)
    design_columns = ["subject_nr", "PrevCongruency", "T1", "T2", "S1", "S2", "S3", "S4",
                      "likelihood_include_mask"]
    outcomes = [] if recovery else ["decision", "response_time"]
    missing = set(design_columns + outcomes) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(sorted(missing))}")
    frame = frame[(frame.subject_nr == subject) & frame.PrevCongruency.notna()].copy()
    if trials is not None:
        frame = frame.iloc[:trials].copy()
    if frame.empty:
        raise ValueError(f"No retained trials for subject_nr={subject}")
    # Keep optional design identifiers for provenance, but discard empirical
    # outcomes in recovery mode so they cannot enter generation or scoring.
    identifiers = [name for name in ("trial_within_subject", "row_id", "Congruency") if name in frame]
    frame = frame[design_columns + identifiers + outcomes].reset_index(drop=True)
    for name in design_columns[1:] + outcomes:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(frame[name].to_numpy(dtype=float)).all():
            raise ValueError(f"Column {name} must contain finite values on every retained trial")
    if not frame.likelihood_include_mask.isin([0, 1]).all():
        raise ValueError("likelihood_include_mask must contain only 0/1 or booleans")
    frame["likelihood_include_mask"] = frame.likelihood_include_mask.astype(bool)
    if set(frame.PrevCongruency.unique()) != {0, 1}:
        raise ValueError("The selected design must contain PrevCongruency levels 0 and 1")
    scored = frame[frame.likelihood_include_mask]
    if set(scored.PrevCongruency.unique()) != {0, 1}:
        raise ValueError("At least one scored trial is required for each PrevCongruency level")
    if not recovery:
        if not frame.decision.isin([0, 1]).all():
            raise ValueError("decision must be 0 or 1 on every retained trial")
        if (frame.response_time <= 0).any():
            raise ValueError("response_time must be positive and in seconds on every retained trial")
        if not scored.response_time.between(*RT_RANGE).all():
            raise ValueError("Scored response_time values must be in seconds within the 0–3 s histogram range; "
                             "review the units and scoring mask")
    return frame


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def synthetic_parameters(model, frame, coordinates):
    """Match the original fitting order, with modes for previous conditions 0/1."""
    if set(frame.PrevCongruency.unique()) != {0., 1.}:
        raise ValueError("This recovery design requires previous-congruency levels 0 and 1.")
    values = iter(coordinates)
    parameters = {}
    for parameter, mechanism in fit_surface(model):
        if parameter == "mode":
            modes = (next(values), next(values))
            value = BatchedTrialParameter(np.where(frame.PrevCongruency.to_numpy() == 0, *modes))
        else:
            value = next(values)
        parameters[f"{mechanism.name}.{parameter}"] = value
    if next(values, None) is not None:
        raise ValueError("Unconsumed generating coordinates.")
    return parameters


def summarize_samples(values, frame):
    """Summarize joint choice/RT samples while preserving sequence generation."""
    result = {}
    for previous in (0, 1):
        mask = frame.likelihood_include_mask.to_numpy(dtype=bool) & (frame.PrevCongruency == previous).to_numpy()
        sample = values[mask].reshape(-1, 2)
        result[str(previous)] = {
            "count": len(sample), "choice_one_probability": float(sample[:, 0].mean()),
            "mean_rt": float(sample[:, 1].mean()),
            "rt_quantiles": np.quantile(sample[:, 1], [.1, .5, .9]).tolist(),
        }
    return result


def main(argv=None, *, recovery=False):
    description = ("Generate a synthetic full-sequence subject and recover its parameters on the GPU."
                   if recovery else __doc__)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=Path, default=SOURCE.parent / "flanker_data_part1.csv")
    parser.add_argument("--subject", type=int, default=1, help="Actual subject_nr value in the CSV (default: 1)")
    parser.add_argument("--trials", type=int, help="Prefix for smoke tests; default is the complete subject")
    parser.add_argument("--estimates", type=int, default=100000, help="Simulated trajectories per proposal (default: 100000)")
    parser.add_argument("--evaluations", type=int, default=5000, help="Total parameter proposals, not generations (default: 5000)")
    parser.add_argument("--population", type=int, default=10, help="CMA-ES population and candidate batch size (default: 10)")
    parser.add_argument("--start", type=int, choices=(0, 1), default=0, help="Which of the two predefined starting points to use")
    parser.add_argument("--optimizer-seed", type=int, default=101)
    parser.add_argument("--simulation-seed", type=int, default=29)
    if recovery:
        parser.add_argument("--data-seed", type=int, default=20260925)
    else:
        parser.add_argument("--predictive-seed", type=int, default=21260925)
    parser.add_argument("--model-seed", type=int, default=29)
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[8101, 8102, 8103])
    parser.add_argument("--predictive-estimates", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--source-revision", help="Commit of an isolated source snapshot without .git")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are never overwritten")
    args = parser.parse_args(argv)
    if min(args.estimates, args.evaluations, args.population, args.predictive_estimates, args.max_steps) < 1:
        parser.error("Counts must be positive")
    if args.trials is not None and args.trials < 2:
        parser.error("A prefix must contain at least two trials")
    independent_seeds = {args.simulation_seed}
    if recovery:
        independent_seeds.add(args.data_seed)
    if (recovery and args.data_seed == args.simulation_seed) or set(args.validation_seeds) & independent_seeds:
        parser.error("Generation, fitting, and validation seeds must be independent")
    if not recovery and args.predictive_seed in independent_seeds | set(args.validation_seeds):
        parser.error("Predictive, fitting, and validation seeds must be independent")
    try:
        frame = load_subject(args.data, args.subject, trials=args.trials, recovery=recovery)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU and CUDA-enabled PyTorch are required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    set_global_seed(args.model_seed)
    pnl.set_num_threads(8)
    torch.set_num_threads(8)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    noise = dict(c_noise=.1, s_noise=.1, d_noise=.1, r_noise=.1)
    model, inputs, outputs = build_model(trials=len(frame), **noise)
    inputs[node(model, "Task Input")] = frame[["T1", "T2"]].to_numpy()
    inputs[node(model, "Stimulus Input")] = frame[["S1", "S2", "S3", "S4"]].to_numpy()
    if recovery:
        truth_parameters = synthetic_parameters(model, frame, TRUTH)
        generator = BatchedCompositionCompiler.compile(model, backend="triton", outputs=outputs, max_steps=args.max_steps)
        generated = generator.run(inputs, [truth_parameters], 1, seed=args.data_seed,
                                  strict_truncation=True, triton_launch_options=LAUNCH).values[0, 0]
        if not np.all(np.isin(generated[..., 0], [0., 1.])):
            raise AssertionError("Invalid generated choices")
        frame["decision"] = generated[:, 0, 0]
        frame["response_time"] = generated[:, 0, 1]
    observations_path = args.output / ("synthetic_subject.csv" if recovery else "observed_subject.csv")
    frame.to_csv(observations_path, index=False)
    observed = frame[["decision", "response_time", "subject_nr", "PrevCongruency"]].copy()
    for name in ("decision", "subject_nr", "PrevCongruency"):
        observed[name] = pd.Categorical(observed[name], categories=[0., 1.] if name == "decision" else None)
    surface = fit_surface(model)
    grids = {key: np.linspace(lo, hi, round((hi - lo) / step) + 1)
             for (key, (lo, hi, _)), step in zip(surface.items(), GRID_STEPS, strict=True)}
    depends = {key: "subject_nr" for key in surface if key[0] in ("termination_threshold", "gain", "slope")}
    depends[("intercept", node(model, "RT_GATE"))] = "subject_nr"
    depends[("mode", node(model, "LC"))] = "PrevCongruency"
    pec = pnl.ParameterEstimationComposition(
        model=model, parameters=grids, depends_on=depends, outcome_variables=list(outputs), data=observed,
        likelihood_include_mask=frame.likelihood_include_mask.to_numpy(dtype=bool),
        optimization_function=pnl.PECOptimizationFunction(
            method="differential_evolution", max_iterations=args.evaluations,
            batched_backend="triton", batched_max_steps=args.max_steps,
            batched_seed=args.simulation_seed, batched_strict_truncation=True,
            batched_bins=100, batched_bin_range=[RT_RANGE], batched_pseudocount=1.,
            batched_smoothing_sigma=.5, batched_categorical_cardinalities=[2],
            batched_fused_likelihood=True, batched_specialize_fixed_parameters=True,
            batched_parameter_batch_size=args.population, batched_triton_launch_options=LAUNCH,
        ), num_estimates=args.estimates, initial_seed=args.simulation_seed,
        same_seed_for_all_parameter_combinations=True,
    )
    pec.controller._pec_input_values_by_node = inputs
    function = pec.controller.function
    names = function.fit_param_names
    if len(names) != len(STARTS[args.start]):
        raise AssertionError(f"Expected eight fit coordinates, got {names}")
    bounds = function.fit_param_bounds
    plan = function._compile_batched_plan()
    indices = function._batched_outcome_indices(plan)
    if recovery:
        replay = plan.run(inputs, [function._batched_parameter_set(TRUTH)], 1, seed=args.data_seed,
                          strict_truncation=True, triton_launch_options=LAUNCH).values[0, 0][..., indices]
        np.testing.assert_array_equal(replay, generated)
    steps = {name: value for name, value in plan.fixed_parameters.items() if name.endswith(".time_step_size")}
    if len(steps) != 5 or any(value != (.02 if name.startswith("LC.") else .01) for name, value in steps.items()):
        raise AssertionError(f"Unexpected model clocks: {steps}")
    initial = dict(zip(names, STARTS[args.start], strict=True))
    for name, value in initial.items():
        lo, hi, step = bounds[name]
        if not lo <= value <= hi or not np.isclose((value - lo) / step, round((value - lo) / step)):
            raise AssertionError(f"Initial value is outside parameter grid: {name}={value}")
    storage = JournalStorage(JournalFileBackend(str(args.output / "optimizer.journal")))
    study = optuna.create_study(
        study_name="dawa_recovery" if recovery else "dawa_fit", storage=storage, direction="maximize",
        sampler=optuna.samplers.CmaEsSampler(x0=initial, sigma0=.2, lr_adapt=True,
                                            popsize=args.population, seed=args.optimizer_seed),
    )
    study.enqueue_trial(initial)
    function.method = study
    manifest = {
        "status": "fitting", "mode": "recovery" if recovery else "empirical_fit",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "git_commit": args.source_revision or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "entry_point": Path(sys.argv[0]).name,
        "source_model_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "design_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "observations_sha256": hashlib.sha256(observations_path.read_bytes()).hexdigest(),
        "trials": len(frame), "scored_trials": int(frame.likelihood_include_mask.sum()),
        "initial": initial, "bounds": bounds,
        "noise": noise, "time_steps": steps, "lc_internal_steps_per_pass": 10,
        "estimator": {"kind": "trial_marginal_histogram", "bins": 100, "rt_range": [0., 3.],
                      "smoothing_sigma": .5, "pseudocount": 1.},
        "launch_options": LAUNCH, "gpu": torch.cuda.get_device_name(), "hostname": platform.node(),
        "torch": torch.__version__, "python": platform.python_version(),
        "data_summary": summarize_samples(frame[["decision", "response_time"]].to_numpy(), frame),
        "setup_seconds": time.perf_counter() - started,
        "note": "One subject; trial-marginal fitting with full simulated latent histories. Budget completion is not convergence.",
    }
    if recovery:
        manifest.update(truth=dict(zip(names, TRUTH, strict=True)), generator_matches_pec_exactly=True,
                        synthetic_sha256=manifest["observations_sha256"])
    save_json(args.output / "manifest.json", manifest)
    objective = function._make_objective_func()
    batch_objective = objective._batched_parameter_sets
    optimization_started = time.perf_counter()
    records = []
    invalid = []

    def logged_batch(candidates):
        begin = time.perf_counter()
        try:
            scores = batch_objective(candidates)
        except BatchedTruncationError:
            # Do not treat truncated paths as valid outcomes. Identify and
            # record offending proposals; unexpected errors still abort.
            scores = []
            for candidate in candidates:
                try:
                    scores.append(float(batch_objective([candidate])[0]))
                except BatchedTruncationError as error:
                    scores.append(-1.e10)
                    invalid.append({"parameters": list(candidate), "reason": str(error)})
        if not np.all(np.isfinite(scores)):
            raise FloatingPointError("Nonfinite objective during fitting")
        elapsed = time.perf_counter() - begin
        with (args.output / "evaluations.jsonl").open("a") as stream:
            for candidate, score in zip(candidates, scores, strict=True):
                record = {"evaluation": len(records) + 1, "parameters": list(candidate), "log_likelihood": float(score),
                          "elapsed_fit_seconds": time.perf_counter() - optimization_started,
                          "batch_seconds": elapsed, "batch_size": len(candidates)}
                records.append(record)
                stream.write(json.dumps(record) + "\n")
        best = max(records, key=lambda row: row["log_likelihood"])
        progress = {"completed_evaluations": len(records), "best": best,
                    "fit_seconds": time.perf_counter() - optimization_started, "invalid_candidates": len(invalid)}
        save_json(args.output / "progress.json", progress)
        print(json.dumps({"progress": progress}), flush=True)
        return np.asarray(scores)

    def logged_objective(*values):
        return float(logged_batch([values])[0])

    logged_objective._batched_parameter_sets = logged_batch
    try:
        fit = function._fit(logged_objective, display_iter=False)
        fit_seconds = time.perf_counter() - optimization_started
        if len(records) != args.evaluations:
            raise AssertionError("Optimizer did not evaluate the requested budget")
        fitted = np.asarray([fit["fitted_params"][name] for name in names])
        if float(fit["optimal_value"]) <= -1.e9:
            raise RuntimeError("No valid fit found")
        np.testing.assert_allclose([study.trials[0].params[name] for name in names], STARTS[args.start], rtol=0, atol=1e-12)
        study.trials_dataframe(attrs=("number", "value", "params", "state", "datetime_start", "datetime_complete")).to_csv(
            args.output / "optimizer_trials.csv", index=False)
        comparison = {"initial": STARTS[args.start], "fitted": fitted}
        if recovery:
            comparison = {"truth": TRUTH, **comparison}
        validation = []
        for seed in args.validation_seeds:
            function.batched_seed = seed
            scores = function._make_objective_func()._batched_parameter_sets(list(comparison.values()))
            score = dict(zip(comparison, map(float, scores), strict=True))
            validation.append({"seed": seed, **score, "fitted_minus_initial": score["fitted"] - score["initial"]})
            if recovery:
                validation[-1]["fitted_minus_truth"] = score["fitted"] - score["truth"]
        predictions = {}
        predictive_seed = args.data_seed + 1000000 if recovery else args.predictive_seed
        predictive_parameters = {"truth": TRUTH, "fitted": fitted} if recovery else {"fitted": fitted}
        for label, values in predictive_parameters.items():
            samples = plan.run(inputs, [function._batched_parameter_set(values)], args.predictive_estimates,
                               seed=predictive_seed, strict_truncation=True,
                               triton_launch_options=LAUNCH).values[0, 0][..., indices]
            predictions[label] = summarize_samples(samples, frame)
        report = {"status": "complete", "mode": manifest["mode"], "subject": args.subject, "initial": initial,
                  "fitted": dict(zip(names, fitted.tolist(), strict=True)),
                  "best_training_log_likelihood": float(fit["optimal_value"]),
                  "initial_training_log_likelihood": records[0]["log_likelihood"],
                  "evaluations": len(records), "fit_seconds": fit_seconds,
                  "total_seconds": time.perf_counter() - started, "invalid_proposals": invalid,
                  "independent_seed_rescoring": validation, "data_summary": manifest["data_summary"],
                  "predictive_summaries": predictions, "predictive_seed": predictive_seed,
                  "optimizer_stop_reason": "Requested evaluation budget completed; convergence not asserted"}
        if recovery:
            widths = np.array([bounds[name][1] - bounds[name][0] for name in names])
            report.update(truth=manifest["truth"], errors=dict(zip(names, (fitted - TRUTH).tolist(), strict=True)),
                          errors_as_fraction_of_bounds=dict(zip(names, ((fitted - TRUTH) / widths).tolist(), strict=True)))
        save_json(args.output / ("recovery.json" if recovery else "fit.json"), report)
        manifest.update(status="complete", fit_seconds=fit_seconds)
        save_json(args.output / "manifest.json", manifest)
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(args.output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
