"""Profile LC modes with nuisance reoptimization and complete-filter rescoring.

These are finite-budget diagnostic profiles, not certified maxima or calibrated
parameter confidence intervals. Every score is an observation-conditioned filter.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
import torch

from psyneulink.core.batched.backend.triton.runtime import BatchedTruncationError
from psyneulink.core.components.functions.nonstateful.fitfunctions import (
    _run_batched_ask_tell_rounds,
)
from psyneulink.core.globals.utilities import set_global_seed
from dawa_batched_simulation import SOURCE, build_model, node
from dawa_conditioned_accuracy import mean_statistics
from dawa_pec_fit import LAUNCH, make_fit_pec, save_json


def canonical_profile_values(values, bounds):
    """Deduplicate by grid coordinate, including floating-point endpoint aliases."""
    lo, hi, step = bounds
    indices = set()
    for value in values:
        coordinate = (value - lo) / step
        if not lo <= value <= hi or not np.isclose(
            coordinate, round(coordinate), rtol=0.0, atol=1.0e-7
        ):
            raise ValueError("Profile coordinate must lie on the fitted parameter grid")
        indices.add(round(coordinate))
    return [lo + index * step for index in sorted(indices)]


def optimize_point(
    evaluate,
    names,
    bounds,
    initials,
    fixed_index,
    fixed_value,
    *,
    evaluations,
    population,
    optimizer_seed,
    sigma=0.06,
    log_batch=None,
):
    """Fix one coordinate; optimize all other coordinates on their original grids."""
    names = list(names)
    free = [i for i in range(len(names)) if i != fixed_index]
    lo, hi, step = bounds[names[fixed_index]]
    if not lo <= fixed_value <= hi or not np.isclose(
        (fixed_value - lo) / step, round((fixed_value - lo) / step)
    ):
        raise ValueError("Profile coordinate must lie on the fitted parameter grid")
    initial_rows = []
    for initial in initials:
        row = np.array(initial, dtype=float, copy=True)
        row[fixed_index] = fixed_value
        if not any(np.array_equal(row, old) for old in initial_rows):
            initial_rows.append(row)
    if not initial_rows or evaluations < len(initial_rows):
        raise ValueError("Evaluation budget must cover every supplied initial point")
    # Select the nuisance starting point using the fitting seed, not validation.
    initial_scores = np.asarray(evaluate(initial_rows), dtype=float)
    best_initial = initial_rows[int(np.argmax(initial_scores))]
    distributions = {
        names[i]: optuna.distributions.FloatDistribution(
            *bounds[names[i]][:2], step=bounds[names[i]][2]
        )
        for i in free
    }
    sampler = optuna.samplers.CmaEsSampler(
        x0={names[i]: best_initial[i] for i in free},
        sigma0=sigma,
        popsize=population,
        seed=optimizer_seed,
        lr_adapt=True,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    for row, score in zip(initial_rows, initial_scores, strict=True):
        trial = optuna.trial.create_trial(
            params={names[i]: float(row[i]) for i in free},
            distributions=distributions,
            value=float(score),
        )
        study.add_trial(trial)
    records = [
        {"parameters": row.tolist(), "score": float(score), "initial": True}
        for row, score in zip(initial_rows, initial_scores, strict=True)
    ]
    if log_batch:
        log_batch(records)

    def batch(rows):
        complete = []
        for values in rows:
            candidate = best_initial.copy()
            candidate[free] = values
            candidate[fixed_index] = fixed_value
            complete.append(candidate)
        scores = np.asarray(evaluate(complete), dtype=float)
        added = [
            {"parameters": row.tolist(), "score": float(score), "initial": False}
            for row, score in zip(complete, scores, strict=True)
        ]
        records.extend(added)
        if log_batch:
            log_batch(added)
        return scores

    remaining = evaluations - len(initial_rows)
    if remaining:
        _run_batched_ask_tell_rounds(
            study,
            distributions,
            [names[i] for i in free],
            population,
            remaining,
            batch,
            startup_trials=1,
            generation_attr=getattr(sampler, "_attr_key_generation", "cma:generation"),
        )
    best = max(records, key=lambda row: row["score"])
    assert len(records) == evaluations
    assert all(row["parameters"][fixed_index] == fixed_value for row in records)
    return {
        "fixed_value": fixed_value,
        "parameters": best["parameters"],
        "training_score": best["score"],
        "best_initial_score": float(initial_scores.max()),
        "evaluations": len(records),
        "initial_parameters": [row.tolist() for row in initial_rows],
        "last_quarter_improvement": best["score"]
        - max(row["score"] for row in records[: max(1, 3 * len(records) // 4)]),
        "optimizer_seed": optimizer_seed,
        "sigma": sigma,
    }


def source_hashes():
    directory = Path(__file__).resolve().parent
    root = directory.parents[3]
    files = list((root / "psyneulink/core/batched").rglob("*.py"))
    files += [
        Path(__file__),
        directory / "dawa_pec_fit.py",
        directory / "dawa_batched_simulation.py",
        directory / "dawa_conditioned_accuracy.py",
        directory / "dawa_conditioned_reference.py",
        SOURCE,
        root / "psyneulink/core/components/functions/nonstateful/fitfunctions.py",
    ]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(files))
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--mode-index", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--values", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9]
    )
    parser.add_argument(
        "--evaluations",
        type=int,
        default=600,
        help="Candidate count per fixed mode, including nuisance starts",
    )
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--estimates", type=int, default=100000)
    parser.add_argument("--fit-seed", type=int, default=17001)
    parser.add_argument("--optimizer-seed", type=int, default=601)
    parser.add_argument(
        "--validation-estimates", type=int, nargs="+", default=[400000, 1000000]
    )
    parser.add_argument(
        "--validation-seeds", type=int, nargs="+", default=list(range(18001, 18009))
    )
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        min(
            args.evaluations,
            args.population,
            args.estimates,
            args.validation_batch_size,
            *args.validation_estimates,
        )
        < 1
    ):
        parser.error("Counts must be positive")
    if (
        len(set(args.validation_seeds)) != len(args.validation_seeds)
        or len(args.validation_seeds) < 2
    ):
        parser.error("Provide at least two unique validation seeds")
    if args.output.exists() != args.resume:
        parser.error("Use a new output directory or --resume an existing one")
    manifests = [json.loads((run / "manifest.json").read_text()) for run in args.runs]
    fits = [json.loads((run / "recovery.json").read_text()) for run in args.runs]
    first = manifests[0]
    if any(
        m["status"] != "complete"
        or m["estimator"]["kind"] != "observation_conditioned_particle_histogram"
        or m.get("synthetic_observation_model", {}).get("kind")
        != "binned_smoothed_uniform_contamination"
        for m in manifests
    ):
        parser.error(
            "Profiles require completed conditioned recovery runs with matched observations"
        )
    for key in (
        "observations_sha256",
        "source_model_sha256",
        "truth",
        "estimator",
        "bounds",
        "noise",
    ):
        if any(m[key] != first[key] for m in manifests):
            parser.error(f"Input runs disagree on {key}")
    if first["source_model_sha256"] != hashlib.sha256(SOURCE.read_bytes()).hexdigest():
        parser.error("The source model differs from the fitted model")
    reserved = set()
    for m in manifests:
        settings = m["arguments"]
        reserved.update(
            [
                settings["data_seed"],
                settings["observation_seed"],
                settings["simulation_seed"],
                *settings["validation_seeds"],
            ]
        )
    if args.fit_seed in reserved or set(args.validation_seeds) & (
        reserved | {args.fit_seed}
    ):
        parser.error(
            "Profile training and validation seeds must be fresh and disjoint from prior experiments"
        )
    frame_path = args.runs[0] / "synthetic_subject.csv"
    if (
        hashlib.sha256(frame_path.read_bytes()).hexdigest()
        != first["observations_sha256"]
    ):
        parser.error("Synthetic observations no longer match their manifest")
    frame = pd.read_csv(frame_path)
    frame["likelihood_include_mask"] = frame.likelihood_include_mask.astype(bool)
    fixed_index = 4 + args.mode_index
    names = list(first["truth"])
    bases = [
        {"label": f"fit_{i}", "parameters": [fit["fitted"][name] for name in names]}
        for i, fit in enumerate(fits)
    ]
    bases.append(
        {"label": "truth", "parameters": [first["truth"][name] for name in names]}
    )
    # Pre-profile validation chooses the anchor; later validation is independent.
    anchor_index = int(
        np.argmax(
            [
                np.mean([r["fitted"] for r in fit["independent_seed_rescoring"]])
                for fit in fits
            ]
        )
    )
    anchor = bases[anchor_index]
    values = canonical_profile_values(
        args.values + [row["parameters"][fixed_index] for row in bases],
        first["bounds"][names[fixed_index]],
    )
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in ("resume", "output", "runs")
    }
    config.update(
        runs=[str(run.resolve()) for run in args.runs],
        values=values,
        observations_sha256=first["observations_sha256"],
        anchor=anchor,
        input_manifest_sha256=[
            hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest()
            for run in args.runs
        ],
    )
    hashes = source_hashes()
    environment = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    report_path = args.output / "profile.json"
    if args.resume:
        report = json.loads(report_path.read_text())
        for key, current in (
            ("config", config),
            ("source_sha256", hashes),
            ("environment", environment),
        ):
            if report[key] != current:
                parser.error(f"Cannot resume: {key} differs")
        if report["status"] == "complete":
            print(f"Already complete: {report_path}")
            return
        report.pop("error", None)
        report["status"] = "running"
    else:
        args.output.mkdir(parents=True)
        report = {
            "status": "running",
            "config": config,
            "source_sha256": hashes,
            "environment": environment,
            "parameter_names": names,
            "bounds": first["bounds"],
            "points": [],
            "validation": [],
            "invalid_proposals": [],
            "note": "Finite-budget nuisance optimization; no global-optimum or parameter-confidence claim. "
            "Pointwise intervals describe Monte Carlo mean comparisons on one synthetic subject, not data-sampling uncertainty.",
        }
    started = time.perf_counter()
    prior_seconds = report.get("total_seconds", 0.0)

    def checkpoint():
        report["total_seconds"] = prior_seconds + time.perf_counter() - started
        save_json(report_path, report)

    checkpoint()
    set_global_seed(first["arguments"]["model_seed"])
    torch.set_num_threads(4)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    model, inputs, outputs = build_model(trials=len(frame), **first["noise"])
    inputs[node(model, "Task Input")] = frame[["T1", "T2"]].to_numpy()
    inputs[node(model, "Stimulus Input")] = frame[["S1", "S2", "S3", "S4"]].to_numpy()
    settings = SimpleNamespace(**first["arguments"])
    settings.pseudocount *= args.estimates / settings.estimates
    settings.estimates, settings.population, settings.evaluations = (
        args.estimates,
        args.population,
        args.evaluations,
    )
    settings.simulation_seed = args.fit_seed
    pec = make_fit_pec(model, inputs, outputs, frame, settings)
    function = pec.controller.function
    assert list(function.fit_param_names) == names
    plan = function._compile_batched_plan()
    objective = function._make_objective_func()._batched_parameter_sets

    def evaluate(rows):
        try:
            scores = np.asarray(objective(rows), dtype=float)
        except BatchedTruncationError:
            scores = []
            for row in rows:
                try:
                    scores.append(float(objective([row])[0]))
                except BatchedTruncationError as error:
                    scores.append(-1e10)
                    report["invalid_proposals"].append(
                        {"parameters": np.asarray(row).tolist(), "reason": str(error)}
                    )
            scores = np.asarray(scores)
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite profile score")
        return scores

    try:
        # Start at the anchor, then move outward; nearby completed optima provide warm starts.
        order = sorted(
            values, key=lambda v: (abs(v - anchor["parameters"][fixed_index]), v)
        )
        for index, value in enumerate(order):
            if any(point["fixed_value"] == value for point in report["points"]):
                continue
            initials = [row["parameters"] for row in bases]
            if report["points"]:
                initials.append(
                    min(report["points"], key=lambda p: abs(p["fixed_value"] - value))[
                        "parameters"
                    ]
                )
            begin = time.perf_counter()

            def log_batch(records):
                with (args.output / "evaluations.jsonl").open("a") as stream:
                    for record in records:
                        stream.write(
                            json.dumps({"fixed_value": value, **record}) + "\n"
                        )
                print(
                    json.dumps(
                        {
                            "mode_index": args.mode_index,
                            "fixed_value": value,
                            "batch_best": max(r["score"] for r in records),
                            "point_seconds": time.perf_counter() - begin,
                        }
                    ),
                    flush=True,
                )

            point = optimize_point(
                evaluate,
                names,
                function.fit_param_bounds,
                initials,
                fixed_index,
                value,
                evaluations=args.evaluations,
                population=args.population,
                optimizer_seed=args.optimizer_seed + index,
                log_batch=log_batch,
            )
            point["seconds"] = time.perf_counter() - begin
            report["points"].append(point)
            checkpoint()
        candidates = bases + [
            {
                "label": f"mode{args.mode_index}={point['fixed_value']:.6g}",
                "parameters": point["parameters"],
            }
            for point in sorted(report["points"], key=lambda p: p["fixed_value"])
        ]
        report["candidates"] = candidates
        parameters = [
            function._batched_parameter_set(row["parameters"]) for row in candidates
        ]
        observed = pec._data_numpy
        include = frame.likelihood_include_mask.to_numpy(dtype=bool)
        done = {
            (row["estimates"], row["seed"], row["label"])
            for row in report["validation"]
        }
        for estimates in args.validation_estimates:
            for seed in args.validation_seeds:
                for first_index in range(
                    0, len(candidates), args.validation_batch_size
                ):
                    stop = min(
                        first_index + args.validation_batch_size, len(candidates)
                    )
                    selected = candidates[first_index:stop]
                    keys = [(estimates, seed, row["label"]) for row in selected]
                    if all(key in done for key in keys):
                        continue
                    if any(key in done for key in keys):
                        raise ValueError("Incomplete saved validation batch")
                    begin = time.perf_counter()
                    _, diagnostics = plan.conditioned_log_likelihood(
                        inputs,
                        parameters[first_index:stop],
                        estimates,
                        data=observed,
                        outcome_indices=function._batched_outcome_indices(plan),
                        categorical_dims=pec.data_categorical_dims,
                        categorical_cardinalities=[2],
                        bins=function.batched_bins,
                        bin_range=function.batched_bin_range,
                        smoothing_sigma=0.5,
                        pseudocount=settings.pseudocount * estimates / args.estimates,
                        seed=seed,
                        include_mask=include,
                        strict_truncation=True,
                        triton_launch_options=LAUNCH,
                        return_diagnostics=True,
                    )
                    logs = np.log(
                        np.asarray(diagnostics["per_trial_densities"])[:, 0].astype(
                            float
                        )
                    )
                    ess = np.asarray(diagnostics["effective_sample_size"])[:, 0]
                    prior = np.asarray(diagnostics["prior_mixture_fraction"])[:, 0]
                    for i, candidate in enumerate(selected):
                        report["validation"].append(
                            {
                                "estimates": estimates,
                                "seed": seed,
                                "label": candidate["label"],
                                "selected_log_score": float(logs[i, include].sum()),
                                "full_log_score": float(logs[i].sum()),
                                "minimum_ess": float(ess[i].min()),
                                "median_ess": float(np.median(ess[i])),
                                "contamination_above_half": int((prior[i] > 0.5).sum()),
                            }
                        )
                    done.update(keys)
                    checkpoint()
                    print(
                        json.dumps(
                            {"validation": keys, "seconds": time.perf_counter() - begin}
                        ),
                        flush=True,
                    )
        report["summary"] = []
        for estimates in args.validation_estimates:
            by_label = {
                candidate["label"]: {
                    row["seed"]: row["selected_log_score"]
                    for row in report["validation"]
                    if row["estimates"] == estimates
                    and row["label"] == candidate["label"]
                }
                for candidate in candidates
            }
            for candidate in candidates:
                label = candidate["label"]
                scores = [by_label[label][seed] for seed in args.validation_seeds]
                differences = [
                    by_label[label][seed] - by_label[anchor["label"]][seed]
                    for seed in args.validation_seeds
                ]
                report["summary"].append(
                    {
                        "estimates": estimates,
                        "label": label,
                        "score": mean_statistics(scores),
                        "difference_from_anchor": mean_statistics(differences),
                    }
                )
        report["status"] = "complete"
        checkpoint()
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        checkpoint()
        raise


if __name__ == "__main__":
    main()
