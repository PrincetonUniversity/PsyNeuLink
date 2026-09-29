"""Measure seed variability and particle-budget drift of conditioned DAWA scores.

Each repetition runs a complete, independent sequential particle filter. Trial
densities from different repetitions are never pooled into a hybrid filter.
The largest tested budget is a finite Monte Carlo reference, not ground truth.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t


DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = DIRECTORY.parents[3]
REFERENCE_ESTIMATES = 100000
SCHEMA_VERSION = 1
IMPLEMENTATION_FILES = (
    "Scripts/Debug/pec_batch_compile/dawa/dawa_conditioned_accuracy.py",
    "Scripts/Debug/pec_batch_compile/dawa/dawa_batched_simulation.py",
    "Scripts/Debug/pec_batch_compile/dawa/dawa_pec_fit.py",
    "Scripts/Debug/pec_batch_compile/dawa/dawa_lca_model/full_lca_model_lc.py",
    "psyneulink/core/batched/compiler.py",
    "psyneulink/core/batched/likelihood.py",
    "psyneulink/core/batched/prep.py",
    "psyneulink/core/batched/backend/triton/runtime.py",
    "psyneulink/core/batched/backend/triton/conditioned.py",
    "psyneulink/core/batched/backend/triton/state.py",
    "psyneulink/core/batched/backend/triton/emit/emitter.py",
)
DEFAULT_CANDIDATES = {
    "reference": "benchmark_4",
    "candidates": [
        {
            "label": "benchmark_1",
            "parameters": [0.30, 0.20, -0.45, 10.0, 0.90, 0.90, 1.0, 5.0],
        },
        {
            "label": "benchmark_2",
            "parameters": [0.40, 0.22, -0.40, 12.0, 0.70, 0.80, 1.5, 5.5],
        },
        {
            "label": "benchmark_3",
            "parameters": [0.50, 0.18, -0.35, 8.0, 0.50, 0.70, 2.0, 6.0],
        },
        {
            "label": "benchmark_4",
            "parameters": [0.60, 0.25, -0.30, 15.0, 0.30, 0.60, 2.5, 7.0],
        },
        {
            "label": "benchmark_4_threshold_plus",
            "parameters": [0.61, 0.25, -0.30, 15.0, 0.30, 0.60, 2.5, 7.0],
        },
        {
            "label": "benchmark_4_mode_plus",
            "parameters": [0.60, 0.25, -0.30, 15.0, 0.35, 0.65, 2.5, 7.0],
        },
    ],
}


def load_candidates(path=None):
    """Read {reference: label, candidates: [{label, parameters: [8 numbers]}]}."""
    payload = DEFAULT_CANDIDATES if path is None else json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise ValueError(
            "Candidate JSON requires a candidates list and a reference label"
        )
    candidates = []
    for item in payload["candidates"]:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("label"), str)
            or not item["label"]
        ):
            raise ValueError("Each candidate requires a nonempty string label")
        values = np.asarray(item.get("parameters"), dtype=float)
        if values.shape != (8,) or not np.isfinite(values).all():
            raise ValueError(
                f"Candidate {item['label']!r} requires eight finite parameters"
            )
        candidates.append({"label": item["label"], "parameters": values.tolist()})
    labels = [item["label"] for item in candidates]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Candidate labels must be nonempty and unique")
    reference = payload.get("reference")
    if reference not in labels:
        raise ValueError("The reference label must identify one candidate")
    return {
        **{
            name: payload[name]
            for name in ("parameter_order", "selection_note")
            if name in payload
        },
        "reference": reference,
        "candidates": candidates,
    }


def mean_statistics(values):
    """Descriptive independent-seed statistics and a Student-t interval for the mean."""
    values = np.asarray(values, dtype=float)
    count = len(values)
    if count == 0 or not np.isfinite(values).all():
        raise ValueError("Statistics require nonempty finite values")
    mean = float(values.mean())
    if count == 1:
        return {"count": 1, "mean": mean, "sd": None, "se": None, "mean_ci95": None}
    sd = float(values.std(ddof=1))
    se = sd / np.sqrt(count)
    half_width = float(student_t.ppf(0.975, count - 1)) * se
    return {
        "count": count,
        "mean": mean,
        "sd": sd,
        "se": float(se),
        "mean_ci95": [mean - half_width, mean + half_width],
    }


def analyze_records(records, candidates, estimates, tolerance=0.2):
    """Pair by seed for contrasts and budget differences; never pool trial factors."""
    lookup = {}
    for row in records:
        key = (row["estimates"], row["label"], row["seed"])
        if key in lookup:
            raise ValueError(f"Duplicate accuracy record: {key}")
        lookup[key] = row
    reference = candidates["reference"]
    largest = max(estimates)
    summary = []
    for budget in estimates:
        for item in candidates["candidates"]:
            label = item["label"]
            seeds = sorted(
                seed for n, name, seed in lookup if n == budget and name == label
            )
            if not seeds:
                continue
            rows = [lookup[(budget, label, seed)] for seed in seeds]
            result = {"estimates": budget, "label": label, "seeds": seeds}
            for field in ("selected_log_score", "full_log_score"):
                values = np.array([row[field] for row in rows])
                result[field] = mean_statistics(values)
                # This averages complete likelihood estimates only. In
                # particular, exp(selected factors) has no general unbiased-
                # likelihood claim when the mask omits interspersed factors.
                result[field]["log_mean_exp"] = float(
                    logsumexp(values) - np.log(len(values))
                )
                paired = [seed for seed in seeds if (budget, reference, seed) in lookup]
                differences = [
                    lookup[(budget, label, seed)][field]
                    - lookup[(budget, reference, seed)][field]
                    for seed in paired
                ]
                result.setdefault("difference_from_reference_candidate", {})[field] = (
                    {
                        "reference": reference,
                        "seeds": paired,
                        **mean_statistics(differences),
                    }
                    if paired
                    else None
                )
                paired = [seed for seed in seeds if (largest, label, seed) in lookup]
                drift = [
                    lookup[(budget, label, seed)][field]
                    - lookup[(largest, label, seed)][field]
                    for seed in paired
                ]
                result.setdefault("difference_from_largest_budget", {})[field] = (
                    {
                        "reference_estimates": largest,
                        "seeds": paired,
                        **mean_statistics(drift),
                    }
                    if paired
                    else None
                )
                paired = [
                    seed
                    for seed in paired
                    if (budget, reference, seed) in lookup
                    and (largest, reference, seed) in lookup
                ]
                drift = [
                    (
                        lookup[(budget, label, seed)][field]
                        - lookup[(budget, reference, seed)][field]
                    )
                    - (
                        lookup[(largest, label, seed)][field]
                        - lookup[(largest, reference, seed)][field]
                    )
                    for seed in paired
                ]
                result.setdefault("reference_candidate_contrast_drift", {})[field] = (
                    {
                        "reference_estimates": largest,
                        "reference_candidate": reference,
                        "seeds": paired,
                        **mean_statistics(drift),
                    }
                    if paired
                    else None
                )
                for comparison in (
                    "difference_from_largest_budget",
                    "reference_candidate_contrast_drift",
                ):
                    statistics = result[comparison][field]
                    if statistics is not None:
                        interval = statistics["mean_ci95"]
                        statistics["mean_ci_within_tolerance"] = (
                            bool(interval[0] >= -tolerance and interval[1] <= tolerance)
                            if interval is not None and budget != largest
                            else None
                        )
            summary.append(result)
    return {
        "reference_candidate": reference,
        "largest_tested_budget": largest,
        "agreement_tolerance_log_units": tolerance,
        "interval_note": "95% Student-t intervals describe independent-seed mean log scores or paired differences. "
        "They are not subject/parameter confidence intervals and do not establish particle convergence.",
        "budget_note": "Drift is lower-budget minus largest-budget score at matching seeds; the largest budget is not ground truth.",
        "pooling_note": "log_mean_exp averages complete replicate likelihood estimates. No trial densities are pooled. "
        "Masked products are selected conditional factors, with no general unbiased full-likelihood claim.",
        "candidates": summary,
    }


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_resume(report, config, hashes, environment):
    """Refuse to mix numerical settings, model implementations or GPU environments."""
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Cannot resume a different accuracy report schema")
    for key, current in (
        ("config", config),
        ("implementation_sha256", hashes),
        ("environment", environment),
    ):
        if report.get(key) != current:
            raise ValueError(f"Cannot resume: {key} differs from the saved report")
    valid_budgets = set(config["estimates"])
    valid_seeds = set(config["seeds"])
    labels = {row["label"] for row in config["candidates"]["candidates"]}
    seen = set()
    for row in report.get("records", []):
        key = (row["estimates"], row["seed"], row["label"])
        if (
            key in seen
            or key[0] not in valid_budgets
            or key[1] not in valid_seeds
            or key[2] not in labels
        ):
            raise ValueError(f"Invalid or duplicate saved accuracy record: {key}")
        if not all(
            np.isfinite(row[field])
            for field in ("selected_log_score", "full_log_score", "device_log_score")
        ):
            raise ValueError(f"Nonfinite saved accuracy record: {key}")
        seen.add(key)
    return seen


def _diagnostic_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "fifth_percentile": float(np.quantile(values, 0.05)),
        "maximum": float(values.max()),
    }


def diagnostic_groups(ess, prior, include, estimates):
    """Distinguish difficult scored observations from unscored history updates."""
    result = {}
    for name, mask in (
        ("all", np.ones(len(include), dtype=bool)),
        ("scored", include),
        ("history_only", ~include),
    ):
        count = int(mask.sum())
        result[name] = {"trials": count}
        if count:
            result[name].update(
                ess=_diagnostic_summary(ess[mask]),
                ess_fraction=_diagnostic_summary(ess[mask] / estimates),
                contamination_responsibility=_diagnostic_summary(prior[mask]),
                trials_with_ess_below_100=int((ess[mask] < 100).sum()),
                trials_with_contamination_above_half=int((prior[mask] > 0.5).sum()),
            )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=DIRECTORY / "dawa_lca_model/flanker_data_part1.csv"
    )
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument(
        "--trials",
        type=int,
        help="Optional ordered prefix; default uses the complete subject",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        help="JSON object with reference label and candidates list",
    )
    parser.add_argument(
        "--estimates", type=int, nargs="+", default=[25000, 100000, 400000, 1000000]
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Explicit independent filter seeds; default is 8101–8120",
    )
    parser.add_argument(
        "--seed-start", type=int, help="First consecutive filter seed (default: 8101)"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        help="Number of consecutive seeds (default: 20); exclusive with --seeds",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model-seed", type=int, default=29)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument(
        "--agreement-tolerance",
        type=float,
        default=0.2,
        help="CI containment margin in log units; not a convergence certificate",
    )
    parser.add_argument(
        "--source-revision", help="Commit of a source snapshot without .git"
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        help="Optionally save per-trial density, ESS and contamination arrays as NPZ",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing matching report without repeating completed batches",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.seeds is not None and (
        args.seed_start is not None or args.repeats is not None
    ):
        parser.error("Use --seeds or --seed-start/--repeats, not both")
    if args.seeds is None:
        start = 8101 if args.seed_start is None else args.seed_start
        repeats = 20 if args.repeats is None else args.repeats
        if repeats < 1:
            parser.error("Repeats must be positive")
        args.seeds = list(range(start, start + repeats))
    if min(*args.estimates, args.batch_size, args.max_steps) < 1:
        parser.error("Budgets, batch size and max steps must be positive")
    if len(set(args.estimates)) != len(args.estimates) or len(set(args.seeds)) != len(
        args.seeds
    ):
        parser.error("Particle budgets and seeds must not contain duplicates")
    if min(args.seeds) < 0 or args.model_seed < 0:
        parser.error("Seeds must be nonnegative")
    if not np.isfinite(args.agreement_tolerance) or args.agreement_tolerance <= 0:
        parser.error("Agreement tolerance must be finite and positive")
    if args.output.exists() != args.resume:
        parser.error(
            "Use --resume for an existing report; otherwise choose a new output path"
        )
    try:
        candidates = load_candidates(args.candidates)
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))

    import torch
    import triton
    from psyneulink.core.batched import BatchedCompositionCompiler
    from psyneulink.core.globals.utilities import set_global_seed
    from dawa_batched_simulation import build_model, fit_surface, node
    from dawa_pec_fit import (
        LAUNCH,
        histogram_settings,
        load_subject,
        synthetic_parameters,
    )

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required")
    try:
        frame = load_subject(args.data, args.subject, trials=args.trials)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    bins, rt_range = histogram_settings(frame, "conditioned")
    observed = frame[["decision", "response_time"]].to_numpy()
    include = frame.likelihood_include_mask.to_numpy(dtype=bool)
    config = {
        "subject": args.subject,
        "trials": args.trials,
        "retained_trials": len(frame),
        "scored_trials": int(include.sum()),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "candidates": candidates,
        "estimates": args.estimates,
        "seeds": args.seeds,
        "batch_size": args.batch_size,
        "model_seed": args.model_seed,
        "max_steps": args.max_steps,
        "agreement_tolerance": args.agreement_tolerance,
        "candidates_file_sha256": hashlib.sha256(
            args.candidates.read_bytes()
        ).hexdigest()
        if args.candidates
        else None,
        "histogram": {
            "bins": bins,
            "rt_range": list(rt_range),
            "bin_width": 0.03,
            "smoothing_sigma": 0.5,
            "pseudocount_at_100000": 1.0,
            "contamination_fraction": 2 * bins / (REFERENCE_ESTIMATES + 2 * bins),
        },
        "launch": LAUNCH,
        "execution": "prepared",
        "resampling": "systematic",
        "all_lca_noise": 0.1,
        "lca_dt": 0.01,
        "diagnostics_directory": str(args.diagnostics_dir.resolve())
        if args.diagnostics_dir
        else None,
    }
    implementation_files = set(IMPLEMENTATION_FILES) | {
        str(path.relative_to(REPOSITORY))
        for path in (REPOSITORY / "psyneulink/core/batched").rglob("*.py")
    }
    hashes = {
        name: hashlib.sha256((REPOSITORY / name).read_bytes()).hexdigest()
        for name in sorted(implementation_files)
    }
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
    }
    if args.resume:
        report = json.loads(args.output.read_text())
        try:
            complete = validate_resume(report, config, hashes, environment)
            if args.source_revision is not None and args.source_revision != report.get(
                "git_commit"
            ):
                raise ValueError(
                    "Cannot resume: source revision differs from the saved report"
                )
        except ValueError as error:
            parser.error(str(error))
    else:
        revision = (
            args.source_revision
            or subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=REPOSITORY,
                text=True,
            ).strip()
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "config": config,
            "implementation_sha256": hashes,
            "environment": environment,
            "git_commit": revision,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "timing_note": "Batch seconds include full-sequence filtering, diagnostics, synchronization and host transfer. "
            "First-use compilation is included; these are accuracy-study timings, not warm score-only benchmarks.",
            "score_note": "selected_log_score and full_log_score sum logs of per-trial FP32 densities in host FP64. "
            "device_log_score is the production selected score; small FP32 reduction differences are expected. "
            "All retained rows condition state, including rows omitted from the selected score.",
            "records": [],
            "batches": [],
            "sessions": [],
        }
        complete = set()
    expected = len(candidates["candidates"]) * len(args.estimates) * len(args.seeds)
    if len(complete) == expected:
        print(
            json.dumps(
                {
                    "status": "already_complete",
                    "records": expected,
                    "output": str(args.output),
                }
            ),
            flush=True,
        )
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.diagnostics_dir:
        args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    report["status"] = "running"
    report.pop("error", None)
    session = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "resumed": args.resume,
        "starting_records": len(complete),
    }
    report["sessions"].append(session)
    atomic_json(args.output, report)
    started = time.perf_counter()
    try:
        set_global_seed(args.model_seed)
        model, inputs, outputs = build_model(
            trials=len(frame), c_noise=0.1, s_noise=0.1, d_noise=0.1, r_noise=0.1
        )
        inputs[node(model, "Task Input")] = frame[["T1", "T2"]].to_numpy()
        inputs[node(model, "Stimulus Input")] = frame[
            ["S1", "S2", "S3", "S4"]
        ].to_numpy()
        plan = BatchedCompositionCompiler.compile(
            model, backend="triton", outputs=outputs, max_steps=args.max_steps
        )
        fitted = {
            f"{mechanism.name}.{parameter}"
            for parameter, mechanism in fit_surface(model)
        }
        plan = plan.specialize_parameters(
            {p.name: p.default for p in plan.ir.params if p.name not in fitted}
        )
        parameters = [
            synthetic_parameters(model, frame, row["parameters"])
            for row in candidates["candidates"]
        ]
        session["setup_seconds"] = time.perf_counter() - started
        fingerprint = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:12]
        for estimates in args.estimates:
            for seed in args.seeds:
                for first in range(0, len(parameters), args.batch_size):
                    stop = min(first + args.batch_size, len(parameters))
                    labels = [
                        row["label"] for row in candidates["candidates"][first:stop]
                    ]
                    keys = [(estimates, seed, label) for label in labels]
                    if all(key in complete for key in keys):
                        continue
                    if any(key in complete for key in keys):
                        raise ValueError(
                            "Saved report has a partial candidate batch; refusing to change its batching"
                        )
                    print(
                        json.dumps(
                            {
                                "starting": {
                                    "estimates": estimates,
                                    "seed": seed,
                                    "candidates": labels,
                                },
                                "completed_records": len(complete),
                                "expected_records": expected,
                            }
                        ),
                        flush=True,
                    )
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    scores, diagnostics = plan.conditioned_log_likelihood(
                        inputs,
                        parameters[first:stop],
                        estimates,
                        data=observed,
                        categorical_dims=[0],
                        bins=bins,
                        bin_range=[rt_range],
                        smoothing_sigma=0.5,
                        pseudocount=estimates / REFERENCE_ESTIMATES,
                        categorical_cardinalities=[2],
                        include_mask=include,
                        seed=seed,
                        strict_truncation=True,
                        triton_launch_options=LAUNCH,
                        return_diagnostics=True,
                    )
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - begin
                    densities = np.asarray(diagnostics["per_trial_densities"])[:, 0]
                    if (
                        densities.shape != (stop - first, len(frame))
                        or not np.isfinite(densities).all()
                        or (densities <= 0).any()
                    ):
                        raise FloatingPointError(
                            "Expected finite positive per-trial conditional densities"
                        )
                    logs = np.log(densities.astype(np.float64))
                    device_scores = np.atleast_1d(scores)
                    if (
                        device_scores.shape != (stop - first,)
                        or not np.isfinite(device_scores).all()
                    ):
                        raise FloatingPointError(
                            "Expected one finite selected score per candidate"
                        )
                    batch_id = f"n{estimates}_s{seed}_c{first}-{stop}"
                    artifact = None
                    if args.diagnostics_dir:
                        artifact = (
                            args.diagnostics_dir / f"{fingerprint}_{batch_id}.npz"
                        )
                        temporary = artifact.with_suffix(".npz.tmp")
                        with temporary.open("wb") as stream:
                            np.savez_compressed(
                                stream,
                                labels=np.asarray(labels),
                                include_mask=include,
                                observations=observed,
                                **{
                                    name: diagnostics[name]
                                    for name in (
                                        "per_trial_densities",
                                        "effective_sample_size",
                                        "prior_mixture_fraction",
                                        "zero_support",
                                    )
                                },
                            )
                        temporary.replace(artifact)
                    batch = {
                        "id": batch_id,
                        "estimates": estimates,
                        "seed": seed,
                        "labels": labels,
                        "seconds_including_diagnostics": elapsed,
                        "session": len(report["sessions"]) - 1,
                        "diagnostics_file": str(artifact.resolve())
                        if artifact
                        else None,
                    }
                    report["batches"].append(batch)
                    for index, label in enumerate(labels):
                        ess = np.asarray(diagnostics["effective_sample_size"])[index, 0]
                        prior = np.asarray(diagnostics["prior_mixture_fraction"])[
                            index, 0
                        ]
                        report["records"].append(
                            {
                                "estimates": estimates,
                                "seed": seed,
                                "label": label,
                                "batch_id": batch_id,
                                "selected_log_score": float(logs[index, include].sum()),
                                "full_log_score": float(logs[index].sum()),
                                "device_log_score": float(device_scores[index]),
                                "ess": _diagnostic_summary(ess),
                                "ess_fraction": _diagnostic_summary(ess / estimates),
                                "contamination_responsibility": _diagnostic_summary(
                                    prior
                                ),
                                "trials_with_ess_below_100": int((ess < 100).sum()),
                                "trials_with_contamination_above_half": int(
                                    (prior > 0.5).sum()
                                ),
                                "diagnostic_groups": diagnostic_groups(
                                    ess, prior, include, estimates
                                ),
                            }
                        )
                    complete.update(keys)
                    report["completed_records"] = len(complete)
                    report["expected_records"] = expected
                    report["analysis"] = analyze_records(
                        report["records"],
                        candidates,
                        args.estimates,
                        args.agreement_tolerance,
                    )
                    session["elapsed_seconds"] = time.perf_counter() - started
                    atomic_json(args.output, report)
        report["status"] = "complete"
        session["elapsed_seconds"] = time.perf_counter() - started
        session["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(args.output, report)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "records": len(complete),
                    "output": str(args.output),
                    "session_seconds": session["elapsed_seconds"],
                }
            ),
            flush=True,
        )
    except BaseException as error:
        report["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        report["error"] = f"{type(error).__name__}: {error}"
        session["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(args.output, report)
        raise


if __name__ == "__main__":
    main()
