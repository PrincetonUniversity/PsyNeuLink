"""Run a complete hierarchical DDM fit with Triton or threaded LLVM.

Generate data once, then run each backend in its own process. Both use the same
smoothed histogram likelihood: LLVM's usual KDE is replaced through the PEC
objective hook so density estimation is not an uncontrolled difference.
Simulation RNG algorithms and floating-point precision still differ.

Results include construction, first scoring calls, warm likelihood timings,
the complete EM fit, convergence, evaluations, and parameter recovery. Raw
results belong in benchmark_results/, which is ignored by Git.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
import pandas as pd
import psyneulink as pnl
import torch

from psyneulink.core.batched.likelihood import histogram_log_likelihood
from psyneulink.core.compositions.hierarchical.transforms import BoundedTransform
from psyneulink.core.compositions.hierarchical import laplaceem


PARAMETERS = ("rate", "threshold")
BOUNDS = ((-1.5, 1.5), (0.3, 1.5))
DT = 0.01
NDT = 0.15


def build_model(rate=0.0, threshold=0.9, seed=0):
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0, rate=rate, threshold=threshold, noise=1.0,
            non_decision_time=NDT, time_step_size=DT, seed=seed,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    return pnl.Composition(pathways=decision), decision


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def generate(args):
    transform = BoundedTransform(lower=np.array(BOUNDS)[:, 0], upper=np.array(BOUNDS)[:, 1])
    rng = np.random.default_rng(args.data_seed)
    mean_z = np.array([0.35, -0.3])
    sd_z = np.array([0.55, 0.4])
    z = rng.normal(mean_z, sd_z, size=(args.subjects, len(PARAMETERS)))
    theta = np.array([transform.to_natural(row) for row in z])
    frames = []
    for subject, row in enumerate(theta):
        model, decision = build_model(*row, seed=args.data_seed + 1000 + subject)
        model.run(
            inputs={decision: np.ones((args.trials, 1))},
            execution_mode=pnl.ExecutionMode.LLVMRun,
        )
        frame = pd.DataFrame(np.asarray(model.results).reshape(args.trials, 2),
                             columns=["decision", "response_time"])
        frame.insert(0, "subject", f"S{subject:02d}")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    args.data.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.data, index=False)
    save_json(args.data.with_suffix(".truth.json"), {
        "parameters": PARAMETERS, "theta": theta.tolist(), "z": z.tolist(),
        "population_mean_z": mean_z.tolist(), "population_sd_z": sd_z.tolist(),
        "subjects": args.subjects, "trials_per_subject": args.trials,
        "data_seed": args.data_seed, "time_step": DT, "non_decision_time": NDT,
        "generator": "LLVMRun", "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
    })
    print(f"Generated {len(data)} trials for {args.subjects} participants: {args.data}", flush=True)


def benchmark(args):
    if args.backend == "triton" and not torch.cuda.is_available():
        raise RuntimeError("The Triton benchmark requires a CUDA GPU")
    data = pd.read_csv(args.data)
    data["decision"] = pd.Categorical(data.decision, categories=[0.0, 1.0])
    labels = tuple(pd.unique(data.subject))
    frames = [data.loc[data.subject == label].drop(columns="subject").reset_index(drop=True)
              for label in labels]
    truth = json.loads(args.data.with_suffix(".truth.json").read_text())
    if truth["data_sha256"] != hashlib.sha256(args.data.read_bytes()).hexdigest():
        raise ValueError("Data file differs from the saved truth manifest")
    rt_max = max(6.0, float(np.ceil(data.response_time.max())))
    density_options = dict(
        bins=int(round(rt_max / args.bin_width)), bin_range=[(0.0, rt_max)],
        smoothing_sigma=args.smoothing_sigma, pseudocount=args.pseudocount,
        categorical_cardinalities=[2],
    )
    model_seconds = []
    first_score_seconds = []
    first_scores = []
    first_batch_score_seconds = []
    participants = []
    calls = np.zeros(len(labels), dtype=int)
    score_seconds = np.zeros(len(labels), dtype=float)
    scalar_calls = np.zeros(len(labels), dtype=int)
    batch_calls = np.zeros(len(labels), dtype=int)
    batch_rows = np.zeros(len(labels), dtype=int)
    batch_score_seconds = np.zeros(len(labels), dtype=float)
    hessian_seconds = []
    trace = []
    fit_started = None
    source_diff = subprocess.check_output(["git", "diff", "HEAD", "--", "psyneulink"])
    args.output.with_suffix(".source.patch").write_bytes(source_diff)
    source_diff_sha256 = hashlib.sha256(source_diff).hexdigest()

    def synchronize():
        if args.backend == "triton":
            torch.cuda.synchronize()

    def timed_call(function, *values, **kwargs):
        synchronize()
        start = time.perf_counter()
        result = function(*values, **kwargs)
        synchronize()
        return result, time.perf_counter() - start

    for subject, frame in enumerate(frames):
        start = time.perf_counter()
        seed = args.simulation_seed + subject
        model, decision = build_model(seed=seed)
        gpu_options = {}
        if args.backend == "triton":
            gpu_options = {
                "batched_backend": "triton", "batched_seed": seed,
                "batched_max_steps": args.max_steps, "batched_strict_truncation": True,
                "batched_bins": density_options["bins"],
                "batched_bin_range": density_options["bin_range"],
                "batched_smoothing_sigma": args.smoothing_sigma,
                "batched_pseudocount": args.pseudocount,
                "batched_categorical_cardinalities": [2],
            }
        pec = pnl.ParameterEstimationComposition(
            model=model,
            parameters={(name, decision): np.linspace(*bounds, 1000)
                        for name, bounds in zip(PARAMETERS, BOUNDS)},
            outcome_variables=list(decision.output_ports), data=frame,
            optimization_function=pnl.PECOptimizationFunction(method=None, **gpu_options),
            num_estimates=args.estimates, initial_seed=seed,
            same_seed_for_all_parameter_combinations=True,
        )
        if args.backend == "llvm":
            observed = frame[["decision", "response_time"]].to_numpy(dtype=float)
            pec.controller.function.set_pec_objective_function(
                lambda simulated, observed=observed: histogram_log_likelihood(
                    simulated, observed, [True, False], device="cpu", **density_options,
                )
            )
        inputs = {decision: np.ones((len(frame), 1))}
        participants.append((pec, inputs))
        model_seconds.append(time.perf_counter() - start)
        score, elapsed = timed_call(pec.log_likelihood, 0.0, 0.9, inputs=inputs)
        first_score_seconds.append(elapsed)
        first_scores.append(score)
        repeated = pec.log_likelihood(0.0, 0.9, inputs=inputs)
        if repeated != score:
            raise RuntimeError(f"Participant {subject} scoring is not deterministic: {score}, {repeated}")
        # Warm the same parameter-batch shape used by this two-parameter full Hessian.
        # Charge its first invocation to setup, rather than hiding compilation in the fit.
        batch_elapsed = 0.0
        if args.backend == "triton" and args.hessian_batch_size > 1 and not args.score_only:
            rows = np.tile([0.0, 0.9], (min(args.hessian_batch_size, 2 * len(PARAMETERS) ** 2), 1))
            scores, batch_elapsed = timed_call(pec.log_likelihood_batch, rows, inputs=inputs)
            np.testing.assert_array_equal(scores, np.full(len(rows), score))
        first_batch_score_seconds.append(batch_elapsed)
        original = pec.log_likelihood
        original_batch = pec.log_likelihood_batch

        def measured(*values, subject=subject, original=original, **kwargs):
            value, elapsed = timed_call(original, *values, **kwargs)
            if not np.isfinite(value):
                raise RuntimeError(f"Non-finite score for participant {subject}: {values}")
            calls[subject] += 1
            scalar_calls[subject] += 1
            score_seconds[subject] += elapsed
            if fit_started is not None and args.trace_output is not None:
                trace.append({"subject_index": subject, "subject": labels[subject],
                              **dict(zip(PARAMETERS, values)), "score": float(value), "seconds": elapsed})
            count = int(calls.sum())
            if fit_started is not None and count % 250 == 0:
                print(f"{args.backend}: {count} likelihood calls, "
                      f"{time.perf_counter() - fit_started:.1f}s fitting", flush=True)
            return value

        pec.log_likelihood = measured

        def measured_batch(rows, subject=subject, original=original_batch, **kwargs):
            values, elapsed = timed_call(original, rows, **kwargs)
            if not np.isfinite(values).all():
                raise RuntimeError(f"Non-finite batch score for participant {subject}")
            calls[subject] += len(rows)
            score_seconds[subject] += elapsed
            batch_calls[subject] += 1
            batch_rows[subject] += len(rows)
            batch_score_seconds[subject] += elapsed
            if fit_started is not None and args.trace_output is not None:
                for row, value in zip(rows, values):
                    trace.append({"subject_index": subject, "subject": labels[subject],
                                  **dict(zip(PARAMETERS, row)), "score": float(value),
                                  "seconds": elapsed / len(rows)})
            return values

        pec.log_likelihood_batch = measured_batch
        print(f"{args.backend}: {labels[subject]} built in {model_seconds[-1]:.3f}s, "
              f"first score {elapsed:.3f}s", flush=True)

    if args.score_only:
        probes = []
        saved_trace = pd.read_csv(args.probe_trace) if args.probe_trace is not None else None
        for subject, (pec, inputs) in enumerate(participants):
            if saved_trace is None:
                points = np.array([[0.0, 0.9]])
                indices = [None]
            else:
                rows = saved_trace.loc[saved_trace.subject_index == subject]
                if rows.empty or set(rows.subject) != {labels[subject]}:
                    raise ValueError(f"Probe trace does not match participant {labels[subject]}")
                # One midpoint per chronological stratum; no selection by runtime or score.
                indices = np.unique(((np.arange(args.probe_points) + 0.5)
                                     * len(rows) / args.probe_points).astype(int))
                points = rows.iloc[indices][list(PARAMETERS)].to_numpy()
            for index, theta in zip(indices, points):
                samples = [timed_call(pec.log_likelihood, *theta, inputs=inputs)
                           for _ in range(args.score_repeats)]
                if len({float(score) for score, _ in samples}) != 1:
                    raise RuntimeError("Scoring probes did not replay exactly")
                probes.append({"subject_index": subject, "subject": labels[subject],
                               "trace_index_within_subject": None if index is None else int(index),
                               "theta": theta.tolist(), "score": float(samples[0][0]),
                               "seconds": [seconds for _, seconds in samples]})
            print(f"{args.backend}: timed {len(points)} parameter vectors for {labels[subject]}", flush=True)
        save_json(args.output, {
            "mode": "scoring_only", "backend": args.backend,
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_diff_sha256": source_diff_sha256,
            "data_sha256": truth["data_sha256"], "threads": args.threads,
            "subjects": len(labels), "trials": len(data), "num_estimates": args.estimates,
            "simulation_seed": args.simulation_seed, "density_options": density_options,
            "model_seconds": model_seconds, "first_score_seconds": first_score_seconds,
            "first_scores": first_scores, "probes": probes,
            "probe_trace": None if args.probe_trace is None else str(args.probe_trace),
            "probe_trace_sha256": None if args.probe_trace is None else
                hashlib.sha256(args.probe_trace.read_bytes()).hexdigest(),
            "note": "Measured likelihood timings only; no hierarchical fit was run.",
        })
        print(f"Saved scoring measurements to {args.output}", flush=True)
        return

    pec, inputs = participants[0]
    warm_scores = [timed_call(pec.log_likelihood, 0.0, 0.9, inputs=inputs)[1]
                   for _ in range(args.score_repeats)]
    calls[:] = 0
    scalar_calls[:] = 0
    score_seconds[:] = 0.0

    def participant_factory(frame, subject_index=None):
        # Prebuilt and warmed to report compilation separately from EM work.
        pd.testing.assert_frame_equal(frame, frames[subject_index])
        return participants[subject_index]

    group_start = time.perf_counter()
    group = pnl.ParameterEstimationComposition(
        data=data, fit_method="hierarchical",
        hierarchical_options={
            "subject_id": "subject", "max_iterations": args.em_iterations,
            "tol": args.em_tol, "curvature": "full", "hessian_step": args.hessian_step,
            "hessian_batch_size": args.hessian_batch_size,
            "estep_options": {"maxiter": args.map_iterations,
                              "xatol": args.map_xatol, "fatol": args.map_fatol},
        },
        distributed_options={"pec_factory": participant_factory},
    )
    group_seconds = time.perf_counter() - group_start
    synchronize()
    fit_started = time.perf_counter()
    original_hessian = laplaceem.full_hessian

    def measured_hessian(*values, **kwargs):
        result, elapsed = timed_call(original_hessian, *values, **kwargs)
        hessian_seconds.append(elapsed)
        return result

    laplaceem.full_hessian = measured_hessian
    try:
        result = group.run()
    finally:
        laplaceem.full_hessian = original_hessian
    synchronize()
    fit_seconds = time.perf_counter() - fit_started
    theta = result.subject_parameters.to_numpy()
    actual = np.asarray(truth["theta"])
    result.em_history.to_csv(args.output.with_suffix(".history.csv"), index=False)
    result.subject_posteriors.to_csv(args.output.with_suffix(".participants.csv"), index=False)
    if args.trace_output is not None:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trace).to_csv(args.trace_output, index=False)
    report = {
        "backend": args.backend, "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_diff_sha256": source_diff_sha256,
        "data_sha256": truth["data_sha256"], "python": platform.python_version(),
        "torch": torch.__version__, "cpu": platform.processor(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "threads": args.threads, "subjects": len(labels), "trials": len(data),
        "num_estimates": args.estimates, "simulation_seed": args.simulation_seed,
        "likelihood": "matched smoothed marginal histogram", "density_options": density_options,
        "solver": result.settings, "model_seconds": model_seconds,
        "first_score_seconds": first_score_seconds, "first_scores": first_scores,
        "first_batch_score_seconds": first_batch_score_seconds,
        "group_setup_seconds": group_seconds, "warm_score_seconds": warm_scores,
        "warm_score_median_seconds": statistics.median(warm_scores),
        "fit_seconds": fit_seconds,
        "setup_plus_fit_seconds": sum(model_seconds) + sum(first_score_seconds)
            + sum(first_batch_score_seconds) + group_seconds + fit_seconds,
        "likelihood_calls": calls.tolist(), "likelihood_seconds": score_seconds.tolist(),
        "scalar_calls": scalar_calls.tolist(), "batch_calls": batch_calls.tolist(),
        "batch_rows": batch_rows.tolist(), "batch_score_seconds": batch_score_seconds.tolist(),
        "hessian_seconds": hessian_seconds,
        "trace_output": None if args.trace_output is None else str(args.trace_output),
        "iterations": result.n_iter, "converged": result.converged,
        "subject_converged": result.subject_converged.tolist(),
        "objective": result.objective, "group_mean_z": result.beta.tolist(),
        "group_variance_z": result.sigma.tolist(), "theta_estimated": theta.tolist(),
        "theta_true": actual.tolist(), "parameter_rmse": np.sqrt(np.mean((theta - actual) ** 2, axis=0)).tolist(),
        "posterior_covariance": result.posterior_covariance.tolist(),
        "history": result.em_history.to_dict(orient="records"),
        "note": "First scoring calls include JIT/cache lookup; no Triton disk-cache purge. "
                "Extra repeatability and timing probes are excluded from setup_plus_fit_seconds. "
                "Backends use different RNG streams and native simulation precision. "
                "Trace seconds for batched rows are elapsed batch time divided by row count.",
    }
    save_json(args.output, report)
    print(json.dumps({key: report[key] for key in [
        "backend", "fit_seconds", "setup_plus_fit_seconds", "iterations", "converged",
        "subject_converged", "likelihood_calls", "parameter_rmse",
    ]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("generate", "triton", "llvm"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--subjects", type=int, default=6)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--simulation-seed", type=int, default=100)
    parser.add_argument("--estimates", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--bin-width", type=float, default=0.05)
    parser.add_argument("--smoothing-sigma", type=float, default=1.0)
    parser.add_argument("--pseudocount", type=float, default=0.1)
    parser.add_argument("--em-iterations", type=int, default=40)
    parser.add_argument("--em-tol", type=float, default=0.01)
    parser.add_argument("--map-iterations", type=int, default=200)
    parser.add_argument("--map-xatol", type=float, default=0.01)
    parser.add_argument("--map-fatol", type=float, default=0.01)
    parser.add_argument("--hessian-step", type=float, default=0.25)
    parser.add_argument("--hessian-batch-size", type=int, default=16, help="Use 1 for scalar Hessian scoring")
    parser.add_argument("--score-repeats", type=int, default=7)
    parser.add_argument("--trace-output", type=Path, help="Record every fit likelihood evaluation as CSV")
    parser.add_argument("--score-only", action="store_true", help="Time scoring without running EM")
    parser.add_argument("--probe-trace", type=Path, help="Sample parameter vectors from a saved fit trace")
    parser.add_argument("--probe-points", type=int, default=5, help="Scoring probes per participant")
    args = parser.parse_args()
    if min(args.subjects, args.trials, args.estimates, args.threads,
           args.em_iterations, args.map_iterations, args.score_repeats, args.max_steps,
           args.probe_points, args.hessian_batch_size) < 1:
        parser.error("Counts must be positive")
    if args.backend != "generate" and args.output is None:
        parser.error("--output is required for a fit")
    if args.probe_trace is not None and not args.score_only:
        parser.error("--probe-trace requires --score-only")
    pnl.set_num_threads(args.threads)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.backend == "generate":
        generate(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    main()
