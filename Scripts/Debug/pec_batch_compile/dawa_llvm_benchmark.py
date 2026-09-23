"""Compare DAWA batch simulation with PEC's threaded LLVM simulation path.

Run each backend/case in a fresh process. Timings include input preparation and
host results, but exclude likelihood estimation and optimization. Both backends
use the default recurrent schedule from dawa_batched_simulation.py.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
import platform
from pathlib import Path
import statistics
import time

import numpy as np
import pandas as pd
import psyneulink as pnl
from psyneulink.core.batched import BatchedCompositionCompiler

from dawa_batched_simulation import SOURCE, build_model, fit_surface, node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("triton", "llvm"), required=True)
    parser.add_argument("--data", type=Path, default=SOURCE.parent / "flanker_data_part1.csv")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--estimates", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2000, help="Triton scheduler cap; LLVM uses its native scheduler")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--threshold", type=float, default=.3)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.trials, args.estimates, args.repeats, args.threads) < 1:
        parser.error("Trials, estimates, repeats and threads must be positive")
    pnl.set_num_threads(args.threads)
    data = pd.read_csv(args.data)
    data = data[(data.subject_nr == args.subject) & data.PrevCongruency.notna()].iloc[:args.trials].copy()
    if len(data) != args.trials:
        parser.error(f"Only {len(data)} eligible trials available for subject {args.subject}")
    composition, inputs, outputs = build_model(trials=args.trials, deterministic=args.deterministic)
    inputs[node(composition, "Task Input")] = data[["T1", "T2"]].to_numpy()
    inputs[node(composition, "Stimulus Input")] = data[["S1", "S2", "S3", "S4"]].to_numpy()
    surface = fit_surface(composition)
    values = [args.threshold if parameter == "termination_threshold" else bounds[2]
              for (parameter, _), bounds in surface.items()]
    parameter_set = {f"{owner.name}.{parameter}": value
                     for (parameter, owner), value in zip(surface, values)}
    setup_start = time.perf_counter()
    observed = data[["decision", "response_time"]].copy()
    observed["decision"] = pd.Categorical(observed.decision, categories=[0., 1.])
    pec = pnl.ParameterEstimationComposition(
        nodes=composition,
        parameters={key: np.array(bounds[:2]) for key, bounds in surface.items()},
        outcome_variables=list(outputs), data=observed,
        optimization_function=pnl.PECOptimizationFunction(method="differential_evolution", max_iterations=1),
        num_estimates=args.estimates, initial_seed=args.seed, same_seed_for_all_parameter_combinations=True,
    )
    if args.backend == "llvm":
        pec.controller.parameters.comp_execution_mode.set("LLVM")
        pec.controller.function.set_pec_objective_function(lambda samples: 0.)

        def run():
            _, samples = pec.log_likelihood(*values, inputs=inputs, return_sim_data=True)
            return np.asarray(samples)
    else:
        plan = BatchedCompositionCompiler.compile(composition, backend="triton", outputs=outputs,
                                                 max_steps=args.max_steps,
                                                 ignored_control_nodes=tuple(pec.pec_control_mechs.values()))

        def run():
            return plan.run(inputs, [parameter_set], args.estimates, seed=args.seed,
                            strict_truncation=True).values[0, 0]

    setup_seconds = time.perf_counter() - setup_start
    report = {
        "backend": args.backend, "subject": args.subject, "trials": args.trials,
        "estimates": args.estimates, "candidates": 1, "threads": pnl.get_num_threads(),
        "seed": args.seed, "deterministic": args.deterministic, "schedule": "recurrent",
        "max_steps": args.max_steps, "parameter_values": parameter_set,
        "data_file": str(args.data), "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "model_source": str(SOURCE), "model_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "platform": platform.platform(), "setup_seconds": setup_seconds,
        "cpu": next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")),
        "versions": {package: version(package) for package in ("psyneulink", "llvmlite", "triton", "torch")},
        "scope": "simulation including preparation and host output; no likelihood or optimizer",
        "warm_seconds": [], "requested_warm_repeats": args.repeats,
        "same_seed_for_all_parameter_combinations": True,
    }
    if args.backend == "triton":
        import torch
        report["gpu"] = torch.cuda.get_device_name()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    for index in range(args.repeats + 1):
        print(f"Starting {args.backend}: trials={args.trials}, estimates={args.estimates}, run={index}", flush=True)
        start = time.perf_counter()
        samples = run()
        elapsed = time.perf_counter() - start
        if samples.shape != (args.trials, args.estimates, 2) or not np.all(np.isfinite(samples)):
            raise AssertionError(f"Invalid simulation output: {samples.shape}")
        if not np.all(np.isin(samples[..., 0], [0., 1.])) or np.any(samples[..., 1] <= .2):
            raise AssertionError("Invalid decisions or response times")
        if previous is not None:
            np.testing.assert_array_equal(samples, previous)
        previous = samples.copy()
        if index == 0:
            report["first_seconds"] = elapsed
        else:
            report["warm_seconds"].append(elapsed)
            report["median_seconds"] = statistics.median(report["warm_seconds"])
        report.update(
            shape=list(samples.shape), dtype=str(samples.dtype), mean_rt=float(samples[..., 1].mean()),
            mean_decision=float(samples[..., 0].mean()),
            rt_quantiles=np.quantile(samples[..., 1], [.1, .5, .9, .99, 1.]).tolist(),
            first_trial_samples=samples[0, :min(8, args.estimates)].tolist(),
            mean_rt_by_trial=samples[..., 1].mean(axis=1).tolist(),
            mean_decision_by_trial=samples[..., 0].mean(axis=1).tolist(),
            mean_rt_estimate_standard_error=(float(samples[..., 1].mean(axis=0).std(ddof=1)
                                                   / np.sqrt(args.estimates)) if args.estimates > 1 else None),
            mean_decision_estimate_standard_error=(float(samples[..., 0].mean(axis=0).std(ddof=1)
                                                         / np.sqrt(args.estimates)) if args.estimates > 1 else None),
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Completed run={index}: {elapsed:.6f}s, mean RT={report['mean_rt']:.6f}", flush=True)
    print(json.dumps({key: report[key] for key in ("backend", "trials", "estimates", "first_seconds",
                                                  "median_seconds", "mean_rt", "mean_decision")}), flush=True)


if __name__ == "__main__":
    main()
