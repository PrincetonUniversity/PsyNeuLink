"""Compare DAWA batch simulation with PEC's threaded LLVM simulation path.

Run each backend/case in a fresh process. Timings include input preparation and
host results, but exclude likelihood estimation and optimization. Both backends
use the default recurrent schedule from dawa_batched_simulation.py.

For multiple noisy components, --independent-noise-streams gives each LLVM PEC
random variable a distinct seed range. PEC otherwise broadcasts one seed to all
components, correlating their draws; Triton always uses separate streams.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
import platform
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import pandas as pd
import psyneulink as pnl
from psyneulink.core.batched import BatchedCompositionCompiler
from psyneulink.core.globals.utilities import set_global_seed

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
    parser.add_argument("--independent-noise-streams", action="store_true",
                        help="Give LLVM PEC's random variables distinct seed offsets; Triton already separates streams")
    parser.add_argument("--specialize-fixed-parameters", action="store_true",
                        help="Triton: compile non-fitted parameter defaults as constants")
    parser.add_argument("--normal-rng", choices=("legacy", "philox4x_v1", "philox4x_fast_v1"),
                        default="philox4x_v1", help="Triton Gaussian generator")
    parser.add_argument("--trial-schedule", choices=("synchronized", "independent"), default="synchronized")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--source-revision", help="Base commit when running a source snapshot without .git")
    for prefix in ("c", "s", "d", "r"):
        parser.add_argument(f"--{prefix}-noise", type=float,
                            help="Override this LCA's Gaussian noise standard deviation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.trials, args.estimates, args.repeats, args.threads) < 1:
        parser.error("Trials, estimates, repeats and threads must be positive")
    noise = {f"{prefix}_noise": getattr(args, f"{prefix}_noise") for prefix in ("c", "s", "d", "r")}
    if any(value is not None and (not np.isfinite(value) or value < 0.) for value in noise.values()):
        parser.error("Noise standard deviations must be finite and nonnegative")
    # Persistent Gaussian control retains construction-time RESULT activity.
    # Reproduce that same initial condition in the separate backend processes.
    set_global_seed(args.seed)
    pnl.set_num_threads(args.threads)
    data = pd.read_csv(args.data)
    data = data[(data.subject_nr == args.subject) & data.PrevCongruency.notna()].iloc[:args.trials].copy()
    if len(data) != args.trials:
        parser.error(f"Only {len(data)} eligible trials available for subject {args.subject}")
    composition, inputs, outputs = build_model(trials=args.trials, deterministic=args.deterministic, **noise)
    inputs[node(composition, "Task Input")] = data[["T1", "T2"]].to_numpy()
    inputs[node(composition, "Stimulus Input")] = data[["S1", "S2", "S3", "S4"]].to_numpy()
    surface = fit_surface(composition)
    values = [args.threshold if parameter == "termination_threshold" else bounds[2]
              for (parameter, _), bounds in surface.items()]
    parameter_set = {f"{owner.name}.{parameter}": value
                     for (parameter, owner), value in zip(surface, values)}
    launch = dict(block_size=args.block_size, num_warps=args.num_warps,
                  normal_rng=args.normal_rng, trial_schedule=args.trial_schedule)
    fixed_parameters = {}
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
    seed_offsets = {}
    if args.independent_noise_streams and args.backend == "llvm":
        # OCM normally broadcasts the SAME seed to every random variable. With
        # several noisy LCAs this correlates their draws. Keep the native PEC
        # threaded evaluator, but separate its per-component seed ranges for
        # comparison with the batch compiler's independent component streams.
        stride = 1 << max(20, (args.seed + args.estimates).bit_length())
        for index, variable in enumerate(pec.controller.random_variables):
            port = variable.parameters.seed.port
            if len(port.mod_afferents) != 1:
                raise AssertionError(f"Expected one randomization control projection for {port.full_name}")
            projection = port.mod_afferents[0]
            offset = index * stride
            projection.function.parameters.intercept.set(float(offset))
            seed_offsets[port.full_name] = offset
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
        if args.specialize_fixed_parameters:
            plan = plan.specialize_parameters({p.name: p.default for p in plan.ir.params
                                               if p.name not in parameter_set})
            fixed_parameters = plan.fixed_parameters

        def run():
            return plan.run(inputs, [parameter_set], args.estimates, seed=args.seed,
                            strict_truncation=True, triton_launch_options=launch).values[0, 0]

    setup_seconds = time.perf_counter() - setup_start
    report = {
        "backend": args.backend, "subject": args.subject, "trials": args.trials,
        "estimates": args.estimates, "candidates": 1, "threads": pnl.get_num_threads(),
        "seed": args.seed, "deterministic": args.deterministic, "schedule": "recurrent",
        "noise_overrides": noise, "time_step_size": .01,
        "lc_time_step_size": .02, "lc_internal_steps_per_pass": 10,
        "source_revision": args.source_revision or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SOURCE.parent, text=True).strip(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "triton_launch_options": launch if args.backend == "triton" else None,
        "fixed_parameters": fixed_parameters,
        "noise_stream_policy": "independent" if args.backend == "triton" or args.independent_noise_streams else "shared_seed",
        "llvm_seed_offsets": seed_offsets,
        "control_initial_activity": node(composition, "Control Units\n[Color, Location]").output_port.defaults.value.tolist(),
        "max_steps": args.max_steps, "parameter_values": parameter_set,
        "data_file": str(args.data), "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "model_source": str(SOURCE), "model_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "platform": platform.platform(), "setup_seconds": setup_seconds,
        "cpu": next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")),
        "versions": {"psyneulink": getattr(pnl, "__version__", "source snapshot; see source_revision"),
                     **{package: version(package) for package in ("llvmlite", "triton", "torch")}},
        "scope": "simulation including preparation and host output; no likelihood or optimizer",
        "warm_seconds": [], "requested_warm_repeats": args.repeats,
        "same_seed_for_all_parameter_combinations": True,
    }
    if args.backend == "triton":
        import torch
        report["gpu"] = torch.cuda.get_device_name()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({**report, "status": "running"}, indent=2) + "\n")
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
            status="complete" if index == args.repeats else "running",
            shape=list(samples.shape), dtype=str(samples.dtype), mean_rt=float(samples[..., 1].mean()),
            mean_decision=float(samples[..., 0].mean()),
            rt_quantiles=np.quantile(samples[..., 1], [.1, .5, .9, .99, 1.]).tolist(),
            first_trial_samples=samples[0, :min(8, args.estimates)].tolist(),
            mean_rt_by_trial=samples[..., 1].mean(axis=1).tolist(),
            mean_decision_by_trial=samples[..., 0].mean(axis=1).tolist(),
            rt_standard_error_by_trial=(samples[..., 1].std(axis=1, ddof=1)
                                       / np.sqrt(args.estimates)).tolist() if args.estimates > 1 else None,
            decision_standard_error_by_trial=(samples[..., 0].std(axis=1, ddof=1)
                                             / np.sqrt(args.estimates)).tolist() if args.estimates > 1 else None,
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
