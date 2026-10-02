"""Compare conditional PEC scoring across separate LLVM and GPU checkouts.

Supply the DAWA full_lca_model_lc.py source with --model-source. Select the
PsyNeuLink implementation through PYTHONPATH. Run each backend separately to
avoid benchmark contention. Raw measurements are written only to --output.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time

import numpy as np
import pandas as pd
import psyneulink as pnl
from psyneulink.core.globals.utilities import set_global_seed

MODEL_PARAMETERS = dict(
    c_gain=10.0,
    c_leak=7.0,
    c_competition=3.0,
    c_bias=0.0,
    c_w=4.0,
    s_gain=5.0,
    s_leak=8.0,
    s_competition=8.0,
    s_bias=-0.45,
    d_gain=5.0,
    d_leak=8.0,
    d_competition=8.0,
    d_bias=-0.45,
    r_gain=5.0,
    r_leak=8.0,
    r_competition=8.0,
    r_bias=-0.45,
    c_noise=0.1,
    s_noise=0.1,
    d_noise=0.1,
    r_noise=0.1,
    r_threshold=0.3,
    non_decision_time=0.2,
    time_step_size=0.01,
    w1=1.0,
    w2=1.2,
    sdr_bias=-0.45,
    lc_base_gain=5.0,
    lc_scaling=1.0,
    lc_mode=0.9,
    lc_input=0.3,
    lc_threshold=0.5,
)


def node(model, name):
    return next(n for n in model.nodes if re.sub(r"-\d+$", "", n.name) == name)


def build_model(source, trials):
    spec = importlib.util.spec_from_file_location("dawa_benchmark_source", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.make_lca_model(**MODEL_PARAMETERS)
    tasks, stimuli = module.conflict_task_sequence(trials, seed=3)
    inputs = {node(model, "Task Input"): tasks, node(model, "Stimulus Input"): stimuli}
    inputs.update(
        {
            node(model, name): np.zeros((trials, 1))
            for name in ("Bias Mechanism", "w1 Mechanism", "w2 Mechanism")
        }
    )
    outputs = [node(model, name).output_port for name in ("DECISION_GATE", "RT_GATE")]
    return model, inputs, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["gpu", "llvm"], required=True)
    parser.add_argument("--particles", type=int, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument(
        "--source-revision", help="Revision of an extracted GPU checkout"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--replicates", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--precision", choices=["fp64", "fp32"], default="fp64")
    parser.add_argument("--bin-offset", type=float, default=0.0)
    parser.add_argument(
        "--save-clouds",
        action="store_true",
        help="Save LLVM predictive clouds in an extra evaluation",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (
        min(args.particles, args.trials, args.threads, args.repeats) < 1
        or args.replicates < 0
    ):
        parser.error(
            "Particle, trial, thread and timing counts must be positive; replicates must be nonnegative."
        )
    if Path(args.output).exists():
        parser.error(
            "Output exists; choose a new path to preserve previous measurements."
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.backend == "llvm":
        from psyneulink.core import llvm as pnlvm

        pnlvm.LLVMBuilderContext.default_float_ty = (
            pnlvm.ir.FloatType() if args.precision == "fp32" else pnlvm.ir.DoubleType()
        )
    pnl.set_num_threads(args.threads)
    set_global_seed(29)
    model, inputs, outputs = build_model(args.model_source, args.trials)
    observations = np.column_stack(
        (
            inputs[node(model, "Stimulus Input")][:, 1],
            0.805 + 0.02 * (np.arange(args.trials) % 6),
        )
    )
    initial = {
        name: np.asarray(node(model, name).output_port.parameters.value.get()).tolist()
        for name in [
            "Control Units\n[Color, Location]",
            "Stimulus Units\n[Red, Blue, Left, Right]",
            "Decision Units\n[Left, Right]",
            "Response Units\n[Left, Right]",
            "LC",
        ]
    }
    epsilon = 200.0 / 100200.0
    domain = [(args.bin_offset, 3.0 + args.bin_offset)]
    report = {
        "status": "running",
        "configuration": {**vars(args), "model_source": str(args.model_source)},
        "model_parameters": MODEL_PARAMETERS,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "simulation_precision": "fp32" if args.backend == "gpu" else args.precision,
        "pnl_source": pnl.__file__,
        "construction_seed": 29,
        "initial_outputs": initial,
        "observations": observations.tolist(),
        "inputs": {n.name: np.asarray(v).tolist() for n, v in inputs.items()},
        "source_sha256": hashlib.sha256(args.model_source.read_bytes()).hexdigest(),
        "contamination": epsilon,
        "timings": [],
        "replicates": [],
    }
    package = Path(pnl.__file__).resolve().parent
    revision = subprocess.run(
        ["git", "-C", str(package.parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    report["revision"] = args.source_revision or (
        revision.stdout.strip() if revision.returncode == 0 else None
    )
    implementation_files = (
        [
            "core/batched/compiler.py",
            "core/batched/likelihood.py",
            "core/batched/backend/triton/conditioned_ops.py",
        ]
        if args.backend == "gpu"
        else [
            "core/llvm/particle.py",
            "core/components/functions/nonstateful/particlefilter.py",
            "core/components/functions/nonstateful/fitfunctions.py",
        ]
    )
    report["implementation_sha256"] = {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in implementation_files
    }
    if args.backend == "gpu":
        import torch
        from psyneulink.core.batched import BatchedCompositionCompiler

        report["gpu"] = torch.cuda.get_device_name()
        plan = BatchedCompositionCompiler.compile(
            model, backend="triton", outputs=outputs, max_steps=4000
        )
        threshold = (
            node(model, "Response Units\n[Left, Right]").name + ".termination_threshold"
        )
        plan = plan.specialize_parameters(
            {p.name: p.default for p in plan.ir.params if p.name != threshold}
        )
        parameters = [{threshold: args.threshold}]
        data = pd.DataFrame(
            {
                "choice": pd.Categorical(observations[:, 0], categories=[0.0, 1.0]),
                "rt": observations[:, 1],
            }
        )
        gpu_function = pnl.PECOptimizationFunction(
            method="differential_evolution",
            max_iterations=1,
            batched_backend="triton",
            batched_max_steps=4000,
            batched_strict_truncation=True,
            batched_specialize_fixed_parameters=True,
            batched_seed=1000,
            batched_bins=100,
            batched_bin_range=domain,
            batched_smoothing_sigma=0.5,
            batched_pseudocount=args.particles / 100000.0,
            batched_categorical_cardinalities=[2],
            conditioned_likelihood=True,
            batched_triton_launch_options={
                "block_size": 32,
                "num_warps": 1,
                "trial_schedule": "independent",
                "normal_rng": "philox4x_fast_v1",
            },
        )
        gpu_pec = pnl.ParameterEstimationComposition(
            model=model,
            parameters={
                (
                    "termination_threshold",
                    node(model, "Response Units\n[Left, Right]"),
                ): [0.25, 0.7]
            },
            outcome_variables=list(outputs),
            data=data,
            num_estimates=args.particles,
            initial_seed=1000,
            same_seed_for_all_parameter_combinations=True,
            optimization_function=gpu_function,
        )

        def score(seed, diagnostics=False, save_clouds=False):
            if not diagnostics:
                gpu_pec.controller.function.batched_seed = seed
                torch.cuda.synchronize()
                start = time.perf_counter()
                value = gpu_pec.log_likelihood(args.threshold, inputs=inputs)
                torch.cuda.synchronize()
                return float(value), time.perf_counter() - start, None, None
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = plan.conditioned_log_likelihood(
                inputs,
                parameters,
                args.particles,
                data=observations,
                categorical_dims=[0],
                bins=100,
                bin_range=domain,
                smoothing_sigma=0.5,
                pseudocount=args.particles / 100000.0,
                categorical_cardinalities=[2],
                seed=seed,
                strict_truncation=True,
                triton_launch_options={
                    "block_size": 32,
                    "num_warps": 1,
                    "trial_schedule": "independent",
                    "normal_rng": "philox4x_fast_v1",
                },
                return_diagnostics=diagnostics,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if diagnostics:
                value, diag = result
                log_densities = np.log(
                    np.asarray(diag["per_trial_densities"], dtype=np.float64)
                ).reshape(-1)
                ess = np.asarray(diag["effective_sample_size"]).reshape(-1)
            else:
                value, log_densities, ess = result, None, None
            return float(value), elapsed, log_densities, ess
    else:
        from psyneulink.core.globals.sampleiterator import SampleIterator
        from psyneulink.core.globals.context import Context

        data = pd.DataFrame(
            {
                "choice": pd.Categorical(observations[:, 0], categories=[0.0, 1.0]),
                "rt": observations[:, 1],
            }
        )
        pec = pnl.ParameterEstimationComposition(
            model=model,
            parameters={
                (
                    "termination_threshold",
                    node(model, "Response Units\n[Left, Right]"),
                ): [0.25, 0.7]
            },
            outcome_variables=list(outputs),
            data=data,
            num_estimates=args.particles,
            initial_seed=1000,
            same_seed_for_all_parameter_combinations=True,
            optimization_function=pnl.PECOptimizationFunction(
                method=None,
                max_iterations=1,
                conditioned_likelihood=True,
                likelihood_options={
                    "kernel": "histogram",
                    "bins": 100,
                    "bin_range": domain,
                    "smoothing_sigma": 0.5,
                    "categorical_values": [[0.0, 1.0]],
                    "contamination_probability": epsilon,
                },
            ),
        )
        ocm = pec.controller
        context = Context(execution_id=None, composition=pec)
        random_dim = ocm.function.parameters.randomization_dimension.get()
        report["noise_stream_policy"] = ocm.parameters.noise_stream_policy.get()
        report["random_streams"] = len(ocm.random_variables)

        def score(seed, diagnostics=False, save_clouds=False):
            # Disjoint seed blocks between repeats. Identical integers across
            # backends do not imply shared random draws.
            ocm._seed_counter = seed
            ocm.function.search_space[random_dim] = SampleIterator(
                ocm.gen_new_seed_sequence(context)
            )
            start = time.perf_counter()
            result = pec.log_likelihood(
                args.threshold,
                inputs=inputs,
                return_sim_data=save_clouds,
                context=context,
            )
            elapsed = time.perf_counter() - start
            if save_clouds:
                value, clouds = result
                np.savez_compressed(
                    Path(args.output).with_suffix(".npz"),
                    clouds=clouds,
                    observations=observations,
                )
            else:
                value = result
            if diagnostics:
                diag = ocm.function.likelihood_diagnostics
                log_densities = np.asarray(diag["per_trial_log_densities"]).copy()
                ess = np.asarray(diag["effective_sample_size"]).copy()
            else:
                log_densities, ess = None, None
            return float(value), elapsed, log_densities, ess

    print(
        json.dumps(
            {
                "stage": "constructed",
                "backend": args.backend,
                "source": pnl.__file__,
                "particles": args.particles,
            }
        ),
        flush=True,
    )
    value, elapsed, _, _ = score(1000)
    report["first_call_seconds"] = elapsed
    report["warmup_score"] = value
    print(
        json.dumps({"stage": "warmup", "seconds": elapsed, "score": value}), flush=True
    )
    for repeat in range(args.repeats):
        repeated, elapsed, _, _ = score(1000)
        assert repeated == value
        report["timings"].append(elapsed)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(
            json.dumps({"stage": "timing", "index": repeat, "seconds": elapsed}),
            flush=True,
        )
    report["median_seconds"] = statistics.median(report["timings"])
    if args.backend == "gpu":
        diagnostic_score, _, _, _ = score(1000, diagnostics=True)
        assert diagnostic_score == report["warmup_score"]
        report["public_api_matches_diagnostic_score"] = True
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(
        json.dumps({"stage": "timed", "median_seconds": report["median_seconds"]}),
        flush=True,
    )
    if args.backend == "llvm" and args.save_clouds:
        replay, _, _, _ = score(1000, save_clouds=True)
        assert replay == report["warmup_score"]
    for i in range(args.replicates):
        seed = 20000 + max(100000, args.particles) * i
        value, elapsed, ld, ess = score(seed, diagnostics=True)
        assert np.isfinite(value)
        report["replicates"].append(
            {
                "seed": seed,
                "score": value,
                "seconds": elapsed,
                "log_densities": ld.tolist(),
                "ess": ess.tolist(),
            }
        )
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                {"stage": "replicate", "index": i, "score": value, "seconds": elapsed}
            ),
            flush=True,
        )
    report["status"] = "complete"
    Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
