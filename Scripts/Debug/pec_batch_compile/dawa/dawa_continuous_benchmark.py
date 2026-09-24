"""Compare threaded CPU and fused GPU likelihoods with continuous GPU sampling.

Uses the recorded two-condition, 4,000-observation workload without fitting any
parameters. Compilation and one warmup per case are excluded. Sampling timings
include deterministic paths and transfers; histogram scoring is reported
separately. These are different numerical estimators, not equal-accuracy fits.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import time

import numpy as np
import torch

from dawa_continuous_recovery import observation_counts, probabilities
from dawa_continuous_study import CONDITIONS
from dawa_likelihood import PARAMETER_NAMES, continuous_flux
from dawa_likelihood.continuous_native import continuous_path_native
from dawa_likelihood.continuous_solver import ContinuousConfig


RESULTS = Path(__file__).with_name("dawa_likelihood") / "study_results"


@contextmanager
def native_timings(enabled=True):
    """Time native entry points; restore them even if the evaluation fails."""
    if not enabled:
        from dawa_likelihood import continuous_flux_gpu
        substeps = set()
        original = continuous_flux_gpu.gpu_flux_block

        def observed(mass, rates, steps, dt, **kwargs):
            substeps.add(steps)
            return original(mass, rates, steps, dt, **kwargs)

        continuous_flux_gpu.gpu_flux_block = observed
        try:
            yield {}, {}, substeps
        finally:
            continuous_flux_gpu.gpu_flux_block = original
        return
    module = continuous_flux.module()
    originals = {name: getattr(module, name) for name in
                 ("coefficients", "forward", "coefficients_vjp", "backward")}
    elapsed, calls, substeps = defaultdict(float), Counter(), set()
    for name, original in originals.items():
        def timed(*args, _name=name, _original=original):
            start = time.perf_counter()
            result = _original(*args)
            elapsed[_name] += time.perf_counter() - start
            calls[_name] += 1
            if _name == "forward":
                substeps.add(args[2])
            return result
        setattr(module, name, timed)
    try:
        yield elapsed, calls, substeps
    finally:
        for name, original in originals.items():
            setattr(module, name, original)


def summarize(runs):
    result = {"seconds": [r["seconds"] for r in runs],
              "median_seconds": statistics.median(r["seconds"] for r in runs)}
    if "phase_seconds" in runs[0]:
        result["phase_median_seconds"] = {
            name: statistics.median(r["phase_seconds"][name] for r in runs)
            for name in runs[0]["phase_seconds"]}
        result["native_calls"] = runs[-1]["native_calls"]
        result["substeps"] = runs[-1]["substeps"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/dawa_continuous_benchmark.json"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--points", type=int, default=129)
    parser.add_argument("--estimates", type=int, default=100000)
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--recompute-rates", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=1, help="Threads in native coefficient and density kernels")
    parser.add_argument("--flux-backend", choices=("native", "triton"), default="native")
    parser.add_argument("--no-gpu-graphs", action="store_true", help="Launch Triton time steps directly, without CUDA graphs")
    parser.add_argument("--parameters", nargs=7, type=float)
    parser.add_argument("--compare", type=Path, help="Check probabilities and gradients against a prior report")
    args = parser.parse_args()
    if min(args.repeats, args.estimates, args.cpu_threads) < 1:
        parser.error("Positive repeat and estimate counts required")
    torch.set_num_threads(1)
    workload = json.loads((RESULTS / "recovery_two.json").read_text())
    previous = json.loads((RESULTS / "performance.json").read_text())
    values = args.parameters or [previous["parameters"][name] for name in PARAMETER_NAMES]
    counts = torch.tensor(workload["training_counts"], dtype=torch.float64)
    edges = torch.tensor(workload["rt_edges"], dtype=torch.float64)
    selected = counts > 0
    cfg = ContinuousConfig(points=args.points, ode_backend="generated", flux_backend=args.flux_backend,
                           recompute_rates=args.recompute_rates, cpu_threads=args.cpu_threads,
                           gpu_graphs=not args.no_gpu_graphs)
    device = "cuda" if cfg.flux_backend == "triton" else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("The Triton flux backend requires CUDA")
    device_counts, device_edges, device_selected = counts.to(device), edges.to(device), selected.to(device)
    conditions = [1, 3]
    report = {
        "scope": __doc__, "platform": platform.platform(), "torch": str(torch.__version__),
        "cpu": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name")), platform.processor()) if Path("/proc/cpuinfo").exists()
                else platform.processor(),
        "threads": args.cpu_threads, "torch_threads": 1, "device": device,
        "dtype": "float64", "repeats": args.repeats, "warmup_calls_per_case": 1,
        "config": asdict(cfg), "parameters": dict(zip(PARAMETER_NAMES, values)),
        "conditions": [CONDITIONS[i][0] for i in conditions], "rt_edges": edges.tolist(),
        "training_counts": counts.tolist(), "training_observations": int(counts.sum()),
        "native_source_sha256": hashlib.sha256(
            Path(continuous_flux.__file__).with_name("continuous_flux_cpu.cpp").read_bytes()).hexdigest(),
    }
    if device == "cuda":
        report["gpu_source_sha256"] = hashlib.sha256(
            Path(continuous_flux.__file__).with_name("continuous_flux_gpu.py").read_bytes()).hexdigest()

    def save():
        if platform.system() == "Linux":
            report["peak_process_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
        if device == "cuda":
            report["gpu"] = torch.cuda.get_device_name()
            report["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024.**2
            report["peak_cuda_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024.**2
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def evaluate(gradient):
        p = torch.tensor(values, dtype=torch.float64, device=device, requires_grad=gradient)
        with native_timings(device == "cpu") as (phases, calls, substeps), torch.set_grad_enabled(gradient):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            prob = probabilities(p, conditions, device_edges, cfg)
            loss = -(device_counts[device_selected] * prob[device_selected].log()).sum() / device_counts.sum()
            grad = torch.autograd.grad(loss, p)[0] if gradient else None
            if device == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
        return {"seconds": seconds, "phase_seconds": dict(phases), "native_calls": dict(calls),
                "substeps": sorted(substeps), "mean_negative_log_likelihood": float(loss.detach()),
                "probabilities": prob.detach().tolist(), "gradient": grad.tolist() if gradient else None}

    for gradient, name in ((False, "forward"), (True, "value_and_gradient")):
        warmup = evaluate(gradient)
        runs = []
        for _ in range(args.repeats):
            runs.append(evaluate(gradient))
            print(f"{name}: {runs[-1]['seconds']:.6f} s", flush=True)
        report[name] = summarize(runs)
        report[name]["warmup_evaluation_seconds"] = warmup["seconds"]
        report[name].update({key: runs[-1][key] for key in
                             ("mean_negative_log_likelihood", "probabilities", "gradient")})
        save()

    if not args.skip_sampling:
        from dawa_likelihood.continuous_monte_carlo import simulate_continuous
        if not torch.cuda.is_available():
            parser.error("CUDA required for sampling; use --skip-sampling for CPU-only profiling")
        p = torch.tensor(values, dtype=torch.float64)
        sample_dt = .00025
        # Same decision horizon as probabilities(), including its extra output cell.
        from dawa_likelihood.model import PARAMETER_BOUNDS
        decision_steps = int(np.ceil((float(edges[-1]) - PARAMETER_BOUNDS[1][0]) / cfg.time_step)) + 1
        horizon = decision_steps * cfg.time_step
        sample_steps = int(np.ceil(horizon / sample_dt))

        def sample():
            torch.cuda.synchronize()
            start = time.perf_counter()
            samples = []
            with torch.no_grad():
                for ci in conditions:
                    _, task, stimulus = CONDITIONS[ci]
                    path = continuous_path_native(p, task, stimulus, steps=sample_steps,
                                                  time_step=sample_dt, ode_step=sample_dt,
                                                  clock_ratio=cfg.lc_clock_ratio)
                    samples.append(simulate_continuous(path, p, estimates=args.estimates, seed=43))
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            start = time.perf_counter()
            prob = np.stack([observation_counts(s, float(p[1]), edges.numpy()) / len(s) for s in samples])
            observed = prob[selected.numpy()]
            zero_bins = int(np.count_nonzero(observed == 0))
            loss = None if zero_bins else float(-(counts.numpy()[selected.numpy()] * np.log(observed)).sum() / float(counts.sum()))
            scoring_seconds = time.perf_counter() - start
            return {"seconds": seconds, "scoring_seconds": scoring_seconds,
                    "mean_negative_log_likelihood": loss, "zero_observed_bins": zero_bins,
                    "probabilities": prob.tolist()}

        sample()
        runs = [sample() for _ in range(args.repeats)]
        report["continuous_sampling"] = {
            **summarize(runs), "gpu": torch.cuda.get_device_name(), "response_dtype": "float32",
            "estimates_per_condition": args.estimates, "time_step": sample_dt, "ode_step": sample_dt,
            "decision_horizon": sample_steps * sample_dt, "seed": 43,
            "histogram_scoring_median_seconds": statistics.median(r["scoring_seconds"] for r in runs),
            "sample_and_score_median_seconds": statistics.median(r["seconds"] + r["scoring_seconds"] for r in runs),
            "mean_negative_log_likelihood": runs[-1]["mean_negative_log_likelihood"],
            "zero_observed_bins": runs[-1]["zero_observed_bins"],
            "maximum_joint_cdf_difference_from_direct": float(np.max(np.abs(
                np.asarray(runs[-1]["probabilities"])[:, :-1].reshape(2, -1, 2).cumsum(1)
                - np.asarray(report["forward"]["probabilities"])[:, :-1].reshape(2, -1, 2).cumsum(1)))),
            "limitations": "Empirical RT-bin probabilities without smoothing or a probability floor; no parameter gradients. Different discretization and precision from the direct solver.",
        }
        print(f"GPU sampling: {report['continuous_sampling']['median_seconds']:.6f} s", flush=True)
        save()

    if args.compare:
        reference = json.loads(args.compare.read_text())
        backend_options = {"flux_backend", "cpu_threads", "gpu_graphs"}
        if {k: v for k, v in report["config"].items() if k not in backend_options} != \
                {k: v for k, v in reference["config"].items() if k not in backend_options}:
            raise ValueError("Comparison numerical settings differ")
        for key in ("parameters", "conditions", "rt_edges", "training_counts"):
            if report[key] != reference[key]:
                raise ValueError(f"Comparison workload differs: {key}")
        report["comparison"] = {"reference": str(args.compare)}
        for name in ("forward", "value_and_gradient"):
            actual, expected = report[name], reference[name]
            np.testing.assert_allclose(actual["probabilities"], expected["probabilities"], atol=2.e-12, rtol=2.e-10)
            if actual["gradient"] is not None:
                np.testing.assert_allclose(actual["gradient"], expected["gradient"], atol=2.e-9, rtol=2.e-8)
            report["comparison"][name + "_speedup"] = expected["median_seconds"] / actual["median_seconds"]
        save()


if __name__ == "__main__":
    main()
