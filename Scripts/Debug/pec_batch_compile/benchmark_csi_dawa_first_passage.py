"""Matched continuous CSI/DAWA first-passage costs and numerical refinement.

One candidate, two conditional trials, the same horizon and noise, 100,000
float32 GPU samples/trial, and float64 direct solves. Nonlinear deterministic
paths use each model's existing native equations. Path construction is timed
separately from the stochastic stage. This is not a sequential subject fit.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "dawa"), str(ROOT / "csi/csi_fit")]

from direct_likelihood.continuous_monte_carlo import simulate_continuous_drift  # noqa: E402
from direct_likelihood.native import native_lca_drift_path  # noqa: E402
from dawa_continuous_study import CONDITIONS, empirical_cdf  # noqa: E402
from dawa_likelihood import PARAMETER_NAMES  # noqa: E402
from dawa_likelihood.continuous_model import ContinuousPath  # noqa: E402
from dawa_likelihood.continuous_native import continuous_path_native  # noqa: E402
from dawa_likelihood.continuous_monte_carlo import simulate_continuous  # noqa: E402
from dawa_likelihood.continuous_solver import ContinuousConfig, ContinuousResponseSolver  # noqa: E402
from psyneulink.core.batched.numerical.first_passage import MovingBoundaryDDMSolver  # noqa: E402


def csi_paths(dt, horizon, trials):
    """Two fresh trials after a 1 s ITI and 50 ms cue; repeated without reuse."""
    task = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64).repeat(trials // 2, 1)
    gain = torch.tensor([10., 20.], dtype=torch.float64).repeat(trials // 2)
    stimulus = torch.tensor([[1., 0., 0., 1.]], dtype=torch.float64).repeat(trials, 1)
    correct = torch.tensor([1., -1.], dtype=torch.float64).repeat(trials // 2)

    def integrate(state, task_input, steps, step_size):
        return native_lca_drift_path(state, task_input, gain, stimulus, correct, steps=steps,
                                     step_size=step_size, leak=12., competition=3.)

    state = torch.zeros_like(task)
    _, state = integrate(state, torch.zeros_like(task), 2000, .0005)
    _, state = integrate(state, task, 100, .0005)
    drift, _ = integrate(state, task, round(horizon / dt), dt)
    threshold = torch.full((trials,), .12, dtype=torch.float64)
    collapse = torch.tensor([-.01, -.02], dtype=torch.float64).repeat(trials // 2)
    return drift, threshold, collapse


def dawa_paths(parameters, dt, horizon, trials):
    return [continuous_path_native(parameters, CONDITIONS[ci][1], CONDITIONS[ci][2],
                                   steps=round(horizon / dt), time_step=dt, ode_step=min(dt, .0005))
            for ci in (1, 3) * (trials // 2)]


def cuda_path(path):
    return ContinuousPath(*(v.cuda() for v in (path.inputs, path.gain, path.gain_rate, path.states)), path.time_step)


def csi_direct(paths, points, dt, horizon, window):
    drift, threshold, collapse = paths
    solver = MovingBoundaryDDMSolver(time_step=dt, spatial_points=points, native_forward=True)
    result = solver.solve_observation_batch(drift=drift, threshold=threshold, collapse_rate=collapse,
                                            interval_low=threshold.new_full(threshold.shape, horizon - window),
                                            interval_high=threshold.new_full(threshold.shape, horizon),
                                            choice=threshold.new_zeros(threshold.shape))
    assert not bool(result.invalid_boundary.any())
    assert float(result.mass_error.max()) < 1.e-8
    assert float(result.minimum_density.min()) >= -1.e-12
    return {"probabilities": torch.stack((result.lower_probability, result.upper_probability,
                                          result.survival_probability), dim=1).tolist(),
            "window_choice0_probability": result.probability.tolist(), "substeps": 1}


def dawa_direct(paths, parameters, points, dt, horizon, window, backend, threads):
    cfg = ContinuousConfig(points=points, time_step=dt, ode_backend="generated", flux_backend=backend,
                           cpu_threads=threads, recompute_rates=False)
    solver = ContinuousResponseSolver(cfg)
    results = []
    device_p = parameters.cuda() if backend == "triton" else parameters
    for path in paths:
        distribution = solver.solve(cuda_path(path) if backend == "triton" else path, device_p)
        assert float(distribution.mass_error) < 1.e-8
        assert float(distribution.minimum_mass) >= -1.e-12
        assert float(distribution.lower_loss) < 1.e-7
        results.append(distribution)
    return {"probabilities": [torch.cat((d.choice_mass.sum(0), d.survival.reshape(1))).cpu().tolist() for d in results],
            "window_choice0_probability": [float(d.interval_probability(0, horizon - window, horizon)) for d in results],
            "substeps": max(d.substeps for d in results)}


def sample(model, paths, parameters, dt, estimates, seed):
    # Split only to respect the samplers' checked uint32 RNG offset domains.
    steps = paths[0].inputs.shape[0] - 1 if model == "dawa" else paths[0].shape[1]
    draws = 4 if model == "dawa" else 3
    limit = (2**32 - 1) // (draws * steps)
    batches = []
    for offset in range(0, estimates, limit):
        count = min(limit, estimates - offset)
        local_seed = seed + offset * 17
        if model == "csi":
            batch = simulate_continuous_drift(*paths, time_step=dt, estimates=count, seed=local_seed)
        else:
            batch = np.stack([simulate_continuous(path, parameters, estimates=count, seed=local_seed + i * 104729)
                              for i, path in enumerate(paths)])
        batches.append(batch)
    return np.concatenate(batches, axis=1)


def summarize_samples(samples, horizon, window):
    return {"probabilities": [[float(np.mean(row[:, 0] == c)) for c in (0, 1, -1)] for row in samples],
            "window_choice0_probability": [float(np.mean((row[:, 0] == 0) & (row[:, 1] >= horizon - window)
                                                          & (row[:, 1] < horizon))) for row in samples],
            "mean_decision_time_including_censoring": samples[:, :, 1].mean(1).tolist()}


def measure(function, repeats):
    function()  # Compilation, graph capture, and allocation warmup excluded.
    seconds = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = function()
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
    return {"seconds": seconds, "median_seconds": statistics.median(seconds)}, result


def csi_cdf(paths, points, dt, cutoffs):
    drift, threshold, collapse = paths
    count = len(cutoffs)
    # Use the interval probabilities themselves: diagnostic cumulative exits
    # extend to a complete solver cell, even if the requested cutoff is off-grid.
    result = MovingBoundaryDDMSolver(time_step=dt, spatial_points=points, native_forward=True).solve_observation_batch(
        drift=drift.repeat_interleave(count * 2, 0), threshold=threshold.repeat_interleave(count * 2),
        collapse_rate=collapse.repeat_interleave(count * 2),
        interval_low=torch.zeros(len(threshold) * count * 2, dtype=threshold.dtype),
        interval_high=torch.tensor(cutoffs, dtype=threshold.dtype).repeat_interleave(2).repeat(len(threshold)),
        choice=torch.tensor([0., 1.], dtype=threshold.dtype).repeat(len(threshold) * count))
    return result.probability.reshape(len(threshold), count, 2).numpy()


def dawa_cdf(paths, parameters, points, dt, cutoffs):
    cfg = ContinuousConfig(points=points, time_step=dt, flux_backend="triton", recompute_rates=False)
    rows = []
    for path in paths:
        d = ContinuousResponseSolver(cfg).solve(cuda_path(path), parameters.cuda())
        rows.append([[float(d.interval_probability(c, 0., t)) for c in (0, 1)] for t in cutoffs])
    return np.asarray(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/csi_dawa_first_passage.json"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--estimates", type=int, default=100000)
    parser.add_argument("--points", type=int, nargs="+", default=[33, 65, 129])
    parser.add_argument("--horizon", type=float, default=1.4)
    parser.add_argument("--trials", type=int, nargs="+", default=[2, 16])
    parser.add_argument("--skip-refinement", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.estimates < 1 or any(n < 2 or n % 2 for n in args.trials):
        parser.error("Positive counts and even trial counts of at least two are required")
    if not 0.1 <= args.horizon <= 3.:
        parser.error("Use a horizon between 0.1 and 3 seconds, keeping CSI boundaries positive")
    if any(n < 9 or n % 2 == 0 for n in args.points):
        parser.error("Use odd spatial point counts of at least nine")
    if abs(args.horizon / .001 - round(args.horizon / .001)) > 1.e-8:
        parser.error("The horizon must align with 1 ms")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    source = json.loads((ROOT / "dawa/dawa_likelihood/study_results/performance_parallel_gpu.json").read_text())
    parameters = torch.tensor([source["parameters"][n] for n in PARAMETER_NAMES], dtype=torch.float64)
    cpuinfo = Path("/proc/cpuinfo")
    cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")), platform.processor()) if cpuinfo.exists() else platform.processor()
    report = {"scope": __doc__, "cpu": cpu, "gpu": torch.cuda.get_device_name(),
              "platform": platform.platform(), "torch": str(torch.__version__), "repeats": args.repeats,
              "horizon": args.horizon, "direct_dt": .001, "sampling_dt": .00025,
              "noise": .1, "window": .025, "estimates_per_trial": args.estimates,
              "dawa_parameters": source["parameters"],
              "csi_parameters": {"gain": [10., 20.], "threshold": [.12, .12], "collapse": [-.01, -.02],
                                 "iti": 1., "cue_duration": .05, "leak": 12., "competition": 3.},
              "timing": "One candidate; fixed conditioned paths; construction timed separately. Stochastic timings include transfers and host probability summaries. Three output categories and a final 25 ms choice-0 bin for both models. No likelihood gradients or optimization. GPU synchronized; warmed medians.",
              "cases": [], "refinement": {}, "source_sha256": {}}
    for path in (Path(__file__), ROOT / "csi/csi_fit/direct_likelihood/continuous_monte_carlo.py",
                 ROOT / "dawa/dawa_likelihood/continuous_monte_carlo.py",
                 ROOT / "dawa/dawa_likelihood/continuous_flux_gpu.py",
                 ROOT / "dawa/dawa_likelihood/continuous_flux_cpu.cpp",
                 ROOT.parents[2] / "psyneulink/core/batched/numerical/first_passage_cpu.cpp"):
        report["source_sha256"][str(path.relative_to(ROOT.parents[2]))] = hashlib.sha256(path.read_bytes()).hexdigest()

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    for trials in args.trials:
        for model in ("csi", "dawa"):
            torch.set_num_threads(1)
            factory = (lambda dt: csi_paths(dt, args.horizon, trials)) if model == "csi" else (
                lambda dt: dawa_paths(parameters, dt, args.horizon, trials))
            prep_direct, direct_paths = measure(lambda: factory(.001), args.repeats)
            prep_sample, sample_paths = measure(lambda: factory(.00025), args.repeats)
            case = {"model": model, "trials": trials, "path_construction_direct": prep_direct,
                    "path_construction_sampling": prep_sample, "direct": []}
            timing, result = measure(lambda: summarize_samples(sample(model, sample_paths, parameters, .00025,
                                                                      args.estimates, 43), args.horizon, .025), args.repeats)
            case["sampling"] = {**timing, **result}
            report["cases"].append(case)
            print(f"{model} trials={trials} sampling {timing['median_seconds']:.6f}s", flush=True)
            grids = args.points if trials == args.trials[0] else [129]
            settings = [("native", 1), ("native", 4)] if model == "csi" else [("native", 1), ("native", 4), ("triton", 1)]
            if trials != args.trials[0]:
                settings = [("native", 4)] if model == "csi" else [("triton", 1)]
            for points in grids:
                for backend, threads in settings:
                    torch.set_num_threads(threads if model == "csi" else 1)
                    function = (lambda: csi_direct(direct_paths, points, .001, args.horizon, .025)) if model == "csi" else (
                        lambda: dawa_direct(direct_paths, parameters, points, .001, args.horizon, .025, backend, threads))
                    timing, result = measure(function, args.repeats)
                    case["direct"].append({"backend": backend, "threads": threads, "points": points,
                                           "density_cells_per_trial": points - 2 if model == "csi" else points**2,
                                           **timing, **result})
                    print(f"{model} trials={trials} {backend}/{threads} grid={points} {timing['median_seconds']:.6f}s, substeps={result['substeps']}", flush=True)
                    save()
            # Forward parity across CPU thread counts and CPU/GPU backends.
            for points in grids:
                rows = [r for r in case["direct"] if r["points"] == points]
                for row in rows[1:]:
                    for key in ("probabilities", "window_choice0_probability"):
                        np.testing.assert_allclose(row[key], rows[0][key], atol=2.e-12, rtol=2.e-10)
            save()

    if not args.skip_refinement:
        cutoffs = np.arange(.1, args.horizon + 1.e-8, .1)
        for model in ("csi", "dawa"):
            torch.set_num_threads(4 if model == "csi" else 1)
            factory = (lambda dt: csi_paths(dt, args.horizon, 2)) if model == "csi" else (
                lambda dt: dawa_paths(parameters, dt, args.horizon, 2))
            solve = (lambda paths, n, dt: csi_cdf(paths, n, dt, cutoffs)) if model == "csi" else (
                lambda paths, n, dt: dawa_cdf(paths, parameters, n, dt, cutoffs))
            reference = solve(factory(.0005), 257, .0005)
            paths = factory(.001)
            rows = []
            for n in (*args.points, 257):
                value = solve(paths, n, .001)
                rows.append({"points": n, "dt": .001, "maximum_joint_cdf_error": float(np.max(np.abs(value - reference))),
                             "joint_cdf": value.tolist()})
            monte_carlo = []
            for dt in (.00025, .000125):
                paths = factory(dt)
                for seed in (43, 71, 103):
                    samples = sample(model, paths, parameters, dt, args.estimates, seed)
                    cdf = np.stack([empirical_cdf(row, cutoffs) for row in samples])
                    delta = cdf - reference
                    monte_carlo.append({"dt": dt, "seed": seed, "maximum_joint_cdf_error": float(np.max(np.abs(delta))),
                                        "joint_cdf": cdf.tolist()})
            report["refinement"][model] = {"cutoffs": cutoffs.tolist(), "reference_points": 257, "reference_dt": .0005,
                                           "reference_joint_cdf": reference.tolist(), "direct": rows, "sampling": monte_carlo}
            print(f"{model} refinement direct: {[(r['points'], round(r['maximum_joint_cdf_error'], 6)) for r in rows]}", flush=True)
            print(f"{model} sampling errors: {[round(r['maximum_joint_cdf_error'], 6) for r in monte_carlo]}", flush=True)
            save()
    save()


if __name__ == "__main__":
    main()
