"""Measure full-subject DAWA candidate evaluation through PEC's GPU objective.

This benchmarks a fixed candidate pool, not optimizer convergence. Each candidate
simulates the complete ordered subject sequence, preserving control state. The
objective is the existing GPU histogram score, not native PEC's CPU fastKDE.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import psyneulink as pnl
import torch

from psyneulink.core.globals.utilities import set_global_seed
from psyneulink.core.batched.likelihood import _sum_histogram_log_likelihood
from dawa_batched_simulation import SOURCE, build_model, fit_surface, node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=SOURCE.parent / "flanker_data_part1.csv")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--estimates", type=int, default=100000)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--profile", type=Path, help="Profile one warmed candidate and write a Chrome trace")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--maxnreg", type=int)
    parser.add_argument("--smoothing-sigma", type=float, default=0., help="Gaussian width in histogram-bin units")
    parser.add_argument("--pseudocount", type=float, default=1., help="Symmetric prior count per joint choice/RT bin")
    parser.add_argument("--materialized", action="store_true", help="Use the original materialized scoring path")
    parser.add_argument("--verify-materialized", action="store_true", help="Compare all trial probabilities to the original scorer after timing")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.estimates, args.repeats, args.max_steps, *args.batch_sizes) < 1:
        parser.error("Counts must be positive")
    if any(not np.isfinite(value) or value < 0 for value in (args.smoothing_sigma, args.pseudocount)):
        parser.error("Smoothing sigma and pseudocount must be finite and nonnegative")

    start = time.perf_counter()
    set_global_seed(args.seed)
    pnl.set_num_threads(8)
    torch.set_num_threads(8)
    data = pd.read_csv(args.data)
    data = data[(data.subject_nr == args.subject) & data.PrevCongruency.notna()].copy()
    if data.empty:
        parser.error("No retained trials for this subject")
    noise = dict(c_noise=.1, s_noise=.1, d_noise=.1, r_noise=.1)
    model, inputs, outputs = build_model(trials=len(data), **noise)
    inputs[node(model, "Task Input")] = data[["T1", "T2"]].to_numpy()
    inputs[node(model, "Stimulus Input")] = data[["S1", "S2", "S3", "S4"]].to_numpy()
    observed = data[["decision", "response_time", "subject_nr", "PrevCongruency"]].copy()
    for name in ("decision", "subject_nr", "PrevCongruency"):
        observed[name] = pd.Categorical(observed[name])
    surface = fit_surface(model)
    depends = {key: "subject_nr" for key in surface if key[0] in ("termination_threshold", "gain", "slope")}
    depends[("intercept", node(model, "RT_GATE"))] = "subject_nr"
    depends[("mode", node(model, "LC"))] = "PrevCongruency"
    pec = pnl.ParameterEstimationComposition(
        model=model, parameters={key: np.asarray(bounds[:2]) for key, bounds in surface.items()},
        depends_on=depends, outcome_variables=list(outputs), data=observed,
        likelihood_include_mask=data.likelihood_include_mask.to_numpy(dtype=bool),
        optimization_function=pnl.PECOptimizationFunction(
            method="differential_evolution", max_iterations=1, batched_backend="triton",
            batched_max_steps=args.max_steps, batched_seed=args.seed,
            batched_strict_truncation=True, batched_bins=100,
            batched_fused_likelihood=not args.materialized,
            batched_bin_range=[(0., 3.)], batched_pseudocount=args.pseudocount,
            batched_smoothing_sigma=args.smoothing_sigma,
            batched_triton_launch_options={"block_size": args.block_size, "num_warps": args.num_warps,
                                           "maxnreg": args.maxnreg},
        ),
        num_estimates=args.estimates, initial_seed=args.seed,
        same_seed_for_all_parameter_combinations=True,
    )
    pec.controller._pec_input_values_by_node = inputs
    function = pec.controller.function
    objective = function._make_objective_func()._batched_parameter_sets
    # Four explicit proposals, in fit_surface order; LC mode has one coordinate
    # per previous-congruency condition, matching the source fitting scripts.
    proposals = [
        [.30, .20, -.45, 10., .90, .90, 1., 5.],
        [.40, .22, -.40, 12., .70, .80, 1.5, 5.5],
        [.50, .18, -.35, 8., .50, .70, 2., 6.],
        [.60, .25, -.30, 15., .30, .60, 2.5, 7.],
    ]
    expected_order = []
    for key in pec.fit_parameters:
        levels = pec.cond_levels[key] if key in pec.cond_levels else [None]
        expected_order.extend((key[0], key[1].name, str(level)) for level in levels)
    if len(expected_order) != 8:
        raise AssertionError(f"Expected eight subject-specific fitting coordinates: {expected_order}")
    candidates = np.asarray(proposals)
    plan = function._compile_batched_plan()
    report = {
        "subject": args.subject, "trials": len(data),
        "scored_trials": int(data.likelihood_include_mask.sum()),
        "estimates_per_trial_per_candidate": args.estimates,
        "candidate_coordinates": expected_order, "candidates": proposals,
        "noise": noise, "lca_dt": .01, "seed": args.seed,
        "max_steps": args.max_steps, "strict_truncation": True,
        "fused_likelihood": not args.materialized,
        "launch_options": {"block_size": args.block_size, "num_warps": args.num_warps, "maxnreg": args.maxnreg},
        "gpu": torch.cuda.get_device_name(), "platform": platform.platform(),
        "torch_version": torch.__version__, "plan_outputs": list(plan.ir.output_names),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "estimator": {"kind": "trial_marginal_histogram", "bins": 100, "rt_range": [0., 3.],
                      "pseudocount": args.pseudocount, "smoothing_sigma": args.smoothing_sigma},
        "setup_seconds": time.perf_counter() - start, "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.profile:
        from torch.profiler import ProfilerActivity, profile, record_function

        objective(candidates[:1])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     profile_memory=True, record_shapes=True) as prof:
            with record_function("DAWA_PEC_objective"):
                objective(candidates[:1])
            torch.cuda.synchronize()
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(args.profile))
        args.profile.with_suffix(".txt").write_text(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
            + "\n" + prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=40)
        )
        report["profile_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        kernels = [getattr(module, "pnl_batched_coevolving_graph_kernel")
                   for name, module in tuple(sys.modules.items())
                   if name.startswith("pnl_batched_") and hasattr(module, "pnl_batched_coevolving_graph_kernel")]
        compiled = [compiled for kernel in kernels for cache in kernel.device_caches.values()
                    for compiled in cache[0].values()]
        report["kernels"] = [{"registers": k.n_regs, "spills": k.n_spills, "metadata": str(k.metadata)} for k in compiled]
        report["profile_trace"] = str(args.profile)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    reference_scores = None
    reference_likelihoods = None
    for batch_size in args.batch_sizes:
        case = {"candidate_batch_size": batch_size, "candidate_count": len(candidates), "runs": []}
        report["cases"].append(case)
        case_scores = None
        for repeat in range(args.repeats + 1):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            print(f"batch_size={batch_size}, repeat={repeat}: starting", flush=True)
            captured_likelihoods = []

            def capture_likelihood(values, include_mask=None):
                captured_likelihoods.append(values.copy())
                return _sum_histogram_log_likelihood(values, include_mask)

            chunks = []
            chunk_seconds = []
            try:
                with patch("psyneulink.core.batched.likelihood._sum_histogram_log_likelihood", capture_likelihood):
                    for i in range(0, len(candidates), batch_size):
                        chunk_start = time.perf_counter()
                        chunks.append(objective(candidates[i:i + batch_size]))
                        chunk_seconds.append(time.perf_counter() - chunk_start)
                scores = np.concatenate(chunks)
                torch.cuda.synchronize()
            except torch.OutOfMemoryError as error:
                case["error"] = str(error)
                save()
                raise
            seconds = time.perf_counter() - start
            if not np.all(np.isfinite(scores)):
                raise AssertionError(f"Nonfinite scores: {scores}")
            likelihoods = np.concatenate(captured_likelihoods)
            if reference_scores is None:
                reference_scores = scores.copy()
                reference_likelihoods = likelihoods.copy()
            else:
                np.testing.assert_array_equal(likelihoods, reference_likelihoods)
                # NumPy's masked FP32 reduction has different summation order
                # for one versus multiple lanes; check trial densities exactly.
                np.testing.assert_allclose(scores, reference_scores, rtol=2e-6, atol=2e-5)
            if case_scores is None:
                case_scores = scores.copy()
            else:
                np.testing.assert_array_equal(scores, case_scores)
            run = {"seconds": seconds, "scores": scores.tolist(),
                   "trial_density_sha256": hashlib.sha256(likelihoods.tobytes(order="C")).hexdigest(),
                   "chunk_seconds": chunk_seconds,
                   "max_score_roundoff_vs_first_batch": float(np.max(np.abs(scores - reference_scores))),
                   "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
            case["runs"].append(run)
            if repeat:
                case["median_seconds"] = statistics.median(row["seconds"] for row in case["runs"][1:])
                case["seconds_per_candidate"] = case["median_seconds"] / len(candidates)
            save()
            print(json.dumps({"batch_size": batch_size, "repeat": repeat, **run}), flush=True)
    if args.verify_materialized:
        captured_likelihoods = []
        original_fused = function.batched_fused_likelihood
        try:
            function.batched_fused_likelihood = False
            with patch("psyneulink.core.batched.likelihood._sum_histogram_log_likelihood", capture_likelihood):
                for row in candidates:
                    objective([row])
        finally:
            function.batched_fused_likelihood = original_fused
        materialized = np.concatenate(captured_likelihoods)
        if args.smoothing_sigma:
            # Integer neighbor counts are reproducible; weighting counts versus
            # individual samples changes FP32 summation order in the oracle.
            np.testing.assert_allclose(reference_likelihoods, materialized, rtol=2e-6, atol=1e-8)
        else:
            np.testing.assert_array_equal(materialized, reference_likelihoods)
        report["materialized_trial_probabilities_match_exactly"] = bool(np.array_equal(materialized, reference_likelihoods))
        report["materialized_validation"] = {
            "passed": True,
            "rtol": 2e-6 if args.smoothing_sigma else 0.,
            "atol": 1e-8 if args.smoothing_sigma else 0.,
            "max_absolute_density_error": float(np.max(np.abs(reference_likelihoods - materialized))),
            "max_relative_density_error": float(np.max(np.abs(reference_likelihoods - materialized) / materialized)),
        }
        save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
