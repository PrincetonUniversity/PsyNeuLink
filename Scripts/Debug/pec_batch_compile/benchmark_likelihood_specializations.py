"""Compare CSI fused fitting and generated sampling at full fitting budgets.

No optimizer or cluster job is launched. The shared workload uses the real
subject's complete task/stimulus sequence and either source-generated observations
or actual recorded data (--recorded, with explicit ceiling history timing).
Both routes use the same histogram settings; this does not extend the public
compiler's empirical-mass semantics.
Candidate microbatching preserves all trials and is valid with CRN=True. Never
split the trial sequence or restart its LCA history to make it fit in memory.
Reports are JSON lines on stdout; no output files are written automatically.
"""

import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from psyneulink.core.batched import (
    BatchedCompositionCompiler, BatchedTrialParameter, LikelihoodEffectContract,
    ObservationField, ObservationSpec, batched_node_op, unregister_batched_instance_op,
)
from psyneulink.core.batched.kernel_ir import diag_slots
from psyneulink.core.batched.likelihood import histogram_log_likelihood
from csi_model_surrogate import make_stab_flex
from csi_triton_vs_llvm import _drift_rate, _node


def emit(record):
    print(json.dumps(record), flush=True)


def make_case(args, dt, *, recorded=False):
    frame = pd.read_csv(args.data)
    frame = frame[(frame.subject_nr == args.subject) & frame.sequence.isin(["NoInstruction", "RealRare", "RealFrequent"])].reset_index(drop=True)
    if args.trials:
        frame = frame.iloc[:args.trials].copy()
    if frame.empty:
        raise ValueError("No subject trials selected")
    comp = make_stab_flex(leak=12.0, competition=3.0, iti=round(1 / dt),
                          csi_switch=round(0.08 / dt), threshold=0.1,
                          threshold_collapse=-0.02 * dt, non_decision_time=0.2,
                          ddm_noise=0.1, lca_noise=0.0,
                          lca_time_step_size=dt, ddm_time_step_size=dt)
    batched_node_op("Drift Rate Value", likelihood_contract=LikelihoodEffectContract())(_drift_rate)
    task = frame[["T1", "T2"]].to_numpy()
    inputs = {
        _node(comp, "Task Input"): task,
        _node(comp, "Stimulus Input"): frame[["S1", "S2", "S3", "S4"]].to_numpy(),
        _node(comp, "Correct Response"): frame[["correct_response"]].to_numpy(),
        _node(comp, "Cue Stimulus Interval"): np.any(task != np.roll(task, 1, axis=0), axis=1)[:, None].astype(float),
    }
    observations = ObservationSpec((
        ObservationField(_node(comp, "DECISION_GATE").output_port, "counting"),
        ObservationField(_node(comp, "RESPONSE_GATE").output_port, "lebesgue" if recorded else "counting", role="event_time",
                         history_timing="ceil_fp32_8ulp" if recorded else "exact"),
    ))
    history = BatchedCompositionCompiler.compile_history_replay(comp, observations, backend="triton", max_steps=round(args.maximum_time / dt))
    observed = history.compile_boundary_trajectories().compile_stochastic_sampler().compile_observation_sampler()
    condition = frame.sequence.map({"NoInstruction": 0, "RealRare": 1, "RealFrequent": 2}).to_numpy()
    lca, ddm = _node(comp, "Task Activations [C1, C2]"), _node(comp, "DDM")
    rows = []
    # Neighborhood of the interior baseline used by the comprehensive recovery
    # study. Condition-dependent values exercise actual candidate/trial lanes.
    for i in range(args.candidates):
        delta = ((i + 1) // 2) * (1 if i % 2 else -1)
        rows.append({
            f"{lca.name}.gain": BatchedTrialParameter((np.array([15., 18., 12.]) + 0.2 * delta)[condition]),
            f"{ddm.name}.threshold": BatchedTrialParameter((np.array([.10, .09, .11]) + .001 * delta)[condition]),
            f"{ddm.name}.threshold_collapse": BatchedTrialParameter((np.array([-.020, -.015, -.025]) - .0005 * delta)[condition] * dt),
            f"{ddm.name}.non_decision_time": BatchedTrialParameter((np.array([.20, .23, .18]) - (abs(delta) % 3) * (.0037 if recorded else dt))[condition]),
        })
    start = time.perf_counter()
    data = (frame[["decision", "response_time"]].to_numpy(dtype=float) if recorded
            else history.simulate_reference(inputs, rows[0], seed=12).observations[0])
    emit(dict(kind="case", dt=dt, subject=args.subject, trials=len(frame), scored_trials=int(frame.likelihood_include_mask.sum()),
              candidates=len(rows), estimates=args.estimates, maximum_time=args.maximum_time,
              synthetic_mean_rt=None if recorded else float(data[:, 1].mean()), actual_mean_rt=float(frame.response_time.mean()),
              data_generation_seconds=time.perf_counter() - start,
              observation_policy="recorded_values_ceiling_history" if recorded else "synthetic_source_lattice_on_real_sequence", device=torch.cuda.get_device_name()))
    if args.verify_endpoints:
        endpoints = history.simulation_plan.compile_observed_endpoints(observations)
        checked, timings = {}, {}
        for method in ("auto", "exhaustive"):
            print(f"dt={dt:g} verifying {method} endpoint reconstruction", file=sys.stderr, flush=True)
            started = time.perf_counter()
            checked[method] = endpoints.reconstruct(inputs, data, rows, method=method)
            timings[method] = time.perf_counter() - started
        np.testing.assert_array_equal(checked["auto"], checked["exhaustive"])
        emit(dict(kind="endpoint_check", dt=dt, count_entries=int(checked["auto"].size),
                  exact_match=True, seconds=timings))
    return history.simulation_plan, observed, inputs, data, rows, frame.likelihood_include_mask.to_numpy(dtype=bool)


def measure(label, operation, args, dt):
    gc.collect()
    torch.cuda.empty_cache()
    print(f"dt={dt:g} {label}: first/warm-up call", file=sys.stderr, flush=True)
    start = time.perf_counter()
    initial, _ = operation()
    torch.cuda.synchronize()
    first = time.perf_counter() - start
    print(f"dt={dt:g} {label} first/warm-up completed: {first:.3f}s", file=sys.stderr, flush=True)
    runs, peaks, parts = [], [], []
    for repeat in range(args.repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result, timing = operation()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        np.testing.assert_array_equal(result, initial)
        runs.append(elapsed)
        peaks.append(torch.cuda.max_memory_allocated() / 1024**3)
        parts.append(timing)
        print(f"dt={dt:g} {label} warm {repeat + 1}: {elapsed:.3f}s", file=sys.stderr, flush=True)
    emit(dict(kind="timing", dt=dt, method=label, first_call_seconds=first, warm_seconds=runs,
              median_seconds=float(np.median(runs)), peak_torch_allocated_gib=max(peaks),
              phases=parts, scores=np.asarray(initial).tolist()))
    return np.asarray(initial)


def run(args, dt):
    simulation, generated, inputs, data, rows, include = make_case(args, dt, recorded=args.recorded)
    histogram = dict(bins=100, smoothing_sigma=0.5, pseudocount=0.1,
                     categorical_cardinalities=[2], include_mask=include)
    fused = generated.compile_histogram_score(categorical_dims=[0], **{key: value for key, value in histogram.items() if key != "include_mask"})
    lanes = len(rows) * len(data) * args.estimates
    phase_times = {}
    slots = len(diag_slots(simulation.kernel_ir))
    emit(dict(kind="memory_preflight", dt=dt, full_population_lanes=lanes,
              generated_sample_device_buffers_gib=lanes * 4 * (2 + 3 + slots) / 1024**3,
              generated_sample_host_device_budget_lower_gib=lanes * (8 * (2 + 3 + slots) + 5) / 1024**3,
              diagnostic_slots=slots, candidate_microbatch=args.microbatch,
              fused_launch=dict(block_size=args.fused_block_size, num_warps=args.fused_num_warps),
              fused_candidate_batch_cap=args.fused_candidate_batch_size,
              fused_estimate_batch_cap=args.fused_estimate_batch_size,
              note="Excludes paths, histogram temporaries, framework memory; generated full population is not allocated."))

    def specialized(strict):
        score = simulation.deterministic_history_log_likelihood(
            inputs, rows, args.estimates, data, [0], **histogram, seed=args.seed,
            common_random_numbers=True, strict_truncation=strict, implementation="handwritten",
            triton_launch_options={"block_size": 32, "num_warps": 1},
        )
        return np.atleast_1d(score), {}

    def generated_default():
        score = simulation.deterministic_history_log_likelihood(
            inputs, rows, args.estimates, data, [0], **histogram, seed=args.seed,
            triton_launch_options={"block_size": args.fused_block_size, "num_warps": args.fused_num_warps},
        )
        return np.atleast_1d(score), {}

    def generated_histogram():
        phase_times.clear()
        scores, sample_time, reduction_time = [], 0.0, 0.0
        for offset in range(0, len(rows), args.microbatch):
            batch = rows[offset:offset + args.microbatch]
            torch.cuda.synchronize()
            start = time.perf_counter()
            samples = generated._run(inputs, data, batch, args.estimates, args.seed, True, None, True,
                                     args.buffer_mib * 1024**2, reference=False, return_device=True)
            torch.cuda.synchronize()
            sample_time += time.perf_counter() - start
            start = time.perf_counter()
            score = histogram_log_likelihood(samples.values, data, [0], **histogram)
            torch.cuda.synchronize()
            reduction_time += time.perf_counter() - start
            scores.extend(np.atleast_1d(score).tolist())
            del samples
        return np.asarray(scores), dict(history_paths_sampling_seconds=sample_time, histogram_seconds=reduction_time,
                                        **phase_times)

    def generated_fused(execution="strict"):
        phase_times.clear()
        result = fused.score(inputs, data, rows, num_estimates=args.estimates, seed=args.seed,
                             include_mask=include, max_buffer_bytes=args.buffer_mib * 1024**2,
                             candidate_batch_size=args.fused_candidate_batch_size,
                             estimate_batch_size=args.fused_estimate_batch_size,
                             execution=execution,
                             triton_launch_options={"block_size": args.fused_block_size, "num_warps": args.fused_num_warps})
        return result.log_likelihood, dict(phase_times, sampled_trials=int(result.sampled_trials.sum()),
                                          window_stopped_lanes=int(result.window_stopped.sum()))

    results = {}
    methods = {"specialized_fit": lambda: specialized(False),
               "specialized_strict": lambda: specialized(True), "generated_histogram": generated_histogram,
               "generated_fused": generated_fused,
               "generated_score_only": lambda: generated_fused("score_only"),
               "generated_window": lambda: generated_fused("window"), "generated_default": generated_default}
    with ExitStack() as stack:
        if args.profile_generated:
            from psyneulink.core.batched.endpoints import ObservedEndpointPlan
            from psyneulink.core.batched.trajectories import BoundaryTrajectoryPlan
            from psyneulink.core.batched.backend.triton import history as history_backend, trajectories as path_backend

            def timed(function, phase):
                def wrapper(*positional, **keywords):
                    key = phase or ("boundary_execution_seconds" if keywords.get("parallel_trial_lanes") else "history_execution_seconds")
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    result = function(*positional, **keywords)
                    torch.cuda.synchronize()
                    phase_times[key] = phase_times.get(key, 0.) + time.perf_counter() - started
                    return result
                return wrapper

            stack.enter_context(patch.object(ObservedEndpointPlan, "reconstruct", timed(ObservedEndpointPlan.reconstruct, "cpu_endpoint_inversion_seconds")))
            original_trace = history_backend._run_history_trace
            stack.enter_context(patch.object(history_backend, "_run_history_trace", timed(original_trace, None)))
            stack.enter_context(patch.object(path_backend, "_run_history_trace", timed(original_trace, None)))
            stack.enter_context(patch.object(BoundaryTrajectoryPlan, "generate_device",
                                             timed(BoundaryTrajectoryPlan.generate_device, "complete_device_path_preparation_seconds")))
            stack.enter_context(patch.object(path_backend, "_validate_device_paths",
                                             timed(path_backend._validate_device_paths, "device_path_validation_seconds")))
        for name in args.methods:
            results[name] = measure(name, methods[name], args, dt)
    if "specialized_fit" in results and "specialized_strict" in results:
        emit(dict(kind="censor_check", dt=dt,
                  max_abs_log_score_difference=float(np.max(np.abs(results["specialized_fit"] - results["specialized_strict"])))))
    if "generated_histogram" in results and "generated_fused" in results:
        # Integer grouping changes FP32 summation order relative to summing
        # individual weighted samples; statistics are checked exactly in tests.
        np.testing.assert_allclose(results["generated_fused"], results["generated_histogram"], rtol=0, atol=2e-4)
        emit(dict(kind="fused_score_check", dt=dt,
                  max_abs_log_score_difference=float(np.max(np.abs(results["generated_fused"] - results["generated_histogram"])))))
    if args.verify_fused_counts:
        print(f"dt={dt:g}: full-budget integer-count verification", file=sys.stderr, flush=True)
        options = dict(num_estimates=args.estimates, seed=args.seed, max_buffer_bytes=args.buffer_mib * 1024**2)
        reduced = fused.score(inputs, data, rows, **options,
                              triton_launch_options={"block_size": args.fused_block_size, "num_warps": args.fused_num_warps})
        window = fused.score(inputs, data, rows, **options, include_mask=include, execution="window",
                             triton_launch_options={"block_size": args.fused_block_size, "num_warps": args.fused_num_warps})
        np.testing.assert_array_equal(window.bin_counts[:, include], reduced.bin_counts[:, include])
        np.testing.assert_array_equal(window.log_likelihood, reduced.log_factors[:, include].sum(-1))
        for index, row in enumerate(rows):
            # CRN=True preserves the stream identity when materializing one
            # candidate at a time; every reference retains the full sequence.
            reference = fused.score(inputs, data, [row], reference=True, **options)
            np.testing.assert_array_equal(reduced.bin_counts[index], reference.bin_counts[0])
        emit(dict(kind="fused_count_check", dt=dt, exact_match=True,
                  simulation_lanes=len(rows) * len(data) * args.estimates,
                  count_entries=int(reduced.bin_counts.size),
                  window_count_entries=int(window.bin_counts[:, include].size),
                  window_exact_match=True, window_stopped_lanes=int(window.window_stopped.sum())))
    if "generated_fused" in results:
        for method in ("generated_score_only", "generated_window"):
            if method in results:
                np.testing.assert_array_equal(results[method], results["generated_fused"])
                emit(dict(kind="execution_score_check", dt=dt, method=method, exact_match=True))
    if "generated_default" in results and "generated_window" in results:
        np.testing.assert_array_equal(results["generated_default"], results["generated_window"])
        emit(dict(kind="default_route_score_check", dt=dt, exact_match=True))
    unregister_batched_instance_op("Drift Rate Value")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "csi_fit/data fitting/data_to_fit_study3.csv")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--recorded", action="store_true", help="Use actual recorded choice/RT data with ceiling history timing")
    parser.add_argument("--trials", type=int, default=0, help="0 means the full fitting sequence; nonzero is only a smoke test")
    parser.add_argument("--dt", type=float, nargs="+", default=[.01, .001])
    parser.add_argument("--candidates", type=int, default=11)
    parser.add_argument("--estimates", type=int, default=100000)
    parser.add_argument("--maximum-time", type=float, default=12.)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--buffer-mib", type=int, default=6144)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--profile-generated", action="store_true", help="Additional synchronized diagnostic phase timers; not enabled for headline comparisons")
    parser.add_argument("--verify-endpoints", action="store_true", help="Compare every candidate/trial endpoint with exhaustive reconstruction outside timed objectives")
    parser.add_argument("--verify-fused-counts", action="store_true", help="Compare all fused bin counts with candidate-microbatched materialized samples outside timing loops")
    parser.add_argument("--fused-block-size", type=int, default=128)
    parser.add_argument("--fused-num-warps", type=int, default=4)
    parser.add_argument("--fused-candidate-batch-size", type=int)
    parser.add_argument("--fused-estimate-batch-size", type=int)
    parser.add_argument("--methods", nargs="+", choices=["specialized_fit", "specialized_strict", "generated_histogram", "generated_fused",
                                                        "generated_score_only", "generated_window", "generated_default"],
                        default=["specialized_fit", "specialized_strict", "generated_histogram", "generated_fused"])
    args = parser.parse_args()
    if args.recorded and args.verify_endpoints:
        parser.error("Exhaustive exact endpoint verification does not apply to recorded ceiling histories")
    if any(value < 1 for value in (args.candidates, args.estimates, args.microbatch, args.repeats, args.buffer_mib)):
        parser.error("Sizes must be positive")
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required")
    for dt in args.dt:
        run(args, dt)


if __name__ == "__main__":
    main()
