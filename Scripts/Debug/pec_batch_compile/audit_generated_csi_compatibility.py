"""Audit generated history/scoring on recorded CSI data; no fit or cluster job.

Random interior candidates vary all three conditions continuously, including
nondecision time, without timestep quantization. Extra high-NDT candidates
probe zero-step histories: these must be reported as unsupported, not silently
clamped or routed through the custom CSI kernel. The custom kernel is used only
as a deterministic count/state/path oracle, not for matched-noise score tests.
Prints JSON to stdout; no results are written automatically.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from psyneulink.core.batched import BatchedTrialParameter, EndpointReconstructionError, unregister_batched_instance_op
from benchmark_likelihood_specializations import make_case, emit


def run(args, dt):
    simulation, observation, inputs, data, _, include = make_case(args, dt, recorded=True)
    try:
        frame = pd.read_csv(args.data)
        frame = frame[(frame.subject_nr == args.subject) & frame.sequence.isin(["NoInstruction", "RealRare", "RealFrequent"])].reset_index(drop=True)
        if args.trials:
            frame = frame.iloc[:args.trials]
        condition = frame.sequence.map({"NoInstruction": 0, "RealRare": 1, "RealFrequent": 2}).to_numpy()
        graph = simulation.ir.graph
        lca = next(node for node in graph.nodes if "Task Activations" in node.name)
        ddm = next(node for node in graph.nodes if node.name.split("-")[0] == "DDM")
        rng = np.random.default_rng(721)
        options = dict(bins=100, smoothing_sigma=.5, pseudocount=.1, categorical_cardinalities=[2])
        scorer = observation.compile_histogram_score(categorical_dims=[0], **options)
        records = []
        for index in range(args.candidates):
            ndt = rng.uniform(.08, .30, 3) if index < args.candidates - 2 else rng.uniform(.45, .60, 3)
            row = {
                lca.params["gain"]: BatchedTrialParameter(rng.uniform(5., 25., 3)[condition]),
                ddm.params["threshold"]: BatchedTrialParameter(rng.uniform(.05, .15, 3)[condition]),
                ddm.params["threshold_collapse"]: BatchedTrialParameter(rng.uniform(-.12, -.03, 3)[condition] * dt),
                ddm.params["non_decision_time"]: BatchedTrialParameter(ndt[condition]),
            }
            print(f"dt={dt:g}: candidate {index + 1}/{args.candidates}", file=sys.stderr, flush=True)
            _, legacy = simulation.deterministic_history_log_likelihood(
                inputs, [row], 1, data, [0], **options, include_mask=include,
                seed=19, return_debug=True,
            )
            zero = int(np.count_nonzero(legacy["observed_steps"] < 1))
            try:
                device = observation.sampler.path_plan.generate_device(inputs, data, [row], max_buffer_bytes=args.buffer_mib * 1024**2)
            except EndpointReconstructionError as error:
                if not zero or error.code != "endpoint.projected_count_below_minimum":
                    raise
                records.append(dict(candidate=index, nondecision_time=ndt.tolist(), status="unsupported_zero_step_history", zero_trials=zero))
                del legacy
                continue
            assert not zero
            np.testing.assert_array_equal(device.history.event_counts, legacy["observed_steps"])
            states = legacy["history_states"].cpu().numpy()
            paths = legacy["drift_paths"].cpu().numpy()
            actual_paths = device.values[..., 0].cpu().numpy()
            np.testing.assert_allclose(device.history.end_states, states, rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(actual_paths, paths, rtol=2e-5, atol=2e-6)
            state_error = float(np.max(np.abs(device.history.end_states - states)))
            path_error = float(np.max(np.abs(actual_paths - paths)))
            del device, legacy, states, paths, actual_paths
            kwargs = dict(num_estimates=args.estimates, seed=19, include_mask=include, max_buffer_bytes=args.buffer_mib * 1024**2)
            fast = scorer.score(inputs, data, [row], execution="window", **kwargs)
            reference = scorer.score(inputs, data, [row], reference=True, **kwargs)
            np.testing.assert_array_equal(fast.bin_counts[:, include], reference.bin_counts[:, include])
            np.testing.assert_array_equal(fast.log_likelihood, reference.log_likelihood)
            records.append(dict(candidate=index, nondecision_time=ndt.tolist(), status="validated",
                                counts_exact=True, max_state_error=state_error, max_drift_error=path_error,
                                histogram_counts_exact=True, log_likelihood=fast.log_likelihood.tolist()))
        emit(dict(kind="recorded_compatibility_audit", dt=dt, subject=args.subject,
                  trials=len(data), scored_trials=int(include.sum()), estimates=args.estimates, candidates=records))
    finally:
        unregister_batched_instance_op("Drift Rate Value")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "csi_fit/data fitting/data_to_fit_study3.csv")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--dt", type=float, nargs="+", default=[.01, .001])
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--estimates", type=int, default=257)
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument("--maximum-time", type=float, default=12.)
    parser.add_argument("--buffer-mib", type=int, default=2048)
    args = parser.parse_args()
    args.verify_endpoints = False
    if args.candidates < 3 or args.estimates < 1 or args.buffer_mib < 1 or args.trials < 0:
        parser.error("Require at least three candidates, positive estimates/budget, and nonnegative trials")
    for dt in args.dt:
        if dt not in (.01, .001):
            parser.error("dt must be .01 or .001")
        run(args, dt)


if __name__ == "__main__":
    main()
