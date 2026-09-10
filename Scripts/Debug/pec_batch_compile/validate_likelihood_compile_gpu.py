"""Local GPU validation of generic conditional sampling, readouts, and mass.

Run in a fresh process with TRITON_INTERPRET unset. Nothing is submitted to a
cluster or written to a result file. Optional shell redirection can preserve the
JSON report outside version control. Timings include preparation, checked
history/path generation, transfers, sampling, and validation: not kernel-only
or production PEC objective timings. The coupled reference uses identical
canonical observed-history starts, not unconditional random trial histories.
"""

import argparse
import json
import sys
import time

import numpy as np

from psyneulink.core.batched import (
    BatchedCompositionCompiler, LikelihoodEffectContract, ObservationField, ObservationSpec,
    batched_node_op, unregister_batched_instance_op,
)
from csi_model_surrogate import make_stab_flex
from csi_triton_vs_llvm import _csi_inputs, _drift_rate, _node


def validate_case(args, dt):
    import torch

    composition = make_stab_flex(
        ddm_noise=0.15, lca_noise=0.0, ddm_time_step_size=dt, lca_time_step_size=dt,
        threshold_collapse=-0.1 * dt, iti=round(0.02 / dt),
        csi_repeat=round(0.02 / dt), csi_switch=round(0.03 / dt),
    )
    # Existing instance-op aliases are registered under the unsuffixed source
    # name so successive model constructions in this process share that alias.
    registration_name = "Drift Rate Value"
    batched_node_op(registration_name, likelihood_contract=LikelihoodEffectContract())(_drift_rate)
    try:
        inputs = _csi_inputs(composition, args.trials)
        inputs[_node(composition, "Cue Stimulus Interval")] = (np.arange(args.trials) % 2)[:, None]
        observations = ObservationSpec((
            ObservationField(_node(composition, "DECISION_GATE").output_port, "counting"),
            ObservationField(_node(composition, "RESPONSE_GATE").output_port, "counting", role="event_time"),
        ))
        started = time.perf_counter()
        mass = BatchedCompositionCompiler.compile_empirical_mass(
            composition, observations, backend="triton", max_steps=round(1.28 / dt),
        )
        lowering_seconds = time.perf_counter() - started
        plan = mass.observation_plan
        histogram = plan.compile_histogram_score(categorical_dims=[0], bins=100, smoothing_sigma=.5,
                                                 pseudocount=.1, categorical_cardinalities=[2])
        history = plan.sampler.path_plan.history_plan
        data = history.simulate_reference(inputs, seed=12).observations[0]
        ndt = f"{_node(composition, 'DDM').name}.non_decision_time"
        gain = f"{_node(composition, 'Task Activations [C1, C2]').name}.gain"
        rows = [{ndt: 0.3 - i * dt, gain: 10.0 - 0.2 * i} for i in range(args.candidates)]
        options = dict(num_estimates=args.estimates, max_buffer_bytes=args.buffer_mib * 1024**2)
        measurements = []
        for common_random in (True, False):
            for seed in args.seeds:
                kwargs = dict(options, seed=seed, common_random_numbers=common_random)
                sampled = plan.sample(inputs, data, rows, **kwargs)
                coupled = plan.simulate_reference(inputs, data, rows, **kwargs)
                score = mass.score(inputs, data, rows, **kwargs)
                ref_score = mass.score(inputs, data, rows, reference=True, **kwargs)
                hist_score = histogram.score(inputs, data, rows, **kwargs)
                hist_reference = histogram.score(inputs, data, rows, reference=True, **kwargs)
                include = np.arange(len(data)) % 3 != 0
                hist_window = histogram.score(inputs, data, rows, execution="window", include_mask=include,
                                              candidate_batch_size=1, estimate_batch_size=1025, **kwargs)
                np.testing.assert_array_equal(hist_window.bin_counts[:, include], hist_reference.bin_counts[:, include])
                np.testing.assert_array_equal(hist_window.log_likelihood, hist_reference.log_factors[:, include].sum(-1))
                count_mismatches = int(np.count_nonzero(sampled.event_counts != coupled.event_counts))
                choice_mismatches = int(np.count_nonzero(sampled.values[..., 0] != coupled.values[..., 0]))
                max_rt_error = float(np.max(np.abs(sampled.values[..., 1] - coupled.values[..., 1])))
                hit_mismatches = int(np.count_nonzero(score.successes != ref_score.successes))
                hist_mismatches = int(np.count_nonzero(hist_score.bin_counts != hist_reference.bin_counts))
                hist_error = float(np.max(np.abs(hist_score.log_likelihood - hist_reference.log_likelihood)))
                if count_mismatches or choice_mismatches or hit_mismatches or hist_mismatches or hist_error or max_rt_error > 1e-6:
                    raise AssertionError((dt, seed, common_random, count_mismatches, choice_mismatches, hit_mismatches, hist_mismatches, hist_error, max_rt_error))
                measurements.append(dict(seed=seed, common_random_numbers=common_random,
                                         count_mismatches=count_mismatches, choice_mismatches=choice_mismatches,
                                         score_hit_mismatches=hit_mismatches, max_rt_error_seconds=max_rt_error,
                                         histogram_count_mismatches=hist_mismatches, histogram_log_score_max_error=hist_error,
                                         window_count_mismatches=0, window_stopped_lanes=int(hist_window.window_stopped.sum()),
                                         zero_hit_factors=int(score.zero_hits.sum())))
        timings = {}
        for label, operation in (
            ("generated_observation_sample", lambda: plan.sample(inputs, data, rows, **options)),
            ("full_coupled_conditional_sample", lambda: plan.simulate_reference(inputs, data, rows, **options)),
            ("empirical_mass_score", lambda: mass.score(inputs, data, rows, **options)),
        ):
            operation()  # Warm this exact seed/signature before measurement.
            elapsed = []
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                started = time.perf_counter()
                operation()
                torch.cuda.synchronize()
                elapsed.append(time.perf_counter() - started)
            timings[label] = dict(median_seconds=float(np.median(elapsed)), runs_seconds=elapsed)
        return dict(dt=dt, max_steps=round(1.28 / dt), lowering_seconds=lowering_seconds,
                    comparisons=measurements, warm_end_to_end=timings)
    finally:
        unregister_batched_instance_op(registration_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, nargs="+", default=[0.01, 0.001])
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--estimates", type=int, default=4097)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--buffer-mib", type=int, default=512)
    args = parser.parse_args()
    if any(value < 1 for value in (args.trials, args.candidates, args.estimates, args.repeats, args.buffer_mib)) or any(dt not in (0.01, 0.001) for dt in args.dt):
        parser.error("Positive sizes and dt in {0.01, 0.001} are required.")
    import torch

    if not torch.cuda.is_available():
        parser.error("A CUDA device is required.")
    report = dict(device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  configuration=vars(args), cases=[])
    for dt in args.dt:
        report["cases"].append(validate_case(args, dt))
        print(f"Validated dt={dt:g} on {report['device']}", file=sys.stderr, flush=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
