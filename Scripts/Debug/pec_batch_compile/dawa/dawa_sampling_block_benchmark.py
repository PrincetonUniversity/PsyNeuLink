"""Replay a saved fitted population to time serial and batched sampling blocks.

Use --serial-only with the earlier source snapshot for a compiler baseline.
Timing excludes warmup; comparisons alternate execution order and require exact
density equality. This is a kernel-call benchmark, not an optimizer experiment.
"""

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import psyneulink as pnl
import torch

from dawa_fitting_budget_study import Study
from dawa_pec_fit import LAUNCH, load_subject
from psyneulink.core.batched.backend.triton.discrete_counts import DiscreteCountEmitter
from psyneulink.core.batched.shifted_histogram import ShiftedHistogramScorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--serial-only", action="store_true")
    args = parser.parse_args()
    pnl.set_num_threads(8)
    torch.set_num_threads(8)
    manifest = json.loads((args.run / "manifest.json").read_text())
    study = Study(
        load_subject(
            args.run / "synthetic_subject.csv", manifest["arguments"]["subject"]
        ),
        2000,
    )
    rows = [
        json.loads(line)
        for line in (args.run / "evaluations.jsonl").read_text().splitlines()
    ]
    parameters = [
        row["parameters"].copy() for row in rows if 1002 <= row["evaluation"] < 1012
    ]
    if len(parameters) != 10:
        raise ValueError("The reference fit must contain evaluations 1002 through 1011")
    for row in parameters:
        row[1] = 0.0
    paramsets = study.parameters(parameters)
    support = np.arange(2001, dtype=np.float32) * np.float32(0.01)
    scorer = ShiftedHistogramScorer(
        support,
        np.linspace(0.1, 0.3, 2001),
        study.observed,
        [0],
        bins=100,
        bin_range=[(0.0, 3.0)],
        smoothing_sigma=0.5,
        categorical_cardinalities=[2],
    )
    options = dict(
        support=support, invalid_candidates="nan", triton_launch_options=LAUNCH
    )

    def emit(cached):
        emitter = DiscreteCountEmitter(
            study.plan.kernel_ir,
            [0, 1],
            [True, False],
            support_size=len(support),
            normal_rng=LAUNCH["normal_rng"],
            trial_schedule=LAUNCH["trial_schedule"],
            stop_truncated=True,
        )
        return emitter.cached_source(len(support)) if cached else emitter.emit()

    host = {}
    for cached in [False] if args.serial_only else [False, True]:
        emit(cached)
        durations = []
        for _ in range(30):
            start = time.perf_counter()
            emit(cached)
            durations.append(time.perf_counter() - start)
        host["cached" if cached else "uncached"] = dict(
            seconds=durations, median_seconds=statistics.median(durations)
        )

    def sample(n, seeds, batched):
        if batched:
            counts = study.plan.discrete_output_count_blocks(
                study.inputs, paramsets, n, study.observed, [0], seeds=seeds, **options
            )
            return [scorer.densities(count, pseudocount=n / 100000) for count in counts]
        results = []
        for seed in seeds:
            count = study.plan.discrete_output_counts(
                study.inputs, paramsets, n, study.observed, [0], seed=seed, **options
            )
            results.append(scorer.densities(count, pseudocount=n / 100000))
        return results

    records = []
    for n, blocks in [
        (1250, 4),
        (2500, 2),
        (5000, 2),
        (10000, 2),
        (20000, 2),
        (100000, 1),
    ]:
        seeds = [7411 + i for i in range(blocks)]
        modes = ["serial"] if args.serial_only else ["serial", "batched"]
        reference = sample(n, seeds, False)
        for mode in modes:
            for expected, actual in zip(
                reference, sample(n, seeds, mode == "batched"), strict=True
            ):
                np.testing.assert_array_equal(expected, actual)
        timings = {mode: [] for mode in modes}
        memory = {mode: [] for mode in modes}
        for repeat in range(args.repeats):
            for mode in modes if repeat % 2 == 0 else modes[::-1]:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                values = sample(n, seeds, mode == "batched")
                torch.cuda.synchronize()
                timings[mode].append(time.perf_counter() - start)
                memory[mode].append(torch.cuda.max_memory_allocated())
                for expected, actual in zip(reference, values, strict=True):
                    np.testing.assert_array_equal(expected, actual)
        record = dict(
            estimates_per_block=n,
            blocks=blocks,
            candidates=10,
            exact_density_equality=True,
            seconds=timings,
            peak_allocated_bytes=memory,
            median_seconds={
                mode: statistics.median(times) for mode, times in timings.items()
            },
        )
        if not args.serial_only:
            record["speedup"] = (
                record["median_seconds"]["serial"] / record["median_seconds"]["batched"]
            )
        records.append(record)
        print(json.dumps(record), flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                launch=LAUNCH,
                host_source=host,
                results=records,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
