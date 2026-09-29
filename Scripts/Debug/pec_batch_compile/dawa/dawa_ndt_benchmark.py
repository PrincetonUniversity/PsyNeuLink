"""Verify and time exact NDT profiling; optionally rescore saved fits on common seeds."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import psyneulink as pnl
import torch

from dawa_fitting_budget_study import Study, total_scores
from dawa_pec_fit import LAUNCH, RT_RANGE, TRUTH, load_subject, save_json
from psyneulink.core.batched.shifted_histogram import ShiftedHistogramScorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Completed recovery run providing observations",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fits", nargs="*", default=[], help="label=completed_run_directory"
    )
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=[9401, 9402, 9403, 9404, 9405],
    )
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pnl.set_num_threads(8)
    torch.set_num_threads(8)
    manifest = json.loads((args.run / "manifest.json").read_text())
    frame = load_subject(
        args.run / "synthetic_subject.csv", manifest["arguments"]["subject"]
    )
    study = Study(frame, manifest["arguments"]["max_steps"])
    support = np.arange(study.plan.ir.max_steps + 1, dtype=np.float32) * np.float32(
        0.01
    )
    grid = np.linspace(0.1, 0.3, 2001)
    scorer = ShiftedHistogramScorer(
        support,
        grid,
        study.observed,
        [0],
        bins=100,
        bin_range=[RT_RANGE],
        smoothing_sigma=0.5,
        categorical_cardinalities=[2],
    )
    base = list(TRUTH)
    base[1] = 0.0

    def counts(rows, n, seed):
        return study.plan.discrete_output_counts(
            study.inputs,
            study.parameters(rows),
            n,
            study.observed,
            [0],
            support=support,
            seed=seed,
            triton_launch_options=LAUNCH,
        )

    report = {
        "gpu": torch.cuda.get_device_name(),
        "representatives": scorer.shifts.tolist(),
        "reference_run": str(args.run),
        "trials": len(frame),
        "checks": [],
        "timing": [],
    }
    if not args.skip_timing:
        n, seed = 257, 7411
        raw = study.plan.run(
            study.inputs,
            study.parameters([base]),
            n,
            seed=seed,
            strict_truncation=True,
            triton_launch_options=LAUNCH,
            keep_device_values=True,
        ).values[0, 0]
        reduced = counts([base], n, seed)
        indices = torch.searchsorted(reduced.support, raw[..., 1].contiguous())
        torch.testing.assert_close(
            reduced.support[indices], raw[..., 1], rtol=0, atol=0
        )
        match = (
            raw[..., 0] == torch.tensor(study.observed[:, 0], device="cuda")[:, None]
        )
        address = (
            torch.arange(len(frame), device="cuda")[:, None] * len(support) + indices
        )
        expected = torch.bincount(
            address[match], minlength=len(frame) * len(support)
        ).reshape(len(frame), len(support))
        torch.testing.assert_close(
            reduced.counts[0, 0].long(), expected, rtol=0, atol=0
        )
        report["materialized_count_check"] = True
        del raw, expected, reduced
        for n in [5000, 100000]:
            reduced = counts([base], n, seed)
            alpha = n / 100000
            shifts = [0.1, 0.1234, 0.2, 0.2199, 0.3]
            check = ShiftedHistogramScorer(
                support,
                shifts,
                study.observed,
                [0],
                bins=100,
                bin_range=[RT_RANGE],
                smoothing_sigma=0.5,
                categorical_cardinalities=[2],
            )
            cached = check.densities(reduced, pseudocount=alpha)[0, 0]
            for k, shift in enumerate(check.shifts):
                row = base.copy()
                row[1] = shift
                direct = study.plan.histogram_likelihood(
                    study.inputs,
                    study.parameters([row]),
                    n,
                    study.observed,
                    [0],
                    bins=100,
                    bin_range=[RT_RANGE],
                    smoothing_sigma=0.5,
                    pseudocount=alpha,
                    categorical_cardinalities=[2],
                    seed=seed,
                    triton_launch_options=LAUNCH,
                )[0, 0]
                np.testing.assert_allclose(cached[k], direct, rtol=2e-6, atol=1e-7)
                report["checks"].append(
                    {
                        "estimates": n,
                        "ndt": float(shift),
                        "max_density_error": float(np.abs(cached[k] - direct).max()),
                        "score_error": float(
                            total_scores(cached[k].astype(float), study.mask)
                            - total_scores(direct.astype(float), study.mask)
                        ),
                    }
                )
            rows = [base] * 10
            ordinary = [list(TRUTH)] * 10
            for kind in ("ordinary", "profile"):
                measurements = []
                for repeat in range(4):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    if kind == "ordinary":
                        study.plan.histogram_likelihood(
                            study.inputs,
                            study.parameters(ordinary),
                            n,
                            study.observed,
                            [0],
                            bins=100,
                            bin_range=[RT_RANGE],
                            smoothing_sigma=0.5,
                            pseudocount=alpha,
                            categorical_cardinalities=[2],
                            seed=seed,
                            triton_launch_options=LAUNCH,
                        )
                    else:
                        reduction = counts(rows, n, seed)
                        density = scorer.densities(reduction, pseudocount=alpha)
                        np.log(density[..., study.mask].astype(float)).sum(-1).max(-1)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    if repeat:
                        measurements.append(elapsed)
                report["timing"].append(
                    {
                        "estimates": n,
                        "candidates": 10,
                        "kind": kind,
                        "seconds": measurements,
                    }
                )
            save_json(args.output / "benchmark.json", report)
    if args.fits:
        labeled = {}
        names = list(manifest["bounds"])
        columns = [
            "PrevCongruency",
            "T1",
            "T2",
            "S1",
            "S2",
            "S3",
            "S4",
            "likelihood_include_mask",
            "decision",
            "response_time",
        ]
        for arg in args.fits:
            label, path = arg.split("=", 1)
            path = Path(path)
            other = load_subject(
                path / "synthetic_subject.csv", manifest["arguments"]["subject"]
            )
            np.testing.assert_array_equal(
                frame[columns].to_numpy(), other[columns].to_numpy()
            )
            result = json.loads((path / "recovery.json").read_text())
            labeled[label] = [result["fitted"][name] for name in names]
        records = []
        for seed in args.validation_seeds:
            density = study.plan.histogram_likelihood(
                study.inputs,
                study.parameters(list(labeled.values())),
                100000,
                study.observed,
                [0],
                bins=100,
                bin_range=[RT_RANGE],
                smoothing_sigma=0.5,
                pseudocount=1.0,
                categorical_cardinalities=[2],
                seed=seed,
                triton_launch_options=LAUNCH,
            )[:, 0]
            record = {
                "seed": seed,
                **dict(
                    zip(
                        labeled,
                        map(float, total_scores(density.astype(float), study.mask)),
                        strict=True,
                    )
                ),
            }
            records.append(record)
            print(record, flush=True)
        report["validation"] = records
        report["parameters"] = labeled
        report["parameter_order"] = names
    save_json(args.output / "benchmark.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
