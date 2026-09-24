#!/usr/bin/env python3
"""Compare CSI histogram resolutions without changing the PNL simulator.

Uses archived local-sweep inputs and parameters. Fixed bins are centered at
integer multiples of their width. The fixed domain has a six-second span;
pseudocount scaling preserves pseudocount density per second, not per bin.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from csi_likelihood_parameter_sweep import GPUProblem, HERE, LAUNCH
from psyneulink.core.batched.histogram_score import prepare_histogram


SETTINGS = (
    ("legacy_100_sigma0.5", None, .5, False),
    ("legacy_100_unsmoothed", None, 0., False),
    ("1ms_unsmoothed_alpha0.1", .001, 0., False),
    ("1ms_unsmoothed", .001, 0., True),
    ("1ms_sigma2ms", .001, 2., True),
    ("5ms_unsmoothed", .005, 0., True),
    ("5ms_sigma2.5ms", .005, .5, True),
    ("10ms_sigma5ms", .01, .5, True),
)
CASES = {"subject-1": [0, 9], "subject-4": [0, 21], "synthetic_moderate": [0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=HERE / "data fitting/audit/likelihood-sweep-local-20260916/deep")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--estimates", type=int, default=100000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.estimates < 1 or args.repeats < 2:
        parser.error("estimates must be positive and repeats must be at least two")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    manifest = {"source": str(args.source), "estimates": args.estimates,
                "seeds": list(range(101, 101+args.repeats)), "settings": SETTINGS,
                "cases": CASES, "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                "simulation_dt": .001, "histogram_domain_span": 6.}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    records, geometry, outlier_records = [], [], []
    for label, candidates in CASES.items():
        source = args.source / label
        saved = np.load(source / "trial_scores.npz")
        input_rows = pd.read_csv(source / "data.csv")
        subject = int(input_rows.subject_nr.iloc[0])
        include = saved["include"]
        direct = saved["direct"][candidates]
        vectors = saved["vectors"][candidates]
        assert np.isfinite(direct[:, include]).all()
        folder = args.output / label
        folder.mkdir()
        shutil.copy2(source / "data.csv", folder / "data.csv")
        pd.read_csv(source / "parameters.csv").iloc[candidates].to_csv(folder / "parameters.csv", index=False)
        gpu = GPUProblem(source / "data.csv", subject, vectors)
        _, original_edges, _, _, _, _ = prepare_histogram(gpu.compiled, gpu.observed, "cpu")
        original_width = float(original_edges[1]-original_edges[0])
        print(f"{label}: legacy bin width {original_width*1000:.4f} ms", flush=True)
        for name, width, sigma, scaled in SETTINGS:
            if width is None:
                histogram = replace(gpu.compiled, smoothing_sigma=sigma)
            else:
                low = -.5*width
                # _bin_edges expands the upper edge by 1e-6 of the range.
                # Undo that expansion so the requested lattice stays fixed.
                high_argument = low + 6. / (1.+1e-6)
                histogram = replace(gpu.compiled, bins=int(round(6./width)),
                                    bin_range=((low, high_argument),), smoothing_sigma=sigma,
                                    pseudocount=.1*width/original_width if scaled else .1)
            _, edges, bins, valid, weights, joint_bins = prepare_histogram(histogram, gpu.observed, "cpu")
            assert valid.all(), "Fixed range must cover every observed history row"
            actual_width = float(edges[1]-edges[0])
            if width is not None:
                np.testing.assert_allclose(edges.numpy(), -.5*width+np.arange(len(edges))*width, atol=7e-7, rtol=0)
            geometry.append({"dataset": label, "setting": name, "width_ms": actual_width*1000,
                             "sigma_ms": sigma*actual_width*1000, "pseudocount": histogram.pseudocount,
                             "total_pseudocount": histogram.pseudocount*joint_bins,
                             "empty_density": histogram.pseudocount/((args.estimates+histogram.pseudocount*joint_bins)*actual_width)})
            logs, counts, durations = [], [], []
            for seed in range(101, 101+args.repeats):
                start = time.perf_counter()
                scored = histogram.score(gpu.stimulus, gpu.observed, gpu.rows,
                                         num_estimates=args.estimates, seed=seed, include_mask=include,
                                         execution="window", candidate_batch_size=2,
                                         max_buffer_bytes=512*1024**2, triton_launch_options=LAUNCH)
                logs.append(scored.log_factors)
                counts.append(scored.bin_counts)
                durations.append(time.perf_counter()-start)
            logs, counts = np.array(logs), np.array(counts)
            weighted = (counts*weights.numpy()[None, None]).sum(-1)
            means = logs.astype(float).mean(0)
            totals = logs[:, :, include].astype(float).sum(-1)
            for i, original in enumerate(candidates):
                gap = means[i, include]-direct[i, include]
                paired = totals[:, i]-totals[:, 0]
                record = {"dataset": label, "candidate": original, "setting": name,
                          "direct": float(direct[i, include].sum()), "gpu": float(totals[:, i].mean()),
                          "gpu_sd": float(totals[:, i].std(ddof=1)), "gap": float(gap.sum()),
                          "mae": float(abs(gap).mean()), "p95": float(np.quantile(abs(gap), .95)),
                          "empty_rows": float((weighted[:, i, include] == 0).sum(-1).mean()),
                          "weighted_counts_under_10_rows": float((weighted[:, i, include] < 10).sum(-1).mean()),
                          "paired_gpu_change": float(paired.mean()), "paired_gpu_sd": float(paired.std(ddof=1)),
                          "direct_change": float((direct[i, include]-direct[0, include]).sum()),
                          "seconds": float(sum(durations))}
                records.append(record)
                if label == "subject-1" and original == 9:
                    trial = 133
                    outlier_records.append({"setting": name, "direct_log_density": float(direct[i, trial]),
                                            "gpu_log_density": float(means[i, trial]),
                                            "gpu_log_density_sd": float(logs[:, i, trial].std(ddof=1)),
                                            "weighted_counts_mean": float(weighted[:, i, trial].mean()),
                                            "gap": float(means[i, trial]-direct[i, trial])})
            np.savez_compressed(folder / f"{name}.npz", gpu=logs, counts=counts, direct=direct,
                                vectors=vectors, include=include, candidates=candidates,
                                edges=edges.numpy(), bins=bins.numpy(), weights=weights.numpy())
            pd.DataFrame(records).to_csv(args.output / "summary.csv", index=False)
            pd.DataFrame(geometry).to_csv(args.output / "histogram_settings.csv", index=False)
            pd.DataFrame(outlier_records).to_csv(args.output / "collapse_outlier.csv", index=False)
            print(f"{label} {name}: anchor gap {records[-len(candidates)]['gap']:.3f}, "
                  f"MAE {records[-len(candidates)]['mae']:.4f}; {sum(durations):.1f}s", flush=True)
        del gpu

    table = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    order = [row[0] for row in SETTINGS]
    for label in CASES:
        anchor = table[(table.dataset == label) & (table.candidate == 0)].set_index("setting").loc[order]
        axes[0].plot(range(len(order)), anchor.mae, "o-", label=label)
        axes[1].plot(range(len(order)), anchor.gpu_sd, "o-", label=label)
    outlier = pd.DataFrame(outlier_records).set_index("setting").loc[order]
    axes[2].plot(range(len(order)), outlier.gap, "o-")
    for ax in axes:
        ax.set_xticks(range(len(order)), order, rotation=60, ha="right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set(title="Anchor per-trial agreement", ylabel="Mean absolute log-density difference")
    axes[1].set(title="Monte Carlo scatter", ylabel="SD of total GPU score across seeds")
    axes[2].set(title="Collapse-deadline outlier", ylabel="GPU − direct log density")
    axes[0].legend(fontsize=8)
    fig.savefig(args.output / "histogram_sensitivity.png", dpi=180)
    fig.savefig(args.output / "histogram_sensitivity.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
