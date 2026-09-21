#!/usr/bin/env python3
"""Summarize saved CSI likelihood sweeps without running new simulations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load_tables(root):
    tables = [pd.read_csv(p) for p in sorted(root.glob("*/summary.csv"))]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def save_figure(fig, root, name):
    fig.savefig(root / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def table_md(frame):
    # Avoid an optional tabulate dependency in the handoff environment.
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        return str(value)
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows += ["| " + " | ".join(cell(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    screen = load_tables(root / "screen")
    deep = load_tables(root / "deep")
    refinement = load_tables(root / "refinement")
    if screen.empty or deep.empty:
        parser.error("root must contain completed screen and deep dataset directories")
    deep_labels = list(deep.dataset.unique())
    real_count = sum(label.startswith("subject-") for label in deep_labels)
    synthetic_count = len(deep_labels) - real_count
    metadata = [json.loads((root / "deep" / label / "metadata.json").read_text()) for label in deep_labels]
    strict_count = sum(item["strict_window_checked"] for item in metadata)
    spatial_points = metadata[0]["config"]["ddm_spatial_points"]
    estimate_text = ", ".join(f"{int(n):,}" for n in sorted(deep.estimates.unique()))
    paired_rows = []
    for scores in sorted((root / "refinement").glob("*/trial_scores.npz")):
        data = np.load(scores)
        table = pd.read_csv(scores.parent / "summary.csv")
        totals = data["gpu"][:, :, data["include"]].astype(float).sum(-1)
        for i in range(1, len(table)):
            row = table.iloc[i]
            delta = totals[:, i] - totals[:, 0]
            paired_rows.append({"dataset": scores.parent.name, "kind": row.kind,
                                "direct_delta": row.direct_log_density-table.iloc[0].direct_log_density,
                                "gpu_delta": delta.mean(), "paired_gpu_sd": delta.std(ddof=1),
                                "gpu_delta_min": delta.min(), "gpu_delta_max": delta.max()})
    paired = pd.DataFrame(paired_rows)
    if len(paired):
        paired.to_csv(root / "refinement_paired_changes.csv", index=False)
    for name, frame in (("screen", screen), ("deep", deep), ("refinement", refinement)):
        if len(frame):
            frame.to_csv(root / f"{name}_summary.csv", index=False)
    screen = screen.sort_values("dataset", key=lambda col: col.str.extract(r"(\d+)")[0].astype(int))
    finite_screen = screen[screen.direct_zero_rows == 0]
    local = deep[(deep.kind == "anchor") | deep.kind.str.startswith("local:")]
    finite_local = local[local.direct_zero_rows == 0]
    metric_rows, strata, local_gaps, outliers = [], [], [], []
    for label, table in deep.groupby("dataset", sort=True):
        data = np.load(root / "deep" / label / "trial_scores.npz")
        original_rows = pd.read_csv(root / "deep" / label / "data.csv")
        anchor = table[table.kind == "anchor"].iloc[0]
        nearby = table[(table.kind == "anchor") | table.kind.str.startswith("local:")]
        valid = nearby[nearby.direct_zero_rows == 0]
        rank = spearmanr(valid.direct_log_density, valid.gpu_log_density_mean).statistic if len(valid) > 2 else np.nan
        metric_rows.append({"Dataset": label, "Direct": anchor.direct_log_density,
                            "GPU": anchor.gpu_log_density_mean, "GPU SD": anchor.gpu_log_density_sd,
                            "Gap/trial": anchor.finite_gap_mean, "Mean |gap|/trial": anchor.finite_gap_mae,
                            "Local rank correlation": rank,
                            "Local zero-support points": int((nearby.direct_zero_rows > 0).sum())})
        include = data["include"]
        for _, row in nearby.iterrows():
            i = int(row.candidate)
            direct = data["direct"][i]
            gpu = data["gpu"][:, i].mean(0)
            finite = include & np.isfinite(direct)
            if row.direct_zero_rows == 0:
                local_gaps.extend(abs(gpu[finite] - direct[finite]))
                indices = np.flatnonzero(finite)
                worst = indices[np.argsort(abs(gpu[indices]-direct[indices]))[-20:]]
                for trial in worst:
                    outliers.append({"dataset": label, "candidate": i, "kind": row.kind,
                                     "trial_index": int(trial), "row_id": int(original_rows.row_id.iloc[trial]),
                                     "condition": original_rows.sequence.iloc[trial],
                                     "choice": data["choice"][trial], "rt": data["rt"][trial],
                                     "direct": direct[trial], "gpu": gpu[trial],
                                     "gap": gpu[trial]-direct[trial]})
            for c in range(3):
                for choice in (0, 1):
                    selected = finite & (data["conditions"] == c) & (data["choice"] == choice)
                    if not selected.any():
                        continue
                    gap = gpu[selected] - direct[selected]
                    strata.append({"dataset": label, "candidate": i, "kind": row.kind,
                                   "condition": ("NoInstruction", "RealRare", "RealFrequent")[c],
                                   "choice": choice, "rows": int(selected.sum()),
                                   "mean_gap": gap.mean(), "mae": abs(gap).mean(),
                                   "p95": np.quantile(abs(gap), .95)})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(root / "dataset_metrics.csv", index=False)
    pd.DataFrame(strata).to_csv(root / "condition_choice_metrics.csv", index=False)
    outlier_frame = pd.DataFrame(outliers).sort_values("gap", key=lambda col: col.abs(), ascending=False).head(100)
    outlier_frame.to_csv(root / "largest_local_trial_gaps.csv", index=False)
    local_gaps = np.array(local_gaps)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    ax = axes[0, 0]
    ax.scatter(finite_screen.direct_log_density, finite_screen.gpu_log_density_mean, s=23, alpha=.8)
    bounds = [min(finite_screen.direct_log_density.min(), finite_screen.gpu_log_density_mean.min()),
              max(finite_screen.direct_log_density.max(), finite_screen.gpu_log_density_mean.max())]
    ax.plot(bounds, bounds, "--", color="0.4")
    ax.set(xlabel="Direct log-density sum", ylabel="GPU log-density sum",
           title=f"Saved solutions: {len(screen)} subjects ({len(finite_screen)} finite direct scores)")
    ax = axes[0, 1]
    ids = screen.dataset.str.extract(r"(\d+)")[0].astype(int)
    values = screen.gap_total / screen.included_rows
    ax.scatter(ids, values, s=18)
    ax.axhline(0, color="0.4", linestyle="--")
    ax.set(xlabel="Subject", ylabel="(GPU − direct) / included trials", title="Signed differences at saved solutions")
    ax = axes[1, 0]
    for label, table in finite_local.groupby("dataset"):
        anchor = table[table.kind == "anchor"]
        if len(anchor):
            ax.scatter(table.direct_log_density-anchor.iloc[0].direct_log_density,
                       table.gpu_log_density_mean-anchor.iloc[0].gpu_log_density_mean,
                       s=20, label=label.replace("subject-", "s"), alpha=.7)
    limits = ax.get_xlim(), ax.get_ylim()
    lo, hi = min(limits[0][0], limits[1][0]), max(limits[0][1], limits[1][1])
    ax.plot([lo, hi], [lo, hi], "--", color="0.4")
    ax.set(xlabel="Direct change from anchor", ylabel="GPU change from anchor",
           title="Local parameter probes with finite direct scores")
    ax.legend(fontsize=7, ncol=2)
    ax = axes[1, 1]
    ordered = np.sort(local_gaps)
    ax.plot(ordered, np.arange(1, len(ordered)+1)/len(ordered))
    ax.axvline(.5, color="0.5", linestyle="--", linewidth=1)
    ax.set(xlabel="Absolute per-trial log-density difference", ylabel="Fraction of trial–parameter pairs",
           xlim=(0, min(2., max(1., np.quantile(ordered, .995)))), ylim=(0, 1.01),
           title="Local agreement (finite direct candidate totals)")
    save_figure(fig, root, "likelihood_sweep_overview")

    heat = local.pivot(index="kind", columns="dataset", values="finite_gap_mean")
    support = local.pivot(index="kind", columns="dataset", values="direct_zero_rows")
    heat = heat.mask(support > 0)
    fig, ax = plt.subplots(figsize=(11, 10), layout="constrained")
    bound = max(.05, float(np.nanquantile(abs(heat.to_numpy()), .95)))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#cccccc")
    artist = ax.imshow(heat, cmap=cmap, vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(heat)), [k.replace("local:", "") for k in heat.index], fontsize=8)
    ax.set_title("Mean signed log-density difference per trial\nGrey: zero direct support or a duplicate bound-clipped probe")
    fig.colorbar(artist, ax=ax, label="GPU − direct (color scale clipped at 95th percentile)")
    save_figure(fig, root, "local_parameter_gaps")

    finite = deep[deep.direct_zero_rows == 0]
    summary = {
        "screen_subjects": len(screen), "screen_finite_subjects": len(finite_screen),
        "screen_median_signed_gap_per_trial": float(finite_screen.finite_gap_mean.median()),
        "screen_median_mae_per_trial": float(finite_screen.finite_gap_mae.median()),
        "screen_max_absolute_total_gap": float(finite_screen.gap_total.abs().max()),
        "screen_max_absolute_mean_gap": float(finite_screen.finite_gap_mean.abs().max()),
        "deep_points": len(deep), "deep_finite_points": len(finite),
        "local_points": len(local), "local_finite_points": len(finite_local),
        "local_trial_pairs": len(local_gaps),
        "local_pair_median_absolute_gap": float(np.median(local_gaps)),
        "local_pair_p95_absolute_gap": float(np.quantile(local_gaps, .95)),
        "local_pair_max_absolute_gap": float(local_gaps.max()),
        "local_pair_fraction_within_half_log_unit": float((local_gaps < .5).mean()),
        "local_rank_min": float(metrics["Local rank correlation"].min()),
        "local_rank_max": float(metrics["Local rank correlation"].max()),
    }
    (root / "statistics.json").write_text(json.dumps(summary, indent=2) + "\n")
    worst = finite_screen.reindex(finite_screen.gap_total.abs().sort_values(ascending=False).index).head(10)
    report = f"""# Local CSI GPU/direct likelihood sweep

The objectives agree approximately near the fitted/generating parameters, but they are not interchangeable. Across {len(local)} anchor/local points, {100*summary['local_pair_fraction_within_half_log_unit']:.2f}% of included trial–parameter pairs differ by less than 0.5 log units. Some nearby parameter rankings differ, and poor joint parameters can produce large histogram-pseudocount effects. The numerical tables below retain those exceptions.

## Scope and methods

- {len(screen)} real datasets at saved direct solutions; deeper sweeps on {real_count} real datasets and {synthetic_count} independently generated GPU datasets.
- {len(deep)} deep parameter–dataset combinations: anchors, one-at-a-time changes of all 13 parameters, eight joint probes per dataset, and search-bound stress cases.
- {estimate_text} GPU simulations per included trial and candidate (seed settings in run manifests). The direct solver uses 8 CPU threads, float64, 1 ms DDM steps, {spatial_points} spatial points, and RK4 LCA integration with maximum step 10 ms.
- GPU: corrected original PNL composition, compiled Triton path, 1 ms steps, 100 RT bins, Gaussian smoothing sigma 0.5 bin, pseudocount 0.1. Observed-history timing uses `ceil_fp32_8ulp`; CSI is rounded to the nearest millisecond in **both** objectives. Metadata retains the unsnapped saved parameters and their direct score.
- Every retained trial contributes to history; only the original inclusion mask contributes to the likelihood. Synthetic datasets retain subject 1's design and mask and replace choices/RTs using a full sequential GPU simulation (independent seeds 10101 and 10102).
- GPU window scoring stops samples only after they cannot contribute to the relevant bins; surviving paths are not renormalized. A 12,000-step cap still raises on true truncation. Strict/window equality was checked at {strict_count}/{len(deep_labels)} deep anchors using 512 estimates.
- Direct interval probabilities are divided by 0.001 s before taking logs. This aligns units, but the direct density and smoothed GPU histogram remain different estimators. No direct zero is floored for comparison.
- Joint probes constrain timing and collapse so observed responses remain in mathematical support. Stress probes deliberately do not impose that constraint. The archived parameter tables contain every attempted point.

## Numerical results

At saved solutions, {len(finite_screen)}/{len(screen)} common-grid direct totals are finite. Among these, the median signed gap is {summary['screen_median_signed_gap_per_trial']:.4f} log units per trial; median subject-level mean absolute gap is {summary['screen_median_mae_per_trial']:.4f}. The largest absolute mean gap is {summary['screen_max_absolute_mean_gap']:.4f}.

Of {len(local)} anchor/local points, {len(finite_local)} have finite direct totals. Across their {len(local_gaps):,} included trial–parameter pairs, the median absolute difference is {summary['local_pair_median_absolute_gap']:.4f}, the 95th percentile is {summary['local_pair_p95_absolute_gap']:.4f}, and {100*summary['local_pair_fraction_within_half_log_unit']:.2f}% are within 0.5 log units (a density ratio of about 1.65). These are descriptive agreement measures, not an equivalence test.

The maximum local per-trial difference is {summary['local_pair_max_absolute_gap']:.3f} log units. `largest_local_trial_gaps.csv` identifies the affected parameters, original row IDs, conditions, and response times.

{table_md(metrics)}

Largest absolute total differences at finite saved solutions:

{table_md(worst[['dataset', 'direct_log_density', 'gpu_log_density_mean', 'gpu_log_density_sd', 'gap_total', 'finite_gap_mae', 'gpu_empty_rows_mean']])}

Across all deep probes, {len(deep)-len(finite)} points contain at least one zero direct probability. A finite GPU score at such a point can reflect histogram smoothing or pseudocounts and must not be interpreted as numerical agreement. See `deep_summary.csv` for the number of unsupported rows and empty GPU bins at every point.

## Interpretation and caveats

The original PNL CSI composition executed by LLVM remains the model reference. Earlier independent model-output checks established the corrected GPU schedule against LLVM; this sweep audits likelihoods and does not rerun or rely on LLVM likelihood evaluation.

Aggregate agreement can hide condition-specific or trial-specific differences. `condition_choice_metrics.csv` reports each local probe by condition and observed choice. Residual differences may reflect rounded history timing, Euler versus RK4 LCA integration, endpoint versus continuous Brownian crossings, histogram smoothing, and Monte Carlo sampling. More estimates reduce sampling error but do not remove these other differences.

The saved population fits used the earlier search bounds. The deep probes use current ranges: gains 5–120, CSI 0–0.3 s, thresholds 0.05–0.3, collapse rates −0.3–0 per second, and NDT 0.1–0.5 s. Subject 1's deep anchor uses the expanded-bound direct solution. CSI snapping can move an optimized point off a sharp support boundary; compare each dataset's `metadata.json` to quantify this separately.

No fits were rerun, no production model semantics were changed by this audit, and no cluster jobs were submitted. The experiment ran on a local RTX 2080 Ti. Compile/setup and scoring durations are recorded separately where available; overlapping audit processes make them unsuitable as dedicated performance benchmarks.

## Files

- `likelihood_sweep_overview.png` / `.pdf`: saved solutions, local score changes, and per-trial agreement.
- `local_parameter_gaps.png` / `.pdf`: each parameter probe by dataset.
- `screen_summary.csv`, `deep_summary.csv`, `dataset_metrics.csv`, `condition_choice_metrics.csv`, `statistics.json`: aggregate results.
- Each dataset directory contains exact input rows, candidate vectors, per-trial scores for every seed, solver diagnostics, and timing/rounding metadata.
- Run manifests record source/data hashes, git state, library/device information, and numerical settings.
"""
    if len(refinement):
        refinement_n = ", ".join(f"{int(n):,}" for n in sorted(refinement.estimates.unique()))
        report += f"\n## Higher-sample rechecks ({refinement_n} estimates)\n\n" + table_md(refinement[["dataset", "kind", "direct_log_density", "gpu_log_density_mean", "gpu_log_density_sd", "gap_total", "finite_gap_mae"]]) + "\n"
        report += "\nSelected local changes relative to the same dataset's anchor (paired seeds):\n\n" + table_md(paired[paired.kind.str.contains(":local:")]) + "\n"
    convergence = root / "convergence/direct_convergence.csv"
    if convergence.exists():
        mesh = pd.read_csv(convergence)
        report += "\n## Direct numerical convergence\n\n" + table_md(mesh[["dataset", "configuration", "delta_log_density", "finite_trial_mae_change", "direct_zero_rows"]]) + "\n"
    collapse_path = root / "collapse_deadline_diagnostic.csv"
    if collapse_path.exists():
        collapse = pd.read_csv(collapse_path)
        selected = collapse[(collapse.candidate == 9) & (collapse.trial_index == 133)]
        report += """
## Largest local outlier: matching the histogram estimator

Subject 1's NoInstruction threshold reduced by 5% produces the largest local discrepancy at trial index 133 (original row ID 202). Its observed RT is 1.012 s, just before the direct collapse deadline at 1.021260 s. The GPU bin is 26.3016 ms wide, with edges 0.986357–1.012659 s; smoothing also draws from neighboring bins.

We integrated the direct PDE's boundary flux over the **same GPU bin edges**, applied the same smoothing weights and 100,000-estimate pseudocount normalization, and retained the direct model's history. That changes the comparison from an approximately 9.42-log-unit gap to 0.52. Most of this outlier is an estimator mismatch. A residual difference remains; matching the estimator does not establish identical simulator dynamics.

The finer PDE mesh changes the sharp point density somewhat, but barely changes the matched histogram result:

""" + table_md(selected[["spatial_points", "dt", "direct_1ms_log_density", "direct_matched_histogram_log_density", "gpu_log_density"]])
        report += "\n\nSee `collapse_deadline_explanation.png` / `.pdf` and `collapse_deadline_diagnostic.csv`. The diagnostic script is archived under `reproduction_sources/`; run it from the repository root.\n"
    report += """

## Reproduction

Use the commands in the parent CSI README to generate a new sweep. To reproduce this exact parameter set, use `csi_likelihood_parameter_sweep.py --replay-from /path/to/this/archive/deep --estimates 20000 --repeats 3 --output /tmp/csi-deep-replay`. This reads the archived candidate vectors and input rows, including the independent synthetic data, rather than generating new points. Use a fresh output directory when changing settings.

The archive includes `working_tree.patch`, `reproduction_sources/`, and source hashes. The sweep was performed on the working tree containing the preceding scheduling fix; the git commit alone is insufficient to identify that code. Multiple manifests preserve resumed invocations. Incomplete directories/log traces are retained from an interrupted population worker and the corrected candidate-generation guard for masked short RTs; the final summaries contain only completed datasets.
"""
    (root / "README.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
