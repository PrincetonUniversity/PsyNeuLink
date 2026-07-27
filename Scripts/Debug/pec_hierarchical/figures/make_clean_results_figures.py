#!/usr/bin/env python3
"""Create a compact, science-first PEC figure set from completed artifacts.

This script performs presentation-only analysis.  It does not simulate data,
train a neural likelihood, or fit a model.  It must run inside SLURM.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path

if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("clean figure generation must run inside a SLURM allocation")

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / f"matplotlib-pec-clean-{os.getpid()}"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import make_final_results_figures as evidence


REPO = Path("/scratch/gpfs/JDC/ap9344/PsyNeuLink")
EVIDENCE_DIR = (
    REPO
    / "Scripts/Debug/pec_hierarchical/figures/final_results_20260726"
)
DEFAULT_OUT = (
    REPO
    / "Scripts/Debug/pec_hierarchical/figures/clean_results_20260727"
)

WHITE = "#FFFFFF"
INK = "#252525"
MUTED = "#666666"
GRID = "#E5E5E5"
TRUTH = "#3F3F3F"
NLE = "#0072B2"
KDE = "#E69F00"
KDE_DARK = "#D55E00"
CORRECTED = "#7B3294"
GREEN = "#009E73"
UNCERTAIN = "#E69F00"

PARAMETERS = evidence.PARAMETERS
SHORT_PARAMETERS = evidence.SHORT_PARAMETERS

plt.rcParams.update(
    {
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "text.color": INK,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "axes.grid.axis": "both",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "figure.titlesize": 18,
    }
)


DDM_METHODS = [
    "nle",
    "full_gaussian_log_pooled_b1200_r0",
    "exact",
    "full_fast_raw_local_b5000_r0",
    "full_fast_raw_local_b120_r0",
]

DDM_LABELS = {
    "nle": "Neural likelihood",
    "full_gaussian_log_pooled_b1200_r0": "Corrected pooled KDE\n(1,200 simulations)",
    "exact": "Exact likelihood",
    "full_fast_raw_local_b5000_r0": "Local KDE\n(5,000 simulations)",
    "full_fast_raw_local_b120_r0": "Original local KDE\n(120 simulations)",
}

DDM_COLORS = {
    "nle": NLE,
    "full_gaussian_log_pooled_b1200_r0": CORRECTED,
    "exact": TRUTH,
    "full_fast_raw_local_b5000_r0": KDE_DARK,
    "full_fast_raw_local_b120_r0": KDE,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.svg"]
    fig.savefig(paths[0], dpi=220, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def add_title(
    fig: plt.Figure,
    title: str,
    subtitle: str,
    *,
    subtitle_y: float = 0.925,
) -> None:
    fig.suptitle(title, x=0.02, y=0.985, ha="left")
    fig.text(0.02, subtitle_y, subtitle, color=MUTED, fontsize=10.5)


def plot_ddm_recovery(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.7), sharey=True)
    parameter_panels = [("drift_rate", "Drift rate"), ("threshold", "Decision threshold")]
    y = np.arange(len(DDM_METHODS))

    for ax, (parameter, title) in zip(axes, parameter_panels, strict=True):
        frame = (
            summary[
                (summary["parameter"] == parameter)
                & summary["method"].isin(DDM_METHODS)
            ]
            .set_index("method")
            .loc[DDM_METHODS]
        )
        values = frame["mean_rmse"].to_numpy(float)
        low = frame["bootstrap_95_low"].to_numpy(float)
        high = frame["bootstrap_95_high"].to_numpy(float)
        bars = ax.barh(
            y,
            values,
            color=[DDM_COLORS[method] for method in DDM_METHODS],
            height=0.62,
            zorder=3,
        )
        ax.errorbar(
            values,
            y,
            xerr=[values - low, high - values],
            fmt="none",
            ecolor=INK,
            elinewidth=1.1,
            capsize=3,
            zorder=4,
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                value + max(values) * 0.025,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontsize=9.5,
                color=MUTED,
            )
        ax.set_title(title)
        ax.set_xlabel("Participant-level RMSE")
        ax.set_xlim(0, max(values) * 1.22)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([DDM_LABELS[method] for method in DDM_METHODS])
    add_title(
        fig,
        "Neural likelihood improves DDM parameter recovery",
        "Mean error across 30 independent simulated datasets; bars show 95% intervals. Lower is better.",
    )
    fig.text(
        0.02,
        0.015,
        "Descriptive comparison: simulator agreement was highly compatible, but the formal validation gate remained incomplete.",
        fontsize=9,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.12, top=0.84, wspace=0.20)
    return save_figure(fig, out_dir, "01_ddm_recovery_clean")


def plot_ddm_cost(latency: pd.DataFrame, out_dir: Path) -> list[Path]:
    selected = {
        "exact": "WFPT exact",
        "nle": "NLE cold",
        "full_fast_raw_local_b120_r0": "fastKDE local · B120",
        "full_gaussian_log_pooled_b1200_r0": "log-FPT Gaussian pooled · B1200",
        "full_fast_raw_local_b5000_r0": "fastKDE local · B5000",
    }
    rows = []
    for method, label in selected.items():
        row = latency[(latency["method"] == method) & (latency["label"] == label)]
        if len(row) != 1:
            raise RuntimeError(f"missing unique latency row for {method}: {label}")
        rows.append(row.iloc[0])
    frame = pd.DataFrame(rows)

    pretty = {
        "exact": "Exact",
        "nle": "Neural likelihood\n(training included)",
        "full_fast_raw_local_b120_r0": "Original local KDE",
        "full_gaussian_log_pooled_b1200_r0": "Corrected pooled KDE",
        "full_fast_raw_local_b5000_r0": "High-budget local KDE",
    }
    offsets = {
        "exact": (7, -16),
        "nle": (8, -3),
        "full_fast_raw_local_b120_r0": (8, 4),
        "full_gaussian_log_pooled_b1200_r0": (8, -16),
        "full_fast_raw_local_b5000_r0": (-8, 7),
    }
    fig, ax = plt.subplots(figsize=(10.6, 6.8))
    for _, row in frame.iterrows():
        method = str(row["method"])
        minutes = float(row["median_seconds"]) / 60.0
        marker = "*" if method == "nle" else "o"
        size = 180 if method == "nle" else 95
        ax.scatter(
            minutes,
            row["rmse"],
            s=size,
            marker=marker,
            color=DDM_COLORS[method],
            edgecolor=WHITE,
            linewidth=0.9,
            zorder=4,
        )
        dx, dy = offsets[method]
        ax.annotate(
            pretty[method],
            (minutes, row["rmse"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="right" if dx < 0 else "left",
            fontsize=10,
            color=INK,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Median end-to-end allocation time (minutes, log scale)")
    ax.set_ylabel("Mean participant-level RMSE")
    ax.set_ylim(0.035, 0.162)
    add_title(
        fig,
        "Neural likelihood reaches low error at much lower computational cost",
        "One DDM fit; the neural-likelihood time includes simulation, training, and hierarchical fitting.",
    )
    fig.text(
        0.02,
        0.015,
        "Queue waiting time is excluded. Times are observed SLURM allocations, not hardware-normalized benchmarks.",
        fontsize=9,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.14, top=0.84)
    return save_figure(fig, out_dir, "02_ddm_accuracy_cost_clean")


def collapse_profile(profile: pd.DataFrame, method: str) -> pd.DataFrame:
    frame = profile.copy()
    frame["collapsed_method"] = frame["method"].map(evidence.collapse_surface_method)
    if method == "exact":
        frame = frame[frame["method"] == "exact"]
    else:
        frame = frame[frame["collapsed_method"] == method]
    grouped = (
        frame.assign(
            rate_value=pd.to_numeric(frame["rate"], errors="coerce"),
            ll_value=pd.to_numeric(frame["centered_log_likelihood"], errors="coerce"),
        )
        .dropna(subset=["rate_value", "ll_value"])
        .groupby("rate_value", as_index=False)
        .agg(
            ll=("ll_value", "mean"),
            holes=("internal_zero_hole_candidate_count", "max"),
        )
        .sort_values("rate_value")
    )
    grouped["relative_ll"] = grouped["ll"] - grouped["ll"].max()
    return grouped


def plot_kde_mechanism(
    surface: pd.DataFrame,
    profile: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    fig, (ax_profile, ax_holes) = plt.subplots(
        1,
        2,
        figsize=(13.2, 6.7),
        gridspec_kw={"width_ratios": [1.65, 1.0]},
    )
    profile_specs = [
        ("exact", "Exact likelihood", TRUTH, 2.7),
        ("nle", "Neural likelihood", NLE, 2.7),
        (
            "gaussian_log_condition_pooled",
            "Corrected pooled KDE",
            CORRECTED,
            2.2,
        ),
        ("fastkde_local", "Original local KDE", KDE_DARK, 2.2),
    ]
    local_group = None
    for method, label, color, linewidth in profile_specs:
        group = collapse_profile(profile, method)
        ax_profile.plot(
            group["rate_value"],
            group["relative_ll"],
            color=color,
            linewidth=linewidth,
            label=label,
            zorder=4,
        )
        if method == "fastkde_local":
            local_group = group

    if local_group is None:
        raise RuntimeError("local fastKDE profile is missing")
    holes = local_group["holes"].to_numpy(float) > 0
    ax_profile.scatter(
        local_group.loc[holes, "rate_value"],
        local_group.loc[holes, "relative_ll"],
        marker="x",
        s=34,
        linewidth=1.4,
        color="#B2182B",
        label="At least one internal zero",
        zorder=6,
    )
    ax_profile.set_xlabel("Candidate drift rate")
    ax_profile.set_ylabel("Log likelihood relative to each curve's maximum")
    ax_profile.set_ylim(-1050, 35)
    ax_profile.set_title("A. Sparse KDE distorts the likelihood curve")
    ax_profile.legend(frameon=False, fontsize=9, loc="lower right")

    hole_specs = [
        ("fastkde_local", "Trial-wise KDE", KDE_DARK),
        ("fastkde_condition_pooled", "Condition-pooled KDE", KDE),
        (
            "gaussian_log_condition_pooled",
            "Corrected pooled KDE",
            CORRECTED,
        ),
    ]
    budgets = [120, 1200, 5000]
    x = np.arange(len(budgets))
    for method, label, color in hole_specs:
        frame = surface[surface["method"] == method].set_index("budget")
        values = np.asarray(
            [100.0 * float(frame.loc[budget, "internal_hole_fraction"]) for budget in budgets]
        )
        ax_holes.plot(
            x,
            values,
            marker="o",
            linewidth=2.3,
            markersize=7,
            color=color,
            label=label,
        )
    ax_holes.set_xticks(x)
    ax_holes.set_xticklabels(["120", "1,200", "5,000"])
    ax_holes.set_xlabel("Simulations per candidate")
    ax_holes.set_ylabel("Candidate values with ≥1 internal zero (%)")
    ax_holes.set_ylim(-3, 95)
    ax_holes.set_title("B. More simulations help, but do not eliminate holes")
    ax_holes.legend(frameon=False, fontsize=9)

    add_title(
        fig,
        "Sparse trial-wise KDE creates artificial gaps in the likelihood",
        "One representative likelihood slice (left) and aggregate diagnostics across simulation budgets (right).",
    )
    fig.text(
        0.02,
        0.015,
        "Internal zero: the observed reaction time is inside the simulated same-choice range, yet the density estimate is zero.",
        fontsize=9,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.13, top=0.83, wspace=0.24)
    return save_figure(fig, out_dir, "03_kde_holes_clean")


def plot_validation(validation: pd.DataFrame, out_dir: Path) -> list[Path]:
    counts = validation["status"].value_counts()
    equivalent = int(counts.get("equivalent", 0))
    uncertain = int(counts.get("inconclusive_design_failure", 0))
    mismatch = int(counts.get("non_equivalent_with_adequate_precision", 0))
    total = equivalent + uncertain + mismatch

    fig, ax = plt.subplots(figsize=(11.2, 3.8))
    ax.barh(
        [0],
        [equivalent],
        color=GREEN,
        height=0.46,
        label="Equivalent",
    )
    ax.barh(
        [0],
        [uncertain],
        left=[equivalent],
        color=UNCERTAIN,
        height=0.46,
        label="Inconclusive",
    )
    ax.text(
        equivalent / 2,
        0,
        f"{equivalent}\nequivalent",
        ha="center",
        va="center",
        color=WHITE,
        fontsize=15,
        fontweight="semibold",
    )
    ax.annotate(
        f"{uncertain} inconclusive",
        (equivalent + uncertain / 2, 0),
        xytext=(-15, 48),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=12,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": MUTED},
    )
    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_xlabel(f"Predeclared simulator comparisons (total {total})")
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.text(
        0.0,
        -0.42,
        f"{mismatch} clear mismatches",
        transform=ax.transAxes,
        fontsize=11.5,
        color=GREEN,
        fontweight="semibold",
    )
    ax.text(
        1.0,
        -0.42,
        "All 637 point estimates were inside the tested margins",
        ha="right",
        transform=ax.transAxes,
        fontsize=10.5,
        color=MUTED,
    )
    add_title(
        fig,
        "The two DDM simulators agreed within the tested margins",
        "Twelve independent replications across a 7 × 7 parameter grid.",
        subtitle_y=0.87,
    )
    fig.text(
        0.02,
        0.015,
        "Formal confirmation remains incomplete because 15 uncertainty intervals were not precise enough.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.31, top=0.69)
    return save_figure(fig, out_dir, "04_simulator_agreement_clean")


def plot_stabflex_rmse(metrics: pd.DataFrame, out_dir: Path) -> list[Path]:
    methods = ["no-pool MLE", "KDE-EM", "neural-EM"]
    labels = ["No pooling", "KDE", "Neural likelihood"]
    colors = ["#A0A0A0", KDE, NLE]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2))

    for ax, parameter, short in zip(
        axes.ravel(), PARAMETERS, SHORT_PARAMETERS, strict=True
    ):
        frame = (
            metrics[metrics["parameter"] == parameter]
            .set_index("method")
            .loc[methods]
        )
        values = frame["rmse"].to_numpy(float)
        reduction = 100.0 * (values[1] - values[2]) / values[1]
        bars = ax.bar(np.arange(3), values, color=colors, width=0.66)
        digits = 3 if parameter == "gain" else 4
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values) * 0.025,
                f"{value:.{digits}f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                color=MUTED,
            )
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(labels)
        ax.set_ylabel("RMSE")
        ax.set_ylim(0, max(values) * 1.24)
        display = short if short == "NDT" else short.capitalize()
        ax.set_title(f"{display} ({reduction:.0f}% lower than KDE)")
        ax.grid(axis="x", visible=False)

    add_title(
        fig,
        "Neural likelihood lowers StabFlex recovery error",
        "Natural parameter units, shown on a separate linear scale for each parameter. Lower is better.",
    )
    fig.text(
        0.02,
        0.015,
        "One 48-participant × 250-trial synthetic dataset. The slope improvement is mainly reduced bias and shrinkage, not recovery of individual differences.",
        fontsize=9,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.11, top=0.84, hspace=0.35, wspace=0.24)
    return save_figure(fig, out_dir, "05_stabflex_rmse_clean")


def plot_stabflex_curves(
    truth: np.ndarray,
    estimates: dict[str, np.ndarray],
    metrics: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    x = np.arange(1, truth.shape[0] + 1)
    for index, (ax, parameter, short) in enumerate(
        zip(axes.ravel(), PARAMETERS, SHORT_PARAMETERS, strict=True)
    ):
        order = np.argsort(truth[:, index])
        nle_corr = float(
            metrics.loc[
                (metrics["method"] == "neural-EM")
                & (metrics["parameter"] == parameter),
                "correlation",
            ].iloc[0]
        )
        ax.plot(
            x,
            truth[order, index],
            color=TRUTH,
            linewidth=2.8,
            label="True value",
            zorder=5,
        )
        ax.plot(
            x,
            estimates["KDE-EM"][order, index],
            color=KDE,
            linewidth=1.5,
            alpha=0.82,
            label="KDE",
        )
        ax.plot(
            x,
            estimates["neural-EM"][order, index],
            color=NLE,
            linewidth=2.0,
            label="Neural likelihood",
        )
        display = short if short == "NDT" else short.capitalize()
        ax.set_title(f"{display}  (neural r = {nle_corr:.2f})")
        ax.set_xlabel("Participants ordered by true value")
        ax.set_ylabel("Parameter value")
        if index == 0:
            ax.legend(frameon=False, fontsize=9)

    add_title(
        fig,
        "Neural likelihood tracks gain and threshold, but not slope",
        "Participants are independently ordered by their true parameter value within each panel.",
    )
    fig.text(
        0.02,
        0.015,
        "Completed 48 × 250 StabFlex study. This is one synthetic dataset, and neural hierarchical fitting reached its iteration limit.",
        fontsize=9,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.84, hspace=0.36, wspace=0.22)
    return save_figure(fig, out_dir, "06_stabflex_curves_clean")


def plot_stabflex_slope(
    truth: np.ndarray,
    estimates: dict[str, np.ndarray],
    out_dir: Path,
) -> list[Path]:
    fixed_ll, _, info_params, _ = evidence.load_stab_slope_diagnostics()
    median_information = np.median(fixed_ll, axis=0)
    slope_index = PARAMETERS.index("slope")

    fig, (ax_scatter, ax_info) = plt.subplots(1, 2, figsize=(12.6, 6.2))
    ax_scatter.plot(
        [0, 0.045],
        [0, 0.045],
        linestyle="--",
        color=MUTED,
        linewidth=1.2,
        label="Perfect recovery",
    )
    ax_scatter.scatter(
        truth[:, slope_index],
        estimates["KDE-EM"][:, slope_index],
        s=40,
        color=KDE,
        alpha=0.72,
        edgecolor=WHITE,
        linewidth=0.5,
        label="KDE",
    )
    ax_scatter.scatter(
        truth[:, slope_index],
        estimates["neural-EM"][:, slope_index],
        s=40,
        color=NLE,
        alpha=0.78,
        edgecolor=WHITE,
        linewidth=0.5,
        label="Neural likelihood",
    )
    ax_scatter.set_xlim(0, 0.045)
    ax_scatter.set_ylim(0, 0.10)
    ax_scatter.set_xlabel("True slope")
    ax_scatter.set_ylabel("Estimated slope")
    ax_scatter.set_title("A. Participant slopes are not recovered")
    ax_scatter.legend(frameon=False, fontsize=9)

    x = np.arange(len(info_params))
    colors = [NLE if name != "slope" else KDE for name in info_params]
    bars = ax_info.bar(x, median_information, color=colors, width=0.62)
    for bar, value in zip(bars, median_information, strict=True):
        ax_info.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.12,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=MUTED,
        )
    ax_info.set_yscale("log")
    ax_info.set_xticks(x)
    ax_info.set_xticklabels(
        [
            "NDT" if name in {"ndt", "non_decision_time"} else name.capitalize()
            for name in info_params
        ]
    )
    ax_info.set_ylabel("Likelihood change across the parameter range (nats, log)")
    ax_info.set_title("B. The likelihood changes very little with slope")

    add_title(
        fig,
        "The StabFlex slope is weakly identified",
        "Recovery in the main study (left) and an earlier 12-participant × 120-trial information diagnostic (right).",
    )
    fig.text(
        0.02,
        0.015,
        "0.04 is not a lower bound. With little slope information, hierarchical shrinkage pulls participant estimates toward one group value.",
        fontsize=9,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.81, wspace=0.27)
    return save_figure(fig, out_dir, "07_stabflex_slope_clean")


def write_readme(out_dir: Path, output_paths: list[Path]) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID", "unknown")
    lines = [
        "# Clean hierarchical PEC figure set",
        "",
        f"Generated from completed artifacts in SLURM job `{job_id}`. No data were simulated and no model was trained or fitted by this job.",
        "",
        "This folder is the presentation-oriented companion to the detailed evidence pack:",
        f"`{EVIDENCE_DIR}`.",
        "",
        "## Reading the figures",
        "",
        "1. `01_ddm_recovery_clean`: neural likelihood has the lowest overall DDM recovery error; corrected pooled KDE approaches it.",
        "2. `02_ddm_accuracy_cost_clean`: neural likelihood remains inexpensive even when one full training run is charged to the fit.",
        "3. `03_kde_holes_clean`: sparse trial-wise KDE creates internal zero-density gaps and a distorted likelihood curve.",
        "4. `04_simulator_agreement_clean`: simulator results were highly compatible, but 15 comparisons lacked enough precision for a complete formal pass.",
        "5. `05_stabflex_rmse_clean`: neural likelihood lowers natural-unit RMSE for all four StabFlex parameters.",
        "6. `06_stabflex_curves_clean`: gain and threshold track participant differences well; NDT is partial; slope is not recovered.",
        "7. `07_stabflex_slope_clean`: slope has far less likelihood information than the other parameters and is strongly shrunk.",
        "",
        "## Minimal terminology",
        "",
        "- **KDE**: kernel-density likelihood estimated directly from model simulations.",
        "- **Neural likelihood / NLE**: a network trained on simulations to approximate the likelihood.",
        "- **RMSE**: root-mean-square parameter recovery error; lower is better.",
        "- **Simulation budget**: number of simulated outcomes used at a candidate parameter value.",
        "",
        "## Essential caveats",
        "",
        "- DDM results are strong descriptive evidence, not a completed confirmatory claim, because the frozen simulator-validation gate was not formally passed.",
        "- StabFlex results come from one synthetic dataset, and neural hierarchical fitting reached its iteration limit.",
        "- Lower slope RMSE for neural likelihood mostly reflects reduced bias and shrinkage; it does not mean participant-level slope differences were recovered.",
        "",
        f"Files emitted: {len(output_paths)} figure files ({len(output_paths) // 2} PNG/SVG pairs).",
    ]
    path = out_dir / "README.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_manifest(
    out_dir: Path,
    output_paths: list[Path],
    readme: Path,
    source_paths: list[Path],
) -> Path:
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "clean presentation figures; descriptive evidence",
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node_list": os.environ.get("SLURM_NODELIST"),
        },
        "host": socket.gethostname(),
        "outputs": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in [*sorted(output_paths), readme]
        ],
        "sources": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_paths
        ],
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] DDM recovery")
    recovery = pd.read_csv(EVIDENCE_DIR / "ddm_recovery_summary.csv")
    output_paths = plot_ddm_recovery(recovery, args.out_dir)

    print("[2/7] DDM accuracy versus cost")
    latency = pd.read_csv(EVIDENCE_DIR / "ddm_accuracy_cost_points.csv")
    output_paths += plot_ddm_cost(latency, args.out_dir)

    print("[3/7] KDE likelihood holes")
    surface, profile = evidence.read_surface_diagnostics()
    output_paths += plot_kde_mechanism(surface, profile, args.out_dir)

    print("[4/7] DDM simulator agreement")
    validation = evidence.read_validation()
    output_paths += plot_validation(validation, args.out_dir)

    print("[5/7] StabFlex recovery error")
    _, truth, estimates = evidence.load_stab_run(evidence.STAB_BIG)
    stab_metrics = evidence.stab_metrics(truth, estimates)
    output_paths += plot_stabflex_rmse(stab_metrics, args.out_dir)

    print("[6/7] StabFlex participant curves")
    output_paths += plot_stabflex_curves(
        truth,
        estimates,
        stab_metrics,
        args.out_dir,
    )

    print("[7/7] StabFlex slope limitation")
    output_paths += plot_stabflex_slope(truth, estimates, args.out_dir)

    readme = write_readme(args.out_dir, output_paths)
    source_paths = [
        Path(__file__).resolve(),
        REPO
        / "Scripts/Debug/pec_hierarchical/figures/submit_clean_results_figures.slurm",
        EVIDENCE_DIR / "ddm_recovery_summary.csv",
        EVIDENCE_DIR / "ddm_accuracy_cost_points.csv",
        evidence.DDM_RUN / "analysis/surface_surface_points.csv",
        evidence.DDM_RUN / "simulator_validation/equivalence_cells.csv",
        evidence.STAB_BIG / "study_data.pkl",
        evidence.STAB_BIG / "theta_mle.npy",
        evidence.STAB_BIG / "theta_kde_em.npy",
        evidence.STAB_BIG / "theta_neural_em.npy",
        evidence.STAB_NLE / "ll_information.npz",
    ]
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing source files:\n" + "\n".join(map(str, missing)))
    manifest = write_manifest(args.out_dir, output_paths, readme, source_paths)
    print(
        json.dumps(
            {
                "complete": True,
                "out_dir": str(args.out_dir),
                "figures": [str(path) for path in output_paths],
                "readme": str(readme),
                "manifest": str(manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
