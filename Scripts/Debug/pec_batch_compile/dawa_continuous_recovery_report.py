"""Compare converged DAWA recovery starts, fixed-parameter profiles and grids."""

import argparse
import json
from pathlib import Path

import numpy as np

from dawa_likelihood import PARAMETER_NAMES
from dawa_likelihood.model import PARAMETER_BOUNDS


def paired_score_difference(probabilities, reference, counts):
    """Mean paired log-score difference and SE for fixed condition sample sizes."""
    selected, total = counts > 0, counts.sum()
    mean = float(np.sum(counts[selected] * (np.log(probabilities[selected]) - np.log(reference[selected]))) / total)
    variance = 0.
    for pc, qc, nc in zip(probabilities, reference, counts):
        keep, size = nc > 0, nc.sum()
        delta = np.log(pc[keep]) - np.log(qc[keep])
        center = np.sum(nc[keep] * delta) / size
        variance += size * np.sum(nc[keep] * (delta - center)**2) / (size - 1)
    return mean, float(np.sqrt(variance) / total)


def summarize(reports):
    reference = reports[0]
    for report in reports[1:]:
        for key in ("truth", "conditions", "training_counts", "heldout_counts", "rt_edges", "points", "time_step"):
            if report[key] != reference[key]:
                raise ValueError(f"Incomparable recovery reports: {key}")
    total = np.asarray(reference["training_counts"]).sum()
    counts = np.asarray(reference["heldout_counts"])
    if np.any(counts.sum(axis=1) < 2):
        raise ValueError("At least two held-out observations per condition are required for paired standard errors.")
    heldout = counts.sum()
    verify_points = reference["fits"][0]["refined_check"]["points"]
    records, fine_probabilities = [], []
    for report in reports:
        for fit in report["fits"]:
            fine = fit["refined_check"]
            if fine["points"] != verify_points:
                raise ValueError("Verification grids must match across reports.")
            p, q = np.asarray(fine["fit_probabilities"]), np.asarray(fine["truth_probabilities"])
            mean, se = paired_score_difference(p, q, counts)
            train = -fit["accepted_losses"][-1]
            # Training and holdout supports can differ, especially in tail bins.
            train_counts = np.asarray(report["training_counts"])
            train_selected = train_counts > 0
            fine_train = float(np.sum(train_counts[train_selected] * np.log(p[train_selected])))
            records.append({"fixed": report["fixed_parameters"], "initial": fit["initial"], "fitted": fit["fitted"],
                            "converged": fit["success"], "termination": fit["message"], "iterations": fit["iterations"],
                            "projected_gradient_norm": fit["projected_gradient_norm"],
                            "training_log_likelihood": total * train, "verification_training_log_likelihood": fine_train,
                            "verification_points": fine["points"],
                            "verification_heldout_log_likelihood": heldout * fine["fit_heldout_per_trial"],
                            "heldout_log_score_difference_from_truth_per_trial": mean,
                            "paired_heldout_standard_error": se,
                            "maximum_joint_cdf_difference_fit_truth": fine["maximum_joint_cdf_difference_fit_truth"],
                            "evaluations": len(fit["evaluations"]),
                            "evaluation_seconds": sum(e["seconds"] for e in fit["evaluations"])})
            fine_probabilities.append(p)
            if "coarse_check" in fit:
                coarse = np.asarray(fit["coarse_check"]["fit_probabilities"])
                change = (coarse[:, :-1] - p[:, :-1]).reshape(len(counts), -1, 2).cumsum(axis=1)
                records[-1]["maximum_joint_cdf_change_under_grid_refinement"] = float(np.abs(change).max())
    best = max(r["training_log_likelihood"] for r in records if not r["fixed"])
    best_fine = max(r["verification_training_log_likelihood"] for r in records if not r["fixed"])
    best_index = max((i for i, r in enumerate(records) if not r["fixed"]), key=lambda i: records[i]["training_log_likelihood"])
    for record, prob in zip(records, fine_probabilities):
        record["training_log_likelihood_gap"] = best - record["training_log_likelihood"]
        record["verification_training_log_likelihood_gap"] = best_fine - record["verification_training_log_likelihood"]
        mean, se = paired_score_difference(prob, fine_probabilities[best_index], counts)
        record["heldout_log_score_difference_from_best_per_trial"] = mean
        record["paired_standard_error_against_best"] = se
    return {"truth": reference["truth"], "conditions": reference["conditions"], "points": reference["points"],
            "time_step": reference["time_step"], "training_trials": int(total), "heldout_trials": int(heldout),
            "records": records,
            "interpretation": "Verification scores re-evaluate the fitted parameters without refitting. Sparse fixed-mode fits are profile samples, not confidence intervals. Paired holdout SE describes sampling variation of the log-score difference."}


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bounds = np.asarray(PARAMETER_BOUNDS)
    low, width = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    truth = (np.array([summary["truth"][name] for name in PARAMETER_NAMES]) - low) / width
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].plot(np.arange(7), truth, "ko", label="Generating values")
    free = [r for r in summary["records"] if not r["fixed"]]
    for i, record in enumerate(free):
        value = (np.array([record["fitted"][name] for name in PARAMETER_NAMES]) - low) / width
        axes[0].plot(np.arange(7), value, "x--", ms=6, alpha=.8, label=f"Start {i + 1}")
    axes[0].set(xticks=np.arange(7), xticklabels=[n.replace("_", " ") for n in PARAMETER_NAMES],
                ylabel="Position within fitting range", title="Multiple starting points", ylim=(-.05, 1.05))
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(fontsize=8)
    constrained = {}
    for record in summary["records"]:
        if record["fixed"]:
            key = tuple(sorted(record["fixed"].items()))
            if key not in constrained or record["training_log_likelihood"] > constrained[key]["training_log_likelihood"]:
                constrained[key] = record
    profiles = sorted([r for r in constrained.values() if "lc_mode" in r["fixed"]], key=lambda r: r["fixed"]["lc_mode"])
    for field, label in (("training_log_likelihood_gap", f"{summary['points']}×{summary['points']} fit grid"),
                         ("verification_training_log_likelihood_gap", "Verification grid, no refit")):
        axes[1].plot([r["fixed"]["lc_mode"] for r in profiles], [r[field] for r in profiles], "o--", label=label)
    axes[1].axhline(0, color="black", lw=.7)
    best = max(free, key=lambda r: r["training_log_likelihood"])
    axes[1].scatter([best["fitted"]["lc_mode"]], [0.], marker="*", color="black", s=100, label="Best unrestricted fit", zorder=4)
    axes[1].axvline(summary["truth"]["lc_mode"], color="gray", lw=.7, ls=":")
    axes[1].set(xlabel="Fixed LC mode", ylabel="Log-likelihood below best unrestricted start",
                title="Re-optimizing the other six parameters")
    if profiles:
        axes[1].legend(fontsize=8)
    other_fixed = [r for r in constrained.values() if "lc_mode" not in r["fixed"]]
    records = free + profiles + other_fixed
    labels = ([f"Start {i + 1}" for i in range(len(free))] + [f"Mode {r['fixed']['lc_mode']:g}" for r in profiles]
              + [", ".join(f"{name.replace('lc_', '')} {value:g}" for name, value in r["fixed"].items()) for r in other_fixed])
    axes[2].errorbar(np.arange(len(records)), [r["heldout_log_score_difference_from_truth_per_trial"] for r in records],
                     yerr=[1.96 * r["paired_heldout_standard_error"] for r in records], fmt="o", capsize=3)
    axes[2].axhline(0, color="black", lw=.7)
    axes[2].set(xticks=np.arange(len(records)), xticklabels=labels, ylabel="Held-out log score / trial − generating model",
                title="Verification grid; ±1.96 paired SE")
    axes[2].tick_params(axis="x", rotation=45)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    for extension in (".png", ".pdf"):
        fig.savefig(output.with_suffix(extension), dpi=160)
    plt.close(fig)


def plot_trajectories(reports, output):
    """Contrast fitted latent LC trajectories with their observable predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from dawa_continuous_study import CONDITIONS
    from dawa_likelihood.continuous_native import continuous_path_native

    torch.set_num_threads(1)
    selected = {}
    for report in reports:
        for fit in report["fits"]:
            key = tuple(sorted(report["fixed_parameters"].items()))
            if key not in selected or fit["accepted_losses"][-1] < selected[key][1]["accepted_losses"][-1]:
                selected[key] = (report, fit)
    reference = reports[0]
    condition = reference["conditions"][0]
    _, task, stimulus = next(c for c in CONDITIONS if c[0] == condition)
    rows = [("Generating model", reference["truth"], reference["fits"][0]["refined_check"]["truth_probabilities"][0])]
    for key, (report, fit) in selected.items():
        label = "Unrestricted fit" if not key else ", ".join(f"{name.replace('_', ' ')} = {value:g}" for name, value in key)
        rows.append((label, fit["fitted"], fit["refined_check"]["fit_probabilities"][0]))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    diagnostics = []
    with torch.no_grad():
        for index, (label, parameters, prob) in enumerate(rows):
            p = torch.tensor([parameters[n] for n in PARAMETER_NAMES], dtype=torch.float64)
            path = continuous_path_native(p, task, stimulus, steps=1400, ode_step=.00025)
            t = np.arange(len(path.gain)) * path.time_step
            color = "black" if index == 0 else f"C{index - 1}"
            style = "-" if index == 0 else ["--", ":", "-."][(index - 1) % 3]
            axes[0, 0].plot(t, path.states[:, 8].numpy(), color=color, ls=style, label=label)
            axes[0, 1].plot(t, path.gain.numpy(), color=color, ls=style)
            cdf = np.asarray(prob[:-1]).reshape(-1, 2).cumsum(axis=0)
            for choice in (0, 1):
                axes[1, choice].plot(reference["rt_edges"][1:], cdf[:, choice], color=color, ls=style)
            diagnostics.append({"label": label, "parameters": parameters, "condition": condition,
                                "ode_step": .00025, "path_time_step": path.time_step, "lc_clock_ratio": 20.,
                                "maximum_lc_fast_state": float(path.states[:, 8].max()), "maximum_gain": float(path.gain.max())})
    axes[0, 0].set(xlabel="Decision time (s)", ylabel="LC fast state v", xlim=(0., .3), title="Latent LC transient")
    axes[0, 1].set(xlabel="Decision time (s)", ylabel="Network gain", xlim=(0., 1.4), title="Gain modulation")
    for choice in (0, 1):
        axes[1, choice].set(xlabel="Response time including NDT (s)", ylabel=f"P(choice={choice}, RT ≤ t)",
                            xlim=(.2, 1.5), title=f"Joint CDF, choice {choice}")
    axes[0, 0].legend(fontsize=8)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(condition.replace("_", " ").capitalize())
    for extension in (".png", ".pdf"):
        fig.savefig(output.with_suffix(extension), dpi=160)
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps(diagnostics, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectories", action="store_true", help="Also compare latent LC trajectories using the native ODE backend")
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.reports]
    summary = summarize(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    plot_summary(summary, args.output)
    if args.trajectories:
        plot_trajectories(reports, args.output.with_name(args.output.stem + "_trajectories"))


if __name__ == "__main__":
    main()
