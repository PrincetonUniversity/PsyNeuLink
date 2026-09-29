"""Summarize completed matched-observation recovery runs and LC-mode profiles.

Intervals describe Monte Carlo uncertainty on one fixed synthetic subject. They
are neither parameter confidence intervals nor uncertainty across subjects.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dawa_conditioned_accuracy import mean_statistics


LABELS = ["Threshold", "Nondecision time", "SDR bias", "Control gain",
          "LC mode 0", "LC mode 1", "LC scaling", "LC base gain"]


def profile_summary(profile):
    """Pair complete-filter scores by seed, rejecting missing/duplicate records."""
    if profile["status"] != "complete":
        raise ValueError("Profiles must be complete")
    config = profile["config"]
    seeds, budgets = config["validation_seeds"], config["validation_estimates"]
    candidates = profile["candidates"]
    labels = [row["label"] for row in candidates]
    if len(set(labels)) != len(labels) or len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("Unique candidate labels and at least two unique seeds are required")
    records = {}
    for row in profile["validation"]:
        key = (row["estimates"], row["seed"], row["label"])
        if key in records or not np.isfinite(row["selected_log_score"]):
            raise ValueError("Duplicate or nonfinite validation score")
        records[key] = row
    expected = {(n, seed, label) for n in budgets for seed in seeds for label in labels}
    if set(records) != expected:
        raise ValueError("Validation scores do not form the complete declared seed matrix")
    anchor = config["anchor"]["label"]
    anchor_index = labels.index(anchor)
    matrices, summaries = {}, []
    for budget in budgets:
        matrix = np.array([[records[(budget, seed, label)]["selected_log_score"] for label in labels] for seed in seeds])
        matrices[str(budget)] = matrix.tolist()
        for index, label in enumerate(labels):
            rows = [records[(budget, seed, label)] for seed in seeds]
            summaries.append({"estimates": budget, "label": label,
                              "score": mean_statistics(matrix[:, index]),
                              "difference_from_anchor": mean_statistics(matrix[:, index] - matrix[:, anchor_index]),
                              "minimum_ess": min(row["minimum_ess"] for row in rows),
                              "median_trial_median_ess": float(np.median([row["median_ess"] for row in rows])),
                              "maximum_contamination_above_half": max(row["contamination_above_half"] for row in rows)})
    largest = max(budgets)
    drift = []
    high = np.array(matrices[str(largest)])
    for budget in budgets:
        if budget == largest:
            continue
        delta = np.array(matrices[str(budget)]) - high
        for index, label in enumerate(labels):
            drift.append({"estimates": budget, "reference_estimates": largest, "label": label,
                          "absolute_score_difference": mean_statistics(delta[:, index]),
                          "anchor_comparison_difference": mean_statistics(delta[:, index] - delta[:, anchor_index])})
    return {"config": config, "environment": profile["environment"], "points": profile["points"],
            "candidates": candidates, "labels": labels, "seeds": seeds,
            "seed_score_matrices": matrices, "summary": summaries, "particle_budget_drift": drift,
            "invalid_proposals": len(profile["invalid_proposals"]), "total_seconds": profile["total_seconds"],
            "source_sha256": profile["source_sha256"]}


def timing_comparison(reference, h100):
    for key in ("data_sha256", "candidates", "histogram", "max_steps", "batch_size", "launch", "execution"):
        if reference["config"][key] != h100["config"][key]:
            raise ValueError(f"Timing workloads differ: {key}")
    if reference["implementation_sha256"] != h100["implementation_sha256"]:
        raise ValueError("Timing sources differ")
    left = {row["estimates"]: row for row in reference["runtime"]["budgets"]}
    result = []
    for row in h100["runtime"]["budgets"]:
        n = row["estimates"]
        if n not in left:
            continue
        result.append({"estimates": n, "reference": left[n], "h100": row,
                       "speedup": left[n]["median_seconds_per_candidate"] / row["median_seconds_per_candidate"]})
    return {"budgets": result, "reference_environment": reference["environment"],
            "h100_environment": h100["environment"], "reference_seeds": reference["config"]["seeds"],
            "h100_seeds": h100["config"]["seeds"], "note": h100["runtime"]["note"],
            "data_sha256": h100["config"]["data_sha256"], "h100_batches": h100["runtime"]["batches"]}


def recovery_comparisons(fits):
    """Compare frozen fits using matched complete-filter replicates."""
    comparisons = []
    for i in range(1, len(fits)):
        left, right = fits[0], fits[i]
        if left["validation_estimates"] != right["validation_estimates"]:
            raise ValueError("Recovery validation particle budgets differ")
        left_scores = {r["seed"]: r["fitted"] for r in left["independent_seed_rescoring"]}
        right_scores = {r["seed"]: r["fitted"] for r in right["independent_seed_rescoring"]}
        if (set(left_scores) != set(right_scores) or len(left_scores) != len(left["independent_seed_rescoring"])
                or len(right_scores) != len(right["independent_seed_rescoring"])):
            raise ValueError("Recovery validation requires matching unique seeds")
        comparisons.append({"left": left["label"], "right": right["label"],
                            "left_minus_right": mean_statistics([left_scores[s] - right_scores[s] for s in sorted(left_scores)])})
    return comparisons


def build_report(runs, profiles, timings):
    manifests = [row["manifest"] for row in runs]
    if not runs or any(m["status"] != "complete" or r["recovery"]["status"] != "complete"
                       for m, r in zip(manifests, runs, strict=True)):
        raise ValueError("Recovery runs must be complete")
    first = manifests[0]
    for key in ("observations_sha256", "source_model_sha256", "truth", "bounds", "estimator", "noise"):
        if any(m[key] != first[key] for m in manifests):
            raise ValueError(f"Recovery runs differ: {key}")
    if first["synthetic_observation_model"]["kind"] != "binned_smoothed_uniform_contamination":
        raise ValueError("This report requires matched conditioned observations")
    modes = [p["config"]["mode_index"] for p in profiles]
    if sorted(modes) != [0, 1]:
        raise ValueError("Provide one profile for each LC mode")
    for p in profiles:
        for key in ("anchor", "validation_estimates", "validation_seeds"):
            if p["config"][key] != profiles[0]["config"][key]:
                raise ValueError(f"Profile comparison protocols differ: {key}")
        if p["source_sha256"] != profiles[0]["source_sha256"] or p["environment"] != profiles[0]["environment"]:
            raise ValueError("Profile sources or environments differ")
        if p["config"]["observations_sha256"] != first["observations_sha256"]:
            raise ValueError("Profiles and recovery runs use different observations")
        if p["parameter_names"] != list(first["truth"]) or p["bounds"] != first["bounds"]:
            raise ValueError("Profile parameter names or bounds differ")
        bases = {c["label"]: c["parameters"] for c in p["candidates"]}
        for i, run in enumerate(runs):
            if bases[f"fit_{i}"] != list(run["recovery"]["fitted"].values()):
                raise ValueError("Profile fitted parameters differ from recovery results")
        if bases["truth"] != list(first["truth"].values()):
            raise ValueError("Profile generating parameters differ")
    fits = []
    for i, run in enumerate(runs):
        recovery = run["recovery"]
        validation = recovery["independent_seed_rescoring"]
        history = run["history"]
        if len(history) != recovery["evaluations"]:
            raise ValueError("Fit history does not match the declared evaluation budget")
        scores = np.array([r["log_likelihood"] for r in history])
        checkpoints = np.unique(np.linspace(0, len(history) - 1, min(101, len(history))).astype(int))
        fits.append({"label": f"fit_{i}", "arguments": run["manifest"]["arguments"],
                     "fitted": recovery["fitted"], "initial": recovery["initial"],
                     "errors": recovery["errors"], "evaluations": recovery["evaluations"],
                     "fit_seconds": recovery["fit_seconds"], "total_seconds": recovery["total_seconds"],
                     "invalid_proposals": len(recovery["invalid_proposals"]),
                     "best_training_log_likelihood": recovery["best_training_log_likelihood"],
                     "last_quarter_improvement": float(scores.max() - scores[:max(1, 3 * len(scores) // 4)].max()),
                     "best_evaluation": int(scores.argmax()) + 1,
                     "training_curve": [{"evaluation": int(index) + 1, "best_score": float(scores[:index + 1].max())}
                                        for index in checkpoints],
                     "validation_estimates": recovery["validation_estimates"],
                     "independent_seed_rescoring": validation,
                     "fitted_minus_truth": mean_statistics([r["fitted"] - r["truth"] for r in validation])})
    return {"schema_version": 1, "status": "complete", "parameter_names": list(first["truth"]),
            "parameter_labels": LABELS, "truth": first["truth"], "bounds": first["bounds"],
            "observations_sha256": first["observations_sha256"], "noise": first["noise"],
            "observation_model": first["synthetic_observation_model"], "estimator": first["estimator"],
            "fits": fits, "pre_profile_fit_comparisons": recovery_comparisons(fits),
            "profiles": [profile_summary(p) for p in sorted(profiles, key=lambda p: p["config"]["mode_index"])],
            "timing": timings,
            "limitations": ["One synthetic subject and one generating parameter vector.",
                            "Matched observation generation does not test smoothing bias on raw physical or empirical RTs.",
                            "Fits differ in optimizer start, optimizer seed and fitting seed.",
                            "Profiles use finite-budget nuisance optimization; global maxima are not guaranteed.",
                            "Intervals describe Monte Carlo replicate means, not parameter confidence or subject variability.",
                            "Particle-budget drift does not test the observation law or model specification."]}


def plot_report(report, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 11), layout="constrained")
    names = report["parameter_names"]
    for column, profile in enumerate(report["profiles"]):
        ax = axes[0, column]
        for budget in profile["config"]["validation_estimates"]:
            rows = []
            for point in sorted(profile["points"], key=lambda p: p["fixed_value"]):
                label = f"mode{column}={point['fixed_value']:.6g}"
                score = next(r["difference_from_anchor"] for r in profile["summary"] if r["label"] == label and r["estimates"] == budget)
                rows.append((point["fixed_value"], score["mean"], *score["mean_ci95"]))
            x, mean, lo, hi = np.array(rows).T
            ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], marker="o", capsize=3,
                        label=f"{budget / 1000:g}k particles")
        ax.axhline(0., color=".3", ls="--", lw=1, label="Pre-profile fit anchor")
        ax.axvline(list(report["truth"].values())[4 + column], color=".5", ls=":", label="Generating mode")
        # Candidates found in the other search are feasible points, not maxima
        # for this coordinate. They expose nuisance-search limitations without
        # rerunning selection on validation scores or inventing an interpolation.
        other = report["profiles"][1 - column]
        cross = []
        for point in other["points"]:
            label = f"mode{1 - column}={point['fixed_value']:.6g}"
            score = next(r["difference_from_anchor"] for r in other["summary"]
                         if r["label"] == label and r["estimates"] == max(other["config"]["validation_estimates"]))
            cross.append((point["parameters"][4 + column], score["mean"], *score["mean_ci95"]))
        x, mean, lo, hi = np.array(cross).T
        ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt="D", ms=4, color=".35", mfc="none",
                    alpha=.7, capsize=2, label="Other search's candidates (largest budget)")
        ax.set(title=f"LC mode {column}: other seven parameters refitted", xlabel="Fixed LC mode",
               ylabel="Log score minus fit anchor (MC 95% CI)")
        ax.legend(fontsize=8)
        points = sorted(profile["points"], key=lambda p: p["fixed_value"])
        for index in (3, 5 - column, 6, 7):
            lo, hi, _ = report["bounds"][names[index]]
            axes[1, column].plot([p["fixed_value"] for p in points],
                                 [(p["parameters"][index] - lo) / (hi - lo) for p in points], "o-",
                                 label=LABELS[index])
        axes[1, column].set(xlabel=f"Fixed LC mode {column}", ylabel="Refitted value (fraction of allowed range)",
                            ylim=(-.03, 1.03), title="Nuisance parameter movement")
        axes[1, column].legend(fontsize=8)
    ax = axes[2, 0]
    truth = np.array(list(report["truth"].values()))
    widths = np.array([report["bounds"][name][1] - report["bounds"][name][0] for name in names])
    for fit in report["fits"]:
        ax.plot(np.arange(len(names)), (np.array(list(fit["fitted"].values())) - truth) / widths, "o-", label=fit["label"])
    ax.axhline(0., color=".5", lw=1)
    ax.set(xticks=np.arange(len(names)), xticklabels=LABELS, ylabel="Error / allowed parameter range", title="Two fits to the same synthetic subject")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend()
    ax = axes[2, 1]
    for fit in report["fits"]:
        ax.plot([r["evaluation"] for r in fit["training_curve"]], [r["best_score"] for r in fit["training_curve"]], label=fit["label"])
    best = max(fit["best_training_log_likelihood"] for fit in report["fits"])
    ax.set(xlabel="Candidate evaluations", ylabel="Best training log score", ylim=(best - 30, best + 2),
           title="Separate fixed-seed searches (final 30 log units)")
    ax.legend()
    fig.suptitle("DAWA: observation-conditioned recovery and LC-mode profiles\nFinite search budgets; error bars measure Monte Carlo uncertainty only")
    fig.savefig(prefix.with_suffix(".png"), dpi=160)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--profiles", type=Path, nargs=2, required=True)
    parser.add_argument("--reference-timing", type=Path, required=True)
    parser.add_argument("--h100-timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output prefix for JSON, PNG and PDF")
    args = parser.parse_args(argv)
    hashes = {}

    def read(path):
        payload = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(payload).hexdigest()
        return payload.decode()

    runs = [{"manifest": json.loads(read(p / "manifest.json")), "recovery": json.loads(read(p / "recovery.json")),
             "history": [json.loads(line) for line in read(p / "evaluations.jsonl").splitlines()]} for p in args.runs]
    profiles = [json.loads(read(p)) for p in args.profiles]
    timing = timing_comparison(json.loads(read(args.reference_timing)), json.loads(read(args.h100_timing)))
    report = build_report(runs, profiles, timing)
    report["input_sha256"] = hashes
    report["renderer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_report(report, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
