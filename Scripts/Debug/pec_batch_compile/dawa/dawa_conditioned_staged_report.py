"""Summarize staged-search calibration and a matched fixed/staged fitting pilot.

Ranking losses refer to the mean largest-count score of a fixed candidate set,
not an exact likelihood or an unknown optimum. Complete masked log scores are
averaged across filter seeds; individual trial densities are never pooled.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def ranking_summary(report):
    if report["status"] != "complete":
        raise ValueError("Ranking calibration must be complete")
    config = report["config"]
    labels = [row["label"] for row in config["candidates"]["candidates"]]
    counts, seeds = config["estimates"], config["seeds"]
    lookup = {(row["estimates"], row["seed"], row["label"]): row["device_log_score"] for row in report["records"]}
    if len(lookup) != len(report["records"]) or len(lookup) != len(counts) * len(seeds) * len(labels):
        raise ValueError("Calibration records must contain exactly one complete count/seed/candidate grid")
    scores = np.array([[[lookup[count, seed, label] for label in labels] for seed in seeds] for count in counts])
    if not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be finite")
    reference = scores[counts.index(max(counts))].mean(axis=0)
    groups = {"all": list(range(len(labels)))}
    near = [i for i, label in enumerate(labels) if "near_rank" in label]
    if near:
        groups["near_optimum_saved_candidates"] = near
    results = {}
    for name, indices in groups.items():
        target = reference[indices]
        rows = []
        for count, values in zip(counts, scores, strict=True):
            values = values[:, indices]
            selected = values.argmax(axis=1)
            correlations = [float(spearmanr(row, target).statistic) for row in values]
            losses = target.max() - target[selected]
            rows.append({
                "estimates": count, "mean_spearman": float(np.mean(correlations)),
                "winner_reference_loss_per_seed": losses.tolist(), "mean_winner_reference_loss": float(losses.mean()),
                "max_winner_reference_loss": float(losses.max()),
                "winner_labels": [labels[indices[i]] for i in selected],
                "mean_score_shift_from_reference": float((values - target).mean()),
            })
        results[name] = {"labels": [labels[i] for i in indices], "budgets": rows}
    return {
        "config": config, "environment": report["environment"], "git_commit": report["git_commit"],
        "implementation_sha256": report["implementation_sha256"],
        "labels": labels, "complete_log_scores": scores.tolist(), "reference_mean_log_scores": reference.tolist(),
        "groups": results,
        "note": "Reference is the largest-count mean on this candidate set, not ground truth. Near-optimum "
                "membership was fixed by saved-fit labels before this calibration. Rank correlations and selection "
                "losses are descriptive; five repeats do not establish a universal safe particle count.",
    }


def fit_comparison(fixed, staged, fixed_manifest, staged_manifest):
    for report in (fixed, staged):
        if report["status"] != "complete":
            raise ValueError("Both fits must be complete")
    for key in ("observations_sha256", "initial", "bounds", "source_model_sha256", "noise", "time_steps", "estimator"):
        if fixed_manifest[key] != staged_manifest[key]:
            raise ValueError(f"Fit comparison requires matching {key}")
    if fixed["validation_estimates"] != staged["validation_estimates"] or fixed["validation_pseudocount"] != staged["validation_pseudocount"]:
        raise ValueError("Validation observation law and particle counts must match")
    a = {row["seed"]: row["fitted"] for row in fixed["independent_seed_rescoring"]}
    b = {row["seed"]: row["fitted"] for row in staged["independent_seed_rescoring"]}
    if set(a) != set(b) or len(a) < 2:
        raise ValueError("Fit comparisons require at least two matching validation seeds")
    # Keep the original pilot readable after unifying the public fit strategy.
    policy_key = "adaptive" if "adaptive" in staged else "staged"
    policy = staged[policy_key]
    if policy_key == "adaptive" and policy.get("policy") != "staged":
        raise ValueError("This comparison requires the conditioned staged policy")
    driver_hash_key = "adaptive_driver_sha256" if policy_key == "adaptive" else "staged_driver_sha256"
    differences = np.array([b[seed] - a[seed] for seed in sorted(a)])
    return {
        "observations_sha256": fixed_manifest["observations_sha256"], "trials": fixed_manifest["trials"],
        "scored_trials": fixed_manifest["scored_trials"], "initial": fixed_manifest["initial"],
        "validation_estimates": fixed["validation_estimates"], "validation_seeds": sorted(a),
        "staged_minus_fixed_validation": {"per_seed": differences.tolist(), "mean": float(differences.mean()),
                                          "mc_standard_error": float(differences.std(ddof=1) / np.sqrt(len(differences)))},
        "fit_seconds_ratio_fixed_over_staged": fixed["fit_seconds"] / staged["fit_seconds"],
        "fixed": {key: fixed[key] for key in ("fitted", "fit_seconds", "total_seconds", "evaluations", "independent_seed_rescoring")},
        "staged": {**{key: staged[key] for key in ("fitted", "fit_seconds", "total_seconds", "evaluations", "independent_seed_rescoring")},
                   "staged": policy},
        "fixed_provenance": {key: fixed_manifest[key] for key in ("git_commit", "driver_sha256", "gpu", "arguments")},
        "staged_provenance": {**{key: staged_manifest[key] for key in ("git_commit", "driver_sha256", "gpu", "arguments")},
                              "staged_driver_sha256": staged_manifest[driver_hash_key]},
        "note": "One subject and start. Timing compares completed fits on the recorded compiler revisions; a historical "
                "fixed fit also predates compiler optimizations, so the ratio is not a pure staged-policy speedup. "
                "Validation averages complete log scores. Monte Carlo SE is not uncertainty across subjects or fits.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True, help="Completed recovery directory")
    parser.add_argument("--staged", type=Path, required=True, help="Completed recovery directory")
    parser.add_argument("--replay", type=Path, help="Optional matched fixed-fit population replay timings")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.accuracy, *(directory / name for directory in (args.fixed, args.staged)
                             for name in ("recovery.json", "manifest.json"))]
    values = [json.loads(path.read_text()) for path in paths]
    report = {"schema_version": 1, "calibration": ranking_summary(values[0]),
              "pilot": fit_comparison(values[1], values[3], values[2], values[4]),
              "inputs_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
              "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.replay is not None:
        replay = json.loads(args.replay.read_text())
        if len(replay.get("batches", [])) != 10 or not all(row["scores_exact"] for row in replay["batches"]):
            raise ValueError("Expected ten completed fixed-fit populations with exact score replay")
        # The first historical population includes its first-use compilation.
        # Current calls were explicitly warmed. Exclude that historical startup
        # from the warm comparison, while retaining every raw timing and score.
        warm = [row for row in replay["batches"] if row["first_evaluation"] != 2]
        baseline_mean = float(np.mean([row["baseline_seconds"] for row in warm]))
        current_mean = float(np.mean([row["current_seconds"] for row in warm]))
        replay["warm_comparison"] = {
            "populations": len(warm), "baseline_mean_seconds": baseline_mean,
            "current_mean_seconds": current_mean, "speedup": baseline_mean / current_mean,
            "note": "Excludes evaluation 2, the first historical population, whose timing includes compilation. "
                    "All current calls were explicitly warmed. Raw all-population means include this startup asymmetry "
                    "and must not be interpreted as a warm compiler speedup.",
        }
        report["fixed_population_replay"] = replay
        report["inputs_sha256"][str(args.replay)] = hashlib.sha256(args.replay.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
