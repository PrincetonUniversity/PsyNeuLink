"""Render a DAWA particle-accuracy study as compact JSON and scientific figures.

Accepts complete or partial reports. Seed matrices retain the complete replicate
scores so the reported standard deviations and paired comparisons are auditable.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np

import dawa_conditioned_accuracy as accuracy_analysis


def runtime_summary(batches, estimates):
    """Exclude each session's first call at a budget, keeping the exclusions explicit."""
    first_calls = set()
    rows = []
    for batch in batches:
        key = (batch.get("session", 0), batch["estimates"])
        first = key not in first_calls
        first_calls.add(key)
        count = len(batch["labels"])
        if (
            not count
            or not np.isfinite(batch["seconds_including_diagnostics"])
            or batch["seconds_including_diagnostics"] < 0
        ):
            raise ValueError(
                "Runtime batches require candidates and finite nonnegative durations"
            )
        rows.append(
            {
                "batch_id": batch["id"],
                "estimates": batch["estimates"],
                "seed": batch["seed"],
                "labels": batch["labels"],
                "seconds": batch["seconds_including_diagnostics"],
                "seconds_per_candidate": batch["seconds_including_diagnostics"] / count,
                "excluded_first_call": first,
            }
        )
    summaries = []
    for budget in estimates:
        kept = [
            row
            for row in rows
            if row["estimates"] == budget and not row["excluded_first_call"]
        ]
        summary = {
            "estimates": budget,
            "included_batches": len(kept),
            "excluded_batches": [
                row["batch_id"]
                for row in rows
                if row["estimates"] == budget and row["excluded_first_call"]
            ],
        }
        if kept:
            values = np.array([row["seconds_per_candidate"] for row in kept])
            summary.update(
                median_seconds_per_candidate=float(np.median(values)),
                quartiles_seconds_per_candidate=np.quantile(
                    values, [0.25, 0.75]
                ).tolist(),
                minimum_seconds_per_candidate=float(values.min()),
                maximum_seconds_per_candidate=float(values.max()),
                aggregate_seconds_per_candidate=sum(row["seconds"] for row in kept)
                / sum(len(row["labels"]) for row in kept),
            )
        summaries.append(summary)
    return {
        "note": "Durations include filtering, diagnostics and host transfer. The first observed call at each "
        "particle budget in each session is excluded to reduce first-use compilation effects. "
        "Remaining times mix candidate chunks; they are not score-only timings or optimizer runtimes.",
        "budgets": summaries,
        "batches": rows,
    }


def _support_summaries(records, estimates, labels):
    summaries = []
    for budget in estimates:
        for label in labels:
            matching = [
                row
                for row in records
                if row["estimates"] == budget and row["label"] == label
            ]
            if not matching:
                continue
            groups = {}
            for group in ("all", "scored", "history_only"):
                rows = [row["diagnostic_groups"][group] for row in matching]
                counts = {row["trials"] for row in rows}
                if len(counts) != 1:
                    raise ValueError(
                        "Replicates disagree on the number of diagnostic trials"
                    )
                count = counts.pop()
                groups[group] = {"trials": count, "replicates": len(rows)}
                if not count:
                    continue
                ess_counts = np.array(
                    [row["trials_with_ess_below_100"] for row in rows]
                )
                contamination_counts = np.array(
                    [row["trials_with_contamination_above_half"] for row in rows]
                )
                groups[group].update(
                    minimum_ess_across_trials_and_seeds=min(
                        row["ess"]["minimum"] for row in rows
                    ),
                    median_of_trial_median_ess=float(
                        np.median([row["ess"]["median"] for row in rows])
                    ),
                    median_of_trial_median_ess_fraction=float(
                        np.median([row["ess_fraction"]["median"] for row in rows])
                    ),
                    maximum_contamination_responsibility=max(
                        row["contamination_responsibility"]["maximum"] for row in rows
                    ),
                    median_of_trial_median_contamination_responsibility=float(
                        np.median(
                            [
                                row["contamination_responsibility"]["median"]
                                for row in rows
                            ]
                        )
                    ),
                    trials_with_ess_below_100={
                        "median": float(np.median(ess_counts)),
                        "minimum": int(ess_counts.min()),
                        "maximum": int(ess_counts.max()),
                    },
                    trials_with_contamination_above_half={
                        "median": float(np.median(contamination_counts)),
                        "minimum": int(contamination_counts.min()),
                        "maximum": int(contamination_counts.max()),
                    },
                )
            summaries.append(
                {
                    "estimates": budget,
                    "label": label,
                    "groups": groups,
                    "maximum_abs_device_selected_score_difference": max(
                        abs(row["device_log_score"] - row["selected_log_score"])
                        for row in matching
                    ),
                }
            )
    return summaries


def trial_variability_summary(raw, labels=None):
    """Aggregate validated NPZ replicates without publishing observations or trial IDs."""
    config = raw["config"]
    labels = [config["candidates"]["reference"]] if labels is None else labels
    valid_labels = {row["label"] for row in config["candidates"]["candidates"]}
    if (
        not labels
        or len(set(labels)) != len(labels)
        or any(label not in valid_labels for label in labels)
    ):
        raise ValueError("Trial diagnostic labels must be unique declared candidates")
    seeds = config["seeds"]
    expected_seeds = set(seeds)
    records = {
        (row["estimates"], row["seed"], row["label"]): row for row in raw["records"]
    }
    included, skipped = [], []
    for budget in config["estimates"]:
        for label in labels:
            present = {
                seed for n, seed, name in records if n == budget and name == label
            }
            if present == expected_seeds and len(seeds) >= 2:
                included.append((budget, label))
            else:
                skipped.append(
                    {
                        "estimates": budget,
                        "label": label,
                        "reason": "incomplete_seed_set"
                        if present != expected_seeds
                        else "fewer_than_two_replicates",
                        "missing_seeds": sorted(expected_seeds - present),
                    }
                )
    included = set(included)
    collected, provenance = {}, []
    shared_mask = None
    fields = (
        "per_trial_densities",
        "effective_sample_size",
        "prior_mixture_fraction",
        "zero_support",
    )
    for batch in raw.get("batches", []):
        selected = [
            label
            for label in batch["labels"]
            if (batch["estimates"], label) in included
        ]
        if not selected:
            continue
        filename = batch.get("diagnostics_file")
        if not filename:
            raise ValueError(f"Batch {batch['id']} has no saved per-trial diagnostics")
        payload = Path(filename).read_bytes()
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            stored_labels = data["labels"].tolist()
            if stored_labels != batch["labels"] or len(set(stored_labels)) != len(
                stored_labels
            ):
                raise ValueError(
                    f"NPZ candidate labels do not match batch {batch['id']}"
                )
            raw_mask = np.asarray(data["include_mask"])
            if (
                raw_mask.shape != (config["retained_trials"],)
                or not np.isin(raw_mask, [False, True]).all()
            ):
                raise ValueError(f"Invalid observation mask in batch {batch['id']}")
            mask = raw_mask.astype(bool)
            if int(mask.sum()) != config["scored_trials"]:
                raise ValueError(f"Scored-trial count differs in batch {batch['id']}")
            if shared_mask is not None and not np.array_equal(mask, shared_mask):
                raise ValueError(
                    f"Observation masks differ between diagnostic batches: {batch['id']}"
                )
            shared_mask = mask
            arrays = {name: np.asarray(data[name]) for name in fields}
            shape = (len(stored_labels), 1, len(mask))
            if any(value.shape != shape for value in arrays.values()):
                raise ValueError(
                    f"NPZ diagnostic shapes do not match batch {batch['id']}"
                )
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise ValueError(f"Nonfinite NPZ diagnostics in batch {batch['id']}")
            if (
                (arrays["per_trial_densities"] <= 0).any()
                or (arrays["effective_sample_size"] <= 0).any()
                or (
                    arrays["effective_sample_size"]
                    > batch["estimates"] * (1.0 + 1.0e-6)
                ).any()
                or (arrays["prior_mixture_fraction"] < 0).any()
                or (arrays["prior_mixture_fraction"] > 1).any()
                or arrays["zero_support"].any()
            ):
                raise ValueError(
                    f"Unsupported NPZ diagnostic values in batch {batch['id']}"
                )
            for label in selected:
                key = (batch["estimates"], batch["seed"], label)
                if key in collected or key not in records:
                    raise ValueError(f"Duplicate or unrecorded NPZ replicate: {key}")
                index = stored_labels.index(label)
                logs = np.log(
                    arrays["per_trial_densities"][index, 0].astype(np.float64)
                )
                if not (
                    np.isclose(
                        logs[mask].sum(),
                        records[key]["selected_log_score"],
                        rtol=0.0,
                        atol=1.0e-8,
                    )
                    and np.isclose(
                        logs.sum(),
                        records[key]["full_log_score"],
                        rtol=0.0,
                        atol=1.0e-8,
                    )
                ):
                    raise ValueError(
                        f"NPZ densities do not reproduce the recorded scores: {key}"
                    )
                collected[key] = (
                    logs,
                    arrays["effective_sample_size"][index, 0].astype(float),
                    arrays["prior_mixture_fraction"][index, 0].astype(float),
                )
        provenance.append(
            {"batch_id": batch["id"], "sha256": hashlib.sha256(payload).hexdigest()}
        )
    summaries = []
    for budget in config["estimates"]:
        for label in labels:
            if (budget, label) not in included:
                continue
            keys = [(budget, seed, label) for seed in seeds]
            if any(key not in collected for key in keys):
                raise ValueError(
                    f"Saved NPZ files do not cover the complete seed set for {budget}, {label}"
                )
            logs, ess, prior = (
                np.stack([collected[key][index] for key in keys]) for index in range(3)
            )
            variance = logs.var(axis=0, ddof=1)
            median_ess, median_prior = np.median(ess, axis=0), np.median(prior, axis=0)
            groups = {}
            for name, mask in (("scored", shared_mask), ("history_only", ~shared_mask)):
                count = int(mask.sum())
                groups[name] = {"trials": count}
                if not count:
                    continue
                total_variance = float(logs[:, mask].sum(axis=1).var(ddof=1))
                trial_variances = variance[mask]
                sum_variances = float(trial_variances.sum())
                descending = np.sort(trial_variances)[::-1]
                groups[name].update(
                    median_per_trial_log_density_sd=float(
                        np.median(np.sqrt(trial_variances))
                    ),
                    sum_per_trial_variances=sum_variances,
                    total_score_variance=total_variance,
                    total_score_sd=float(np.sqrt(total_variance)),
                    covariance_contribution_to_score_variance=total_variance
                    - sum_variances,
                    top_trial_variance_share={
                        str(k): float(descending[:k].sum() / sum_variances)
                        if sum_variances > 0
                        else None
                        for k in (1, 5, 10)
                    },
                    trials_with_median_contamination_above_half=int(
                        (median_prior[mask] > 0.5).sum()
                    ),
                    trials_with_median_contamination_above_99pct=int(
                        (median_prior[mask] > 0.99).sum()
                    ),
                    trials_with_median_ess_below_100=int(
                        (median_ess[mask] < 100).sum()
                    ),
                    median_of_trial_median_ess=float(np.median(median_ess[mask])),
                    median_of_trial_median_contamination=float(
                        np.median(median_prior[mask])
                    ),
                )
            summaries.append(
                {
                    "estimates": budget,
                    "label": label,
                    "replicates": len(seeds),
                    "groups": groups,
                }
            )
    return {
        "labels": labels,
        "required_seeds": seeds,
        "summaries": summaries,
        "skipped": skipped,
        "diagnostic_file_sha256": provenance,
        "note": "Only complete configured seed sets are summarized. Variance shares describe sums of individual "
        "trial log-density variances, not shares of total-score variance: between-trial covariance can "
        "increase or decrease the latter. High ESS can also reflect almost uniform contamination ancestry, "
        "rather than good model support. No observed values, per-trial arrays or trial identifiers are retained.",
    }


def compact_report(
    raw, *, contrast_labels=None, trial_diagnostics=False, trial_labels=None
):
    """Recompute statistics from saved replicate scores and remove machine-local paths."""
    config = raw["config"]
    candidates = config["candidates"]
    labels = [row["label"] for row in candidates["candidates"]]
    reference = candidates["reference"]
    budgets = config["estimates"]
    if contrast_labels is None:
        contrast_labels = [
            label
            for label in labels
            if label != reference and "stress" not in label.lower()
        ]
        if not contrast_labels:
            contrast_labels = [label for label in labels if label != reference]
    if len(set(contrast_labels)) != len(contrast_labels) or any(
        label not in labels or label == reference for label in contrast_labels
    ):
        raise ValueError(
            "Contrast labels must be unique non-reference candidate labels"
        )
    records = raw.get("records", [])
    if not records:
        raise ValueError("The report contains no completed candidate records")
    expected_seeds = set(config["seeds"])
    for row in records:
        if (
            row["estimates"] not in budgets
            or row["label"] not in labels
            or row["seed"] not in expected_seeds
        ):
            raise ValueError("Record identifiers do not match the study configuration")
    analysis = accuracy_analysis.analyze_records(
        records, candidates, budgets, config.get("agreement_tolerance", 0.2)
    )
    analysis_source = Path(accuracy_analysis.__file__).resolve()
    lookup = {(row["estimates"], row["seed"], row["label"]): row for row in records}
    matrices = []
    for budget in budgets:
        seeds = [
            seed
            for seed in config["seeds"]
            if any((budget, seed, label) in lookup for label in labels)
        ]
        if not seeds:
            continue
        matrix = {"estimates": budget, "seeds": seeds, "candidate_labels": labels}
        for field in ("selected_log_score", "full_log_score", "device_log_score"):
            matrix[field] = [
                [
                    lookup[(budget, seed, label)][field]
                    if (budget, seed, label) in lookup
                    else None
                    for label in labels
                ]
                for seed in seeds
            ]
        matrices.append(matrix)
    expected = len(budgets) * len(config["seeds"]) * len(labels)
    report = {
        "schema_version": 1,
        "source_status": raw["status"],
        "completed_records": len(records),
        "expected_records": expected,
        "complete": len(records) == expected and raw["status"] == "complete",
        "config": {
            key: value
            for key, value in config.items()
            if key != "diagnostics_directory"
        },
        "environment": raw["environment"],
        "git_commit": raw["git_commit"],
        "implementation_sha256": raw["implementation_sha256"],
        "analysis_provenance": {
            "module": accuracy_analysis.__name__,
            "path": str(analysis_source),
            "sha256": hashlib.sha256(analysis_source.read_bytes()).hexdigest(),
        },
        "score_note": raw.get("score_note"),
        "seed_score_matrices": matrices,
        "analysis": analysis,
        "support_diagnostics": _support_summaries(records, budgets, labels),
        "runtime": runtime_summary(raw.get("batches", []), budgets),
        "figure": {
            "contrast_labels": contrast_labels,
            "reference_label": reference,
            "score_field": "selected_log_score",
            "uncertainty_note": "Panels A–B show single-evaluation Monte Carlo SD. Panel C shows "
            "95% Student-t confidence intervals for mean paired contrasts. "
            "Panel D shows timing medians and interquartile ranges.",
        },
        "interpretation": "The largest particle budget is a finite reference, not ground truth. "
        "A budget-drift mean is within the declared tolerance only when its entire 95% interval "
        "is inside that margin; an interval containing zero alone does not establish agreement. "
        "These checks do not establish parameter recovery or convergence of an optimizer.",
    }
    if trial_diagnostics:
        report["trial_diagnostics"] = trial_variability_summary(raw, trial_labels)
    elif trial_labels is not None:
        raise ValueError("Trial diagnostic labels require trial_diagnostics=True")
    return report


def _label(name):
    return (
        name.replace("_minus_001", " −0.001")
        .replace("_plus_001", " +0.001")
        .replace("_", " ")
    )


def render_figure(report, png, pdf):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    config = report["config"]
    labels = [row["label"] for row in config["candidates"]["candidates"]]
    contrasts = report["figure"]["contrast_labels"]
    budgets = sorted(config["estimates"])
    lookup = {
        (row["estimates"], row["label"]): row
        for row in report["analysis"]["candidates"]
    }
    colors = dict(
        zip(
            labels, plt.get_cmap("tab10").colors * (len(labels) // 10 + 1), strict=False
        )
    )
    with plt.rc_context(
        {"font.size": 10, "axes.titlesize": 11, "pdf.fonttype": 42, "ps.fonttype": 42}
    ):
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4))
        axes = axes.ravel()
        handles = []
        for label in labels:
            points = [
                (budget, lookup[(budget, label)]["selected_log_score"]["sd"])
                for budget in budgets
                if (budget, label) in lookup
                and lookup[(budget, label)]["selected_log_score"]["sd"] is not None
            ]
            (line,) = axes[0].plot(
                [p[0] for p in points],
                [p[1] for p in points],
                "o-",
                color=colors[label],
                markersize=4,
                label=_label(label),
            )
            handles.append(line)
        for index, label in enumerate(contrasts):
            points = []
            for budget in budgets:
                row = lookup.get((budget, label))
                comparison = (
                    None
                    if row is None
                    else row["difference_from_reference_candidate"][
                        "selected_log_score"
                    ]
                )
                if comparison is not None and comparison["sd"] is not None:
                    points.append((budget, comparison))
            axes[1].plot(
                [p[0] for p in points],
                [p[1]["sd"] for p in points],
                "o-",
                color=colors[label],
                markersize=4,
            )
            offset = np.exp((index - (len(contrasts) - 1) / 2) * 0.035)
            means = np.array([p[1]["mean"] for p in points])
            if points:
                intervals = np.array([p[1]["mean_ci95"] for p in points])
                axes[2].errorbar(
                    np.array([p[0] for p in points]) * offset,
                    means,
                    yerr=np.maximum(
                        0.0,
                        np.stack((means - intervals[:, 0], intervals[:, 1] - means)),
                    ),
                    fmt="o-",
                    color=colors[label],
                    markersize=4,
                    capsize=3,
                    linewidth=1.2,
                )
        timing = [
            row for row in report["runtime"]["budgets"] if row["included_batches"]
        ]
        if timing:
            middle = np.array([row["median_seconds_per_candidate"] for row in timing])
            interval = np.array(
                [row["quartiles_seconds_per_candidate"] for row in timing]
            )
            axes[3].errorbar(
                [row["estimates"] for row in timing],
                middle,
                yerr=np.stack((middle - interval[:, 0], interval[:, 1] - middle)),
                fmt="o-",
                color="#363636",
                capsize=4,
            )
        else:
            axes[3].text(
                0.5,
                0.5,
                "No timings remain after first-call exclusions",
                ha="center",
                transform=axes[3].transAxes,
            )
        axes[0].set_title("A  Single-evaluation score variability")
        axes[0].set_ylabel("SD of selected log score")
        axes[1].set_title("B  Single-evaluation comparison variability")
        axes[1].set_ylabel("SD of paired candidate − anchor score")
        axes[2].set_title("C  Mean paired contrasts and 95% intervals")
        axes[2].set_ylabel("Mean candidate − anchor log score")
        axes[2].axhline(0.0, color="#777777", linewidth=0.8, linestyle="--", zorder=0)
        axes[3].set_title("D  Runtime with diagnostics (median and IQR)")
        axes[3].set_ylabel("Seconds per candidate")
        for axis in axes:
            axis.set_xscale("log")
            axis.set_xticks(budgets)
            axis.xaxis.set_major_formatter(
                FuncFormatter(
                    lambda value, _: (
                        f"{value / 1e6:g}m" if value >= 1e6 else f"{value / 1000:g}k"
                    )
                )
            )
            axis.set_xlim(budgets[0] / 1.25, budgets[-1] * 1.25)
            axis.set_xlabel("Particles per filter")
            axis.grid(True, color="#d9d9d9", linewidth=0.6)
            axis.spines[["top", "right"]].set_visible(False)
        for axis in (axes[0], axes[1], axes[3]):
            axis.set_ylim(bottom=0.0)
        status = "Complete" if report["complete"] else "Partial"
        seeds = sorted(
            {
                row["selected_log_score"]["count"]
                for row in report["analysis"]["candidates"]
            }
        )
        seed_note = str(seeds[0]) if len(seeds) == 1 else f"{seeds[0]}–{seeds[-1]}"
        fig.suptitle(
            f"DAWA conditioned likelihood accuracy · subject {config['subject']} · {status.lower()} study\n"
            f"{report['environment']['gpu']} · {config['retained_trials']} retained / {config['scored_trials']} scored trials · "
            f"{seed_note} seeds per available candidate/budget",
            fontsize=13,
            y=0.99,
        )
        fig.legend(
            handles,
            [_label(label) for label in labels],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.06),
            ncol=min(4, len(labels)),
            frameon=False,
            fontsize=9,
        )
        fig.text(
            0.5,
            0.015,
            "A–B: SD of one complete filter evaluation. C: uncertainty of the mean, not single-evaluation variability.\n"
            "Largest budget is a finite reference. D includes diagnostics; the first call at each budget/session is excluded.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0.0, 0.13, 1.0, 0.93))
        fig.savefig(png, dpi=180)
        fig.savefig(pdf)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output prefix; creates .json, .png and .pdf",
    )
    parser.add_argument(
        "--contrast-labels",
        nargs="+",
        help="Non-reference labels for contrast panels (default omits labels containing stress)",
    )
    parser.add_argument(
        "--trial-diagnostics",
        action="store_true",
        help="Load saved NPZ files for aggregate per-trial replicate variability",
    )
    parser.add_argument(
        "--trial-labels",
        nargs="+",
        help="Candidates for trial diagnostics (default: reference candidate)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.trial_labels is not None and not args.trial_diagnostics:
        parser.error("--trial-labels requires --trial-diagnostics")
    paths = {
        kind: args.output.with_suffix(f".{kind}") for kind in ("json", "png", "pdf")
    }
    if args.input.resolve() in {path.resolve() for path in paths.values()}:
        parser.error("Output must not overwrite the input report")
    if not args.overwrite and any(path.exists() for path in paths.values()):
        parser.error("Output exists; choose a new prefix or explicitly use --overwrite")
    raw_bytes = args.input.read_bytes()
    report = compact_report(
        json.loads(raw_bytes),
        contrast_labels=args.contrast_labels,
        trial_diagnostics=args.trial_diagnostics,
        trial_labels=args.trial_labels,
    )
    report["raw_report_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    report["renderer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["rendered_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = {
        kind: path.with_name(path.stem + ".tmp" + path.suffix)
        for kind, path in paths.items()
    }
    try:
        render_figure(report, temporary["png"], temporary["pdf"])
        temporary["json"].write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        for kind, path in paths.items():
            temporary[kind].replace(path)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "complete": report["complete"],
                "records": report["completed_records"],
                "outputs": {key: str(path) for key, path in paths.items()},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
