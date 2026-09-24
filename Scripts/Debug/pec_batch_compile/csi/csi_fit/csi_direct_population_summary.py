#!/usr/bin/env python3
"""Summarize an empirical all-subject CSI direct-likelihood fit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from direct_likelihood.model import CONDITIONS, parameter_bounds, parameter_names


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data fitting" / "data_to_fit_study3.csv"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _load_results(results_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    results = []
    errors = []
    for path in sorted(results_root.glob("subject-*/fit.json")):
        try:
            payload = json.loads(path.read_text())
            subject = int(payload["subject_nr"])
            vector = np.asarray(payload["parameter_vector"], dtype=float)
            log_likelihood = float(payload["log_likelihood"])
            if vector.shape != (len(parameter_names()),):
                raise ValueError(f"expected 13 parameters; found {vector.shape}")
            if not np.all(np.isfinite(vector)) or not math.isfinite(log_likelihood):
                raise ValueError("parameters and likelihood must be finite")
            directory_subject = int(path.parent.name.removeprefix("subject-"))
            if subject != directory_subject:
                raise ValueError(
                    f"payload subject {subject} != directory subject {directory_subject}"
                )
            payload["_path"] = path
            payload["_vector"] = vector
            results.append(payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{path}: {error}")
    return results, errors


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    data = pd.read_csv(args.data)
    expected_subjects = sorted(int(value) for value in data.subject_nr.unique())
    modeled_data = data[data["sequence"].isin(CONDITIONS)]
    modeled_rows = modeled_data.groupby("subject_nr").size().astype(int)
    included_rows = (
        modeled_data.groupby("subject_nr")["likelihood_include_mask"].sum().astype(int)
    )
    results, invalid_results = _load_results(args.results_root)
    completed_subjects = sorted(int(result["subject_nr"]) for result in results)
    duplicates = sorted(
        subject
        for subject in set(completed_subjects)
        if completed_subjects.count(subject) > 1
    )
    missing_subjects = sorted(set(expected_subjects) - set(completed_subjects))

    default_lower, default_upper = parameter_bounds()
    rows = []
    fresh_score_failures = []
    for result in sorted(results, key=lambda item: int(item["subject_nr"])):
        subject = int(result["subject_nr"])
        vector = result["_vector"]
        bounds_payload = result.get("parameter_bounds")
        if bounds_payload is None:
            lower, upper = default_lower, default_upper
        else:
            lower = np.asarray(bounds_payload["lower"], dtype=float)
            upper = np.asarray(bounds_payload["upper"], dtype=float)
            if lower.shape != vector.shape or upper.shape != vector.shape:
                raise ValueError(
                    f"subject {subject}: recorded parameter bounds must "
                    "contain 13 values"
                )
            if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
                raise ValueError(
                    f"subject {subject}: recorded parameter bounds are not finite"
                )
            if np.any(upper <= lower):
                raise ValueError(
                    f"subject {subject}: recorded parameter bounds are invalid"
                )
        scale = upper - lower
        scaled = (vector - lower) / scale
        boundary_indices = np.flatnonzero(
            (scaled <= args.boundary_tolerance)
            | (scaled >= 1.0 - args.boundary_tolerance)
        )
        start_rows = [
            row
            for row in result.get("run_results", ())
            if row.get("phase") == "start"
        ]
        best_initial = max(
            (float(row["initial_log_likelihood"]) for row in start_rows),
            default=math.nan,
        )
        fresh_score_path = result["_path"].with_name("fresh-score.json")
        fresh_log_likelihood = math.nan
        fresh_score_difference = math.nan
        fresh_score_valid = False
        if fresh_score_path.exists():
            try:
                fresh_score = json.loads(fresh_score_path.read_text())
                fresh_log_likelihood = float(fresh_score["log_likelihood"])
                fresh_score_difference = (
                    fresh_log_likelihood - float(result["log_likelihood"])
                )
                diagnostics = fresh_score["diagnostics"]
                fresh_score_valid = (
                    math.isfinite(fresh_log_likelihood)
                    and abs(fresh_score_difference) <= 1.0e-8
                    and not diagnostics["invalid_included_rows"]
                    and not diagnostics["zero_probability_included_rows"]
                )
                if not fresh_score_valid:
                    fresh_score_failures.append(
                        f"subject {subject}: fresh-score validation failed"
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                fresh_score_failures.append(
                    f"{fresh_score_path}: {error}"
                )
        row = {
            "subject_nr": subject,
            "modeled_rows": int(modeled_rows.loc[subject]),
            "included_rows": int(included_rows.loc[subject]),
            "log_likelihood": float(result["log_likelihood"]),
            "log_likelihood_per_row": (
                float(result["log_likelihood"]) / int(included_rows.loc[subject])
            ),
            "best_initial_log_likelihood": best_initial,
            "improvement_over_best_initial": (
                float(result["log_likelihood"]) - best_initial
            ),
            "fresh_log_likelihood": fresh_log_likelihood,
            "fresh_score_difference": fresh_score_difference,
            "fresh_score_valid": fresh_score_valid,
            "success": bool(result.get("success", False)),
            "stationary": bool(result.get("stationary", False)),
            "coordinate_stationary": bool(
                result.get("coordinate_stationary", False)
            ),
            "evaluations": int(result.get("evaluations", 0)),
            "iterations": int(result.get("iterations", 0)),
            "projected_gradient_inf_norm": float(
                result.get("projected_gradient_inf_norm", math.nan)
            ),
            "rejected_start_attempts": int(
                result.get("rejected_start_attempts", 0)
            ),
            "boundary_parameters": ";".join(
                parameter_names()[index] for index in boundary_indices
            ),
            "gain_upper_bound": float(upper[0]),
            "threshold_upper_bound": float(upper[4]),
            "non_decision_time_upper_bound": float(upper[10]),
            "gain_parameterization": result.get(
                "gain_parameterization", "linear"
            ),
            "result_path": str(result["_path"].resolve()),
        }
        row.update(dict(zip(parameter_names(), vector, strict=True)))
        rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.output_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "data": args.data.resolve(),
        "results_root": args.results_root.resolve(),
        "expected_subject_count": len(expected_subjects),
        "completed_subject_count": len(completed_subjects),
        "missing_subjects": missing_subjects,
        "duplicate_subjects": duplicates,
        "invalid_results": invalid_results,
        "fresh_score_count": sum(
            math.isfinite(row["fresh_log_likelihood"]) for row in rows
        ),
        "fresh_score_valid_count": sum(row["fresh_score_valid"] for row in rows),
        "fresh_score_failures": fresh_score_failures,
        "successful_optimizer_count": sum(row["success"] for row in rows),
        "stationary_count": sum(row["stationary"] for row in rows),
        "coordinate_stationary_count": sum(
            row["coordinate_stationary"] for row in rows
        ),
        "boundary_fit_count": sum(bool(row["boundary_parameters"]) for row in rows),
        "log_likelihood": {
            "minimum": min((row["log_likelihood"] for row in rows), default=None),
            "median": (
                float(np.median([row["log_likelihood"] for row in rows]))
                if rows
                else None
            ),
            "maximum": max((row["log_likelihood"] for row in rows), default=None),
        },
        "table": args.output_csv.resolve(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n"
    )
    print(json.dumps(summary, indent=2, default=_json_default))
    if args.require_complete and (
        missing_subjects
        or duplicates
        or invalid_results
        or fresh_score_failures
        or any(not row["fresh_score_valid"] for row in rows)
    ):
        raise SystemExit(1)
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--boundary-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--require-complete", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.boundary_tolerance < 0.0 or args.boundary_tolerance >= 0.5:
        raise ValueError("boundary-tolerance must lie in [0, 0.5).")
    summarize(args)


if __name__ == "__main__":
    main()
