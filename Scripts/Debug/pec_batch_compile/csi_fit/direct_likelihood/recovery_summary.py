"""Aggregation helpers for small direct-likelihood recovery matrices."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
import json

import numpy as np

from .model import parameter_bounds, parameter_names


def load_recovery_results(paths: Iterable[Path]) -> list[dict[str, object]]:
    """Load recovery JSON files and retain each source path."""
    results = []
    for path in paths:
        payload = json.loads(path.read_text())
        payload["source_path"] = str(path)
        results.append(payload)
    return results


def summarize_recovery_results(
    results: Iterable[dict[str, object]],
    *,
    bound_tolerance: float = 1.0e-5,
) -> dict[str, object]:
    """Compute run-level and parameter-level recovery diagnostics."""
    if bound_tolerance < 0.0 or bound_tolerance >= 0.5:
        raise ValueError("bound_tolerance must be in [0, 0.5).")
    names = tuple(parameter_names())
    lower, upper = parameter_bounds()
    scale = upper - lower
    run_rows = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)

    for result in results:
        result_names = tuple(result["parameter_names"])
        if result_names != names:
            raise ValueError("Recovery parameter names do not match the model.")
        truth = np.asarray(result["truth_parameter_vector"], dtype=float)
        recovered = np.asarray(result["recovered_parameter_vector"], dtype=float)
        if truth.shape != lower.shape or recovered.shape != lower.shape:
            raise ValueError("Recovery vectors do not match the model dimension.")
        error = recovered - truth
        scaled_error = error / scale
        scaled_location = (recovered - lower) / scale
        lower_hits = scaled_location <= bound_tolerance
        upper_hits = scaled_location >= 1.0 - bound_tolerance
        fit = result["fit"]
        label = str(result.get("truth_label", "unspecified"))
        run_row = {
            "truth_label": label,
            "simulation_seed": int(result["simulation_seed"]),
            "source_path": result.get("source_path"),
            "scaled_parameter_rmse": float(
                np.sqrt(np.mean(np.square(scaled_error)))
            ),
            "log_likelihood_gain_over_truth": float(
                result["recovered_log_likelihood"]
                - result["truth_log_likelihood"]
            ),
            "fit_success": bool(fit["success"]),
            "stationary": bool(fit["stationary"]),
            "coordinate_stationary": bool(fit["coordinate_stationary"]),
            "evaluations": int(fit["evaluations"]),
            "iterations": int(fit["iterations"]),
            "rejected_start_attempts": int(fit["rejected_start_attempts"]),
            "bound_hit_count": int(np.count_nonzero(lower_hits | upper_hits)),
            "lower_bound_parameters": [
                name for name, hit in zip(names, lower_hits, strict=True) if hit
            ],
            "upper_bound_parameters": [
                name for name, hit in zip(names, upper_hits, strict=True) if hit
            ],
            "parameter_error": error,
            "scaled_parameter_error": scaled_error,
        }
        run_rows.append(run_row)
        grouped[label].append(run_row)

    groups = []
    for label, rows in sorted(grouped.items()):
        errors = np.stack([np.asarray(row["parameter_error"]) for row in rows])
        scaled_errors = np.stack(
            [np.asarray(row["scaled_parameter_error"]) for row in rows]
        )
        run_rmse = np.asarray([row["scaled_parameter_rmse"] for row in rows])
        likelihood_gains = np.asarray(
            [row["log_likelihood_gain_over_truth"] for row in rows]
        )
        parameter_metrics = []
        for parameter_index, name in enumerate(names):
            parameter_error = errors[:, parameter_index]
            parameter_scaled_error = scaled_errors[:, parameter_index]
            parameter_metrics.append(
                {
                    "parameter": name,
                    "bias": float(np.mean(parameter_error)),
                    "rmse": float(
                        np.sqrt(np.mean(np.square(parameter_error)))
                    ),
                    "scaled_bias": float(np.mean(parameter_scaled_error)),
                    "scaled_rmse": float(
                        np.sqrt(np.mean(np.square(parameter_scaled_error)))
                    ),
                }
            )
        groups.append(
            {
                "truth_label": label,
                "recoveries": len(rows),
                "mean_scaled_parameter_rmse": float(np.mean(run_rmse)),
                "median_scaled_parameter_rmse": float(np.median(run_rmse)),
                "maximum_scaled_parameter_rmse": float(np.max(run_rmse)),
                "mean_log_likelihood_gain_over_truth": float(
                    np.mean(likelihood_gains)
                ),
                "fit_success_count": sum(row["fit_success"] for row in rows),
                "stationary_count": sum(row["stationary"] for row in rows),
                "coordinate_stationary_count": sum(
                    row["coordinate_stationary"] for row in rows
                ),
                "runs_with_bound_hits": sum(
                    row["bound_hit_count"] > 0 for row in rows
                ),
                "parameter_metrics": parameter_metrics,
            }
        )

    return {
        "parameter_names": names,
        "bound_tolerance_in_normalized_coordinates": bound_tolerance,
        "recoveries": run_rows,
        "groups": groups,
    }
