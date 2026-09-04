#!/usr/bin/env python3
"""CLI for the continuous-time CSI direct-likelihood prototype."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from direct_likelihood import (
    CONDITIONS,
    CSITrialData,
    ContinuousCSILikelihood,
    ContinuousCSIParameters,
    SolverConfig,
    generator_validation_report,
    simulate_sequential_trials,
)
from direct_likelihood.fit import (
    fit_cmaes,
    fit_lbfgsb,
    fit_staged,
    parallel_finite_difference_objective,
)
from direct_likelihood.model import parameter_bounds, parameter_names
from direct_likelihood.recovery_summary import (
    load_recovery_results,
    summarize_recovery_results,
)
from direct_likelihood.validation import validation_report


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data fitting" / "data_to_fit_study3.csv"


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON")


def _write_json(payload: dict[str, object], output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, default=_json_value)
    if output is None:
        print(rendered)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
        print(output)


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--subject",
        type=int,
        required=True,
        help="Actual subject_nr value (not the legacy script's one-based array index).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--float32", action="store_true")
    parser.add_argument("--ddm-time-step", type=float, default=0.001)
    parser.add_argument("--ddm-spatial-points", type=int, default=65)
    parser.add_argument("--lca-max-step", type=float, default=0.01)
    parser.add_argument(
        "--lca-integration-method",
        choices=("rk4", "euler"),
        default="rk4",
        help="LCA integrator; Euler is intended for legacy-semantic diagnostics.",
    )
    parser.add_argument("--ddm-checkpoint-steps", type=int, default=32)
    parser.add_argument(
        "--ddm-bucket-size",
        type=int,
        default=256,
        help="Duration-sorted DDM batch size; zero puts all trials in one batch.",
    )
    parser.add_argument(
        "--compile-ddm-steps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Compile the reusable PDE step (enabled by default for fit; "
            "compilation adds startup time)."
        ),
    )
    parser.add_argument(
        "--custom-ddm-adjoint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use the compiled whole-PDE reverse pass (enabled by default for "
            "autograd fitting)."
        ),
    )
    parser.add_argument(
        "--custom-lca-adjoint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use compact LCA integration/drift adjoints (enabled by default "
            "for autograd fitting)."
        ),
    )
    parser.add_argument(
        "--native-lca-scan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use the fused native CPU subject scan (enabled by default on "
            "the CPU; its extension is compiled once on first use)."
        ),
    )
    parser.add_argument(
        "--native-ddm-forward",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use the fused native CPU PDE forward solve (enabled by default "
            "when the native build toolchain is available)."
        ),
    )
    parser.add_argument("--no-lca-checkpoint", action="store_true")


def _load_problem(args):
    dtype = torch.float32 if args.float32 else torch.float64
    trials = CSITrialData.from_csv(
        args.data, args.subject, dtype=dtype, device=args.device
    )
    exact_gradient = (
        (
            args.command in {"fit", "recover", "staged-fit"}
            and getattr(args, "gradient_method", None) == "autograd"
        )
        or (
            args.command == "benchmark"
            and getattr(args, "with_gradient", False)
        )
    )
    finite_difference_fit = (
        args.command == "fit"
        and getattr(args, "gradient_method", None)
        == "parallel-finite-difference"
    )
    custom_ddm_adjoint = (
        exact_gradient
        if args.custom_ddm_adjoint is None
        else args.custom_ddm_adjoint
    )
    custom_lca_adjoint = (
        exact_gradient
        if args.custom_lca_adjoint is None
        else args.custom_lca_adjoint
    )
    compile_ddm_steps = (
        finite_difference_fit or custom_ddm_adjoint
        if args.compile_ddm_steps is None
        else args.compile_ddm_steps
    )
    if args.native_lca_scan is None:
        from direct_likelihood.native import native_kernels_available

        native_lca_scan = (
            torch.device(args.device).type == "cpu" and native_kernels_available()
        )
    else:
        native_lca_scan = args.native_lca_scan
    native_ddm_forward = (
        native_lca_scan
        if args.native_ddm_forward is None
        else args.native_ddm_forward
    )
    likelihood = ContinuousCSILikelihood(
        SolverConfig(
            ddm_time_step=args.ddm_time_step,
            ddm_spatial_points=args.ddm_spatial_points,
            lca_max_step=args.lca_max_step,
            lca_integration_method=args.lca_integration_method,
            ddm_checkpoint_steps=args.ddm_checkpoint_steps,
            checkpoint_lca=not args.no_lca_checkpoint,
            ddm_bucket_size=args.ddm_bucket_size,
            compile_ddm_steps=compile_ddm_steps,
            custom_ddm_adjoint=custom_ddm_adjoint,
            custom_lca_adjoint=custom_lca_adjoint,
            compile_lca_adjoint=custom_lca_adjoint,
            native_lca_scan=native_lca_scan,
            native_ddm_forward=native_ddm_forward,
        )
    )
    return trials, likelihood, dtype


def _load_parameters(path: Path | None, *, dtype, device):
    if path is None:
        return ContinuousCSIParameters.defaults(dtype=dtype, device=device)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if len(frame) != 1:
            raise ValueError(
                f"Parameter CSV must contain exactly one row; got {len(frame)}."
            )
        parameters = ContinuousCSIParameters.from_legacy_row(frame.iloc[0])
        return ContinuousCSIParameters.from_vector(
            parameters.vector().to(dtype=dtype, device=device)
        )
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        if "parameter_vector" in payload:
            payload = payload["parameter_vector"]
        elif "recovered_parameter_vector" in payload:
            payload = payload["recovered_parameter_vector"]
        else:
            payload = [payload[name] for name in parameter_names()]
    vector = torch.as_tensor(payload, dtype=dtype, device=device)
    return ContinuousCSIParameters.from_vector(vector)


def _score(args) -> None:
    trials, likelihood, dtype = _load_problem(args)
    parameters = _load_parameters(args.parameters, dtype=dtype, device=args.device)
    with torch.no_grad():
        result = likelihood.score(
            parameters, trials, collect_timings=args.profile
        )
    payload = {
        "subject_nr": trials.subject_nr,
        "rows": len(trials),
        "included_rows": int(trials.include.sum()),
        "log_likelihood": result.log_likelihood,
        "parameter_names": parameter_names(),
        "parameter_vector": parameters.vector(),
        "legacy_parameterization": parameters.as_legacy_dict(),
        "diagnostics": result.diagnostics,
        "timings": result.timings,
    }
    _write_json(payload, args.output)
    if args.trial_output is not None:
        args.trial_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "row_id": trials.row_id.detach().cpu().numpy(),
                "include": trials.include.detach().cpu().numpy(),
                "decision_time": result.decision_time.detach().cpu().numpy(),
                "likelihood": result.probability.detach().cpu().numpy(),
                "log_likelihood": result.per_trial_log_likelihood.detach().cpu().numpy(),
            }
        ).to_csv(args.trial_output, index=False)


def _export_pnl_parameters(args) -> None:
    parameters = _load_parameters(
        args.parameters, dtype=torch.float64, device="cpu"
    )
    row = parameters.as_legacy_dict()
    continuous_csi_count = row["Cue Stimulus Interval.slope"]
    rounded_csi_count = int(np.rint(continuous_csi_count))
    row["Cue Stimulus Interval.slope"] = rounded_csi_count
    row["subject_nr"] = args.subject
    row["source_parameter_file"] = str(args.parameters.resolve())
    row["direct_continuous_csi_count"] = continuous_csi_count
    row["pnl_rounded_csi_count"] = rounded_csi_count
    row["csi_rounding_error_seconds"] = (
        rounded_csi_count - continuous_csi_count
    ) * 0.01
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(args.output, index=False)
    print(args.output)


def _pnl_rescore_summary(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_csv(path)
    required = {"validation_simulation_seed", "validation_log_likelihood"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"PNL rescore CSV is missing columns: {sorted(missing)}")
    values = frame["validation_log_likelihood"].to_numpy(dtype=float)
    summary = {
        "source_path": str(path),
        "rows": len(frame),
        "simulation_seeds": frame["validation_simulation_seed"].tolist(),
        "mean": float(np.mean(values)),
        "standard_deviation": (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }
    for column in ("validation_num_estimates", "backend", "bins"):
        if column in frame and frame[column].nunique(dropna=False) == 1:
            summary[column] = frame[column].iloc[0]
    return frame, summary


def _compare_pnl(args) -> None:
    trials, likelihood, dtype = _load_problem(args)
    direct_parameters = _load_parameters(
        args.direct_parameters, dtype=dtype, device=args.device
    )
    pnl_parameters = _load_parameters(
        args.pnl_parameters, dtype=dtype, device=args.device
    )
    with torch.no_grad():
        direct_at_direct = likelihood.score(direct_parameters, trials)
        direct_at_pnl = likelihood.score(pnl_parameters, trials)

    direct_rescore, direct_rescore_summary = _pnl_rescore_summary(
        args.direct_pnl_rescore
    )
    pnl_rescore, pnl_rescore_summary = _pnl_rescore_summary(
        args.pnl_pnl_rescore
    )
    paired = direct_rescore[
        ["validation_simulation_seed", "validation_log_likelihood"]
    ].merge(
        pnl_rescore[
            ["validation_simulation_seed", "validation_log_likelihood"]
        ],
        on="validation_simulation_seed",
        suffixes=("_direct_vector", "_pnl_vector"),
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError("PNL rescore files have no simulation seeds in common.")
    paired["direct_minus_pnl"] = (
        paired["validation_log_likelihood_direct_vector"]
        - paired["validation_log_likelihood_pnl_vector"]
    )
    lower, upper = parameter_bounds()
    vector_difference = (
        direct_parameters.vector().detach().cpu().numpy()
        - pnl_parameters.vector().detach().cpu().numpy()
    )
    _write_json(
        {
            "subject_nr": trials.subject_nr,
            "direct_parameter_source": str(args.direct_parameters),
            "pnl_parameter_source": str(args.pnl_parameters),
            "comparison_note": (
                "Compare vector rankings within each objective; direct and PNL "
                "likelihood values are not on a common numerical scale."
            ),
            "parameter_names": parameter_names(),
            "direct_parameter_vector": direct_parameters.vector(),
            "pnl_parameter_vector": pnl_parameters.vector(),
            "normalized_parameter_difference": vector_difference / (upper - lower),
            "normalized_parameter_difference_rmse": float(
                np.sqrt(np.mean(np.square(vector_difference / (upper - lower))))
            ),
            "direct_likelihood": {
                "at_direct_vector": direct_at_direct.log_likelihood,
                "at_pnl_vector": direct_at_pnl.log_likelihood,
                "direct_minus_pnl": (
                    direct_at_direct.log_likelihood - direct_at_pnl.log_likelihood
                ),
                "preferred_vector": (
                    "direct" if direct_at_direct.log_likelihood >= direct_at_pnl.log_likelihood
                    else "pnl"
                ),
            },
            "pnl_simulator_likelihood": {
                "direct_vector": direct_rescore_summary,
                "pnl_vector": pnl_rescore_summary,
                "paired_seeds": paired.to_dict(orient="records"),
                "mean_paired_direct_minus_pnl": float(
                    paired["direct_minus_pnl"].mean()
                ),
                "paired_difference_standard_deviation": (
                    float(paired["direct_minus_pnl"].std(ddof=1))
                    if len(paired) > 1
                    else 0.0
                ),
                "preferred_vector": (
                    "direct"
                    if paired["direct_minus_pnl"].mean() >= 0.0
                    else "pnl"
                ),
            },
        },
        args.output,
    )


def _semantic_ladder(args) -> None:
    """Factor the direct/PNL ranking gap into controlled semantic changes."""
    trials, base_likelihood, dtype = _load_problem(args)
    direct_exact = _load_parameters(
        args.direct_parameters, dtype=dtype, device=args.device
    )
    pnl_parameters = _load_parameters(
        args.pnl_parameters, dtype=dtype, device=args.device
    )
    rounded_vector = direct_exact.vector().clone()
    rounded_vector[3] = (
        torch.round(rounded_vector[3] / args.legacy_time_step)
        * args.legacy_time_step
    )
    direct_rounded = ContinuousCSIParameters.from_vector(rounded_vector)

    fine_rk4 = replace(
        base_likelihood.config,
        lca_integration_method="rk4",
    )
    fine_binned = replace(
        fine_rk4,
        rt_bin_count=args.legacy_bins,
    )
    fine_reset = replace(
        fine_binned,
        reset_lca_each_trial=True,
        native_lca_scan=False,
    )
    fine_binned_euler = replace(
        fine_binned,
        lca_integration_method="euler",
        native_lca_scan=False,
    )
    legacy_rk4 = replace(
        fine_binned,
        ddm_time_step=args.legacy_time_step,
        ddm_spatial_points=args.legacy_spatial_points,
        lca_max_step=args.legacy_time_step,
        # Crank--Nicolson can ring and produce a zero first-passage flux on a
        # time grid this coarse.  Fully implicit stepping keeps this diagnostic
        # finite and isolates resolution rather than an avoidable PDE artifact.
        rannacher_steps=args.legacy_rannacher_steps,
    )
    legacy_euler = replace(
        legacy_rk4,
        lca_integration_method="euler",
        native_lca_scan=False,
    )
    endpoint_euler = replace(
        legacy_euler,
        ddm_process="endpoint",
        ddm_spatial_points=args.endpoint_spatial_points,
        endpoint_evidence_domain=args.endpoint_evidence_domain,
        native_ddm_forward=False,
    )
    specifications = (
        (
            "continuous_exact_csi_rk4",
            fine_rk4,
            direct_exact,
            "continuous CSI and the reference RK4/PDE mesh",
        ),
        (
            "continuous_rounded_csi_rk4",
            fine_rk4,
            direct_rounded,
            "only the direct vector's CSI is rounded to the legacy clock",
        ),
        (
            "fine_mesh_rounded_csi_fixed_rt_bins",
            fine_binned,
            direct_rounded,
            "fixed empirical-range RT bins matching the PNL histogram estimator",
        ),
        (
            "fine_mesh_rounded_csi_fixed_rt_bins_reset_history",
            fine_reset,
            direct_rounded,
            "diagnostic ablation: reset LCA at each trial; fixed RT bins",
        ),
        (
            "fine_mesh_rounded_csi_fixed_rt_bins_euler",
            fine_binned_euler,
            direct_rounded,
            "Euler LCA on the reference mesh; fixed RT bins",
        ),
        (
            "10ms_mesh_rounded_csi_rk4",
            legacy_rk4,
            direct_rounded,
            "10 ms fully implicit DDM/LCA mesh with RK4; CSI remains rounded",
        ),
        (
            "10ms_mesh_rounded_csi_euler",
            legacy_euler,
            direct_rounded,
            "closest stable direct analogue of legacy timing and LCA updates",
        ),
        (
            "10ms_endpoint_ddm_rounded_csi_euler",
            endpoint_euler,
            direct_rounded,
            "deterministic legacy Euler random walk with endpoint-only crossing",
        ),
    )

    levels = []
    with torch.no_grad():
        for name, config, direct_parameters, description in specifications:
            likelihood = ContinuousCSILikelihood(config)
            direct_result = likelihood.score(
                direct_parameters, trials, collect_timings=True
            )
            pnl_result = likelihood.score(
                pnl_parameters, trials, collect_timings=True
            )
            direct_value = float(direct_result.log_likelihood)
            pnl_value = float(pnl_result.log_likelihood)
            rt_bin_width = float(
                direct_result.diagnostics["rt_bin_width"]
            )
            included_count = int(trials.include.sum())
            log_density_normalization = included_count * math.log(rt_bin_width)
            levels.append(
                {
                    "name": name,
                    "description": description,
                    "ddm_time_step": config.ddm_time_step,
                    "ddm_spatial_points": config.ddm_spatial_points,
                    "ddm_process": config.ddm_process,
                    "lca_max_step": config.lca_max_step,
                    "lca_integration_method": config.lca_integration_method,
                    "reset_lca_each_trial": config.reset_lca_each_trial,
                    "rt_bin_count": config.rt_bin_count,
                    "direct_vector_csi_duration": float(
                        direct_parameters.csi_duration
                    ),
                    "pnl_vector_csi_duration": float(
                        pnl_parameters.csi_duration
                    ),
                    "direct_vector_log_likelihood": direct_value,
                    "pnl_vector_log_likelihood": pnl_value,
                    "direct_vector_log_density": (
                        direct_value - log_density_normalization
                    ),
                    "pnl_vector_log_density": (
                        pnl_value - log_density_normalization
                    ),
                    "direct_minus_pnl": direct_value - pnl_value,
                    "rt_bin_width": rt_bin_width,
                    "preferred_vector": (
                        "direct" if direct_value >= pnl_value else "pnl"
                    ),
                    "direct_vector_maximum_mass_error": float(
                        direct_result.diagnostics["maximum_mass_error"]
                    ),
                    "pnl_vector_maximum_mass_error": float(
                        pnl_result.diagnostics["maximum_mass_error"]
                    ),
                    "elapsed_seconds": (
                        direct_result.timings["total_seconds"]
                        + pnl_result.timings["total_seconds"]
                    ),
                }
            )

    def contrast(name: str, changed: int, baseline: int) -> dict[str, object]:
        changed_level = levels[changed]
        baseline_level = levels[baseline]
        return {
            "name": name,
            "changed_level": changed_level["name"],
            "baseline_level": baseline_level["name"],
            "direct_vector_log_likelihood_change": (
                changed_level["direct_vector_log_likelihood"]
                - baseline_level["direct_vector_log_likelihood"]
            ),
            "pnl_vector_log_likelihood_change": (
                changed_level["pnl_vector_log_likelihood"]
                - baseline_level["pnl_vector_log_likelihood"]
            ),
            "ranking_gap_change": (
                changed_level["direct_minus_pnl"]
                - baseline_level["direct_minus_pnl"]
            ),
        }

    _write_json(
        {
            "subject_nr": trials.subject_nr,
            "direct_parameter_source": str(args.direct_parameters),
            "pnl_parameter_source": str(args.pnl_parameters),
            "purpose": (
                "Within-objective ranking diagnostic. Absolute direct and PNL "
                "simulator likelihood scales remain incomparable."
            ),
            "statistical_semantics": {
                "direct": (
                    "Sequential conditional likelihood: persistent LCA state "
                    "is propagated through the participant's observed prior RTs."
                ),
                "pnl_simulator": (
                    "Trialwise histogram composite likelihood: each Monte Carlo "
                    "lane simulates the full sequence, so a trial's histogram "
                    "averages over simulated prior RT histories."
                ),
                "consequence": (
                    "The numerical ladder cannot remove this conditional-versus-"
                    "marginal-history distinction."
                ),
            },
            "levels": levels,
            "contrasts": [
                contrast("CSI rounding", 1, 0),
                contrast("fixed histogram RT bins", 2, 1),
                contrast("reset-history ablation", 3, 2),
                contrast("Euler versus RK4 on reference mesh", 4, 2),
                contrast("10 ms versus reference mesh under RK4", 5, 2),
                contrast("Euler versus RK4 on 10 ms mesh", 6, 5),
                contrast("endpoint DDM versus continuous 10 ms PDE", 7, 6),
                contrast("closest legacy numerical semantics versus reference", 7, 0),
            ],
        },
        args.output,
    )


def _fit(args) -> None:
    trials, likelihood, dtype = _load_problem(args)
    initial_vectors = [
        _load_parameters(path, dtype=dtype, device=args.device)
        .vector()
        .detach()
        .cpu()
        .numpy()
        for path in args.initial_parameters
    ]
    results = []
    if args.optimizer in {"lbfgsb", "both"}:
        results.append(
            fit_lbfgsb(
                likelihood,
                trials,
                starts=args.starts,
                max_iterations=args.max_iterations,
                seed=args.seed,
                initial_vectors=initial_vectors,
                gradient_method=args.gradient_method,
                finite_difference_workers=args.finite_difference_workers,
                finite_difference_relative_step=(
                    args.finite_difference_relative_step
                ),
                include_default_start=not args.no_default_start,
                max_start_attempts=args.max_start_attempts,
                random_start_candidates=args.random_start_candidates,
                function_tolerance=args.optimizer_ftol,
                gradient_tolerance=args.optimizer_gtol,
                maximum_line_search_steps=args.max_line_search_steps,
                polish_restarts=args.polish_restarts,
                polish_iterations=args.polish_iterations,
                polish_method=args.polish_method,
                coordinate_polish=not args.no_coordinate_polish,
                coordinate_step=args.coordinate_step,
                coordinate_levels=args.coordinate_levels,
                coordinate_rounds=args.coordinate_rounds,
                coordinate_cycles=args.coordinate_cycles,
                coordinate_improvement_tolerance=(
                    args.coordinate_improvement_tolerance
                ),
                stationarity_tolerance=args.stationarity_tolerance,
            )
        )
    if args.optimizer in {"cmaes", "both"}:
        results.append(
            fit_cmaes(
                likelihood,
                trials,
                evaluations=args.cma_evaluations,
                seed=args.seed,
                initial_vectors=initial_vectors,
            )
        )
    best = max(results, key=lambda result: result.log_likelihood)
    payload = {
        "subject_nr": trials.subject_nr,
        "best_method": best.method,
        "log_likelihood": best.log_likelihood,
        "success": best.success,
        "evaluations": best.evaluations,
        "iterations": best.iterations,
        "projected_gradient_inf_norm": (
            best.projected_gradient_inf_norm
        ),
        "stationary": best.stationary,
        "coordinate_stationary": best.coordinate_stationary,
        "rejected_start_attempts": best.rejected_start_attempts,
        "run_results": best.run_results,
        "message": best.message,
        "parameter_names": parameter_names(),
        "parameter_vector": best.parameter_vector,
        "all_runs": [result.__dict__ for result in results],
    }
    _write_json(payload, args.output)


def _validate(args) -> None:
    _write_json(validation_report(run_pnl=not args.skip_pnl), args.output)


def _staged_fit(args) -> None:
    trials, likelihood, dtype = _load_problem(args)
    initial_vectors = [
        _load_parameters(path, dtype=dtype, device=args.device)
        .vector()
        .detach()
        .cpu()
        .numpy()
        for path in args.initial_parameters
    ]
    result = fit_staged(
        likelihood,
        trials,
        seed=args.seed,
        initial_vectors=initial_vectors,
        coarse_ddm_time_step=args.coarse_ddm_time_step,
        coarse_ddm_spatial_points=args.coarse_ddm_spatial_points,
        coarse_lca_max_step=args.coarse_lca_max_step,
        coarse_cma_evaluations=args.coarse_cma_evaluations,
        default_starts=args.default_starts,
        default_max_iterations=args.default_max_iterations,
        random_start_candidates=args.random_start_candidates,
        fine=not args.skip_fine,
        fine_ddm_time_step=args.fine_ddm_time_step,
        fine_ddm_spatial_points=args.fine_ddm_spatial_points,
        fine_lca_max_step=args.fine_lca_max_step,
        fine_max_iterations=args.fine_max_iterations,
    )
    payload = {
        "subject_nr": trials.subject_nr,
        "method": (
            "coarse-cmaes/default-lbfgsb/fine-lbfgsb"
            if result.fine_result is not None
            else "coarse-cmaes/default-lbfgsb"
        ),
        "final_mesh": result.final_mesh,
        "log_likelihood": result.log_likelihood,
        "parameter_names": parameter_names(),
        "parameter_vector": result.parameter_vector,
        "meshes": result.meshes,
        "stage_seconds": result.stage_seconds,
        "coarse_result": result.coarse_result.__dict__,
        "default_result": result.default_result.__dict__,
        "fine_result": (
            result.fine_result.__dict__
            if result.fine_result is not None
            else None
        ),
    }
    _write_json(payload, args.output)


def _generator_validation(args) -> None:
    simulation_time_steps = (
        tuple(args.simulation_time_step)
        if args.simulation_time_step is not None
        else (0.002, 0.001, 0.0005)
    )
    cutoffs = (
        tuple(args.cutoff)
        if args.cutoff is not None
        else (0.25, 0.5, 0.75, 1.0, 1.5)
    )
    _write_json(
        generator_validation_report(
            paths=args.paths,
            seed=args.seed,
            simulation_time_steps=simulation_time_steps,
            reference_time_step=args.reference_time_step,
            reference_spatial_points=args.reference_spatial_points,
            cutoffs=cutoffs,
        ),
        args.output,
    )


def _recovery_frame(trials: CSITrialData) -> pd.DataFrame:
    condition = trials.condition_index.detach().cpu().numpy()
    return pd.DataFrame(
        {
            "subject_nr": trials.subject_nr,
            "sequence": [CONDITIONS[int(index)] for index in condition],
            "T1": trials.task[:, 0].detach().cpu().numpy(),
            "T2": trials.task[:, 1].detach().cpu().numpy(),
            "S1": trials.stimulus[:, 0].detach().cpu().numpy(),
            "S2": trials.stimulus[:, 1].detach().cpu().numpy(),
            "S3": trials.stimulus[:, 2].detach().cpu().numpy(),
            "S4": trials.stimulus[:, 3].detach().cpu().numpy(),
            "correct_response": (
                trials.correct_response.detach().cpu().numpy()
            ),
            "decision": trials.choice.detach().cpu().numpy(),
            "response_time": trials.response_time.detach().cpu().numpy(),
            "likelihood_include_mask": (
                trials.include.detach().cpu().numpy()
            ),
            "row_id": trials.row_id.detach().cpu().numpy(),
        }
    )


def _recover(args) -> None:
    template, likelihood, dtype = _load_problem(args)
    if args.truth_parameters is None:
        truth_vector = torch.tensor(
            [
                15.0, 18.0, 12.0, 0.08,
                0.10, 0.09, 0.11,
                -0.020, -0.015, -0.025,
                0.20, 0.23, 0.18,
            ],
            dtype=dtype,
            device=args.device,
        )
        truth = ContinuousCSIParameters.from_vector(truth_vector)
    else:
        truth = _load_parameters(
            args.truth_parameters, dtype=dtype, device=args.device
        )
    simulation = simulate_sequential_trials(
        likelihood,
        truth,
        template,
        seed=args.simulation_seed,
        simulation_time_step=args.simulation_time_step,
        maximum_decision_time=args.maximum_decision_time,
        bridge_correction=args.bridge_correction,
    )
    if args.simulated_output is not None:
        args.simulated_output.parent.mkdir(parents=True, exist_ok=True)
        _recovery_frame(simulation.trials).to_csv(
            args.simulated_output, index=False
        )
    with torch.no_grad():
        truth_score = likelihood.score(truth, simulation.trials)
    fit = fit_lbfgsb(
        likelihood,
        simulation.trials,
        starts=args.starts,
        max_iterations=args.max_iterations,
        seed=args.seed,
        max_start_attempts=args.max_start_attempts,
        random_start_candidates=args.random_start_candidates,
        function_tolerance=args.optimizer_ftol,
        gradient_tolerance=args.optimizer_gtol,
        maximum_line_search_steps=args.max_line_search_steps,
        polish_restarts=args.polish_restarts,
        polish_iterations=args.polish_iterations,
        polish_method=args.polish_method,
        coordinate_polish=not args.no_coordinate_polish,
        coordinate_step=args.coordinate_step,
        coordinate_levels=args.coordinate_levels,
        coordinate_rounds=args.coordinate_rounds,
        coordinate_cycles=args.coordinate_cycles,
        coordinate_improvement_tolerance=(
            args.coordinate_improvement_tolerance
        ),
        stationarity_tolerance=args.stationarity_tolerance,
    )
    truth_array = truth.vector().detach().cpu().numpy()
    error = fit.parameter_vector - truth_array
    lower, upper = parameter_bounds()
    payload = {
        "subject_nr": template.subject_nr,
        "truth_label": args.truth_label,
        "simulation_seed": args.simulation_seed,
        "simulation_time_step": args.simulation_time_step,
        "bridge_correction": simulation.bridge_correction,
        "maximum_simulated_decision_time": simulation.maximum_decision_time,
        "mean_simulated_decision_time": simulation.mean_decision_time,
        "simulated_accuracy": float(simulation.trials.choice.mean()),
        "included_rows": int(simulation.trials.include.sum()),
        "parameter_names": parameter_names(),
        "truth_parameter_vector": truth_array,
        "recovered_parameter_vector": fit.parameter_vector,
        "parameter_error": error,
        "scaled_parameter_error": error / (upper - lower),
        "scaled_parameter_rmse": float(
            np.sqrt(np.mean(np.square(error / (upper - lower))))
        ),
        "truth_log_likelihood": truth_score.log_likelihood,
        "recovered_log_likelihood": fit.log_likelihood,
        "fit": fit.__dict__,
    }
    _write_json(payload, args.output)


def _recovery_summary(args) -> None:
    summary = summarize_recovery_results(
        load_recovery_results(args.input),
        bound_tolerance=args.bound_tolerance,
    )
    _write_json(summary, args.output)
    if args.csv_output is not None:
        rows = []
        for run in summary["recoveries"]:
            row = {
                key: value
                for key, value in run.items()
                if key not in {
                    "parameter_error",
                    "scaled_parameter_error",
                    "lower_bound_parameters",
                    "upper_bound_parameters",
                }
            }
            for name, error, scaled_error in zip(
                summary["parameter_names"],
                run["parameter_error"],
                run["scaled_parameter_error"],
                strict=True,
            ):
                row[f"error:{name}"] = error
                row[f"scaled_error:{name}"] = scaled_error
            row["lower_bound_parameters"] = ";".join(
                run["lower_bound_parameters"]
            )
            row["upper_bound_parameters"] = ";".join(
                run["upper_bound_parameters"]
            )
            rows.append(row)
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.csv_output, index=False)


def _benchmark(args) -> None:
    if args.repeats < 1:
        raise ValueError("--repeats must be at least one.")
    if args.warmups < 0:
        raise ValueError("--warmups cannot be negative.")
    trials, likelihood, dtype = _load_problem(args)
    parameters = _load_parameters(args.parameters, dtype=dtype, device=args.device)
    objective_gradient_seconds = []
    if args.with_gradient:
        for _ in range(args.warmups):
            vector = parameters.vector().detach().clone().requires_grad_()
            warm_result = likelihood.score_vector(vector, trials)
            (-warm_result.log_likelihood).backward()
        runs = []
        for _ in range(args.repeats):
            vector = parameters.vector().detach().clone().requires_grad_()
            start = time.perf_counter()
            result = likelihood.score_vector(
                vector, trials, collect_timings=True
            )
            (-result.log_likelihood).backward()
            objective_gradient_seconds.append(time.perf_counter() - start)
            runs.append(result)
    else:
        with torch.no_grad():
            for _ in range(args.warmups):
                likelihood.score(parameters, trials)
            runs = [
                likelihood.score(parameters, trials, collect_timings=True)
                for _ in range(args.repeats)
            ]
    timing_names = tuple(runs[0].timings)
    timing_summary = {
        name: {
            "median": statistics.median(run.timings[name] for run in runs),
            "minimum": min(run.timings[name] for run in runs),
            "maximum": max(run.timings[name] for run in runs),
        }
        for name in timing_names
    }
    final = runs[-1]
    required = int(final.diagnostics["drift_cells_required"])
    computed = int(final.diagnostics["drift_cells_computed"])
    payload = {
        "subject_nr": trials.subject_nr,
        "rows": len(trials),
        "included_rows": int(trials.include.sum()),
        "repeats": args.repeats,
        "warmups": args.warmups,
        "log_likelihood": final.log_likelihood,
        "timings": timing_summary,
        "objective_gradient_seconds": (
            {
                "median": statistics.median(objective_gradient_seconds),
                "minimum": min(objective_gradient_seconds),
                "maximum": max(objective_gradient_seconds),
            }
            if objective_gradient_seconds
            else None
        ),
        "ddm_bucket_count": final.diagnostics["ddm_bucket_count"],
        "drift_cells_required": required,
        "drift_cells_computed": computed,
        "drift_cell_efficiency": required / computed if computed else 1.0,
    }
    _write_json(payload, args.output)


def _parse_grid_level(value: str) -> tuple[float, int, float]:
    try:
        ddm_step, spatial_points, lca_step = value.split(",")
        parsed = (float(ddm_step), int(spatial_points), float(lca_step))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "A grid level must be DDM_DT,SPATIAL_POINTS,LCA_DT."
        ) from error
    if parsed[0] <= 0.0 or parsed[1] < 5 or parsed[2] <= 0.0:
        raise argparse.ArgumentTypeError("Grid-level values must be positive.")
    return parsed


def _grid_refinement(args) -> None:
    trials, base_likelihood, dtype = _load_problem(args)
    parameters = _load_parameters(args.parameters, dtype=dtype, device=args.device)
    results = []
    with torch.no_grad():
        for ddm_step, spatial_points, lca_step in args.level:
            likelihood = ContinuousCSILikelihood(
                replace(
                    base_likelihood.config,
                    ddm_time_step=ddm_step,
                    ddm_spatial_points=spatial_points,
                    lca_max_step=lca_step,
                )
            )
            results.append(
                likelihood.score(parameters, trials, collect_timings=True)
            )

    reference = results[-1]
    included = trials.include
    reference_probability = reference.probability[included]
    levels = []
    for specification, result in zip(args.level, results, strict=True):
        current_probability = result.probability[included]
        positive = (current_probability > 0.0) & (reference_probability > 0.0)
        if bool(torch.any(positive)):
            log_difference = torch.abs(
                torch.log(current_probability[positive])
                - torch.log(reference_probability[positive])
            )
            mean_log_difference = float(torch.mean(log_difference))
            maximum_log_difference = float(torch.amax(log_difference))
        else:
            mean_log_difference = float("inf")
            maximum_log_difference = float("inf")
        levels.append(
            {
                "ddm_time_step": specification[0],
                "ddm_spatial_points": specification[1],
                "lca_max_step": specification[2],
                "log_likelihood": result.log_likelihood,
                "log_likelihood_difference_from_finest": (
                    result.log_likelihood - reference.log_likelihood
                ),
                "mean_absolute_trial_log_probability_difference": (
                    mean_log_difference
                ),
                "maximum_absolute_trial_log_probability_difference": (
                    maximum_log_difference
                ),
                "maximum_lca_state_difference": torch.amax(
                    torch.abs(
                        result.lca_state_after_trial
                        - reference.lca_state_after_trial
                    )
                ),
                "maximum_mass_error": result.diagnostics["maximum_mass_error"],
                "zero_probability_rows": result.diagnostics[
                    "zero_probability_included_rows"
                ],
                "timings": result.timings,
            }
        )
    _write_json(
        {
            "subject_nr": trials.subject_nr,
            "reference": "final --level entry",
            "levels": levels,
        },
        args.output,
    )


def _endpoint_grid_refinement(args) -> None:
    """Check spatial convergence of the deterministic endpoint DDM ranking."""
    trials, base_likelihood, dtype = _load_problem(args)
    direct_parameters = _load_parameters(
        args.direct_parameters, dtype=dtype, device=args.device
    )
    pnl_parameters = _load_parameters(
        args.pnl_parameters, dtype=dtype, device=args.device
    )
    rounded_vector = direct_parameters.vector().clone()
    rounded_vector[3] = (
        torch.round(rounded_vector[3] / args.endpoint_time_step)
        * args.endpoint_time_step
    )
    direct_parameters = ContinuousCSIParameters.from_vector(rounded_vector)
    levels = []
    with torch.no_grad():
        for spatial_points in args.spatial_points:
            config = replace(
                base_likelihood.config,
                ddm_process="endpoint",
                ddm_time_step=args.endpoint_time_step,
                ddm_spatial_points=spatial_points,
                endpoint_evidence_domain=args.endpoint_evidence_domain,
                lca_max_step=args.endpoint_time_step,
                lca_integration_method="euler",
                rt_bin_count=args.legacy_bins,
                native_lca_scan=False,
                native_ddm_forward=False,
            )
            likelihood = ContinuousCSILikelihood(config)
            direct_result = likelihood.score(
                direct_parameters, trials, collect_timings=True
            )
            pnl_result = likelihood.score(
                pnl_parameters, trials, collect_timings=True
            )
            direct_value = float(direct_result.log_likelihood)
            pnl_value = float(pnl_result.log_likelihood)
            rt_bin_width = float(
                direct_result.diagnostics["rt_bin_width"]
            )
            included_count = int(trials.include.sum())
            log_density_normalization = included_count * math.log(rt_bin_width)
            levels.append(
                {
                    "spatial_points": spatial_points,
                    "direct_vector_log_likelihood": direct_value,
                    "pnl_vector_log_likelihood": pnl_value,
                    "direct_vector_log_density": (
                        direct_value - log_density_normalization
                    ),
                    "pnl_vector_log_density": (
                        pnl_value - log_density_normalization
                    ),
                    "direct_minus_pnl": direct_value - pnl_value,
                    "rt_bin_width": rt_bin_width,
                    "maximum_mass_error": max(
                        float(direct_result.diagnostics["maximum_mass_error"]),
                        float(pnl_result.diagnostics["maximum_mass_error"]),
                    ),
                    "zero_probability_rows": {
                        "direct": direct_result.diagnostics[
                            "zero_probability_included_rows"
                        ],
                        "pnl": pnl_result.diagnostics[
                            "zero_probability_included_rows"
                        ],
                    },
                    "elapsed_seconds": (
                        direct_result.timings["total_seconds"]
                        + pnl_result.timings["total_seconds"]
                    ),
                }
            )
    reference = levels[-1]
    for level in levels:
        level["ranking_gap_difference_from_finest"] = (
            level["direct_minus_pnl"] - reference["direct_minus_pnl"]
        )
    reset_likelihood = ContinuousCSILikelihood(
        replace(config, reset_lca_each_trial=True)
    )
    with torch.no_grad():
        reset_direct = reset_likelihood.score(
            direct_parameters, trials, collect_timings=True
        )
        reset_pnl = reset_likelihood.score(
            pnl_parameters, trials, collect_timings=True
        )
    reset_direct_value = float(reset_direct.log_likelihood)
    reset_pnl_value = float(reset_pnl.log_likelihood)
    reset_bin_width = float(reset_direct.diagnostics["rt_bin_width"])
    reset_normalization = int(trials.include.sum()) * math.log(reset_bin_width)
    reset_history_ablation = {
        "spatial_points": config.ddm_spatial_points,
        "description": (
            "Sensitivity bound only: resetting history is not the PNL "
            "simulator's marginal-history objective."
        ),
        "direct_vector_log_likelihood": reset_direct_value,
        "pnl_vector_log_likelihood": reset_pnl_value,
        "direct_vector_log_density": (
            reset_direct_value - reset_normalization
        ),
        "pnl_vector_log_density": reset_pnl_value - reset_normalization,
        "direct_minus_pnl": reset_direct_value - reset_pnl_value,
        "ranking_gap_change_from_persistent": (
            reset_direct_value - reset_pnl_value - reference["direct_minus_pnl"]
        ),
        "maximum_mass_error": max(
            float(reset_direct.diagnostics["maximum_mass_error"]),
            float(reset_pnl.diagnostics["maximum_mass_error"]),
        ),
        "elapsed_seconds": (
            reset_direct.timings["total_seconds"]
            + reset_pnl.timings["total_seconds"]
        ),
    }
    _write_json(
        {
            "subject_nr": trials.subject_nr,
            "ddm_process": "10 ms Gaussian Euler endpoint crossing",
            "reference": "final --spatial-points entry",
            "endpoint_time_step": args.endpoint_time_step,
            "endpoint_evidence_domain": args.endpoint_evidence_domain,
            "legacy_bins": args.legacy_bins,
            "levels": levels,
            "reset_history_ablation": reset_history_ablation,
        },
        args.output,
    )


def _gradient_benchmark(args) -> None:
    trials, likelihood, dtype = _load_problem(args)
    parameters = _load_parameters(
        args.parameters, dtype=dtype, device=args.device
    )
    vector = parameters.vector().detach().cpu().numpy()
    start = time.perf_counter()
    objective, gradient = parallel_finite_difference_objective(
        likelihood,
        trials,
        vector,
        workers=args.workers,
        relative_step=args.relative_step,
    )
    elapsed = time.perf_counter() - start
    _write_json(
        {
            "subject_nr": trials.subject_nr,
            "method": "parallel-finite-difference",
            "negative_log_likelihood": objective,
            "gradient": gradient,
            "parameter_names": parameter_names(),
            "workers": args.workers,
            "relative_step": args.relative_step,
            "elapsed_seconds": elapsed,
        },
        args.output,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    score = commands.add_parser("score", help="Evaluate one parameter vector.")
    _add_model_arguments(score)
    score.add_argument("--parameters", type=Path)
    score.add_argument("--output", type=Path)
    score.add_argument("--trial-output", type=Path)
    score.add_argument("--profile", action="store_true")
    score.set_defaults(action=_score)

    export_pnl = commands.add_parser(
        "export-pnl-parameters",
        help="Convert a direct parameter JSON to the legacy one-row PNL CSV.",
    )
    export_pnl.add_argument("--parameters", type=Path, required=True)
    export_pnl.add_argument("--subject", type=int, required=True)
    export_pnl.add_argument("--output", type=Path, required=True)
    export_pnl.set_defaults(action=_export_pnl_parameters)

    compare_pnl = commands.add_parser(
        "compare-pnl",
        help="Compare direct and PNL vectors under both likelihood objectives.",
    )
    _add_model_arguments(compare_pnl)
    compare_pnl.add_argument("--direct-parameters", type=Path, required=True)
    compare_pnl.add_argument("--pnl-parameters", type=Path, required=True)
    compare_pnl.add_argument("--direct-pnl-rescore", type=Path, required=True)
    compare_pnl.add_argument("--pnl-pnl-rescore", type=Path, required=True)
    compare_pnl.add_argument("--output", type=Path, required=True)
    compare_pnl.set_defaults(action=_compare_pnl)

    semantic_ladder = commands.add_parser(
        "semantic-ladder",
        help="Factor direct/PNL vector rankings across legacy numerical semantics.",
    )
    _add_model_arguments(semantic_ladder)
    semantic_ladder.add_argument(
        "--direct-parameters", type=Path, required=True
    )
    semantic_ladder.add_argument(
        "--pnl-parameters", type=Path, required=True
    )
    semantic_ladder.add_argument("--legacy-time-step", type=float, default=0.01)
    semantic_ladder.add_argument(
        "--legacy-spatial-points", type=int, default=65
    )
    semantic_ladder.add_argument("--legacy-bins", type=int, default=100)
    semantic_ladder.add_argument(
        "--legacy-rannacher-steps",
        type=int,
        default=1_000_000,
        help="Fully implicit by default to prevent coarse-grid CN ringing.",
    )
    semantic_ladder.add_argument(
        "--endpoint-spatial-points", type=int, default=511
    )
    semantic_ladder.add_argument(
        "--endpoint-evidence-domain", type=float, default=0.35
    )
    semantic_ladder.add_argument("--output", type=Path, required=True)
    semantic_ladder.set_defaults(action=_semantic_ladder)

    fit = commands.add_parser("fit", help="Fit one participant.")
    _add_model_arguments(fit)
    fit.add_argument(
        "--optimizer", choices=("lbfgsb", "cmaes", "both"), default="lbfgsb"
    )
    fit.add_argument("--starts", type=int, default=4)
    fit.add_argument(
        "--initial-parameters",
        type=Path,
        action="append",
        default=[],
        help=(
            "JSON parameter vector to use as a start; repeat for multiple "
            "starts. Validated defaults and random starts fill the remainder."
        ),
    )
    fit.add_argument("--max-iterations", type=int, default=200)
    fit.add_argument("--no-default-start", action="store_true")
    fit.add_argument("--max-start-attempts", type=int, default=100)
    fit.add_argument(
        "--random-start-candidates",
        type=int,
        default=16,
        help="Score this many valid random candidates and optimize the best.",
    )
    fit.add_argument("--optimizer-ftol", type=float, default=1.0e-12)
    fit.add_argument("--optimizer-gtol", type=float, default=1.0e-5)
    fit.add_argument("--max-line-search-steps", type=int, default=50)
    fit.add_argument("--polish-restarts", type=int, default=1)
    fit.add_argument("--polish-iterations", type=int, default=100)
    fit.add_argument(
        "--polish-method", choices=("lbfgsb", "powell"), default="lbfgsb"
    )
    fit.add_argument("--no-coordinate-polish", action="store_true")
    fit.add_argument("--coordinate-step", type=float, default=1.0e-3)
    fit.add_argument("--coordinate-levels", type=int, default=3)
    fit.add_argument("--coordinate-rounds", type=int, default=2)
    fit.add_argument("--coordinate-cycles", type=int, default=2)
    fit.add_argument(
        "--coordinate-improvement-tolerance", type=float, default=1.0e-7
    )
    fit.add_argument("--stationarity-tolerance", type=float, default=1.0e-2)
    fit.add_argument("--cma-evaluations", type=int, default=500)
    fit.add_argument("--seed", type=int, default=1)
    fit.add_argument(
        "--gradient-method",
        choices=("parallel-finite-difference", "autograd"),
        default="autograd",
        help=(
            "Use the exact compiled custom adjoints (default) or parallel "
            "deterministic perturbation scores."
        ),
    )
    fit.add_argument("--finite-difference-workers", type=int)
    fit.add_argument(
        "--finite-difference-relative-step", type=float, default=1.0e-5
    )
    fit.add_argument("--output", type=Path, required=True)
    fit.set_defaults(action=_fit)

    staged = commands.add_parser(
        "staged-fit",
        help="Run coarse CMA-ES, default L-BFGS-B, then fine polishing.",
    )
    _add_model_arguments(staged)
    staged.add_argument(
        "--initial-parameters", type=Path, action="append", default=[]
    )
    staged.add_argument("--seed", type=int, default=1)
    staged.add_argument("--coarse-ddm-time-step", type=float, default=0.002)
    staged.add_argument("--coarse-ddm-spatial-points", type=int, default=33)
    staged.add_argument("--coarse-lca-max-step", type=float, default=0.02)
    staged.add_argument("--coarse-cma-evaluations", type=int, default=1_000)
    staged.add_argument("--default-starts", type=int, default=2)
    staged.add_argument("--default-max-iterations", type=int, default=100)
    staged.add_argument("--random-start-candidates", type=int, default=32)
    staged.add_argument("--skip-fine", action="store_true")
    staged.add_argument("--fine-ddm-time-step", type=float, default=0.0005)
    staged.add_argument("--fine-ddm-spatial-points", type=int, default=129)
    staged.add_argument("--fine-lca-max-step", type=float, default=0.005)
    staged.add_argument("--fine-max-iterations", type=int, default=50)
    staged.add_argument("--output", type=Path, required=True)
    staged.set_defaults(action=_staged_fit, gradient_method="autograd")

    recover = commands.add_parser(
        "recover",
        help="Generate and refit one seeded synthetic sequential dataset.",
    )
    _add_model_arguments(recover)
    recover.add_argument("--truth-parameters", type=Path)
    recover.add_argument("--truth-label", default="interior")
    recover.add_argument("--simulation-seed", type=int, default=17)
    recover.add_argument("--simulation-time-step", type=float, default=0.0005)
    recover.add_argument("--maximum-decision-time", type=float, default=3.0)
    recover.add_argument(
        "--bridge-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detect within-step crossings with a Brownian-bridge test.",
    )
    recover.add_argument("--starts", type=int, default=2)
    recover.add_argument("--max-iterations", type=int, default=100)
    recover.add_argument("--seed", type=int, default=1)
    recover.add_argument("--max-start-attempts", type=int, default=100)
    recover.add_argument(
        "--random-start-candidates", type=int, default=32
    )
    recover.add_argument("--optimizer-ftol", type=float, default=1.0e-12)
    recover.add_argument("--optimizer-gtol", type=float, default=1.0e-5)
    recover.add_argument("--max-line-search-steps", type=int, default=50)
    recover.add_argument("--polish-restarts", type=int, default=1)
    recover.add_argument("--polish-iterations", type=int, default=50)
    recover.add_argument(
        "--polish-method", choices=("lbfgsb", "powell"), default="lbfgsb"
    )
    recover.add_argument("--no-coordinate-polish", action="store_true")
    recover.add_argument("--coordinate-step", type=float, default=1.0e-3)
    recover.add_argument("--coordinate-levels", type=int, default=3)
    recover.add_argument("--coordinate-rounds", type=int, default=2)
    recover.add_argument("--coordinate-cycles", type=int, default=2)
    recover.add_argument(
        "--coordinate-improvement-tolerance", type=float, default=1.0e-7
    )
    recover.add_argument("--stationarity-tolerance", type=float, default=1.0e-2)
    recover.add_argument("--simulated-output", type=Path)
    recover.add_argument("--output", type=Path, required=True)
    recover.set_defaults(action=_recover, gradient_method="autograd")

    recovery_summary = commands.add_parser(
        "recovery-summary",
        help="Aggregate a matrix of recovery JSON results.",
    )
    recovery_summary.add_argument(
        "--input", type=Path, action="append", required=True
    )
    recovery_summary.add_argument("--bound-tolerance", type=float, default=1e-5)
    recovery_summary.add_argument("--output", type=Path, required=True)
    recovery_summary.add_argument("--csv-output", type=Path)
    recovery_summary.set_defaults(action=_recovery_summary)

    validate = commands.add_parser(
        "validate", help="Run analytic and PNL continuum-refinement checks."
    )
    validate.add_argument("--skip-pnl", action="store_true")
    validate.add_argument("--output", type=Path)
    validate.set_defaults(action=_validate)

    generator_validation = commands.add_parser(
        "generator-validation",
        help="Compare Euler first-passage samples with a fine PDE reference.",
    )
    generator_validation.add_argument("--paths", type=int, default=50_000)
    generator_validation.add_argument("--seed", type=int, default=9182)
    generator_validation.add_argument(
        "--simulation-time-step",
        type=float,
        action="append",
        default=None,
        help="Simulation mesh; repeat from coarse to fine.",
    )
    generator_validation.add_argument(
        "--reference-time-step", type=float, default=0.00025
    )
    generator_validation.add_argument(
        "--reference-spatial-points", type=int, default=257
    )
    generator_validation.add_argument(
        "--cutoff", type=float, action="append", default=None
    )
    generator_validation.add_argument("--output", type=Path)
    generator_validation.set_defaults(action=_generator_validation)

    benchmark = commands.add_parser(
        "benchmark", help="Time the major phases of one subject score."
    )
    _add_model_arguments(benchmark)
    benchmark.add_argument("--parameters", type=Path)
    benchmark.add_argument("--repeats", type=int, default=3)
    benchmark.add_argument("--warmups", type=int, default=1)
    benchmark.add_argument(
        "--with-gradient",
        action="store_true",
        help="Time an exact objective plus all 13 parameter gradients.",
    )
    benchmark.add_argument("--output", type=Path)
    benchmark.set_defaults(action=_benchmark)

    grid = commands.add_parser(
        "grid-refinement",
        help="Compare subject likelihoods across numerical meshes.",
    )
    _add_model_arguments(grid)
    grid.add_argument("--parameters", type=Path)
    grid.add_argument(
        "--level",
        type=_parse_grid_level,
        action="append",
        required=True,
        metavar="DDM_DT,POINTS,LCA_DT",
        help="Add a mesh from coarse to fine; the final entry is the reference.",
    )
    grid.add_argument("--output", type=Path)
    grid.set_defaults(action=_grid_refinement)

    endpoint_grid = commands.add_parser(
        "endpoint-grid-refinement",
        help="Check endpoint-crossing DDM ranking over evidence grids.",
    )
    _add_model_arguments(endpoint_grid)
    endpoint_grid.add_argument(
        "--direct-parameters", type=Path, required=True
    )
    endpoint_grid.add_argument(
        "--pnl-parameters", type=Path, required=True
    )
    endpoint_grid.add_argument(
        "--spatial-points", type=int, action="append", required=True
    )
    endpoint_grid.add_argument("--endpoint-time-step", type=float, default=0.01)
    endpoint_grid.add_argument(
        "--endpoint-evidence-domain", type=float, default=0.35
    )
    endpoint_grid.add_argument("--legacy-bins", type=int, default=100)
    endpoint_grid.add_argument("--output", type=Path, required=True)
    endpoint_grid.set_defaults(action=_endpoint_grid_refinement)

    gradient = commands.add_parser(
        "gradient-benchmark",
        help="Time one parallel finite-difference objective and gradient.",
    )
    _add_model_arguments(gradient)
    gradient.add_argument("--parameters", type=Path)
    gradient.add_argument("--workers", type=int)
    gradient.add_argument("--relative-step", type=float, default=1.0e-5)
    gradient.add_argument("--output", type=Path)
    gradient.set_defaults(action=_gradient_benchmark)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
