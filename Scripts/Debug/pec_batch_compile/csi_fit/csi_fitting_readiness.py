#!/usr/bin/env python3
"""Generate and fit crossed CSI parameter-recovery experiments.

The workflow keeps data generation and inference methods explicit. Synthetic
data can come from the continuous first-passage model or the legacy-compatible
GPU endpoint process at 10 ms or 1 ms. Each dataset can then be fit by the
continuous direct likelihood or by the corrected deterministic-history GPU
likelihood at either endpoint resolution.

One ``run-cell`` invocation owns one generator/fitter/seed combination and is
therefore safe to use as a Slurm array task. Outputs are resumable and include
all physical-unit parameters needed to compare methods.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from psyneulink.core.batched import (
    BatchedCompositionCompiler,
    BatchedTrialParameter,
    batched_node_op,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[3]
DIRECT_DRIVER = SCRIPT_DIR / "csi_direct_likelihood.py"
GPU_FIT_DRIVER = (
    SCRIPT_DIR
    / "data fitting"
    / "expectation_fit_study3.2_real_sequences_single_csi_leak12.py"
)
MODEL_PATH = (
    SCRIPT_DIR / "data fitting" / "expectation_model_study2_study3.py"
)
DEFAULT_DATA = SCRIPT_DIR / "data fitting" / "data_to_fit_study3.csv"

sys.path.insert(0, str(SCRIPT_DIR))
from direct_likelihood import (  # noqa: E402
    CONDITIONS,
    CSITrialData,
    ContinuousCSILikelihood,
    ContinuousCSIParameters,
    SolverConfig,
    simulate_sequential_trials,
)
from direct_likelihood.model import (  # noqa: E402
    parameter_bounds,
    parameter_names,
)


DEFAULT_TRUTH = np.asarray(
    [
        15.0,
        18.0,
        12.0,
        0.08,
        0.10,
        0.09,
        0.11,
        -0.020,
        -0.015,
        -0.025,
        0.20,
        0.23,
        0.18,
    ],
    dtype=float,
)
GENERATOR_TIME_STEPS = {
    "gpu-10ms": 0.01,
    "gpu-1ms": 0.001,
}
FITTER_TIME_STEPS = {
    "gpu-10ms": 0.01,
    "gpu-1ms": 0.001,
}


@batched_node_op("Drift Rate Value")
def _batched_csi_drift_rate(x0, x1, x2, x3, x4, x5, x6):
    """Triton transcription of the original seven-input CSI drift UDF."""
    a = 1.0 / (1.0 + tl.exp(-((x0 - x1) + 4.0 * x4 - 4.0)))
    b = 1.0 / (1.0 + tl.exp(-((x1 - x0) + 4.0 * x4 - 4.0)))
    c = 1.0 / (1.0 + tl.exp(-((x2 - x3) + 4.0 * x5 - 4.0)))
    d = 1.0 / (1.0 + tl.exp(-((x3 - x2) + 4.0 * x5 - 4.0)))
    positive = 1.0 / (1.0 + tl.exp(-(a - b + c - d)))
    negative = 1.0 / (1.0 + tl.exp(-(-a + b - c + d)))
    return (positive - negative) * x6


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _legacy_value(row: pd.Series, base: str, condition: str | None = None) -> float:
    wanted = f"{base}[{condition}]" if condition is not None else base
    matches = [
        key
        for key in row.index
        if str(key) == wanted or str(key).endswith(wanted)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one fitted column matching {wanted!r}; found {matches}."
        )
    return float(row[matches[0]])


def _legacy_row_to_physical_vector(
    row: pd.Series, *, model_time_step: float | None = None
) -> np.ndarray:
    """Convert a legacy scheduler-unit fit row to the 13 physical parameters."""
    if model_time_step is None:
        model_time_step = float(row.get("model_time_step", 0.01))
    if not math.isfinite(model_time_step) or model_time_step <= 0.0:
        raise ValueError("A legacy fit's model time step must be positive.")
    gains = [
        _legacy_value(row, "Task Activations [C1, C2].gain", condition)
        for condition in CONDITIONS
    ]
    csi_duration = _legacy_value(row, "Cue Stimulus Interval.slope") * model_time_step
    thresholds = [
        _legacy_value(row, "Threshold Mechanism.intercept", condition)
        for condition in CONDITIONS
    ]
    collapse_rates = [
        _legacy_value(
            row,
            "Threshold Mechanism.offset-integrator_function",
            condition,
        )
        / model_time_step
        for condition in CONDITIONS
    ]
    non_decision_times = [
        _legacy_value(row, "DDM.non_decision_time", condition)
        for condition in CONDITIONS
    ]
    return np.asarray(
        [
            *gains,
            csi_duration,
            *thresholds,
            *collapse_rates,
            *non_decision_times,
        ],
        dtype=float,
    )


def _load_parameter_vector(path: Path | None) -> np.ndarray:
    if path is None:
        vector = DEFAULT_TRUTH.copy()
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if len(frame) != 1:
            raise ValueError(
                f"Parameter CSV must contain one row; found {len(frame)}."
            )
        vector = _legacy_row_to_physical_vector(frame.iloc[0])
    else:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            for key in (
                "parameter_vector",
                "truth_parameter_vector",
                "recovered_parameter_vector",
            ):
                if key in payload:
                    payload = payload[key]
                    break
            else:
                payload = [payload[name] for name in parameter_names()]
        vector = np.asarray(payload, dtype=float)
    if vector.shape != (13,) or not np.all(np.isfinite(vector)):
        raise ValueError("The truth parameter vector must contain 13 finite values.")
    lower, upper = parameter_bounds()
    if np.any(vector < lower) or np.any(vector > upper):
        raise ValueError("The truth parameter vector lies outside the fitting bounds.")
    return vector


def _parameters(vector: np.ndarray) -> ContinuousCSIParameters:
    return ContinuousCSIParameters.from_vector(
        torch.as_tensor(vector, dtype=torch.float64)
    )


def _load_model_module():
    specification = importlib.util.spec_from_file_location(
        "_csi_fitting_readiness_model", MODEL_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load the CSI model from {MODEL_PATH}.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _node(composition, base_name: str):
    matches = tuple(
        node
        for node in composition.nodes
        if re.sub(r"-\d+$", "", node.name) == base_name
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one CSI node named {base_name!r}; found {len(matches)}."
        )
    return matches[0]


def _subject_frame(path: Path, subject: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame[
        (frame["subject_nr"] == subject)
        & frame["sequence"].isin(CONDITIONS)
    ].reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"subject_nr={subject} has no retained CSI rows in {path}.")
    return selected


def _gpu_inputs(composition, trials: CSITrialData):
    count = len(trials)
    task = trials.task.detach().cpu().numpy()
    return {
        _node(composition, "Task Input"): task,
        _node(composition, "Stimulus Input"): (
            trials.stimulus.detach().cpu().numpy()
        ),
        _node(composition, "Correct Response"): (
            trials.correct_response.detach().cpu().numpy().reshape(-1, 1)
        ),
        _node(composition, "Cue Stimulus Interval"): (
            trials.is_switch.detach().cpu().numpy().reshape(-1, 1)
        ),
        _node(composition, "Threshold Mechanism"): np.zeros((count, 1)),
    }


def _gpu_parameter_set(
    composition,
    trials: CSITrialData,
    vector: np.ndarray,
    time_step: float,
) -> tuple[dict[str, Any], np.ndarray]:
    parameters = _parameters(vector)
    condition = trials.condition_index.detach().cpu().numpy()
    rounded_csi_steps = round(float(parameters.csi_duration) / time_step)
    realized = vector.copy()
    realized[3] = rounded_csi_steps * time_step
    return (
        {
            f"{_node(composition, 'Task Activations [C1, C2]').name}.gain": (
                BatchedTrialParameter(
                    parameters.gain.detach().cpu().numpy()[condition]
                )
            ),
            f"{_node(composition, 'Cue Stimulus Interval').name}.slope": (
                float(rounded_csi_steps)
            ),
            f"{_node(composition, 'Threshold Mechanism').name}.intercept": (
                BatchedTrialParameter(
                    parameters.threshold.detach().cpu().numpy()[condition]
                )
            ),
            (
                f"{_node(composition, 'Threshold Mechanism').name}."
                "offset-integrator_function"
            ): BatchedTrialParameter(
                parameters.collapse_rate.detach().cpu().numpy()[condition]
                * time_step
            ),
            f"{_node(composition, 'DDM').name}.non_decision_time": (
                BatchedTrialParameter(
                    parameters.non_decision_time.detach().cpu().numpy()[condition]
                )
            ),
        },
        realized,
    )


def _generate_continuous(
    trials: CSITrialData,
    vector: np.ndarray,
    *,
    seed: int,
    simulation_time_step: float,
    maximum_decision_time: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from direct_likelihood.native import native_kernels_available

    use_native_lca = native_kernels_available()
    likelihood = ContinuousCSILikelihood(
        SolverConfig(
            ddm_time_step=0.001,
            lca_max_step=0.01,
            native_lca_scan=use_native_lca,
        )
    )
    result = simulate_sequential_trials(
        likelihood,
        _parameters(vector),
        trials,
        seed=seed,
        simulation_time_step=simulation_time_step,
        maximum_decision_time=maximum_decision_time,
        bridge_correction=True,
    )
    return (
        result.trials.choice.detach().cpu().numpy(),
        result.trials.response_time.detach().cpu().numpy(),
        {
            "simulation_time_step": simulation_time_step,
            "bridge_correction": result.bridge_correction,
            "native_lca_scan": use_native_lca,
            "maximum_decision_time": result.maximum_decision_time,
            "mean_decision_time": result.mean_decision_time,
            "realized_parameter_vector": vector,
        },
    )


def _generate_gpu(
    trials: CSITrialData,
    vector: np.ndarray,
    *,
    seed: int,
    time_step: float,
    maximum_decision_time: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    model = _load_model_module()
    steps_per_second = round(1.0 / time_step)
    max_steps = math.ceil(maximum_decision_time / time_step)
    composition = model.make_stab_flex(
        gain=10.0,
        leak=12.0,
        competition=3.0,
        lca_time_step_size=time_step,
        non_decision_time=0.2,
        starting_value=0.0,
        threshold=0.12,
        threshold_collapse=0.0,
        ddm_noise=0.1,
        lca_noise=0.0,
        iti=steps_per_second,
        csi_repeat=0,
        csi_switch=0,
        ddm_time_step_size=time_step,
    )
    outputs = (
        _node(composition, "DECISION_GATE").output_port,
        _node(composition, "RESPONSE_GATE").output_port,
    )
    plan = BatchedCompositionCompiler.compile(
        composition,
        backend="triton",
        outputs=outputs,
        max_steps=max_steps,
    )
    parameter_set, realized = _gpu_parameter_set(
        composition, trials, vector, time_step
    )
    result = plan.run(
        inputs=_gpu_inputs(composition, trials),
        parameter_sets=[parameter_set],
        num_estimates=1,
        seed=seed,
        common_random_numbers=True,
        strict_truncation=True,
    )
    values = np.asarray(result.values[0, 0, :, 0], dtype=float)
    if values.shape != (len(trials), 2):
        raise RuntimeError(f"Unexpected GPU generation shape {values.shape}.")
    condition = trials.condition_index.detach().cpu().numpy()
    is_switch = trials.is_switch.detach().cpu().numpy()
    decision_time = (
        values[:, 1]
        - realized[10:13][condition]
        - is_switch * realized[3]
    )
    return (
        values[:, 0],
        values[:, 1],
        {
            "model_time_step": time_step,
            "max_steps": max_steps,
            "maximum_simulation_time": max_steps * time_step,
            "strict_truncation": True,
            "maximum_decision_time": float(np.max(decision_time)),
            "mean_decision_time": float(np.mean(decision_time)),
            "realized_parameter_vector": realized,
            "csi_rounding_error": float(realized[3] - vector[3]),
        },
    )


def generate_dataset(
    *,
    generator: str,
    source_data: Path,
    subject: int,
    truth_vector: np.ndarray,
    simulation_seed: int,
    continuous_simulation_time_step: float,
    maximum_decision_time: float,
    data_output: Path,
    metadata_output: Path,
) -> dict[str, Any]:
    """Generate one complete stateful sequence and retain its original inputs."""
    source_frame = _subject_frame(source_data, subject)
    trials = CSITrialData.from_csv(source_data, subject, dtype=torch.float64)
    start = time.perf_counter()
    if generator == "continuous":
        choice, response_time, diagnostics = _generate_continuous(
            trials,
            truth_vector,
            seed=simulation_seed,
            simulation_time_step=continuous_simulation_time_step,
            maximum_decision_time=maximum_decision_time,
        )
    else:
        choice, response_time, diagnostics = _generate_gpu(
            trials,
            truth_vector,
            seed=simulation_seed,
            time_step=GENERATOR_TIME_STEPS[generator],
            maximum_decision_time=maximum_decision_time,
        )
    frame = source_frame.copy()
    frame["decision"] = choice
    frame["response_time"] = response_time
    frame["synthetic_generator"] = generator
    frame["synthetic_seed"] = simulation_seed
    if "row_id" not in frame:
        frame["row_id"] = np.arange(len(frame), dtype=int)
    data_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_output, index=False)
    payload = {
        "generator": generator,
        "source_data": source_data.resolve(),
        "subject_nr": subject,
        "simulation_seed": simulation_seed,
        "rows": len(frame),
        "included_rows": int(frame["likelihood_include_mask"].astype(bool).sum()),
        "simulated_accuracy": float(np.mean(choice)),
        "mean_response_time": float(np.mean(response_time)),
        "truth_parameter_names": parameter_names(),
        "truth_parameter_vector": truth_vector,
        "generation_seconds": time.perf_counter() - start,
        "diagnostics": diagnostics,
        "data_output": data_output.resolve(),
    }
    _write_json(payload, metadata_output)
    return payload


def _direct_fit_command(args, data_path: Path, fit_output: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(DIRECT_DRIVER),
        "fit",
        "--data",
        str(data_path),
        "--subject",
        str(args.subject),
        "--ddm-time-step",
        str(args.direct_ddm_time_step),
        "--ddm-spatial-points",
        str(args.direct_spatial_points),
        "--lca-max-step",
        str(args.direct_lca_max_step),
        "--starts",
        str(args.direct_starts),
        "--max-iterations",
        str(args.direct_max_iterations),
        "--random-start-candidates",
        str(args.direct_random_start_candidates),
        "--seed",
        str(args.optimizer_seed),
        "--output",
        str(fit_output),
    ]


def _gpu_fit_command(
    args, data_path: Path, fit_output: Path, time_step: float
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(GPU_FIT_DRIVER),
        "--backend",
        "triton",
        # The generated file contains one subject, while the legacy driver's
        # subject argument is a one-based position rather than subject_nr.
        "--subject-id",
        "1",
        "--data-file",
        str(data_path),
        "--model-time-step",
        str(time_step),
        "--maximum-simulation-time",
        str(args.gpu_fit_maximum_time),
        "--num-estimates",
        str(args.gpu_num_estimates),
        "--max-iterations",
        str(args.gpu_max_iterations),
        "--bins",
        str(args.gpu_bins),
        "--smoothing-sigma",
        str(args.gpu_smoothing_sigma),
        "--pseudocount",
        str(args.gpu_pseudocount),
        "--parameter-batch-size",
        str(args.gpu_parameter_batch_size),
        "--optimizer-seed",
        str(args.optimizer_seed),
        "--simulation-seed",
        str(args.likelihood_seed),
        "--triton-block-size",
        str(args.triton_block_size),
        "--triton-num-warps",
        str(args.triton_num_warps),
        "--deterministic-observed-history",
        "--skip-posterior-predictive",
        "--fit-output",
        str(fit_output),
        "--run-label",
        args.run_label,
    ]
    if args.gpu_fit_strict_truncation:
        command.append("--strict-truncation")
    return command


def _fit_vector(path: Path, fitter: str) -> tuple[np.ndarray, float, dict[str, Any]]:
    if fitter == "direct":
        payload = json.loads(path.read_text())
        vector = np.asarray(payload["parameter_vector"], dtype=float)
        return vector, float(payload["log_likelihood"]), payload
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise RuntimeError(f"GPU fit output must have one row; found {len(frame)}.")
    row = frame.iloc[0]
    vector = _legacy_row_to_physical_vector(row)
    return vector, float(row["log_likelihood"]), row.to_dict()


def _run_cell(args) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    data_output = output_dir / "synthetic_data.csv"
    generation_output = output_dir / "generation.json"
    fit_output = output_dir / (
        "fit.json" if args.fitter == "direct" else "fit.csv"
    )
    result_output = output_dir / "result.json"
    truth = _load_parameter_vector(args.truth_parameters)

    if result_output.exists() and not args.force and not args.dry_run:
        print(result_output)
        return

    if args.fitter == "direct":
        command = _direct_fit_command(args, data_output, fit_output)
    else:
        command = _gpu_fit_command(
            args,
            data_output,
            fit_output,
            FITTER_TIME_STEPS[args.fitter],
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "generator": args.generator,
                    "fitter": args.fitter,
                    "output_dir": str(output_dir),
                    "truth_parameter_vector": truth.tolist(),
                    "fit_command": command,
                },
                indent=2,
            )
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.force or not data_output.exists() or not generation_output.exists():
        generation = generate_dataset(
            generator=args.generator,
            source_data=args.data,
            subject=args.subject,
            truth_vector=truth,
            simulation_seed=args.simulation_seed,
            continuous_simulation_time_step=(
                args.continuous_simulation_time_step
            ),
            maximum_decision_time=args.generation_maximum_time,
            data_output=data_output,
            metadata_output=generation_output,
        )
    else:
        generation = json.loads(generation_output.read_text())

    start = time.perf_counter()
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
    fit_seconds = time.perf_counter() - start
    recovered, log_likelihood, fit_payload = _fit_vector(
        fit_output, args.fitter
    )
    recovery_target = np.asarray(
        generation["diagnostics"].get("realized_parameter_vector", truth),
        dtype=float,
    )
    if recovery_target.shape != truth.shape:
        raise RuntimeError(
            "Generator diagnostics returned an invalid realized parameter vector."
        )
    lower, upper = parameter_bounds()
    error = recovered - recovery_target
    scaled_error = error / (upper - lower)
    payload = {
        "run_label": args.run_label,
        "generator": args.generator,
        "fitter": args.fitter,
        "subject_nr": args.subject,
        "simulation_seed": args.simulation_seed,
        "optimizer_seed": args.optimizer_seed,
        "likelihood_seed": args.likelihood_seed,
        "parameter_names": parameter_names(),
        "requested_truth_parameter_vector": truth,
        "truth_parameter_vector": recovery_target,
        "recovered_parameter_vector": recovered,
        "parameter_error": error,
        "scaled_parameter_error": scaled_error,
        "scaled_parameter_rmse": float(np.sqrt(np.mean(scaled_error**2))),
        "log_likelihood": log_likelihood,
        "fit_seconds": fit_seconds,
        "generation": generation,
        "fit_output": fit_output,
        "fit_metadata": fit_payload,
    }
    _write_json(payload, result_output)
    print(result_output)


def _generate(args) -> None:
    truth = _load_parameter_vector(args.truth_parameters)
    payload = generate_dataset(
        generator=args.generator,
        source_data=args.data,
        subject=args.subject,
        truth_vector=truth,
        simulation_seed=args.simulation_seed,
        continuous_simulation_time_step=args.continuous_simulation_time_step,
        maximum_decision_time=args.generation_maximum_time,
        data_output=args.output,
        metadata_output=args.metadata_output,
    )
    print(json.dumps(payload, indent=2, default=_json_default))


def _result_paths(paths: list[Path]) -> list[Path]:
    results = []
    for path in paths:
        if path.is_dir():
            results.extend(path.rglob("result.json"))
        elif path.is_file():
            results.append(path)
    return sorted(set(result.resolve() for result in results))


def _summarize(args) -> None:
    paths = _result_paths(args.paths)
    if not paths:
        raise SystemExit("No result.json files were found.")
    payloads = [json.loads(path.read_text()) for path in paths]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        groups[(payload["generator"], payload["fitter"])].append(payload)
    summaries = []
    for (generator, fitter), runs in sorted(groups.items()):
        rmse = np.asarray([run["scaled_parameter_rmse"] for run in runs])
        summaries.append(
            {
                "generator": generator,
                "fitter": fitter,
                "runs": len(runs),
                "mean_scaled_parameter_rmse": float(np.mean(rmse)),
                "median_scaled_parameter_rmse": float(np.median(rmse)),
                "maximum_scaled_parameter_rmse": float(np.max(rmse)),
            }
        )
    report = {
        "parameter_names": parameter_names(),
        "result_count": len(payloads),
        "groups": summaries,
        "results": [str(path) for path in paths],
    }
    _write_json(report, args.output)
    if args.csv_output is not None:
        rows = []
        for payload, path in zip(payloads, paths):
            row = {
                "result": str(path),
                "generator": payload["generator"],
                "fitter": payload["fitter"],
                "subject_nr": payload["subject_nr"],
                "simulation_seed": payload["simulation_seed"],
                "optimizer_seed": payload["optimizer_seed"],
                "likelihood_seed": payload["likelihood_seed"],
                "scaled_parameter_rmse": payload["scaled_parameter_rmse"],
                "log_likelihood": payload["log_likelihood"],
                "fit_seconds": payload["fit_seconds"],
            }
            for name, truth, recovered, error in zip(
                payload["parameter_names"],
                payload["truth_parameter_vector"],
                payload["recovered_parameter_vector"],
                payload["scaled_parameter_error"],
            ):
                row[f"truth::{name}"] = truth
                row[f"recovered::{name}"] = recovered
                row[f"scaled_error::{name}"] = error
            rows.append(row)
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.csv_output, index=False)
    print(args.output)


def _add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--generator",
        choices=("continuous", *GENERATOR_TIME_STEPS),
        required=True,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--truth-parameters", type=Path)
    parser.add_argument("--simulation-seed", type=int, default=17)
    parser.add_argument(
        "--continuous-simulation-time-step", type=float, default=0.0005
    )
    parser.add_argument("--generation-maximum-time", type=float, default=12.0)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="Generate one synthetic dataset.")
    _add_generation_arguments(generate)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--metadata-output", type=Path, required=True)
    generate.set_defaults(action=_generate)

    cell = commands.add_parser(
        "run-cell", help="Generate and fit one crossed-recovery matrix cell."
    )
    _add_generation_arguments(cell)
    cell.add_argument(
        "--fitter", choices=("direct", *FITTER_TIME_STEPS), required=True
    )
    cell.add_argument("--output-dir", type=Path, required=True)
    cell.add_argument("--run-label", default="fitting-readiness")
    cell.add_argument("--optimizer-seed", type=int, default=1)
    cell.add_argument("--likelihood-seed", type=int, default=1001)
    cell.add_argument("--force", action="store_true")
    cell.add_argument("--dry-run", action="store_true")
    cell.add_argument("--direct-ddm-time-step", type=float, default=0.001)
    cell.add_argument("--direct-spatial-points", type=int, default=65)
    cell.add_argument("--direct-lca-max-step", type=float, default=0.01)
    cell.add_argument("--direct-starts", type=int, default=2)
    cell.add_argument("--direct-max-iterations", type=int, default=100)
    cell.add_argument("--direct-random-start-candidates", type=int, default=32)
    cell.add_argument("--gpu-num-estimates", type=int, default=100_000)
    cell.add_argument("--gpu-max-iterations", type=int, default=5_000)
    cell.add_argument("--gpu-bins", type=int, default=100)
    cell.add_argument("--gpu-smoothing-sigma", type=float, default=0.5)
    cell.add_argument("--gpu-pseudocount", type=float, default=0.1)
    cell.add_argument("--gpu-parameter-batch-size", type=int, default=11)
    cell.add_argument("--gpu-fit-maximum-time", type=float, default=12.0)
    cell.add_argument("--gpu-fit-strict-truncation", action="store_true")
    cell.add_argument("--triton-block-size", type=int, default=32)
    cell.add_argument("--triton-num-warps", type=int, default=1)
    cell.set_defaults(action=_run_cell)

    summarize = commands.add_parser(
        "summarize", help="Aggregate completed crossed-recovery cells."
    )
    summarize.add_argument("paths", type=Path, nargs="+")
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--csv-output", type=Path)
    summarize.set_defaults(action=_summarize)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    for name in (
        "continuous_simulation_time_step",
        "generation_maximum_time",
        "gpu_fit_maximum_time",
    ):
        if hasattr(args, name) and getattr(args, name) <= 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive.")
    args.action(args)


if __name__ == "__main__":
    main()
