#!/usr/bin/env python3
"""Compare continuous and GPU-sampled CSI likelihoods on one parameter surface.

The likelihood values are not directly comparable: the direct implementation
uses continuous first passage and the GPU implementation samples PNL's 10 ms
endpoint process.  This driver therefore compares candidate rankings and also
reports the GPU's across-seed Monte Carlo variability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import optuna
import pandas as pd
import psyneulink as pnl
from scipy import stats
import torch

from psyneulink.core.batched import batched_node_op

from direct_likelihood import (
    CSITrialData,
    ContinuousCSILikelihood,
    ContinuousCSIParameters,
    SolverConfig,
)
from direct_likelihood.model import parameter_bounds
from direct_likelihood.native import native_kernels_available


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data fitting" / "data_to_fit_study3.csv"
DEFAULT_PARAMETERS = (
    HERE / "direct_likelihood" / "fits" / "direct_fit_subject_1_fine_mesh_pnl.csv"
)


@batched_node_op("Drift Rate Value")
def _batched_drift_rate(x0, x1, x2, x3, x4, x5, x6):
    """Triton form of the CSI model's seven-input drift-rate function."""

    a = 1.0 / (1.0 + tl.exp(-((x0 - x1) + 4.0 * x4 - 4.0)))
    b = 1.0 / (1.0 + tl.exp(-((x1 - x0) + 4.0 * x4 - 4.0)))
    c = 1.0 / (1.0 + tl.exp(-((x2 - x3) + 4.0 * x5 - 4.0)))
    d = 1.0 / (1.0 + tl.exp(-((x3 - x2) + 4.0 * x5 - 4.0)))
    positive = 1.0 / (1.0 + tl.exp(-(a - b + c - d)))
    negative = 1.0 / (1.0 + tl.exp(-(-a + b - c + d)))
    return (positive - negative) * x6


def _node(composition, prefix):
    for node in composition.nodes:
        if node.name.startswith(prefix):
            return node
    raise KeyError(f"No node starts with {prefix!r}.")


def _inputs(composition, task, stimulus, correct_response):
    is_switch = np.any(task != np.roll(task, 1, axis=0), axis=1)
    return {
        _node(composition, "Task Input"): [[np.asarray(value)] for value in task],
        _node(composition, "Stimulus Input"): [
            [np.asarray(value)] for value in stimulus
        ],
        _node(composition, "Correct Response"): [
            [np.asarray(value)] for value in correct_response
        ],
        _node(composition, "Cue Stimulus Interval"): [
            [np.asarray([float(value)])] for value in is_switch
        ],
        _node(composition, "Threshold Mechanism"): [
            [np.asarray([0.0])] for _ in task
        ],
    }


def _legacy_row(vector):
    parameters = ContinuousCSIParameters.from_vector(
        torch.as_tensor(vector, dtype=torch.float64)
    )
    return parameters.as_legacy_dict()


def _candidate_surface(base_vector, count, local_fraction, seed):
    """Generate reproducible local axis probes and joint perturbations."""

    if count < 1:
        raise ValueError("candidate count must be positive.")
    lower, upper = parameter_bounds()
    span = upper - lower
    # These are the resolutions of the legacy PNL search coordinates after
    # converting collapse increments from per-10-ms step to units per second.
    steps = np.asarray(
        [0.1, 0.1, 0.1, 0.01, 0.0005, 0.0005, 0.0005,
         0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        dtype=float,
    )
    base = np.clip(np.asarray(base_vector, dtype=float), lower, upper)
    records = [(base, "base")]
    seen = {tuple(np.round(base, decimals=12))}

    def add(delta, kind):
        candidate = np.clip(base + delta, lower, upper)
        # Perturb in whole legacy search steps while retaining an unsnapped
        # direct-fit center as the first candidate.
        candidate = np.clip(
            base + np.round((candidate - base) / steps) * steps,
            lower,
            upper,
        )
        key = tuple(np.round(candidate, decimals=12))
        if key not in seen:
            seen.add(key)
            records.append((candidate, kind))

    for fraction in (0.5 * local_fraction, local_fraction):
        for dimension in range(base.size):
            delta = np.zeros_like(base)
            delta[dimension] = fraction * span[dimension]
            add(delta, f"axis+:{dimension}:{fraction:g}")
            add(-delta, f"axis-:{dimension}:{fraction:g}")
            if len(records) >= count:
                break
        if len(records) >= count:
            break

    rng = np.random.default_rng(seed)
    attempts = 0
    while len(records) < count:
        fraction = rng.uniform(0.25 * local_fraction, local_fraction)
        direction = rng.uniform(-1.0, 1.0, size=base.size)
        add(direction * fraction * span, f"joint:{fraction:g}")
        attempts += 1
        if attempts > 100 * count:
            raise RuntimeError("Could not generate enough distinct candidates.")

    rows = []
    vectors = []
    for index, (vector, kind) in enumerate(records[:count]):
        rows.append({"candidate_id": index, "candidate_kind": kind, **_legacy_row(vector)})
        vectors.append(vector)
    return pd.DataFrame(rows), np.asarray(vectors)


def _build_gpu_problem(data, subject, estimates, bins, smoothing, pseudocount,
                       batch_size, seed, max_steps, time_step=0.01):
    if not np.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("GPU time_step must be finite and positive.")
    iti_steps = int(round(1.0 / time_step))
    if not np.isclose(iti_steps * time_step, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("GPU time_step must divide the one-second ITI exactly.")
    model_directory = HERE / "data fitting"
    sys.path.insert(0, str(model_directory))
    try:
        from expectation_model_study2_study3 import make_stab_flex
    finally:
        sys.path.pop(0)

    frame = pd.read_csv(data)
    frame["decision"] = frame["decision"].astype("category")
    frame["sequence"] = frame["sequence"].astype("category")
    frame = frame[
        (frame.subject_nr == subject)
        & frame.sequence.isin(["RealRare", "RealFrequent", "NoInstruction"])
    ].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No CSI rows found for subject {subject}.")

    task = frame[["T1", "T2"]].to_numpy()
    stimulus = frame[["S1", "S2", "S3", "S4"]].to_numpy()
    correct_response = frame["correct_response"].to_numpy()
    include = frame["likelihood_include_mask"].to_numpy(dtype=bool)
    observed = frame[["decision", "response_time", "sequence"]]

    composition = make_stab_flex(
        gain=10.0,
        leak=12.0,
        competition=3.0,
        iti=iti_steps,
        csi_switch=0,
        threshold=0.06,
        threshold_collapse=0.0,
        non_decision_time=0.3,
        lca_time_step_size=time_step,
        ddm_time_step_size=time_step,
        lca_noise=0.0,
        ddm_noise=0.1,
    )
    cue = _node(composition, "Cue Stimulus Interval")
    lca = _node(composition, "Task Activations [C1, C2]")
    threshold = _node(composition, "Threshold Mechanism")
    ddm = _node(composition, "DDM")
    decision_gate = _node(composition, "DECISION_GATE")
    response_gate = _node(composition, "RESPONSE_GATE")
    fit_parameters = {
        ("gain", lca): np.linspace(5.0, 35.0, 301),
        ("slope", cue): np.linspace(0.0, 0.30 / time_step, 31),
        ("intercept", threshold): np.linspace(0.05, 0.25, 401),
        ("offset-integrator_function", threshold): np.linspace(
            -0.3 * time_step, 0.0, 301
        ),
        ("non_decision_time", ddm): np.linspace(0.1, 0.4, 301),
    }
    pec = pnl.ParameterEstimationComposition(
        name="CSI likelihood surface comparison",
        nodes=composition,
        parameters=fit_parameters,
        depends_on={
            ("gain", lca): "sequence",
            ("intercept", threshold): "sequence",
            ("offset-integrator_function", threshold): "sequence",
            ("non_decision_time", ddm): "sequence",
        },
        outcome_variables=[
            decision_gate.output_ports[0],
            response_gate.output_ports[0],
        ],
        data=observed,
        likelihood_include_mask=include,
        optimization_function=pnl.PECOptimizationFunction(
            method=optuna.samplers.CmaEsSampler(
                popsize=batch_size,
                seed=1,
            ),
            max_iterations=1,
            conditioned_likelihood=False,
            deterministic_history_likelihood=True,
            batched_bins=bins,
            batched_smoothing_sigma=smoothing,
            batched_pseudocount=pseudocount,
            batched_categorical_cardinalities=[2],
            batched_seed=seed,
            batched_backend="triton",
            batched_parameter_batch_size=batch_size,
            batched_max_steps=max_steps,
            batched_triton_launch_options={"block_size": 128, "num_warps": 4},
        ),
        num_estimates=estimates,
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    pec.controller.function.parameters.save_values.set(True)
    return pec, _inputs(composition, task, stimulus, correct_response)


def _fit_values(frame, fit_names):
    def canonical(name):
        mechanism, separator, parameter = str(name).partition(".")
        mechanism = re.sub(r"-\d+$", "", mechanism)
        return f"{mechanism}{separator}{parameter}"

    columns = {canonical(column): column for column in frame}
    missing = [name for name in fit_names if canonical(name) not in columns]
    if missing:
        raise ValueError(f"Candidate table is missing PNL parameters: {missing}")
    return [
        tuple(float(row[columns[canonical(name)]]) for name in fit_names)
        for _, row in frame.iterrows()
    ]


def _rescale_legacy_time_step(frame, time_step, *, source_time_step=0.01):
    """Preserve physical CSI duration and collapse rate at a new step size."""

    if not np.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("time_step must be finite and positive.")
    result = frame.copy()
    csi_column = "Cue Stimulus Interval.slope"
    if csi_column not in result:
        raise ValueError(f"Parameter table is missing {csi_column!r}.")
    result[csi_column] *= source_time_step / time_step
    collapse_columns = [
        column
        for column in result
        if column.startswith("Threshold Mechanism.offset-integrator_function[")
    ]
    if len(collapse_columns) != 3:
        raise ValueError(
            "Parameter table must contain three condition-specific boundary "
            "collapse columns."
        )
    result[collapse_columns] *= time_step / source_time_step
    return result


def _score_direct(vectors, trials, *, rt_bins):
    native = native_kernels_available()
    likelihood = ContinuousCSILikelihood(
        SolverConfig(
            ddm_time_step=0.001,
            ddm_spatial_points=65,
            lca_max_step=0.01,
            rt_bin_count=rt_bins,
            native_lca_scan=native,
            native_ddm_forward=native,
        )
    )
    scores = []
    started = time.perf_counter()
    with torch.no_grad():
        for vector in vectors:
            result = likelihood.score_vector(
                torch.as_tensor(vector, dtype=torch.float64), trials
            )
            scores.append(float(result.log_likelihood))
    return np.asarray(scores), time.perf_counter() - started


def _rank_metrics(reference, sampled, *, top_count=10):
    finite = np.isfinite(reference) & np.isfinite(sampled)
    reference = np.asarray(reference)[finite]
    sampled = np.asarray(sampled)[finite]
    if reference.size < 2:
        return {"finite_candidates": int(reference.size)}
    reference_order = np.argsort(-reference)
    sampled_order = np.argsort(-sampled)
    reference_rank = np.empty(reference.size, dtype=int)
    sampled_rank = np.empty(sampled.size, dtype=int)
    reference_rank[reference_order] = np.arange(reference.size)
    sampled_rank[sampled_order] = np.arange(sampled.size)
    k = min(top_count, reference.size)
    return {
        "finite_candidates": int(reference.size),
        "spearman": float(stats.spearmanr(reference, sampled).statistic),
        "kendall_tau": float(stats.kendalltau(reference, sampled).statistic),
        "reference_best_candidate": int(np.flatnonzero(finite)[reference_order[0]]),
        "sampled_best_candidate": int(np.flatnonzero(finite)[sampled_order[0]]),
        "reference_best_rank_under_sampled": int(sampled_rank[reference_order[0]] + 1),
        "sampled_best_rank_under_reference": int(reference_rank[sampled_order[0]] + 1),
        f"top_{k}_overlap": int(
            len(set(reference_order[:k]).intersection(sampled_order[:k]))
        ),
    }


def _score_gpu(pec, inputs, candidate_values, seeds, batch_size):
    fit_function = pec.controller.function
    # Public scoring both installs the node-keyed inputs and performs the one-
    # time compilation. Subsequent batches reuse the cached plan.
    fit_function.batched_seed = seeds[0]
    pec.log_likelihood(*candidate_values[0], inputs=inputs)
    all_scores = []
    durations = []
    for seed in seeds:
        fit_function.batched_seed = seed
        objective = fit_function._batched_objective_func()
        evaluate = objective._batched_parameter_sets
        seed_scores = []
        started = time.perf_counter()
        for begin in range(0, len(candidate_values), batch_size):
            seed_scores.extend(
                evaluate(candidate_values[begin:begin + batch_size]).tolist()
            )
        durations.append(time.perf_counter() - started)
        all_scores.append(seed_scores)
    return np.asarray(all_scores), durations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--base-parameters", type=Path, default=DEFAULT_PARAMETERS)
    parser.add_argument("--candidates", type=int, default=121)
    parser.add_argument("--candidate-seed", type=int, default=20260901)
    parser.add_argument("--local-fraction", type=float, default=0.05)
    parser.add_argument("--estimates", type=int, default=100_000)
    parser.add_argument("--gpu-seeds", nargs="+", type=int, default=[21, 22, 23, 24, 25])
    parser.add_argument("--batch-size", type=int, default=11)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--pseudocount", type=float, default=0.0)
    parser.add_argument("--gpu-time-step", type=float, default=0.01)
    parser.add_argument("--max-time", type=float, default=12.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Explicit GPU cap; by default ceil(max-time / gpu-time-step).",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    if args.base_parameters.suffix.lower() != ".csv":
        raise ValueError("--base-parameters must be a legacy one-row CSV.")
    base_frame = pd.read_csv(args.base_parameters)
    if len(base_frame) != 1:
        raise ValueError("--base-parameters must contain exactly one row.")
    base = ContinuousCSIParameters.from_legacy_row(base_frame.iloc[0]).vector().numpy()
    candidates, vectors = _candidate_surface(
        base, args.candidates, args.local_fraction, args.candidate_seed
    )

    trials = CSITrialData.from_csv(
        args.data, args.subject, dtype=torch.float64, device="cpu"
    )
    direct_scores, direct_seconds = _score_direct(vectors, trials, rt_bins=None)
    binned_scores, binned_seconds = _score_direct(vectors, trials, rt_bins=args.bins)

    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else int(np.ceil(args.max_time / args.gpu_time_step))
    )
    pec, inputs = _build_gpu_problem(
        args.data,
        args.subject,
        args.estimates,
        args.bins,
        args.smoothing,
        args.pseudocount,
        args.batch_size,
        args.gpu_seeds[0],
        max_steps,
        args.gpu_time_step,
    )
    gpu_candidates = _rescale_legacy_time_step(
        candidates, args.gpu_time_step
    )
    values = _fit_values(
        gpu_candidates, pec.controller.function.fit_param_names
    )
    gpu_scores, gpu_seconds = _score_gpu(
        pec, inputs, values, args.gpu_seeds, args.batch_size
    )
    gpu_mean = np.mean(gpu_scores, axis=0)
    gpu_sd = np.std(gpu_scores, axis=0, ddof=1) if len(args.gpu_seeds) > 1 else np.zeros(args.candidates)

    scores = candidates.copy()
    scores["direct_continuous"] = direct_scores
    scores["direct_continuous_binned"] = binned_scores
    for seed, seed_scores in zip(args.gpu_seeds, gpu_scores):
        scores[f"gpu_seed_{seed}"] = seed_scores
    scores["gpu_mean"] = gpu_mean
    scores["gpu_sd"] = gpu_sd

    per_seed = {
        str(seed): {
            "continuous": _rank_metrics(direct_scores, seed_scores),
            "continuous_binned": _rank_metrics(binned_scores, seed_scores),
        }
        for seed, seed_scores in zip(args.gpu_seeds, gpu_scores)
    }
    seed_correlations = []
    for first in range(len(args.gpu_seeds)):
        for second in range(first + 1, len(args.gpu_seeds)):
            seed_correlations.append(
                float(stats.spearmanr(gpu_scores[first], gpu_scores[second]).statistic)
            )
    summary = {
        "subject": args.subject,
        "rows": len(trials),
        "included_rows": int(trials.include.sum()),
        "candidates": args.candidates,
        "local_fraction_of_parameter_range": args.local_fraction,
        "gpu_estimates_per_candidate": args.estimates,
        "gpu_time_step": args.gpu_time_step,
        "gpu_max_steps": max_steps,
        "gpu_seeds": args.gpu_seeds,
        "gpu_bins": args.bins,
        "gpu_smoothing": args.smoothing,
        "gpu_pseudocount": args.pseudocount,
        "continuous_vs_gpu_mean": _rank_metrics(direct_scores, gpu_mean),
        "continuous_binned_vs_gpu_mean": _rank_metrics(binned_scores, gpu_mean),
        "per_gpu_seed": per_seed,
        "gpu_seed_pair_spearman": {
            "minimum": float(np.min(seed_correlations)) if seed_correlations else 1.0,
            "median": float(np.median(seed_correlations)) if seed_correlations else 1.0,
            "maximum": float(np.max(seed_correlations)) if seed_correlations else 1.0,
        },
        "gpu_monte_carlo_sd": {
            "median": float(np.median(gpu_sd)),
            "maximum": float(np.max(gpu_sd)),
        },
        "timings_seconds": {
            "direct_continuous_all_candidates": direct_seconds,
            "direct_continuous_binned_all_candidates": binned_seconds,
            "gpu_by_seed_after_compilation": gpu_seconds,
        },
        "interpretation": (
            "Compare rankings, not raw values: GPU samples a discrete endpoint "
            f"process at dt={args.gpu_time_step:g} s; direct scores use "
            "continuous first passage."
        ),
    }

    prefix = args.output_prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    scores_path = prefix.with_suffix(".scores.csv")
    summary_path = prefix.with_suffix(".summary.json")
    scores.to_csv(scores_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(scores_path)
    print(summary_path)


if __name__ == "__main__":
    main()
