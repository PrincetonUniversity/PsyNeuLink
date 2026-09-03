#!/usr/bin/env python3
"""Audit CSI Triton simulations against the original LLVM composition.

The LLVM and Triton backends use different random-number stream layouts, so
stochastic trajectories are not expected to match lane by lane.  This audit
first checks exact deterministic execution, then compares response accuracy
and response-time distributions over independently initialized full-sequence
replicates.  It deliberately bypasses both likelihood implementations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

import psyneulink as pnl
from psyneulink.core.batched import BatchedCompositionCompiler, batched_node_op
from psyneulink.core.globals.utilities import set_global_seed


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "data fitting" / "expectation_model_study2_study3.py"


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


@dataclass(frozen=True)
class Scenario:
    name: str
    gain: float
    csi_switch: float
    threshold: float
    collapse: float
    non_decision_time: float


SCENARIOS = (
    Scenario(
        name="baseline_collapsing",
        gain=10.0,
        csi_switch=10.0,
        threshold=0.12,
        collapse=-0.001,
        non_decision_time=0.20,
    ),
    Scenario(
        name="low_gain_constant_boundary",
        gain=6.0,
        csi_switch=0.0,
        threshold=0.16,
        collapse=0.0,
        non_decision_time=0.25,
    ),
    Scenario(
        name="high_gain_fast_collapse",
        gain=25.0,
        csi_switch=20.0,
        threshold=0.10,
        collapse=-0.001,
        non_decision_time=0.35,
    ),
)


BASE_MODEL_OPTIONS = {
    "gain": 10.0,
    "leak": 12.0,
    "competition": 3.0,
    "lca_time_step_size": 0.01,
    "non_decision_time": 0.20,
    "starting_value": 0.0,
    "threshold": 0.12,
    "threshold_collapse": -0.001,
    "ddm_noise": 0.1,
    "lca_noise": 0.0,
    "iti": 100,
    "csi_repeat": 0,
    "csi_switch": 10,
    "ddm_time_step_size": 0.01,
}


def _load_model_module():
    specification = importlib.util.spec_from_file_location(
        "_csi_gpu_llvm_audit_model", MODEL_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load the CSI model from {MODEL_PATH}.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _node(composition, base_name):
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


def _trial_sequence(model, trial_count: int, seed: int):
    tasks, stimuli, correct, _ = model.generate_mixed_task_sequence(
        trial_count,
        switch_frequency=0.5,
        incongruence_frequency=0.5,
        seed=seed,
    )
    task = np.asarray(tasks, dtype=float)
    stimulus = np.asarray(stimuli, dtype=float)
    correct_response = np.asarray(correct, dtype=float).reshape(-1)
    cue = np.asarray(
        [
            0.0 if np.array_equal(task[index], task[index - 1]) else 1.0
            for index in range(trial_count)
        ],
        dtype=float,
    )
    congruent = np.sign(stimulus[:, 0] - stimulus[:, 1]) == np.sign(
        stimulus[:, 2] - stimulus[:, 3]
    )
    return {
        "task": task,
        "stimulus": stimulus,
        "correct_response": correct_response,
        "cue": cue,
        "congruent": congruent,
    }


def _inputs(composition, trials):
    count = len(trials["cue"])
    return {
        _node(composition, "Task Input"): trials["task"],
        _node(composition, "Stimulus Input"): trials["stimulus"],
        _node(composition, "Correct Response"): trials[
            "correct_response"
        ].reshape(-1, 1),
        _node(composition, "Cue Stimulus Interval"): trials["cue"].reshape(
            -1, 1
        ),
        _node(composition, "Threshold Mechanism"): np.zeros((count, 1)),
    }


def _outputs(composition):
    return (
        _node(composition, "DECISION_GATE").output_port,
        _node(composition, "RESPONSE_GATE").output_port,
    )


def _selected_composition_results(composition, outputs):
    result_indices = []
    for output in outputs:
        matches = tuple(
            index
            for index, cim_input in enumerate(
                composition.output_CIM.input_ports
            )
            if any(
                projection.sender is output
                for projection in cim_input.path_afferents
            )
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"Could not locate output {output.full_name!r} in CSI results."
            )
        result_indices.append(matches[0])
    return np.asarray(
        [
            [
                float(np.asarray(trial[index]).reshape(-1)[0])
                for index in result_indices
            ]
            for trial in composition.results
        ]
    )


def _scenario_parameter_sets(composition, scenarios):
    lca = _node(composition, "Task Activations [C1, C2]")
    cue = _node(composition, "Cue Stimulus Interval")
    threshold = _node(composition, "Threshold Mechanism")
    ddm = _node(composition, "DDM")
    return [
        {
            f"{lca.name}.gain": scenario.gain,
            f"{cue.name}.slope": scenario.csi_switch,
            f"{threshold.name}.intercept": scenario.threshold,
            f"{threshold.name}.offset-integrator_function": scenario.collapse,
            f"{ddm.name}.non_decision_time": scenario.non_decision_time,
        }
        for scenario in scenarios
    ]


def _gpu_simulations(
    model,
    trials,
    scenarios,
    *,
    replicates: int,
    seed: int,
    max_steps: int,
    ddm_noise: float,
):
    composition = model.make_stab_flex(
        **{**BASE_MODEL_OPTIONS, "ddm_noise": ddm_noise}
    )
    outputs = _outputs(composition)
    start = time.perf_counter()
    plan = BatchedCompositionCompiler.compile(
        composition,
        backend="triton",
        outputs=outputs,
        max_steps=max_steps,
    )
    parameter_sets = _scenario_parameter_sets(composition, scenarios)
    compiled_seconds = time.perf_counter() - start
    start = time.perf_counter()
    result = plan.run(
        inputs=_inputs(composition, trials),
        parameter_sets=parameter_sets,
        num_estimates=replicates,
        seed=seed,
        common_random_numbers=True,
        strict_truncation=True,
    )
    run_seconds = time.perf_counter() - start
    # Runtime layout is [parameter, subject, trial, estimate, outcome].
    values = np.transpose(result.values[:, 0], (0, 2, 1, 3))
    expected = (len(scenarios), replicates, len(trials["cue"]), 2)
    if values.shape != expected:
        raise RuntimeError(
            f"Unexpected Triton output shape {values.shape}; expected {expected}."
        )
    return values.astype(float), {
        "compile_seconds": compiled_seconds,
        "run_seconds": run_seconds,
    }


def _llvm_simulations(
    model,
    trials,
    scenarios,
    *,
    replicates: int,
    seed: int,
    ddm_noise: float,
):
    values = np.empty(
        (len(scenarios), replicates, len(trials["cue"]), 2), dtype=float
    )
    scenario_seconds = {}
    for scenario_index, scenario in enumerate(scenarios):
        # LLVM owns a sequential random stream. Resetting the Composition before
        # each run restores the canonical LCA/DDM state while the stream moves
        # forward to fresh draws for the next independent sequence replicate.
        set_global_seed(seed + scenario_index)
        composition = model.make_stab_flex(
            **{
                **BASE_MODEL_OPTIONS,
                "gain": scenario.gain,
                "csi_switch": scenario.csi_switch,
                "threshold": scenario.threshold,
                "threshold_collapse": scenario.collapse,
                "non_decision_time": scenario.non_decision_time,
                "ddm_noise": ddm_noise,
            }
        )
        inputs = _inputs(composition, trials)
        outputs = _outputs(composition)
        start = time.perf_counter()
        for replicate in range(replicates):
            composition.reset(clear_results=True)
            composition.run(
                inputs=inputs,
                execution_mode=pnl.ExecutionMode.LLVMRun,
            )
            values[scenario_index, replicate] = (
                _selected_composition_results(composition, outputs)
            )
        scenario_seconds[scenario.name] = time.perf_counter() - start
    return values, {
        "run_seconds": float(sum(scenario_seconds.values())),
        "scenario_seconds": scenario_seconds,
    }


def _standardized_difference(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    difference = float(np.mean(left) - np.mean(right))
    standard_error = math.sqrt(
        float(np.var(left, ddof=1)) / len(left)
        + float(np.var(right, ddof=1)) / len(right)
    )
    if standard_error == 0.0:
        z_score = 0.0 if difference == 0.0 else math.inf
    else:
        z_score = difference / standard_error
    return difference, standard_error, z_score


def _group_masks(trials):
    return {
        "all": np.ones(len(trials["cue"]), dtype=bool),
        "repeat": trials["cue"] == 0.0,
        "switch": trials["cue"] != 0.0,
        "congruent": trials["congruent"],
        "incongruent": ~trials["congruent"],
        "correct_positive": trials["correct_response"] > 0.0,
        "correct_negative": trials["correct_response"] < 0.0,
    }


def _rt_grid_error(values, trials, scenario):
    csi_seconds = trials["cue"] * scenario.csi_switch * 0.01
    decision_steps = (
        values[..., 1]
        - scenario.non_decision_time
        - csi_seconds[None, :]
    ) / 0.01
    return float(np.max(np.abs(decision_steps - np.round(decision_steps))))


def _summarize_scenario(gpu, llvm, trials, scenario):
    # The model uses ``correct_response`` to flip the drift into a
    # correctness-aligned frame. DECISION_OUTCOME is consequently 1 for a
    # correct response and 0 for an error, regardless of response direction.
    gpu_accuracy = gpu[..., 0] > 0.5
    llvm_accuracy = llvm[..., 0] > 0.5
    groups: dict[str, Any] = {}
    passed = True
    for name, mask in _group_masks(trials).items():
        if not np.any(mask):
            continue
        gpu_accuracy_by_replicate = np.mean(gpu_accuracy[:, mask], axis=1)
        llvm_accuracy_by_replicate = np.mean(llvm_accuracy[:, mask], axis=1)
        accuracy_difference, accuracy_se, accuracy_z = (
            _standardized_difference(
                gpu_accuracy_by_replicate, llvm_accuracy_by_replicate
            )
        )
        gpu_rt_by_replicate = np.mean(gpu[:, mask, 1], axis=1)
        llvm_rt_by_replicate = np.mean(llvm[:, mask, 1], axis=1)
        rt_difference, rt_se, rt_z = _standardized_difference(
            gpu_rt_by_replicate, llvm_rt_by_replicate
        )
        gpu_rt = gpu[:, mask, 1].reshape(-1)
        llvm_rt = llvm[:, mask, 1].reshape(-1)
        ks = ks_2samp(gpu_rt, llvm_rt)
        group_passed = (
            abs(accuracy_difference) <= max(0.015, 4.0 * accuracy_se)
            and abs(rt_difference) <= max(0.01, 4.0 * rt_se)
        )
        passed = passed and group_passed
        groups[name] = {
            "trials": int(np.sum(mask)),
            "gpu_accuracy": float(np.mean(gpu_accuracy[:, mask])),
            "llvm_accuracy": float(np.mean(llvm_accuracy[:, mask])),
            "accuracy_difference": accuracy_difference,
            "accuracy_standard_error": accuracy_se,
            "accuracy_z": accuracy_z,
            "gpu_mean_rt": float(np.mean(gpu_rt)),
            "llvm_mean_rt": float(np.mean(llvm_rt)),
            "mean_rt_difference": rt_difference,
            "mean_rt_standard_error": rt_se,
            "mean_rt_z": rt_z,
            "gpu_rt_quantiles": np.quantile(
                gpu_rt, [0.1, 0.5, 0.9]
            ).tolist(),
            "llvm_rt_quantiles": np.quantile(
                llvm_rt, [0.1, 0.5, 0.9]
            ).tolist(),
            "rt_wasserstein": float(wasserstein_distance(gpu_rt, llvm_rt)),
            "rt_ks_statistic": float(ks.statistic),
            "rt_ks_pvalue_descriptive": float(ks.pvalue),
            "passes_mean_checks": group_passed,
        }

    trial_accuracy_difference = np.mean(gpu_accuracy, axis=0) - np.mean(
        llvm_accuracy, axis=0
    )
    trial_rt_difference = np.mean(gpu[..., 1], axis=0) - np.mean(
        llvm[..., 1], axis=0
    )
    return {
        "scenario": asdict(scenario),
        "passes_mean_checks": passed,
        "gpu_decisions": sorted(np.unique(gpu[..., 0]).tolist()),
        "llvm_decisions": sorted(np.unique(llvm[..., 0]).tolist()),
        "gpu_rt_grid_max_error_steps": _rt_grid_error(gpu, trials, scenario),
        "llvm_rt_grid_max_error_steps": _rt_grid_error(llvm, trials, scenario),
        "maximum_trial_accuracy_difference": float(
            np.max(np.abs(trial_accuracy_difference))
        ),
        "maximum_trial_mean_rt_difference": float(
            np.max(np.abs(trial_rt_difference))
        ),
        "groups": groups,
    }


def _json_default(value):
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def run_audit(args):
    model = _load_model_module()
    trials = _trial_sequence(model, args.trials, args.sequence_seed)
    selected = set(args.scenario or ())
    scenarios = tuple(
        scenario
        for scenario in SCENARIOS
        if not selected or scenario.name in selected
    )
    unknown = selected.difference(scenario.name for scenario in SCENARIOS)
    if unknown:
        raise ValueError(f"Unknown scenarios: {sorted(unknown)}")

    deterministic_gpu, deterministic_gpu_timing = _gpu_simulations(
        model,
        trials,
        scenarios,
        replicates=1,
        seed=args.gpu_seed,
        max_steps=args.max_steps,
        ddm_noise=0.0,
    )
    deterministic_llvm, deterministic_llvm_timing = _llvm_simulations(
        model,
        trials,
        scenarios,
        replicates=1,
        seed=args.llvm_seed,
        ddm_noise=0.0,
    )
    deterministic_error = np.max(
        np.abs(deterministic_gpu - deterministic_llvm), axis=(1, 2, 3)
    )

    gpu, gpu_timing = _gpu_simulations(
        model,
        trials,
        scenarios,
        replicates=args.gpu_replicates,
        seed=args.gpu_seed,
        max_steps=args.max_steps,
        ddm_noise=0.1,
    )
    llvm, llvm_timing = _llvm_simulations(
        model,
        trials,
        scenarios,
        replicates=args.llvm_replicates,
        seed=args.llvm_seed,
        ddm_noise=0.1,
    )
    summaries = [
        _summarize_scenario(
            gpu[index], llvm[index], trials, scenario
        )
        for index, scenario in enumerate(scenarios)
    ]
    deterministic_passed = bool(np.all(deterministic_error <= 1.0e-5))
    payload = {
        "purpose": (
            "Compare the CSI Triton and original LLVM simulators directly; "
            "neither likelihood calculation is exercised."
        ),
        "model_source": MODEL_PATH,
        "base_model_options": BASE_MODEL_OPTIONS,
        "trial_count": args.trials,
        "sequence_seed": args.sequence_seed,
        "gpu_replicates": args.gpu_replicates,
        "llvm_replicates": args.llvm_replicates,
        "gpu_seed": args.gpu_seed,
        "llvm_seed": args.llvm_seed,
        "max_steps": args.max_steps,
        "deterministic_maximum_absolute_errors": {
            scenario.name: float(deterministic_error[index])
            for index, scenario in enumerate(scenarios)
        },
        "deterministic_passed": deterministic_passed,
        "statistical_passed": all(
            summary["passes_mean_checks"] for summary in summaries
        ),
        "timings": {
            "deterministic_gpu": deterministic_gpu_timing,
            "deterministic_llvm": deterministic_llvm_timing,
            "stochastic_gpu": gpu_timing,
            "stochastic_llvm": llvm_timing,
        },
        "scenarios": summaries,
        "interpretation": {
            "mean_check_rule": (
                "For each stratum, absolute GPU-minus-LLVM accuracy must be "
                "within max(0.015, 4 cluster-SE), and mean RT within "
                "max(0.010 seconds, 4 cluster-SE)."
            ),
            "cluster_unit": "one independently reset full-sequence replicate",
            "ks_pvalues": (
                "Descriptive only: trial observations within a sequence are "
                "history-dependent and are not independent KS samples."
            ),
        },
    }
    return payload


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--gpu-replicates", type=int, default=2_000)
    parser.add_argument("--llvm-replicates", type=int, default=2_000)
    parser.add_argument("--sequence-seed", type=int, default=17)
    parser.add_argument("--gpu-seed", type=int, default=29)
    parser.add_argument("--llvm-seed", type=int, default=31)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5_000,
        help=(
            "GPU execution cap. The audit default exceeds the 1,200-step "
            "fitting cap so constant-boundary tail trials are not censored."
        ),
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(scenario.name for scenario in SCENARIOS),
        help="Run only this scenario; repeat to select multiple scenarios.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--full-json",
        action="store_true",
        help="Print the complete JSON report instead of a compact summary.",
    )
    return parser


def _print_summary(payload):
    print(
        "deterministic_passed="
        f"{payload['deterministic_passed']} "
        f"statistical_passed={payload['statistical_passed']}"
    )
    for summary in payload["scenarios"]:
        overall = summary["groups"]["all"]
        name = summary["scenario"]["name"]
        deterministic_error = payload[
            "deterministic_maximum_absolute_errors"
        ][name]
        print(
            f"{name}: deterministic_max_error={deterministic_error:.3g}, "
            f"accuracy GPU/LLVM={overall['gpu_accuracy']:.4f}/"
            f"{overall['llvm_accuracy']:.4f} "
            f"(difference={overall['accuracy_difference']:+.4f}, "
            f"z={overall['accuracy_z']:+.2f}), mean RT GPU/LLVM="
            f"{overall['gpu_mean_rt']:.4f}/{overall['llvm_mean_rt']:.4f}s "
            f"(difference={overall['mean_rt_difference']:+.4f}s, "
            f"z={overall['mean_rt_z']:+.2f}), "
            f"RT Wasserstein={overall['rt_wasserstein']:.4f}s"
        )


def main():
    args = make_parser().parse_args()
    if args.trials < 8 or args.trials % 8:
        raise SystemExit("--trials must be a positive multiple of eight.")
    if args.gpu_replicates < 2 or args.llvm_replicates < 2:
        raise SystemExit("Stochastic replicate counts must be at least two.")
    payload = run_audit(args)
    rendered = json.dumps(payload, indent=2, default=_json_default)
    if args.full_json:
        print(rendered)
    else:
        _print_summary(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if not payload["deterministic_passed"] or not payload["statistical_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
