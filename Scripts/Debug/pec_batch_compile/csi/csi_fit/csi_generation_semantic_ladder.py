#!/usr/bin/env python3
"""Decompose CSI generator differences at a fixed one-millisecond clock.

This driver changes one numerical semantic at a time in the independent
sequential generator.  It is intentionally separate from fitting: one command
generates one complete synthetic subject sequence for safe local or Slurm
parallelism.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
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
from direct_likelihood.native import native_kernels_available  # noqa: E402


TRUTH_VECTOR = np.asarray(
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


@dataclass(frozen=True)
class GeneratorVariant:
    simulation_time_step: float
    lca_max_step: float
    lca_integration_method: str
    bridge_correction: bool
    description: str


VARIANTS = {
    "reference-0.5ms-rk4-bridge": GeneratorVariant(
        simulation_time_step=0.0005,
        lca_max_step=0.01,
        lca_integration_method="rk4",
        bridge_correction=True,
        description="Current continuous recovery generator.",
    ),
    "rk4-1ms-bridge": GeneratorVariant(
        simulation_time_step=0.001,
        lca_max_step=0.01,
        lca_integration_method="rk4",
        bridge_correction=True,
        description="Change only the stochastic simulation step to 1 ms.",
    ),
    "rk4-1ms-endpoint": GeneratorVariant(
        simulation_time_step=0.001,
        lca_max_step=0.01,
        lca_integration_method="rk4",
        bridge_correction=False,
        description="At 1 ms, replace continuous crossing with endpoint checks.",
    ),
    "euler-1ms-bridge": GeneratorVariant(
        simulation_time_step=0.001,
        lca_max_step=0.001,
        lca_integration_method="euler",
        bridge_correction=True,
        description="Use PNL's 1 ms Euler/post-step LCA with bridge crossings.",
    ),
    "euler-1ms-endpoint": GeneratorVariant(
        simulation_time_step=0.001,
        lca_max_step=0.001,
        lca_integration_method="euler",
        bridge_correction=False,
        description="Closest CPU mirror of the GPU 1 ms stochastic process.",
    ),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _run(args: argparse.Namespace) -> None:
    variant = VARIANTS[args.variant]
    trials = CSITrialData.from_csv(args.data, args.subject, dtype=torch.float64)
    parameters = ContinuousCSIParameters.from_vector(
        torch.as_tensor(TRUTH_VECTOR, dtype=torch.float64)
    )
    use_native = native_kernels_available()
    likelihood = ContinuousCSILikelihood(
        SolverConfig(
            ddm_time_step=0.001,
            lca_max_step=variant.lca_max_step,
            lca_integration_method=variant.lca_integration_method,
            native_lca_scan=use_native,
        )
    )
    start = time.perf_counter()
    result = simulate_sequential_trials(
        likelihood,
        parameters,
        trials,
        seed=args.seed,
        simulation_time_step=variant.simulation_time_step,
        maximum_decision_time=args.maximum_decision_time,
        bridge_correction=variant.bridge_correction,
    )
    elapsed = time.perf_counter() - start
    choice = result.trials.choice.detach().cpu().numpy()
    response_time = result.trials.response_time.detach().cpu().numpy()
    condition_index = trials.condition_index.detach().cpu().numpy()
    is_switch = trials.is_switch.detach().cpu().numpy()
    decision_time = (
        response_time
        - TRUTH_VECTOR[10:13][condition_index]
        - is_switch * TRUTH_VECTOR[3]
    )
    condition_summary = {}
    for index, condition in enumerate(CONDITIONS):
        selected = condition_index == index
        condition_summary[condition] = {
            "rows": int(np.sum(selected)),
            "mean_response_time": float(np.mean(response_time[selected])),
            "accuracy": float(np.mean(choice[selected])),
        }
    payload = {
        "variant": args.variant,
        "variant_specification": asdict(variant),
        "subject_nr": args.subject,
        "simulation_seed": args.seed,
        "truth_parameter_vector": TRUTH_VECTOR,
        "rows": len(trials),
        "mean_response_time": float(np.mean(response_time)),
        "mean_decision_time": float(np.mean(decision_time)),
        "maximum_decision_time": float(np.max(decision_time)),
        "accuracy": float(np.mean(choice)),
        "condition_summary": condition_summary,
        "response_time": response_time,
        "choice": choice,
        "elapsed_seconds": elapsed,
        "native_lca": use_native,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    print(args.output)
    print(
        f"{args.variant}: mean RT={payload['mean_response_time'] * 1000:.3f} ms, "
        f"accuracy={payload['accuracy']:.5f}, elapsed={elapsed:.3f} s"
    )


def _mapping(args: argparse.Namespace) -> None:
    variants = tuple(VARIANTS)
    offset = args.task_id - 1
    task_count = len(variants) * len(args.seeds)
    if offset < 0 or offset >= task_count:
        raise ValueError(f"task-id must be in [1, {task_count}].")
    variant = variants[offset // len(args.seeds)]
    seed = args.seeds[offset % len(args.seeds)]
    print(json.dumps({"task_id": args.task_id, "variant": variant, "seed": seed}))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Generate one complete sequence.")
    run.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    run.add_argument("--data", type=Path, default=DEFAULT_DATA)
    run.add_argument("--subject", type=int, default=1)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--maximum-decision-time", type=float, default=12.0)
    run.add_argument("--output", type=Path, required=True)
    run.set_defaults(action=_run)

    mapping = commands.add_parser("mapping", help="Print an array-task mapping.")
    mapping.add_argument("--task-id", type=int, required=True)
    mapping.add_argument("--seeds", type=int, nargs="+", required=True)
    mapping.set_defaults(action=_mapping)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if getattr(args, "maximum_decision_time", 1.0) <= 0.0:
        raise ValueError("maximum-decision-time must be positive.")
    args.action(args)


if __name__ == "__main__":
    main()
