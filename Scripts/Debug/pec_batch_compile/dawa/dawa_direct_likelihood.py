"""Prototype DAWA direct likelihood, gradients, and bounded gradient fitting.

Run `validate` for native-model parity, Monte Carlo, quadrature refinement, and
finite-difference gradient checks. `score` and `fit` use a documented fixed
history approximation for empirical RTs; see dawa_likelihood/README.md.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from dawa_likelihood import DEFAULT_PARAMETERS, PARAMETER_NAMES, SolverConfig, sequence_likelihood
from dawa_likelihood.model import PARAMETER_BOUNDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "score", "fit"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--points", type=int, default=97)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("dawa_lca_model") / "flanker_data_part1.csv")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--parameters", type=float, nargs=7, default=DEFAULT_PARAMETERS, metavar="VALUE")
    parser.add_argument("--history-reference-ndt", type=float, default=.2,
                        help="Freeze latent history steps by rounding (observed RT - this value)/0.01")
    parser.add_argument("--measurement-sd", type=float, default=.01)
    parser.add_argument("--resolution", type=float, default=.001)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--estimates", type=int, default=100000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.threads, args.trials, args.max_steps, args.iterations, args.estimates) < 1:
        parser.error("Thread, trial, step, iteration, and estimate counts must be positive")
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; select --device cpu")

    def save(report):
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")

    report = {"device": args.device, "torch": torch.__version__, "dtype": "float64"}
    if args.device == "cuda":
        report["gpu"] = torch.cuda.get_device_name()
    if args.command == "validate":
        from dawa_likelihood.validation import native_replay_error, finite_difference_check, refinement_check, gradient_grid_check
        print("Checking deterministic multi-trial replay...", flush=True)
        errors = [native_replay_error(), native_replay_error((.4, .2, -.3, 13., .5, 2., 6.))]
        if max(errors) > 1.e-10:
            raise AssertionError(f"Native replay mismatch: {errors}")
        report["native_replay_max_errors"] = errors
        save(report)
        print("Checking quadrature refinement against batched GPU samples...", flush=True)
        report["refinement"] = refinement_check(device=args.device, estimates=args.estimates)
        save(report)
        print("Checking all seven gradients and checkpoint equivalence...", flush=True)
        report["gradient"] = finite_difference_check(device=args.device, points=args.points)
        save(report)
        print("Refining spatial and crossing quadrature for the gradients...", flush=True)
        report["gradient_refinement"] = gradient_grid_check(device=args.device)
    else:
        import pandas as pd
        frame = pd.read_csv(args.data)
        frame = frame[(frame.subject_nr == args.subject) & frame.PrevCongruency.notna()].iloc[:args.trials].copy()
        if len(frame) != args.trials:
            parser.error(f"Only {len(frame)} eligible trials available")
        included = frame.likelihood_include_mask.to_numpy(dtype=bool)
        if not included.any():
            parser.error("This slice has no included observations; increase --trials to retain the warm-up history")
        if "decision_steps" in frame:
            history = frame.decision_steps.to_numpy()
            history_mode = "supplied decision_steps; conditional on known latent history"
        else:
            history = np.rint((frame.response_time.to_numpy() - args.history_reference_ndt) / .01).astype(int)
            history_mode = "fixed plug-in history; not a marginalized sequential RT likelihood"
        if np.any(history < 1) or np.any(history > args.max_steps):
            parser.error("Inferred history exceeds the horizon or is nonpositive; adjust history reference/horizon")
        config = SolverConfig(points=args.points)
        evaluations = []

        def evaluate(values):
            start = time.perf_counter()
            p = torch.tensor(values, dtype=torch.float64, device=args.device, requires_grad=True)
            result = sequence_likelihood(p, frame[["T1", "T2"]].to_numpy(), frame[["S1", "S2", "S3", "S4"]].to_numpy(),
                                         frame.decision.to_numpy(), frame.response_time.to_numpy(), history,
                                         max_steps=args.max_steps, config=config,
                                         include=included,
                                         measurement_sd=args.measurement_sd, resolution=args.resolution)
            value = float(result.log_likelihood.detach())
            if not np.isfinite(value):
                raise FloatingPointError("Zero/nonfinite observation probability; proposal is infeasible")
            gradient = torch.autograd.grad(result.log_likelihood, p)[0]
            grad = gradient.detach().cpu().numpy()
            if not np.all(np.isfinite(grad)):
                raise FloatingPointError("Nonfinite likelihood gradient; proposal is infeasible")
            info = {"log_likelihood": value, "parameters": dict(zip(PARAMETER_NAMES, map(float, values))),
                    "gradient": grad.tolist(), "seconds": time.perf_counter() - start,
                    "maximum_mass_error": max(float(d.mass_error.detach()) for d in result.distributions if d is not None),
                    "maximum_quadrature_defect": max(float(d.quadrature_defect.detach()) for d in result.distributions if d is not None),
                    "maximum_lower_loss": max(float(d.lower_loss.detach()) for d in result.distributions if d is not None),
                    "maximum_survival_at_horizon": max(float(d.survival.detach()) for d in result.distributions if d is not None)}
            evaluations.append(info)
            print(f"evaluation {len(evaluations)}: log likelihood={value:.6f}, {info['seconds']:.3f}s", flush=True)
            return value, grad

        report.update({"subject": args.subject, "trials": args.trials, "included_trials": int(included.sum()), "points": args.points,
                       "max_steps": args.max_steps, "history_mode": history_mode,
                       "history_reference_ndt": args.history_reference_ndt, "history_steps": history.tolist(),
                       "measurement_sd": args.measurement_sd, "rt_resolution": args.resolution,
                       "parameter_scope": "one shared seven-parameter vector", "evaluations": evaluations})
        if args.command == "score":
            evaluate(args.parameters)
        else:
            from dawa_likelihood.fit import fit_projected_gradient

            def objective(values):
                value, gradient = evaluate(values)
                save(report)
                return value, gradient

            result = fit_projected_gradient(objective, args.parameters, PARAMETER_BOUNDS, iterations=args.iterations)
            result["parameters"] = dict(zip(PARAMETER_NAMES, result["parameters"]))
            report["fit"] = result
    save(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
