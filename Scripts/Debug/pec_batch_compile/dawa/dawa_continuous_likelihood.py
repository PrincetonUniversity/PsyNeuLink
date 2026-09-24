"""Continuous-time DAWA likelihood and gradients, using CSI-style RT bins.

The deterministic network is a coupled ODE with instantaneous gain modulation.
A 2D absorbing Fokker-Planck solver scores choices and recorded RT intervals.
History is reconstructed from RT minus each candidate's nondecision time.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from dawa_likelihood import DEFAULT_PARAMETERS, PARAMETER_NAMES, ContinuousConfig, continuous_sequence_likelihood
from dawa_likelihood.model import PARAMETER_BOUNDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "score", "fit"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--points", type=int, default=65)
    parser.add_argument("--time-step", type=float, default=.001)
    parser.add_argument("--ode-step", type=float, default=.0005)
    parser.add_argument("--ode-backend", choices=("torch", "generated"), default="torch")
    parser.add_argument("--flux-backend", choices=("torch", "native", "triton"), default="torch")
    parser.add_argument("--cpu-threads", type=int, default=1, help="Threads in native finite-volume kernels")
    parser.add_argument("--no-gpu-graphs", action="store_true")
    parser.add_argument("--retain-rates", action="store_true", help="Retain coefficient blocks to save recomputation")
    parser.add_argument("--lower-bound", type=float, default=-.25)
    parser.add_argument("--lc-clock-ratio", type=float, default=20.)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("dawa_lca_model") / "flanker_data_part1.csv")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--parameters", type=float, nargs=7, default=DEFAULT_PARAMETERS, metavar="VALUE")
    parser.add_argument("--resolution", type=float, default=.001)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--estimates", type=int, default=100000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.threads, args.cpu_threads, args.trials, args.iterations, args.estimates) < 1:
        parser.error("Thread, trial, iteration, and estimate counts must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; select --device cpu")
    if args.flux_backend == "native" and args.device != "cpu":
        parser.error("The native flux backend requires --device cpu")
    if args.flux_backend == "triton" and args.device != "cuda":
        parser.error("The Triton flux backend requires --device cuda")
    torch.set_num_threads(args.threads)
    cfg = ContinuousConfig(points=args.points, time_step=args.time_step, ode_step=args.ode_step,
                           lower_bound=args.lower_bound, lc_clock_ratio=args.lc_clock_ratio,
                           ode_backend=args.ode_backend, flux_backend=args.flux_backend,
                           cpu_threads=args.cpu_threads, gpu_graphs=not args.no_gpu_graphs,
                           recompute_rates=not args.retain_rates)
    report = {"process": "continuous", "device": args.device, "dtype": "float64", "torch": torch.__version__,
              "history": "candidate-dependent RT - nondecision time; RT-bin-center reconstruction",
              "observation": "recorded RT interval; no added Gaussian measurement noise"}
    if args.device == "cuda":
        report["gpu"] = torch.cuda.get_device_name()

    def save():
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")

    if args.command == "validate":
        from dawa_likelihood.continuous_validation import run_validation
        report["audit_scope"] = "Fixed oracle and refinement cases; numerical settings are recorded per case."
        report["model"] = {"lc_clock_ratio": 20., "response_noise": .1, "lower_bound": -.25, "ode_step": .0005,
                           "default_parameters": dict(zip(PARAMETER_NAMES, DEFAULT_PARAMETERS)),
                           "refinement_and_monte_carlo_task": [1, 0], "refinement_and_monte_carlo_stimulus": [0, 1, 0, 1]}
        run_validation(report, save, device=args.device, estimates=args.estimates)
    else:
        import pandas as pd
        frame = pd.read_csv(args.data)
        frame = frame[(frame.subject_nr == args.subject) & frame.PrevCongruency.notna()].iloc[:args.trials].copy()
        if len(frame) != args.trials:
            parser.error(f"Only {len(frame)} eligible trials available")
        included = frame.likelihood_include_mask.to_numpy(dtype=bool)
        if not included.any():
            parser.error("This slice has no included observations; increase --trials to retain warm-up history")
        report.update({"subject": args.subject, "trials": args.trials, "included_trials": int(included.sum()),
                       "points": cfg.points, "time_step": cfg.time_step, "ode_step": cfg.ode_step,
                       "ode_backend": cfg.ode_backend,
                       "flux_backend": cfg.flux_backend,
                       "cpu_threads": cfg.cpu_threads, "gpu_graphs": cfg.gpu_graphs,
                       "recompute_rates": cfg.recompute_rates,
                       "lc_clock_ratio": cfg.lc_clock_ratio, "lower_bound": cfg.lower_bound,
                       "rt_resolution": args.resolution, "parameter_scope": "one shared seven-parameter vector",
                       "evaluations": []})

        def evaluate(values):
            start = time.perf_counter()
            p = torch.tensor(values, dtype=torch.float64, device=args.device, requires_grad=True)
            result = continuous_sequence_likelihood(p, frame[["T1", "T2"]].to_numpy(),
                                                    frame[["S1", "S2", "S3", "S4"]].to_numpy(),
                                                    frame.decision.to_numpy(), frame.response_time.to_numpy(),
                                                    config=cfg, include=included, resolution=args.resolution)
            value = float(result.log_likelihood.detach())
            if not np.isfinite(value):
                raise FloatingPointError("Zero/nonfinite observation probability; proposal is infeasible")
            gradient = torch.autograd.grad(result.log_likelihood, p)[0].cpu().numpy()
            if not np.all(np.isfinite(gradient)):
                raise FloatingPointError("Nonfinite likelihood gradient; proposal is infeasible")
            ds = [d for d in result.distributions if d is not None]
            record = {"log_likelihood": value, "parameters": dict(zip(PARAMETER_NAMES, map(float, values))),
                      "gradient": gradient.tolist(), "seconds": time.perf_counter() - start,
                      "maximum_mass_error": max(float(d.mass_error.detach()) for d in ds),
                      "maximum_lower_loss": max(float(d.lower_loss.detach()) for d in ds),
                      "minimum_mass": min(float(d.minimum_mass.detach()) for d in ds),
                      "maximum_cfl": max(d.maximum_cfl for d in ds),
                      "maximum_substeps": max(d.substeps for d in ds)}
            report["evaluations"].append(record)
            print(f"evaluation {len(report['evaluations'])}: log likelihood={value:.6f}, {record['seconds']:.3f}s", flush=True)
            save()
            return value, gradient

        if args.command == "score":
            evaluate(args.parameters)
        else:
            from dawa_likelihood.fit import fit_projected_gradient
            result = fit_projected_gradient(evaluate, args.parameters, PARAMETER_BOUNDS, iterations=args.iterations)
            result["parameters"] = dict(zip(PARAMETER_NAMES, result["parameters"]))
            report["fit"] = result
    save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
