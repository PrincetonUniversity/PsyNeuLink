"""Warm CPU first-passage score/adjoint benchmark, independent of CSI equations.

Run from the repository root with .venv/bin/python. No files are written. This
measures a fitting-sized PDE stage, not full subject preprocessing or fitting.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("backend", "csi_compat"), default="backend")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--trials", type=int, default=480)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=7)
    options = parser.parse_args()
    if min(options.threads, options.trials, options.repeats) < 1 or options.steps < 956:
        parser.error("Use positive threads/trials/repeats and at least 956 time steps.")
    torch.set_num_threads(options.threads)
    torch.set_num_interop_threads(1)
    dtype = torch.float64
    lanes, steps = options.trials, options.steps
    t = (torch.arange(steps, dtype=dtype) + .5) * .001
    phase = torch.linspace(0, 2, lanes, dtype=dtype)
    values = dict(
        drift=(.1 + .06 * torch.sin(8 * t[None, :] + phase[:, None])).requires_grad_(),
        threshold=torch.linspace(.08, .14, lanes, dtype=dtype).requires_grad_(),
        collapse_rate=torch.linspace(-.025, -.01, lanes, dtype=dtype).requires_grad_(),
        interval_low=torch.linspace(.2513, .9513, lanes, dtype=dtype).requires_grad_(),
        interval_high=torch.linspace(.2553, .9553, lanes, dtype=dtype).requires_grad_(),
        choice=(torch.arange(lanes) % 2).to(dtype),
    )
    if options.route == "csi_compat":
        sys.path.insert(0, str(Path(__file__).resolve().parent / "csi" / "csi_fit"))
        from direct_likelihood.solver import MovingBoundaryDDMSolver

        solver = MovingBoundaryDDMSolver(time_step=.001, spatial_points=65, noise=.1,
                                         native_forward=True, custom_adjoint=True)
    else:
        from psyneulink.core.batched.numerical import FirstPassageProblem, compile_first_passage

        problem = FirstPassageProblem(
            process="continuous_time", stochastic_dimensions=1, drift_dependence="time_only",
            diffusion="constant_scalar", boundary="symmetric_linear", initial_state="point_center",
            coefficient_source="conditioned_deterministic", observation="choice_rt_interval",
        )
        solver = compile_first_passage(problem, noise=.1)

    def run(grad):
        start = time.perf_counter()
        with torch.set_grad_enabled(grad):
            result = solver.solve_observation_batch(**values)
            value = result.probability.log().sum()
            gradients = torch.autograd.grad(value, tuple(v for v in values.values() if v.requires_grad)) if grad else ()
        elapsed = time.perf_counter() - start
        hashes = [hashlib.sha256(v.detach().numpy().tobytes()).hexdigest() for v in (result.probability, *gradients)]
        return elapsed, float(value.detach()), hashes

    output = dict(route=options.route, threads=options.threads, trials=lanes, steps=steps,
                  time_step=.001, spatial_points=65, dtype="float64", warmups=2)
    for grad in (False, True):
        for _ in range(2):
            run(grad)
        results = [run(grad) for _ in range(options.repeats)]
        output["gradient" if grad else "score"] = dict(
            seconds=statistics.median(r[0] for r in results), runs=[r[0] for r in results],
            value=results[-1][1], hashes=results[-1][2],
        )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
