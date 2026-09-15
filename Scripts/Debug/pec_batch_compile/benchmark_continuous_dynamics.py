"""Compare generated continuous equations with the CSI handwritten drift kernel.

This research-only fixture is not registered as a compiler recognizer. Run from
the repository root. Prints timings and numerical differences; writes no files.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import sympy as sp

from psyneulink.core.batched.continuous_ir import ContinuousDynamics, Sigmoid
from psyneulink.core.batched.numerical.dynamics import compile_continuous_phase


def csi_equations():
    x, y, gain = sp.symbols("x y gain", real=True)
    task0, task1, s0, s1, s2, s3, response = sp.symbols("task0 task1 stimulus0 stimulus1 stimulus2 stimulus3 response", real=True)
    with sp.evaluate(False):
        control0, control1 = Sigmoid(gain * x), Sigmoid(gain * y)
        a, b = Sigmoid(s0 - s1 + 4 * control0 - 4), Sigmoid(s1 - s0 + 4 * control0 - 4)
        c, d = Sigmoid(s2 - s3 + 4 * control1 - 4), Sigmoid(s3 - s2 + 4 * control1 - 4)
        contrast = a - b + c - d
        drift = (Sigmoid(contrast) - Sigmoid(-contrast)) * response
        return ContinuousDynamics(
            states=(x, y), inputs=(task0, task1, s0, s1, s2, s3, response), parameters=(gain,),
            drift=(-12 * x + task0 - 3 * control1, -12 * y + task1 - 3 * control0), readouts=(("drift", drift),),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=480)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(args.trials, args.steps, args.threads, args.repeats) <= 0:
        parser.error("All counts must be positive.")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "csi_fit"))
    from direct_likelihood.native import native_lca_drift_path

    plan = compile_continuous_phase(csi_equations())
    x = torch.linspace(-.06, .04, args.trials * 2, dtype=torch.float64).reshape(args.trials, 2).requires_grad_()
    u = torch.tensor([[1., 0., 1., 0., .2, .8, 1.], [0., 1., .2, .8, .9, .1, -1.]], dtype=torch.float64)
    u = u.repeat((args.trials + 1) // 2, 1)[:args.trials].contiguous().requires_grad_()
    p = torch.linspace(9, 24, args.trials, dtype=torch.float64)[:, None].requires_grad_()
    duration = torch.full((args.trials,), args.steps * .001, dtype=torch.float64)
    start_time = torch.zeros(args.trials, dtype=torch.float64)
    steps = torch.full((args.trials,), args.steps, dtype=torch.int64)

    def run(route, grad):
        begin = time.perf_counter()
        with torch.set_grad_enabled(grad):
            if route == "generated":
                result = plan.integrate(state=x, inputs=u, parameters=p, duration=duration, start_time=start_time, steps=steps)
                values = result.readouts[..., 0], result.final_state
            else:
                values = native_lca_drift_path(x, u[:, :2], p[:, 0], u[:, 2:6], u[:, 6],
                                               steps=args.steps, step_size=.001, leak=12., competition=3.)
            gradients = torch.autograd.grad(values[0].sum() + values[1].sum(), (x, u, p)) if grad else ()
        elapsed = time.perf_counter() - begin
        return elapsed, tuple(t.detach() for t in (*values, *gradients))

    output = dict(trials=args.trials, steps=args.steps, threads=args.threads, dt=.001, dtype="float64", warmups=2)
    for grad in (False, True):
        timings = {route: [] for route in ("handwritten", "generated")}
        for route in timings:
            for _ in range(2):
                run(route, grad)
        last = {}
        for repetition in range(args.repeats):
            routes = tuple(timings) if repetition % 2 == 0 else tuple(reversed(timings))
            for route in routes:
                elapsed, last[route] = run(route, grad)
                timings[route].append(elapsed)
        differences = []
        for actual, reference in zip(last["generated"], last["handwritten"], strict=True):
            torch.testing.assert_close(actual, reference, rtol=2e-9, atol=2e-10)
            differences.append(float((actual - reference).abs().max()))
        output["gradient" if grad else "forward"] = dict(
            seconds={route: statistics.median(runs) for route, runs in timings.items()},
            runs=timings, max_absolute_differences=differences,
        )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
