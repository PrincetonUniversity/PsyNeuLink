"""Warmed timing and parity of DAWA's reference and optional native backends."""

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from dawa_likelihood.continuous_likelihood import continuous_sequence_likelihood
from dawa_likelihood.continuous_model import continuous_path
from dawa_likelihood.continuous_native import continuous_path_native
from dawa_likelihood.continuous_solver import ContinuousConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/dawa_continuous_profile.json"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Positive repeat count required")
    torch.set_num_threads(1)
    report = {"device": "cpu", "dtype": "float64", "threads": 1, "repeats": args.repeats,
              "scope": "One fresh-trial likelihood and gradient; not full-subject fitting throughput.",
              "observation": {"task": [1, 0], "stimulus": [0, 1, 0, 1], "choice": 1, "rt": .9537},
              "numerical": {"points": 65, "time_step": .001, "ode_step": .0005},
              "deterministic": {}, "likelihood": {}}

    def evaluate(ode, flux=None):
        p = torch.tensor([.3, .2, -.45, 10., .9, 1., 5.], dtype=torch.float64, requires_grad=True)
        start = time.perf_counter()
        if flux is None:
            fun = continuous_path if ode == "torch" else continuous_path_native
            path = fun(p, [1, 0], [0, 1, 0, 1], steps=1000)
            value = path.inputs.square().mean() + path.gain.mean() + path.gain_rate.mean()
        else:
            cfg = ContinuousConfig(points=65, ode_backend=ode, flux_backend=flux)
            value = continuous_sequence_likelihood(p, [[1, 0]], [[0, 1, 0, 1]], [1], [.9537], config=cfg).log_likelihood
        gradient = torch.autograd.grad(value, p)[0].numpy()
        return time.perf_counter() - start, float(value.detach()), gradient

    for section, routes in (("deterministic", (("torch", None), ("generated", None))),
                            ("likelihood", (("torch", "torch"), ("generated", "torch"), ("generated", "native")))):
        timings, outputs = {route: [] for route in routes}, {}
        for route in routes:
            evaluate(*route)  # All extension loading/JIT setup is excluded below.
        for repeat in range(args.repeats):
            for route in (routes if repeat % 2 == 0 else tuple(reversed(routes))):
                elapsed, value, gradient = evaluate(*route)
                timings[route].append(elapsed)
                outputs[route] = (value, gradient)
                print(section, route, elapsed, flush=True)
        reference_value, reference_gradient = outputs[routes[0]]
        for route in routes:
            value, gradient = outputs[route]
            np.testing.assert_allclose(value, reference_value, atol=2.e-7, rtol=2.e-7)
            np.testing.assert_allclose(gradient, reference_gradient, atol=2.e-6, rtol=2.e-6)
            report[section]["/".join(x for x in route if x)] = {
                "seconds": timings[route], "median_seconds": statistics.median(timings[route]),
                "value": value, "gradient": gradient.tolist(), "value_difference": abs(value - reference_value),
                "maximum_gradient_difference": float(np.abs(gradient - reference_gradient).max())}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
