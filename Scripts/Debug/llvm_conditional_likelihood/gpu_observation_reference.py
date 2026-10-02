"""Verify or regenerate the CPU test fixture using the GPU feature checkout.

Select the GPU PsyNeuLink checkout through PYTHONPATH. Without --output this
checks the existing fixture. --output writes a new capture for review.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from psyneulink.core.batched.likelihood import _HistogramObservationWeights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "tests/composition/pec/particlefilter_gpu_reference.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--source-revision", help="GPU revision to record when regenerating"
    )
    args = parser.parse_args()
    reference = json.loads(args.fixture.read_text())
    captures = []
    for case in reference["cases"]:
        data = np.asarray(case["data"])
        simulated = torch.as_tensor(
            case["simulated"], dtype=torch.float32, device="cuda"
        )
        simulated = simulated.reshape(1, 1, -1, data.shape[1])
        count = simulated.shape[-2]
        cardinalities = [len(values) for values in case["categorical_values"]]
        cells = case["bins"] ** (
            data.shape[1] - sum(case["categorical_dims"])
        ) * np.prod(cardinalities)
        epsilon = case["contamination_probability"]
        alpha = epsilon * count / ((1 - epsilon) * cells)
        observation = _HistogramObservationWeights(
            data,
            case["categorical_dims"],
            bins=case["bins"],
            bin_range=case["bin_range"],
            smoothing_sigma=case["smoothing_sigma"],
            pseudocount=alpha,
            categorical_cardinalities=cardinalities,
            dtype=torch.float32,
            device="cuda",
            strict_observations=True,
            source_normalized=True,
            fused=True,
        )
        capture = dict(
            case,
            edge_bits=[
                edge.cpu().numpy().view(np.uint32).tolist()
                for edge in observation.edges
            ],
            log_densities=[],
            weights=[],
            responsibilities=[],
        )
        for trial in range(len(data)):
            weights, density = observation(simulated, trial)
            total = weights.sum()
            capture["log_densities"].append(float(torch.log(density).item()))
            capture["weights"].append(
                (weights / total).cpu().numpy().reshape(-1).tolist()
            )
            capture["responsibilities"].append(float((alpha / total).item()))
        if args.output is None:
            for expected, actual in zip(case["edge_bits"], capture["edge_bits"]):
                np.testing.assert_array_equal(actual, expected)
            for key in ("log_densities", "weights", "responsibilities"):
                np.testing.assert_allclose(
                    capture[key], case[key], atol=1e-9, rtol=2e-6, err_msg=case["name"]
                )
        captures.append(capture)
    if args.output is not None:
        implementation = Path(inspect.getfile(_HistogramObservationWeights))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "source_commit": args.source_revision,
                    "source": "psyneulink/core/batched/likelihood.py:_HistogramObservationWeights (fused CUDA)",
                    "implementation_sha256": hashlib.sha256(
                        implementation.read_bytes()
                    ).hexdigest(),
                    "torch": torch.__version__,
                    "device": torch.cuda.get_device_name(),
                    "cases": captures,
                },
                indent=2,
            )
            + "\n"
        )
    print(
        f"{'Captured' if args.output else 'Verified'} {len(captures)} GPU observation cases."
    )


if __name__ == "__main__":
    main()
