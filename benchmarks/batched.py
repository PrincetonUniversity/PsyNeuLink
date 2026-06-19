"""ASV benchmarks for the batched Triton compiler (GPU `triton` backend).

Tracks steady-state run time of `BatchedSimulationPlan.run()` for representative
DDM and stability-flexibility cases as the batched compiler evolves. The kernel
is compiled and warmed up in `setup()`, so the timed methods measure only the
batched simulation (not one-time compilation). Each warmup also records an
output checksum (`track_checksum_*`) to surface correctness drift across commits.

Requires CUDA + triton (compiled GPU path); benchmarks skip otherwise. Run with:

    .venv/bin/asv run            # benchmarks current HEAD (existing env)
    .venv/bin/asv publish && .venv/bin/asv preview
"""

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "tests" / "composition" / "pec"))

import psyneulink as pnl  # noqa: E402
from psyneulink.core.batched import BatchedCompositionCompiler  # noqa: E402


TRIALS = 128
PARAM_SETS = 8
SEED = 11


def _require_cuda():
    try:
        import torch
    except ImportError:
        raise NotImplementedError("torch is not installed")
    if not torch.cuda.is_available():
        raise NotImplementedError("CUDA device is not available")


class DDM:
    """Single-DDM model, lowered to the generated ddm_graph kernel."""

    params = [1024, 4096]
    param_names = ["estimates"]

    def setup(self, estimates):
        _require_cuda()
        decision = pnl.DDM(
            function=pnl.DriftDiffusionIntegrator(
                starting_value=0.0, rate=1.0, noise=0.1, threshold=0.05,
                non_decision_time=0.0, time_step_size=0.01,
            ),
            output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
            name="DDM",
        )
        comp = pnl.Composition(pathways=decision)
        self._plan = BatchedCompositionCompiler.compile(comp, backend="triton", max_steps=64)
        self._inputs = {decision: np.ones((TRIALS, 1), dtype=float)}
        self._param_sets = [
            {"rate": 1.0, "threshold": 0.05, "noise": 0.1, "time_step_size": 0.01}
            for _ in range(PARAM_SETS)
        ]
        self._estimates = estimates
        # Warm up: triggers kernel compilation + GPU warm-up, and records checksum.
        result = self._run()
        self._checksum = float(np.sum(result.values))

    def _run(self):
        return self._plan.run(
            inputs=self._inputs, parameter_sets=self._param_sets,
            num_estimates=self._estimates, seed=SEED,
        )

    def time_run(self, estimates):
        self._run()

    def track_checksum(self, estimates):
        return self._checksum

    track_checksum.unit = "sum"


class StabilityFlexibility:
    """Stateful LCA + DDM graph; trials looped inside each lane."""

    params = [1024]
    param_names = ["estimates"]

    def setup(self, estimates):
        _require_cuda()
        from test_stab_flex_pec_fit import (
            generate_trial_sequence,
            make_input_dict,
            make_stab_flex,
        )

        comp = make_stab_flex(
            lca_time_step_size=0.01, ddm_time_step_size=0.01,
            threshold=0.05, ddm_noise=0.1, lca_noise=0.0,
        )
        task, stimulus, cue, correct = generate_trial_sequence(TRIALS, 0.5, seed=3)
        self._inputs = make_input_dict(
            comp, task[:TRIALS], stimulus[:TRIALS], cue[:TRIALS], correct[:TRIALS]
        )
        self._plan = BatchedCompositionCompiler.compile(comp, backend="triton", max_steps=256)
        self._param_sets = [
            {"DDM.threshold": 0.05, "DDM.noise": 0.1, "Task Activations [Act1, Act2].noise": 0.0}
            for _ in range(PARAM_SETS)
        ]
        self._estimates = estimates
        result = self._run()
        self._checksum = float(np.sum(result.values))

    def _run(self):
        return self._plan.run(
            inputs=self._inputs, parameter_sets=self._param_sets,
            num_estimates=self._estimates, seed=3,
        )

    def time_run(self, estimates):
        self._run()

    def track_checksum(self, estimates):
        return self._checksum

    track_checksum.unit = "sum"
