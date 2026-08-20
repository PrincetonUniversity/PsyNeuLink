"""ASV benchmarks for the batched Triton compiler (GPU `triton` backend).

Tracks steady-state run time of `BatchedSimulationPlan.run()` across commits for
representative models, swept over the number of estimates (GPU lanes):

- `DDM`         — single DDM (ddm fusion)
- `DDMGraph`    — transfer -> DDM (ddm_graph fusion)
- `LCA`         — deterministic width-2 LCA with a static trial threshold
  (stateful_graph fusion)
- `StabilityFlexibility` — toy LCA + DDM (stateful_graph fusion)
- `CSISurrogate` — stochastic co-evolving CSI research model with the historical
  five-parameter recovery surface (coevolving_graph fusion)

The kernel is compiled and warmed up in `setup()`, so the timed methods measure
only the batched simulation (not one-time compilation). Each warmup records an
output checksum (`track_checksum`) to surface correctness drift across commits.

Requires CUDA + triton (compiled GPU path); benchmarks skip otherwise. Run with:

    .venv/bin/asv run --set-commit-hash $(git rev-parse HEAD)
    .venv/bin/asv publish && .venv/bin/asv preview
"""

import re
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "tests" / "composition" / "pec"))
sys.path.insert(0, str(_REPO / "Scripts" / "Debug" / "pec_batch_compile"))

import psyneulink as pnl  # noqa: E402
from psyneulink.core.batched import BatchedCompositionCompiler, batched_node_op  # noqa: E402


# The CSI surrogate's drift-rate UDF, registered as an instance op (the body uses
# `tl`, resolved at emission). Module-level so it is registered once on import.
@batched_node_op("Drift Rate Value")
def _csi_drift_rate(x0, x1, x2, x3, x4, x5, x6):
    a = 1.0 / (1.0 + tl.exp(-((x0 - x1) + 4.0 * x4 - 4.0)))
    b = 1.0 / (1.0 + tl.exp(-((x1 - x0) + 4.0 * x4 - 4.0)))
    c = 1.0 / (1.0 + tl.exp(-((x2 - x3) + 4.0 * x5 - 4.0)))
    d = 1.0 / (1.0 + tl.exp(-((x3 - x2) + 4.0 * x5 - 4.0)))
    pos = 1.0 / (1.0 + tl.exp(-(a - b + c - d)))
    neg = 1.0 / (1.0 + tl.exp(-(-a + b - c + d)))
    return (pos - neg) * x6


TRIALS = 128
PARAM_SETS = 8

# Estimate sweeps: cheap (trial-lane) models get a wider range than the heavier
# stateful (per-lane trial loop) models, to keep total asv runtime reasonable.
STATELESS_ESTIMATES = [1024, 4096, 16384]
STATEFUL_ESTIMATES = [256, 1024, 4096]


def _require_cuda():
    try:
        import torch
    except ImportError:
        raise NotImplementedError("torch is not installed")
    if not torch.cuda.is_available():
        raise NotImplementedError("CUDA device is not available")


def _ddm_function():
    return pnl.DriftDiffusionIntegrator(
        starting_value=0.0, rate=1.0, noise=0.1, threshold=0.05,
        non_decision_time=0.0, time_step_size=0.01,
    )


def _build_ddm():
    decision = pnl.DDM(
        function=_ddm_function(),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME], name="DDM",
    )
    comp = pnl.Composition(pathways=decision)
    plan = BatchedCompositionCompiler.compile(comp, backend="triton", max_steps=64)
    inputs = {decision: np.ones((TRIALS, 1), dtype=float)}
    param_sets = [
        {"rate": 1.0, "threshold": 0.05, "noise": 0.1, "time_step_size": 0.01}
        for _ in range(PARAM_SETS)
    ]
    return plan, inputs, param_sets


def _build_ddm_graph():
    source = pnl.TransferMechanism(input_shapes=1, name="stimulus")
    decision = pnl.DDM(
        function=_ddm_function(),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME], name="DDM",
    )
    comp = pnl.Composition(pathways=[[source, decision]])
    plan = BatchedCompositionCompiler.compile(comp, backend="triton", max_steps=64)
    inputs = {source: np.ones((TRIALS, 1), dtype=float)}
    param_sets = [
        {"DDM.rate": 1.0, "DDM.threshold": 0.05, "DDM.noise": 0.1, "DDM.time_step_size": 0.01}
        for _ in range(PARAM_SETS)
    ]
    return plan, inputs, param_sets


def _build_lca():
    task = pnl.TransferMechanism(input_shapes=2, function=pnl.Linear, name="Task")
    lca = pnl.LCAMechanism(
        input_shapes=2, function=pnl.Logistic(gain=1.0), leak=0.5, competition=1.0,
        self_excitation=0.0, noise=0.0, termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=64, time_step_size=0.01,
        reset_stateful_function_when=pnl.Never(), name="LCA",
    )
    readout = pnl.TransferMechanism(input_shapes=2, function=pnl.Linear(slope=1.0), name="Readout")
    comp = pnl.Composition(pathways=[[task, lca, readout]])
    plan = BatchedCompositionCompiler.compile(comp, backend="triton", max_steps=64)
    inputs = {task: np.tile(np.array([1.0, 0.0]), (TRIALS, 1))}
    param_sets = [dict() for _ in range(PARAM_SETS)]
    return plan, inputs, param_sets


def _build_stab_flex():
    from test_stab_flex_pec_fit import generate_trial_sequence, make_input_dict, make_stab_flex

    comp = make_stab_flex(
        lca_time_step_size=0.01, ddm_time_step_size=0.01,
        threshold=0.05, ddm_noise=0.1, lca_noise=0.0,
    )
    task, stimulus, cue, correct = generate_trial_sequence(TRIALS, 0.5, seed=3)
    inputs = make_input_dict(comp, task[:TRIALS], stimulus[:TRIALS], cue[:TRIALS], correct[:TRIALS])
    plan = BatchedCompositionCompiler.compile(comp, backend="triton", max_steps=256)
    param_sets = [
        {"DDM.threshold": 0.05, "DDM.noise": 0.1}
        for _ in range(PARAM_SETS)
    ]
    return plan, inputs, param_sets


def _build_csi():
    """Build the stochastic CSI surrogate and its five recovery parameters."""

    from csi_model_surrogate import generate_mixed_task_sequence, make_stab_flex

    comp = make_stab_flex(
        iti=0,
        csi_repeat=0,
        csi_switch=10,
        threshold=0.12,
        threshold_collapse=-0.001,
        non_decision_time=0.3,
        ddm_noise=0.1,
        lca_noise=0.0,
        lca_time_step_size=0.01,
        ddm_time_step_size=0.01,
    )

    def node(base):
        return next(
            candidate
            for candidate in comp.nodes
            if re.sub(r"-\d+$", "", candidate.name) == base
        )

    tasks, stimuli, correct, _ = generate_mixed_task_sequence(
        TRIALS,
        switch_frequency=0.5,
        incongruence_frequency=0.5,
        seed=3,
    )
    task_values = np.asarray(tasks, dtype=float)
    cue_values = np.zeros((TRIALS, 1), dtype=float)
    cue_values[1:, 0] = np.any(task_values[1:] != task_values[:-1], axis=1)
    inputs = {
        node("Stimulus Input"): np.asarray(stimuli, dtype=float),
        node("Task Input"): task_values,
        node("Correct Response"): np.asarray(correct, dtype=float).reshape(TRIALS, 1),
        node("Cue Stimulus Interval"): cue_values,
    }
    plan = BatchedCompositionCompiler.compile(
        comp,
        backend="triton",
        max_steps=512,
    )

    lca = node("Task Activations [C1, C2]")
    cue = node("Cue Stimulus Interval")
    threshold_source = node("Threshold Mechanism")
    ddm = node("DDM")
    gains = np.linspace(5.0, 20.0, PARAM_SETS)
    switch_steps = np.rint(np.linspace(0.0, 50.0, PARAM_SETS))
    thresholds = np.linspace(0.08, 0.25, PARAM_SETS)
    collapses = np.linspace(-0.003, 0.0, PARAM_SETS)
    non_decision_times = np.linspace(0.0, 0.6, PARAM_SETS)
    param_sets = [
        {
            f"{lca.name}.gain": float(gain),
            f"{cue.name}.slope": float(switch_step),
            f"{threshold_source.name}.intercept": float(threshold),
            f"{threshold_source.name}.offset-integrator_function": float(collapse),
            f"{ddm.name}.non_decision_time": float(non_decision_time),
        }
        for gain, switch_step, threshold, collapse, non_decision_time in zip(
            gains,
            switch_steps,
            thresholds,
            collapses,
            non_decision_times,
        )
    ]
    return plan, inputs, param_sets


class _BatchedBenchmark:
    """Shared setup/run logic. Subclasses set `params`, `_seed`, and `_build`."""

    param_names = ["estimates"]
    _seed = 11

    def setup(self, estimates):
        _require_cuda()
        self._plan, self._inputs, self._param_sets = self._build()
        self._estimates = estimates
        # Warm up: triggers kernel compilation + GPU warm-up, and records checksum.
        self._checksum = float(np.sum(self._run().values))

    def _run(self):
        return self._plan.run(
            inputs=self._inputs, parameter_sets=self._param_sets,
            num_estimates=self._estimates, seed=self._seed,
        )

    def time_run(self, estimates):
        self._run()

    def track_checksum(self, estimates):
        return self._checksum

    track_checksum.unit = "sum"


class DDM(_BatchedBenchmark):
    params = STATELESS_ESTIMATES
    _build = staticmethod(_build_ddm)


class DDMGraph(_BatchedBenchmark):
    params = STATELESS_ESTIMATES
    _build = staticmethod(_build_ddm_graph)


class LCA(_BatchedBenchmark):
    params = STATEFUL_ESTIMATES
    _seed = 3
    _build = staticmethod(_build_lca)


class StabilityFlexibility(_BatchedBenchmark):
    params = STATEFUL_ESTIMATES
    _seed = 3
    _build = staticmethod(_build_stab_flex)


class CSISurrogate(_BatchedBenchmark):
    """Stochastic co-evolving CSI model with five runtime recovery parameters."""

    params = STATEFUL_ESTIMATES
    _seed = 3
    _build = staticmethod(_build_csi)
