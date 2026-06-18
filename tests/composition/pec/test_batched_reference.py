"""Validate the batched compiler against real PsyNeuLink execution.

The batched Triton kernels (run here on CPU through Triton's interpreter via the
``triton_cpu`` backend) are checked against PsyNeuLink's own Python-mode
execution — the authoritative reference.  There is no hand-written numpy
reference: the oracle is the real component math.

- Deterministic (noise=0) cases must match PNL exactly (fp32 tolerance).
- Stochastic cases match PNL summary statistics over many samples (the batched
  RNG is deterministic-per-kernel, not bitwise-identical to PNL's streams).

PEC-scale stochastic / GPU-compiled validation lives in
``Scripts/Debug/pec_batch_compile/pec_grid_correctness_check.py`` (PNL LLVM
oracle), which runs as its own process because Triton interpret (CPU) and
compiled (GPU) modes cannot coexist in one process.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import psyneulink as pnl

from psyneulink.core.batched import BatchedCompositionCompiler


requires_triton = pytest.mark.skipif(
    importlib.util.find_spec("triton") is None or importlib.util.find_spec("torch") is None,
    reason="torch + triton are required for batched CPU (interpret) execution",
)

pytestmark = [pytest.mark.composition, requires_triton]


def _pnl_python_outcomes(comp, inputs, output_specs=None):
    """Run ``comp`` in PNL Python mode and return per-trial outcome rows.

    Each trial's terminal output ports are flattened and concatenated, then (if
    ``output_specs`` is given) reordered/selected to match the batched
    compiler's outputs.  A composition may expose more terminal outputs than the
    batched plan models, so we map each batched output to its ``comp.results``
    column via the output-CIM port names (``OUTPUT_CIM_<node>_<port>``).
    """

    comp.run(inputs=inputs, execution_mode=pnl.ExecutionMode.Python)
    num_trials = len(comp.results)
    if output_specs is None:
        return np.asarray(
            [np.concatenate([np.asarray(p, dtype=float).reshape(-1) for p in trial]) for trial in comp.results],
            dtype=float,
        )

    # comp.results[trial][i] is the value of output_CIM.output_ports[i] (possibly
    # multi-width).  Those ports are named OUTPUT_CIM_<node>_<port>, so map each
    # batched output spec to its port and concatenate in spec order — a node may
    # expose several ports (e.g. DDM DECISION_OUTCOME + RESPONSE_TIME) and a
    # composition may expose ports the batched plan does not model.
    cim_names = [op.name for op in comp.output_CIM.output_ports]
    per_port = {
        nm: np.asarray([np.asarray(comp.results[t][i], dtype=float).reshape(-1) for t in range(num_trials)])
        for i, nm in enumerate(cim_names)
    }
    selected = [
        per_port[next(nm for nm in cim_names if f"{spec.node}_{spec.port}" in nm)]
        for spec in output_specs
    ]
    return np.concatenate(selected, axis=1)


def _assert_matches_pnl_python(comp, inputs, *, max_steps=None, atol=1e-4, seed=0):
    """Batched (triton_cpu) outcomes must equal PNL Python-mode outcomes."""

    kwargs = {} if max_steps is None else {"max_steps": max_steps}
    plan = BatchedCompositionCompiler.compile(comp, backend="triton_cpu", **kwargs)
    batched = plan.run(inputs=inputs, parameter_sets=[{}], num_estimates=1, seed=seed)
    reference = _pnl_python_outcomes(comp, inputs, output_specs=plan.ir.graph.outputs)

    # batched.values: [param, subject, trial, estimate, outcome]
    got = batched.values[0, 0, :, 0, :]
    np.testing.assert_allclose(got, reference, atol=atol, rtol=1e-4)
    return batched


def test_linear_transfer_matches_pnl_python():
    mech = pnl.TransferMechanism(
        input_shapes=2, function=pnl.Linear(slope=2.0, intercept=1.0), name="linear"
    )
    comp = pnl.Composition(pathways=mech)
    _assert_matches_pnl_python(comp, {mech: np.array([[1.0, 2.0], [3.0, 4.0]])})


def test_logistic_transfer_matches_pnl_python():
    mech = pnl.TransferMechanism(
        input_shapes=2, function=pnl.Logistic(gain=2.0), name="logistic"
    )
    comp = pnl.Composition(pathways=mech)
    _assert_matches_pnl_python(comp, {mech: np.array([[0.5, -1.0], [0.0, 2.0]])})


def test_mapping_projection_graph_matches_pnl_python():
    source = pnl.TransferMechanism(
        input_shapes=2, function=pnl.Linear(slope=1.0, intercept=0.0), name="source"
    )
    target = pnl.TransferMechanism(
        input_shapes=1, function=pnl.Linear(slope=3.0, intercept=1.0), name="target"
    )
    comp = pnl.Composition(pathways=[[source, pnl.MappingProjection(matrix=[[1.0], [2.0]]), target]])
    _assert_matches_pnl_python(comp, {source: np.array([[2.0, 4.0], [1.0, 1.0]])})


def test_ddm_deterministic_matches_pnl_python():
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0, rate=1.0, noise=0.0, threshold=0.05,
            non_decision_time=0.0, time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    comp = pnl.Composition(pathways=decision)
    _assert_matches_pnl_python(comp, {decision: np.array([[1.0], [-1.0]])}, max_steps=64)


def test_ddm_behind_transfer_deterministic_matches_pnl_python():
    source = pnl.TransferMechanism(input_shapes=1, name="stimulus")
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0, rate=1.0, noise=0.0, threshold=0.05,
            non_decision_time=0.0, time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    comp = pnl.Composition(pathways=[[source, decision]])
    _assert_matches_pnl_python(comp, {source: np.array([[1.0], [-1.0]])}, max_steps=64)


def _make_bare_lca(*, leak, competition, self_excitation, gain, dt, cue):
    """An isolated width-2 LCA driven by a cue (-> step count), read out by an
    identity transfer (the LCA is a stateful intermediate node, never terminal).
    """

    task = pnl.TransferMechanism(input_shapes=2, function=pnl.Linear, name="Task")
    cue_in = pnl.TransferMechanism(input_shapes=1, function=pnl.Linear, name="Cue")
    lca = pnl.LCAMechanism(
        input_shapes=2, function=pnl.Logistic(gain=gain), leak=leak, competition=competition,
        self_excitation=self_excitation, noise=0.0, termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=1200, time_step_size=dt, name="LCA",
    )
    readout = pnl.TransferMechanism(input_shapes=2, function=pnl.Linear(slope=1.0), name="Readout")
    ctl = pnl.ControlMechanism(
        monitor_for_control=cue_in,
        control_signals=[(pnl.TERMINATION_THRESHOLD, lca)],
        modulation=pnl.OVERRIDE,
    )
    comp = pnl.Composition()
    for node in (task, cue_in, lca, readout, ctl):
        comp.add_node(node)
    comp.add_projection(sender=task, receiver=lca)
    comp.add_projection(pnl.MappingProjection(matrix=np.eye(2)), sender=lca, receiver=readout)
    return comp, task, cue_in


def test_bare_lca_width2_matches_documented_recurrence():
    """Isolated width-2 LCA op: lowers/runs standalone and computes its specified
    leaky-competing recurrence.

    The batched width-2 LCA is a *documented approximation*, not a full PsyNeuLink
    ``LCAMechanism`` (BATCH_COMPILE_WIP.md, "LCA Caveats"); it initializes its
    activation state at 0 rather than ``logistic(0)``, so it is exactly
    PNL-equivalent only in the wiring it was tuned for (covered by the
    stability-flexibility reference test).  Here we isolate the op and check it
    computes the recurrence it claims, for a cue-driven step count.
    """

    leak, competition, self_excitation, gain, dt, cue = 0.5, 1.0, 0.0, 1.0, 0.1, 5.0
    comp, task, cue_in = _make_bare_lca(
        leak=leak, competition=competition, self_excitation=self_excitation, gain=gain, dt=dt, cue=cue,
    )
    batched = BatchedCompositionCompiler.compile(comp, backend="triton_cpu", max_steps=64).run(
        inputs={task: np.array([[1.0, 0.0]]), cue_in: np.array([[cue]])},
        parameter_sets=[{}], num_estimates=1, seed=0,
    )
    got = batched.values[0, 0, 0, 0, :]

    pre = np.zeros(2)
    act = np.zeros(2)
    inp = np.array([1.0, 0.0])
    for _ in range(int(np.ceil(cue))):
        rec = np.array([
            self_excitation * act[0] - competition * act[1],
            -competition * act[0] + self_excitation * act[1],
        ])
        pre = pre + (inp + rec - leak * pre) * dt
        act = 1.0 / (1.0 + np.exp(-gain * pre))

    np.testing.assert_allclose(got, act, atol=1e-5)


def _make_stab_flex_deterministic():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_stab_flex_pec_fit import generate_trial_sequence, make_input_dict, make_stab_flex

    comp = make_stab_flex(
        lca_time_step_size=0.01, ddm_time_step_size=0.01,
        threshold=0.05, ddm_noise=0.0, lca_noise=0.0,
    )
    task, stimulus, cue, correct = generate_trial_sequence(16, 0.5, seed=3)
    inputs = make_input_dict(comp, task[:2], stimulus[:2], cue[:2], correct[:2])
    return comp, inputs


def test_stability_flexibility_lca_deterministic_matches_pnl_python():
    """The stateful LCA+DDM graph (noise=0) must match PNL Python-mode outcomes.

    This is the correctness check for the batched (width-2) LCA lowering against
    the real PsyNeuLink ``LCAMechanism`` path.
    """

    comp, inputs = _make_stab_flex_deterministic()
    # The composition contains a real LCAMechanism (the stateful node under test).
    assert any(type(node).__name__ == "LCAMechanism" for node in comp.nodes)
    _assert_matches_pnl_python(comp, inputs, max_steps=256, seed=3)


def test_ddm_stochastic_matches_pnl_python_statistics():
    """Zero-drift DDM: batched and PNL summary statistics agree over many samples."""

    threshold, noise, dt = 0.2, 0.3, 0.01
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0, rate=1.0, noise=noise, threshold=threshold,
            non_decision_time=0.0, time_step_size=dt,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    comp = pnl.Composition(pathways=decision)

    n = 256
    plan = BatchedCompositionCompiler.compile(comp, backend="triton_cpu", max_steps=2000)
    batched = plan.run(inputs={decision: np.array([[0.0]])}, parameter_sets=[{}], num_estimates=n, seed=7)
    bvals = batched.values[0, 0, 0, :, :]

    # PNL reference samples: re-seed the integrator's random_state per run.
    from psyneulink.core.globals.utilities import _SeededPhilox

    ref = []
    for i in range(n):
        decision.function.parameters.random_state.set(_SeededPhilox([i + 1]))
        comp.run(inputs={decision: [[0.0]]}, execution_mode=pnl.ExecutionMode.Python)
        ref.append([float(np.asarray(p).reshape(-1)[0]) for p in comp.results[-1]])
    ref = np.asarray(ref, dtype=float)

    # Zero drift -> ~50/50 decisions; means should agree within sampling tolerance.
    assert abs(bvals[:, 0].mean() - ref[:, 0].mean()) < 0.12
    assert abs(bvals[:, 1].mean() - ref[:, 1].mean()) < 0.1
    assert np.all(np.isfinite(bvals))
