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

import sys
from pathlib import Path

import numpy as np
import pytest

import psyneulink as pnl

from psyneulink.core.batched import (
    BatchedCompileError,
    BatchedCompositionCompiler,
    BatchedDiagnosticCode,
    batched_node_op,
    unregister_batched_instance_op,
)


pytestmark = [
    pytest.mark.batched,
    pytest.mark.composition,
    pytest.mark.triton,
    pytest.mark.triton_interpreter,
]


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


def _make_bare_lca(*, leak, competition, self_excitation, gain, dt, steps):
    """An isolated deterministic width-2 LCA with a static trial step count."""

    lca = pnl.LCAMechanism(
        input_shapes=2, function=pnl.Logistic(gain=gain), leak=leak, competition=competition,
        self_excitation=self_excitation, noise=0.0, termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=steps, time_step_size=dt,
        reset_stateful_function_when=pnl.Never(), name="LCA",
    )
    return pnl.Composition(pathways=lca), lca


def test_bare_lca_width2_matches_pnl_python():
    """The deterministic static-threshold width-2 subset must match PsyNeuLink.

    This replaces the legacy handwritten recurrence oracle with a static trial
    threshold inside the exact deterministic support boundary. The shared
    semantic corpus separately constructs fresh models to pin first-step
    initialization, persistence, Logistic parameters, and threshold behavior.
    """

    leak, competition, self_excitation, gain, dt, steps = 0.5, 1.0, 0.0, 1.0, 0.1, 5
    comp, lca = _make_bare_lca(
        leak=leak,
        competition=competition,
        self_excitation=self_excitation,
        gain=gain,
        dt=dt,
        steps=steps,
    )
    _assert_matches_pnl_python(
        comp,
        {lca: np.array([[1.0, 0.0]])},
        max_steps=64,
    )


def test_fires_once_integrating_transfer_matches_pnl_python():
    """A fires-once, reset-each-trial integrator_mode transfer lowered as a
    stateless single affine integrator step must match PNL Python mode.

    Covers a non-trivial AdaptiveIntegrator rate (0.5) and a non-identity output
    function (Logistic), so the affine fold `function((1-rate)*init + rate*x)` is
    actually exercised, not just identity.
    """

    mech = pnl.TransferMechanism(
        input_shapes=2, function=pnl.Logistic(gain=1.5),
        integrator_mode=True, integration_rate=0.5,
        reset_stateful_function_when=pnl.AtTrialStart(), name="integ",
    )
    comp = pnl.Composition()
    comp.add_node(mech)
    comp.scheduler.add_condition(mech, pnl.AtPass(0))
    _assert_matches_pnl_python(comp, {mech: np.array([[0.4, -0.7]])})


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


def test_stability_flexibility_explicit_at_pass_zero_origins_matches_pnl_python():
    """Adding explicit ``AtPass(0)`` conditions on the origin nodes must not
    change lowering or results.

    The CSI surrogate model schedules its origins with ``AtPass(0)`` ("fire only
    on pass 0").  That is exactly the batched origin default (each node computes
    once per trial), so the schedule classifier accepts it as a static graph.
    This pins the equivalence: the deterministic stab-flex model still lowers and
    matches PNL Python mode once those explicit conditions are present.
    """

    comp, inputs = _make_stab_flex_deterministic()
    for origin in comp.get_nodes_by_role(pnl.NodeRole.ORIGIN):
        comp.scheduler.add_condition(origin, pnl.AtPass(0))

    report = BatchedCompositionCompiler.diagnose(comp, backend="triton_cpu")
    assert report.is_supported
    assert not report.rejected_conditions
    assert report.metadata["schedule_kind"] == "static_graph"

    _assert_matches_pnl_python(comp, inputs, max_steps=256, seed=3)


_CSI_DIR = Path(__file__).resolve().parents[3] / "Scripts" / "Debug" / "pec_batch_compile"
_UNMODELED_CSI_CONTROL_DETAIL = (
    "coevolving Always/WhenFinished execution falls outside the typed "
    "controlled-finished subset and requires executable conditional pass regions"
)


def _make_csi_surrogate(**kwargs):
    """Build the CSI reference model retained from the original runtime tests."""

    sys.path.insert(0, str(_CSI_DIR))
    from csi_model_surrogate import make_stab_flex

    return make_stab_flex(**kwargs)


def _register_csi_drift_rate():
    # Faithful tl transcription of csi_model_surrogate.drift_rate_fct.
    @batched_node_op("Drift Rate Value")
    def drift_rate(x0, x1, x2, x3, x4, x5, x6):
        a = 1.0 / (1.0 + tl.exp(-((x0 - x1) + 4.0 * x4 - 4.0)))
        b = 1.0 / (1.0 + tl.exp(-((x1 - x0) + 4.0 * x4 - 4.0)))
        c = 1.0 / (1.0 + tl.exp(-((x2 - x3) + 4.0 * x5 - 4.0)))
        d = 1.0 / (1.0 + tl.exp(-((x3 - x2) + 4.0 * x5 - 4.0)))
        pos = 1.0 / (1.0 + tl.exp(-(a - b + c - d)))
        neg = 1.0 / (1.0 + tl.exp(-(-a + b - c + d)))
        return (pos - neg) * x6


def _lca_termination_override(composition):
    """Find the semantic control edge without relying on CSI component names."""

    matches = []
    for node in composition.nodes:
        if not isinstance(node, pnl.ControlMechanism):
            continue
        for signal in node.control_signals:
            for projection in signal.efferents:
                receiver = projection.receiver
                if (
                    isinstance(receiver.owner, pnl.LCAMechanism)
                    and receiver.name == pnl.TERMINATION_THRESHOLD
                ):
                    matches.append(node)
    assert len(matches) == 1
    return matches[0]


def _assert_csi_control_rejection(composition):
    controller = _lca_termination_override(composition)
    report = BatchedCompositionCompiler.diagnose(
        composition,
        backend="triton_cpu",
        max_steps=800,
    )

    assert not report.model_supported
    assert len(report.model_diagnostics) == 1
    diagnostic = report.model_diagnostics[0]
    assert diagnostic.component == controller.name
    assert diagnostic.component_id == f"node:{controller.name}"
    assert diagnostic.code == BatchedDiagnosticCode.MODEL_SCHEDULE_NOT_EXECUTABLE
    assert diagnostic.reason == "batched schedule kind is not executable yet"
    assert diagnostic.detail == _UNMODELED_CSI_CONTROL_DETAIL
    return report


def test_csi_drift_rate_udf_registration_exposes_schedule_rejection():
    """The UDF clears its blocker; CSI remains outside executable control IR."""

    comp = _make_csi_surrogate(iti=0, csi_repeat=0, csi_switch=0)
    before = BatchedCompositionCompiler.diagnose(comp, backend="triton_cpu")
    udf_diagnostics = [
        diagnostic
        for diagnostic in before.model_diagnostics
        if diagnostic.code == BatchedDiagnosticCode.MODEL_FUNCTION_UNSUPPORTED
        and "Drift Rate Value" in diagnostic.component
    ]
    assert len(udf_diagnostics) == 1

    try:
        _register_csi_drift_rate()
        _assert_csi_control_rejection(comp)
    finally:
        unregister_batched_instance_op("Drift Rate Value")


@pytest.mark.parametrize(
    "iti, threshold_collapse",
    [
        pytest.param(0, 0.0, id="zero_iti_fixed_threshold"),
        pytest.param(0, -0.001, id="zero_iti_collapsing_threshold"),
        pytest.param(10, 0.0, id="delayed_iti_fixed_threshold"),
        pytest.param(10, -0.001, id="delayed_iti_collapsing_threshold"),
    ],
)
def test_csi_noise_free_model_variants_reject_unmodeled_control(
    iti,
    threshold_collapse,
):
    """Retain the former end-to-end, ITI, collapse, and Python/LLVM variants.

    These deterministic variants cannot honestly reach an execution-mode
    oracle until KernelIR executes their controlled LCA/DDM pass region.
    """

    try:
        _register_csi_drift_rate()
        comp = _make_csi_surrogate(
            iti=iti,
            csi_repeat=0,
            csi_switch=0,
            threshold_collapse=threshold_collapse,
            ddm_noise=0.0,
            lca_noise=0.0,
        )
        _assert_csi_control_rejection(comp)
    finally:
        unregister_batched_instance_op("Drift Rate Value")


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


def test_csi_multi_parameter_lane_model_rejects_before_launch():
    """The former non-decision-time lane sweep must not bypass model analysis."""

    try:
        _register_csi_drift_rate()
        comp = _make_csi_surrogate(
            iti=0,
            csi_repeat=0,
            csi_switch=0,
            threshold_collapse=-0.001,
            ddm_noise=0.0,
            lca_noise=0.0,
        )
        diagnosed = _assert_csi_control_rejection(comp)

        with pytest.raises(BatchedCompileError) as error:
            BatchedCompositionCompiler.compile(
                comp,
                backend="triton_cpu",
                max_steps=400,
            )

        assert error.value.capability_report == diagnosed
        diagnostic = diagnosed.model_diagnostics[0]
        assert diagnostic.formatted_reason in str(error.value)
    finally:
        unregister_batched_instance_op("Drift Rate Value")
