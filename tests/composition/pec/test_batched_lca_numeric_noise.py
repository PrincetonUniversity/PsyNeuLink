"""Numeric-noise semantics for the batched width-two LCA implementation."""

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompositionCompiler,
    batched_node_op,
    unregister_batched_instance_op,
)

from test_batched_csi_coevolving_acceptance import (
    _DRIFT_NODE_NAME,
    _csi_drift_rate,
    _make_stab_flex,
    _node,
    _selected_python_results,
)


pytestmark = [pytest.mark.batched, pytest.mark.composition]


_LCA_INPUTS = np.asarray(
    [
        [1.0, -0.25],
        [-0.6, 0.85],
        [0.35, -1.1],
    ],
    dtype=float,
)


def _standalone_lca(noise):
    lca = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(gain=1.3),
        leak=0.3,
        competition=0.4,
        self_excitation=0.2,
        noise=noise,
        time_step_size=0.2,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=3,
        reset_stateful_function_when=pnl.Never(),
        name="numeric noise LCA",
    )
    return pnl.Composition(pathways=lca), lca


def _python_lca_values(noise):
    composition, lca = _standalone_lca(noise)
    composition.run(
        inputs={lca: _LCA_INPUTS.copy()},
        execution_mode=pnl.ExecutionMode.Python,
    )
    return np.asarray(
        [np.asarray(trial[0], dtype=float).reshape(-1) for trial in composition.results]
    )


@pytest.mark.parametrize(
    "noise",
    (
        pytest.param(1.0e-9, id="near-zero"),
        pytest.param(0.125, id="positive"),
        pytest.param(-0.2, id="negative"),
        pytest.param(np.asarray([0.125, 0.125]), id="broadcast-vector"),
    ),
)
def test_standalone_numeric_lca_noise_matches_python(noise, batched_backend):
    expected = _python_lca_values(noise)
    composition, lca = _standalone_lca(noise)
    plan = BatchedCompositionCompiler.compile(
        composition,
        backend=batched_backend,
        outputs=(lca.output_port,),
        max_steps=16,
    )
    result = plan.run(
        inputs={lca: _LCA_INPUTS.copy()},
        parameter_sets=[{}],
        num_estimates=1,
        seed=0,
    )

    np.testing.assert_allclose(
        result.values[0, 0, :, 0, :],
        expected,
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_standalone_lca_noise_is_a_runtime_parameter(batched_backend):
    noise_values = (0.125, -0.2, 0.0)
    composition, lca = _standalone_lca(noise_values[0])
    plan = BatchedCompositionCompiler.compile(
        composition,
        backend=batched_backend,
        outputs=(lca.output_port,),
        max_steps=16,
    )
    noise_name = plan.ir.graph.node(lca.name).params["noise"]
    noise_parameter = next(
        parameter for parameter in plan.ir.params if parameter.name == noise_name
    )
    assert noise_parameter.default == noise_values[0]
    assert noise_parameter.runtime_mutable

    result = plan.run(
        inputs={lca: _LCA_INPUTS.copy()},
        parameter_sets=[{noise_name: noise} for noise in noise_values],
        num_estimates=1,
        seed=0,
    )
    expected = np.asarray([_python_lca_values(noise) for noise in noise_values])

    assert not np.allclose(result.values[0], result.values[1])
    np.testing.assert_allclose(
        result.values[:, 0, :, 0, :],
        expected,
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_numeric_lca_initialization_policy_is_reauthenticated():
    composition, lca = _standalone_lca(0.125)
    plan = BatchedCompositionCompiler.compile(
        composition,
        backend="triton_cpu",
        outputs=(lca.output_port,),
        max_steps=16,
    )
    kernel = plan.kernel_ir
    nodes = tuple(
        replace(
            node,
            attrs={**node.attrs, "initialize_noise_sender": False},
        )
        if node.name == lca.name
        else node
        for node in kernel.graph.nodes
    )
    with pytest.raises(ValueError, match="initialization policy"):
        replace(kernel, graph=replace(kernel.graph, nodes=nodes))


def _scheduled_stepwise_lca(reset_condition):
    lca = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(gain=1.3),
        leak=0.3,
        competition=0.4,
        self_excitation=0.2,
        noise=0.125,
        time_step_size=0.2,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=3,
        execute_until_finished=False,
        reset_stateful_function_when=reset_condition,
        name="scheduled numeric noise LCA",
    )
    follower = pnl.TransferMechanism(input_shapes=2, name="LCA follower")
    composition = pnl.Composition()
    composition.add_nodes([lca, follower])
    composition.add_projection(sender=lca, receiver=follower)
    composition.scheduler.add_condition(lca, pnl.Always())
    composition.scheduler.add_condition(follower, pnl.WhenFinished(lca))
    return composition, lca, follower


@pytest.mark.parametrize(
    "reset_factory",
    (
        pytest.param(pnl.Never, id="persistent"),
        pytest.param(pnl.AtTrialStart, id="trial-reset"),
    ),
)
def test_numeric_lca_noise_matches_python_across_reset_policies(
    reset_factory,
    batched_backend,
):
    python_composition, python_lca, _ = _scheduled_stepwise_lca(reset_factory())
    python_composition.run(
        inputs={python_lca: _LCA_INPUTS.copy()},
        execution_mode=pnl.ExecutionMode.Python,
    )
    expected = np.asarray(
        [
            np.asarray(trial[0], dtype=float).reshape(-1)
            for trial in python_composition.results
        ]
    )

    composition, lca, follower = _scheduled_stepwise_lca(reset_factory())
    plan = BatchedCompositionCompiler.compile(
        composition,
        backend=batched_backend,
        outputs=(follower.output_port,),
        max_steps=16,
    )
    result = plan.run(
        inputs={lca: _LCA_INPUTS.copy()},
        parameter_sets=[{}],
        num_estimates=1,
        seed=0,
    )

    np.testing.assert_allclose(
        result.values[0, 0, :, 0, :],
        expected,
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def _constant_noise():
    return 0.125


@pytest.mark.parametrize(
    "noise_factory",
    (
        pytest.param(lambda: _constant_noise, id="callable"),
        pytest.param(lambda: pnl.NormalDist(), id="distribution"),
        pytest.param(lambda: [0.1, 0.2], id="non-broadcast-vector"),
        pytest.param(lambda: np.nan, id="nan"),
        pytest.param(lambda: np.inf, id="infinity"),
        pytest.param(lambda: 1.0e40, id="outside-fp32"),
    ),
)
def test_non_scalar_or_nonfinite_lca_noise_remains_fail_closed(
    noise_factory: Callable,
):
    composition, lca = _standalone_lca(noise_factory())
    report = BatchedCompositionCompiler.diagnose(
        composition,
        backend="triton_cpu",
        outputs=(lca.output_port,),
        max_steps=16,
    )

    matches = tuple(
        diagnostic
        for diagnostic in report.model_diagnostics
        if diagnostic.component == lca.name
        and diagnostic.reason == "unsupported LCA noise for batched v2"
    )
    assert not report.model_supported
    assert len(matches) == 1
    assert matches[0].detail == (
        "requires a finite float32 scalar or broadcast-scalar numeric value"
    )


@pytest.fixture
def registered_numeric_noise_csi_drift_rate():
    batched_node_op(_DRIFT_NODE_NAME)(_csi_drift_rate)
    try:
        yield
    finally:
        unregister_batched_instance_op(_DRIFT_NODE_NAME)


def _csi_numeric_noise_model(noise):
    composition = _make_stab_flex(
        iti=0,
        csi_repeat=0,
        csi_switch=1,
        threshold_collapse=-0.001,
        ddm_noise=0.0,
        lca_noise=noise,
    )
    stimulus = _node(composition, "Stimulus Input")
    task = _node(composition, "Task Input")
    correct = _node(composition, "Correct Response")
    cue = _node(composition, "Cue Stimulus Interval")
    decision_gate = _node(composition, "DECISION_GATE")
    response_gate = _node(composition, "RESPONSE_GATE")
    inputs = {
        stimulus: np.asarray(
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            dtype=float,
        ),
        task: np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float),
        correct: np.asarray([[1.0], [-1.0]], dtype=float),
        cue: np.asarray([[1.0], [3.0]], dtype=float),
    }
    return composition, inputs, (decision_gate.output_port, response_gate.output_port)


def test_csi_numeric_lca_noise_matches_fresh_python(
    registered_numeric_noise_csi_drift_rate,
    batched_backend,
):
    noise_values = (0.05, -0.05, 0.2)
    expected = []
    for noise in noise_values:
        python_composition, python_inputs, python_outputs = (
            _csi_numeric_noise_model(noise)
        )
        python_composition.run(
            inputs=python_inputs,
            execution_mode=pnl.ExecutionMode.Python,
        )
        expected.append(
            _selected_python_results(python_composition, python_outputs)
        )
    expected = np.asarray(expected)

    composition, inputs, outputs = _csi_numeric_noise_model(noise_values[0])
    plan = BatchedCompositionCompiler.compile(
        composition,
        backend=batched_backend,
        outputs=outputs,
        max_steps=128,
    )
    lca = _node(composition, "Task Activations [C1, C2]")
    noise_name = plan.ir.graph.node(lca.name).params["noise"]
    noise_parameter = next(
        parameter for parameter in plan.ir.params if parameter.name == noise_name
    )
    assert noise_parameter.default == noise_values[0]
    assert noise_parameter.runtime_mutable

    result = plan.run(
        inputs=inputs,
        parameter_sets=(
            {},
            {noise_name: noise_values[1]},
            {noise_name: noise_values[2]},
        ),
        num_estimates=1,
        seed=0,
    )

    assert not np.allclose(result.values[0], result.values[1])
    np.testing.assert_allclose(
        result.values[:, 0, :, 0, :],
        expected,
        rtol=1.0e-5,
        atol=1.0e-6,
    )
