"""Acceptance contract for the first executable co-evolving CSI model.

This uses the real CSI surrogate rather than a compiler-shaped stand-in.  The
accepted slice deliberately keeps the cue transform identity-shaped while it
exercises the controlled LCA transition, the co-evolving drift/DDM region, the
collapsing threshold, persistent LCA state, and the two finished-gated output
mechanisms.
"""

import importlib.util
from pathlib import Path
import re

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompositionCompiler,
    batched_node_op,
    unregister_batched_instance_op,
)
from psyneulink.core.batched.backend.triton.graph_emit import (
    triton_graph_kernel_source,
)
from psyneulink.core.batched.graph import COEVOLVING_GRAPH_FUSION
from psyneulink.core.batched.kernel_ir import iter_kernel_ops
from psyneulink.core.batched.prep import normalize_parameter_sets


pytestmark = [pytest.mark.batched, pytest.mark.composition]


_DRIFT_NODE_NAME = "Drift Rate Value"
_CSI_PATH = (
    Path(__file__).resolve().parents[3]
    / "Scripts"
    / "Debug"
    / "pec_batch_compile"
    / "csi_model_surrogate.py"
)
_EXPECTED = np.asarray(
    [
        [1.0, 0.54],
        [1.0, 0.59],
    ],
    dtype=float,
)


def _csi_drift_rate(x0, x1, x2, x3, x4, x5, x6):
    """Inspectable Triton transcription of the CSI nested-logistic UDF."""

    a = 1.0 / (1.0 + tl.exp(-((x0 - x1) + 4.0 * x4 - 4.0)))
    b = 1.0 / (1.0 + tl.exp(-((x1 - x0) + 4.0 * x4 - 4.0)))
    c = 1.0 / (1.0 + tl.exp(-((x2 - x3) + 4.0 * x5 - 4.0)))
    d = 1.0 / (1.0 + tl.exp(-((x3 - x2) + 4.0 * x5 - 4.0)))
    positive = 1.0 / (1.0 + tl.exp(-(a - b + c - d)))
    negative = 1.0 / (1.0 + tl.exp(-(-a + b - c + d)))
    return (positive - negative) * x6


@pytest.fixture
def registered_csi_drift_rate():
    batched_node_op(_DRIFT_NODE_NAME)(_csi_drift_rate)
    try:
        yield
    finally:
        unregister_batched_instance_op(_DRIFT_NODE_NAME)


def _make_stab_flex():
    module_spec = importlib.util.spec_from_file_location(
        "_batched_csi_model_surrogate",
        _CSI_PATH,
    )
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.make_stab_flex(
        iti=0,
        csi_repeat=0,
        csi_switch=1,
        threshold_collapse=-0.001,
        ddm_noise=0.0,
        lca_noise=0.0,
    )


def _node(composition, base_name):
    matches = tuple(
        node
        for node in composition.nodes
        if re.sub(r"-\d+$", "", node.name) == base_name
    )
    assert len(matches) == 1
    return matches[0]


def _model(*, correct_values=None, ddm_rate=None, one_trial=False):
    composition = _make_stab_flex()
    stimulus = _node(composition, "Stimulus Input")
    task = _node(composition, "Task Input")
    correct = _node(composition, "Correct Response")
    cue = _node(composition, "Cue Stimulus Interval")
    decision_gate = _node(composition, "DECISION_GATE")
    response_gate = _node(composition, "RESPONSE_GATE")
    if ddm_rate is not None:
        ddm = _node(composition, "DDM")
        ddm.function.parameters.rate.set(float(ddm_rate))
    trial_slice = slice(None, 1) if one_trial else slice(None)
    if correct_values is None:
        correct_values = [[1.0], [-1.0]]
    inputs = {
        stimulus: np.asarray(
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            dtype=float,
        )[trial_slice],
        task: np.asarray(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=float,
        )[trial_slice],
        correct: np.asarray(correct_values, dtype=float)[trial_slice],
        cue: np.asarray([[1.0], [3.0]], dtype=float)[trial_slice],
    }
    outputs = (decision_gate.output_port, response_gate.output_port)
    return composition, inputs, outputs


def _selected_python_results(composition, outputs):
    result_indices = []
    for output in outputs:
        matches = tuple(
            index
            for index, cim_input in enumerate(composition.output_CIM.input_ports)
            if any(
                projection.sender is output
                for projection in cim_input.path_afferents
            )
        )
        assert len(matches) == 1
        result_indices.append(matches[0])
    return np.asarray(
        [
            [
                float(np.asarray(trial[index]).reshape(-1)[0])
                for index in result_indices
            ]
            for trial in composition.results
        ],
        dtype=float,
    )


def _run_compiled_csi(backend, **model_options):
    composition, inputs, outputs = _model(**model_options)
    plan = BatchedCompositionCompiler.compile(
        composition,
        backend=backend,
        outputs=outputs,
        max_steps=128,
    )
    result = plan.run(
        inputs=inputs,
        parameter_sets=[{}],
        num_estimates=1,
        seed=0,
    )
    return result.values[0, 0, :, 0, :]


def test_csi_compiles_to_one_lane_local_coevolving_region(
    registered_csi_drift_rate,
):
    composition, _, outputs = _model()
    plan = BatchedCompositionCompiler.compile(
        composition,
        backend="triton_cpu",
        outputs=outputs,
        max_steps=128,
    )
    kernel = plan.kernel_ir
    graph = kernel.graph

    assert kernel.executable
    assert graph.executable
    assert kernel.fusion_kind == COEVOLVING_GRAPH_FUSION
    assert len(kernel.modulations) == 1
    assert len(kernel.effective_parameters) == 1
    assert len(kernel.finished_values) == 2
    assert tuple(output.node for output in graph.outputs) == tuple(
        output.owner.name for output in outputs
    )

    all_ops = iter_kernel_ops(kernel)
    regions = tuple(
        op
        for op in all_ops
        if op.kind == "ForPasses"
        and op.attrs.get("trace_kind") == "lane_local_coevolving"
    )
    assert len(regions) == 1
    region = regions[0]
    assert region.attrs["declaration_only"] is False
    assert region.attrs["max_steps"] == 128

    stepper = graph.node(_node(composition, "Task Activations [C1, C2]").name)
    terminator = graph.node(_node(composition, "DDM").name)
    finished_by_component = {
        value.component_id: value for value in kernel.finished_values
    }
    assert region.attrs["stepper_component_id"] == stepper.component_id
    assert region.attrs["terminator_component_id"] == terminator.component_id
    assert region.attrs["stepper_finished_value_id"] == (
        finished_by_component[stepper.component_id].value_id
    )
    assert region.attrs["terminator_finished_value_id"] == (
        finished_by_component[terminator.component_id].value_id
    )

    modulation = kernel.modulations[0]
    assert region.attrs["effective_parameter_id"] == (
        modulation.effective_parameter_id
    )
    assert len(region.inputs) == 1
    assert region.inputs[0].name == (
        f"effective:{modulation.effective_parameter_id}"
    )
    assert region.attrs["terminator_trial_states"] == (
        ("value", 1, 0.0),
        ("steps", 1, 0.0),
        ("finished", 1, 0.0),
    )
    assert region.attrs["terminator_initial_control_value"] == 1.0
    assert region.attrs["terminator_control_storage"] == "lane_persistent"
    assert region.attrs["terminator_control_update"] == (
        "ordered_threshold_override"
    )
    assert region.attrs["completion_cleanup"] == (
        "fold_terminator_control_state"
    )

    region_steps = tuple(
        op
        for op in region.attrs["body"]
        if op.kind in {"CallMechanism", "StepMechanism"}
    )
    assert {op.target for op in region_steps} >= {
        stepper.name,
        terminator.name,
    }
    assert sum(op.kind == "InitializeEffectiveParameter" for op in kernel.ops) == 1
    assert sum(op.kind == "ApplyModulation" for op in all_ops) == 1
    assert sum(op.kind == "StoreOutput" for op in all_ops) == 2

    source = triton_graph_kernel_source(kernel)
    assert re.search(
        r"n\d+_coevolving_required_passes = .*\.to\(tl\.int64\)",
        source,
    )
    assert re.search(
        r"n\d+_coevolving_start_pass = "
        r"n\d+_coevolving_required_passes - 1$",
        source,
        flags=re.MULTILINE,
    )
    assert re.search(
        r"n\d+_coevolving_held_control = tl\.full\(\(BLOCK,\), 1\.0, ",
        source,
    )
    assert "coevolving_has_control_update" in source


@pytest.mark.parametrize(
    "argument, replacement",
    (
        ("threshold", 0.1),
        ("threshold_collapse", -0.002),
        ("noise", 0.1),
        ("starting_value", 0.01),
        ("offset", 0.01),
    ),
)
def test_csi_first_coevolving_ddm_boundary_is_default_only(
    registered_csi_drift_rate,
    argument,
    replacement,
):
    composition, _, outputs = _model()
    plan = BatchedCompositionCompiler.compile(
        composition,
        backend="triton_cpu",
        outputs=outputs,
        max_steps=128,
    )
    terminator = next(
        node for node in plan.kernel_ir.graph.nodes if node.component_type == "DDM"
    )
    parameter_name = terminator.params[argument]
    parameter = next(
        item for item in plan.ir.params if item.name == parameter_name
    )

    assert not parameter.runtime_mutable
    assert parameter.runtime_constraint == (
        "first coevolving DDM boundary is frozen in KernelIR"
    )
    with pytest.raises(ValueError, match="is fixed at"):
        normalize_parameter_sets(
            [{parameter_name: replacement}],
            plan.ir,
        )


@pytest.mark.triton
@pytest.mark.triton_interpreter
def test_csi_deterministic_interpreter_matches_fresh_python(
    registered_csi_drift_rate,
):
    python_composition, python_inputs, python_outputs = _model()
    python_composition.run(
        inputs=python_inputs,
        execution_mode=pnl.ExecutionMode.Python,
    )
    expected = _selected_python_results(python_composition, python_outputs)
    np.testing.assert_allclose(expected, _EXPECTED, rtol=0.0, atol=1e-12)

    actual = _run_compiled_csi("triton_cpu")

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actual, _EXPECTED, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "model_options, expected",
    (
        (
            {"correct_values": [[0.0]], "one_trial": True},
            np.asarray([[0.0, 0.92]]),
        ),
        (
            {"ddm_rate": 40.0},
            np.asarray([[1.0, 0.33], [1.0, 0.34]]),
        ),
        (
            {"ddm_rate": 500.0},
            np.asarray([[1.0, 0.32], [1.0, 0.34]]),
        ),
    ),
    ids=(
        "boundary-crosses-zero",
        "persistent-threshold-control-value",
        "one-step-threshold-cleanup",
    ),
)
@pytest.mark.triton
@pytest.mark.triton_interpreter
def test_csi_interpreter_matches_ddm_boundary_transition_oracle(
    registered_csi_drift_rate,
    model_options,
    expected,
):
    python_composition, python_inputs, python_outputs = _model(**model_options)
    python_composition.run(
        inputs=python_inputs,
        execution_mode=pnl.ExecutionMode.Python,
    )
    python_result = _selected_python_results(
        python_composition,
        python_outputs,
    )
    np.testing.assert_allclose(python_result, expected, rtol=0.0, atol=1e-12)

    actual = _run_compiled_csi("triton_cpu", **model_options)

    np.testing.assert_allclose(actual, python_result, rtol=1e-5, atol=1e-6)


@pytest.mark.triton
@pytest.mark.triton_gpu
def test_csi_deterministic_gpu_matches_oracle(registered_csi_drift_rate):
    actual = _run_compiled_csi("triton")

    np.testing.assert_allclose(actual, _EXPECTED, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        _run_compiled_csi(
            "triton",
            correct_values=[[0.0]],
            one_trial=True,
        ),
        [[0.0, 0.92]],
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _run_compiled_csi("triton", ddm_rate=40.0),
        [[1.0, 0.33], [1.0, 0.34]],
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _run_compiled_csi("triton", ddm_rate=500.0),
        [[1.0, 0.32], [1.0, 0.34]],
        rtol=1e-5,
        atol=1e-6,
    )
