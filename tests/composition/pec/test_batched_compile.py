import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import psyneulink as pnl

from psyneulink.core.batched import (
    BatchedGraphIR,
    BatchedCompileError,
    BatchedCompositionCompiler,
)
from psyneulink.core.batched.kernel_ir import (
    STATEFUL_LANE_LAYOUT,
    TRIAL_LANE_LAYOUT,
    iter_kernel_ops,
    lower_to_kernel_ir,
)
from psyneulink.core.components.functions.userdefinedfunction import UserDefinedFunction
from psyneulink.core.components.functions.nonstateful.fitfunctions import PECOptimizationFunction


pytestmark = pytest.mark.usefixtures("set_threads_to_one")


def _make_ddm_comp(noise=0.0):
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0,
            rate=1.0,
            noise=noise,
            threshold=0.05,
            non_decision_time=0.0,
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    return pnl.Composition(pathways=decision), decision


def _triton_available():
    if importlib.util.find_spec("triton") is None or importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.cuda.is_available()


def _with_graph_fusion_kind(plan, fusion_kind):
    graph = replace(plan.ir.graph, fusion_kind=fusion_kind)
    ir = replace(plan.ir, graph=graph, metadata={**plan.ir.metadata, "fusion_kind": fusion_kind})
    return replace(plan, ir=ir)


@pytest.mark.composition
def test_batched_compiler_rejects_reference_backend_name():
    comp, _ = _make_ddm_comp(noise=0.0)

    with pytest.raises(BatchedCompileError, match="Unknown batched backend 'reference'"):
        BatchedCompositionCompiler.diagnose(comp, backend="reference")

    with pytest.raises(BatchedCompileError, match="Unknown batched backend 'reference'"):
        BatchedCompositionCompiler.compile(comp, backend="reference")


@pytest.mark.composition
def test_batched_ir_debug_ddm_deterministic():
    comp, decision = _make_ddm_comp(noise=0.0)
    plan = BatchedCompositionCompiler.compile(comp, backend="ir_debug")
    result = plan.run(
        inputs={decision: np.array([[1.0], [-1.0]], dtype=float)},
        parameter_sets=[{"rate": 1.0, "threshold": 0.05, "time_step_size": 0.01}],
        num_estimates=2,
        seed=1,
    )

    assert result.values.shape == (1, 1, 2, 2, 2)
    np.testing.assert_allclose(result.values[0, 0, :, :, 0], [[1.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(result.values[0, 0, :, :, 1], [[0.05, 0.05], [0.05, 0.05]])


@pytest.mark.composition
def test_batched_ir_debug_ddm_common_random_numbers():
    comp, decision = _make_ddm_comp(noise=0.2)
    plan = BatchedCompositionCompiler.compile(comp, backend="ir_debug")
    params = [
        {"rate": 0.0, "threshold": 0.05, "noise": 0.2, "time_step_size": 0.01},
        {"rate": 0.0, "threshold": 0.05, "noise": 0.2, "time_step_size": 0.01},
    ]
    result = plan.run(
        inputs={decision: np.array([[0.0], [0.0]], dtype=float)},
        parameter_sets=params,
        num_estimates=4,
        seed=7,
        common_random_numbers=True,
    )

    np.testing.assert_array_equal(result.values[0], result.values[1])


def _make_linear_projection_comp():
    source = pnl.TransferMechanism(
        input_shapes=2,
        function=pnl.Linear(slope=1.0, intercept=0.0),
        name="source",
    )
    target = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Linear(slope=3.0, intercept=1.0),
        name="target",
    )
    comp = pnl.Composition(pathways=[[source, pnl.MappingProjection(matrix=[[1.0], [2.0]]), target]])
    return comp, source, target


@pytest.mark.composition
def test_batched_ir_debug_transfer_only_generic_graph():
    mech = pnl.TransferMechanism(
        input_shapes=2,
        function=pnl.Linear(slope=2.0, intercept=1.0),
        name="linear",
    )
    comp = pnl.Composition(pathways=mech)
    plan = BatchedCompositionCompiler.compile(comp, backend="ir_debug")
    result = plan.run(
        inputs={mech: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)},
        parameter_sets=[{}],
        num_estimates=2,
    )

    assert isinstance(plan.ir.graph, BatchedGraphIR)
    assert plan.ir.model_kind == "graph"
    assert plan.ir.graph.fusion_kind == "stateless_graph"
    assert result.values.shape == (1, 1, 2, 2, 2)
    np.testing.assert_allclose(result.values[0, 0, :, :, :], [[[3.0, 5.0], [3.0, 5.0]], [[7.0, 9.0], [7.0, 9.0]]])


@pytest.mark.composition
def test_kernel_ir_transfer_only_structure():
    mech = pnl.TransferMechanism(
        input_shapes=2,
        function=pnl.Linear(slope=2.0, intercept=1.0),
        name="linear",
    )
    comp = pnl.Composition(pathways=mech)
    plan = BatchedCompositionCompiler.compile(comp, backend="ir_debug")
    kernel = lower_to_kernel_ir(plan.ir)
    op_kinds = [op.kind for op in iter_kernel_ops(kernel)]

    assert kernel.lane_layout.kind == TRIAL_LANE_LAYOUT
    assert op_kinds == ["LoadInput", "ElementwiseLinear", "StoreOutput"]


@pytest.mark.composition
def test_batched_ir_debug_mapping_projection_generic_graph():
    comp, source, _ = _make_linear_projection_comp()
    result = BatchedCompositionCompiler.compile(comp, backend="ir_debug").run(
        inputs={source: np.array([[2.0, 4.0]], dtype=float)},
        parameter_sets=[{}],
        num_estimates=1,
    )

    assert result.values.shape == (1, 1, 1, 1, 1)
    np.testing.assert_allclose(result.values[0, 0, 0, 0], [31.0])


@pytest.mark.composition
def test_kernel_ir_dense_projection_structure():
    comp, source, _ = _make_linear_projection_comp()
    plan = BatchedCompositionCompiler.compile(comp, backend="ir_debug")
    kernel = lower_to_kernel_ir(plan.ir)
    op_kinds = [op.kind for op in iter_kernel_ops(kernel)]

    assert kernel.lane_layout.kind == TRIAL_LANE_LAYOUT
    assert "DenseMatVec" in op_kinds
    assert "CombineSum" in op_kinds


@pytest.mark.composition
def test_batched_ir_debug_product_combine_generic_graph():
    left = pnl.TransferMechanism(input_shapes=2, name="left")
    right = pnl.TransferMechanism(input_shapes=2, name="right")
    product = pnl.TransferMechanism(
        input_shapes=2,
        function=pnl.Linear(slope=1.0, intercept=0.0),
        input_ports=pnl.InputPort(combine=pnl.PRODUCT),
        name="product",
    )
    comp = pnl.Composition()
    for node in (left, right, product):
        comp.add_node(node)
    comp.add_projection(sender=left, receiver=product)
    comp.add_projection(sender=right, receiver=product)

    result = BatchedCompositionCompiler.compile(comp, backend="ir_debug").run(
        inputs={
            left: np.array([[2.0, 3.0]], dtype=float),
            right: np.array([[4.0, 5.0]], dtype=float),
        },
        parameter_sets=[{}],
        num_estimates=1,
    )

    np.testing.assert_allclose(result.values[0, 0, 0, 0], [8.0, 15.0])


@pytest.mark.composition
def test_kernel_ir_product_combine_structure():
    left = pnl.TransferMechanism(input_shapes=2, name="left")
    right = pnl.TransferMechanism(input_shapes=2, name="right")
    product = pnl.TransferMechanism(
        input_shapes=2,
        function=pnl.Linear(slope=1.0, intercept=0.0),
        input_ports=pnl.InputPort(combine=pnl.PRODUCT),
        name="product",
    )
    comp = pnl.Composition()
    for node in (left, right, product):
        comp.add_node(node)
    comp.add_projection(sender=left, receiver=product)
    comp.add_projection(sender=right, receiver=product)

    plan = BatchedCompositionCompiler.compile(comp, backend="ir_debug")
    kernel = lower_to_kernel_ir(plan.ir)
    op_kinds = [op.kind for op in iter_kernel_ops(kernel)]

    assert "CombineProduct" in op_kinds


@pytest.mark.composition
def test_batched_ir_debug_ddm_behind_transfer_generic_graph():
    source = pnl.TransferMechanism(input_shapes=1, name="stimulus")
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0,
            rate=1.0,
            noise=0.0,
            threshold=0.05,
            non_decision_time=0.0,
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    comp = pnl.Composition(pathways=[[source, decision]])
    plan = BatchedCompositionCompiler.compile(comp, backend="ir_debug", max_steps=64)
    result = plan.run(
        inputs={source: np.array([[1.0], [-1.0]], dtype=float)},
        parameter_sets=[{"rate": 1.0, "threshold": 0.05, "noise": 0.0, "time_step_size": 0.01}],
        num_estimates=1,
    )

    assert plan.ir.model_kind == "graph"
    assert plan.ir.graph.fusion_kind == "ddm_graph"
    np.testing.assert_allclose(result.values[0, 0, :, 0, 0], [1.0, 0.0])
    np.testing.assert_allclose(result.values[0, 0, :, 0, 1], [0.05, 0.05])


@pytest.mark.composition
def test_kernel_ir_ddm_graph_structure():
    source = pnl.TransferMechanism(input_shapes=1, name="stimulus")
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0,
            rate=1.0,
            noise=0.0,
            threshold=0.05,
            non_decision_time=0.0,
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    comp = pnl.Composition(pathways=[[source, decision]])
    plan = BatchedCompositionCompiler.compile(comp, backend="ir_debug", max_steps=64)
    kernel = lower_to_kernel_ir(plan.ir)
    op_kinds = [op.kind for op in iter_kernel_ops(kernel)]

    assert kernel.lane_layout.kind == TRIAL_LANE_LAYOUT
    assert "DDMIntegrateUntilFinished" in op_kinds
    assert kernel.rng_streams[0].name == "DDM.ddm"


@pytest.mark.composition
def test_batched_compiler_rejects_unsupported_custom_function():
    mech = pnl.TransferMechanism(
        input_shapes=1,
        function=UserDefinedFunction(custom_function=lambda variable: variable),
        name="custom",
    )
    comp = pnl.Composition(pathways=mech)
    report = BatchedCompositionCompiler.diagnose(comp)

    assert not report.is_supported
    assert "UserDefinedFunction" in "; ".join(report.unsupported_reasons)
    with pytest.raises(BatchedCompileError):
        BatchedCompositionCompiler.compile(comp)


@pytest.mark.composition
def test_batched_compiler_rejects_unsupported_scheduler_condition():
    mech = pnl.TransferMechanism(input_shapes=1, name="linear")
    comp = pnl.Composition(pathways=mech)
    comp.scheduler.add_condition(mech, pnl.Never())
    report = BatchedCompositionCompiler.diagnose(comp)

    assert not report.is_supported
    assert "Never" in "; ".join(report.unsupported_reasons)


@pytest.mark.composition
def test_batched_compiler_rejects_unsupported_lca_width():
    lca = pnl.LCAMechanism(input_shapes=3, name="wide_lca")
    comp = pnl.Composition(pathways=lca)
    report = BatchedCompositionCompiler.diagnose(comp)

    assert not report.is_supported
    assert "unsupported LCA width" in "; ".join(report.unsupported_reasons)


@pytest.mark.composition
def test_batched_compiler_rejects_unsupported_lca_function():
    lca = pnl.LCAMechanism(input_shapes=2, function=pnl.Linear(), name="linear_lca")
    comp = pnl.Composition(pathways=lca)
    report = BatchedCompositionCompiler.diagnose(comp)

    assert not report.is_supported
    assert "unsupported LCA function" in "; ".join(report.unsupported_reasons)


@pytest.mark.composition
def test_pec_can_compile_batched_diagnostic():
    comp, decision = _make_ddm_comp(noise=0.0)
    data = pd.DataFrame({"decision": [1.0], "response_time": [0.05]})
    data["decision"] = data["decision"].astype("category")
    pec = pnl.ParameterEstimationComposition(
        name="pec",
        nodes=[comp],
        parameters={("threshold", decision): [0.05, 0.1]},
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        optimization_function=PECOptimizationFunction(method="differential_evolution"),
        num_estimates=1,
        initial_seed=1,
    )

    report = pec.can_compile_batched(backend="ir_debug")
    assert report.is_supported
    assert report.model_kind == "ddm"
    assert isinstance(report.metadata["fusion_kind"], str)


@pytest.mark.composition
def test_stability_flexibility_ir_debug_smoke():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_stab_flex_pec_fit import generate_trial_sequence, make_input_dict, make_stab_flex

    comp = make_stab_flex(
        lca_time_step_size=0.01,
        ddm_time_step_size=0.01,
        threshold=0.05,
        ddm_noise=0.0,
        lca_noise=0.0,
    )
    task, stimulus, cue, correct = generate_trial_sequence(16, 0.5, seed=3)
    inputs = make_input_dict(comp, task[:2], stimulus[:2], cue[:2], correct[:2])
    plan = BatchedCompositionCompiler.compile(comp, backend="ir_debug")
    assert plan.ir.model_kind == "stability_flexibility"
    assert plan.ir.graph.fusion_kind == "stateful_graph"
    assert plan.ir.graph.metadata["stability_flexibility_roles"]["lca_node"].startswith("Task Activations")
    result = plan.run(
        inputs=inputs,
        parameter_sets=[{"threshold": 0.05, "ddm_noise": 0.0, "lca_noise": 0.0}],
        num_estimates=1,
        seed=3,
    )

    assert result.values.shape == (1, 1, 2, 1, 2)
    assert np.all(np.isfinite(result.values))


@pytest.mark.composition
def test_kernel_ir_stability_flexibility_stateful_structure():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_stab_flex_pec_fit import make_stab_flex

    comp = make_stab_flex(
        lca_time_step_size=0.01,
        ddm_time_step_size=0.01,
        threshold=0.05,
        ddm_noise=0.0,
        lca_noise=0.0,
    )
    plan = BatchedCompositionCompiler.compile(comp, backend="ir_debug", max_steps=256)
    kernel = lower_to_kernel_ir(plan.ir)
    op_kinds = [op.kind for op in iter_kernel_ops(kernel)]

    assert kernel.lane_layout.kind == STATEFUL_LANE_LAYOUT
    assert op_kinds[0] == "InitializeState"
    assert "ForTrials" in op_kinds
    assert "LCAIntegrateUntilFinished" in op_kinds
    assert "DDMIntegrateUntilFinished" in op_kinds
    assert {stream.step_extent for stream in kernel.rng_streams} == {"LCA_MAX_STEPS", "MAX_STEPS"}


@pytest.mark.triton
@pytest.mark.skipif(not _triton_available(), reason="Triton CUDA backend is not available")
def test_triton_stateless_graph_matches_ir_debug():
    comp, source, _ = _make_linear_projection_comp()
    inputs = {source: np.array([[2.0, 4.0], [1.0, 1.0]], dtype=float)}
    ir_debug = BatchedCompositionCompiler.compile(comp, backend="ir_debug").run(
        inputs=inputs,
        parameter_sets=[{}],
        num_estimates=2,
    )
    triton = BatchedCompositionCompiler.compile(comp, backend="triton").run(
        inputs=inputs,
        parameter_sets=[{}],
        num_estimates=2,
    )

    np.testing.assert_allclose(triton.values, ir_debug.values, rtol=1e-6, atol=1e-6)


@pytest.mark.triton
@pytest.mark.skipif(not _triton_available(), reason="Triton CUDA backend is not available")
def test_triton_ddm_matches_ir_debug_deterministic():
    comp, decision = _make_ddm_comp(noise=0.0)
    inputs = {decision: np.array([[1.0], [-1.0]], dtype=float)}
    params = [{"rate": 1.0, "threshold": 0.05, "noise": 0.0, "time_step_size": 0.01}]
    ir_debug = BatchedCompositionCompiler.compile(comp, backend="ir_debug", max_steps=64).run(
        inputs=inputs,
        parameter_sets=params,
        num_estimates=2,
        seed=11,
    )
    triton = BatchedCompositionCompiler.compile(comp, backend="triton", max_steps=64).run(
        inputs=inputs,
        parameter_sets=params,
        num_estimates=2,
        seed=11,
    )

    np.testing.assert_allclose(triton.values, ir_debug.values, rtol=1e-6, atol=1e-6)


@pytest.mark.triton
@pytest.mark.skipif(not _triton_available(), reason="Triton CUDA backend is not available")
def test_triton_ddm_behind_transfer_generated_graph_matches_ir_debug():
    source = pnl.TransferMechanism(input_shapes=1, name="stimulus")
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0,
            rate=1.0,
            noise=0.0,
            threshold=0.05,
            non_decision_time=0.0,
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    comp = pnl.Composition(pathways=[[source, decision]])
    inputs = {source: np.array([[1.0], [-1.0]], dtype=float)}
    params = [{"rate": 1.0, "threshold": 0.05, "noise": 0.0, "time_step_size": 0.01}]

    ir_debug = BatchedCompositionCompiler.compile(comp, backend="ir_debug", max_steps=64).run(
        inputs=inputs,
        parameter_sets=params,
        num_estimates=2,
        seed=11,
    )
    triton_plan = BatchedCompositionCompiler.compile(comp, backend="triton", max_steps=64)
    assert triton_plan.ir.graph.fusion_kind == "ddm_graph"
    triton = triton_plan.run(
        inputs=inputs,
        parameter_sets=params,
        num_estimates=2,
        seed=11,
    )

    np.testing.assert_allclose(triton.values, ir_debug.values, rtol=1e-6, atol=1e-6)


@pytest.mark.triton
@pytest.mark.skipif(not _triton_available(), reason="Triton CUDA backend is not available")
def test_triton_stability_flexibility_smoke():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_stab_flex_pec_fit import generate_trial_sequence, make_input_dict, make_stab_flex

    comp = make_stab_flex(
        lca_time_step_size=0.01,
        ddm_time_step_size=0.01,
        threshold=0.05,
        ddm_noise=0.0,
        lca_noise=0.0,
    )
    task, stimulus, cue, correct = generate_trial_sequence(16, 0.5, seed=4)
    inputs = make_input_dict(comp, task[:2], stimulus[:2], cue[:2], correct[:2])
    params = [{"threshold": 0.05, "ddm_noise": 0.0, "lca_noise": 0.0}]
    ir_debug = BatchedCompositionCompiler.compile(comp, backend="ir_debug", max_steps=256).run(
        inputs=inputs,
        parameter_sets=params,
        num_estimates=1,
        seed=4,
    )
    triton_plan = BatchedCompositionCompiler.compile(comp, backend="triton", max_steps=256)
    assert triton_plan.ir.graph.fusion_kind == "stateful_graph"
    result = triton_plan.run(
        inputs=inputs,
        parameter_sets=params,
        num_estimates=1,
        seed=4,
    )

    assert result.values.shape == (1, 1, 2, 1, 2)
    assert np.all(np.isfinite(result.values))
    np.testing.assert_allclose(result.values, ir_debug.values, rtol=1e-5, atol=1e-5)


@pytest.mark.triton
@pytest.mark.skipif(not _triton_available(), reason="Triton CUDA backend is not available")
def test_triton_stability_flexibility_stateful_graph_matches_monolithic_fallback():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_stab_flex_pec_fit import generate_trial_sequence, make_input_dict, make_stab_flex

    comp = make_stab_flex(
        lca_time_step_size=0.01,
        ddm_time_step_size=0.01,
        threshold=0.05,
        ddm_noise=0.0,
        lca_noise=0.0,
    )
    task, stimulus, cue, correct = generate_trial_sequence(16, 0.5, seed=5)
    inputs = make_input_dict(comp, task[:2], stimulus[:2], cue[:2], correct[:2])
    params = [{"threshold": 0.05, "ddm_noise": 0.0, "lca_noise": 0.0}]
    generated_plan = BatchedCompositionCompiler.compile(comp, backend="triton", max_steps=256)
    monolithic_plan = _with_graph_fusion_kind(generated_plan, "stability_flexibility")

    generated = generated_plan.run(inputs=inputs, parameter_sets=params, num_estimates=1, seed=5)
    monolithic = monolithic_plan.run(inputs=inputs, parameter_sets=params, num_estimates=1, seed=5)

    np.testing.assert_allclose(generated.values, monolithic.values, rtol=1e-5, atol=1e-5)
