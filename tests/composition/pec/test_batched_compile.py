import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import psyneulink as pnl

from psyneulink.core.batched import (
    BatchedGraphIR,
    BatchedCompileError,
    BatchedCompositionCompiler,
    BatchedDiagnosticCode,
    batched_node_op,
    unregister_batched_instance_op,
)
from psyneulink.core.batched import specs as batched_specs
from psyneulink.core.batched.specs import BatchedOpSpecError
from psyneulink.core.batched.backend.triton.api import (
    TritonOpError,
    pnl_triton_op,
)
from psyneulink.core.batched.backend.triton.graph_emit import triton_graph_kernel_source
from psyneulink.core.batched.kernel_ir import (
    STATEFUL_LANE_LAYOUT,
    TRIAL_LANE_LAYOUT,
    iter_kernel_ops,
    lower_to_kernel_ir,
)
from psyneulink.core.components.functions.userdefinedfunction import UserDefinedFunction
from psyneulink.core.components.functions.nonstateful.fitfunctions import PECOptimizationFunction


pytestmark = [
    pytest.mark.batched,
    pytest.mark.usefixtures("set_threads_to_one"),
]


_TRITON_TEST_GLOBAL = 2.0


@pnl_triton_op(constexpr=("limit",))
def _test_pnl_triton_add(x, limit):
    return x + limit


def _test_pnl_triton_uses_global(x):
    return x + _TRITON_TEST_GLOBAL


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


# Numeric execution uses the externally configured Triton interpreter. Pure
# lowering, diagnostic, and source-emission tests remain backend-independent.
requires_triton = pytest.mark.triton_interpreter


@pytest.mark.composition
def test_batched_compiler_rejects_reference_backend_name():
    comp, _ = _make_ddm_comp(noise=0.0)

    with pytest.raises(BatchedCompileError, match="Unknown batched backend 'reference'"):
        BatchedCompositionCompiler.diagnose(comp, backend="reference")

    with pytest.raises(BatchedCompileError, match="Unknown batched backend 'reference'"):
        BatchedCompositionCompiler.compile(comp, backend="reference")


@pytest.mark.composition
def test_pnl_triton_op_extracts_inspectable_helper_source_without_triton_import():
    assert _test_pnl_triton_add.name == "_test_pnl_triton_add"
    assert "@triton.jit" in _test_pnl_triton_add.source
    assert "limit: tl.constexpr" in _test_pnl_triton_add.source
    assert "pnl_triton_op" not in _test_pnl_triton_add.source


@pytest.mark.composition
def test_pnl_triton_op_rejects_closures_and_globals():
    scale = 2.0

    def uses_closure(x):
        return x + scale

    with pytest.raises(TritonOpError, match="cannot close over values"):
        pnl_triton_op(uses_closure)

    with pytest.raises(TritonOpError, match="unsupported free variables"):
        pnl_triton_op(_test_pnl_triton_uses_global)


@pytest.mark.composition
@requires_triton
def test_batched_ir_debug_ddm_deterministic():
    comp, decision = _make_ddm_comp(noise=0.0)
    plan = BatchedCompositionCompiler.compile(comp, backend="triton_cpu")
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
@requires_triton
def test_batched_ir_debug_ddm_common_random_numbers():
    comp, decision = _make_ddm_comp(noise=0.2)
    plan = BatchedCompositionCompiler.compile(comp, backend="triton_cpu")
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
@requires_triton
def test_batched_ir_debug_transfer_only_generic_graph():
    mech = pnl.TransferMechanism(
        input_shapes=2,
        function=pnl.Linear(slope=2.0, intercept=1.0),
        name="linear",
    )
    comp = pnl.Composition(pathways=mech)
    plan = BatchedCompositionCompiler.compile(comp, backend="triton_cpu")
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
    plan = BatchedCompositionCompiler.compile(comp, backend="triton_cpu")
    kernel = lower_to_kernel_ir(plan.ir)
    op_kinds = [op.kind for op in iter_kernel_ops(kernel)]

    assert kernel.lane_layout.kind == TRIAL_LANE_LAYOUT
    assert op_kinds == ["LoadInput", "CallFunction", "StoreOutput"]


@pytest.mark.composition
@requires_triton
def test_batched_ir_debug_mapping_projection_generic_graph():
    comp, source, _ = _make_linear_projection_comp()
    result = BatchedCompositionCompiler.compile(comp, backend="triton_cpu").run(
        inputs={source: np.array([[2.0, 4.0]], dtype=float)},
        parameter_sets=[{}],
        num_estimates=1,
    )

    assert result.values.shape == (1, 1, 1, 1, 1)
    np.testing.assert_allclose(result.values[0, 0, 0, 0], [31.0])


@pytest.mark.composition
def test_registry_diagnostics_accept_supported_components():
    comp, _, _ = _make_linear_projection_comp()
    report = BatchedCompositionCompiler.diagnose(comp, backend="triton")

    assert report.is_supported
    assert not any("missing Triton" in reason for reason in report.unsupported_reasons)


@pytest.mark.composition
def test_registry_diagnostics_report_missing_triton_implementation(monkeypatch):
    import dataclasses

    from psyneulink.core.components.functions.nonstateful.transferfunctions import Linear

    batched_specs.ensure_builtin_specs()
    spec = batched_specs._FUNCTION_SPECS[Linear]
    monkeypatch.setitem(
        batched_specs._SPECS_BY_KEY,
        spec.key,
        dataclasses.replace(spec, triton_template=None),
    )
    comp, _, _ = _make_linear_projection_comp()
    report = BatchedCompositionCompiler.diagnose(comp, backend="triton")

    assert not report.is_supported
    assert "missing Triton implementation" in "; ".join(report.unsupported_reasons)


@pytest.mark.composition
@requires_triton
def test_batched_op_decorator_registers_custom_elementwise_function():
    @batched_specs.batched_op(pnl.Exponential)
    def exponential(x, rate, scale):
        return scale * tl.exp(rate * x)  # tl: triton.language (resolved at emission)

    try:
        mech = pnl.TransferMechanism(
            input_shapes=2,
            function=pnl.Exponential(rate=2.0, scale=1.5),
            name="exp_mech",
        )
        comp = pnl.Composition(pathways=mech)
        report = BatchedCompositionCompiler.diagnose(comp, backend="triton")
        assert report.is_supported

        plan = BatchedCompositionCompiler.compile(comp, backend="triton_cpu")
        result = plan.run(
            inputs={mech: np.array([[0.0, 1.0]], dtype=float)},
            parameter_sets=[{}],
            num_estimates=1,
        )
        np.testing.assert_allclose(
            result.values[0, 0, 0, 0],
            [1.5, 1.5 * np.exp(2.0)],
            rtol=1e-5,
        )
    finally:
        removed = batched_specs._FUNCTION_SPECS.pop(pnl.Exponential, None)
        if removed is not None:
            batched_specs._SPECS_BY_KEY.pop(removed.key, None)


@pytest.mark.composition
def test_batched_op_decorator_rejects_unknown_parameter_name():
    with pytest.raises(BatchedOpSpecError, match="does not match a Parameter"):
        @batched_specs.batched_op(pnl.Exponential)
        def bad(x, not_a_parameter):
            return x + not_a_parameter

    assert pnl.Exponential not in batched_specs._FUNCTION_SPECS


@pytest.mark.composition
def test_batched_op_specs_do_not_apply_to_subclasses():
    class CustomLinear(pnl.Linear):
        pass

    mech = pnl.TransferMechanism(
        input_shapes=1,
        function=CustomLinear(),
        name="custom_linear",
    )
    comp = pnl.Composition(pathways=mech)
    report = BatchedCompositionCompiler.diagnose(comp)

    assert not report.is_supported
    assert "CustomLinear" in "; ".join(report.unsupported_reasons)


def _make_udf_reducer_comp():
    """A->reducer<-B graph: a UDF node that reduces its 2-wide combined input.

    The UDF node ('Reducer') is a ProcessingMechanism whose class already owns a
    PassthroughMechanismSpec, so only an instance-level op can give it a kernel.
    """

    def _prod(variable):
        v = np.asarray(variable, dtype=float).reshape(-1)
        return v[0] * v[1] if v.size >= 2 else 0.0

    a = pnl.ProcessingMechanism(input_shapes=1, name="A")
    b = pnl.ProcessingMechanism(input_shapes=1, name="B")
    reducer = pnl.ProcessingMechanism(
        name="Reducer",
        input_ports=[{pnl.NAME: "in", pnl.INPUT_SHAPES: 2, pnl.COMBINE: pnl.SUM}],
        function=pnl.UserDefinedFunction(custom_function=_prod),
    )
    comp = pnl.Composition()
    comp.add_node(a)
    comp.add_node(b)
    comp.add_node(reducer)
    comp.add_projection(sender=a, receiver=reducer, projection=pnl.MappingProjection(matrix=np.array([[1.0, 0.0]])))
    comp.add_projection(sender=b, receiver=reducer, projection=pnl.MappingProjection(matrix=np.array([[0.0, 1.0]])))
    return comp, a, b


@pytest.mark.composition
@requires_triton
def test_batched_node_op_registers_instance_level_udf_reduction():
    # A UDF node that the class-keyed registry cannot express (all UDFs share one
    # class) is given an instance-level op that reduces its whole input vector.
    comp, a, b = _make_udf_reducer_comp()
    try:
        @batched_node_op("Reducer")
        def reducer(x0, x1):
            return x0 * x1  # tl arithmetic over the 2 combined input components

        report = BatchedCompositionCompiler.diagnose(comp, backend="triton_cpu")
        assert report.is_supported, report.unsupported_reasons

        plan = BatchedCompositionCompiler.compile(comp, backend="triton_cpu")
        result = plan.run(
            inputs={a: np.array([[3.0]]), b: np.array([[5.0]])},
            parameter_sets=[{}],
            num_estimates=1,
        )
        # combined input is [3, 5] (SUM of the two routed projections) -> 3 * 5.
        np.testing.assert_allclose(result.values[0, 0, 0, 0], [15.0], rtol=1e-5)
    finally:
        unregister_batched_instance_op("Reducer")


@pytest.mark.composition
def test_batched_node_op_registration_is_instance_scoped_and_reversible():
    comp, _, _ = _make_udf_reducer_comp()

    # Without a registered instance op, the UDF node is rejected.
    rejected = BatchedCompositionCompiler.diagnose(comp)
    assert not rejected.is_supported
    assert any(
        diagnostic.component == "Reducer"
        for diagnostic in rejected.model_diagnostics
    )

    try:
        @batched_node_op("Reducer")
        def reducer(x0, x1):
            return x0 * x1

        assert "Reducer" in batched_specs._INSTANCE_SPECS
        key = batched_specs._INSTANCE_SPECS["Reducer"].key
        assert key == "instance:Reducer"
        assert key in batched_specs._SPECS_BY_KEY
        # Instance-keyed: ProcessingMechanism as a class is unaffected.
        assert pnl.ProcessingMechanism not in batched_specs._MECHANISM_SPECS
    finally:
        unregister_batched_instance_op("Reducer")

    assert "Reducer" not in batched_specs._INSTANCE_SPECS
    assert "instance:Reducer" not in batched_specs._SPECS_BY_KEY


@pytest.mark.composition
def test_triton_stateless_graph_source_structure():
    comp, _, _ = _make_linear_projection_comp()
    plan = BatchedCompositionCompiler.compile(comp, backend="triton_cpu")
    source = triton_graph_kernel_source(lower_to_kernel_ir(plan.ir))

    compile(source, "<pnl batched kernel>", "exec")
    assert "pnl_batched_stateless_graph_kernel" in source
    assert "_pnl_triton_linear" in source
    assert "_pnl_triton_projection_term" in source
    assert "@triton.jit" in source


@pytest.mark.composition
def test_single_ddm_uses_generated_graph_kernel():
    comp, _ = _make_ddm_comp(noise=0.0)
    plan = BatchedCompositionCompiler.compile(comp, backend="triton_cpu", max_steps=64)

    assert plan.ir.model_kind == "ddm"
    assert plan.ir.graph.fusion_kind == "ddm_graph"
    source = triton_graph_kernel_source(lower_to_kernel_ir(plan.ir))
    compile(source, "<pnl batched kernel>", "exec")
    assert "pnl_batched_ddm_graph_kernel" in source
    assert "_pnl_triton_ddm" in source


@pytest.mark.composition
def test_kernel_ir_dense_projection_structure():
    comp, source, _ = _make_linear_projection_comp()
    plan = BatchedCompositionCompiler.compile(comp, backend="triton_cpu")
    kernel = lower_to_kernel_ir(plan.ir)
    op_kinds = [op.kind for op in iter_kernel_ops(kernel)]

    assert kernel.lane_layout.kind == TRIAL_LANE_LAYOUT
    assert "CallProjection" in op_kinds
    assert "CombineSum" in op_kinds


@pytest.mark.composition
@requires_triton
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

    result = BatchedCompositionCompiler.compile(comp, backend="triton_cpu").run(
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

    plan = BatchedCompositionCompiler.compile(comp, backend="triton_cpu")
    kernel = lower_to_kernel_ir(plan.ir)
    op_kinds = [op.kind for op in iter_kernel_ops(kernel)]

    assert "CombineProduct" in op_kinds


@pytest.mark.composition
@requires_triton
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
    plan = BatchedCompositionCompiler.compile(comp, backend="triton_cpu", max_steps=64)
    result = plan.run(
        inputs={source: np.array([[1.0], [-1.0]], dtype=float)},
        parameter_sets=[
            {
                "DDM.rate": 1.0,
                "DDM.threshold": 0.05,
                "DDM.noise": 0.0,
                "DDM.time_step_size": 0.01,
            }
        ],
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
    plan = BatchedCompositionCompiler.compile(comp, backend="triton_cpu", max_steps=64)
    kernel = lower_to_kernel_ir(plan.ir)
    op_kinds = [op.kind for op in iter_kernel_ops(kernel)]

    assert kernel.lane_layout.kind == TRIAL_LANE_LAYOUT
    assert "CallMechanism" in op_kinds
    assert kernel.rng_streams[0].name == "DDM.rng"


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
def test_batched_compiler_rejects_integrator_mode_transfer():
    mech = pnl.TransferMechanism(input_shapes=1, integrator_mode=True, name="integ")
    comp = pnl.Composition(pathways=mech)
    report = BatchedCompositionCompiler.diagnose(comp)

    assert not report.is_supported
    assert "integrator_mode" in "; ".join(report.unsupported_reasons)


@pytest.mark.composition
def test_batched_fires_once_integrating_transfer_is_supported():
    # integrator_mode + per-trial reset + fires once (AtPass) -> lowerable as a
    # stateless single-step integrator.
    mech = pnl.TransferMechanism(
        input_shapes=1, function=pnl.Linear(slope=2.0),
        integrator_mode=True, integration_rate=1.0,
        reset_stateful_function_when=pnl.AtTrialStart(), name="integ",
    )
    comp = pnl.Composition(pathways=mech)
    comp.scheduler.add_condition(mech, pnl.AtPass(0))
    report = BatchedCompositionCompiler.diagnose(comp)

    assert report.is_supported, report.unsupported_reasons
    assert not any("integrator_mode" in reason for reason in report.unsupported_reasons)


@pytest.mark.composition
def test_batched_integrating_transfer_without_fire_once_schedule_is_rejected():
    # Reset each trial but NO fires-once schedule: it would accumulate within the
    # trial, so the stateless fold is unsound and the node must stay rejected.
    mech = pnl.TransferMechanism(
        input_shapes=1, integrator_mode=True,
        reset_stateful_function_when=pnl.AtTrialStart(), name="integ",
    )
    comp = pnl.Composition(pathways=mech)
    report = BatchedCompositionCompiler.diagnose(comp)

    assert not report.is_supported
    assert "integrator_mode" in "; ".join(report.unsupported_reasons)


@pytest.mark.composition
def test_batched_compiler_accepts_stateless_transfer():
    mech = pnl.TransferMechanism(input_shapes=1, function=pnl.Linear(slope=2.0), name="plain")
    comp = pnl.Composition(pathways=mech)
    report = BatchedCompositionCompiler.diagnose(comp)

    assert report.is_supported
    assert not any("integrator_mode" in reason for reason in report.unsupported_reasons)


@pytest.mark.composition
def test_csi_surrogate_with_iti_reports_each_remaining_semantic_blocker():
    # With iti>0, Task Input fires at AtPass(iti): a delayed within-trial onset.
    # Diagnosis recognizes that condition instead of reporting it independently,
    # but the composition still fails closed: generic co-evolving
    # Always/WhenFinished regions and the controlled LCA-finished predicate that
    # starts the DDM are not executable in KernelIR. Its drift-rate UDF also has
    # no instance op in this test. CSI is an acceptance composition, not a model
    # kind or special compiler path.
    csi_dir = Path(__file__).resolve().parents[3] / "Scripts" / "Debug" / "pec_batch_compile"
    sys.path.insert(0, str(csi_dir))
    from csi_model_surrogate import make_stab_flex

    comp = make_stab_flex(iti=10, csi_repeat=10, csi_switch=10, threshold_collapse=-0.001)
    report = BatchedCompositionCompiler.diagnose(comp)

    assert not report.is_supported
    assert "AtPass" not in "; ".join(report.unsupported_reasons)
    assert not report.rejected_conditions
    rejected = {d.component for d in report.rejected_nodes}
    assert not any("Threshold Mechanism" in name for name in rejected)
    assert not any(
        "Task Input" in d.component and "integrator_mode" in d.reason
        for d in report.rejected_nodes
    )
    assert any("Drift Rate Value" in name for name in rejected)
    schedule_diagnostic = next(
        diagnostic
        for diagnostic in report.rejected_nodes
        if diagnostic.component.startswith("CSI Override")
    )
    assert schedule_diagnostic.code == BatchedDiagnosticCode.MODEL_SCHEDULE_NOT_EXECUTABLE
    assert "LCA finished predicate" in schedule_diagnostic.detail


@pytest.mark.composition
def test_batched_compiler_rejects_unsupported_scheduler_condition():
    mech = pnl.TransferMechanism(input_shapes=1, name="linear")
    comp = pnl.Composition(pathways=mech)
    comp.scheduler.add_condition(mech, pnl.Never())
    report = BatchedCompositionCompiler.diagnose(comp)

    assert not report.is_supported
    assert "Never" in "; ".join(report.unsupported_reasons)


@pytest.mark.composition
def test_batched_compiler_reports_unimplemented_precomputed_scheduler():
    source = pnl.TransferMechanism(input_shapes=1, name="source")
    target = pnl.TransferMechanism(input_shapes=1, name="target")
    comp = pnl.Composition(pathways=[[source, target]])
    # EveryNCalls(source, 1) is exactly the scheduler's implicit processing-edge
    # default and is therefore executable as an ordinary static graph.  A
    # two-call predicate genuinely requires a precomputed/dynamic schedule.
    comp.scheduler.add_condition(target, pnl.EveryNCalls(source, 2))

    report = BatchedCompositionCompiler.diagnose(comp)

    assert not report.is_supported
    assert report.metadata["schedule_kind"] == "precomputed_trace"
    unsupported = "; ".join(report.unsupported_reasons)
    assert "EveryNCalls" in unsupported
    assert "not executable yet" in unsupported


@pytest.mark.composition
def test_batched_compiler_accepts_at_pass_zero_condition():
    # AtPass(0) ("fire only on pass 0") is the batched origin default (each node
    # computes once per trial), so it lowers as a static graph, not a rejection.
    mech = pnl.TransferMechanism(input_shapes=1, function=pnl.Linear(slope=2.0), name="linear")
    comp = pnl.Composition(pathways=mech)
    comp.scheduler.add_condition(mech, pnl.AtPass(0))
    report = BatchedCompositionCompiler.diagnose(comp)

    assert report.is_supported
    assert not report.rejected_conditions
    assert report.metadata["schedule_kind"] == "static_graph"


@pytest.mark.composition
def test_batched_compiler_defers_at_pass_nonzero_condition():
    # AtPass(n>0) is a delayed within-trial onset (e.g. ITI); it is recognized
    # but not executable yet, and must be deferred rather than silently mis-timed
    # as a static graph.
    mech = pnl.TransferMechanism(input_shapes=1, name="linear")
    comp = pnl.Composition(pathways=mech)
    comp.scheduler.add_condition(mech, pnl.AtPass(3))
    report = BatchedCompositionCompiler.diagnose(comp)

    assert not report.is_supported
    assert report.metadata["schedule_kind"] == "precomputed_trace"
    unsupported = "; ".join(report.unsupported_reasons)
    assert "AtPass" in unsupported
    assert "not executable yet" in unsupported


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

    report = pec.can_compile_batched(backend="triton_cpu")
    assert report.is_supported
    assert report.model_kind == "ddm"
    assert isinstance(report.metadata["fusion_kind"], str)


@pytest.mark.composition
@requires_triton
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
    plan = BatchedCompositionCompiler.compile(comp, backend="triton_cpu")
    assert plan.ir.model_kind == "graph"
    assert plan.ir.graph.fusion_kind == "stateful_graph"
    assert plan.ir.graph.metadata["schedule_kind"] == "static_graph"
    assert plan.capability_report.metadata["schedule_kind"] == "static_graph"
    assert "stability_flexibility_roles" not in plan.ir.graph.metadata
    result = plan.run(
        inputs=inputs,
        parameter_sets=[
            {
                "DDM.threshold": 0.05,
                "DDM.noise": 0.0,
            }
        ],
        num_estimates=1,
        seed=3,
    )

    assert result.values.shape == (1, 1, 2, 1, 2)
    assert np.all(np.isfinite(result.values))


@pytest.mark.composition
@requires_triton
def test_stability_flexibility_rejects_old_model_specific_parameter_names():
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
    plan = BatchedCompositionCompiler.compile(comp, backend="triton_cpu")

    with pytest.raises(ValueError, match="Unknown batched parameter"):
        plan.run(
            inputs=inputs,
            parameter_sets=[{"threshold": 0.05, "ddm_noise": 0.0, "lca_noise": 0.0}],
            num_estimates=1,
            seed=3,
        )


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
    plan = BatchedCompositionCompiler.compile(comp, backend="triton_cpu", max_steps=256)
    kernel = lower_to_kernel_ir(plan.ir)
    op_kinds = [op.kind for op in iter_kernel_ops(kernel)]

    assert kernel.lane_layout.kind == STATEFUL_LANE_LAYOUT
    assert op_kinds[0] == "InitializeState"
    assert "ForTrials" in op_kinds
    mechanism_types = {
        op.attrs["component_type"]
        for op in iter_kernel_ops(kernel)
        if op.kind == "CallMechanism"
    }
    assert mechanism_types == {"LCAMechanism", "DDM"}
    assert [(stream.name, stream.step_extent) for stream in kernel.rng_streams] == [
        ("DDM.rng", "MAX_STEPS")
    ]
