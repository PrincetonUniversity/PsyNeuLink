"""Acceptance contract for counted ``WhenFinished`` execution.

The primary cases deliberately use an ordinary stateless follower.  A dynamic
finished-state executor must therefore be a scheduler capability, not a hidden
LCA/DDM pairing rule.  Each semantic case builds a fresh Python oracle and a
fresh compiler input.  Nearby scheduler, reset, control, and co-evolution
semantics remain structured fail-closed boundaries.

The provenance strings point to existing execution-mode-expanded tests whose
LCA, ``WhenFinished``, multi-trial, DDM, termination, and control semantics this
corpus combines.
"""

from collections.abc import Callable
from dataclasses import dataclass
import itertools

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompileError,
    BatchedCompositionCompiler,
    BatchedDiagnosticCode,
)
from psyneulink.core.batched.graph import STATEFUL_GRAPH_FUSION, lower_composition
from psyneulink.core.batched.ir import BatchedFinishedValueSpec
from psyneulink.core.batched.kernel_ir import (
    STATEFUL_LANE_LAYOUT,
    iter_kernel_ops,
)

from batched_semantic_test_support import (
    SemanticCase,
    SemanticModel,
    assert_matches_python,
)


pytestmark = [pytest.mark.batched, pytest.mark.composition]


_LCA_LLVM_PROVENANCE = (
    "tests/mechanisms/test_lca.py::TestLCA::test_LCAMechanism_threshold"
)
_WHEN_FINISHED_LLVM_PROVENANCE = (
    "tests/scheduling/test_scheduler.py::"
    "TestFeedback::test_time_termination_measures"
)
_MULTI_TRIAL_LLVM_PROVENANCE = (
    "tests/composition/test_composition.py::"
    "TestRun::test_run_2_mechanisms_with_multiple_trials_of_input_values"
)
_DDM_LLVM_PROVENANCE = (
    "tests/mechanisms/test_ddm_mechanism.py::"
    "test_ddm_is_finished_with_dependency"
)
_TERMINATION_LLVM_PROVENANCE = (
    "tests/scheduling/test_scheduler.py::TestFeedback::test_run_term_conditions"
)
_CONTROL_LLVM_PROVENANCE = (
    "tests/composition/test_control.py::"
    "TestControlMechanisms::test_control_of_mech_port[OVERRIDE]"
)

_TRIAL_INPUTS = np.array(
    [[1.0, -1.0], [0.5, 0.25], [-0.5, 1.5]],
    dtype=float,
)
_PERSISTENT_RESULTS = np.array(
    [[1.0202979047745746], [1.0691310439370374], [-0.12468102134796921]],
    dtype=float,
)
_ONE_TRIAL_TRACE = (
    frozenset({"producer"}),
    frozenset({"producer"}),
    frozenset({"producer"}),
    frozenset({"follower"}),
)
_EXPECTED_TRACE = _ONE_TRIAL_TRACE * len(_TRIAL_INPUTS)

_LCA_DDM_INPUTS = np.array(
    [[2.0, -2.0], [-2.0, 2.0], [2.0, -2.0]],
    dtype=float,
)
_LCA_DDM_RESULTS = np.array(
    [[1.0, 0.01], [0.0, 0.01], [1.0, 0.01]],
    dtype=float,
)

_UNMODELED_COEVOLUTION_DETAIL = (
    "coevolving Always/WhenFinished execution requires explicit finished "
    "predicates and conditional pass regions in KernelIR"
)


def _stepwise_lca(*, name: str, reset_condition, threshold: float = 3.0):
    return pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(gain=1.0),
        leak=0.0,
        competition=0.0,
        self_excitation=0.0,
        noise=0.0,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=threshold,
        time_step_size=0.5,
        execute_until_finished=False,
        reset_stateful_function_when=reset_condition,
        name=name,
    )


def _generic_finished_case(
    name: str,
    *,
    threshold: float = 3.0,
    reverse_insertion: bool = False,
    renamed_boundary: bool = False,
    reset_factory: Callable[[], object] = pnl.Never,
) -> SemanticCase:
    build_number = itertools.count()

    def build():
        suffix = next(build_number)
        if renamed_boundary:
            producer_name = f"z / finished producer [{suffix}]"
            follower_name = f"a:stateless follower[{suffix}]"
        else:
            producer_name = f"generic finished producer {suffix}"
            follower_name = f"generic stateless follower {suffix}"

        producer = _stepwise_lca(
            name=producer_name,
            reset_condition=reset_factory(),
            threshold=threshold,
        )
        follower = pnl.TransferMechanism(
            input_shapes=1,
            function=pnl.Linear(slope=2.0, intercept=-0.25),
            name=follower_name,
        )
        composition = pnl.Composition(name=f"dynamic finished {name} {suffix}")
        composition.add_nodes(
            [follower, producer] if reverse_insertion else [producer, follower]
        )
        composition.add_projection(
            sender=producer,
            receiver=follower,
            projection=pnl.MappingProjection(matrix=[[1.0], [-1.0]]),
        )
        composition.scheduler.add_condition(producer, pnl.Always())
        composition.scheduler.add_condition(follower, pnl.WhenFinished(producer))
        return SemanticModel(
            composition=composition,
            inputs={producer: _TRIAL_INPUTS.copy()},
            outputs=(follower.output_port,),
        )

    return SemanticCase(
        name=name,
        build=build,
        provenance=(
            f"{_LCA_LLVM_PROVENANCE}; {_WHEN_FINISHED_LLVM_PROVENANCE}; "
            f"{_MULTI_TRIAL_LLVM_PROVENANCE}"
        ),
        max_steps=16,
        atol=1e-6,
        rtol=1e-5,
    )


@dataclass(frozen=True)
class _ExecutableAcceptance:
    case: SemanticCase
    expected_count: int
    expected_values: np.ndarray | None = None


EXECUTABLE_ACCEPTANCE_CASES = (
    _ExecutableAcceptance(
        case=_generic_finished_case("persistent_state"),
        expected_count=3,
        expected_values=_PERSISTENT_RESULTS,
    ),
    _ExecutableAcceptance(
        case=_generic_finished_case(
            "renamed_reverse_insertion",
            reverse_insertion=True,
            renamed_boundary=True,
        ),
        expected_count=3,
        expected_values=_PERSISTENT_RESULTS,
    ),
    _ExecutableAcceptance(
        case=_generic_finished_case("count_one", threshold=1.0),
        expected_count=1,
    ),
    _ExecutableAcceptance(
        case=_generic_finished_case(
            "host_fp64_threshold_rounds_up",
            threshold=1.00000001,
        ),
        expected_count=2,
    ),
    _ExecutableAcceptance(
        case=_generic_finished_case("fractional_threshold", threshold=2.2),
        expected_count=3,
    ),
)

def _run_python_oracle(model: SemanticModel):
    model.composition.run(
        inputs=model.inputs,
        execution_mode=pnl.ExecutionMode.Python,
    )
    values = _selected_results(model)

    producer = next(iter(model.inputs))
    follower = model.outputs[0].owner
    roles = {producer: "producer", follower: "follower"}
    trace = tuple(
        frozenset(roles[node] for node in execution_set)
        for execution_set in model.composition.scheduler.execution_list[
            model.composition.default_execution_id
        ]
    )
    return values, trace


def _selected_results(model: SemanticModel) -> np.ndarray:
    result_indices = [
        _result_index(model.composition, output)
        for output in model.outputs
    ]
    return np.asarray(
        [
            np.concatenate(
                [
                    np.asarray(trial[index], dtype=float).reshape(-1)
                    for index in result_indices
                ]
            )
            for trial in model.composition.results
        ],
        dtype=float,
    )


def _result_index(composition, output_port) -> int:
    matches = []
    for index, cim_input in enumerate(composition.output_CIM.input_ports):
        if any(
            projection.sender is output_port
            for projection in cim_input.path_afferents
        ):
            matches.append(index)
    assert len(matches) == 1, output_port.full_name
    return matches[0]


def _expected_trace(execution_count: int):
    one_trial = (
        (frozenset({"producer"}),) * execution_count
        + (frozenset({"follower"}),)
    )
    return one_trial * len(_TRIAL_INPUTS)


def _assert_executable_ir_contract(acceptance, model, backend):
    producer = next(iter(model.inputs))
    follower = model.outputs[0].owner
    history_before = dict(model.composition.scheduler.execution_list)

    lowering = lower_composition(
        model.composition,
        outputs=model.outputs,
    )
    graph = lowering.graph
    assert graph is not None
    assert graph.executable
    assert lowering.schedule_kind == "precomputed_trace"
    assert not lowering.rejected_nodes
    assert not lowering.rejected_conditions
    assert graph.fusion_kind == STATEFUL_GRAPH_FUSION
    assert graph.metadata["schedule_kind"] == "precomputed_trace"
    assert graph.metadata["scheduler_requires_pass_region"] is True
    assert graph.execution_order == (producer.name, follower.name)

    producer_spec = graph.node(producer.name)
    follower_spec = graph.node(follower.name)
    conditions = {condition.component_id: condition for condition in graph.scheduler}
    producer_condition = conditions[producer_spec.component_id]
    follower_condition = conditions[follower_spec.component_id]
    assert producer_condition.condition_type == "Always"
    assert producer_condition.consideration_set_id < (
        follower_condition.consideration_set_id
    )
    assert follower_condition.condition_type == "WhenFinished"
    assert follower_condition.dependency_component_ids == (
        producer_spec.component_id,
    )

    assert len(graph.finished_values) == 1
    finished = graph.finished_values[0]
    assert type(finished) is BatchedFinishedValueSpec
    assert finished.node == producer.name
    assert finished.component_id == producer_spec.component_id
    assert finished.predicate_kind == "execution_count_at_least"
    assert finished.attrs == {"count": acceptance.expected_count}
    assert finished.storage == "combinational"
    assert finished.width == 1
    assert finished.dtype == "bool"
    assert follower_condition.finished_value_ids == (finished.value_id,)
    assert finished.producer_consideration_set_id == (
        producer_condition.consideration_set_id
    )

    producer_states = tuple(
        state for state in graph.states
        if state.component_id == producer_spec.component_id
    )
    producer_resets = tuple(
        reset for reset in graph.resets
        if reset.component_id == producer_spec.component_id
    )
    assert len(producer_states) == 3
    assert len(producer_resets) == 1
    assert producer_resets[0].condition_type == "Never"
    assert producer_resets[0].state_ids == tuple(
        state.state_id for state in producer_states
    )

    plan = BatchedCompositionCompiler.compile(
        model.composition,
        backend=backend,
        outputs=model.outputs,
        max_steps=acceptance.case.max_steps,
    )
    assert dict(model.composition.scheduler.execution_list) == history_before
    report = plan.capability_report
    assert report.model_supported
    assert report.codegen_ready is True
    assert report.backend_available
    assert report.can_execute
    assert not report.model_diagnostics
    assert not report.codegen_diagnostics
    assert not report.backend_diagnostics
    assert report.metadata["fusion_kind"] == STATEFUL_GRAPH_FUSION
    assert report.metadata["schedule_kind"] == "precomputed_trace"

    kernel = plan.kernel_ir
    assert kernel.executable
    assert kernel.lane_layout.kind == STATEFUL_LANE_LAYOUT
    assert kernel.graph.executable
    assert kernel.states == graph.states
    assert kernel.finished_values == graph.finished_values
    assert kernel.schedule_trace is not None
    trace = kernel.schedule_trace
    expected_steps = tuple(
        (
            pass_index,
            producer_condition.consideration_set_id,
            (producer_spec.component_id,),
        )
        for pass_index in range(acceptance.expected_count)
    ) + (
        (
            acceptance.expected_count - 1,
            follower_condition.consideration_set_id,
            (follower_spec.component_id,),
        ),
    )
    assert tuple(
        (step.pass_index, step.consideration_set_id, step.component_ids)
        for step in trace.steps
    ) == expected_steps
    assert trace.num_passes == acceptance.expected_count
    assert trace.component_execution_count == acceptance.expected_count + 1

    role_by_component_id = {
        producer_spec.component_id: "producer",
        follower_spec.component_id: "follower",
    }
    typed_role_trace = tuple(
        frozenset(role_by_component_id[component_id] for component_id in step.component_ids)
        for step in trace.steps
    )
    assert typed_role_trace == _expected_trace(acceptance.expected_count)[
        : acceptance.expected_count + 1
    ]

    ops = iter_kernel_ops(kernel)
    producer_steps = tuple(
        op
        for op in ops
        if op.kind == "StepMechanism" and op.target == producer.name
    )
    assert len(producer_steps) == acceptance.expected_count
    assert tuple(op.attrs["execution_index"] for op in producer_steps) == tuple(
        range(acceptance.expected_count)
    )
    expected_state_ids = tuple(state.state_id for state in producer_states)
    assert all(op.attrs["state_ids"] == expected_state_ids for op in producer_steps)
    assert all(op.attrs["active_lanes"] == "all" for op in producer_steps)
    assert not any(
        op.kind == "CallMechanism" and op.target == producer.name
        for op in ops
    )
    assert sum(
        op.kind == "CallFunction" and op.target == follower.name
        for op in ops
    ) == 1
    assert sum(op.kind == "StoreOutput" for op in ops) == len(graph.outputs)
    return plan


@pytest.mark.parametrize(
    "acceptance",
    EXECUTABLE_ACCEPTANCE_CASES,
    ids=lambda acceptance: acceptance.case.name,
)
def test_generic_later_set_finished_matches_python(
    acceptance,
    batched_backend,
):
    python_model = acceptance.case.build()
    python_values, python_trace = _run_python_oracle(python_model)
    if acceptance.expected_values is not None:
        np.testing.assert_allclose(
            python_values,
            acceptance.expected_values,
            rtol=1e-12,
            atol=1e-12,
        )
    assert python_trace == _expected_trace(acceptance.expected_count)

    compiled_model = acceptance.case.build()
    assert compiled_model.composition is not python_model.composition
    assert next(iter(compiled_model.inputs)) is not next(iter(python_model.inputs))
    assert compiled_model.outputs[0].owner is not python_model.outputs[0].owner
    _assert_executable_ir_contract(
        acceptance,
        compiled_model,
        batched_backend,
    )

    comparison = assert_matches_python(
        acceptance.case,
        backend=batched_backend,
    )
    np.testing.assert_allclose(
        comparison.python_values,
        python_values,
        rtol=1e-12,
        atol=1e-12,
    )


def _lca_ddm_finished_case() -> SemanticCase:
    build_number = itertools.count()

    def build():
        suffix = next(build_number)
        producer = _stepwise_lca(
            name=f"typed finished producer {suffix}",
            reset_condition=pnl.Never(),
        )
        follower = pnl.DDM(
            function=pnl.DriftDiffusionIntegrator(
                starting_value=0.0,
                rate=1.0,
                noise=0.0,
                threshold=1.0,
                non_decision_time=0.0,
                time_step_size=0.01,
            ),
            output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
            execute_until_finished=False,
            reset_stateful_function_when=pnl.AtTrialStart(),
            name=f"typed finished follower {suffix}",
        )
        composition = pnl.Composition(name=f"typed dynamic finished {suffix}")
        composition.add_nodes([producer, follower])
        composition.add_projection(
            sender=producer,
            receiver=follower,
            projection=pnl.MappingProjection(matrix=[[1.0], [-1.0]]),
        )
        composition.scheduler.add_condition(producer, pnl.Always())
        composition.scheduler.add_condition(follower, pnl.WhenFinished(producer))
        return SemanticModel(
            composition=composition,
            inputs={producer: _LCA_DDM_INPUTS.copy()},
            outputs=(
                follower.output_ports[pnl.DECISION_OUTCOME],
                follower.output_ports[pnl.RESPONSE_TIME],
            ),
        )

    return SemanticCase(
        name="typed_lca_ddm_finished_dependency",
        build=build,
        provenance=(
            f"{_LCA_LLVM_PROVENANCE}; {_WHEN_FINISHED_LLVM_PROVENANCE}; "
            f"{_DDM_LLVM_PROVENANCE}; {_MULTI_TRIAL_LLVM_PROVENANCE}"
        ),
        max_steps=16,
    )


def test_lca_ddm_declares_exact_finished_value_before_execution_support():
    """The existing typed declaration stays intact while execution fails closed."""

    case = _lca_ddm_finished_case()
    python_model = case.build()
    python_values, python_trace = _run_python_oracle(python_model)
    np.testing.assert_allclose(
        python_values,
        _LCA_DDM_RESULTS,
        rtol=1e-12,
        atol=1e-12,
    )
    assert python_trace == _EXPECTED_TRACE

    batched_model = case.build()
    producer = next(iter(batched_model.inputs))
    follower = batched_model.outputs[0].owner
    lowering = lower_composition(
        batched_model.composition,
        outputs=batched_model.outputs,
    )
    graph = lowering.graph
    assert graph is not None
    assert not graph.executable
    assert graph.execution_order == (producer.name, follower.name)

    producer_spec = graph.node(producer.name)
    follower_spec = graph.node(follower.name)
    conditions = {condition.component_id: condition for condition in graph.scheduler}
    producer_condition = conditions[producer_spec.component_id]
    follower_condition = conditions[follower_spec.component_id]
    assert producer_condition.condition_type == "Always"
    assert follower_condition.condition_type == "WhenFinished"
    assert follower_condition.dependency_component_ids == (
        producer_spec.component_id,
    )
    assert len(graph.finished_values) == 1
    finished = graph.finished_values[0]
    assert type(finished) is BatchedFinishedValueSpec
    assert finished.node == producer.name
    assert finished.component_id == producer_spec.component_id
    assert finished.value_id == 0
    assert finished.dtype == "bool"
    assert finished.predicate_kind == "execution_count_at_least"
    assert finished.attrs == {"count": 3}
    assert follower_condition.finished_value_ids == (finished.value_id,)
    assert (
        finished.producer_consideration_set_id
        < follower_condition.consideration_set_id
    )

    report = BatchedCompositionCompiler.diagnose(
        batched_model.composition,
        backend="triton_cpu",
        outputs=batched_model.outputs,
        max_steps=case.max_steps,
    )
    assert not report.model_supported
    assert len(report.model_diagnostics) == 1
    diagnostic = report.model_diagnostics[0]
    assert diagnostic.code == BatchedDiagnosticCode.MODEL_SCHEDULE_NOT_EXECUTABLE
    assert diagnostic.component == producer.name
    assert diagnostic.component_id == f"node:{producer.name}"
    assert diagnostic.reason == "batched schedule kind is not executable yet"
    assert diagnostic.detail == _UNMODELED_COEVOLUTION_DETAIL

    with pytest.raises(BatchedCompileError) as error:
        BatchedCompositionCompiler.compile(
            batched_model.composition,
            backend="triton_cpu",
            outputs=batched_model.outputs,
            max_steps=case.max_steps,
        )
    assert error.value.capability_report == report


@dataclass(frozen=True)
class _DiagnosticModel:
    composition: pnl.Composition
    outputs: tuple
    component: object


@dataclass(frozen=True)
class _RejectionCase:
    name: str
    build: Callable[[], _DiagnosticModel]
    provenance: str
    code: str
    component_kind: str
    reason: str
    detail_contains: str


def _same_set_finished_dependency():
    producer = pnl.TransferMechanism(input_shapes=1, name="same-set producer")
    follower = pnl.TransferMechanism(input_shapes=1, name="same-set follower")
    composition = pnl.Composition(name="same-set finished dependency")
    composition.add_nodes([producer, follower])
    composition.scheduler.add_condition(follower, pnl.WhenFinished(producer))
    return _DiagnosticModel(composition, (follower.output_port,), follower)


def _multiple_finished_producers():
    left = pnl.TransferMechanism(input_shapes=1, name="left finished producer")
    right = pnl.TransferMechanism(input_shapes=1, name="right finished producer")
    follower = pnl.TransferMechanism(input_shapes=1, name="multi-finished follower")
    composition = pnl.Composition(name="multiple finished producers")
    composition.add_nodes([follower, right, left])
    composition.add_projection(sender=left, receiver=follower)
    composition.add_projection(sender=right, receiver=follower)
    composition.scheduler.add_condition(
        follower,
        pnl.All(pnl.WhenFinished(left), pnl.WhenFinished(right)),
    )
    return _DiagnosticModel(composition, (follower.output_port,), follower)


def _unsupported_trial_termination():
    mechanism = pnl.TransferMechanism(
        input_shapes=1,
        name="custom termination mechanism",
    )
    composition = pnl.Composition(
        pathways=mechanism,
        termination_processing={pnl.TimeScale.TRIAL: pnl.AtPass(4)},
        name="custom finished termination",
    )
    return _DiagnosticModel(composition, (mechanism.output_port,), composition)


def _unsupported_finished_control():
    source = pnl.TransferMechanism(input_shapes=1, name="finished control monitor")
    target = pnl.TransferMechanism(input_shapes=1, name="finished control target")
    controller = pnl.ControlMechanism(
        function=pnl.Identity(),
        monitor_for_control=source,
        control_signals=[(pnl.SLOPE, target)],
        modulation=pnl.OVERRIDE,
        name="finished-state controller",
    )
    composition = pnl.Composition(name="finished control boundary")
    composition.add_nodes([target, controller, source])
    composition.scheduler.add_condition(controller, pnl.WhenFinished(source))
    return _DiagnosticModel(composition, (target.output_port,), controller)


REJECTION_CASES = (
    _RejectionCase(
        name="same_set_dependency",
        build=_same_set_finished_dependency,
        provenance=_WHEN_FINISHED_LLVM_PROVENANCE,
        code=BatchedDiagnosticCode.MODEL_SCHEDULE_NOT_EXECUTABLE,
        component_kind="node",
        reason="batched schedule kind is not executable yet",
        detail_contains="WhenFinished requires dynamic_lane_local",
    ),
    _RejectionCase(
        name="multiple_finished_producers",
        build=_multiple_finished_producers,
        provenance=_WHEN_FINISHED_LLVM_PROVENANCE,
        code=BatchedDiagnosticCode.MODEL_SCHEDULER_CONDITION_UNSUPPORTED,
        component_kind="node",
        reason="unsupported scheduler condition for static batched graph",
        detail_contains="All",
    ),
    _RejectionCase(
        name="custom_trial_termination",
        build=_unsupported_trial_termination,
        provenance=_TERMINATION_LLVM_PROVENANCE,
        code=BatchedDiagnosticCode.MODEL_SCHEDULER_TERMINATION_UNSUPPORTED,
        component_kind="composition",
        reason="unsupported scheduler termination for batched v2",
        detail_contains="AtPass(args=(4,)",
    ),
    _RejectionCase(
        name="finished_control",
        build=_unsupported_finished_control,
        provenance=(
            f"{_WHEN_FINISHED_LLVM_PROVENANCE}; {_CONTROL_LLVM_PROVENANCE}"
        ),
        code=BatchedDiagnosticCode.MODEL_UNSUPPORTED,
        component_kind="component",
        reason="unsupported generic ControlMechanism for batched v2",
        detail_contains=".slope",
    ),
)


@pytest.mark.parametrize("case", REJECTION_CASES, ids=lambda case: case.name)
def test_dynamic_finished_boundaries_have_structured_rejections(case):
    model = case.build()
    report = BatchedCompositionCompiler.diagnose(
        model.composition,
        backend="triton_cpu",
        outputs=model.outputs,
        max_steps=16,
    )

    assert not report.model_supported, case.provenance
    matches = [
        diagnostic
        for diagnostic in report.model_diagnostics
        if diagnostic.code == case.code
        and diagnostic.component == getattr(model.component, "name", None)
    ]
    assert len(matches) == 1, case.provenance
    diagnostic = matches[0]
    assert diagnostic.component_id == (
        f"{case.component_kind}:{model.component.name}"
    )
    assert diagnostic.reason == case.reason
    assert case.detail_contains in diagnostic.detail
    assert all(item.component_id is not None for item in report.model_diagnostics)

    with pytest.raises(BatchedCompileError) as error:
        BatchedCompositionCompiler.compile(
            model.composition,
            backend="triton_cpu",
            outputs=model.outputs,
            max_steps=16,
        )
    assert error.value.capability_report == report
