"""Acceptance contract for typed per-trial reset of retained batched state.

The executable target is the existing fixed-count scheduler boundary: one
stepwise LCA scheduled ``Always`` projects to a stateless TransferMechanism in
a strictly later consideration set, scheduled ``WhenFinished(LCA)``.  This
module requires ``AtTrialStart`` to reset the LCA's exact retained states before
pass zero of every trial.  ``Never`` is retained as a contrasting persistence
case.  Dynamic scheduling, control, and an LCA-to-DDM follower remain explicit
fail-closed boundaries.
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
from psyneulink.core.batched.graph import (
    STATEFUL_GRAPH_FUSION,
    lower_composition,
)
from psyneulink.core.batched.ir import (
    BatchedCompositionIR,
    BatchedResetSpec,
)
from psyneulink.core.batched.kernel_ir import (
    STATEFUL_LANE_LAYOUT,
    iter_kernel_ops,
    lower_to_kernel_ir,
)
from psyneulink.core.scheduling.condition import Condition

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
_RESET_LLVM_PROVENANCE = (
    "tests/mechanisms/test_integrator_mechanism.py::"
    "TestStatefulness::test_reset_stateful_function_when_composition"
)
_MULTI_TRIAL_LLVM_PROVENANCE = (
    "tests/composition/test_composition.py::"
    "TestRun::test_run_2_mechanisms_with_multiple_trials_of_input_values"
)
_DDM_LLVM_PROVENANCE = (
    "tests/mechanisms/test_ddm_mechanism.py::"
    "test_ddm_is_finished_with_dependency"
)
_CONTROL_LLVM_PROVENANCE = (
    "tests/composition/test_control.py::"
    "TestControlMechanisms::test_control_of_mech_port[OVERRIDE]"
)

_TRIAL_INPUTS = np.array(
    [[1.0, -1.0], [0.5, 0.25], [-0.5, 1.5]],
    dtype=float,
)
_AT_TRIAL_START_RESULTS = np.array(
    [[1.6461957739972028], [-0.031647070293334245], [-1.751958401180123]],
    dtype=float,
)
_NEVER_RESULTS = np.array(
    [[1.6461957739972028], [1.530671300749336], [-0.1541358098825767]],
    dtype=float,
)
_EXECUTIONS_PER_TRIAL = 3
_ONE_TRIAL_TRACE = (
    frozenset({"producer"}),
    frozenset({"producer"}),
    frozenset({"producer"}),
    frozenset({"follower"}),
)
_EXPECTED_TRACE = _ONE_TRIAL_TRACE * len(_TRIAL_INPUTS)

_UNMODELED_COEVOLUTION_DETAIL = (
    "coevolving Always/WhenFinished execution requires explicit finished "
    "predicates and conditional pass regions in KernelIR"
)


def _stepwise_lca(*, name: str, reset_condition, function_bias: float = 0.25):
    return pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(
            gain=1.5,
            bias=function_bias,
            x_0=-0.1,
            scale=1.2,
            offset=0.05,
        ),
        leak=0.0,
        competition=0.0,
        self_excitation=0.0,
        noise=0.0,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=_EXECUTIONS_PER_TRIAL,
        time_step_size=0.5,
        execute_until_finished=False,
        reset_stateful_function_when=reset_condition,
        name=name,
    )


def _reset_case(
    name: str,
    *,
    reset_factory: Callable[[], object],
    reverse_insertion: bool = False,
    renamed_boundary: bool = False,
    newline_name: bool = False,
    function_bias: float = 0.25,
) -> SemanticCase:
    build_number = itertools.count()

    def build():
        suffix = next(build_number)
        if newline_name:
            producer_name = f"reset producer {suffix}\ncomment injection"
            follower_name = f"post-reset follower {suffix}"
        elif renamed_boundary:
            producer_name = f"z / reset producer [{suffix}]"
            follower_name = f"a:post-reset follower[{suffix}]"
        else:
            producer_name = f"reset producer {suffix}"
            follower_name = f"post-reset follower {suffix}"

        producer = _stepwise_lca(
            name=producer_name,
            reset_condition=reset_factory(),
            function_bias=function_bias,
        )
        follower = pnl.TransferMechanism(
            input_shapes=1,
            function=pnl.Linear(slope=2.0, intercept=-0.25),
            name=follower_name,
        )
        composition = pnl.Composition(name=f"typed trial reset {name} {suffix}")
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
            f"{_RESET_LLVM_PROVENANCE}; {_MULTI_TRIAL_LLVM_PROVENANCE}"
        ),
        max_steps=16,
        atol=1e-6,
        rtol=1e-5,
    )


@dataclass(frozen=True)
class _ResetAcceptance:
    case: SemanticCase
    condition_type: str
    expected_values: np.ndarray


AT_TRIAL_START_CASES = (
    _ResetAcceptance(
        case=_reset_case(
            "at_trial_start",
            reset_factory=pnl.AtTrialStart,
        ),
        condition_type="AtTrialStart",
        expected_values=_AT_TRIAL_START_RESULTS,
    ),
    _ResetAcceptance(
        case=_reset_case(
            "at_trial_start_renamed_reverse_insertion",
            reset_factory=pnl.AtTrialStart,
            reverse_insertion=True,
            renamed_boundary=True,
        ),
        condition_type="AtTrialStart",
        expected_values=_AT_TRIAL_START_RESULTS,
    ),
    _ResetAcceptance(
        case=_reset_case(
            "at_trial_start_newline_name",
            reset_factory=pnl.AtTrialStart,
            newline_name=True,
        ),
        condition_type="AtTrialStart",
        expected_values=_AT_TRIAL_START_RESULTS,
    ),
)

NEVER_CASE = _ResetAcceptance(
    case=_reset_case(
        "never_persists",
        reset_factory=pnl.Never,
    ),
    condition_type="Never",
    expected_values=_NEVER_RESULTS,
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


def _kernel_ir(lowering):
    graph = lowering.graph
    assert graph is not None
    return lower_to_kernel_ir(
        BatchedCompositionIR(
            model_kind=lowering.model_kind,
            node_names=tuple(node.name for node in graph.nodes),
            params=lowering.params,
            output_names=tuple(output.name for output in graph.outputs),
            graph=graph,
        )
    )


def _assert_reset_ir_contract(acceptance: _ResetAcceptance):
    model = acceptance.case.build()
    producer = next(iter(model.inputs))
    follower = model.outputs[0].owner
    history_before = dict(model.composition.scheduler.execution_list)
    lowering = lower_composition(
        model.composition,
        outputs=model.outputs,
    )
    assert dict(model.composition.scheduler.execution_list) == history_before

    graph = lowering.graph
    assert graph is not None
    assert graph.executable
    assert lowering.schedule_kind == "precomputed_trace"
    assert not lowering.rejected_nodes
    assert not lowering.rejected_conditions
    assert graph.fusion_kind == STATEFUL_GRAPH_FUSION
    assert graph.execution_order == (producer.name, follower.name)
    assert not graph.rng_streams

    producer_spec = graph.node(producer.name)
    follower_spec = graph.node(follower.name)
    conditions = {condition.component_id: condition for condition in graph.scheduler}
    producer_condition = conditions[producer_spec.component_id]
    follower_condition = conditions[follower_spec.component_id]
    assert producer_condition.condition_type == "Always"
    assert follower_condition.condition_type == "WhenFinished"
    assert producer_condition.consideration_set_id < (
        follower_condition.consideration_set_id
    )
    assert follower_condition.dependency_component_ids == (
        producer_spec.component_id,
    )

    assert len(graph.finished_values) == 1
    finished = graph.finished_values[0]
    assert finished.component_id == producer_spec.component_id
    assert finished.predicate_kind == "execution_count_at_least"
    assert finished.attrs == {"count": _EXECUTIONS_PER_TRIAL}
    assert follower_condition.finished_value_ids == (finished.value_id,)

    producer_states = tuple(
        state
        for state in graph.states
        if state.component_id == producer_spec.component_id
    )
    assert len(producer_states) == 2
    state_ids = tuple(state.state_id for state in producer_states)
    assert len(graph.resets) == 1
    reset = graph.resets[0]
    assert type(reset) is BatchedResetSpec
    assert reset.node == producer.name
    assert reset.component_id == producer_spec.component_id
    assert reset.condition_type == acceptance.condition_type
    assert reset.state_ids == state_ids
    assert reset.attrs == {}
    assert reset.region == "trial"

    kernel = _kernel_ir(lowering)
    assert kernel.executable
    assert kernel.lane_layout.kind == STATEFUL_LANE_LAYOUT
    assert kernel.states == graph.states
    assert kernel.resets == graph.resets
    assert kernel.schedule_trace is not None
    assert kernel.schedule_trace.num_passes == _EXECUTIONS_PER_TRIAL
    assert tuple(op.kind for op in kernel.ops) == (
        "InitializeState",
        "ForTrials",
    )

    initialize, trials = kernel.ops
    assert len(initialize.outputs) == len(state_ids)
    trial_body = tuple(trials.attrs["body"])
    if acceptance.condition_type == "AtTrialStart":
        assert tuple(op.kind for op in trial_body) == (
            "ResetState",
            "ForPasses",
            "StoreOutput",
        )
        reset_op, passes, _ = trial_body
        assert reset_op.target == producer.name
        assert reset_op.attrs["component_id"] == producer_spec.component_id
        assert reset_op.attrs["state_ids"] == state_ids
        assert reset_op.attrs["condition_type"] == "AtTrialStart"
        assert not reset_op.inputs
        assert len(tuple(op for op in iter_kernel_ops(kernel) if op.kind == "ResetState")) == 1
    else:
        assert tuple(op.kind for op in trial_body) == (
            "ForPasses",
            "StoreOutput",
        )
        passes, _ = trial_body
        assert not any(op.kind == "ResetState" for op in iter_kernel_ops(kernel))

    assert passes.attrs["declaration_only"] is False
    assert passes.attrs["trace_kind"] == "precomputed"
    producer_steps = tuple(
        op
        for op in iter_kernel_ops(kernel)
        if op.kind == "StepMechanism" and op.target == producer.name
    )
    assert len(producer_steps) == _EXECUTIONS_PER_TRIAL
    assert tuple(op.attrs["execution_index"] for op in producer_steps) == tuple(
        range(_EXECUTIONS_PER_TRIAL)
    )
    assert all(op.attrs["state_ids"] == state_ids for op in producer_steps)
    return model, kernel


def test_python_oracle_distinguishes_trial_reset_from_persistence():
    reset_values = []
    for acceptance in AT_TRIAL_START_CASES:
        values, trace = _run_python_oracle(acceptance.case.build())
        np.testing.assert_allclose(
            values,
            acceptance.expected_values,
            rtol=1e-12,
            atol=1e-12,
        )
        assert trace == _EXPECTED_TRACE
        reset_values.append(values)

    never_values, never_trace = _run_python_oracle(NEVER_CASE.case.build())
    np.testing.assert_allclose(
        never_values,
        NEVER_CASE.expected_values,
        rtol=1e-12,
        atol=1e-12,
    )
    assert never_trace == _EXPECTED_TRACE
    np.testing.assert_allclose(reset_values[0][0], never_values[0], rtol=0, atol=0)
    assert not np.allclose(reset_values[0][1:], never_values[1:])


@pytest.mark.parametrize(
    "acceptance",
    AT_TRIAL_START_CASES,
    ids=lambda acceptance: acceptance.case.name,
)
def test_at_trial_start_has_typed_reset_before_pass_zero(acceptance):
    _assert_reset_ir_contract(acceptance)


def test_never_preserves_state_without_a_per_trial_reset_op():
    _assert_reset_ir_contract(NEVER_CASE)


@pytest.mark.parametrize(
    "acceptance",
    AT_TRIAL_START_CASES,
    ids=lambda acceptance: acceptance.case.name,
)
def test_at_trial_start_matches_fresh_python(acceptance, batched_backend):
    comparison = assert_matches_python(
        acceptance.case,
        backend=batched_backend,
    )
    np.testing.assert_allclose(
        comparison.python_values,
        acceptance.expected_values,
        rtol=1e-12,
        atol=1e-12,
    )


def test_never_persistence_still_matches_fresh_python(batched_backend):
    comparison = assert_matches_python(
        NEVER_CASE.case,
        backend=batched_backend,
    )
    np.testing.assert_allclose(
        comparison.python_values,
        NEVER_CASE.expected_values,
        rtol=1e-12,
        atol=1e-12,
    )


def test_trial_reset_initializer_uses_each_parameter_and_estimate_lane(
    batched_backend,
):
    biases = (-0.5, 0.75)
    batched_case = _reset_case(
        "trial_reset_parameter_lanes",
        reset_factory=pnl.AtTrialStart,
    )
    batched_model = batched_case.build()
    producer = next(iter(batched_model.inputs))
    plan = BatchedCompositionCompiler.compile(
        batched_model.composition,
        backend=batched_backend,
        outputs=batched_model.outputs,
        max_steps=batched_case.max_steps,
    )
    bias_parameter = plan.ir.graph.node(producer.name).params["bias"]
    result = plan.run(
        inputs=batched_model.inputs,
        parameter_sets=tuple(
            {bias_parameter: bias}
            for bias in biases
        ),
        num_estimates=2,
        seed=0,
    )

    expected = []
    for index, bias in enumerate(biases):
        python_case = _reset_case(
            f"trial_reset_parameter_lane_python_{index}",
            reset_factory=pnl.AtTrialStart,
            function_bias=bias,
        )
        values, trace = _run_python_oracle(python_case.build())
        assert trace == _EXPECTED_TRACE
        expected.append(values)
    expected = np.asarray(expected)[:, None, :, None, :]
    expected = np.broadcast_to(expected, result.values.shape)

    assert result.values.shape == (2, 1, len(_TRIAL_INPUTS), 2, 1)
    assert not np.allclose(result.values[0], result.values[1])
    np.testing.assert_allclose(
        result.values,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


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


def _dynamic_same_set_dependency():
    producer = pnl.TransferMechanism(input_shapes=1, name="dynamic producer")
    follower = pnl.TransferMechanism(input_shapes=1, name="dynamic follower")
    composition = pnl.Composition(name="dynamic same-set reset boundary")
    composition.add_nodes([producer, follower])
    composition.scheduler.add_condition(follower, pnl.WhenFinished(producer))
    return _DiagnosticModel(composition, (follower.output_port,), follower)


def _controlled_finished_dependency():
    source = pnl.TransferMechanism(input_shapes=1, name="reset control monitor")
    target = pnl.TransferMechanism(input_shapes=1, name="reset control target")
    controller = pnl.ControlMechanism(
        function=pnl.Identity(),
        monitor_for_control=source,
        control_signals=[(pnl.SLOPE, target)],
        modulation=pnl.OVERRIDE,
        name="reset boundary controller",
    )
    composition = pnl.Composition(name="controlled reset boundary")
    composition.add_nodes([target, controller, source])
    composition.scheduler.add_condition(controller, pnl.WhenFinished(source))
    return _DiagnosticModel(composition, (target.output_port,), controller)


def _lca_to_ddm_dependency():
    producer = _stepwise_lca(
        name="reset boundary LCA producer",
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
        name="reset boundary DDM follower",
    )
    composition = pnl.Composition(name="LCA DDM reset boundary")
    composition.add_nodes([producer, follower])
    composition.add_projection(
        sender=producer,
        receiver=follower,
        projection=pnl.MappingProjection(matrix=[[1.0], [-1.0]]),
    )
    composition.scheduler.add_condition(producer, pnl.Always())
    composition.scheduler.add_condition(follower, pnl.WhenFinished(producer))
    return _DiagnosticModel(
        composition,
        (
            follower.output_ports[pnl.DECISION_OUTCOME],
            follower.output_ports[pnl.RESPONSE_TIME],
        ),
        producer,
    )


def _unsupported_lca_reset(reset_condition, name):
    producer = _stepwise_lca(
        name=f"{name} producer",
        reset_condition=reset_condition,
    )
    follower = pnl.TransferMechanism(
        input_shapes=1,
        name=f"{name} follower",
    )
    composition = pnl.Composition(name=f"{name} reset boundary")
    composition.add_nodes([producer, follower])
    composition.add_projection(
        sender=producer,
        receiver=follower,
        projection=pnl.MappingProjection(matrix=[[1.0], [-1.0]]),
    )
    composition.scheduler.add_condition(producer, pnl.Always())
    composition.scheduler.add_condition(follower, pnl.WhenFinished(producer))
    return _DiagnosticModel(composition, (follower.output_port,), producer)


def _at_pass_reset():
    return _unsupported_lca_reset(pnl.AtPass(0), "AtPass")


def _reset_condition_name_impostor():
    impostor = type("AtTrialStart", (Condition,), {})(lambda: False)
    return _unsupported_lca_reset(impostor, "impostor")


REJECTION_CASES = (
    _RejectionCase(
        name="dynamic_same_set_dependency",
        build=_dynamic_same_set_dependency,
        provenance=_WHEN_FINISHED_LLVM_PROVENANCE,
        code=BatchedDiagnosticCode.MODEL_SCHEDULE_NOT_EXECUTABLE,
        component_kind="node",
        reason="batched schedule kind is not executable yet",
        detail_contains="WhenFinished requires dynamic_lane_local",
    ),
    _RejectionCase(
        name="controlled_finished_dependency",
        build=_controlled_finished_dependency,
        provenance=(
            f"{_WHEN_FINISHED_LLVM_PROVENANCE}; {_CONTROL_LLVM_PROVENANCE}"
        ),
        code=BatchedDiagnosticCode.MODEL_UNSUPPORTED,
        component_kind="component",
        reason="unsupported generic ControlMechanism for batched v2",
        detail_contains="->",
    ),
    _RejectionCase(
        name="lca_to_ddm_dependency",
        build=_lca_to_ddm_dependency,
        provenance=(
            f"{_LCA_LLVM_PROVENANCE}; {_WHEN_FINISHED_LLVM_PROVENANCE}; "
            f"{_DDM_LLVM_PROVENANCE}"
        ),
        code=BatchedDiagnosticCode.MODEL_SCHEDULE_NOT_EXECUTABLE,
        component_kind="node",
        reason="batched schedule kind is not executable yet",
        detail_contains=_UNMODELED_COEVOLUTION_DETAIL,
    ),
    _RejectionCase(
        name="at_pass_reset",
        build=_at_pass_reset,
        provenance=_RESET_LLVM_PROVENANCE,
        code=BatchedDiagnosticCode.MODEL_UNSUPPORTED,
        component_kind="component",
        reason="unsupported LCA reset policy for batched v2",
        detail_contains="AtPass",
    ),
    _RejectionCase(
        name="reset_condition_name_impostor",
        build=_reset_condition_name_impostor,
        provenance=_RESET_LLVM_PROVENANCE,
        code=BatchedDiagnosticCode.MODEL_UNSUPPORTED,
        component_kind="component",
        reason="unsupported LCA reset policy for batched v2",
        detail_contains="AtTrialStart",
    ),
)


@pytest.mark.parametrize("case", REJECTION_CASES, ids=lambda case: case.name)
def test_reset_slice_keeps_dynamic_control_and_ddm_fail_closed(case):
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
