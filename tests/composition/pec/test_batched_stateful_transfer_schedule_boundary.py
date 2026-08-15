"""Python-oracle boundary for scheduled integrating TransferMechanisms.

This corpus adapts existing LLVM-covered reset and scheduler cases to the
batched axes.  The compiler currently supports only the exact single-update
optimization: a canonical ``AtTrialStart`` reset and an explicit ``AtPass(n)``
prove that an AdaptiveIntegrator or SimpleIntegrator advances once from its
initializer in every trial.  General retained TransferMechanism state remains
fail-closed.

CSI is not identified here.  The cases exercise the ordinary integrator,
scheduler, projection, and dependency semantics that CSI needs.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import itertools

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompileError,
    BatchedCompositionCompiler,
    BatchedCompositionIR,
    BatchedDiagnosticCode,
)
from psyneulink.core.batched.graph import lower_composition
from psyneulink.core.batched.kernel_ir import iter_kernel_ops, lower_to_kernel_ir
from psyneulink.core.scheduling.condition import Condition

from batched_semantic_test_support import (
    SemanticCase,
    SemanticModel,
    assert_matches_python,
)


pytestmark = [pytest.mark.batched, pytest.mark.composition]


_TRANSFER_RESET_LLVM_PROVENANCE = (
    "tests/mechanisms/test_transfer_mechanism.py::"
    "test_integrator_mode_reset_in_composition"
)
_INTEGRATOR_RESET_LLVM_PROVENANCE = (
    "tests/mechanisms/test_integrator_mechanism.py::"
    "TestStatefulness::test_reset_stateful_function_when_composition"
)
_AT_PASS_PROVENANCE = (
    "tests/scheduling/test_condition.py::"
    "TestCondition::TestTime::test_AtPass_in_middle"
)

_INPUTS = np.array([[1.0], [-2.0], [0.5]])
_BUILD_NUMBERS = itertools.count()


@dataclass(frozen=True)
class _ScheduledModel(SemanticModel):
    stateful: object
    roles: Mapping[object, str]


@dataclass(frozen=True)
class _FoldAcceptance:
    case: SemanticCase
    expected_affine: tuple[float, float]
    expected_schedule_kind: str
    expected_trace_steps: tuple[tuple[int, int, str], ...]
    expected_python_trace: tuple[frozenset[str], ...]
    expected_values: np.ndarray


@dataclass(frozen=True)
class _RejectionCase:
    name: str
    build: Callable[[], _ScheduledModel]
    provenance: str
    expected_values: np.ndarray
    expected_trace: tuple[frozenset[str], ...]


def _adaptive_integrator():
    return pnl.AdaptiveIntegrator(
        rate=0.35,
        initializer=0.4,
        offset=0.0,
        noise=0.0,
    )


def _simple_integrator():
    return pnl.SimpleIntegrator(
        rate=0.6,
        initializer=-0.25,
        offset=0.1,
        noise=0.0,
    )


def _make_two_node_model(
    *,
    name: str,
    integrator_factory: Callable,
    reset_factory: Callable,
    producer_condition_factory: Callable,
    follower_condition_factory: Callable | None = None,
    projection_matrix: float = 1.0,
    follower_slope: float = 1.0,
    follower_intercept: float = 0.0,
) -> _ScheduledModel:
    suffix = next(_BUILD_NUMBERS)
    producer = pnl.TransferMechanism(
        input_shapes=1,
        integrator_mode=True,
        integrator_function=integrator_factory(),
        function=pnl.Linear(slope=1.5, intercept=-0.2),
        reset_stateful_function_when=reset_factory(),
        name=f"{name} producer {suffix}",
    )
    follower = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Linear(
            slope=follower_slope,
            intercept=follower_intercept,
        ),
        name=f"{name} follower {suffix}",
    )
    composition = pnl.Composition(
        pathways=[[
            producer,
            pnl.MappingProjection(matrix=[[projection_matrix]]),
            follower,
        ]],
        name=f"{name} composition {suffix}",
    )
    composition.scheduler.add_condition(
        producer,
        producer_condition_factory(),
    )
    if follower_condition_factory is not None:
        composition.scheduler.add_condition(
            follower,
            follower_condition_factory(),
        )
    return _ScheduledModel(
        composition=composition,
        inputs={producer: _INPUTS.copy()},
        outputs=(follower.output_port,),
        stateful=producer,
        roles={producer: "stateful", follower: "follower"},
    )


def _fold_case(
    *,
    name: str,
    integrator_factory: Callable,
    pass_index: int,
    expected_affine: tuple[float, float],
    expected_schedule_kind: str,
    expected_values,
) -> _FoldAcceptance:
    def build():
        return _make_two_node_model(
            name=name,
            integrator_factory=integrator_factory,
            reset_factory=pnl.AtTrialStart,
            producer_condition_factory=lambda: pnl.AtPass(pass_index),
            projection_matrix=1.25,
            follower_slope=-2.0,
            follower_intercept=0.25,
        )

    trace_steps = ()
    if pass_index:
        trace_steps = (
            (pass_index, 0, "stateful"),
            (pass_index, 1, "follower"),
        )
    one_trial_python_trace = (
        *(frozenset() for _ in range(pass_index)),
        frozenset({"stateful"}),
        frozenset({"follower"}),
    )
    return _FoldAcceptance(
        case=SemanticCase(
            name=name,
            build=build,
            provenance=(
                f"{_TRANSFER_RESET_LLVM_PROVENANCE}; "
                f"{_INTEGRATOR_RESET_LLVM_PROVENANCE}; "
                f"{_AT_PASS_PROVENANCE}"
            ),
            parameter_sets=({}, {}),
            num_estimates=2,
            max_steps=8,
            atol=1e-6,
            rtol=1e-5,
        ),
        expected_affine=expected_affine,
        expected_schedule_kind=expected_schedule_kind,
        expected_trace_steps=trace_steps,
        expected_python_trace=tuple(
            step
            for _ in range(len(_INPUTS))
            for step in one_trial_python_trace
        ),
        expected_values=np.asarray(expected_values, dtype=float).reshape(-1, 1),
    )


FOLD_ACCEPTANCES = (
    _fold_case(
        name="adaptive_at_pass_zero",
        integrator_factory=_adaptive_integrator,
        pass_index=0,
        expected_affine=(0.35, 0.26),
        expected_schedule_kind="static_graph",
        expected_values=(-1.5375, 2.4, -0.88125),
    ),
    _fold_case(
        name="simple_delayed_at_pass_three",
        integrator_factory=_simple_integrator,
        pass_index=3,
        expected_affine=(0.6, -0.15),
        expected_schedule_kind="precomputed_trace",
        expected_values=(-0.9375, 5.8125, 0.1875),
    ),
)


def _repeated_adaptive_model():
    return _make_two_node_model(
        name="repeated adaptive",
        integrator_factory=lambda: pnl.AdaptiveIntegrator(
            rate=0.25,
            initializer=0.4,
            offset=0.0,
            noise=0.0,
        ),
        reset_factory=pnl.AtTrialStart,
        producer_condition_factory=pnl.Always,
        follower_condition_factory=lambda: pnl.AtPass(3),
    )


def _never_persistent_simple_model():
    return _make_two_node_model(
        name="persistent simple",
        integrator_factory=_simple_integrator,
        reset_factory=pnl.Never,
        producer_condition_factory=lambda: pnl.AtPass(0),
    )


def _implicit_dependency_adaptive_model():
    suffix = next(_BUILD_NUMBERS)
    source = pnl.TransferMechanism(
        input_shapes=1,
        name=f"implicit dependency source {suffix}",
    )
    stateful = pnl.TransferMechanism(
        input_shapes=1,
        integrator_mode=True,
        integrator_function=pnl.AdaptiveIntegrator(
            rate=0.25,
            initializer=0.4,
            offset=0.0,
            noise=0.0,
        ),
        function=pnl.Linear(slope=1.5, intercept=-0.2),
        reset_stateful_function_when=pnl.AtTrialStart(),
        name=f"implicit dependency adaptive {suffix}",
    )
    follower = pnl.TransferMechanism(
        input_shapes=1,
        name=f"implicit dependency follower {suffix}",
    )
    composition = pnl.Composition(
        pathways=[[source, stateful, follower]],
        name=f"implicit dependency composition {suffix}",
    )
    composition.scheduler.add_condition(source, pnl.AtPass(2))
    return _ScheduledModel(
        composition=composition,
        inputs={source: _INPUTS.copy()},
        outputs=(follower.output_port,),
        stateful=stateful,
        roles={
            source: "source",
            stateful: "stateful",
            follower: "follower",
        },
    )


def _at_pass_reset_simple_model():
    return _make_two_node_model(
        name="AtPass reset simple",
        integrator_factory=_simple_integrator,
        reset_factory=lambda: pnl.AtPass(0),
        producer_condition_factory=lambda: pnl.AtPass(0),
    )


def _at_trial_start_impostor_simple_model():
    impostor = type("AtTrialStart", (Condition,), {})(lambda: False)
    return _make_two_node_model(
        name="AtTrialStart impostor simple",
        integrator_factory=_simple_integrator,
        reset_factory=lambda: impostor,
        producer_condition_factory=lambda: pnl.AtPass(0),
    )


def _unsupported_accumulator_model():
    return _make_two_node_model(
        name="unsupported accumulator",
        integrator_factory=lambda: pnl.AccumulatorIntegrator(
            rate=0.5,
            increment=0.1,
            initializer=0.2,
            noise=0.0,
        ),
        reset_factory=pnl.AtTrialStart,
        producer_condition_factory=lambda: pnl.AtPass(0),
    )


def _three_trials(*steps):
    return tuple(step for _ in range(len(_INPUTS)) for step in steps)


_STATEFUL = frozenset({"stateful"})
_FOLLOWER = frozenset({"follower"})
_SOURCE = frozenset({"source"})
_EMPTY = frozenset()


REJECTION_CASES = (
    _RejectionCase(
        name="repeated_adaptive_with_delayed_follower",
        build=_repeated_adaptive_model,
        provenance=(
            f"{_TRANSFER_RESET_LLVM_PROVENANCE}; {_AT_PASS_PROVENANCE}"
        ),
        expected_values=np.array([[1.015234375], [-2.0609375], [0.5025390625]]),
        expected_trace=_three_trials(
            _STATEFUL,
            _STATEFUL,
            _STATEFUL,
            _STATEFUL,
            _FOLLOWER,
        ),
    ),
    _RejectionCase(
        name="never_persists_between_trials",
        build=_never_persistent_simple_model,
        provenance=_INTEGRATOR_RESET_LLVM_PROVENANCE,
        expected_values=np.array([[0.475], [-1.175], [-0.575]]),
        expected_trace=_three_trials(_STATEFUL, _FOLLOWER),
    ),
    _RejectionCase(
        name="implicit_dependency_is_not_an_explicit_single_fire_proof",
        build=_implicit_dependency_adaptive_model,
        provenance=(
            f"{_TRANSFER_RESET_LLVM_PROVENANCE}; {_AT_PASS_PROVENANCE}"
        ),
        expected_values=np.array([[0.625], [-0.5], [0.4375]]),
        expected_trace=_three_trials(
            _EMPTY,
            _EMPTY,
            _SOURCE,
            _STATEFUL,
            _FOLLOWER,
        ),
    ),
    _RejectionCase(
        name="AtPass_reset_is_not_AtTrialStart",
        build=_at_pass_reset_simple_model,
        provenance=_INTEGRATOR_RESET_LLVM_PROVENANCE,
        expected_values=np.array([[0.475], [-2.225], [0.025]]),
        expected_trace=_three_trials(_STATEFUL, _FOLLOWER),
    ),
    _RejectionCase(
        name="condition_name_does_not_establish_reset_identity",
        build=_at_trial_start_impostor_simple_model,
        provenance=_INTEGRATOR_RESET_LLVM_PROVENANCE,
        expected_values=np.array([[0.475], [-1.175], [-0.575]]),
        expected_trace=_three_trials(_STATEFUL, _FOLLOWER),
    ),
    _RejectionCase(
        name="unimplemented_integrator_function",
        build=_unsupported_accumulator_model,
        provenance=_TRANSFER_RESET_LLVM_PROVENANCE,
        expected_values=np.array([[0.1], [0.1], [0.1]]),
        expected_trace=_three_trials(_STATEFUL, _FOLLOWER),
    ),
)


def _run_python_oracle(model: _ScheduledModel):
    model.composition.run(
        inputs=model.inputs,
        execution_mode=pnl.ExecutionMode.Python,
    )
    indices = [_result_index(model.composition, output) for output in model.outputs]
    values = np.asarray(
        [
            np.concatenate([
                np.asarray(trial[index], dtype=float).reshape(-1)
                for index in indices
            ])
            for trial in model.composition.results
        ],
        dtype=float,
    )
    trace = tuple(
        frozenset(model.roles[node] for node in execution_set)
        for execution_set in model.composition.scheduler.execution_list[
            model.composition.default_execution_id
        ]
    )
    return values, trace


def _result_index(composition, output_port):
    matches = [
        index
        for index, cim_input in enumerate(composition.output_CIM.input_ports)
        if any(
            projection.sender is output_port
            for projection in cim_input.path_afferents
        )
    ]
    assert len(matches) == 1, output_port.full_name
    return matches[0]


@pytest.mark.parametrize(
    "acceptance",
    FOLD_ACCEPTANCES,
    ids=lambda acceptance: acceptance.case.name,
)
def test_single_update_fold_has_explicit_graph_and_kernel_ir_contract(acceptance):
    model = acceptance.case.build()
    lowering = lower_composition(
        model.composition,
        outputs=model.outputs,
    )

    assert not lowering.rejected_nodes, acceptance.case.provenance
    assert not lowering.rejected_conditions, acceptance.case.provenance
    assert lowering.model_kind is not None
    assert lowering.schedule_kind == acceptance.expected_schedule_kind
    graph = lowering.graph
    assert graph is not None
    assert graph.executable
    assert graph.fusion_kind == "stateless_graph"
    assert graph.states == ()
    assert graph.resets == ()

    stateful_node = graph.node(model.stateful.name)
    np.testing.assert_allclose(
        stateful_node.attrs["integrator_pre"],
        acceptance.expected_affine,
        rtol=0.0,
        atol=1e-12,
    )

    semantic_ir = BatchedCompositionIR(
        model_kind=lowering.model_kind,
        node_names=tuple(node.name for node in graph.nodes),
        params=lowering.params,
        output_names=tuple(output.name for output in graph.outputs),
        max_steps=acceptance.case.max_steps,
        graph=graph,
    )
    kernel_ir = lower_to_kernel_ir(semantic_ir)
    assert kernel_ir.executable
    assert kernel_ir.states == ()
    assert kernel_ir.resets == ()
    function_ops = [
        op
        for op in iter_kernel_ops(kernel_ir)
        if op.kind == "CallFunction" and op.target == model.stateful.name
    ]
    assert len(function_ops) == 1
    np.testing.assert_allclose(
        function_ops[0].attrs["integrator_pre"],
        acceptance.expected_affine,
        rtol=0.0,
        atol=1e-12,
    )

    if not acceptance.expected_trace_steps:
        assert kernel_ir.schedule_trace is None
    else:
        assert kernel_ir.schedule_trace is not None
        role_ids = {
            role: graph.node(node.name).component_id
            for node, role in model.roles.items()
        }
        assert tuple(
            (
                step.pass_index,
                step.consideration_set_id,
                step.component_ids,
            )
            for step in kernel_ir.schedule_trace.steps
        ) == tuple(
            (pass_index, set_id, (role_ids[role],))
            for pass_index, set_id, role in acceptance.expected_trace_steps
        )
        assert kernel_ir.schedule_trace.num_passes == 4
        assert kernel_ir.schedule_trace.component_execution_count == 2


@pytest.mark.parametrize(
    "acceptance",
    FOLD_ACCEPTANCES,
    ids=lambda acceptance: acceptance.case.name,
)
def test_single_update_fold_matches_fresh_python_in_every_lane(
    acceptance,
    batched_backend,
):
    oracle_values, oracle_trace = _run_python_oracle(acceptance.case.build())
    np.testing.assert_allclose(
        oracle_values,
        acceptance.expected_values,
        rtol=0.0,
        atol=1e-12,
    )
    assert oracle_trace == acceptance.expected_python_trace

    comparison = assert_matches_python(
        acceptance.case,
        backend=batched_backend,
    )
    np.testing.assert_allclose(
        comparison.python_values,
        acceptance.expected_values,
        rtol=0.0,
        atol=1e-12,
    )
    assert comparison.batched_values.shape == (2, 1, 3, 2, 1)


@pytest.mark.parametrize("case", REJECTION_CASES, ids=lambda case: case.name)
def test_general_stateful_transfer_schedule_cases_fail_closed(case):
    python_model = case.build()
    python_values, python_trace = _run_python_oracle(python_model)
    np.testing.assert_allclose(
        python_values,
        case.expected_values,
        rtol=0.0,
        atol=1e-12,
    )
    assert python_trace == case.expected_trace

    compile_model = case.build()
    report = BatchedCompositionCompiler.diagnose(
        compile_model.composition,
        backend="triton_cpu",
        outputs=compile_model.outputs,
        max_steps=8,
    )

    assert not report.model_supported, case.provenance
    assert report.codegen_ready is None
    matches = [
        diagnostic
        for diagnostic in report.model_diagnostics
        if diagnostic.code
        == BatchedDiagnosticCode.MODEL_STATEFUL_TRANSFER_UNSUPPORTED
        and diagnostic.component == compile_model.stateful.name
    ]
    assert len(matches) == 1, case.provenance
    diagnostic = matches[0]
    assert diagnostic.component_id == f"node:{compile_model.stateful.name}"
    assert diagnostic.reason == (
        "unsupported stateful transfer (integrator_mode) for batched v2"
    )
    assert diagnostic.detail == "integrator_mode=True"

    with pytest.raises(BatchedCompileError) as error:
        BatchedCompositionCompiler.compile(
            compile_model.composition,
            backend="triton_cpu",
            outputs=compile_model.outputs,
            max_steps=8,
        )
    assert error.value.capability_report == report
