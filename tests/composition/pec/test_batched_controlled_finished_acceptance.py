"""Acceptance boundary for dynamically controlled ``WhenFinished`` counts.

These cases isolate generic scheduler and control semantics used by larger
models: a scalar cue overrides an LCA's trial termination threshold, the LCA
executes once per pass, and a strictly later stateless TransferMechanism runs
when that LCA is finished.  The LCA is the first supported stateful producer;
its class and names are not model-recognition signals.

Fresh Python compositions pin the semantic oracle, including trial-varying and
control-parameter-varying execution counts.  Batched compilation must continue
to reject this boundary structurally until the effective controlled threshold
is represented in executable IR.  The rejection assertions are the conversion
point for later Python/interpreter/GPU parity tests.
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
from psyneulink.core.batched.ir import (
    BatchedFinishedValueSpec,
    BatchedResetSpec,
    BatchedSchedulerSpec,
)


pytestmark = [pytest.mark.batched, pytest.mark.composition]


_CONTROL_LLVM_PROVENANCE = (
    "tests/composition/test_control.py::"
    "TestControlMechanisms::test_control_of_mech_port"
)
_SCHEDULER_LLVM_PROVENANCE = (
    "tests/scheduling/test_scheduler.py::TestFeedback::test_time_termination_measures"
)
_LCA_LLVM_PROVENANCE = (
    "tests/mechanisms/test_lca.py::TestLCA::test_LCAMechanism_threshold"
)
_PROVENANCE = (
    f"{_CONTROL_LLVM_PROVENANCE}; {_SCHEDULER_LLVM_PROVENANCE}; {_LCA_LLVM_PROVENANCE}"
)

_TASK_INPUTS = np.asarray(
    [[1.0, -1.0], [0.5, 0.25], [-0.5, 1.5]],
    dtype=float,
)
_CUE_INPUTS = (1.0, 2.0, 3.0)
_BUILD_NUMBERS = itertools.count()


@dataclass(frozen=True)
class _ControlledFinishedModel:
    composition: pnl.Composition
    inputs: dict
    output: object
    cue: object
    task: object
    controller: object
    producer: object
    follower: object


@dataclass(frozen=True)
class _ControlledFinishedCase:
    name: str
    reset_factory: Callable[[], object]
    controller_intercept: float
    expected_counts: tuple[int, ...]
    expected_values: np.ndarray
    provenance: str = _PROVENANCE

    def build(self) -> _ControlledFinishedModel:
        build_number = next(_BUILD_NUMBERS)
        stem = f"controlled finished {self.name} {build_number}"
        cue = pnl.TransferMechanism(
            input_shapes=1,
            function=pnl.Linear(),
            name=f"{stem} cue",
        )
        task = pnl.TransferMechanism(
            input_shapes=2,
            function=pnl.Linear(),
            name=f"{stem} task",
        )
        producer = pnl.LCAMechanism(
            input_shapes=2,
            function=pnl.Logistic(gain=1.0),
            leak=0.0,
            competition=0.0,
            self_excitation=0.0,
            noise=0.0,
            termination_measure=pnl.TimeScale.TRIAL,
            # This deliberately differs from every effective controlled value.
            termination_threshold=9.0,
            time_step_size=0.5,
            execute_until_finished=False,
            reset_stateful_function_when=self.reset_factory(),
            name=f"{stem} producer",
        )
        controller = pnl.ControlMechanism(
            function=pnl.Linear(
                slope=1.0,
                intercept=self.controller_intercept,
            ),
            monitor_for_control=cue,
            control_signals=[(pnl.TERMINATION_THRESHOLD, producer)],
            modulation=pnl.OVERRIDE,
            name=f"{stem} controller",
        )
        follower = pnl.TransferMechanism(
            input_shapes=1,
            function=pnl.Linear(slope=2.0, intercept=-0.25),
            name=f"{stem} follower",
        )

        composition = pnl.Composition(name=f"{stem} composition")
        composition.add_nodes([task, cue, controller, producer, follower])
        composition.add_projection(sender=task, receiver=producer)
        composition.add_projection(
            sender=producer,
            receiver=follower,
            projection=pnl.MappingProjection(matrix=[[1.0], [-1.0]]),
        )
        composition.scheduler.add_condition(cue, pnl.AtPass(0))
        composition.scheduler.add_condition(task, pnl.AtPass(0))
        composition.scheduler.add_condition(controller, pnl.AtPass(0))
        composition.scheduler.add_condition(producer, pnl.Always())
        composition.scheduler.add_condition(follower, pnl.WhenFinished(producer))

        return _ControlledFinishedModel(
            composition=composition,
            inputs={
                task: _TASK_INPUTS.copy(),
                cue: np.asarray(_CUE_INPUTS, dtype=float).reshape(-1, 1),
            },
            output=follower.output_port,
            cue=cue,
            task=task,
            controller=controller,
            producer=producer,
            follower=follower,
        )


RESET_INTERCEPT_ZERO = _ControlledFinishedCase(
    name="reset_intercept_zero",
    reset_factory=pnl.AtTrialStart,
    controller_intercept=0.0,
    expected_counts=(1, 2, 3),
    expected_values=np.asarray(
        [
            [0.23983732480741837],
            [-0.12943433936788695],
            [-1.4176584685525668],
        ],
        dtype=float,
    ),
)
RESET_INTERCEPT_ONE = _ControlledFinishedCase(
    name="reset_intercept_one",
    reset_factory=pnl.AtTrialStart,
    controller_intercept=1.0,
    expected_counts=(2, 3, 4),
    expected_values=np.asarray(
        [
            [0.6742343145200196],
            [-0.0769758015573534],
            [-1.617265410904876],
        ],
        dtype=float,
    ),
)
PERSISTENT_INTERCEPT_ZERO = _ControlledFinishedCase(
    name="persistent_intercept_zero",
    reset_factory=pnl.Never,
    controller_intercept=0.0,
    expected_counts=(1, 2, 3),
    expected_values=np.asarray(
        [
            [0.23983732480741837],
            [0.33647015903160593],
            [-0.8872411541841685],
        ],
        dtype=float,
    ),
)

CONTROLLED_FINISHED_CASES = (
    RESET_INTERCEPT_ZERO,
    RESET_INTERCEPT_ONE,
    PERSISTENT_INTERCEPT_ZERO,
)


def _result_index(composition, output_port) -> int:
    matches = []
    for index, cim_input in enumerate(composition.output_CIM.input_ports):
        if any(
            projection.sender is output_port for projection in cim_input.path_afferents
        ):
            matches.append(index)
    assert len(matches) == 1, output_port.full_name
    return matches[0]


def _selected_results(model: _ControlledFinishedModel) -> np.ndarray:
    result_index = _result_index(model.composition, model.output)
    return np.asarray(
        [
            np.asarray(trial[result_index], dtype=float).reshape(-1)
            for trial in model.composition.results
        ],
        dtype=float,
    )


def _role_trace(model: _ControlledFinishedModel):
    roles = {
        model.cue: "cue",
        model.task: "task",
        model.controller: "controller",
        model.producer: "producer",
        model.follower: "follower",
    }
    execution_list = model.composition.scheduler.execution_list[
        model.composition.default_execution_id
    ]
    return tuple(
        frozenset(roles[node] for node in execution_set)
        for execution_set in execution_list
    )


def _expected_role_trace(execution_counts):
    trace = []
    for execution_count in execution_counts:
        trace.extend(
            (
                frozenset({"cue", "task"}),
                frozenset({"controller"}),
            )
        )
        trace.extend(frozenset({"producer"}) for _ in range(execution_count))
        trace.append(frozenset({"follower"}))
    return tuple(trace)


def _run_python(model: _ControlledFinishedModel):
    model.composition.run(
        inputs=model.inputs,
        execution_mode=pnl.ExecutionMode.Python,
    )
    return _selected_results(model), _role_trace(model)


def _scheduler_history(composition):
    return {
        execution_id: tuple(
            frozenset(execution_set) for execution_set in execution_list
        )
        for execution_id, execution_list in composition.scheduler.execution_list.items()
    }


def _component_id(lowering, component) -> int:
    matches = [
        component_id
        for component_id, bound_component in lowering.bindings.nodes_by_id.items()
        if bound_component is component
    ]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize(
    "case",
    CONTROLLED_FINISHED_CASES,
    ids=lambda case: case.name,
)
def test_python_oracle_preserves_controlled_finished_semantics(case):
    model = case.build()
    independent_model = case.build()

    assert independent_model.composition is not model.composition
    assert independent_model.producer is not model.producer
    values, trace = _run_python(model)

    np.testing.assert_allclose(values, case.expected_values, rtol=1e-12, atol=1e-12)
    assert trace == _expected_role_trace(case.expected_counts), case.provenance
    assert model.producer.num_executions_before_finished == case.expected_counts[-1]


def test_python_control_parameter_rows_produce_independent_execution_counts():
    zero_model = RESET_INTERCEPT_ZERO.build()
    one_model = RESET_INTERCEPT_ONE.build()

    zero_values, zero_trace = _run_python(zero_model)
    one_values, one_trace = _run_python(one_model)

    assert zero_trace == _expected_role_trace((1, 2, 3))
    assert one_trace == _expected_role_trace((2, 3, 4))
    assert not np.allclose(zero_values, one_values)


def test_python_reset_policy_changes_state_history_not_controlled_counts():
    reset_model = RESET_INTERCEPT_ZERO.build()
    persistent_model = PERSISTENT_INTERCEPT_ZERO.build()

    reset_values, reset_trace = _run_python(reset_model)
    persistent_values, persistent_trace = _run_python(persistent_model)

    assert reset_trace == persistent_trace
    np.testing.assert_allclose(reset_values[0], persistent_values[0])
    assert not np.allclose(reset_values[1:], persistent_values[1:])


@pytest.mark.parametrize(
    "case",
    CONTROLLED_FINISHED_CASES,
    ids=lambda case: case.name,
)
def test_batched_controlled_finished_schedule_remains_structured_rejection(case):
    model = case.build()
    history_before = _scheduler_history(model.composition)

    lowering = lower_composition(
        model.composition,
        outputs=(model.output,),
    )
    graph = lowering.graph
    assert graph is not None
    assert not graph.executable
    assert graph.fusion_kind == STATEFUL_GRAPH_FUSION
    assert graph.metadata["schedule_kind"] == "precomputed_trace"
    assert not graph.rng_streams

    cue_id = _component_id(lowering, model.cue)
    task_id = _component_id(lowering, model.task)
    controller_id = _component_id(lowering, model.controller)
    producer_id = _component_id(lowering, model.producer)
    follower_id = _component_id(lowering, model.follower)

    control_node = graph.node(model.controller.name)
    assert control_node.attrs["absorbed_control"] == {
        "source": model.cue.name,
        "target": model.producer.name,
        "parameter": pnl.TERMINATION_THRESHOLD,
        "modulation": "OVERRIDE",
    }
    assert model.controller.name not in graph.execution_order
    assert set(graph.execution_order) == {
        model.cue.name,
        model.task.name,
        model.producer.name,
        model.follower.name,
    }

    schedule = {spec.component_id: spec for spec in graph.scheduler}
    assert all(type(spec) is BatchedSchedulerSpec for spec in schedule.values())
    assert schedule[cue_id].condition_type == "AtPass"
    assert schedule[cue_id].attrs == {
        "pass_index": 0,
        "time_scale": "ENVIRONMENT_STATE_UPDATE",
    }
    assert schedule[task_id].condition_type == "AtPass"
    assert schedule[task_id].attrs == schedule[cue_id].attrs
    assert schedule[controller_id].condition_type == "AtPass"
    assert schedule[controller_id].attrs == schedule[cue_id].attrs
    assert schedule[producer_id].condition_type == "Always"
    assert schedule[producer_id].attrs == {}
    assert schedule[follower_id].condition_type == "WhenFinished"
    assert schedule[follower_id].dependencies == (model.producer.name,)
    assert schedule[follower_id].dependency_component_ids == (producer_id,)

    assert len(graph.finished_values) == 1
    finished = graph.finished_values[0]
    assert type(finished) is BatchedFinishedValueSpec
    assert finished.node == model.producer.name
    assert finished.component_id == producer_id
    assert schedule[follower_id].finished_value_ids == (finished.value_id,)

    assert len(graph.resets) == 1
    reset = graph.resets[0]
    assert type(reset) is BatchedResetSpec
    assert reset.node == model.producer.name
    assert reset.component_id == producer_id
    assert reset.condition_type == type(case.reset_factory()).__name__
    assert reset.state_ids
    assert {
        state.component_id
        for state in graph.states
        if state.state_id in reset.state_ids
    } == {producer_id}

    report = BatchedCompositionCompiler.diagnose(
        model.composition,
        backend="triton_cpu",
        outputs=(model.output,),
        max_steps=16,
    )
    assert _scheduler_history(model.composition) == history_before
    assert not report.model_supported
    assert report.codegen_ready is None
    assert report.metadata["fusion_kind"] == STATEFUL_GRAPH_FUSION
    assert report.metadata["schedule_kind"] == "precomputed_trace"
    assert len(report.model_diagnostics) == 1

    diagnostic = report.model_diagnostics[0]
    assert (
        diagnostic.code == BatchedDiagnosticCode.MODEL_SCHEDULER_CONDITION_UNSUPPORTED
    )
    assert diagnostic.component == model.controller.name
    assert diagnostic.component_id == f"node:{model.controller.name}"
    assert diagnostic.reason == (
        "unsupported absorbed control scheduler condition for batched v2"
    )
    assert diagnostic.detail == (
        f"controller {model.controller.name} uses "
        "AtPass(0, time_scale=ENVIRONMENT_STATE_UPDATE)"
    )

    with pytest.raises(BatchedCompileError) as error:
        BatchedCompositionCompiler.compile(
            model.composition,
            backend="triton_cpu",
            outputs=(model.output,),
            max_steps=16,
        )
    assert error.value.capability_report == report
    assert diagnostic.formatted_reason in str(error.value)
    assert _scheduler_history(model.composition) == history_before
