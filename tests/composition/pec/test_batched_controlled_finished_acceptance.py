"""Acceptance boundary for dynamically controlled ``WhenFinished`` counts.

These cases isolate generic scheduler and control semantics used by larger
models: a scalar cue overrides an LCA's trial termination threshold, the LCA
executes once per pass, and a strictly later stateless TransferMechanism runs
when that LCA is finished.  The LCA is the first supported stateful producer;
its class and names are not model-recognition signals.

Fresh Python compositions pin the semantic oracle, including trial-varying and
control-parameter-varying execution counts.  The exact one-controller topology
is executable; nearby control and dynamic-scheduler shapes remain structured
fail-closed boundaries.
"""

from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass, replace
import itertools

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedAbsorbedProjectionSpec,
    BatchedCompileError,
    BatchedCompositionCompiler,
    BatchedCompositionIR,
    BatchedDiagnosticCode,
    BatchedEffectiveParameterSpec,
    BatchedModulationSpec,
    BatchedParameterBindingSpec,
    BatchedPortSpec,
)
from psyneulink.core.batched import specs as batched_specs
from psyneulink.core.batched.backend.triton.graph_emit import (
    triton_graph_kernel_source,
)
from psyneulink.core.batched.backend.triton.runtime import (
    BatchedTruncationError,
)
from psyneulink.core.batched.graph import STATEFUL_GRAPH_FUSION, lower_composition
from psyneulink.core.batched.ir import (
    BatchedFinishedValueSpec,
    BatchedResetSpec,
    BatchedSchedulerSpec,
)
from psyneulink.core.batched.kernel_ir import lower_to_kernel_ir


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

    def build(
        self,
        *,
        controller_class=pnl.ControlMechanism,
        controller_function=None,
        monitor_projection=None,
        target_parameter=pnl.TERMINATION_THRESHOLD,
    ) -> _ControlledFinishedModel:
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
        if controller_function is None:
            controller_function = pnl.Linear(
                slope=1.0,
                intercept=self.controller_intercept,
            )
        controller_kwargs = {}
        if monitor_projection is None:
            controller_kwargs["monitor_for_control"] = cue
        else:
            controller_kwargs["default_variable"] = [0.0]
        controller = controller_class(
            function=controller_function,
            control_signals=[(target_parameter, producer)],
            modulation=pnl.OVERRIDE,
            name=f"{stem} controller",
            **controller_kwargs,
        )
        follower = pnl.TransferMechanism(
            input_shapes=1,
            function=pnl.Linear(slope=2.0, intercept=-0.25),
            name=f"{stem} follower",
        )

        composition = pnl.Composition(name=f"{stem} composition")
        composition.add_nodes([task, cue, controller, producer, follower])
        if monitor_projection is not None:
            composition.add_projection(
                sender=cue,
                receiver=controller.input_port,
                projection=monitor_projection,
            )
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


def test_controlled_finished_lowering_is_execution_history_invariant():
    model = RESET_INTERCEPT_ONE.build()
    before = lower_composition(model.composition, outputs=(model.output,))

    _run_python(model)
    after = lower_composition(model.composition, outputs=(model.output,))

    assert before.graph is not None
    assert after.graph is not None
    assert before.params == after.params
    assert before.rejected_nodes == after.rejected_nodes
    assert before.rejected_conditions == after.rejected_conditions
    for field in (
        "nodes",
        "ports",
        "states",
        "scheduler",
        "consideration_sets",
        "finished_values",
        "absorbed_projections",
        "effective_parameters",
        "modulations",
        "resets",
        "termination",
        "execution_order",
        "executable",
    ):
        assert getattr(before.graph, field) == getattr(after.graph, field), field


def test_controlled_finished_rejects_parameter_port_name_spoof():
    model = RESET_INTERCEPT_ZERO.build(target_parameter=pnl.BIAS)
    control_projection = model.controller.control_signals[0].efferents[0]
    bias_port = control_projection.receiver
    threshold_port = model.producer.parameter_ports[pnl.TERMINATION_THRESHOLD]
    bias_port.name = pnl.TERMINATION_THRESHOLD
    threshold_port.name = pnl.BIAS

    lowering = lower_composition(model.composition, outputs=(model.output,))
    matches = [
        diagnostic
        for diagnostic in lowering.rejected_nodes
        if diagnostic.component == model.controller.name
        and diagnostic.reason
        == "unsupported control target parameter identity for batched v2"
    ]
    assert len(matches) == 1
    assert lowering.graph is None or not lowering.graph.modulations

    report = BatchedCompositionCompiler.diagnose(
        model.composition,
        outputs=(model.output,),
        max_steps=16,
    )
    assert not report.model_supported
    with pytest.raises(BatchedCompileError):
        BatchedCompositionCompiler.compile(
            model.composition,
            outputs=(model.output,),
            max_steps=16,
        )


def test_controlled_finished_rejects_function_type_name_spoof():
    class Identity(pnl.Linear):
        pass

    model = RESET_INTERCEPT_ZERO.build(
        controller_function=Identity(slope=2.0),
    )
    lowering = lower_composition(model.composition, outputs=(model.output,))
    matches = [
        diagnostic
        for diagnostic in lowering.rejected_nodes
        if diagnostic.component == model.controller.name
        and diagnostic.reason
        == "unsupported generic ControlMechanism for batched v2"
    ]
    assert len(matches) == 1
    assert lowering.graph is None or not lowering.graph.modulations

    report = BatchedCompositionCompiler.diagnose(
        model.composition,
        outputs=(model.output,),
        max_steps=16,
    )
    assert not report.model_supported
    with pytest.raises(BatchedCompileError):
        BatchedCompositionCompiler.compile(
            model.composition,
            outputs=(model.output,),
            max_steps=16,
        )


def test_controlled_finished_rejects_projection_type_name_spoof():
    class MappingProjection(pnl.MappingProjection):
        def _execute(self, *args, **kwargs):
            return 2.0 * super()._execute(*args, **kwargs)

    model = RESET_INTERCEPT_ZERO.build(
        monitor_projection=MappingProjection(matrix=[[1.0]]),
    )
    lowering = lower_composition(model.composition, outputs=(model.output,))
    matches = [
        diagnostic
        for diagnostic in lowering.rejected_nodes
        if diagnostic.component == model.controller.name
        and diagnostic.reason
        == "unsupported generic control projection for batched v2"
    ]
    assert len(matches) == 1
    assert lowering.graph is None or not lowering.graph.modulations

    report = BatchedCompositionCompiler.diagnose(
        model.composition,
        outputs=(model.output,),
        max_steps=16,
    )
    assert not report.model_supported
    with pytest.raises(BatchedCompileError):
        BatchedCompositionCompiler.compile(
            model.composition,
            outputs=(model.output,),
            max_steps=16,
        )


def test_controlled_finished_rejects_controller_type_name_spoof():
    class ControlMechanism(pnl.ControlMechanism):
        def _execute(self, variable=None, context=None, runtime_params=None):
            value = super()._execute(
                variable=variable,
                context=context,
                runtime_params=runtime_params,
            )
            return np.asarray(value) * 2.0

    model = RESET_INTERCEPT_ZERO.build(
        controller_class=ControlMechanism,
        controller_function=pnl.Identity(),
    )
    lowering = lower_composition(model.composition, outputs=(model.output,))
    matches = [
        diagnostic
        for diagnostic in lowering.rejected_nodes
        if diagnostic.component == model.controller.name
        and diagnostic.reason == "unsupported node for batched v2"
    ]
    assert len(matches) == 1
    assert lowering.graph is None

    report = BatchedCompositionCompiler.diagnose(
        model.composition,
        outputs=(model.output,),
        max_steps=16,
    )
    assert not report.model_supported
    with pytest.raises(BatchedCompileError):
        BatchedCompositionCompiler.compile(
            model.composition,
            outputs=(model.output,),
            max_steps=16,
        )


@pytest.mark.parametrize(
    "case",
    CONTROLLED_FINISHED_CASES,
    ids=lambda case: case.name,
)
def test_batched_controlled_finished_schedule_is_executable(case):
    model = case.build()
    history_before = _scheduler_history(model.composition)

    lowering = lower_composition(
        model.composition,
        outputs=(model.output,),
    )
    graph = lowering.graph
    assert graph is not None
    assert graph.executable
    assert graph.fusion_kind == STATEFUL_GRAPH_FUSION
    assert graph.metadata["schedule_kind"] == "dynamic_lane_local"
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

    assert len(graph.modulations) == 1
    modulation = graph.modulations[0]
    assert type(modulation) is BatchedModulationSpec
    assert modulation.modulation_id == 0
    assert modulation.effective_parameter_id == 0
    assert modulation.mode == "OVERRIDE"
    assert modulation.width == 1
    assert modulation.dtype == "float32"
    assert modulation.controller == model.controller.name
    assert modulation.controller_component_id == controller_id
    assert modulation.controller_input_port == model.controller.input_ports[0].name
    assert modulation.control_signal_port == model.controller.control_signals[0].name
    assert modulation.source == model.cue.name
    assert modulation.source_component_id == cue_id
    assert modulation.source_port == model.cue.output_port.name
    assert modulation.target == model.producer.name
    assert modulation.target_component_id == producer_id
    assert modulation.target_parameter == pnl.TERMINATION_THRESHOLD
    source_node = graph.node(model.cue.name)
    target_node = graph.node(model.producer.name)
    assert modulation.source_port_id == source_node.output_port_ids[0]
    assert dict(target_node.parameter_port_ids)[pnl.TERMINATION_THRESHOLD] == (
        modulation.target_parameter_port_id
    )

    signal = model.controller.control_signals[0]
    control_projection = signal.efferents[0]
    monitor_projection = model.controller.input_ports[0].path_afferents[0]
    assert lowering.bindings.modulation_by_id(0) is control_projection
    assert lowering.bindings.port_by_id(modulation.source_port_id) is (
        monitor_projection.sender
    )
    assert lowering.bindings.port_by_id(modulation.controller_input_port_id) is (
        monitor_projection.receiver
    )
    assert lowering.bindings.port_by_id(modulation.control_signal_port_id) is signal
    assert lowering.bindings.port_by_id(modulation.target_parameter_port_id) is (
        control_projection.receiver
    )

    assert len(graph.absorbed_projections) == 2
    monitor_spec = graph.absorbed_projections[modulation.monitor_projection_id]
    control_spec = graph.absorbed_projections[modulation.control_projection_id]
    assert type(monitor_spec) is BatchedAbsorbedProjectionSpec
    assert monitor_spec.kind == "MappingProjection"
    assert monitor_spec.initial_value == ()
    assert monitor_spec.sender_component_id == modulation.source_component_id
    assert monitor_spec.sender_port_id == modulation.source_port_id
    assert monitor_spec.receiver_component_id == modulation.controller_component_id
    assert monitor_spec.receiver_port_id == modulation.controller_input_port_id
    assert type(control_spec) is BatchedAbsorbedProjectionSpec
    assert control_spec.kind == "ControlProjection"
    assert control_spec.initial_value == (1.0,)
    assert control_spec.sender_component_id == modulation.controller_component_id
    assert control_spec.sender_port_id == modulation.control_signal_port_id
    assert control_spec.receiver_component_id == modulation.target_component_id
    assert control_spec.receiver_port_id == modulation.target_parameter_port_id
    assert (
        lowering.bindings.absorbed_projection_by_id(modulation.monitor_projection_id)
        is monitor_projection
    )
    assert (
        lowering.bindings.absorbed_projection_by_id(modulation.control_projection_id)
        is control_projection
    )

    expected_controller_params = ("slope", "intercept", "scale", "offset")
    assert all(
        type(binding) is BatchedParameterBindingSpec
        for binding in modulation.controller_param_bindings
    )
    assert (
        tuple(binding.argument for binding in modulation.controller_param_bindings)
        == expected_controller_params
    )
    controller_params = {
        binding.argument: binding.parameter
        for binding in modulation.controller_param_bindings
    }
    assert control_node.params == controller_params
    assert control_node.attrs["spec_key"] == modulation.controller_function_spec_key
    controller_param_specs = {param.parameter_id: param for param in lowering.params}
    assert all(
        controller_param_specs[binding.parameter_id].name == binding.parameter
        for binding in modulation.controller_param_bindings
    )
    assert all(
        not controller_param_specs[binding.parameter_id].runtime_mutable
        for binding in modulation.controller_param_bindings
    )
    assert all(
        controller_param_specs[binding.parameter_id].owner_component_id == controller_id
        for binding in modulation.controller_param_bindings
    )
    assert {
        binding.argument: controller_param_specs[binding.parameter_id].default
        for binding in modulation.controller_param_bindings
    } == {
        "slope": 1.0,
        "intercept": case.controller_intercept,
        "scale": 1.0,
        "offset": 0.0,
    }
    cue_param_names = set(graph.node(model.cue.name).params.values())
    assert cue_param_names
    cue_param_specs = {
        param.name: param for param in lowering.params if param.name in cue_param_names
    }
    assert set(cue_param_specs) == cue_param_names
    assert all(not param.runtime_mutable for param in cue_param_specs.values())

    assert all(type(port) is BatchedPortSpec for port in graph.ports)
    assert tuple(port.port_id for port in graph.ports) == tuple(range(len(graph.ports)))

    assert len(graph.effective_parameters) == 1
    effective_parameter = graph.effective_parameters[0]
    assert type(effective_parameter) is BatchedEffectiveParameterSpec
    assert effective_parameter.effective_parameter_id == (
        modulation.effective_parameter_id
    )
    assert effective_parameter.target == modulation.target
    assert effective_parameter.target_component_id == modulation.target_component_id
    assert effective_parameter.target_parameter == modulation.target_parameter
    assert effective_parameter.target_parameter_port_id == (
        modulation.target_parameter_port_id
    )
    assert effective_parameter.base_value == (9.0,)
    assert effective_parameter.initial_modulation_value == (1.0,)
    assert effective_parameter.storage == "lane_persistent"
    assert effective_parameter.reset == "Never"
    assert effective_parameter.update_event == "after_controller_execution"
    assert effective_parameter.sample_event == "at_target_parameter_update"

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
    assert finished.predicate_kind == ("execution_count_at_least_effective_parameter")
    assert finished.attrs == {
        "effective_parameter_id": modulation.effective_parameter_id,
        "target_parameter_port_id": modulation.target_parameter_port_id,
        "rounding": "ceil",
        "minimum": 1,
        "maximum": 2**24,
    }
    assert schedule[follower_id].finished_value_ids == (finished.value_id,)
    assert (
        schedule[cue_id].consideration_set_id
        < schedule[controller_id].consideration_set_id
        < schedule[producer_id].consideration_set_id
        < schedule[follower_id].consideration_set_id
    )

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

    semantic_ir = BatchedCompositionIR(
        model_kind=lowering.model_kind,
        node_names=tuple(node.name for node in graph.nodes),
        params=lowering.params,
        output_names=tuple(output.name for output in graph.outputs),
        max_steps=16,
        graph=graph,
    )
    kernel_ir = lower_to_kernel_ir(semantic_ir)
    assert kernel_ir.executable
    assert kernel_ir.ports == graph.ports
    assert kernel_ir.absorbed_projections == graph.absorbed_projections
    assert kernel_ir.effective_parameters == graph.effective_parameters
    assert kernel_ir.modulations == graph.modulations
    assert kernel_ir.finished_values == graph.finished_values
    source = triton_graph_kernel_source(kernel_ir)
    compile(source, "<controlled-finished-kernel>", "exec")

    report = BatchedCompositionCompiler.diagnose(
        model.composition,
        backend="triton_cpu",
        outputs=(model.output,),
        max_steps=16,
    )
    assert _scheduler_history(model.composition) == history_before
    assert report.model_supported
    assert report.codegen_ready is True
    assert report.metadata["fusion_kind"] == STATEFUL_GRAPH_FUSION
    assert report.metadata["schedule_kind"] == "dynamic_lane_local"
    assert not report.model_diagnostics
    assert not report.codegen_diagnostics
    assert _scheduler_history(model.composition) == history_before


@pytest.mark.parametrize(
    "case",
    CONTROLLED_FINISHED_CASES,
    ids=lambda case: case.name,
)
def test_batched_controlled_finished_matches_fresh_python(case, batched_backend):
    python_model = case.build()
    python_values, python_trace = _run_python(python_model)
    assert python_trace == _expected_role_trace(case.expected_counts)

    compiled_model = case.build()
    plan = BatchedCompositionCompiler.compile(
        compiled_model.composition,
        backend=batched_backend,
        outputs=(compiled_model.output,),
        max_steps=16,
    )
    result = plan.run(
        inputs=compiled_model.inputs,
        parameter_sets=({},),
        num_estimates=1,
        seed=0,
    )

    assert result.values.shape == (1, 1, len(_TASK_INPUTS), 1, 1)
    np.testing.assert_allclose(
        result.values[0, 0, :, 0, :],
        python_values,
        rtol=1e-5,
        atol=1e-6,
    )
    assert result.metadata["truncation"][compiled_model.producer.name] == 0.0


def test_identity_controller_freezes_shorter_subject_lane(batched_backend):
    task_inputs = np.asarray([[1.0, -1.0], [1.0, -1.0]], dtype=float)
    cue_inputs = np.asarray([[1.0], [3.0]], dtype=float)

    python_model = RESET_INTERCEPT_ZERO.build(
        controller_function=pnl.Identity(),
    )
    python_model.inputs[python_model.task] = task_inputs
    python_model.inputs[python_model.cue] = cue_inputs
    python_values, python_trace = _run_python(python_model)
    assert python_trace == _expected_role_trace((1, 3))

    compiled_model = RESET_INTERCEPT_ZERO.build(
        controller_function=pnl.Identity(),
    )
    compiled_model.inputs[compiled_model.task] = task_inputs
    compiled_model.inputs[compiled_model.cue] = cue_inputs
    plan = BatchedCompositionCompiler.compile(
        compiled_model.composition,
        backend=batched_backend,
        outputs=(compiled_model.output,),
        max_steps=16,
    )
    result = plan.run(
        inputs=compiled_model.inputs,
        parameter_sets=({},),
        num_estimates=1,
        subject_slices=(slice(0, 1), slice(1, 2)),
        seed=0,
    )

    assert result.values.shape == (1, 2, 1, 1, 1)
    np.testing.assert_allclose(
        result.values[0, :, 0, 0, :],
        python_values,
        rtol=1e-5,
        atol=1e-6,
    )
    assert result.metadata["truncation"][compiled_model.producer.name] == 0.0


def test_dynamic_control_rejects_unequal_subject_trial_counts(batched_backend):
    model = RESET_INTERCEPT_ZERO.build(controller_function=pnl.Identity())
    model.inputs[model.task] = np.repeat([[1.0, -1.0]], 3, axis=0)
    model.inputs[model.cue] = np.asarray([[3.0], [1.0], [1.0]], dtype=float)
    plan = BatchedCompositionCompiler.compile(
        model.composition,
        backend=batched_backend,
        outputs=(model.output,),
        max_steps=2,
    )

    with pytest.raises(ValueError, match="same number of trials"):
        plan.run(
            inputs=model.inputs,
            parameter_sets=({},),
            num_estimates=1,
            subject_slices=(slice(0, 1), slice(1, 3)),
            seed=0,
        )


@pytest.mark.parametrize(
    ("task_trials", "cue_trials"),
    ((2, 3), (3, 2)),
    ids=("longer-count-source", "longer-task-input"),
)
def test_dynamic_control_rejects_mismatched_input_trial_counts(
    batched_backend,
    task_trials,
    cue_trials,
):
    model = RESET_INTERCEPT_ZERO.build(controller_function=pnl.Identity())
    model.inputs[model.task] = np.repeat(
        [[1.0, -1.0]],
        task_trials,
        axis=0,
    )
    model.inputs[model.cue] = np.arange(1, cue_trials + 1, dtype=float)[:, None]
    plan = BatchedCompositionCompiler.compile(
        model.composition,
        backend=batched_backend,
        outputs=(model.output,),
        max_steps=3,
    )

    with pytest.raises(ValueError, match="same trial counts"):
        plan.run(
            inputs=model.inputs,
            parameter_sets=({},),
            num_estimates=1,
            seed=0,
        )


def test_dynamic_control_rejects_fractional_typed_count_source(batched_backend):
    model = RESET_INTERCEPT_ONE.build()
    model.inputs[model.task] = np.asarray([[1.0, -1.0]], dtype=float)
    model.inputs[model.cue] = np.asarray([[1.0 + 2.0**-25]], dtype=float)
    plan = BatchedCompositionCompiler.compile(
        model.composition,
        backend=batched_backend,
        outputs=(model.output,),
        max_steps=4,
    )

    with pytest.raises(ValueError, match="nonnegative integer values"):
        plan.run(
            inputs=model.inputs,
            parameter_sets=({},),
            num_estimates=1,
            seed=0,
        )


def test_dynamic_count_cap_reports_only_lane_strictly_above_cap(
    batched_backend,
):
    task_inputs = np.repeat([[1.0, -1.0]], 3, axis=0)
    cue_inputs = np.asarray([[1.0], [2.0], [3.0]], dtype=float)

    python_model = RESET_INTERCEPT_ZERO.build(
        controller_function=pnl.Identity(),
    )
    python_model.inputs[python_model.task] = task_inputs
    python_model.inputs[python_model.cue] = cue_inputs
    python_values, _ = _run_python(python_model)

    compiled_model = RESET_INTERCEPT_ZERO.build(
        controller_function=pnl.Identity(),
    )
    compiled_model.inputs[compiled_model.task] = task_inputs
    compiled_model.inputs[compiled_model.cue] = cue_inputs
    plan = BatchedCompositionCompiler.compile(
        compiled_model.composition,
        backend=batched_backend,
        outputs=(compiled_model.output,),
        max_steps=2,
    )
    with pytest.warns(UserWarning, match="truncated bounded loops"):
        result = plan.run(
            inputs=compiled_model.inputs,
            parameter_sets=({},),
            num_estimates=1,
            seed=0,
        )

    np.testing.assert_allclose(
        result.values[0, 0, :2, 0, :],
        python_values[:2],
        rtol=1e-5,
        atol=1e-6,
    )
    exact_inputs = {
        component: values[1:2]
        for component, values in compiled_model.inputs.items()
    }
    exact_result = plan.run(
        inputs=exact_inputs,
        parameter_sets=({},),
        num_estimates=1,
        seed=0,
    )
    np.testing.assert_allclose(
        exact_result.values[0, 0, 0, 0, :],
        python_values[1],
        rtol=1e-5,
        atol=1e-6,
    )
    assert exact_result.metadata["truncation"][compiled_model.producer.name] == 0.0
    assert result.metadata["truncation"][compiled_model.producer.name] == (
        pytest.approx(1.0 / 3.0)
    )
    with pytest.raises(BatchedTruncationError, match="max_steps=2"):
        plan.run(
            inputs=compiled_model.inputs,
            parameter_sets=({},),
            num_estimates=1,
            seed=0,
            strict_truncation=True,
        )


def test_absorbed_controller_missing_triton_template_is_reported(monkeypatch):
    batched_specs.ensure_builtin_specs()
    spec = batched_specs._FUNCTION_SPECS[pnl.Linear]
    monkeypatch.setitem(
        batched_specs._SPECS_BY_KEY,
        spec.key,
        replace(spec, triton_template=None),
    )
    model = RESET_INTERCEPT_ZERO.build()

    report = BatchedCompositionCompiler.diagnose(
        model.composition,
        backend="triton",
        outputs=(model.output,),
        max_steps=16,
    )

    assert report.model_supported
    assert report.codegen_ready is False
    assert any(
        diagnostic.component == model.controller.name
        and diagnostic.code == BatchedDiagnosticCode.CODEGEN_OP_MISSING
        for diagnostic in report.codegen_diagnostics
    )


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"modulation_id": -1}, "identities"),
        ({"controller_component_id": True}, "identities"),
        ({"controller": ""}, "labels"),
        ({"mode": "PREFIX_OVERRIDE"}, "scalar float32 OVERRIDE"),
        ({"width": 2}, "scalar float32 OVERRIDE"),
        ({"dtype": "float64"}, "scalar float32 OVERRIDE"),
        ({"absorbed_identity_chain": False}, "identity projection chain"),
        (
            {"controller_param_bindings": ()},
            "implementation and parameter bindings",
        ),
        ({"controller_function_spec_key": ""}, "implementation and parameter bindings"),
    ],
)
def test_modulation_schema_rejects_malformed_declarations(changes, message):
    model = RESET_INTERCEPT_ONE.build()
    modulation = lower_composition(
        model.composition,
        outputs=(model.output,),
    ).graph.modulations[0]

    with pytest.raises(ValueError, match=message):
        replace(modulation, **changes)


def _lower_replaced_controlled_graph(*, graph_changes=None, modulation_changes=None):
    model = RESET_INTERCEPT_ONE.build()
    lowering = lower_composition(model.composition, outputs=(model.output,))
    graph = lowering.graph
    if modulation_changes is not None:
        graph = replace(
            graph,
            modulations=(replace(graph.modulations[0], **modulation_changes),),
        )
    if graph_changes is not None:
        graph = replace(graph, **graph_changes)
    return _lower_graph_to_kernel_ir(lowering, graph)


def _lower_graph_to_kernel_ir(lowering, graph):
    semantic_ir = BatchedCompositionIR(
        model_kind=lowering.model_kind,
        node_names=tuple(node.name for node in graph.nodes),
        params=lowering.params,
        output_names=tuple(output.name for output in graph.outputs),
        max_steps=16,
        graph=graph,
    )
    return lower_to_kernel_ir(semantic_ir)


@pytest.mark.parametrize(
    ("field", "forged_value"),
    [
        ("base_value", (4.0,)),
        ("initial_modulation_value", (7.0,)),
    ],
)
def test_kernel_ir_rejects_forged_effective_parameter_values(
    field,
    forged_value,
):
    model = RESET_INTERCEPT_ONE.build()
    lowering = lower_composition(model.composition, outputs=(model.output,))
    graph = lowering.graph
    effective_parameter = replace(
        graph.effective_parameters[0],
        **{field: forged_value},
    )
    graph = replace(
        graph,
        effective_parameters=(effective_parameter,),
    )

    with pytest.raises(ValueError, match="target or initial values"):
        _lower_graph_to_kernel_ir(lowering, graph)


def test_kernel_ir_rejects_forged_modulation_component_identity():
    model = RESET_INTERCEPT_ONE.build()
    graph = lower_composition(
        model.composition,
        outputs=(model.output,),
    ).graph
    modulation = graph.modulations[0]

    with pytest.raises(ValueError, match="component identity|controller-component"):
        _lower_replaced_controlled_graph(
            modulation_changes={
                "controller_component_id": modulation.source_component_id,
            },
        )


def test_kernel_ir_rejects_duplicate_modulation_target():
    model = RESET_INTERCEPT_ONE.build()
    lowering = lower_composition(model.composition, outputs=(model.output,))
    graph = lowering.graph
    duplicate = replace(
        graph.modulations[0],
        modulation_id=1,
        effective_parameter_id=1,
    )
    duplicate_effective_parameter = replace(
        graph.effective_parameters[0],
        effective_parameter_id=1,
    )
    duplicate_finished = replace(
        graph.finished_values[0],
        value_id=1,
        attrs={
            **graph.finished_values[0].attrs,
            "effective_parameter_id": 1,
        },
    )
    graph = replace(
        graph,
        effective_parameters=(
            *graph.effective_parameters,
            duplicate_effective_parameter,
        ),
        modulations=(*graph.modulations, duplicate),
        finished_values=(*graph.finished_values, duplicate_finished),
    )
    semantic_ir = BatchedCompositionIR(
        model_kind=lowering.model_kind,
        node_names=tuple(node.name for node in graph.nodes),
        params=lowering.params,
        output_names=tuple(output.name for output in graph.outputs),
        max_steps=16,
        graph=graph,
    )

    with pytest.raises(ValueError):
        lower_to_kernel_ir(semantic_ir)


def test_kernel_ir_rejects_erased_dynamic_finished_binding():
    model = RESET_INTERCEPT_ONE.build()
    graph = lower_composition(
        model.composition,
        outputs=(model.output,),
    ).graph
    finished = replace(graph.finished_values[0], predicate_kind="dynamic", attrs={})

    with pytest.raises(
        ValueError,
        match="exact GraphIR node|exact bijection",
    ):
        _lower_replaced_controlled_graph(
            graph_changes={"finished_values": (finished,)},
        )


def test_executable_graph_materializes_declared_modulation_effect():
    kernel = _lower_replaced_controlled_graph()

    assert kernel.executable
    assert sum(op.kind == "InitializeEffectiveParameter" for op in kernel.ops) == 1
    trial_body = kernel.ops[-1].attrs["body"]
    dynamic_region = next(
        op
        for op in trial_body
        if op.kind == "ForPasses"
        and op.attrs.get("trace_kind") == "lane_local_dynamic"
    )
    assert all(op.kind != "ApplyModulation" for op in trial_body)
    program = dynamic_region.attrs["program"]
    assert sum(
        effect.kind == "ApplyModulation"
        for consideration_set in program.consideration_sets
        for member in consideration_set.members
        for effect in member.effects
    ) == 1


def test_kernel_ir_rejects_complete_control_effect_ledger_erasure():
    model = RESET_INTERCEPT_ONE.build()
    lowering = lower_composition(model.composition, outputs=(model.output,))
    graph = lowering.graph
    follower_id = _component_id(lowering, model.follower)
    scheduler = tuple(
        replace(condition, finished_value_ids=())
        if condition.component_id == follower_id
        else condition
        for condition in graph.scheduler
    )
    forged = replace(
        graph,
        absorbed_projections=(),
        effective_parameters=(),
        modulations=(),
        finished_values=(),
        scheduler=scheduler,
        executable=True,
        metadata={
            **graph.metadata,
            "scheduler_executable": True,
            "scheduler_requires_pass_region": False,
        },
    )

    with pytest.raises(ValueError):
        _lower_graph_to_kernel_ir(lowering, forged)


def test_kernel_ir_rejects_erased_registered_controller_transform():
    model = RESET_INTERCEPT_ONE.build()
    lowering = lower_composition(model.composition, outputs=(model.output,))
    graph = lowering.graph
    modulation = graph.modulations[0]
    controller_nodes = tuple(
        replace(
            node,
            params={},
            attrs={
                **node.attrs,
                "spec_key": "",
                "control_function": "identity",
            },
        )
        if node.component_id == modulation.controller_component_id
        else node
        for node in graph.nodes
    )
    forged = replace(
        graph,
        nodes=controller_nodes,
        modulations=(
            replace(
                modulation,
                controller_function_spec_key="",
                controller_param_bindings=(),
            ),
        ),
    )

    with pytest.raises(ValueError):
        _lower_graph_to_kernel_ir(lowering, forged)


def test_override_mode_requires_exact_canonical_value():
    model = RESET_INTERCEPT_ZERO.build()
    model.controller.control_signals[0].modulation = "PREFIX_OVERRIDE"

    lowering = lower_composition(model.composition, outputs=(model.output,))

    # Without an exact control edge, the stepwise LCA has no complete finished
    # predicate and the semantic graph must fail closed rather than retain a
    # partially typed modulation declaration.
    assert lowering.graph is None
    diagnostics = [
        diagnostic
        for diagnostic in lowering.rejected_nodes
        if diagnostic.component == model.controller.name
    ]
    assert len(diagnostics) == 1
    assert diagnostics[0].reason == "unsupported control modulation for batched v2"
    assert diagnostics[0].detail == "PREFIX_OVERRIDE"


@pytest.mark.parametrize(
    "endpoint, forged_endpoint",
    (
        ("source_port_id", "controller_input_port_id"),
        ("controller_input_port_id", "control_signal_port_id"),
        ("control_signal_port_id", "target_parameter_port_id"),
        ("target_parameter_port_id", "source_port_id"),
    ),
)
def test_kernel_ir_rejects_each_forged_modulation_endpoint_port_role(
    endpoint,
    forged_endpoint,
):
    model = RESET_INTERCEPT_ONE.build()
    lowering = lower_composition(model.composition, outputs=(model.output,))
    graph = lowering.graph
    modulation = graph.modulations[0]
    forged_port_id = getattr(modulation, forged_endpoint)
    assert forged_port_id != getattr(modulation, endpoint)
    forged_modulation = replace(modulation, **{endpoint: forged_port_id})

    finished_values = graph.finished_values
    effective_parameters = graph.effective_parameters
    if endpoint == "target_parameter_port_id":
        finished = graph.finished_values[0]
        finished_values = (
            replace(
                finished,
                attrs={
                    **finished.attrs,
                    "target_parameter_port_id": forged_port_id,
                },
            ),
        )
        effective_parameters = (
            replace(
                graph.effective_parameters[0],
                target_parameter_port_id=forged_port_id,
            ),
        )
    forged_graph = replace(
        graph,
        effective_parameters=effective_parameters,
        modulations=(forged_modulation,),
        finished_values=finished_values,
    )

    with pytest.raises(ValueError):
        _lower_graph_to_kernel_ir(lowering, forged_graph)


def test_kernel_ir_rejects_forged_modulation_target_parameter_label():
    model = RESET_INTERCEPT_ONE.build()
    lowering = lower_composition(model.composition, outputs=(model.output,))
    graph = lowering.graph
    forged = replace(
        graph.modulations[0],
        target_parameter="not_the_termination_threshold",
    )
    forged_effective_parameter = replace(
        graph.effective_parameters[0],
        target_parameter=forged.target_parameter,
    )

    with pytest.raises(ValueError):
        _lower_graph_to_kernel_ir(
            lowering,
            replace(
                graph,
                effective_parameters=(forged_effective_parameter,),
                modulations=(forged,),
            ),
        )


def test_kernel_ir_rejects_forged_modulation_source_role():
    model = RESET_INTERCEPT_ONE.build()
    lowering = lower_composition(model.composition, outputs=(model.output,))
    graph = lowering.graph
    forged = replace(
        graph.modulations[0],
        source=model.follower.name,
        source_component_id=_component_id(lowering, model.follower),
        source_port_id=graph.outputs[0].port_id,
    )

    with pytest.raises(ValueError):
        _lower_graph_to_kernel_ir(
            lowering,
            replace(graph, modulations=(forged,)),
        )


def test_kernel_ir_rejects_coherent_monitor_route_to_late_source():
    model = RESET_INTERCEPT_ONE.build()
    lowering = lower_composition(model.composition, outputs=(model.output,))
    graph = lowering.graph
    modulation = graph.modulations[0]
    follower_id = _component_id(lowering, model.follower)
    follower_port_id = graph.node(model.follower.name).output_port_ids[0]
    monitor = graph.absorbed_projections[modulation.monitor_projection_id]
    forged_monitor = replace(
        monitor,
        sender=model.follower.name,
        sender_component_id=follower_id,
        sender_port=model.follower.output_port.name,
        sender_port_id=follower_port_id,
    )
    forged_modulation = replace(
        modulation,
        source=model.follower.name,
        source_component_id=follower_id,
        source_port=model.follower.output_port.name,
        source_port_id=follower_port_id,
    )
    absorbed = tuple(
        forged_monitor if item.projection_id == monitor.projection_id else item
        for item in graph.absorbed_projections
    )

    with pytest.raises(ValueError):
        _lower_graph_to_kernel_ir(
            lowering,
            replace(
                graph,
                absorbed_projections=absorbed,
                modulations=(forged_modulation,),
            ),
        )


def test_kernel_ir_rejects_forged_modulation_target_role():
    model = RESET_INTERCEPT_ONE.build()
    lowering = lower_composition(model.composition, outputs=(model.output,))
    graph = lowering.graph
    modulation = graph.modulations[0]
    cue_slope = model.cue.parameter_ports["slope"]
    cue_slope_port_id = next(
        port_id
        for port_id, port in lowering.bindings.ports_by_id.items()
        if port is cue_slope
    )
    cue_id = _component_id(lowering, model.cue)
    forged_modulation = replace(
        modulation,
        target=model.cue.name,
        target_component_id=cue_id,
        target_parameter="slope",
        target_parameter_port_id=cue_slope_port_id,
    )
    finished = graph.finished_values[0]
    forged_finished = replace(
        finished,
        node=model.cue.name,
        component_id=cue_id,
        attrs={
            **finished.attrs,
            "target_parameter_port_id": cue_slope_port_id,
        },
    )
    forged_effective_parameter = replace(
        graph.effective_parameters[0],
        target=model.cue.name,
        target_component_id=cue_id,
        target_parameter="slope",
        target_parameter_port_id=cue_slope_port_id,
    )
    forged_graph = replace(
        graph,
        effective_parameters=(forged_effective_parameter,),
        modulations=(forged_modulation,),
        finished_values=(forged_finished,),
    )

    with pytest.raises(ValueError):
        _lower_graph_to_kernel_ir(lowering, forged_graph)


def test_kernel_ir_rejects_coherent_target_parameter_port_relabeling():
    model = RESET_INTERCEPT_ONE.build()
    lowering = lower_composition(model.composition, outputs=(model.output,))
    graph = lowering.graph
    modulation = graph.modulations[0]
    target_id = modulation.target_component_id
    target_ports = {
        port.name: port
        for port in graph.ports
        if port.owner_component_id == target_id and port.kind == "ParameterPort"
    }
    bias_port = target_ports["bias"]
    termination_port = target_ports[pnl.TERMINATION_THRESHOLD]
    swapped_ports = tuple(
        replace(port, name=pnl.TERMINATION_THRESHOLD)
        if port.port_id == bias_port.port_id
        else replace(port, name="bias")
        if port.port_id == termination_port.port_id
        else port
        for port in graph.ports
    )
    control_route = graph.absorbed_projections[modulation.control_projection_id]
    swapped_route = replace(
        control_route,
        receiver_port_id=bias_port.port_id,
    )
    absorbed = tuple(
        swapped_route if item.projection_id == control_route.projection_id else item
        for item in graph.absorbed_projections
    )
    effective = replace(
        graph.effective_parameters[0],
        target_parameter_port_id=bias_port.port_id,
    )
    finished = replace(
        graph.finished_values[0],
        attrs={
            **graph.finished_values[0].attrs,
            "target_parameter_port_id": bias_port.port_id,
        },
    )
    forged_modulation = replace(
        modulation,
        target_parameter_port_id=bias_port.port_id,
    )

    with pytest.raises(ValueError):
        _lower_graph_to_kernel_ir(
            lowering,
            replace(
                graph,
                ports=swapped_ports,
                absorbed_projections=absorbed,
                effective_parameters=(effective,),
                modulations=(forged_modulation,),
                finished_values=(finished,),
            ),
        )


def test_kernel_ir_rejects_orphan_effective_finished_value():
    model = RESET_INTERCEPT_ONE.build()
    lowering = lower_composition(model.composition, outputs=(model.output,))
    graph = lowering.graph
    assert graph.finished_values[0].predicate_kind == (
        "execution_count_at_least_effective_parameter"
    )

    with pytest.raises(ValueError):
        _lower_graph_to_kernel_ir(
            lowering,
            replace(graph, effective_parameters=(), modulations=()),
        )


def _malform_control_input_function(model):
    model.controller.input_ports[0].function.parameters.scale.set(2.0, None)


def _add_unprojected_control_input(model):
    model.controller.add_ports(
        [
            pnl.InputPort(
                reference_value=[0.0],
                name=f"{model.controller.name} extra input",
            )
        ]
    )


@pytest.mark.parametrize(
    "mutation, expected_reason",
    (
        (
            _malform_control_input_function,
            "unsupported control input semantics for batched v2",
        ),
        (
            _add_unprojected_control_input,
            "unsupported control input routing for batched v2",
        ),
    ),
    ids=("malformed-input-port", "extra-input-port"),
)
def test_malformed_control_input_never_produces_typed_modulation(
    mutation,
    expected_reason,
):
    model = RESET_INTERCEPT_ZERO.build()
    mutation(model)

    lowering = lower_composition(model.composition, outputs=(model.output,))

    assert lowering.graph is None
    assert lowering.bindings.modulations_by_id == {}
    diagnostics = [
        diagnostic
        for diagnostic in lowering.rejected_nodes
        if diagnostic.component == model.controller.name
    ]
    assert len(diagnostics) == 1
    assert diagnostics[0].reason == expected_reason


def test_controller_function_spec_is_frozen_in_kernel_registry_snapshot(monkeypatch):
    model = RESET_INTERCEPT_ONE.build()
    lowering = lower_composition(model.composition, outputs=(model.output,))
    graph = lowering.graph
    modulation = graph.modulations[0]
    registered_spec = batched_specs.lookup_spec(modulation.controller_function_spec_key)
    spec_key = "test:controlled-finished-controller-linear"
    controller_spec = replace(registered_spec, key=spec_key)
    monkeypatch.setitem(
        batched_specs._SPECS_BY_KEY,
        spec_key,
        controller_spec,
    )
    controller_node = graph.node(model.controller.name)
    graph = replace(
        graph,
        nodes=tuple(
            replace(
                node,
                attrs={**node.attrs, "spec_key": spec_key},
            )
            if node.component_id == modulation.controller_component_id
            else node
            for node in graph.nodes
        ),
        modulations=(
            replace(
                modulation,
                controller_function_spec_key=spec_key,
            ),
        ),
    )
    assert controller_node.attrs["spec_key"] != spec_key
    kernel = _lower_graph_to_kernel_ir(lowering, graph)

    assert kernel.op_specs.lookup_spec(spec_key) is controller_spec

    def replacement_linear(x, slope, intercept, scale, offset):
        return x + slope + intercept + scale + offset

    replacement_spec = replace(controller_spec, body=replacement_linear)
    monkeypatch.setitem(
        batched_specs._SPECS_BY_KEY,
        spec_key,
        replacement_spec,
    )

    assert batched_specs.lookup_spec(spec_key) is replacement_spec
    assert kernel.op_specs.lookup_spec(spec_key) is controller_spec
    replacement_kernel = _lower_graph_to_kernel_ir(lowering, graph)
    assert replacement_kernel.op_specs.lookup_spec(spec_key) is replacement_spec


def test_modulation_controller_parameter_bindings_are_deeply_immutable():
    model = RESET_INTERCEPT_ONE.build()
    graph = lower_composition(
        model.composition,
        outputs=(model.output,),
    ).graph
    modulation = graph.modulations[0]
    bindings = modulation.controller_param_bindings
    assert type(bindings) is tuple
    original_binding = bindings[0]

    with pytest.raises(TypeError):
        bindings[0] = bindings[1]

    with pytest.raises(FrozenInstanceError):
        original_binding.argument = "forged_argument"


@pytest.mark.parametrize(
    "forgery",
    ("argument_name", "parameter_name"),
)
def test_kernel_ir_rejects_forged_controller_parameter_binding_names(forgery):
    model = RESET_INTERCEPT_ONE.build()
    lowering = lower_composition(model.composition, outputs=(model.output,))
    graph = lowering.graph
    modulation = graph.modulations[0]
    bindings = list(modulation.controller_param_bindings)
    if forgery == "argument_name":
        bindings[0] = replace(bindings[0], argument="forged_argument")
    else:
        cue_parameter_name = graph.node(model.cue.name).params["slope"]
        cue_parameter = next(
            parameter
            for parameter in lowering.params
            if parameter.name == cue_parameter_name
        )
        bindings[0] = replace(
            bindings[0],
            parameter=cue_parameter.name,
            parameter_id=cue_parameter.parameter_id,
        )

    forged_bindings = tuple(bindings)
    forged_modulation = replace(
        modulation,
        controller_param_bindings=forged_bindings,
    )
    forged_params = {binding.argument: binding.parameter for binding in forged_bindings}
    forged_nodes = tuple(
        replace(node, params=forged_params)
        if node.component_id == modulation.controller_component_id
        else node
        for node in graph.nodes
    )
    forged_graph = replace(
        graph,
        nodes=forged_nodes,
        modulations=(forged_modulation,),
    )

    with pytest.raises(ValueError):
        _lower_graph_to_kernel_ir(lowering, forged_graph)
