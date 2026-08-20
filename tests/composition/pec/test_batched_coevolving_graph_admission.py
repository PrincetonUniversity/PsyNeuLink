"""Structural admission tests for the first executable CSI GraphIR subset."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    batched_node_op,
    unregister_batched_instance_op,
)
from psyneulink.core.batched.graph import (
    COEVOLVING_GRAPH_FUSION,
    _dynamic_controlled_coevolving_graph_eligible,
    lower_composition,
)
from psyneulink.core.batched.ir import BatchedCompositionIR
from psyneulink.core.batched.kernel_ir import lower_to_kernel_ir


pytestmark = [pytest.mark.batched, pytest.mark.composition]

_CSI_DIR = Path(__file__).resolve().parents[3] / "Scripts" / "Debug" / "pec_batch_compile"


@pytest.fixture
def lower_csi():
    @batched_node_op("Drift Rate Value")
    def drift_rate(x0, x1, x2, x3, x4, x5, x6):
        a = 1.0 / (1.0 + tl.exp(-((x0 - x1) + 4.0 * x4 - 4.0)))
        b = 1.0 / (1.0 + tl.exp(-((x1 - x0) + 4.0 * x4 - 4.0)))
        c = 1.0 / (1.0 + tl.exp(-((x2 - x3) + 4.0 * x5 - 4.0)))
        d = 1.0 / (1.0 + tl.exp(-((x3 - x2) + 4.0 * x5 - 4.0)))
        pos = 1.0 / (1.0 + tl.exp(-(a - b + c - d)))
        neg = 1.0 / (1.0 + tl.exp(-(-a + b - c + d)))
        return (pos - neg) * x6

    sys.path.insert(0, str(_CSI_DIR))
    from csi_model_surrogate import make_stab_flex

    def lower(**overrides):
        controller_intercept = overrides.pop(
            "controller_intercept_override",
            None,
        )
        task_onset = overrides.pop("task_onset_override", None)
        cue_scale = overrides.pop("cue_scale_override", None)
        threshold_scale = overrides.pop("threshold_source_scale", None)
        threshold_offset = overrides.pop("threshold_source_offset", None)
        threshold_integrator_mode = overrides.pop(
            "threshold_integrator_mode",
            None,
        )
        threshold_execute_until_finished = overrides.pop(
            "threshold_execute_until_finished",
            None,
        )
        ddm_execute_until_finished = overrides.pop(
            "ddm_execute_until_finished",
            None,
        )
        options = {
            "iti": 0,
            "csi_repeat": 0,
            "csi_switch": 1,
            "threshold_collapse": -0.001,
            "ddm_noise": 0.0,
            "lca_noise": 0.0,
        }
        options.update(overrides)
        composition = make_stab_flex(**options)
        cue_source = next(
            node
            for node in composition.nodes
            if node.name.startswith("Cue Stimulus Interval")
        )
        controller = next(
            node
            for node in composition.nodes
            if node.name.startswith("CSI Override")
        )
        task = next(
            node
            for node in composition.nodes
            if node.name.startswith("Task Input")
        )
        if controller_intercept is not None:
            controller.function.parameters.intercept.set(
                controller_intercept,
            )
        if task_onset is not None:
            composition.scheduler.remove_condition(task)
            composition.scheduler.add_condition(task, pnl.AtPass(task_onset))
        if cue_scale is not None:
            cue_source.function.parameters.scale.set(cue_scale)
        threshold_source = next(
            node
            for node in composition.nodes
            if node.name.startswith("Threshold Mechanism")
        )
        if threshold_scale is not None:
            threshold_source.function.parameters.scale.set(threshold_scale)
        if threshold_offset is not None:
            threshold_source.function.parameters.offset.set(threshold_offset)
        if threshold_integrator_mode is not None:
            threshold_source.parameters.integrator_mode.set(
                threshold_integrator_mode
            )
        if threshold_execute_until_finished is not None:
            threshold_source.parameters.execute_until_finished.set(
                threshold_execute_until_finished
            )
        if ddm_execute_until_finished is not None:
            ddm = next(
                node
                for node in composition.nodes
                if node.name.startswith("DDM")
            )
            ddm.parameters.execute_until_finished.set(
                ddm_execute_until_finished
            )
        return lower_composition(composition)

    try:
        yield lower
    finally:
        unregister_batched_instance_op("Drift Rate Value")


def _roles(graph):
    modulation = graph.modulations[0]
    stepper = graph.node(modulation.target)
    controller = graph.node(modulation.controller)
    source = graph.node(modulation.source)
    terminator_finished = next(
        value for value in graph.finished_values if value.predicate_kind == "dynamic"
    )
    terminator = graph.node(terminator_finished.node)
    threshold_controller = next(
        node
        for node in graph.nodes
        if node.component_type == "ControlMechanism"
        and node.component_id != controller.component_id
    )
    sets = {
        item.consideration_set_id: item for item in graph.consideration_sets
    }
    drift = graph.node(sets[3].nodes[0])
    gates = tuple(graph.node(name) for name in sets[5].nodes)
    origins = tuple(graph.node(name) for name in sets[0].nodes)
    return {
        "source": source,
        "controller": controller,
        "threshold_controller": threshold_controller,
        "stepper": stepper,
        "drift": drift,
        "terminator": terminator,
        "gates": gates,
        "origins": origins,
        "sets": sets,
    }


def test_canonical_csi_graph_is_admitted_with_exact_six_set_schedule(lower_csi):
    lowering = lower_csi()
    graph = lowering.graph

    assert graph is not None
    assert graph.executable
    assert graph.metadata["scheduler_executable"]
    assert graph.fusion_kind == COEVOLVING_GRAPH_FUSION
    assert not lowering.rejected_nodes
    assert not lowering.rejected_conditions
    assert _dynamic_controlled_coevolving_graph_eligible(graph, lowering.params)

    role = _roles(graph)
    sets = role["sets"]
    assert tuple(sorted(sets)) == tuple(range(6))
    assert sets[0].component_ids == tuple(
        node.component_id for node in role["origins"]
    )
    assert sets[1].component_ids == tuple(
        node.component_id
        for node in graph.nodes
        if node.component_id
        in {
            role["threshold_controller"].component_id,
            role["controller"].component_id,
        }
    )
    assert sets[2].component_ids == (role["stepper"].component_id,)
    assert sets[3].component_ids == (role["drift"].component_id,)
    assert sets[4].component_ids == (role["terminator"].component_id,)
    assert sets[5].component_ids == tuple(
        node.component_id for node in role["gates"]
    )
    assert all(item.region == "pass" and item.inputs_frozen for item in sets.values())

    conditions = {
        condition.component_id: condition for condition in graph.scheduler
    }
    assert all(
        conditions[node.component_id].condition_type == "AtPass"
        and conditions[node.component_id].attrs["pass_index"] == 0
        for node in role["origins"]
    )
    assert conditions[role["controller"].component_id].condition_type == "AtPass"
    assert conditions[role["stepper"].component_id].condition_type == "Always"
    for node in (
        role["threshold_controller"],
        role["drift"],
        role["terminator"],
    ):
        condition = conditions[node.component_id]
        assert condition.condition_type == "WhenFinished"
        assert condition.dependency_component_ids == (role["stepper"].component_id,)
        assert condition.finished_value_ids == (0,)
    for gate in role["gates"]:
        condition = conditions[gate.component_id]
        assert condition.condition_type == "WhenFinished"
        assert condition.dependency_component_ids == (role["terminator"].component_id,)
        assert condition.finished_value_ids == (1,)


def test_canonical_csi_graph_authenticates_control_count_and_ddm_rng(lower_csi):
    lowering = lower_csi()
    graph = lowering.graph
    role = _roles(graph)

    assert len(graph.modulations) == 1
    modulation = graph.modulations[0]
    effective = graph.effective_parameters[0]
    assert modulation.source_component_id == role["source"].component_id
    assert modulation.controller_component_id == role["controller"].component_id
    assert modulation.target_component_id == role["stepper"].component_id
    assert modulation.target_parameter == "termination_threshold"
    assert effective.target_component_id == role["stepper"].component_id
    assert effective.base_value == (1.0,)
    assert effective.initial_modulation_value == (1.0,)
    assert effective.reset == "Never"

    stepper_finished = graph.finished_values[0]
    assert stepper_finished.predicate_kind == (
        "execution_count_at_least_effective_parameter"
    )
    assert stepper_finished.attrs == {
        "effective_parameter_id": 0,
        "target_parameter_port_id": effective.target_parameter_port_id,
        "rounding": "ceil",
        "minimum": 1,
        "maximum": 2 ** 24,
    }
    assert graph.finished_values[1].predicate_kind == "dynamic"

    threshold_control = role["threshold_controller"].attrs["absorbed_control"]
    assert threshold_control == {
        "source": threshold_control["source"],
        "target": role["terminator"].name,
        "parameter": "threshold",
        "modulation": "OVERRIDE",
    }
    assert role["threshold_controller"].attrs[
        "absorbed_control_initial_value"
    ] == 1.0
    assert len(graph.rng_streams) == 1
    rng = graph.rng_streams[0]
    assert rng.name == f"{role['terminator'].name}.rng"
    assert rng.node == role["terminator"].name
    assert rng.component_id == role["terminator"].component_id
    assert rng.stream_id == 0
    assert rng.width == 1
    assert rng.step_extent == "MAX_STEPS"


def test_coevolving_admission_rejects_structural_forgeries(lower_csi):
    lowering = lower_csi()
    graph = lowering.graph
    role = _roles(graph)
    eligible = _dynamic_controlled_coevolving_graph_eligible
    assert eligible(graph, lowering.params)

    stepper_id = role["stepper"].component_id
    scheduler = tuple(
        replace(condition, consideration_set_id=3)
        if condition.component_id == stepper_id
        else condition
        for condition in graph.scheduler
    )
    assert not eligible(replace(graph, scheduler=scheduler), lowering.params)

    finished = tuple(
        replace(value, attrs={**value.attrs, "rounding": "floor"})
        if value.component_id == stepper_id
        else value
        for value in graph.finished_values
    )
    assert not eligible(replace(graph, finished_values=finished), lowering.params)

    threshold_id = role["threshold_controller"].component_id
    nodes = tuple(
        replace(
            node,
            attrs={
                **node.attrs,
                "absorbed_control": {
                    **node.attrs["absorbed_control"],
                    "parameter": "noise",
                },
            },
        )
        if node.component_id == threshold_id
        else node
        for node in graph.nodes
    )
    assert not eligible(replace(graph, nodes=nodes), lowering.params)

    nodes = tuple(
        replace(
            node,
            attrs={
                **node.attrs,
                "absorbed_control_initial_value": 0.5,
            },
        )
        if node.component_id == threshold_id
        else node
        for node in graph.nodes
    )
    assert not eligible(replace(graph, nodes=nodes), lowering.params)

    rng_streams = (replace(graph.rng_streams[0], step_extent="TRIAL"),)
    assert not eligible(replace(graph, rng_streams=rng_streams), lowering.params)

    controller_intercept = role["controller"].params["intercept"]
    parameters = tuple(
        replace(parameter, default=1.0)
        if parameter.name == controller_intercept
        else parameter
        for parameter in lowering.params
    )
    assert not eligible(graph, parameters)


def test_coevolving_kernel_lowering_reauthenticates_projection_values(lower_csi):
    lowering = lower_csi()
    graph = lowering.graph
    task_projection = next(
        projection
        for projection in graph.projections
        if np.asarray(projection.matrix).shape == (2, 2)
    )
    forged_graph = replace(
        graph,
        projections=tuple(
            replace(
                projection,
                matrix=np.eye(2, dtype=np.float32) * np.float32(2.0),
            )
            if projection.projection_id == task_projection.projection_id
            else projection
            for projection in graph.projections
        ),
    )

    kernel = lower_to_kernel_ir(
        BatchedCompositionIR(
            model_kind=lowering.model_kind,
            node_names=tuple(node.name for node in forged_graph.nodes),
            params=lowering.params,
            output_names=tuple(output.name for output in forged_graph.outputs),
            max_steps=128,
            graph=forged_graph,
        )
    )

    assert not _dynamic_controlled_coevolving_graph_eligible(
        forged_graph,
        lowering.params,
    )
    assert not kernel.executable
    assert not any(
        op.kind == "ForPasses"
        and op.attrs.get("trace_kind") == "lane_local_coevolving"
        for trial_op in kernel.ops
        for op in trial_op.attrs.get("body", ())
    )


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"csi_switch": 0}, id="zero_switch_csi"),
        pytest.param({"csi_repeat": 2}, id="nonzero_repeat_csi"),
        pytest.param({"iti": 3}, id="delayed_task_onset"),
        pytest.param(
            {"iti": 2, "csi_repeat": 3, "csi_switch": 4},
            id="combined_affine_count",
        ),
    ],
)
def test_affine_csi_counts_and_nonzero_iti_are_admitted(lower_csi, overrides):
    lowering = lower_csi(**overrides)

    assert lowering.graph is not None
    assert lowering.graph.executable
    assert lowering.graph.metadata["scheduler_executable"]
    assert not lowering.rejected_nodes
    assert not lowering.rejected_conditions
    assert _dynamic_controlled_coevolving_graph_eligible(
        lowering.graph,
        lowering.params,
    )


def test_affine_csi_graph_exposes_mutable_counts_and_exact_iti(lower_csi):
    lowering = lower_csi(iti=2, csi_repeat=3, csi_switch=4)
    graph = lowering.graph
    role = _roles(graph)
    parameters = {parameter.name: parameter for parameter in lowering.params}
    source_parameters = {
        argument: parameters[name]
        for argument, name in role["source"].params.items()
    }
    controller_parameters = {
        argument: parameters[name]
        for argument, name in role["controller"].params.items()
    }

    assert source_parameters["slope"].default == 4.0
    assert source_parameters["intercept"].default == 3.0
    assert source_parameters["slope"].runtime_mutable
    assert source_parameters["intercept"].runtime_mutable
    assert not source_parameters["scale"].runtime_mutable
    assert not source_parameters["offset"].runtime_mutable
    assert all(
        not parameter.runtime_mutable
        for parameter in controller_parameters.values()
    )
    assert controller_parameters["intercept"].default == 2.0

    task = next(
        node
        for node in role["origins"]
        if node.component_type == "TransferMechanism"
    )
    task_condition = next(
        condition
        for condition in graph.scheduler
        if condition.component_id == task.component_id
    )
    assert task.attrs["onset_step"] == 2
    assert task_condition.condition_type == "AtPass"
    assert task_condition.attrs == {
        "pass_index": 2,
        "time_scale": "ENVIRONMENT_STATE_UPDATE",
    }
    assert graph.metadata["coevolve_warmup"] == 2


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(
            {"iti": 2, "controller_intercept_override": 3},
            id="controller_task_onset_mismatch",
        ),
        pytest.param(
            {"iti": 2, "task_onset_override": 3},
            id="task_controller_onset_mismatch",
        ),
        pytest.param({"csi_switch": -1}, id="negative_switch_default"),
        pytest.param({"csi_switch": 1.5}, id="fractional_switch_default"),
        pytest.param({"cue_scale_override": 2}, id="nonidentity_cue_scale"),
    ],
)
def test_invalid_affine_count_or_iti_contract_remains_fail_closed(
    lower_csi,
    overrides,
):
    lowering = lower_csi(**overrides)

    assert lowering.graph is not None
    assert not lowering.graph.executable
    assert not lowering.graph.metadata["scheduler_executable"]
    assert lowering.rejected_nodes or lowering.rejected_conditions


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"threshold_source_scale": 2.0}, id="source_scale"),
        pytest.param({"threshold_source_offset": 0.01}, id="source_offset"),
        pytest.param(
            {"threshold_integrator_mode": False},
            id="integrator_mode_disabled",
        ),
        pytest.param(
            {"threshold_execute_until_finished": True},
            id="source_execute_until_finished",
        ),
        pytest.param(
            {"ddm_execute_until_finished": True},
            id="ddm_execute_until_finished",
        ),
    ],
)
def test_noncanonical_threshold_chain_is_not_absorbed(lower_csi, overrides):
    lowering = lower_csi(**overrides)

    assert lowering.graph is None or not lowering.graph.executable
    assert any(
        diagnostic.reason == "unsupported generic ControlMechanism for batched v2"
        and diagnostic.detail.endswith("->DDM.threshold")
        for diagnostic in lowering.rejected_nodes
    )
