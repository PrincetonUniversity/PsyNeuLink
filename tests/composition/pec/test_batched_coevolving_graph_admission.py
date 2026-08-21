"""Focused GraphIR admission contracts for the generic CSI schedule."""

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
    _dynamic_scheduled_graph_eligible,
    lower_composition,
)
from psyneulink.core.batched.ir import BatchedCompositionIR
from psyneulink.core.batched.kernel_ir import iter_kernel_ops, lower_to_kernel_ir


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


def _count_control_nodes(graph):
    modulation = graph.modulations[0]
    return tuple(
        graph.node(name)
        for name in (modulation.source, modulation.controller, modulation.target)
    )


def test_canonical_csi_is_admitted_by_the_generic_scheduler(lower_csi):
    lowering = lower_csi()

    assert lowering.graph is not None
    assert lowering.graph.executable
    assert lowering.graph.fusion_kind == COEVOLVING_GRAPH_FUSION
    assert not lowering.rejected_nodes
    assert not lowering.rejected_conditions
    assert _dynamic_scheduled_graph_eligible(lowering.graph, lowering.params)


def test_generic_csi_folded_affine_control_rejects_authority_forgeries(
    lower_csi,
):
    lowering = lower_csi(iti=3)
    graph = lowering.graph
    folded = graph.folded_affine_controls[0]
    folded_effective = graph.effective_parameters[folded.effective_parameter_id]
    foreign_parameter = next(
        parameter
        for parameter in lowering.params
        if parameter.owner_component_id != folded.target_component_id
    )

    for forged in (
        replace(graph, folded_affine_controls=()),
        replace(
            graph,
            folded_affine_controls=(
                replace(folded, effective_parameter_id=0),
            ),
        ),
        replace(
            graph,
            folded_affine_controls=(
                replace(folded, controller_output_port_id=0),
            ),
        ),
        replace(
            graph,
            folded_affine_controls=(
                replace(
                    folded,
                    delta_parameter=foreign_parameter.name,
                    delta_parameter_id=foreign_parameter.parameter_id,
                ),
            ),
        ),
        replace(
            graph,
            folded_affine_controls=(
                replace(folded, initial_value=(0.5,)),
            ),
        ),
        replace(
            graph,
            effective_parameters=(
                graph.effective_parameters[0],
                replace(
                    folded_effective,
                    initial_modulation_value=(0.5,),
                ),
            ),
        ),
    ):
        assert not _dynamic_scheduled_graph_eligible(forged, lowering.params)

    with pytest.raises(ValueError, match="affine scheduler-update policy"):
        replace(folded, update_expression="base_plus_delta")
    with pytest.raises(ValueError, match="affine scheduler-update policy"):
        replace(folded, clock_component_id=folded.target_component_id)


def test_generic_csi_folded_parameters_require_exact_mutable_contract(lower_csi):
    lowering = lower_csi(iti=3)
    graph = lowering.graph
    folded = graph.folded_affine_controls[0]
    target = graph.node(folded.target)
    controller = graph.node(folded.controller)
    source_name = controller.attrs["absorbed_control"]["source"]
    params = {parameter.parameter_id: parameter for parameter in lowering.params}
    base = params[folded.base_parameter_id]
    delta = params[folded.delta_parameter_id]

    assert base.aliases == tuple(dict.fromkeys((
        "ddm.threshold",
        "DDM.threshold",
        f"{target.name}.threshold",
        f"{source_name}.intercept",
    )))
    assert delta.aliases == tuple(dict.fromkeys((
        "ddm.threshold_collapse",
        "DDM.threshold_collapse",
        f"{target.name}.threshold_collapse",
        f"{source_name}.offset-integrator_function",
    )))

    for parameter_id, changes, effective_base in (
        (base.parameter_id, {"default": -0.01}, (-0.01,)),
        (base.parameter_id, {"minimum": None}, None),
        (base.parameter_id, {"minimum_inclusive": False}, None),
        (base.parameter_id, {"maximum": 1.0}, None),
        (base.parameter_id, {"maximum_inclusive": False}, None),
        (base.parameter_id, {"aliases": base.aliases[1:]}, None),
        (base.parameter_id, {"owner_scope": "mechanism"}, None),
        (base.parameter_id, {"runtime_mutable": False}, None),
        (base.parameter_id, {"runtime_constraint": "forged"}, None),
        (delta.parameter_id, {"default": 0.01}, None),
        (delta.parameter_id, {"minimum": -1.0}, None),
        (delta.parameter_id, {"minimum_inclusive": False}, None),
        (delta.parameter_id, {"maximum": None}, None),
        (delta.parameter_id, {"maximum_inclusive": False}, None),
        (delta.parameter_id, {"aliases": delta.aliases[:-1]}, None),
    ):
        forged_params = tuple(
            replace(parameter, **changes)
            if parameter.parameter_id == parameter_id
            else parameter
            for parameter in lowering.params
        )
        forged_graph = graph
        if effective_base is not None:
            forged_graph = replace(
                graph,
                effective_parameters=tuple(
                    replace(parameter, base_value=effective_base)
                    if parameter.effective_parameter_id
                    == folded.effective_parameter_id
                    else parameter
                    for parameter in graph.effective_parameters
                ),
            )
        assert not _dynamic_scheduled_graph_eligible(
            forged_graph,
            forged_params,
        )


@pytest.mark.parametrize(
    "overrides",
    (
        pytest.param({"threshold": -0.01}, id="negative-threshold"),
        pytest.param({"threshold_collapse": 0.01}, id="positive-collapse"),
    ),
)
def test_invalid_live_folded_parameter_sign_is_diagnosed_fail_closed(
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
    (
        pytest.param({"threshold": 0.0}, id="zero-threshold"),
        pytest.param({"threshold_collapse": 0.0}, id="zero-collapse"),
    ),
)
def test_live_folded_parameter_zero_boundaries_are_admitted(lower_csi, overrides):
    lowering = lower_csi(**overrides)

    assert lowering.graph is not None
    assert lowering.graph.executable
    assert _dynamic_scheduled_graph_eligible(lowering.graph, lowering.params)


def test_generic_scheduler_preserves_supported_dense_projection(
    lower_csi,
):
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

    assert _dynamic_scheduled_graph_eligible(forged_graph, lowering.params)
    assert kernel.executable
    dynamic_region = next(
        op
        for op in iter_kernel_ops(kernel)
        if op.kind == "ForPasses"
        and op.attrs.get("trace_kind") == "lane_local_dynamic"
    )
    projection_call = next(
        op
        for consideration_set in dynamic_region.attrs["program"].consideration_sets
        for member in consideration_set.members
        for op in member.body
        if op.kind == "CallProjection"
        and op.attrs["projection_id"] == task_projection.projection_id
    )
    np.testing.assert_array_equal(
        projection_call.attrs["matrix"],
        np.eye(2, dtype=np.float32) * np.float32(2.0),
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
    assert _dynamic_scheduled_graph_eligible(
        lowering.graph,
        lowering.params,
    )


def test_affine_csi_graph_exposes_mutable_counts_and_exact_iti(lower_csi):
    lowering = lower_csi(iti=2, csi_repeat=3, csi_switch=4)
    graph = lowering.graph
    source, controller, _ = _count_control_nodes(graph)
    parameters = {parameter.name: parameter for parameter in lowering.params}
    source_parameters = {
        argument: parameters[name]
        for argument, name in source.params.items()
    }
    controller_parameters = {
        argument: parameters[name]
        for argument, name in controller.params.items()
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
        for node in graph.nodes
        if node.name.startswith("Task Input")
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
