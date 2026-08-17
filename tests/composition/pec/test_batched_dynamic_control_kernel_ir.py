"""Typed KernelIR lowering for lane-local controlled finished counts.

The graph capability remains intentionally non-executable in this checkpoint.
These tests pin the complete automatically lowered effect inventory and
placement that a later emitter may consume without rediscovering semantics.
"""

from dataclasses import replace
import itertools

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched.backend.triton.graph_emit import (
    triton_graph_kernel_source,
)
from psyneulink.core.batched.graph import lower_composition
from psyneulink.core.batched.ir import BatchedCompositionIR
from psyneulink.core.batched.kernel_ir import (
    KernelValue,
    diag_slots,
    lower_to_kernel_ir,
    node_input_value_name,
    validate_kernel_ir,
)


pytestmark = [pytest.mark.batched, pytest.mark.composition]


_BUILD_NUMBERS = itertools.count()


def _dynamic_control_kernel(*, controller_function=None, reset_when=None):
    stem = f"direct dynamic control KIR {next(_BUILD_NUMBERS)}"
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
        termination_threshold=9.0,
        time_step_size=0.5,
        execute_until_finished=False,
        reset_stateful_function_when=(
            pnl.AtTrialStart() if reset_when is None else reset_when
        ),
        name=f"{stem} producer",
    )
    controller = pnl.ControlMechanism(
        function=(
            pnl.Linear(slope=1.0, intercept=1.0)
            if controller_function is None
            else controller_function
        ),
        control_signals=[(pnl.TERMINATION_THRESHOLD, producer)],
        modulation=pnl.OVERRIDE,
        monitor_for_control=cue,
        name=f"{stem} controller",
    )
    follower = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Linear(slope=2.0, intercept=-0.25),
        name=f"{stem} follower",
    )

    composition = pnl.Composition()
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

    lowering = lower_composition(composition, outputs=(follower.output_port,))
    graph = lowering.graph
    assert graph is not None
    assert not graph.executable
    semantic_ir = BatchedCompositionIR(
        model_kind=lowering.model_kind,
        node_names=tuple(node.name for node in graph.nodes),
        params=lowering.params,
        output_names=tuple(output.name for output in graph.outputs),
        max_steps=16,
        graph=graph,
    )
    kernel = lower_to_kernel_ir(semantic_ir)
    assert not kernel.executable
    return kernel


def _replace_trial_body(kernel, body):
    trials = kernel.ops[-1]
    return replace(
        kernel,
        ops=(*kernel.ops[:-1], replace(trials, attrs={"body": tuple(body)})),
    )


def _trial_body(kernel):
    return kernel.ops[-1].attrs["body"]


def _trial_op_index(kernel, kind):
    matches = tuple(
        index for index, op in enumerate(_trial_body(kernel)) if op.kind == kind
    )
    assert len(matches) == 1
    return matches[0]


def _lower_replaced_graph(kernel, graph):
    return lower_to_kernel_ir(
        BatchedCompositionIR(
            model_kind=kernel.model_kind,
            node_names=tuple(node.name for node in graph.nodes),
            params=kernel.params,
            output_names=kernel.output_names,
            max_steps=kernel.max_steps,
            graph=graph,
        )
    )


def test_dynamic_control_kernel_ir_is_fully_lowered_but_not_executable():
    kernel = _dynamic_control_kernel()

    assert not kernel.graph.executable
    assert not kernel.executable
    assert tuple(op.kind for op in kernel.ops) == (
        "InitializeState",
        "InitializeEffectiveParameter",
        "ForTrials",
    )
    apply_index = _trial_op_index(kernel, "ApplyModulation")
    trial_body = _trial_body(kernel)
    assert trial_body[apply_index - 1].kind == "CallFunction"
    assert trial_body[apply_index + 1].attrs["trace_kind"] == "lane_local_counted"
    dynamic_region = trial_body[apply_index + 1]
    dynamic_step = dynamic_region.attrs["body"][-1]
    assert dynamic_region.inputs[0].name == "effective:0"
    assert dynamic_region.outputs[:-1] == dynamic_step.outputs
    assert dynamic_region.outputs[-1].name == "dynamic-truncated:0"
    assert dynamic_step.attrs["active_lanes"] == "parent_finished_predicate"
    assert dynamic_step.attrs["loop_counter"] == "parent_pass_index"
    assert dynamic_step.attrs["finished_value_id"] == 0
    assert dynamic_step.attrs["effective_parameter_id"] == 0
    assert dynamic_step.attrs["target_parameter_port_id"] == (
        kernel.modulations[0].target_parameter_port_id
    )
    assert "execution_index" not in dynamic_step.attrs
    assert trial_body[apply_index + 2].kind == "StoreFlag"
    assert trial_body[apply_index + 2].inputs == (dynamic_region.outputs[-1],)
    assert diag_slots(kernel) == ((dynamic_step.target, "truncated"),)
    with pytest.raises(ValueError, match="non-executable"):
        triton_graph_kernel_source(kernel)


def test_identity_controller_is_lowered_without_a_registry_key():
    kernel = _dynamic_control_kernel(controller_function=pnl.Identity())
    apply_index = _trial_op_index(kernel, "ApplyModulation")
    controller_call = _trial_body(kernel)[apply_index - 1]

    assert controller_call.kind == "CallFunction"
    assert controller_call.attrs["function_type"] == "Identity"
    assert controller_call.attrs["spec_key"] == ""
    assert controller_call.attrs["params"] == {}


def test_never_reset_omits_the_trial_reset_prefix():
    kernel = _dynamic_control_kernel(reset_when=pnl.Never())

    assert kernel.resets[0].condition_type == "Never"
    assert all(op.kind != "ResetState" for op in _trial_body(kernel))
    assert _trial_body(kernel)[0].kind == "LoadInput"


def test_dynamic_materialization_falls_back_atomically_outside_boundary():
    kernel = _dynamic_control_kernel()
    graph = replace(
        kernel.graph,
        metadata={
            **kernel.graph.metadata,
            "schedule_kind": "unsupported",
        },
    )
    declaration = _lower_replaced_graph(kernel, graph)

    assert tuple(op.kind for op in declaration.ops) == (
        "InitializeState",
        "ForTrials",
    )
    region = declaration.ops[-1].attrs["body"][-1]
    assert region.kind == "ForPasses"
    assert region.attrs["declaration_only"] is True
    assert all(
        op.kind not in {"InitializeEffectiveParameter", "ApplyModulation"}
        for op in declaration.ops
    )


@pytest.mark.parametrize(
    "field, forged_value",
    [
        ("target", "forged target"),
        ("base_value", (7.0,)),
        ("initial_modulation_value", (5.0,)),
        ("target_parameter_port_id", 0),
    ],
)
def test_complete_kernel_rejects_forged_effective_initializer(field, forged_value):
    kernel = _dynamic_control_kernel()
    initializer = kernel.ops[1]
    attrs = {**initializer.attrs, field: forged_value}
    forged = replace(initializer, attrs=attrs)

    with pytest.raises(ValueError, match="does not exactly match"):
        replace(kernel, ops=(kernel.ops[0], forged, kernel.ops[2]))


@pytest.mark.parametrize(
    "field",
    [
        "modulation_id",
        "controller_component_id",
        "control_signal_port_id",
        "target_component_id",
        "target_parameter_port_id",
        "effective_parameter_id",
    ],
)
def test_complete_kernel_rejects_forged_apply_identity(field):
    kernel = _dynamic_control_kernel()
    body = list(_trial_body(kernel))
    apply_index = _trial_op_index(kernel, "ApplyModulation")
    apply = body[apply_index]

    with pytest.raises(
        ValueError,
        match="held-value|declaration-ID|does not exactly match",
    ):
        body[apply_index] = replace(
            apply,
            attrs={**apply.attrs, field: apply.attrs[field] + 1},
        )
        _replace_trial_body(kernel, body)


def test_complete_kernel_rejects_apply_outside_controller_region_boundary():
    kernel = _dynamic_control_kernel()
    body = list(_trial_body(kernel))
    apply_index = _trial_op_index(kernel, "ApplyModulation")
    apply = body.pop(apply_index)
    body.insert(apply_index + 1, apply)

    with pytest.raises(ValueError, match="immediately follow"):
        _replace_trial_body(kernel, body)


@pytest.mark.parametrize(
    "field, forged_value",
    [
        ("finished_value_id", 1),
        ("target_component_id", 0),
        ("target_parameter_port_id", 0),
        ("producer_consideration_set_id", 0),
        ("max_steps", 15),
    ],
)
def test_complete_kernel_rejects_forged_lane_local_region_identity(
    field,
    forged_value,
):
    kernel = _dynamic_control_kernel()
    body = list(_trial_body(kernel))
    region_index = _trial_op_index(kernel, "ForPasses")
    region = body[region_index]
    attrs = {**region.attrs, field: forged_value}
    outputs = region.outputs
    if field == "finished_value_id":
        outputs = (*outputs[:-1], KernelValue("dynamic-truncated:1", 1))
    body[region_index] = replace(region, outputs=outputs, attrs=attrs)

    with pytest.raises(ValueError, match="does not exactly match|dominating operation"):
        _replace_trial_body(kernel, body)


@pytest.mark.parametrize(
    "field, forged_value",
    [
        ("finished_value_id", 1),
        ("effective_parameter_id", 1),
        ("target_parameter_port_id", 0),
        ("component_id", 0),
    ],
)
def test_complete_kernel_rejects_forged_dynamic_step_identity(field, forged_value):
    kernel = _dynamic_control_kernel()
    body = list(_trial_body(kernel))
    region_index = _trial_op_index(kernel, "ForPasses")
    region = body[region_index]
    region_body = list(region.attrs["body"])
    step = region_body[-1]
    region_body[-1] = replace(
        step,
        attrs={**step.attrs, field: forged_value},
    )
    body[region_index] = replace(
        region,
        attrs={**region.attrs, "body": tuple(region_body)},
    )

    with pytest.raises(
        ValueError,
        match="target|finished-value owner|effective parameter",
    ):
        _replace_trial_body(kernel, body)


def test_lane_local_step_requires_parent_loop_counter_sentinel():
    kernel = _dynamic_control_kernel()
    region = _trial_body(kernel)[_trial_op_index(kernel, "ForPasses")]
    step = region.attrs["body"][-1]

    with pytest.raises(ValueError, match="parent pass counter"):
        replace(
            step,
            attrs={**step.attrs, "loop_counter": "execution_index"},
        )


def test_lane_local_region_requires_one_final_dynamic_step():
    kernel = _dynamic_control_kernel()
    region = _trial_body(kernel)[_trial_op_index(kernel, "ForPasses")]

    with pytest.raises(ValueError, match="one-step structure"):
        replace(
            region,
            attrs={**region.attrs, "body": region.attrs["body"][:-1]},
        )


def test_partial_dynamic_effect_inventory_is_rejected():
    kernel = _dynamic_control_kernel()

    with pytest.raises(ValueError, match="dominating operation|initialize exactly once"):
        replace(kernel, ops=(kernel.ops[0], kernel.ops[-1]))


def test_dynamic_step_cannot_claim_a_precomputed_active_lane_policy():
    kernel = _dynamic_control_kernel()
    body = list(_trial_body(kernel))
    region_index = _trial_op_index(kernel, "ForPasses")
    region = body[region_index]
    region_body = list(region.attrs["body"])
    step = region_body[-1]
    attrs = {
        key: value
        for key, value in step.attrs.items()
        if key
        not in {
            "loop_counter",
            "finished_value_id",
            "effective_parameter_id",
            "target_parameter_port_id",
        }
    }
    attrs.update(active_lanes="all", execution_index=0)
    region_body[-1] = replace(step, attrs=attrs)
    body[region_index] = replace(
        region,
        attrs={**region.attrs, "body": tuple(region_body)},
    )

    with pytest.raises(ValueError, match="active-lane policy"):
        _replace_trial_body(kernel, body)


def test_dynamic_controller_must_consume_exact_declared_source_value():
    kernel = _dynamic_control_kernel()
    body = list(_trial_body(kernel))
    apply_index = _trial_op_index(kernel, "ApplyModulation")
    controller_call = body[apply_index - 1]
    body[apply_index - 1] = replace(
        controller_call,
        inputs=(KernelValue(node_input_value_name(kernel.graph, controller_call.target), 1),),
    )

    with pytest.raises(ValueError, match="dominating operation|exact typed"):
        _replace_trial_body(kernel, body)


def test_dynamic_program_cannot_erase_follower_and_output_effects():
    kernel = _dynamic_control_kernel()
    output_node = kernel.outputs[0].node
    body = tuple(
        op
        for op in _trial_body(kernel)
        if op.target != output_node and op.kind != "StoreOutput"
    )

    with pytest.raises(ValueError, match="complete compiler-derived"):
        _replace_trial_body(kernel, body)


def test_dynamic_program_cannot_duplicate_controller_execution():
    kernel = _dynamic_control_kernel()
    body = list(_trial_body(kernel))
    apply_index = _trial_op_index(kernel, "ApplyModulation")
    controller_call = body[apply_index - 1]
    body.insert(apply_index - 1, controller_call)

    with pytest.raises(ValueError, match="complete compiler-derived"):
        _replace_trial_body(kernel, body)


@pytest.mark.parametrize(
    "forge",
    [
        lambda matrix: np.zeros_like(matrix),
        lambda matrix: matrix.astype(np.float64),
        lambda matrix: matrix[None, ...],
    ],
    ids=("values", "dtype", "shape"),
)
def test_dynamic_program_cannot_forge_target_projection_matrix(forge):
    kernel = _dynamic_control_kernel()
    body = list(_trial_body(kernel))
    region_index = _trial_op_index(kernel, "ForPasses")
    region = body[region_index]
    region_body = list(region.attrs["body"])
    projection_index = next(
        index
        for index, op in enumerate(region_body)
        if op.kind == "CallProjection"
    )
    projection = region_body[projection_index]
    region_body[projection_index] = replace(
        projection,
        attrs={
            **projection.attrs,
            "matrix": forge(projection.attrs["matrix"]),
        },
    )
    body[region_index] = replace(
        region,
        attrs={**region.attrs, "body": tuple(region_body)},
    )

    with pytest.raises(ValueError, match="complete compiler-derived"):
        _replace_trial_body(kernel, body)


def test_kernel_projection_snapshot_does_not_alias_graph_authority():
    kernel = _dynamic_control_kernel()
    region = _trial_body(kernel)[_trial_op_index(kernel, "ForPasses")]
    projection = next(
        op for op in region.attrs["body"] if op.kind == "CallProjection"
    )
    graph_projection = next(
        item
        for item in kernel.graph.projections
        if item.projection_id == projection.attrs["projection_id"]
    )

    assert not np.shares_memory(projection.attrs["matrix"], graph_projection.matrix)
    projection.attrs["matrix"][...] = 0.0
    with pytest.raises(ValueError, match="complete compiler-derived"):
        validate_kernel_ir(kernel)


def test_bool_at_pass_index_does_not_materialize_dynamic_program():
    kernel = _dynamic_control_kernel()
    source_id = kernel.modulations[0].source_component_id
    scheduler = tuple(
        replace(
            condition,
            attrs={**condition.attrs, "pass_index": False},
        )
        if condition.component_id == source_id
        else condition
        for condition in kernel.graph.scheduler
    )
    graph = replace(kernel.graph, scheduler=scheduler)

    declaration = _lower_replaced_graph(kernel, graph)

    assert all(op.kind != "InitializeEffectiveParameter" for op in declaration.ops)
    assert declaration.ops[-1].attrs["body"][-1].attrs["declaration_only"] is True


def test_duplicate_trial_termination_does_not_materialize_dynamic_program():
    kernel = _dynamic_control_kernel()
    trial_termination = next(
        item
        for item in kernel.graph.termination
        if item.time_scale == "ENVIRONMENT_STATE_UPDATE"
    )
    graph = replace(
        kernel.graph,
        termination=(*kernel.graph.termination, trial_termination),
    )

    declaration = _lower_replaced_graph(kernel, graph)

    assert all(op.kind != "InitializeEffectiveParameter" for op in declaration.ops)
    assert declaration.ops[-1].attrs["body"][-1].attrs["declaration_only"] is True
