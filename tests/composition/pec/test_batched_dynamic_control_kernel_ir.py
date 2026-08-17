"""Typed KernelIR lowering for lane-local controlled finished counts."""

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


def _dynamic_control_kernel(
    *,
    controller_function=None,
    reset_when=None,
    expect_executable=True,
):
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
    assert graph.executable is expect_executable
    semantic_ir = BatchedCompositionIR(
        model_kind=lowering.model_kind,
        node_names=tuple(node.name for node in graph.nodes),
        params=lowering.params,
        output_names=tuple(output.name for output in graph.outputs),
        max_steps=16,
        graph=graph,
    )
    kernel = lower_to_kernel_ir(semantic_ir)
    assert kernel.executable is expect_executable
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


def test_dynamic_control_kernel_ir_is_fully_lowered_and_executable():
    kernel = _dynamic_control_kernel()

    assert kernel.graph.executable
    assert kernel.executable
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
    source = triton_graph_kernel_source(kernel)
    compile(source, "<dynamic-control-kernel>", "exec")
    assert source.index("n_effective_0_0 = tl.full") < source.index("trial_idx = 0")
    assert source.index("while trial_idx < num_trials") < source.index(
        "# reset component"
    )
    assert "n_n1_output_0_0 = _pnl_triton_linear(" in source
    assert "n_effective_0_0 = tl.where(mask, n_n1_output_0_0" in source
    assert "tl.ceil(n_effective_0_0)" in source
    assert "16777216.0" in source
    block_passes_line = next(
        line for line in source.splitlines() if "_block_passes =" in line
    )
    assert "tl.max(tl.where(mask," in block_passes_line
    assert "MAX_STEPS" in block_passes_line
    assert "LCA_MAX_STEPS" not in block_passes_line
    assert source.count(" = _pnl_triton_lca_width2_step(") == 1
    assert "_finished = tl.where(mask & (" in source
    assert "_required_passes > MAX_STEPS" in source
    assert source.index("tl.store(diag") < source.index("tl.store(out")


def test_identity_controller_is_lowered_without_a_registry_key():
    kernel = _dynamic_control_kernel(controller_function=pnl.Identity())
    apply_index = _trial_op_index(kernel, "ApplyModulation")
    controller_call = _trial_body(kernel)[apply_index - 1]

    assert controller_call.kind == "CallFunction"
    assert controller_call.attrs["function_type"] == "Identity"
    assert controller_call.attrs["spec_key"] == ""
    assert controller_call.attrs["params"] == {}
    source = triton_graph_kernel_source(kernel)
    modulation = kernel.modulations[0]
    source_value = f"n_n{modulation.source_component_id}_output_0_0"
    controller_value = f"n_n{modulation.controller_component_id}_output_0_0"
    assert f"{controller_value} = {source_value}" in source
    assert f"n_effective_0_0 = tl.where(mask, {controller_value}" in source


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

    with pytest.raises(
        ValueError,
        match="complete typed|dominating operation|initialize exactly once",
    ):
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

    with pytest.raises(
        ValueError,
        match="complete compiler-derived|exactly one StoreOutput",
    ):
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


def test_non_boolean_frozen_inputs_do_not_materialize_dynamic_program():
    kernel = _dynamic_control_kernel()
    consideration_sets = (
        replace(kernel.graph.consideration_sets[0], inputs_frozen=1),
        *kernel.graph.consideration_sets[1:],
    )
    graph = replace(kernel.graph, consideration_sets=consideration_sets)

    declaration = _lower_replaced_graph(kernel, graph)

    assert all(op.kind != "InitializeEffectiveParameter" for op in declaration.ops)
    assert declaration.ops[-1].attrs["body"][-1].attrs["declaration_only"] is True


def test_nonexact_termination_dependency_does_not_materialize_dynamic_program():
    kernel = _dynamic_control_kernel()
    trial_termination = kernel.graph.termination[0]
    dependency_ids = (False, *trial_termination.dependency_component_ids[1:])
    termination = (
        replace(
            trial_termination,
            dependency_component_ids=dependency_ids,
        ),
        *kernel.graph.termination[1:],
    )
    graph = replace(kernel.graph, termination=termination)

    declaration = _lower_replaced_graph(kernel, graph)

    assert all(op.kind != "InitializeEffectiveParameter" for op in declaration.ops)
    assert declaration.ops[-1].attrs["body"][-1].attrs["declaration_only"] is True


@pytest.mark.parametrize(
    "termination_index, changes",
    [
        (0, {"attrs": []}),
        (1, {"dependency_component_ids": []}),
    ],
    ids=("trial-attrs-list", "run-dependencies-list"),
)
def test_untyped_empty_termination_fields_do_not_materialize_dynamic_program(
    termination_index,
    changes,
):
    kernel = _dynamic_control_kernel()
    termination = list(kernel.graph.termination)
    termination[termination_index] = replace(
        termination[termination_index],
        **changes,
    )
    graph = replace(kernel.graph, termination=tuple(termination))

    declaration = _lower_replaced_graph(kernel, graph)

    assert all(op.kind != "InitializeEffectiveParameter" for op in declaration.ops)
    assert declaration.ops[-1].attrs["body"][-1].attrs["declaration_only"] is True


@pytest.mark.parametrize(
    "changes",
    [
        {"projection_id": 0.0},
        {"sender_port_id": 1016},
        {"receiver_port_id": "float-current"},
    ],
    ids=("projection-id-float", "sender-port-wrong-owner", "receiver-port-float"),
)
def test_forged_ordinary_projection_identity_is_rejected(changes):
    kernel = _dynamic_control_kernel()
    projection = kernel.graph.projections[0]
    if changes.get("receiver_port_id") == "float-current":
        changes = {
            "receiver_port_id": float(projection.receiver_port_id),
        }
    projections = (
        replace(projection, **changes),
        *kernel.graph.projections[1:],
    )
    graph = replace(kernel.graph, projections=projections)

    with pytest.raises(ValueError, match="ordinary projection"):
        _lower_replaced_graph(kernel, graph)


def test_forged_ordinary_projection_implementation_is_rejected():
    kernel = _dynamic_control_kernel()
    projection = kernel.graph.projections[0]
    linear_key = kernel.graph.node(projection.sender).attrs["spec_key"]
    graph = replace(
        kernel.graph,
        projections=(
            replace(projection, spec_key=linear_key),
            *kernel.graph.projections[1:],
        ),
    )

    with pytest.raises(ValueError, match="ordinary projection"):
        _lower_replaced_graph(kernel, graph)


def test_sub_float32_controller_count_transform_is_not_executable():
    kernel = _dynamic_control_kernel(
        controller_function=pnl.Linear(slope=1.0, intercept=1e-8),
        expect_executable=False,
    )

    assert all(op.kind != "InitializeEffectiveParameter" for op in kernel.ops)
    assert kernel.ops[-1].attrs["body"][-1].attrs["declaration_only"] is True


def test_removed_typed_count_source_does_not_materialize_dynamic_program():
    kernel = _dynamic_control_kernel()
    target = kernel.graph.node(kernel.modulations[0].target)
    attrs = dict(target.attrs)
    attrs.pop("termination_input_node")
    graph = replace(
        kernel.graph,
        nodes=tuple(
            replace(node, attrs=attrs) if node.name == target.name else node
            for node in kernel.graph.nodes
        ),
    )

    declaration = _lower_replaced_graph(kernel, graph)

    assert not declaration.executable
    assert all(op.kind != "InitializeEffectiveParameter" for op in declaration.ops)
    assert declaration.ops[-1].attrs["body"][-1].attrs["declaration_only"] is True


@pytest.mark.parametrize(
    ("role", "mutation"),
    [
        ("follower", "wrong-spec"),
        ("follower", "missing-slope"),
        ("target", "missing-leak"),
    ],
    ids=("ordinary-spec", "ordinary-binding", "mechanism-binding"),
)
def test_forged_node_implementation_is_rejected(role, mutation):
    kernel = _dynamic_control_kernel()
    target = kernel.graph.node(kernel.modulations[0].target)
    follower = kernel.graph.node(kernel.outputs[0].node)
    node = target if role == "target" else follower
    if mutation == "wrong-spec":
        logistic_key = next(
            state.function_initializer.spec_key
            for state in kernel.states
            if state.function_initializer is not None
        )
        replacement = replace(
            node,
            attrs={**node.attrs, "spec_key": logistic_key},
        )
    else:
        missing = "leak" if mutation == "missing-leak" else "slope"
        replacement = replace(
            node,
            params={
                argument: parameter
                for argument, parameter in node.params.items()
                if argument != missing
            },
        )
    graph = replace(
        kernel.graph,
        nodes=tuple(
            replacement if candidate.name == node.name else candidate
            for candidate in kernel.graph.nodes
        ),
    )

    with pytest.raises(ValueError, match="executable node"):
        _lower_replaced_graph(kernel, graph)


def test_forged_target_input_width_is_rejected():
    kernel = _dynamic_control_kernel()
    target = kernel.graph.node(kernel.modulations[0].target)
    graph = replace(
        kernel.graph,
        nodes=tuple(
            replace(node, input_width=1) if node.name == target.name else node
            for node in kernel.graph.nodes
        ),
    )

    with pytest.raises(ValueError, match="parameter/shape signature"):
        _lower_replaced_graph(kernel, graph)


def test_extra_controller_input_does_not_materialize_dynamic_program():
    kernel = _dynamic_control_kernel()
    modulation = kernel.modulations[0]
    controller = kernel.graph.node(modulation.controller)
    controller_input = next(
        port
        for port in kernel.graph.ports
        if port.port_id == modulation.controller_input_port_id
    )
    extra_input = replace(
        controller_input,
        port_id=max(port.port_id for port in kernel.graph.ports) + 1,
        name=f"{controller_input.name} forged extra",
    )
    graph = replace(
        kernel.graph,
        nodes=tuple(
            replace(
                node,
                input_width=2,
                input_port_ids=(*node.input_port_ids, extra_input.port_id),
            )
            if node.name == controller.name
            else node
            for node in kernel.graph.nodes
        ),
        ports=(*kernel.graph.ports, extra_input),
    )

    declaration = _lower_replaced_graph(kernel, graph)

    assert not declaration.executable
    assert all(op.kind != "InitializeEffectiveParameter" for op in declaration.ops)
    assert declaration.ops[-1].attrs["body"][-1].attrs["declaration_only"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("input_width", True),
        ("input_width", 1.0),
        ("output_width", True),
        ("output_width", 1.0),
    ),
    ids=("input-bool", "input-float", "output-bool", "output-float"),
)
def test_noninteger_controller_width_does_not_materialize_dynamic_program(
    field,
    value,
):
    kernel = _dynamic_control_kernel()
    controller = kernel.graph.node(kernel.modulations[0].controller)
    graph = replace(
        kernel.graph,
        nodes=tuple(
            replace(node, **{field: value})
            if node.name == controller.name
            else node
            for node in kernel.graph.nodes
        ),
    )

    declaration = _lower_replaced_graph(kernel, graph)

    assert not declaration.executable
    assert all(op.kind != "InitializeEffectiveParameter" for op in declaration.ops)
    assert declaration.ops[-1].attrs["body"][-1].attrs["declaration_only"] is True


def test_duplicate_first_set_member_does_not_materialize_dynamic_program():
    kernel = _dynamic_control_kernel()
    first_set = kernel.graph.consideration_sets[0]
    duplicated_set = replace(
        first_set,
        nodes=(*first_set.nodes, first_set.nodes[0]),
        component_ids=(*first_set.component_ids, first_set.component_ids[0]),
    )
    graph = replace(
        kernel.graph,
        consideration_sets=(
            duplicated_set,
            *kernel.graph.consideration_sets[1:],
        ),
        execution_order=(
            *duplicated_set.nodes,
            *kernel.graph.execution_order[2:],
        ),
    )

    declaration = _lower_replaced_graph(kernel, graph)

    assert all(op.kind != "InitializeEffectiveParameter" for op in declaration.ops)
    assert declaration.ops[-1].attrs["body"][-1].attrs["declaration_only"] is True


@pytest.mark.parametrize("condition_type", ("AtPass", "WhenFinished"))
def test_non_pass_scheduler_region_does_not_materialize_dynamic_program(
    condition_type,
):
    kernel = _dynamic_control_kernel()
    scheduler = tuple(
        replace(condition, region="trial")
        if condition.condition_type == condition_type
        else condition
        for condition in kernel.graph.scheduler
    )
    graph = replace(kernel.graph, scheduler=scheduler)

    declaration = _lower_replaced_graph(kernel, graph)

    assert all(op.kind != "InitializeEffectiveParameter" for op in declaration.ops)
    assert declaration.ops[-1].attrs["body"][-1].attrs["declaration_only"] is True


def test_at_pass_dependency_does_not_materialize_dynamic_program():
    kernel = _dynamic_control_kernel()
    source_id = kernel.modulations[0].source_component_id
    target = kernel.graph.node(kernel.modulations[0].target)
    scheduler = tuple(
        replace(
            condition,
            dependencies=(target.name,),
            dependency_component_ids=(target.component_id,),
        )
        if condition.component_id == source_id
        else condition
        for condition in kernel.graph.scheduler
    )
    graph = replace(kernel.graph, scheduler=scheduler)

    declaration = _lower_replaced_graph(kernel, graph)

    assert all(op.kind != "InitializeEffectiveParameter" for op in declaration.ops)
    assert declaration.ops[-1].attrs["body"][-1].attrs["declaration_only"] is True


def test_untyped_follower_attrs_do_not_materialize_dynamic_program():
    kernel = _dynamic_control_kernel()
    scheduler = tuple(
        replace(
            condition,
            attrs=[("predicate", "is_finished")],
        )
        if condition.condition_type == "WhenFinished"
        else condition
        for condition in kernel.graph.scheduler
    )
    graph = replace(kernel.graph, scheduler=scheduler)

    declaration = _lower_replaced_graph(kernel, graph)

    assert all(op.kind != "InitializeEffectiveParameter" for op in declaration.ops)
    assert declaration.ops[-1].attrs["body"][-1].attrs["declaration_only"] is True


def test_executable_flags_require_exact_booleans():
    kernel = _dynamic_control_kernel()
    graph = replace(kernel.graph, executable=1)

    with pytest.raises(ValueError, match="exact booleans"):
        _lower_replaced_graph(kernel, graph)


def test_kernel_cannot_override_graph_capability_rejection():
    kernel = _dynamic_control_kernel()
    rejected_graph = replace(kernel.graph, executable=False)
    nonexecutable = replace(
        kernel,
        graph=rejected_graph,
        executable=False,
    )

    with pytest.raises(ValueError, match="GraphIR capability authority"):
        replace(nonexecutable, executable=True)


def test_boolean_graph_component_identity_does_not_materialize_dynamic_program():
    kernel = _dynamic_control_kernel()
    controller_id = kernel.modulations[0].controller_component_id
    nodes = tuple(
        replace(node, component_id=True)
        if node.component_id == controller_id
        else node
        for node in kernel.graph.nodes
    )
    trial_termination = kernel.graph.termination[0]
    dependency_ids = tuple(
        True if component_id == controller_id else component_id
        for component_id in trial_termination.dependency_component_ids
    )
    termination = (
        replace(
            trial_termination,
            dependency_component_ids=dependency_ids,
        ),
        *kernel.graph.termination[1:],
    )
    graph = replace(
        kernel.graph,
        nodes=nodes,
        termination=termination,
    )

    declaration = _lower_replaced_graph(kernel, graph)

    assert all(op.kind != "InitializeEffectiveParameter" for op in declaration.ops)
    assert declaration.ops[-1].attrs["body"][-1].attrs["declaration_only"] is True


@pytest.mark.parametrize(
    "changes",
    [
        {"component_id": 3.0},
        {"initial_value": (0.0,)},
        {"name": "forged retained state"},
        {"node": "missing retained-state owner"},
    ],
    ids=("component-id-float", "initializer-width", "name", "owner"),
)
def test_forged_retained_state_declaration_is_rejected(changes):
    kernel = _dynamic_control_kernel()
    states = (
        replace(kernel.graph.states[0], **changes),
        *kernel.graph.states[1:],
    )
    graph = replace(kernel.graph, states=states)

    with pytest.raises(ValueError, match="retained state"):
        _lower_replaced_graph(kernel, graph)


@pytest.mark.parametrize(
    "changes",
    [
        {"input_value": (0.0,)},
        {"params": {}},
    ],
    ids=("input-width", "parameter-bindings"),
)
def test_forged_retained_state_function_initializer_is_rejected(changes):
    kernel = _dynamic_control_kernel()
    state_index = next(
        index
        for index, state in enumerate(kernel.graph.states)
        if state.function_initializer is not None
    )
    state = kernel.graph.states[state_index]
    initializer = replace(state.function_initializer, **changes)
    states = list(kernel.graph.states)
    states[state_index] = replace(state, function_initializer=initializer)
    graph = replace(kernel.graph, states=tuple(states))

    with pytest.raises(ValueError, match="retained-state function initializer"):
        _lower_replaced_graph(kernel, graph)
