"""Typed KernelIR lowering for lane-local controlled finished counts."""

from dataclasses import replace
import itertools
import re

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched.backend.triton.graph_emit import (
    triton_graph_kernel_source,
)
from psyneulink.core.batched.graph import lower_composition
from psyneulink.core.batched.ir import BatchedCompositionIR
from psyneulink.core.batched.kernel_ir import (
    KernelDynamicScheduleProgram,
    KernelLoopCarry,
    KernelValue,
    diag_slots,
    lower_to_kernel_ir,
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


def _dynamic_region(kernel):
    return next(
        op
        for op in _trial_body(kernel)
        if op.kind == "ForPasses"
        and op.attrs.get("trace_kind") == "lane_local_dynamic"
    )


def _dynamic_program(kernel):
    program = _dynamic_region(kernel).attrs["program"]
    assert type(program) is KernelDynamicScheduleProgram
    return program


def _program_members(program):
    return tuple(
        member
        for consideration_set in program.consideration_sets
        for member in consideration_set.members
    )


def _program_member(program, component_id):
    return next(
        member
        for member in _program_members(program)
        if member.component_id == component_id
    )


def _replace_program_member(program, component_id, replacement):
    consideration_sets = tuple(
        replace(
            item,
            members=tuple(
                replacement if member.component_id == component_id else member
                for member in item.members
            ),
        )
        for item in program.consideration_sets
    )
    return replace(program, consideration_sets=consideration_sets)


def _replace_dynamic_region(
    kernel,
    *,
    program=None,
    outputs=None,
    **attr_changes,
):
    body = list(_trial_body(kernel))
    region = _dynamic_region(kernel)
    program = region.attrs["program"] if program is None else program
    attrs = {**region.attrs, "program": program, **attr_changes}
    outputs = region.outputs if outputs is None else outputs
    body[body.index(region)] = replace(region, outputs=outputs, attrs=attrs)
    return _replace_trial_body(kernel, body)


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
    region = _dynamic_region(kernel)
    program = _dynamic_program(kernel)
    members = _program_members(program)
    by_id = {member.component_id: member for member in members}

    assert kernel.graph.executable and kernel.executable
    assert tuple(op.kind for op in kernel.ops) == (
        "InitializeState",
        "InitializeEffectiveParameter",
        "ForTrials",
    )
    assert set(region.attrs) == {
        "region", "body", "declaration_only", "trace_kind", "program", "max_passes"
    }
    assert (
        region.attrs["body"],
        region.attrs["trace_kind"],
        region.attrs["program"],
        region.attrs["max_passes"],
    ) == ((), "lane_local_dynamic", program, 16)
    set_signature = "|".join(
        f"{item.consideration_set_id}:"
        f"{','.join(str(member.component_id) for member in item.members)}"
        for item in program.consideration_sets
    )
    assert set_signature == "0:0,2|1:1|2:3|3:4"
    predicate_signature = "|".join(
        f"{component_id}:{member.predicate.kind}:"
        f"{member.predicate.pass_index}:"
        f"{member.predicate.dependency_component_ids}:"
        f"{member.predicate.finished_value_ids}"
        for component_id, member in sorted(by_id.items())
    )
    assert predicate_signature == (
        "0:AtPass:0:():()|1:AtPass:0:():()|2:AtPass:0:():()|"
        "3:Always:None:():()|4:WhenFinished:None:(3,):(0,)"
    )
    assert (
        program.trial_termination.kind,
        program.trial_termination.dependency_component_ids,
    ) == ("AllHaveRun", (0, 1, 2, 3, 4))

    slot_signature = "|".join(
        f"{slot.value.name}:{slot.value.dtype}"
        for slot in program.scheduler_state_slots
    )
    assert slot_signature == (
        "schedule:pass-index:int32|schedule:execution-count:0:int32|"
        "schedule:has-run:0:bool|schedule:execution-count:1:int32|"
        "schedule:has-run:1:bool|schedule:execution-count:2:int32|"
        "schedule:has-run:2:bool|schedule:execution-count:3:int32|"
        "schedule:has-run:3:bool|schedule:execution-count:4:int32|"
        "schedule:has-run:4:bool|schedule:finished:3:0:bool"
    )
    assert all(slot.value.width == 1 for slot in program.scheduler_state_slots)
    carry_signature = "|".join(
        f"{carry.kind}:{carry.owner_component_id}:"
        f"{carry.value_id}:{carry.value.name}"
        for carry in program.loop_carries
    )
    assert carry_signature == (
        "state:3:0:n3:state:0|state:3:1:n3:state:1|"
        "state:3:2:n3:state:2|effective_parameter:3:0:effective:0|"
        "output:0:1:n0:output:0|output:1:10:n1:output:0|"
        "output:2:16:n2:output:0|output:3:25:n3:output:0|"
        "output:4:40:n4:output:0|diagnostic:3:0:dynamic-truncated:0"
    )
    assert region.inputs == tuple(
        carry.value
        for carry in program.loop_carries
        if carry.kind in {"state", "effective_parameter"}
    )
    assert region.outputs == tuple(carry.value for carry in program.loop_carries)

    publications = tuple(
        (member.component_id, publication)
        for member in members
        for publication in member.publications
    )
    publication_signature = "|".join(sorted(
        f"{item.kind}:{item.owner_component_id}:{item.value_id}"
        for _, item in publications
    ))
    assert publication_signature == (
        "output:0:1|output:1:10|output:2:16|output:3:25|output:4:40|"
        "state:3:0|state:3:1|state:3:2"
    )
    assert all(
        owner == publication.owner_component_id
        and f":candidate:c{owner}" in publication.source.name
        for owner, publication in publications
    )
    body_signature = "|".join(
        f"{component_id}:{','.join(op.kind for op in by_id[component_id].body)}"
        for component_id in range(5)
    )
    assert body_signature == (
        "0:LoadInput,CallFunction|1:CallFunction|2:LoadInput,CallFunction|"
        "3:CallProjection,CombineSum,StepMechanism|"
        "4:CallProjection,CombineSum,CallFunction"
    )
    effect = by_id[1].effects[0]
    assert (
        effect.kind,
        effect.attrs["modulation_id"],
        effect.attrs["controller_component_id"],
        effect.attrs["effective_parameter_id"],
    ) == ("ApplyModulation", 0, 1, 0)
    step = next(op for op in by_id[3].body if op.kind == "StepMechanism")
    assert step.attrs["active_lanes"] == "parent_member_predicate"
    assert step.attrs["state_ids"] == (0, 1, 2)
    assert step.inputs[-3:] == tuple(
        carry.value for carry in program.loop_carries[:3]
    )
    assert tuple(
        (budget.component_id, budget.maximum)
        for budget in program.execution_budgets
    ) == ((0, 1), (1, 1), (2, 1), (3, 16), (4, 16))

    candidates = {
        value.name for member in members for op in member.body for value in op.outputs
    }
    carries = {carry.value.name for carry in program.loop_carries}
    stores = tuple(
        op
        for op in _trial_body(kernel)
        if op.kind in {"StoreFlag", "StoreOutput"}
    )
    assert {op.kind for op in stores} == {"StoreFlag", "StoreOutput"}
    assert all(
        value.name in carries - candidates
        for op in stores
        for value in op.inputs
    )
    assert diag_slots(kernel) == ((kernel.finished_values[0].node, "truncated"),)

    source = triton_graph_kernel_source(kernel)
    compile(source, "<dynamic-control-kernel>", "exec")
    markers = tuple(
        f"# dynamic scheduler consideration set {set_id}" for set_id in range(4)
    )
    assert tuple(source.index(marker) for marker in markers) == tuple(
        sorted(source.index(marker) for marker in markers)
    )
    assert "dynamic_round = 0" in source
    assert "dynamic_round < MAX_STEPS" in source
    assert source.index("dynamic_s0_n2_active =") < source.index(
        "n_n0_output_0_candidate"
    )
    assert source.index("n_n2_output_0_candidate") < source.index(
        "n_n0_output_0_dynamic_current_0 = tl.where"
    )
    assert re.search(
        r"dynamic_truncated_0_dynamic_current_0 = tl\.where\(mask & "
        r"\(n_schedule_finished_\d+_0_0 == 0\), 1\.0, 0\.0\)",
        source,
    )
    assert source.count(" = _pnl_triton_lca_width2_step(") == 1
    assert all(
        marker not in source
        for marker in ("lane_local_counted", "_block_passes", "_required_passes")
    )
    assert source.index("tl.store(diag") < source.index("tl.store(out")


def test_identity_controller_is_lowered_without_a_registry_key():
    kernel = _dynamic_control_kernel(controller_function=pnl.Identity())
    controller_id = kernel.modulations[0].controller_component_id
    controller_member = _program_member(_dynamic_program(kernel), controller_id)
    controller_call = controller_member.body[-1]

    assert controller_call.kind == "CallFunction"
    assert controller_call.attrs["function_type"] == "Identity"
    assert controller_call.attrs["spec_key"] == ""
    assert controller_call.attrs["params"] == {}
    source = triton_graph_kernel_source(kernel)
    assert "_pnl_triton_identity(" not in source
    assert f"candidate_c{controller_id}" in source


def test_never_reset_omits_the_trial_reset_prefix():
    kernel = _dynamic_control_kernel(reset_when=pnl.Never())

    assert kernel.resets[0].condition_type == "Never"
    assert all(op.kind != "ResetState" for op in _trial_body(kernel))
    assert _trial_body(kernel)[0].attrs["trace_kind"] == "lane_local_dynamic"


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

    with pytest.raises(ValueError, match="exactly match|must exactly match"):
        replace(kernel, ops=(kernel.ops[0], forged, kernel.ops[2]))


@pytest.mark.parametrize(
    "forgery",
    (
        "set-order",
        "predicate",
        "slot",
        "carry",
        "budget",
        "effect",
        "state",
    ),
)
def test_complete_kernel_rejects_forged_dynamic_program(forgery):
    kernel = _dynamic_control_kernel()
    program = _dynamic_program(kernel)
    region_outputs = None
    producer_id = kernel.modulations[0].target_component_id
    controller_id = kernel.modulations[0].controller_component_id

    if forgery == "set-order":
        first, second, *remainder = program.consideration_sets
        program = replace(
            program,
            consideration_sets=(
                replace(first, members=second.members),
                replace(second, members=first.members),
                *remainder,
            ),
        )
    elif forgery == "predicate":
        member = _program_member(program, producer_id)
        predicate = replace(member.predicate, kind="AtPass", pass_index=0)
        program = _replace_program_member(
            program, producer_id, replace(member, predicate=predicate)
        )
    elif forgery == "slot":
        first, *remainder = program.scheduler_state_slots
        value = replace(first.value, name=f"{first.value.name}:forged")
        program = replace(
            program,
            scheduler_state_slots=(replace(first, value=value), *remainder),
        )
    elif forgery == "carry":
        forged = KernelLoopCarry(
            kind="diagnostic",
            owner_component_id=controller_id,
            value_id=max(carry.value_id for carry in program.loop_carries) + 1,
            value=KernelValue("forged:diagnostic", 1),
        )
        program = replace(program, loop_carries=(*program.loop_carries, forged))
        region_outputs = (*_dynamic_region(kernel).outputs, forged.value)
    elif forgery == "budget":
        budget = next(
            item
            for item in program.execution_budgets
            if item.component_id == producer_id
        )
        program = replace(
            program,
            execution_budgets=tuple(
                replace(item, maximum=item.maximum - 1)
                if item is budget
                else item
                for item in program.execution_budgets
            ),
        )
    elif forgery == "effect":
        member = _program_member(program, controller_id)
        effect = member.effects[0]
        attrs = {
            **effect.attrs,
            "modulation_id": effect.attrs["modulation_id"] + 1,
        }
        program = _replace_program_member(
            program,
            controller_id,
            replace(member, effects=(replace(effect, attrs=attrs),)),
        )
    else:
        member = _program_member(program, producer_id)
        step = next(op for op in member.body if op.kind == "StepMechanism")
        forged_step = replace(
            step,
            attrs={
                **step.attrs,
                "state_ids": tuple(
                    state_id + len(step.attrs["state_ids"]) + 1
                    for state_id in step.attrs["state_ids"]
                ),
            },
        )
        body = tuple(forged_step if op is step else op for op in member.body)
        program = _replace_program_member(
            program, producer_id, replace(member, body=body)
        )

    with pytest.raises(ValueError, match="compiler-derived|exact|dominating"):
        _replace_dynamic_region(
            kernel,
            program=program,
            outputs=region_outputs,
        )


def test_complete_kernel_rejects_forged_dynamic_global_cap():
    kernel = _dynamic_control_kernel()

    with pytest.raises(ValueError, match="compiler-derived|global pass cap|exact"):
        _replace_dynamic_region(kernel, max_passes=kernel.max_steps - 1)


@pytest.mark.parametrize("store_kind", ("StoreFlag", "StoreOutput"))
def test_member_local_candidate_cannot_escape_dynamic_region(store_kind):
    kernel = _dynamic_control_kernel()
    candidate = next(
        publication.source
        for member in _program_members(_dynamic_program(kernel))
        for publication in member.publications
        if publication.source.width == 1
    )
    body = list(_trial_body(kernel))
    store_index = next(
        index for index, op in enumerate(body) if op.kind == store_kind
    )
    body[store_index] = replace(body[store_index], inputs=(candidate,))

    with pytest.raises(ValueError, match="dominating|compiler-derived|carry"):
        _replace_trial_body(kernel, body)


def test_kernel_projection_snapshot_does_not_alias_graph_authority():
    kernel = _dynamic_control_kernel()
    projection = next(
        op
        for member in _program_members(_dynamic_program(kernel))
        for op in member.body
        if op.kind == "CallProjection"
    )
    graph_projection = next(
        item
        for item in kernel.graph.projections
        if item.projection_id == projection.attrs["projection_id"]
    )

    assert not np.shares_memory(projection.attrs["matrix"], graph_projection.matrix)
    projection.attrs["matrix"][...] = 0.0
    with pytest.raises(ValueError, match="compiler-derived|exact"):
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


@pytest.mark.parametrize(
    ("node_index", "attr", "value"),
    (
        (0, "onset_step", 5),
        (1, "onset_step", 5),
        (2, "onset_step", 5),
        (3, "onset_step", 5),
        (4, "onset_step", 5),
        (2, "integrator_pre", (1.0,)),
        (4, "integrator_pre", (1.0,)),
    ),
)
def test_forged_component_execution_attr_does_not_materialize_dynamic_program(
    node_index,
    attr,
    value,
):
    kernel = _dynamic_control_kernel()
    forged_node = kernel.graph.nodes[node_index]
    graph = replace(
        kernel.graph,
        nodes=tuple(
            replace(node, attrs={**node.attrs, attr: value})
            if node is forged_node
            else node
            for node in kernel.graph.nodes
        ),
    )

    declaration = _lower_replaced_graph(kernel, graph)

    assert not declaration.executable
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
