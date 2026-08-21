"""Acceptance for a generic scheduled terminator in the dynamic executor.

This deliberately is not the CSI topology.  It is the smallest graph that
requires a stateful component to publish its own dynamic ``is_finished`` value
and two readouts while consuming a lane-local RNG clock::

    Always LCA -> WhenFinished(LCA) DDM -> WhenFinished(DDM) decision gate
                                      +-> WhenFinished(DDM) RT gate

The DDM executes one integration step per eligible scheduler pass.  Its local
integration cap is therefore distinct from the fuel needed to execute the
whole schedule, including the upstream LCA and downstream readouts.
"""

from dataclasses import dataclass, replace
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import BatchedCompileError, BatchedCompositionCompiler
from psyneulink.core.batched.backend.triton.graph_emit import (
    triton_graph_kernel_source,
)
from psyneulink.core.batched.backend.triton.emit.emitter import (
    TritonGraphEmitter,
)
from psyneulink.core.batched.backend.triton.emit.lanes import RNG_STREAM_STRIDE
from psyneulink.core.batched.graph import _dynamic_scheduled_graph_eligible
from psyneulink.core.batched.kernel_ir import (
    KernelDynamicScheduleProgram,
    iter_kernel_ops,
)


pytestmark = [pytest.mark.batched, pytest.mark.composition]


_BUILD_NUMBERS = itertools.count()
_DETERMINISTIC_INPUTS = np.asarray(
    [[1.0, -1.0], [-0.5, 1.5], [1.0, -1.0]],
    dtype=float,
)
_DETERMINISTIC_RESULTS = np.asarray(
    [[1.5, 2.05], [-0.5, 2.05], [1.5, 2.05]],
    dtype=float,
)


@dataclass(frozen=True)
class _ScheduledTerminatorModel:
    composition: pnl.Composition
    inputs: dict
    outputs: tuple
    stepper: pnl.LCAMechanism
    terminator: pnl.DDM
    gates: tuple


def _build_scheduled_terminator(
    *,
    noise=0.0,
    starting_value=0.0,
    threshold=0.25,
    inputs=_DETERMINISTIC_INPUTS,
) -> _ScheduledTerminatorModel:
    build_number = next(_BUILD_NUMBERS)
    stem = f"generic scheduled terminator {build_number}"
    stepper = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(gain=1.0),
        leak=0.0,
        competition=0.0,
        self_excitation=0.0,
        noise=0.0,
        termination_measure=pnl.TimeScale.TRIAL,
        # A fixed, inspectable scheduler predicate.  The DDM remains active
        # long enough that this Always member must keep running after count 3.
        termination_threshold=3,
        time_step_size=0.5,
        execute_until_finished=False,
        reset_stateful_function_when=pnl.AtTrialStart(),
        name=f"{stem} stepper",
    )
    terminator = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=starting_value,
            rate=1.0,
            noise=noise,
            threshold=threshold,
            non_decision_time=0.2,
            time_step_size=0.1,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        execute_until_finished=False,
        reset_stateful_function_when=pnl.AtTrialStart(),
        name=f"{stem} terminator",
    )
    decision_gate = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Linear(slope=2.0, intercept=-0.5),
        name=f"{stem} decision gate",
    )
    response_time_gate = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Linear(slope=3.0, intercept=0.25),
        name=f"{stem} response-time gate",
    )

    composition = pnl.Composition(name=stem)
    composition.add_nodes(
        [stepper, terminator, decision_gate, response_time_gate]
    )
    composition.add_projection(
        sender=stepper,
        receiver=terminator,
        projection=pnl.MappingProjection(matrix=[[1.0], [-1.0]]),
    )
    composition.add_projection(
        sender=terminator.output_ports[pnl.DECISION_OUTCOME],
        receiver=decision_gate,
    )
    composition.add_projection(
        sender=terminator.output_ports[pnl.RESPONSE_TIME],
        receiver=response_time_gate,
    )
    composition.scheduler.add_condition(stepper, pnl.Always())
    composition.scheduler.add_condition(
        terminator,
        pnl.WhenFinished(stepper),
    )
    composition.scheduler.add_condition(
        decision_gate,
        pnl.WhenFinished(terminator),
    )
    composition.scheduler.add_condition(
        response_time_gate,
        pnl.WhenFinished(terminator),
    )
    return _ScheduledTerminatorModel(
        composition=composition,
        inputs={stepper: np.asarray(inputs, dtype=float)},
        outputs=(decision_gate.output_port, response_time_gate.output_port),
        stepper=stepper,
        terminator=terminator,
        gates=(decision_gate, response_time_gate),
    )


def _selected_python_results(model: _ScheduledTerminatorModel) -> np.ndarray:
    indices = []
    for output in model.outputs:
        matches = tuple(
            index
            for index, cim_input in enumerate(
                model.composition.output_CIM.input_ports
            )
            if any(
                projection.sender is output
                for projection in cim_input.path_afferents
            )
        )
        assert len(matches) == 1
        indices.append(matches[0])
    return np.asarray(
        [
            [
                float(np.asarray(trial[index]).reshape(-1)[0])
                for index in indices
            ]
            for trial in model.composition.results
        ],
        dtype=float,
    )


def _dynamic_region(kernel):
    regions = tuple(
        op
        for op in iter_kernel_ops(kernel)
        if op.kind == "ForPasses"
        and op.attrs.get("trace_kind") == "lane_local_dynamic"
    )
    assert len(regions) == 1
    return regions[0]


def _program_members(program):
    return tuple(
        member
        for consideration_set in program.consideration_sets
        for member in consideration_set.members
    )


def _replace_trial_body(kernel, body):
    trials = kernel.ops[-1]
    return replace(
        kernel,
        ops=(*kernel.ops[:-1], replace(trials, attrs={**trials.attrs, "body": body})),
    )


def _replace_dynamic_program(kernel, program):
    trials = kernel.ops[-1]
    body = list(trials.attrs["body"])
    region = _dynamic_region(kernel)
    body[body.index(region)] = replace(
        region,
        attrs={**region.attrs, "program": program},
    )
    return _replace_trial_body(kernel, tuple(body))


def _replace_program_member(program, component_id, replacement):
    return replace(
        program,
        consideration_sets=tuple(
            replace(
                item,
                members=tuple(
                    replacement
                    if member.component_id == component_id
                    else member
                    for member in item.members
                ),
            )
            for item in program.consideration_sets
        ),
    )


def test_backward_when_finished_dependency_remains_fail_closed():
    """A later producer needs another round, which is not in the typed tier."""

    stem = f"backward scheduled terminator {next(_BUILD_NUMBERS)}"
    gate = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Linear(slope=2.0, intercept=0.5),
        name=f"{stem} gate",
    )
    lca_source = pnl.TransferMechanism(input_shapes=2, name=f"{stem} LCA source")
    ddm_source = pnl.TransferMechanism(input_shapes=1, name=f"{stem} DDM source")
    middle = pnl.TransferMechanism(input_shapes=2, name=f"{stem} middle")
    terminator = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            rate=1.0,
            noise=0.0,
            threshold=0.35,
            time_step_size=0.1,
        ),
        execute_until_finished=False,
        reset_stateful_function_when=pnl.AtTrialStart(),
        name=f"{stem} terminator",
    )
    stepper = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(gain=1.0),
        leak=0.0,
        competition=0.0,
        self_excitation=0.0,
        noise=0.0,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=3,
        time_step_size=0.5,
        execute_until_finished=False,
        reset_stateful_function_when=pnl.AtTrialStart(),
        name=f"{stem} stepper",
    )
    composition = pnl.Composition(name=stem)
    composition.add_nodes(
        [gate, lca_source, ddm_source, middle, terminator, stepper]
    )
    composition.add_projection(sender=lca_source, receiver=middle)
    composition.add_projection(sender=middle, receiver=stepper)
    composition.add_projection(sender=ddm_source, receiver=terminator)
    for node in (lca_source, ddm_source, middle, stepper):
        composition.scheduler.add_condition(node, pnl.Always())
    composition.scheduler.add_condition(terminator, pnl.WhenFinished(stepper))
    composition.scheduler.add_condition(gate, pnl.WhenFinished(terminator))

    report = BatchedCompositionCompiler.diagnose(
        composition,
        backend="triton_cpu",
        outputs=(gate.output_port,),
        max_steps=4,
    )
    assert not report.model_supported
    with pytest.raises(BatchedCompileError):
        BatchedCompositionCompiler.compile(
            composition,
            backend="triton_cpu",
            outputs=(gate.output_port,),
            max_steps=4,
        )


def test_scheduled_terminator_has_typed_dynamic_program():
    model = _build_scheduled_terminator(noise=0.25)
    plan = BatchedCompositionCompiler.compile(
        model.composition,
        backend="triton_cpu",
        outputs=model.outputs,
        max_steps=4,
    )
    kernel = plan.kernel_ir
    region = _dynamic_region(kernel)
    program = region.attrs["program"]
    assert type(program) is KernelDynamicScheduleProgram
    assert region.attrs["trace_kind"] == "lane_local_dynamic"
    assert "max_passes" not in region.attrs

    nodes = {node.name: node for node in kernel.graph.nodes}
    stepper_id = nodes[model.stepper.name].component_id
    terminator_id = nodes[model.terminator.name].component_id
    terminator_member = next(
        member
        for member in _program_members(program)
        if member.component_id == terminator_id
    )
    assert terminator_member.predicate.kind == "WhenFinished"
    assert terminator_member.predicate.dependency_component_ids == (stepper_id,)

    trial_states = tuple(
        carry
        for carry in program.loop_carries
        if carry.kind == "trial_state"
        and carry.owner_component_id == terminator_id
    )
    assert len(trial_states) == 3
    assert tuple(carry.value_id for carry in trial_states) == (0, 1, 2)
    assert tuple(carry.value.name.rsplit(".", 1)[-1] for carry in trial_states) == (
        "value",
        "steps",
        "finished",
    )
    starting_parameter = next(
        parameter
        for parameter in kernel.params
        if parameter.name
        == nodes[model.terminator.name].params["starting_value"]
    )
    assert (
        trial_states[0].initial_value,
        trial_states[0].initial_parameter_id,
    ) == (None, starting_parameter.parameter_id)
    assert tuple(
        (carry.initial_value, carry.initial_parameter_id)
        for carry in trial_states[1:]
    ) == (((0.0,), None), ((0.0,), None))

    step = next(op for op in terminator_member.body if op.kind == "StepMechanism")
    assert step.attrs["trial_state_ids"] == tuple(
        carry.value_id for carry in trial_states
    )
    assert step.attrs["finished_trial_state_id"] == trial_states[-1].value_id

    finished = next(
        value
        for value in kernel.finished_values
        if value.component_id == terminator_id
    )
    finished_publications = tuple(
        publication
        for publication in terminator_member.publications
        if publication.kind == "finished"
    )
    assert len(finished_publications) == 1
    assert (
        finished_publications[0].owner_component_id,
        finished_publications[0].value_id,
    ) == (terminator_id, finished.value_id)

    output_port_ids = tuple(
        port.port_id
        for port in kernel.ports
        if port.owner_component_id == terminator_id and port.kind == "OutputPort"
    )
    output_publications = tuple(
        publication
        for publication in terminator_member.publications
        if publication.kind == "output"
    )
    assert len(output_publications) == 2
    assert tuple(publication.value_id for publication in output_publications) == (
        output_port_ids
    )

    rng_slots = tuple(
        slot
        for slot in program.scheduler_state_slots
        if slot.kind == "rng_clock"
    )
    assert len(kernel.rng_streams) == len(rng_slots) == 1
    stream = kernel.rng_streams[0]
    assert (
        rng_slots[0].owner_component_id,
        rng_slots[0].rng_stream_id,
    ) == (terminator_id, stream.stream_id)
    assert step.attrs["rng_stream_ids"] == (stream.stream_id,)

    budgets = {
        budget.component_id: budget
        for budget in program.execution_budgets
    }
    # Four DDM steps are component-local.  The whole schedule also needs the
    # three-call LCA warmup; repeatable members receive the independent fuel.
    terminator_budget = budgets[terminator_id]
    assert (
        terminator_budget.maximum,
        terminator_budget.finished_value_id,
        terminator_budget.unfinished_maximum,
        terminator_budget.post_finish,
    ) == (4, finished.value_id, 4, "stop")
    stepper_finished = next(
        value
        for value in kernel.finished_values
        if value.component_id == stepper_id
    )
    stepper_budget = budgets[stepper_id]
    assert (
        stepper_budget.maximum,
        stepper_budget.finished_value_id,
        stepper_budget.unfinished_maximum,
        stepper_budget.post_finish,
    ) == (program.schedule_fuel, stepper_finished.value_id, 4, "continue")
    assert all(
        (
            budgets[kernel.graph.node(gate.name).component_id].maximum,
            budgets[kernel.graph.node(gate.name).component_id].post_finish,
        )
        == (program.schedule_fuel, "unrestricted")
        for gate in model.gates
    )
    assert program.schedule_fuel == 7
    assert program.schedule_fuel > kernel.max_steps

    source = triton_graph_kernel_source(kernel)
    assert "lane_local_coevolving" not in source
    trial_loop_index = source.index("while trial_idx < num_trials")
    for carry in trial_states:
        safe_name = "n_" + "".join(
            character if character.isalnum() else "_"
            for character in carry.value.name
        )
        initializer = (
            f"{safe_name}_dynamic_current_0 = "
            + (
                f"param_{carry.initial_parameter_id}_value"
                if carry.initial_parameter_id is not None
                else "tl.full((BLOCK,), 0.0, tl.float32)"
            )
        )
        assert source.count(initializer) == 1
        assert trial_loop_index < source.index(initializer)

    rng_clock_var = (
        "n_"
        + "".join(
            character if character.isalnum() else "_"
            for character in rng_slots[0].value.name
        )
        + "_0"
    )
    ddm_step_calls = tuple(
        line
        for line in source.splitlines()
        if " = _pnl_triton_ddm_step(" in line
    )
    assert len(ddm_step_calls) == 1
    assert "draw = tl.randn(seed, rng_base + step)" in source
    assert rng_clock_var in ddm_step_calls[0]
    assert all(
        forbidden not in ddm_step_calls[0]
        for forbidden in (
            "dynamic_round",
            "schedule_pass_index",
            f"schedule_execution_count_{terminator_id}",
        )
    )


def test_distinct_scheduled_rng_owners_use_global_stream_inventory():
    model = _build_scheduled_terminator(noise=0.25)
    second_terminator = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0,
            rate=1.0,
            noise=0.35,
            threshold=0.3,
            non_decision_time=0.1,
            time_step_size=0.1,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        execute_until_finished=False,
        reset_stateful_function_when=pnl.AtTrialStart(),
        name=f"{model.composition.name} second terminator",
    )
    second_gate = pnl.TransferMechanism(
        input_shapes=1,
        name=f"{model.composition.name} second gate",
    )
    model.composition.add_nodes([second_terminator, second_gate])
    model.composition.add_projection(
        sender=model.stepper,
        receiver=second_terminator,
        projection=pnl.MappingProjection(matrix=[[1.0], [-1.0]]),
    )
    model.composition.add_projection(
        sender=second_terminator.output_ports[pnl.DECISION_OUTCOME],
        receiver=second_gate,
    )
    model.composition.scheduler.add_condition(
        second_terminator,
        pnl.WhenFinished(model.stepper),
    )
    model.composition.scheduler.add_condition(
        second_gate,
        pnl.WhenFinished(second_terminator),
    )

    plan = BatchedCompositionCompiler.compile(
        model.composition,
        backend="triton_cpu",
        outputs=(*model.outputs, second_gate.output_port),
        max_steps=4,
    )
    streams = plan.kernel_ir.rng_streams
    assert tuple(stream.stream_id for stream in streams) == (0, 1)
    assert {stream.node for stream in streams} == {
        model.terminator.name,
        second_terminator.name,
    }
    with pytest.raises(ValueError, match="RNG stream"):
        replace(plan.kernel_ir, rng_streams=streams[::-1])

    emitter = TritonGraphEmitter(plan.kernel_ir)
    emitter._index_rng_streams()
    assert tuple(emitter.rng_stream_offset(stream.node) for stream in streams) == (
        0,
        RNG_STREAM_STRIDE,
    )

    duplicate_owner = replace(
        streams[1],
        node=streams[0].node,
        component_id=streams[0].component_id,
    )
    emitter.kernel = SimpleNamespace(
        rng_streams=(streams[0], duplicate_owner),
    )
    with pytest.raises(ValueError, match="at most one stream declaration"):
        emitter._index_rng_streams()


def test_scheduled_terminator_schema_rejects_semantic_forgeries():
    model = _build_scheduled_terminator(noise=0.25)
    kernel = BatchedCompositionCompiler.compile(
        model.composition,
        backend="triton_cpu",
        outputs=model.outputs,
        max_steps=4,
    ).kernel_ir
    program = _dynamic_region(kernel).attrs["program"]
    terminator_id = kernel.graph.node(model.terminator.name).component_id

    forged_fusion = None if kernel.fusion_kind is not None else "forged"
    with pytest.raises(ValueError, match="fusion kind"):
        replace(kernel, fusion_kind=forged_fusion)

    for dimensions in (
        kernel.lane_layout.dimensions[:-1],
        list(kernel.lane_layout.dimensions),
    ):
        forged_layout = replace(kernel.lane_layout, dimensions=dimensions)
        with pytest.raises(ValueError, match="lane layout"):
            replace(kernel, lane_layout=forged_layout)

    stream = kernel.rng_streams[0]
    rng_forgeries = (
        ("stream_id", stream.stream_id + 1),
        ("name", f"{stream.name}.forged"),
        ("node", f"{stream.node}.forged"),
        ("component_id", stream.component_id + 1),
        ("width", stream.width + 1),
        ("width", float(stream.width)),
        ("step_extent", f"{stream.step_extent}.forged"),
    )
    for field, value in rng_forgeries:
        forged_stream = replace(stream, **{field: value})
        with pytest.raises(ValueError, match="RNG stream"):
            replace(kernel, rng_streams=(forged_stream,))

    graph_stream = kernel.graph.rng_streams[0]
    for field, value in (
        ("name", ""),
        ("node", ""),
        ("width", 0),
        ("step_extent", ""),
        ("component_id", -1),
        ("stream_id", -1),
    ):
        forged_kernel_stream = replace(stream, **{field: value})
        forged_graph = replace(
            kernel.graph,
            rng_streams=(replace(graph_stream, **{field: value}),),
        )
        with pytest.raises(ValueError, match="RNG stream"):
            replace(
                kernel,
                graph=forged_graph,
                rng_streams=(forged_kernel_stream,),
            )

    for field, value in (
        ("stream_id", False),
        ("node", f"{graph_stream.node}.forged"),
        ("width", True),
    ):
        forged_graph = replace(
            kernel.graph,
            rng_streams=(replace(graph_stream, **{field: value}),),
        )
        assert not _dynamic_scheduled_graph_eligible(forged_graph, kernel.params)

    with pytest.raises(ValueError, match="schedule fuel"):
        replace(program, schedule_fuel=program.schedule_fuel - 1)

    trial_state = next(
        carry
        for carry in program.loop_carries
        if carry.kind == "trial_state"
        and carry.owner_component_id == terminator_id
    )
    with pytest.raises(ValueError, match="loop carry"):
        replace(trial_state, initial_value=(1.0,))
    wrong_parameter = next(
        parameter
        for parameter in kernel.params
        if parameter.owner_component_id == terminator_id
        and parameter.parameter_id != trial_state.initial_parameter_id
    )
    forged_state = replace(
        trial_state,
        initial_parameter_id=wrong_parameter.parameter_id,
    )
    forged_program = replace(
        program,
        loop_carries=tuple(
            forged_state if carry is trial_state else carry
            for carry in program.loop_carries
        ),
    )
    with pytest.raises(ValueError, match="compiler-derived|exact"):
        _replace_dynamic_program(kernel, forged_program)

    member = next(
        member
        for member in _program_members(program)
        if member.component_id == terminator_id
    )
    step = next(op for op in member.body if op.kind == "StepMechanism")
    forged_step = replace(
        step,
        attrs={**step.attrs, "rng_stream_ids": ()},
    )
    forged_member = replace(
        member,
        body=tuple(forged_step if op is step else op for op in member.body),
    )
    forged_program = _replace_program_member(
        program,
        terminator_id,
        forged_member,
    )
    with pytest.raises(ValueError, match="compiler-derived|exact|RNG"):
        _replace_dynamic_program(kernel, forged_program)


def test_deterministic_scheduled_terminator_matches_fresh_python(
    batched_backend,
):
    python_model = _build_scheduled_terminator()
    python_model.composition.run(
        inputs=python_model.inputs,
        execution_mode=pnl.ExecutionMode.Python,
    )
    expected = _selected_python_results(python_model)
    np.testing.assert_allclose(
        expected,
        _DETERMINISTIC_RESULTS,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    execution_list = python_model.composition.scheduler.execution_list[
        python_model.composition.default_execution_id
    ]
    assert sum(python_model.stepper in item for item in execution_list) == 18
    assert sum(python_model.terminator in item for item in execution_list) == 12
    assert all(
        sum(gate in item for item in execution_list) == 3
        for gate in python_model.gates
    )

    compiled_model = _build_scheduled_terminator()
    plan = BatchedCompositionCompiler.compile(
        compiled_model.composition,
        backend=batched_backend,
        outputs=compiled_model.outputs,
        max_steps=4,
    )
    result = plan.run(
        inputs=compiled_model.inputs,
        parameter_sets=[{}],
        num_estimates=1,
        seed=17,
    )

    np.testing.assert_allclose(
        result.values[0, 0, :, 0, :],
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert result.metadata["truncation"][compiled_model.stepper.name] == 0.0
    assert result.metadata["truncation"][compiled_model.terminator.name] == 0.0


def test_component_cap_is_not_the_schedule_fuel(batched_backend):
    exact_model = _build_scheduled_terminator()
    exact_plan = BatchedCompositionCompiler.compile(
        exact_model.composition,
        backend=batched_backend,
        outputs=exact_model.outputs,
        max_steps=4,
    )
    exact_result = exact_plan.run(
        inputs=exact_model.inputs,
        parameter_sets=[{}],
        num_estimates=1,
        seed=19,
    )
    assert exact_result.metadata["truncation"][exact_model.terminator.name] == 0.0

    below_model = _build_scheduled_terminator()
    below_plan = BatchedCompositionCompiler.compile(
        below_model.composition,
        backend=batched_backend,
        outputs=below_model.outputs,
        max_steps=3,
    )
    with pytest.warns(UserWarning, match="truncated bounded loops"):
        below_result = below_plan.run(
            inputs=below_model.inputs,
            parameter_sets=[{}],
            num_estimates=1,
            seed=19,
        )
    assert below_result.metadata["truncation"][below_model.stepper.name] == 0.0
    assert below_result.metadata["truncation"][below_model.terminator.name] == 1.0


def test_runtime_starting_value_lanes_match_fresh_python(batched_backend):
    starting_values = (0.05, -0.05)
    expected = []
    for starting_value in starting_values:
        python_model = _build_scheduled_terminator(
            starting_value=starting_value,
        )
        python_model.composition.run(
            inputs=python_model.inputs,
            execution_mode=pnl.ExecutionMode.Python,
        )
        expected.append(_selected_python_results(python_model))

    compiled_model = _build_scheduled_terminator()
    parameter_name = f"{compiled_model.terminator.name}.starting_value"
    plan = BatchedCompositionCompiler.compile(
        compiled_model.composition,
        backend=batched_backend,
        outputs=compiled_model.outputs,
        max_steps=6,
    )
    result = plan.run(
        inputs=compiled_model.inputs,
        parameter_sets=[
            {parameter_name: starting_value}
            for starting_value in starting_values
        ],
        num_estimates=1,
        seed=31,
    )

    np.testing.assert_allclose(
        result.values[:, 0, :, 0, :],
        np.asarray(expected),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert result.metadata["truncation"][compiled_model.terminator.name] == 0.0


def test_stochastic_scheduled_terminator_replays_seed_and_changes_seed(
    batched_backend,
):
    model = _build_scheduled_terminator(
        noise=0.35,
        threshold=0.01,
        inputs=np.zeros((2, 2), dtype=float),
    )
    plan = BatchedCompositionCompiler.compile(
        model.composition,
        backend=batched_backend,
        outputs=model.outputs,
        max_steps=4,
    )
    common = {
        "inputs": model.inputs,
        "parameter_sets": [{}],
        "num_estimates": 64,
    }
    first = plan.run(**common, seed=107).values
    replay = plan.run(**common, seed=107).values
    changed = plan.run(**common, seed=108).values

    np.testing.assert_array_equal(first, replay)
    assert not np.array_equal(first, changed)
