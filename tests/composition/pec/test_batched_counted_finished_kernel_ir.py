"""KernelIR contract for precomputed counted-finished stateful schedules.

These tests deliberately exercise GraphIR -> KernelIR lowering directly.  The
public capability boundary remains closed until the complete counted-finished
slice is promoted, so the fixture starts from a supported atomic LCA graph and
changes only its object-free finished predicate and schedule classification.
No live PsyNeuLink scheduler object is interpreted by KernelIR or the emitter.
"""

from dataclasses import replace
import re

import pytest

import psyneulink as pnl
from psyneulink.core.batched import specs
from psyneulink.core.batched.backend.triton.graph_emit import (
    triton_graph_kernel_source,
)
from psyneulink.core.batched.graph import lower_composition
from psyneulink.core.batched.ir import BatchedCompositionIR
from psyneulink.core.batched.kernel_ir import (
    iter_kernel_ops,
    lower_to_kernel_ir,
)


pytestmark = [pytest.mark.batched, pytest.mark.composition]


_FINISHED_AFTER = 3
_STEP_HELPER = "_pnl_triton_lca_width2_step"


def _counted_finished_kernel(*, weighted_op_budget=None):
    producer = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(gain=1.0),
        leak=0.0,
        competition=0.0,
        self_excitation=0.0,
        noise=0.0,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=_FINISHED_AFTER,
        time_step_size=0.5,
        # Use the already-supported atomic form to obtain a complete GraphIR.
        # The replacement below is the direct-IR declaration under test.
        execute_until_finished=True,
        reset_stateful_function_when=pnl.Never(),
        name="counted finished producer",
    )
    follower = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Linear(slope=2.0, intercept=-0.25),
        name="stateless finished follower",
    )
    composition = pnl.Composition(name="generic counted finished graph")
    composition.add_nodes([producer, follower])
    composition.add_projection(
        sender=producer,
        receiver=follower,
        projection=pnl.MappingProjection(matrix=[[1.0], [-1.0]]),
    )
    composition.scheduler.add_condition(producer, pnl.Always())
    composition.scheduler.add_condition(follower, pnl.WhenFinished(producer))

    lowering = lower_composition(
        composition,
        outputs=(follower.output_port,),
    )
    graph = lowering.graph
    assert graph is not None
    assert graph.executable
    assert len(graph.finished_values) == 1
    assert graph.finished_values[0].predicate_kind == "dynamic"

    finished_value = replace(
        graph.finished_values[0],
        predicate_kind="execution_count_at_least",
        attrs={"count": _FINISHED_AFTER},
    )
    metadata = {
        **graph.metadata,
        "schedule_kind": "precomputed_trace",
        "scheduler_requires_pass_region": True,
    }
    if weighted_op_budget is not None:
        metadata["schedule_trace_weighted_op_budget"] = weighted_op_budget
    graph = replace(
        graph,
        finished_values=(finished_value,),
        metadata=metadata,
    )
    composition_ir = BatchedCompositionIR(
        model_kind=lowering.model_kind,
        node_names=tuple(node.name for node in graph.nodes),
        params=lowering.params,
        output_names=tuple(output.name for output in graph.outputs),
        graph=graph,
    )
    return lower_to_kernel_ir(composition_ir), producer, follower


def _trial_schedule(kernel):
    assert tuple(op.kind for op in kernel.ops) == (
        "InitializeState",
        "ForTrials",
    )
    initialize, trials = kernel.ops
    trial_body = tuple(trials.attrs["body"])
    assert tuple(op.kind for op in trial_body) == ("ForPasses", "StoreOutput")
    return initialize, trial_body[0], trial_body[1]


def test_counted_finished_stateful_trace_uses_one_step_mechanism_ops():
    kernel, producer, follower = _counted_finished_kernel()

    assert kernel.executable
    assert kernel.schedule_trace is not None
    assert kernel.schedule_trace.component_execution_count == _FINISHED_AFTER + 1
    initialize, passes, store = _trial_schedule(kernel)
    assert len(tuple(op for op in kernel.ops if op.kind == "InitializeState")) == 1
    assert tuple(value.name for value in initialize.outputs) == (
        "n0:state:0",
        "n0:state:1",
    )
    assert passes.attrs["declaration_only"] is False
    assert passes.attrs["trace_kind"] == "precomputed"

    consideration_sets = tuple(passes.attrs["body"])
    assert all(op.kind == "ExecuteConsiderationSet" for op in consideration_sets)
    nested_ops = tuple(
        child
        for consideration_set in consideration_sets
        for child in consideration_set.attrs["body"]
    )
    producer_steps = tuple(
        op
        for op in nested_ops
        if op.kind == "StepMechanism" and op.target == producer.name
    )
    assert len(producer_steps) == _FINISHED_AFTER
    assert tuple(op.attrs["execution_index"] for op in producer_steps) == tuple(
        range(_FINISHED_AFTER)
    )
    assert all(op.attrs["state_ids"] == (0, 1) for op in producer_steps)
    assert all(op.attrs["active_lanes"] == "all" for op in producer_steps)
    assert not any(
        op.kind == "CallMechanism" and op.target == producer.name
        for op in iter_kernel_ops(kernel)
    )

    follower_calls = tuple(
        op
        for op in nested_ops
        if op.kind == "CallFunction" and op.target == follower.name
    )
    assert len(follower_calls) == 1
    assert sum(op.kind == "StoreOutput" for op in iter_kernel_ops(kernel)) == 1
    assert store.target == f"{follower.name}.RESULT"
    assert all(
        child.kind not in {"StoreOutput", "StoreFlag"}
        for consideration_set in consideration_sets
        for child in consideration_set.attrs["body"]
    )


def test_counted_finished_steps_use_the_frozen_mechanism_spec():
    kernel, producer, _ = _counted_finished_kernel()
    producer_node = kernel.graph.node(producer.name)
    producer_steps = tuple(
        op
        for op in iter_kernel_ops(kernel)
        if op.kind == "StepMechanism" and op.target == producer.name
    )

    assert producer_steps
    assert {op.attrs["spec_key"] for op in producer_steps} == {
        producer_node.attrs["spec_key"]
    }
    frozen_spec = kernel.op_specs.lookup_spec(producer_node.attrs["spec_key"])
    assert frozen_spec is specs.mechanism_spec_for(producer)
    assert frozen_spec.can_step


@pytest.mark.parametrize(
    "attr, value, message",
    (
        ("component_id", 1, "component id 1, expected 0"),
        ("state_ids", (1,), "state IDs \\(1,\\), expected \\(0, 1\\)"),
        ("execution_index", 1, "execution index 1, expected 0"),
    ),
)
def test_counted_finished_step_identity_and_order_are_enforced(
    attr,
    value,
    message,
):
    kernel, _, _ = _counted_finished_kernel()
    initialize, trials = kernel.ops
    passes, store = trials.attrs["body"]
    first_set, *remaining_sets = passes.attrs["body"]
    first_step_index = next(
        index
        for index, op in enumerate(first_set.attrs["body"])
        if op.kind == "StepMechanism"
    )
    first_body = list(first_set.attrs["body"])
    first_step = first_body[first_step_index]
    first_body[first_step_index] = replace(
        first_step,
        attrs={**first_step.attrs, attr: value},
    )
    forged_set = replace(
        first_set,
        attrs={**first_set.attrs, "body": tuple(first_body)},
    )
    forged_passes = replace(
        passes,
        attrs={**passes.attrs, "body": (forged_set, *remaining_sets)},
    )
    forged_trials = replace(
        trials,
        attrs={**trials.attrs, "body": (forged_passes, store)},
    )

    with pytest.raises(ValueError, match=message):
        replace(kernel, ops=(initialize, forged_trials))


def test_counted_finished_step_count_must_match_the_typed_trace():
    kernel, _, _ = _counted_finished_kernel()
    initialize, trials = kernel.ops
    passes, store = trials.attrs["body"]
    sets = list(passes.attrs["body"])
    final_producer_set = sets[-2]
    final_body = list(final_producer_set.attrs["body"])
    final_step = next(op for op in final_body if op.kind == "StepMechanism")
    final_body.append(
        replace(
            final_step,
            attrs={**final_step.attrs, "execution_index": _FINISHED_AFTER},
        )
    )
    sets[-2] = replace(
        final_producer_set,
        attrs={**final_producer_set.attrs, "body": tuple(final_body)},
    )
    forged_passes = replace(
        passes,
        attrs={**passes.attrs, "body": tuple(sets)},
    )
    forged_trials = replace(
        trials,
        attrs={**trials.attrs, "body": (forged_passes, store)},
    )

    with pytest.raises(ValueError, match="does not match its typed schedule trace"):
        replace(kernel, ops=(initialize, forged_trials))


def test_counted_finished_set_body_must_match_declared_component_identity():
    kernel, _, _ = _counted_finished_kernel()
    initialize, trials = kernel.ops
    passes, store = trials.attrs["body"]
    sets = list(passes.attrs["body"])
    producer_set = sets[-2]
    follower_set = sets[-1]
    sets[-2] = replace(
        producer_set,
        attrs={**producer_set.attrs, "body": follower_set.attrs["body"]},
    )
    sets[-1] = replace(
        follower_set,
        attrs={**follower_set.attrs, "body": producer_set.attrs["body"]},
    )
    forged_passes = replace(passes, attrs={**passes.attrs, "body": tuple(sets)})
    forged_trials = replace(
        trials,
        attrs={**trials.attrs, "body": (forged_passes, store)},
    )

    with pytest.raises(ValueError, match="is not one of its declared component IDs"):
        replace(kernel, ops=(initialize, forged_trials))


def test_counted_finished_state_and_output_effects_match_graph_ir():
    kernel, _, _ = _counted_finished_kernel()
    initialize, trials = kernel.ops
    passes, store = trials.attrs["body"]

    with pytest.raises(ValueError, match="retained-state declarations"):
        replace(kernel, states=kernel.states[:-1])

    forged_trials = replace(
        trials,
        attrs={**trials.attrs, "body": (passes, store, store)},
    )
    with pytest.raises(ValueError, match="exactly one StoreOutput"):
        replace(kernel, ops=(initialize, forged_trials))


def test_counted_finished_source_calls_one_step_helper_once_per_execution():
    kernel, _, _ = _counted_finished_kernel()

    source = triton_graph_kernel_source(kernel)
    helper_calls = tuple(
        line
        for line in source.splitlines()
        if re.search(rf"=\s*{re.escape(_STEP_HELPER)}\(", line)
    )

    assert source.count(f"def {_STEP_HELPER}(") == 1
    assert len(helper_calls) == _FINISHED_AFTER
    assert re.search(r"\bstep\b", source) is None
    assert source.count("tl.store(out + lane_out") == 1


def test_counted_finished_step_return_cardinality_is_exact():
    kernel, producer, _ = _counted_finished_kernel()
    spec_key = kernel.graph.node(producer.name).attrs["spec_key"]
    original = kernel.op_specs.lookup_spec(spec_key)

    def extra_result(*args, **kwargs):
        return tuple(original.step_emit(*args, **kwargs)) + ("extra",)

    replacement = replace(original, step_emit=extra_result)
    frozen = specs.BatchedOpSpecSnapshot(
        {**kernel.op_specs.specs_by_key, spec_key: replacement}
    )
    forged = replace(kernel, op_specs=frozen)

    with pytest.raises(ValueError, match="returned 3 value\\(s\\), expected 2"):
        triton_graph_kernel_source(forged)


def test_counted_finished_weighted_expansion_is_bounded_before_unrolling():
    with pytest.raises(
        ValueError,
        match=r"weighted op expansion \d+ exceeds budget 1",
    ):
        _counted_finished_kernel(weighted_op_budget=1)
