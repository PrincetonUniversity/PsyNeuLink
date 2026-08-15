"""Backend-neutral state-reset effects for scheduled KernelIR."""

from dataclasses import replace

import pytest

import psyneulink as pnl
from psyneulink.core.batched.backend.triton.graph_emit import (
    triton_graph_kernel_source,
)
from psyneulink.core.batched.graph import lower_composition
from psyneulink.core.batched.ir import BatchedCompositionIR
from psyneulink.core.batched.kernel_ir import KernelIR, KernelOp, lower_to_kernel_ir


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _counted_lca_kernel(*, reset_at_trial_start: bool):
    producer = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(gain=1.5, bias=0.25),
        leak=0.0,
        competition=0.0,
        self_excitation=0.0,
        noise=0.0,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=3,
        time_step_size=0.5,
        execute_until_finished=False,
        reset_stateful_function_when=pnl.Never(),
        name="reset-state producer",
    )
    follower = pnl.TransferMechanism(
        input_shapes=1,
        name="reset-state follower",
    )
    composition = pnl.Composition(name="direct reset-state KernelIR")
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
    assert graph is not None and graph.executable
    if reset_at_trial_start:
        graph = replace(
            graph,
            resets=(
                replace(graph.resets[0], condition_type="AtTrialStart"),
            ),
        )
    kernel = lower_to_kernel_ir(
        BatchedCompositionIR(
            model_kind=lowering.model_kind,
            node_names=tuple(node.name for node in graph.nodes),
            params=lowering.params,
            output_names=tuple(output.name for output in graph.outputs),
            graph=graph,
        )
    )
    return kernel, producer


def _trial_body(kernel: KernelIR):
    trials = tuple(op for op in kernel.ops if op.kind == "ForTrials")
    assert len(trials) == 1
    return tuple(trials[0].attrs["body"])


def _replace_trial_body(kernel: KernelIR, body):
    trials = next(op for op in kernel.ops if op.kind == "ForTrials")
    replacement = replace(trials, attrs={**trials.attrs, "body": tuple(body)})
    return replace(
        kernel,
        ops=tuple(replacement if op is trials else op for op in kernel.ops),
    )


def test_at_trial_start_reset_is_a_prefix_effect_with_exact_state_identity():
    kernel, producer = _counted_lca_kernel(reset_at_trial_start=True)

    assert kernel.executable
    body = _trial_body(kernel)
    assert tuple(op.kind for op in body) == (
        "ResetState",
        "ForPasses",
        "StoreOutput",
    )
    reset = body[0]
    declaration = kernel.resets[0]
    producer_states = tuple(
        state for state in kernel.states if state.component_id == declaration.component_id
    )
    assert reset.target == producer.name
    assert reset.inputs == ()
    assert reset.attrs == {
        "component_id": declaration.component_id,
        "state_ids": tuple(state.state_id for state in producer_states),
        "condition_type": "AtTrialStart",
        "region": "trial",
    }
    assert tuple(value.width for value in reset.outputs) == tuple(
        state.width for state in producer_states
    )


def test_never_reset_preserves_state_without_a_reset_effect():
    kernel, _ = _counted_lca_kernel(reset_at_trial_start=False)
    assert tuple(op.kind for op in _trial_body(kernel)) == (
        "ForPasses",
        "StoreOutput",
    )


def test_reset_source_reapplies_literal_and_function_initializers_before_pass_zero():
    kernel, _ = _counted_lca_kernel(reset_at_trial_start=True)
    source = triton_graph_kernel_source(kernel)

    reset_comment = source.index("# reset reset-state producer state at trial start")
    assert source.index("while trial_idx < num_trials") < reset_comment
    assert reset_comment < source.index("# precomputed scheduler pass 0")
    # The pre-state literal and the Logistic-derived activation are initialized
    # once outside ForTrials and reapplied by ResetState inside every trial.
    assert source.count("n0_state_0_0 = tl.full") == 2
    assert source.count("n0_state_1_0 =") == 2


@pytest.mark.parametrize(
    "mutation, detail",
    (
        ("wrong-component", "does not exactly match"),
        ("late", "unconditional prefix"),
        ("duplicate", "exactly one ResetState"),
    ),
)
def test_reset_effect_identity_and_placement_are_enforced(mutation, detail):
    kernel, _ = _counted_lca_kernel(reset_at_trial_start=True)
    body = _trial_body(kernel)
    reset = body[0]

    if mutation == "wrong-component":
        bad_reset = replace(
            reset,
            attrs={**reset.attrs, "component_id": reset.attrs["component_id"] + 1},
        )
        body = (bad_reset, *body[1:])
    elif mutation == "late":
        body = (body[1], reset, *body[2:])
    else:
        body = (reset, reset, *body[1:])

    with pytest.raises(ValueError, match=detail):
        _replace_trial_body(kernel, body)


def test_reset_effect_mutation_is_revalidated_before_source_emission():
    kernel, _ = _counted_lca_kernel(reset_at_trial_start=True)
    reset = _trial_body(kernel)[0]
    reset.attrs["state_ids"] = (reset.attrs["state_ids"][0],)

    with pytest.raises(ValueError, match="ResetState"):
        triton_graph_kernel_source(kernel)


def test_reset_op_rejects_partial_state_outputs_at_construction():
    kernel, _ = _counted_lca_kernel(reset_at_trial_start=True)
    reset = _trial_body(kernel)[0]

    with pytest.raises(ValueError, match="matching its state outputs"):
        KernelOp(
            kind="ResetState",
            target=reset.target,
            outputs=reset.outputs[:1],
            attrs=reset.attrs,
        )
