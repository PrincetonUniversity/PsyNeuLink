import pytest

import psyneulink as pnl
from psyneulink.core.batched.graph import (
    STATEFUL_GRAPH_FUSION,
    lower_composition,
)


@pytest.mark.composition
def test_dependency_order_overrides_target_before_source_insertion():
    source = pnl.TransferMechanism(input_shapes=1, name="source")
    target = pnl.TransferMechanism(input_shapes=1, name="target")
    composition = pnl.Composition()
    composition.add_nodes([target, source])
    composition.add_projection(sender=source, receiver=target)

    result = lower_composition(composition)

    assert not result.rejected_nodes
    assert result.graph.execution_order == ("source", "target")
    assert tuple(op.target for op in result.graph.ops[:2]) == ("source", "target")
    assert [(projection.sender, projection.receiver) for projection in result.graph.projections] == [
        ("source", "target")
    ]


def _disconnected_stateful_composition(*, stateful_first):
    stateful = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(),
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=2,
        name="a_stateful",
    )
    terminator = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0,
            rate=1.0,
            noise=0.0,
            threshold=0.05,
            non_decision_time=0.0,
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="z_terminator",
    )
    composition = pnl.Composition()
    ordered = [stateful, terminator] if stateful_first else [terminator, stateful]
    composition.add_nodes(ordered)
    composition.scheduler.add_condition(stateful, pnl.Always())
    return composition


@pytest.mark.composition
@pytest.mark.parametrize("stateful_first", [True, False])
def test_disconnected_stateful_node_does_not_trigger_coevolution(stateful_first):
    result = lower_composition(
        _disconnected_stateful_composition(stateful_first=stateful_first)
    )

    assert not result.rejected_nodes
    # The deterministic tie-break puts the stateful node first, so a positional
    # coevolution heuristic would get this wrong.  There is no dependency path.
    assert result.graph.execution_order == ("a_stateful", "z_terminator")
    assert result.graph.fusion_kind == STATEFUL_GRAPH_FUSION
