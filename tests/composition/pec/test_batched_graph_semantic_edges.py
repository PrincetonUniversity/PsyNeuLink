import itertools

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import BatchedCompositionCompiler
from psyneulink.core.batched.graph import lower_composition
from psyneulink.core.batched.ir import BatchedCompositionIR
from psyneulink.core.batched.kernel_ir import iter_kernel_ops, lower_to_kernel_ir

from batched_semantic_test_support import (
    SemanticCase,
    SemanticModel,
    assert_matches_python,
)


pytestmark = [
    pytest.mark.batched,
    pytest.mark.composition,
]


def _add_projection(composition, sender, receiver, matrix):
    composition.add_projection(
        sender=sender,
        receiver=receiver,
        projection=pnl.MappingProjection(matrix=np.asarray(matrix, dtype=float)),
    )


def _prefix_input_case(*, longer_name_first):
    build_number = itertools.count()

    def build():
        # Keep the exact prefix relationship on both fresh builds without
        # colliding with the global PsyNeuLink name registry: A/AB, then A1/A1B.
        index = next(build_number)
        short_name = "A" if index == 0 else f"A{index}"
        short = pnl.TransferMechanism(input_shapes=1, name=short_name)
        long = pnl.TransferMechanism(input_shapes=1, name=f"{short_name}B")
        receiver = pnl.TransferMechanism(input_shapes=1, name=f"prefix-result-{index}")
        composition = pnl.Composition()
        composition.add_nodes([short, long, receiver])
        _add_projection(composition, short, receiver, [[2.0]])
        _add_projection(composition, long, receiver, [[3.0]])

        short_values = np.asarray([[1.0], [2.0], [-1.0]])
        long_values = np.asarray([[10.0], [20.0], [4.0]])
        ordered_inputs = (
            ((long, long_values), (short, short_values))
            if longer_name_first
            else ((short, short_values), (long, long_values))
        )
        return SemanticModel(
            composition=composition,
            inputs=dict(ordered_inputs),
            outputs=(receiver.output_port,),
        )

    order = "AB_before_A" if longer_name_first else "A_before_AB"
    return SemanticCase(
        name=f"exact_prefix_inputs_{order}",
        build=build,
        provenance="checkpoint-2 regression: exact input identity for A versus AB",
    )


def _duplicate_name_case():
    build_number = itertools.count()

    def build():
        index = next(build_number)
        requested_name = f"duplicate-{index}"
        first = pnl.TransferMechanism(input_shapes=1, name=requested_name)
        second = pnl.TransferMechanism(input_shapes=1, name=requested_name)
        receiver = pnl.TransferMechanism(input_shapes=1, name=f"duplicate-result-{index}")
        assert first.name != second.name

        composition = pnl.Composition()
        composition.add_nodes([first, second, receiver])
        _add_projection(composition, first, receiver, [[2.0]])
        _add_projection(composition, second, receiver, [[-4.0]])
        return SemanticModel(
            composition=composition,
            # Put the automatically suffixed name first to exercise exact input
            # binding as well as distinct graph identities.
            inputs={
                second: np.asarray([[3.0], [7.0]]),
                first: np.asarray([[11.0], [-2.0]]),
            },
            outputs=(receiver.output_port,),
        )

    return SemanticCase(
        name="duplicate_auto_suffixed_names",
        build=build,
        provenance="checkpoint-2 regression: duplicate PsyNeuLink names get distinct identities",
    )


def _sanitized_name_collision_case():
    build_number = itertools.count()

    def build():
        index = next(build_number)
        dashed = pnl.TransferMechanism(input_shapes=1, name=f"collision-{index}")
        underscored = pnl.TransferMechanism(input_shapes=1, name=f"collision_{index}")
        receiver = pnl.TransferMechanism(input_shapes=1, name=f"collision-result-{index}")
        composition = pnl.Composition()
        composition.add_nodes([dashed, underscored, receiver])
        _add_projection(composition, dashed, receiver, [[5.0]])
        _add_projection(composition, underscored, receiver, [[-2.0]])
        return SemanticModel(
            composition=composition,
            inputs={
                underscored: np.asarray([[4.0], [-3.0]]),
                dashed: np.asarray([[1.0], [8.0]]),
            },
            outputs=(receiver.output_port,),
        )

    return SemanticCase(
        name="sanitized_identifier_collision",
        build=build,
        provenance="checkpoint-2 regression: codegen symbols do not define component identity",
    )


def _target_before_source_case():
    build_number = itertools.count()

    def build():
        index = next(build_number)
        source = pnl.TransferMechanism(input_shapes=2, name=f"late-source-{index}")
        target = pnl.TransferMechanism(input_shapes=3, name=f"early-target-{index}")
        composition = pnl.Composition()
        composition.add_nodes([target, source])
        _add_projection(
            composition,
            source,
            target,
            [[1.0, -2.0, 0.5], [3.0, 0.25, -1.0]],
        )
        return SemanticModel(
            composition=composition,
            inputs={source: np.asarray([[1.0, 2.0], [-4.0, 0.5]])},
            outputs=(target.output_port,),
        )

    return SemanticCase(
        name="target_inserted_before_source",
        build=build,
        provenance="checkpoint-2 regression: execution order follows dependencies, not insertion",
    )


def _vector_output_case():
    build_number = itertools.count()

    def build():
        index = next(build_number)
        vector = pnl.TransferMechanism(input_shapes=4, name=f"vector-output-{index}")
        composition = pnl.Composition(pathways=vector)
        return SemanticModel(
            composition=composition,
            inputs={
                vector: np.asarray(
                    [[1.0, -2.0, 3.5, 0.25], [-4.0, 5.0, 0.0, 9.0]]
                )
            },
            outputs=(vector.output_port,),
        )

    return SemanticCase(
        name="vector_primary_output",
        build=build,
        provenance="checkpoint-2 output contract: preserve every primary-port component",
    )


def _reordered_output_case():
    build_number = itertools.count()

    def build():
        index = next(build_number)
        first = pnl.TransferMechanism(input_shapes=1, name=f"first-output-{index}")
        second = pnl.TransferMechanism(input_shapes=2, name=f"second-output-{index}")
        composition = pnl.Composition()
        composition.add_nodes([first, second])
        return SemanticModel(
            composition=composition,
            inputs={
                first: np.asarray([[1.0], [2.0]]),
                second: np.asarray([[10.0, 20.0], [30.0, 40.0]]),
            },
            outputs=(second.output_port, first.output_port),
        )

    return SemanticCase(
        name="explicit_reordered_primary_outputs",
        build=build,
        provenance="checkpoint-2 output contract: explicit output order is observable",
    )


def _rectangular_fan_in_case():
    build_number = itertools.count()

    def build():
        index = next(build_number)
        left = pnl.TransferMechanism(input_shapes=2, name=f"fan-in-left-{index}")
        right = pnl.TransferMechanism(input_shapes=1, name=f"fan-in-right-{index}")
        receiver = pnl.TransferMechanism(input_shapes=3, name=f"fan-in-result-{index}")
        composition = pnl.Composition()
        composition.add_nodes([left, right, receiver])
        _add_projection(
            composition,
            left,
            receiver,
            [[1.0, 2.0, 0.0], [-1.0, 0.5, 3.0]],
        )
        _add_projection(composition, right, receiver, [[4.0, -2.0, 0.25]])
        return SemanticModel(
            composition=composition,
            inputs={
                left: np.asarray([[1.0, 2.0], [-3.0, 0.5]]),
                right: np.asarray([[5.0], [-2.0]]),
            },
            outputs=(receiver.output_port,),
        )

    return SemanticCase(
        name="rectangular_fan_in",
        build=build,
        provenance="checkpoint-2 projection contract: sum unequal rectangular sources",
    )


def _rectangular_fan_out_case():
    build_number = itertools.count()

    def build():
        index = next(build_number)
        source = pnl.TransferMechanism(input_shapes=2, name=f"fan-out-source-{index}")
        scalar = pnl.TransferMechanism(input_shapes=1, name=f"fan-out-scalar-{index}")
        vector = pnl.TransferMechanism(input_shapes=3, name=f"fan-out-vector-{index}")
        composition = pnl.Composition()
        composition.add_nodes([source, scalar, vector])
        _add_projection(composition, source, scalar, [[2.0], [-1.0]])
        _add_projection(
            composition,
            source,
            vector,
            [[1.0, 0.0, 2.0], [0.5, -3.0, 1.0]],
        )
        return SemanticModel(
            composition=composition,
            inputs={source: np.asarray([[1.0, 2.0], [-4.0, 0.5]])},
            outputs=(scalar.output_port, vector.output_port),
        )

    return SemanticCase(
        name="rectangular_fan_out",
        build=build,
        provenance="checkpoint-2 projection contract: one source feeds unequal receivers",
    )


def _separate_receiver_input_ports_case():
    build_number = itertools.count()

    def build():
        index = next(build_number)
        left = pnl.TransferMechanism(input_shapes=1, name=f"left-source-{index}")
        right = pnl.TransferMechanism(input_shapes=1, name=f"right-source-{index}")
        receiver = pnl.TransferMechanism(
            input_ports=[
                {pnl.NAME: "left-input", pnl.INPUT_SHAPES: 1},
                {pnl.NAME: "right-input", pnl.INPUT_SHAPES: 1},
            ],
            name=f"separate-input-receiver-{index}",
        )
        composition = pnl.Composition()
        composition.add_nodes([left, right, receiver])
        composition.add_projection(sender=left, receiver=receiver.input_ports[0])
        composition.add_projection(sender=right, receiver=receiver.input_ports[1])
        return SemanticModel(
            composition=composition,
            inputs={
                left: np.asarray([[2.0], [-3.0]]),
                right: np.asarray([[5.0], [7.0]]),
            },
            # Reverse the two automatic RESULT-N ports to exercise both port
            # identity and flattened output ordering.
            outputs=(receiver.output_ports[1], receiver.output_ports[0]),
        )

    return SemanticCase(
        name="separate_receiver_input_ports",
        build=build,
        provenance=(
            "tests/composition/test_interfaces.py:92-119; per-InputPort "
            "mechanism variable and RESULT-N routing"
        ),
    )


def _per_port_sum_product_case():
    build_number = itertools.count()

    def build():
        index = next(build_number)
        sources = [
            pnl.TransferMechanism(input_shapes=1, name=f"port-source-{slot}-{index}")
            for slot in range(4)
        ]
        receiver = pnl.TransferMechanism(
            input_ports=[
                {
                    pnl.NAME: "sum-input",
                    pnl.INPUT_SHAPES: 1,
                    pnl.FUNCTION: pnl.LinearCombination(operation=pnl.SUM),
                },
                {
                    pnl.NAME: "product-input",
                    pnl.INPUT_SHAPES: 1,
                    # This PNL spelling intentionally does not populate the
                    # convenience InputPort.combine attribute.
                    pnl.FUNCTION: pnl.LinearCombination(operation=pnl.PRODUCT),
                },
            ],
            function=pnl.Linear(slope=2.0, intercept=1.0),
            name=f"per-port-combine-{index}",
        )
        composition = pnl.Composition()
        composition.add_nodes([*sources, receiver])
        for source in sources[:2]:
            composition.add_projection(sender=source, receiver=receiver.input_ports[0])
        for source in sources[2:]:
            composition.add_projection(sender=source, receiver=receiver.input_ports[1])
        trial_values = (
            np.asarray([[2.0], [-1.0]]),
            np.asarray([[3.0], [4.0]]),
            np.asarray([[4.0], [-2.0]]),
            np.asarray([[5.0], [3.0]]),
        )
        return SemanticModel(
            composition=composition,
            inputs=dict(zip(sources, trial_values)),
            outputs=tuple(receiver.output_ports),
        )

    return SemanticCase(
        name="per_port_sum_and_product",
        build=build,
        provenance=(
            "tests/ports/test_input_ports.py:14-49,70-76; SUM and PRODUCT "
            "must be applied independently per receiver InputPort"
        ),
    )


def _ddm_reordered_outputs_case(*, fan_out):
    build_number = itertools.count()

    def build():
        index = next(build_number)
        stimulus = pnl.TransferMechanism(input_shapes=1, name=f"ddm-stimulus-{index}")
        decision = pnl.DDM(
            function=pnl.DriftDiffusionIntegrator(
                starting_value=0.0,
                rate=1.0,
                noise=0.0,
                threshold=0.05,
                non_decision_time=0.2,
                time_step_size=0.01,
            ),
            output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
            name=f"port-ddm-{index}",
        )
        composition = pnl.Composition(pathways=[[stimulus, decision]])
        inputs = {stimulus: np.asarray([[1.0], [-1.0]])}
        if not fan_out:
            outputs = (
                decision.output_ports[pnl.RESPONSE_TIME],
                decision.output_ports[pnl.DECISION_OUTCOME],
            )
        else:
            outcome = pnl.TransferMechanism(
                input_shapes=1,
                function=pnl.Linear(slope=10.0, intercept=1.0),
                name=f"outcome-readout-{index}",
            )
            response_time = pnl.TransferMechanism(
                input_shapes=1,
                function=pnl.Linear(slope=10.0, intercept=2.0),
                name=f"rt-readout-{index}",
            )
            composition.add_nodes([outcome, response_time])
            composition.add_projection(
                sender=decision.output_ports[pnl.DECISION_OUTCOME],
                receiver=outcome,
            )
            composition.add_projection(
                sender=decision.output_ports[pnl.RESPONSE_TIME],
                receiver=response_time,
            )
            outputs = (response_time.output_port, outcome.output_port)
        return SemanticModel(
            composition=composition,
            inputs=inputs,
            outputs=outputs,
        )

    return SemanticCase(
        name=("ddm_named_output_fan_out" if fan_out else "ddm_reordered_named_outputs"),
        build=build,
        provenance=(
            "tests/mechanisms/test_ddm_mechanism.py::TestOutputPorts; preserve "
            "DECISION_OUTCOME versus RESPONSE_TIME identity"
        ),
        max_steps=64,
    )


PARITY_CASES = (
    _prefix_input_case(longer_name_first=False),
    _prefix_input_case(longer_name_first=True),
    _duplicate_name_case(),
    _sanitized_name_collision_case(),
    _target_before_source_case(),
    _vector_output_case(),
    _reordered_output_case(),
    _rectangular_fan_in_case(),
    _rectangular_fan_out_case(),
    _separate_receiver_input_ports_case(),
    _per_port_sum_product_case(),
    _ddm_reordered_outputs_case(fan_out=False),
    _ddm_reordered_outputs_case(fan_out=True),
)


@pytest.mark.parametrize("case", PARITY_CASES, ids=lambda case: case.name)
def test_graph_edge_case_matches_python(case, batched_backend):
    assert_matches_python(case, backend=batched_backend)


def _assert_structured_rejection(composition, reason, *, outputs=None):
    report = BatchedCompositionCompiler.diagnose(
        composition,
        backend="triton_cpu",
        outputs=outputs,
    )
    matches = [diagnostic for diagnostic in report.model_diagnostics if diagnostic.reason == reason]
    assert matches, report.to_dict()
    assert not report.model_supported
    assert report.codegen_ready is None

    diagnostic = matches[0]
    serialized = diagnostic.to_dict()
    assert serialized["code"].startswith("model.")
    assert serialized["component"]
    assert serialized["component_id"]
    assert serialized["reason"] == reason
    return diagnostic


def _lower_case_without_backend(case):
    """Build GraphIR and KernelIR without requiring Triton to be installed."""

    model = case.build()
    lowering = lower_composition(model.composition, outputs=tuple(model.outputs))
    assert lowering.graph is not None
    assert not lowering.rejected_nodes
    assert not lowering.rejected_conditions
    graph = lowering.graph
    ir = BatchedCompositionIR(
        model_kind=lowering.model_kind,
        node_names=tuple(node.name for node in graph.nodes),
        params=lowering.params,
        output_names=tuple(output.name for output in graph.outputs),
        max_steps=256,
        graph=graph,
    )
    return model, graph, lower_to_kernel_ir(ir)


def test_separate_receiver_input_ports_lower_by_stable_port_identity():
    model, graph, kernel_ir = _lower_case_without_backend(
        _separate_receiver_input_ports_case()
    )
    receiver = model.outputs[0].owner
    receiver_port_ids = {projection.receiver_port_id for projection in graph.projections}
    operations = iter_kernel_ops(kernel_ir)
    combines = [
        operation
        for operation in operations
        if operation.kind == "CombineSum" and operation.target == receiver.name
    ]
    concatenate = next(
        operation
        for operation in operations
        if operation.kind == "Concatenate" and operation.target == receiver.name
    )

    assert len(receiver_port_ids) == 2
    assert {operation.attrs["receiver_port_id"] for operation in combines} == receiver_port_ids
    assert concatenate.attrs["port_ids"] == tuple(
        port_id
        for _, _, _, port_id, _, _ in graph.node(receiver.name).attrs["input_ports"]
    )


def test_each_receiver_port_lowers_its_own_sum_or_product():
    model, graph, kernel_ir = _lower_case_without_backend(_per_port_sum_product_case())
    receiver = model.outputs[0].owner
    expected = {
        port.name: ("CombineProduct" if index else "CombineSum")
        for index, port in enumerate(receiver.input_ports)
    }
    combines = {
        operation.attrs["receiver_port"]: operation.kind
        for operation in iter_kernel_ops(kernel_ir)
        if operation.kind in {"CombineSum", "CombineProduct"}
        and operation.target == receiver.name
    }

    assert combines == expected
    assert {
        operation.attrs["receiver_port_id"]
        for operation in iter_kernel_ops(kernel_ir)
        if operation.target == receiver.name
        and operation.kind in {"CombineSum", "CombineProduct"}
    } == {projection.receiver_port_id for projection in graph.projections}


def test_derived_output_port_function_remains_fail_closed():
    mechanism = pnl.TransferMechanism(
        input_shapes=3,
        output_ports=[pnl.RESULT, pnl.MEAN],
        name="non-primary-output",
    )
    composition = pnl.Composition(pathways=mechanism)
    requested_outputs = (mechanism.output_ports[pnl.MEAN],)

    node_diagnostic = _assert_structured_rejection(
        composition,
        "unsupported OutputPort function for batched v2",
        outputs=requested_outputs,
    )
    assert node_diagnostic.component == mechanism.name
    assert pnl.MEAN in node_diagnostic.detail

    output_diagnostic = _assert_structured_rejection(
        composition,
        "unsupported output-port routing for batched v2",
        outputs=requested_outputs,
    )
    assert output_diagnostic.component == mechanism.name
    assert output_diagnostic.detail == pnl.MEAN
