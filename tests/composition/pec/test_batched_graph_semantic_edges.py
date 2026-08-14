import itertools

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import BatchedCompositionCompiler

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


def test_separate_receiver_input_ports_have_structured_rejection_until_port_routing_lands():
    left = pnl.TransferMechanism(input_shapes=1, name="left-source")
    right = pnl.TransferMechanism(input_shapes=1, name="right-source")
    receiver = pnl.TransferMechanism(
        input_ports=[
            {pnl.NAME: "left-input", pnl.INPUT_SHAPES: 1},
            {pnl.NAME: "right-input", pnl.INPUT_SHAPES: 1},
        ],
        name="separate-input-receiver",
    )
    composition = pnl.Composition()
    composition.add_nodes([left, right, receiver])
    composition.add_projection(sender=left, receiver=receiver.input_ports[0])
    composition.add_projection(sender=right, receiver=receiver.input_ports[1])

    node_diagnostic = _assert_structured_rejection(
        composition,
        "unsupported multi-port input routing for batched v2",
    )
    assert node_diagnostic.component == receiver.name
    assert node_diagnostic.detail == "input_ports=2"

    projection_diagnostic = _assert_structured_rejection(
        composition,
        "unsupported multi-port projection routing for batched v2",
    )
    assert f"{right.name}.RESULT" in projection_diagnostic.detail
    assert f"{receiver.name}.right-input" in projection_diagnostic.detail


def test_non_primary_output_has_structured_rejection_until_output_port_lowering_lands():
    mechanism = pnl.TransferMechanism(
        input_shapes=3,
        output_ports=[pnl.RESULT, pnl.MEAN],
        name="non-primary-output",
    )
    composition = pnl.Composition(pathways=mechanism)
    requested_outputs = (mechanism.output_ports[pnl.MEAN],)

    node_diagnostic = _assert_structured_rejection(
        composition,
        "unsupported multi-port output routing for batched v2",
        outputs=requested_outputs,
    )
    assert node_diagnostic.component == mechanism.name
    assert node_diagnostic.detail == "output_ports=2"

    output_diagnostic = _assert_structured_rejection(
        composition,
        "unsupported output-port routing for batched v2",
        outputs=requested_outputs,
    )
    assert output_diagnostic.component == mechanism.name
    assert output_diagnostic.detail == pnl.MEAN
