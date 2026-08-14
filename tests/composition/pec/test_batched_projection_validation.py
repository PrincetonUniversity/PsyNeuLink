import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import BatchedCompileError, BatchedCompositionCompiler
from psyneulink.core.batched.graph import lower_composition


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _projection_model(*, matrix=None, function=None, feedback=False):
    source = pnl.TransferMechanism(input_shapes=2, name="source")
    receiver = pnl.TransferMechanism(input_shapes=2, name="receiver")
    kwargs = {"matrix": np.eye(2) if matrix is None else matrix}
    if function is not None:
        kwargs["function"] = function
    projection = pnl.MappingProjection(**kwargs)
    composition = pnl.Composition()
    composition.add_nodes([source, receiver])
    composition.add_projection(
        projection=projection,
        sender=source,
        receiver=receiver,
        feedback=feedback,
    )
    return composition, source, receiver, projection


def _assert_projection_rejected(composition, receiver, projection, reason):
    outputs = (receiver.output_port,)
    report = BatchedCompositionCompiler.diagnose(
        composition,
        backend="triton_cpu",
        outputs=outputs,
    )

    matches = [
        diagnostic
        for diagnostic in report.model_diagnostics
        if diagnostic.component == projection.name and diagnostic.reason == reason
    ]
    assert len(matches) == 1, report.to_dict()
    assert not report.model_supported
    assert report.codegen_ready is None
    assert matches[0].code.startswith("model.")

    with pytest.raises(BatchedCompileError) as error:
        BatchedCompositionCompiler.compile(
            composition,
            backend="triton_cpu",
            outputs=outputs,
        )
    assert error.value.capability_report == report
    return matches[0]


def test_projection_owned_by_another_composition_is_not_lowered():
    source = pnl.TransferMechanism(input_shapes=1, name="source")
    receiver = pnl.TransferMechanism(input_shapes=1, name="receiver")
    first_composition = pnl.Composition(pathways=[[source, receiver]])
    stale_projection = next(
        projection
        for projection in first_composition.projections
        if projection.sender.owner is source and projection.receiver.owner is receiver
    )

    composition = pnl.Composition()
    composition.add_nodes([source, receiver])
    assert stale_projection in receiver.input_port.path_afferents
    assert stale_projection not in composition.projections

    lowering = lower_composition(composition, outputs=(receiver.output_port,))

    assert not lowering.rejected_nodes
    assert not lowering.rejected_conditions
    assert lowering.graph is not None
    assert not lowering.graph.projections
    assert {(input_spec.node, input_spec.port) for input_spec in lowering.graph.inputs} == {
        (source.name, source.input_port.name),
        (receiver.name, receiver.input_port.name),
    }


def test_feedback_mapping_projection_is_rejected_until_delay_state_is_modeled():
    composition, _, receiver, projection = _projection_model(feedback=True)

    diagnostic = _assert_projection_rejected(
        composition,
        receiver,
        projection,
        "unsupported feedback projection for batched v2",
    )
    assert diagnostic.detail == "source->receiver"


@pytest.mark.parametrize(
    "function, expected",
    [
        (pnl.MatrixTransform(normalize=True), "normalize=True"),
        (pnl.MatrixTransform(operation=pnl.L0), "operation='difference'"),
    ],
)
def test_unmodeled_mapping_projection_function_semantics_are_rejected(
    function,
    expected,
):
    composition, _, receiver, projection = _projection_model(function=function)

    diagnostic = _assert_projection_rejected(
        composition,
        receiver,
        projection,
        "unsupported MappingProjection function for batched v2",
    )
    assert "MatrixTransform(operation=DOT_PRODUCT, normalize=False)" in diagnostic.detail
    assert expected in diagnostic.detail


@pytest.mark.parametrize("parameter_name", ("weight", "exponent"))
def test_unmodeled_mapping_projection_weighting_is_rejected(parameter_name):
    composition, _, receiver, projection = _projection_model()
    getattr(projection.parameters, parameter_name).set(2.0, None)

    diagnostic = _assert_projection_rejected(
        composition,
        receiver,
        projection,
        "unsupported MappingProjection function for batched v2",
    )
    assert f"{parameter_name}=2.0" in diagnostic.detail


@pytest.mark.parametrize(
    "matrix, expected_dtype",
    [
        (np.asarray([[1 + 2j, 0], [0, 1 - 1j]]), "complex"),
        (np.asarray([[1.0, np.inf], [0.0, 1.0]]), "float"),
        (np.asarray([[1e40, 0.0], [0.0, 1.0]]), "float"),
    ],
    ids=("complex", "nonfinite", "float32-overflow"),
)
def test_unrepresentable_mapping_projection_matrix_is_rejected(
    matrix,
    expected_dtype,
):
    composition, _, receiver, projection = _projection_model(matrix=matrix)

    diagnostic = _assert_projection_rejected(
        composition,
        receiver,
        projection,
        "unsupported MappingProjection matrix for batched v2",
    )
    assert "finite real values representable as float32" in diagnostic.detail
    assert expected_dtype in diagnostic.detail
