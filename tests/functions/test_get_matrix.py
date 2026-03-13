import psyneulink as pnl

import pytest
import numpy as np


@pytest.mark.parametrize(
    "initializer",
    [
        pnl.KaimingMatrix(),
        pnl.RandomMatrix(),
        pnl.XavierMatrix(),
        pnl.OrthogonalMatrix(),
        pnl.RANDOM_CONNECTIVITY_MATRIX,
        pnl.KAIMING_MATRIX,
        pnl.XAVIER_MATRIX,
        pnl.ORTHOGONAL_MATRIX,
    ],
)
def test_matrix_initializer_in_mapping_projection(initializer):

    sender = pnl.ProcessingMechanism(input_shapes=3)
    receiver = pnl.ProcessingMechanism(input_shapes=5)

    proj = pnl.MappingProjection(
        sender=sender,
        receiver=receiver,
        matrix=initializer
    )

    comp = pnl.Composition(pathways=[[sender, proj, receiver]])

    # run once to instantiate projection
    comp.run(inputs={sender: [[1, 2, 3]]})

    mat = proj.parameters.matrix.get(None)

    # correct shape
    assert mat.shape == (3, 5)

    # ensure matrix was actually initialized
    assert not np.all(mat == 0)



def test_kaiming_matrix_fan_direction_end_to_end():

    # fan="in": should depend on sender size
    sender_small_in = pnl.TransferMechanism(input_shapes=2)
    receiver_large_in = pnl.TransferMechanism(input_shapes=10)

    sender_large_in = pnl.TransferMechanism(input_shapes=10)
    receiver_small_in = pnl.TransferMechanism(input_shapes=2)

    proj_small_large_in = pnl.MappingProjection(
        sender=sender_small_in,
        receiver=receiver_large_in,
        matrix=pnl.KaimingMatrix(),
    )
    proj_large_small_in = pnl.MappingProjection(
        sender=sender_large_in,
        receiver=receiver_small_in,
        matrix=pnl.KaimingMatrix(),
    )

    comp_small_large_in = pnl.Composition(
        pathways=[[sender_small_in, proj_small_large_in, receiver_large_in]]
    )
    comp_large_small_in = pnl.Composition(
        pathways=[[sender_large_in, proj_large_small_in, receiver_small_in]]
    )

    comp_small_large_in.run(inputs={sender_small_in: [[1.0, 1.0]]})
    comp_large_small_in.run(inputs={sender_large_in: [np.ones(10).tolist()]})

    mat_small_large_in = proj_small_large_in.parameters.matrix.get(None)
    mat_large_small_in = proj_large_small_in.parameters.matrix.get(None)

    assert mat_small_large_in.shape == (2, 10)
    assert mat_large_small_in.shape == (10, 2)

    # larger sender -> smaller std
    assert np.std(mat_large_small_in) < np.std(mat_small_large_in)

    # fan="out": should depend on receiver size
    sender_small_out = pnl.TransferMechanism(input_shapes=2)
    receiver_large_out = pnl.TransferMechanism(input_shapes=10)

    sender_large_out = pnl.TransferMechanism(input_shapes=10)
    receiver_small_out = pnl.TransferMechanism(input_shapes=2)

    proj_small_large_out = pnl.MappingProjection(
        sender=sender_small_out,
        receiver=receiver_large_out,
        matrix=pnl.KaimingMatrix(distribution="normal", fan="out", gain=2.0),
    )
    proj_large_small_out = pnl.MappingProjection(
        sender=sender_large_out,
        receiver=receiver_small_out,
        matrix=pnl.KaimingMatrix(distribution="normal", fan="out", gain=2.0),
    )

    comp_small_large_out = pnl.Composition(
        pathways=[[sender_small_out, proj_small_large_out, receiver_large_out]]
    )
    comp_large_small_out = pnl.Composition(
        pathways=[[sender_large_out, proj_large_small_out, receiver_small_out]]
    )

    comp_small_large_out.run(inputs={sender_small_out: [[1.0, 1.0]]})
    comp_large_small_out.run(inputs={sender_large_out: [np.ones(10).tolist()]})

    mat_small_large_out = proj_small_large_out.parameters.matrix.get(None)
    mat_large_small_out = proj_large_small_out.parameters.matrix.get(None)

    assert mat_small_large_out.shape == (2, 10)
    assert mat_large_small_out.shape == (10, 2)

    # larger receiver -> smaller std
    assert np.std(mat_small_large_out) < np.std(mat_large_small_out)
