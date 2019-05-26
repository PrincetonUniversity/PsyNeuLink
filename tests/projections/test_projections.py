import numpy as np
import psyneulink as pnl
import pytest


@pytest.mark.parametrize(
    'projection_type, sender_variable, receiver_variable, projection_value, function_value',
    [
        (pnl.MappingProjection, [0, 0, 0], [0, 0], np.array([0, 0]), np.array([0, 0]))
    ]
)
def test_value_shapes_with_matrix(projection_type, sender_variable, receiver_variable, projection_value, function_value):
    A = pnl.TransferMechanism(default_variable=sender_variable)
    B = pnl.TransferMechanism(default_variable=receiver_variable)
    P = projection_type(sender=A, receiver=B)

    assert P.defaults.value.shape == projection_value.shape
    assert P.function.defaults.value.shape == function_value.shape

def test_for_use_and_suppression_of_identity_function():
    A = pnl.ProcessingMechanism(name='M1', size=2)
    B = pnl.ProcessingMechanism(name='M2', size=2)
    C = pnl.ProcessingMechanism(name='M2', size=3)
    comp = pnl.Composition()
    comp.add_linear_processing_pathway([A, B, C])
    P1=A.efferents[0]
    P2=B.efferents[0]
    assert P1.matrix == pnl.IDENTITY_MATRIX
    assert isinstance(P1.function, pnl.Identity)
    assert (P2.matrix == [[1.,1.,1.],[1.,1.,1.]]).all()
    assert isinstance(P2.function, pnl.LinearMatrix)
    P1.suppress_identity_function = True
    assert (P1.matrix == np.identity(2)).all()
    assert isinstance(P1.function, pnl.LinearMatrix)
    P1.suppress_identity_function = False
    assert P1.matrix == pnl.IDENTITY_MATRIX
    assert isinstance(P1.function, pnl.Identity)

    A = pnl.ProcessingMechanism(name='M1', size=2)
    B = pnl.ProcessingMechanism(name='M2', size=2)
    P = pnl.MappingProjection(suppress_identity_function=True)
    comp.add_linear_processing_pathway([A, P, B])
    assert (P.matrix == np.identity(2)).all()
    assert isinstance(P.function, pnl.LinearMatrix)