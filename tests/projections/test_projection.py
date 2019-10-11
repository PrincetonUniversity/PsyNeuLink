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
