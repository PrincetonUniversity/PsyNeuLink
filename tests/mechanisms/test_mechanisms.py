import numpy as np
import psyneulink as pnl
import pytest


class TestMechanism:

    @pytest.mark.mechanism
    @pytest.mark.parametrize(
        'mechanism, default_variable, result_variable',
        [
            (pnl.TransferMechanism, [0], np.array([[0]])),
            (pnl.IntegratorMechanism, [0], np.array([[0]])),
        ]
    )
    def test_transfer_mech_instantiation(self, mechanism, default_variable, result_variable):
        T = mechanism(default_variable=default_variable)

        assert T.defaults.variable == result_variable
        assert T.defaults.value == result_variable

        assert T.function.defaults.variable == result_variable
        assert T.function.defaults.value == result_variable

        assert T.input_state.defaults.variable == result_variable[0]
        assert T.input_state.defaults.value == result_variable[0]

        assert T.input_state.function.defaults.variable == result_variable[0]
        assert T.input_state.function.defaults.value == result_variable[0]

    @pytest.mark.mechanism
    @pytest.mark.parametrize(
        'mechanism_type, default_variable, mechanism_value, function_value',
        [
            (pnl.ObjectiveMechanism, [0, 0, 0], np.array([[0, 0, 0]]), np.array([[0, 0, 0]]))
        ]
    )
    def test_value_shapes(self, mechanism_type, default_variable, mechanism_value, function_value):
        M = mechanism_type(default_variable=default_variable)

        assert M.defaults.value.shape == mechanism_value.shape
        assert M.function.defaults.value.shape == function_value.shape



