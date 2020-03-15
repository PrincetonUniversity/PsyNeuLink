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

        assert T.input_port.defaults.variable == result_variable[0]
        assert T.input_port.defaults.value == result_variable[0]

        assert T.input_port.function.defaults.variable == result_variable[0]
        assert T.input_port.function.defaults.value == result_variable[0]

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


class TestReinitializeValues:

    def test_reset_state_integrator_mechanism(self):
        A = pnl.IntegratorMechanism(name='A', function=pnl.DriftDiffusionIntegrator())

        # Execute A twice
        #  [0] saves decision variable only (not time)
        original_output = [A.execute(1.0)[0], A.execute(1.0)[0]]

        # SAVING STATE  - - - - - - - - - - - - - - - - - - - - - - - - -
        reinitialize_values = []
        for attr in A.function.stateful_attributes:
            reinitialize_values.append(getattr(A.function, attr))

        # Execute A twice AFTER saving the state so that it continues accumulating.
        # We expect the next two outputs to repeat once we reset the state b/c we will return it to the current state
        output_after_saving_state = [A.execute(1.0)[0], A.execute(1.0)[0]]

        # RESETTING STATE - - - - - - - - - - - - - - - - - - - - - - - -
        A.reinitialize(*reinitialize_values)

        # We expect these results to match the results from immediately after saving the state
        output_after_reinitialization = [A.execute(1.0)[0], A.execute(1.0)[0]]

        assert np.allclose(output_after_saving_state, output_after_reinitialization)
        assert np.allclose(original_output, [np.array([[1.0]]), np.array([[2.0]])])
        assert np.allclose(output_after_reinitialization, [np.array([[3.0]]), np.array([[4.0]])])

    def test_reset_state_transfer_mechanism(self):
        A = pnl.TransferMechanism(name='A', integrator_mode=True)

        # Execute A twice
        original_output = [A.execute(1.0), A.execute(1.0)]

        # SAVING STATE  - - - - - - - - - - - - - - - - - - - - - - - - -
        reinitialize_values = []

        for attr in A.integrator_function.stateful_attributes:
            reinitialize_values.append(getattr(A.integrator_function, attr))

        # Execute A twice AFTER saving the state so that it continues accumulating.
        # We expect the next two outputs to repeat once we reset the state b/c we will return it to the current state
        output_after_saving_state = [A.execute(1.0), A.execute(1.0)]

        # RESETTING STATE - - - - - - - - - - - - - - - - - - - - - - - -
        A.reinitialize(*reinitialize_values)

        # We expect these results to match the results from immediately after saving the state
        output_after_reinitialization = [A.execute(1.0), A.execute(1.0)]

        assert np.allclose(output_after_saving_state, output_after_reinitialization)
        assert np.allclose(original_output, [np.array([[0.5]]), np.array([[0.75]])])
        assert np.allclose(output_after_reinitialization, [np.array([[0.875]]), np.array([[0.9375]])])
