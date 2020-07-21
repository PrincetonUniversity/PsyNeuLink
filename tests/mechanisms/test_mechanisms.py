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


class TestMechanismFunctionParameters:
    f = pnl.Linear()
    i = pnl.SimpleIntegrator()
    mech_1 = pnl.TransferMechanism(function=f, integrator_function=i)
    mech_2 = pnl.TransferMechanism(function=f, integrator_function=i)
    integrator_mechanism = pnl.IntegratorMechanism(function=i)

    @pytest.mark.parametrize(
        "f, g",
        [
            pytest.param(
                mech_1.defaults.function,
                mech_2.defaults.function,
                id="function_defaults",
            ),
            pytest.param(
                mech_1.defaults.function,
                mech_1.parameters.function.get(),
                id="function_default-and-value",
            ),
            pytest.param(
                mech_1.defaults.function,
                mech_2.parameters.function.get(),
                id="function_default-and-other-value",
            ),
            pytest.param(
                mech_1.defaults.integrator_function,
                mech_2.defaults.integrator_function,
                id="integrator_function_defaults",
            ),
            pytest.param(
                mech_1.defaults.integrator_function,
                mech_1.parameters.integrator_function.get(),
                id="integrator_function_default-and-value",
            ),
            pytest.param(
                mech_1.defaults.integrator_function,
                mech_2.parameters.integrator_function.get(),
                id="integrator_function_default-and-other-value",
            ),
        ],
    )
    def test_function_parameter_distinctness(self, f, g):
        assert f is not g

    @pytest.mark.parametrize(
        "f, owner",
        [
            pytest.param(
                mech_1.parameters.function.get(),
                mech_1,
                id='function'
            ),
            pytest.param(
                integrator_mechanism.class_defaults.function,
                integrator_mechanism.class_parameters.function,
                id="class_default_function"
            ),
            pytest.param(
                mech_1.defaults.function,
                mech_1.parameters.function,
                id="default_function"
            ),
            pytest.param(
                mech_1.parameters.termination_measure.get(),
                mech_1,
                id='termination_measure'
            ),
            pytest.param(
                mech_1.class_defaults.termination_measure,
                mech_1.class_parameters.termination_measure,
                id="class_default_termination_measure"
            ),
            pytest.param(
                mech_1.defaults.termination_measure,
                mech_1.parameters.termination_measure,
                id="default_termination_measure"
            ),
        ]
    )
    def test_function_parameter_ownership(self, f, owner):
        assert f.owner is owner

    @pytest.mark.parametrize(
        'param_name, function',
        [
            ('function', f),
            ('integrator_function', i),
        ]
    )
    def test_function_parameter_assignment(self, param_name, function):
        # mech_1 should use the exact instances, mech_2 should have copies
        assert getattr(self.mech_1.parameters, param_name).get() is function
        assert getattr(self.mech_2.parameters, param_name).get() is not function

class TestResetValues:

    def test_reset_state_integrator_mechanism(self):
        A = pnl.IntegratorMechanism(name='A', function=pnl.DriftDiffusionIntegrator())

        # Execute A twice
        #  [0] saves decision variable only (not time)
        original_output = [A.execute(1.0)[0], A.execute(1.0)[0]]

        # SAVING STATE  - - - - - - - - - - - - - - - - - - - - - - - - -
        reset_stateful_functions_to = []
        for attr in A.function.stateful_attributes:
            reset_stateful_functions_to.append(getattr(A.function, attr))

        # Execute A twice AFTER saving the state so that it continues accumulating.
        # We expect the next two outputs to repeat once we reset the state b/c we will return it to the current state
        output_after_saving_state = [A.execute(1.0)[0], A.execute(1.0)[0]]

        # RESETTING STATE - - - - - - - - - - - - - - - - - - - - - - - -
        A.reset(*reset_stateful_functions_to)

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
        reset_stateful_functions_to = []

        for attr in A.integrator_function.stateful_attributes:
            reset_stateful_functions_to.append(getattr(A.integrator_function, attr))

        # Execute A twice AFTER saving the state so that it continues accumulating.
        # We expect the next two outputs to repeat once we reset the state b/c we will return it to the current state
        output_after_saving_state = [A.execute(1.0), A.execute(1.0)]

        # RESETTING STATE - - - - - - - - - - - - - - - - - - - - - - - -
        A.reset(*reset_stateful_functions_to)

        # We expect these results to match the results from immediately after saving the state
        output_after_reinitialization = [A.execute(1.0), A.execute(1.0)]

        assert np.allclose(output_after_saving_state, output_after_reinitialization)
        assert np.allclose(original_output, [np.array([[0.5]]), np.array([[0.75]])])
        assert np.allclose(output_after_reinitialization, [np.array([[0.875]]), np.array([[0.9375]])])
