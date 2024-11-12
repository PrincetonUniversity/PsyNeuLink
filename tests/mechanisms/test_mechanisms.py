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

    @pytest.mark.parametrize(
        'noise',
        [pnl.GaussianDistort, pnl.NormalDist]
    )
    def test_noise_assignment_equivalence(self, noise):
        t1 = pnl.TransferMechanism(name='t1', input_shapes=2, noise=noise())
        t2 = pnl.TransferMechanism(name='t2', input_shapes=2)
        t2.integrator_function.parameters.noise.set(noise())

        t1.integrator_function.noise.seed.base = 0
        t2.integrator_function.noise.base.seed = 0

        for _ in range(5):
            np.testing.assert_equal(t1.execute([1, 1]), t2.execute([1, 1]))

    @pytest.mark.parametrize(
        'noise, included_parameter_ports, excluded_parameter_ports, noise_statefulness',
        [
            (0, ['noise'], ['seed'], True),
            ([0], ['noise'], ['seed'], True),
            ([0, 0], ['noise'], ['seed'], True),
            ([0, pnl.NormalDist()], [], ['noise', 'seed'], False),
            (pnl.NormalDist, ['seed'], ['noise'], False),
            ([pnl.NormalDist(), pnl.NormalDist()], [], ['noise', 'seed'], False),
        ]
    )
    def test_numeric_noise_specifications(
        self,
        noise,
        included_parameter_ports,
        excluded_parameter_ports,
        noise_statefulness
    ):
        try:
            size = len(noise)
        except TypeError:
            size = 1

        t = pnl.TransferMechanism(input_shapes=size, noise=noise)

        assert all(p in t.parameter_ports for p in included_parameter_ports)
        assert all(p not in t.parameter_ports for p in excluded_parameter_ports)

        assert t.parameters.noise.stateful is noise_statefulness

    @pytest.mark.parametrize(
        'noise',
        [
            [0, pnl.NormalDist()],
            pnl.NormalDist,
            [pnl.NormalDist(), pnl.NormalDist()]
        ]
    )
    def test_noise_change_warning_to_numeric(self, noise):
        try:
            size = len(noise)
        except TypeError:
            size = 1

        t = pnl.TransferMechanism(input_shapes=size, noise=noise)

        with pytest.warns(
            UserWarning,
            match='Setting noise to a numeric value after instantiation.*'
        ):
            t.parameters.noise.set(0)

    @pytest.mark.parametrize(
        'noise',
        [
            0,
            [0],
            [0, 0],
        ]
    )
    def test_noise_change_warning_to_function(self, noise):
        try:
            size = len(noise)
        except TypeError:
            size = 1

        t = pnl.TransferMechanism(input_shapes=size, noise=noise)

        with pytest.warns(
            UserWarning,
            match='Setting noise to a value containing functions after instantiation.*'
        ):
            t.parameters.noise.set(pnl.NormalDist)


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
        A = pnl.IntegratorMechanism(name='A', function=pnl.DriftDiffusionIntegrator(time_step_size=1.0))

        # Execute A twice
        #  [0] saves decision variable only (not time)
        original_output = [A.execute(1.0), A.execute(1.0)]

        # SAVING STATE  - - - - - - - - - - - - - - - - - - - - - - - - -
        reset_stateful_functions_to = {}
        for attr in A.function.stateful_attributes:
            reset_stateful_functions_to[attr] = getattr(A.function, attr)

        print(reset_stateful_functions_to)
        # Execute A twice AFTER saving the state so that it continues accumulating.
        # We expect the next two outputs to repeat once we reset the state b/c we will return it to the current state
        output_after_saving_state = [A.execute(1.0), A.execute(1.0)]

        # RESETTING STATE - - - - - - - - - - - - - - - - - - - - - - - -
        A.reset(**reset_stateful_functions_to)

        # We expect these results to match the results from immediately after saving the state
        output_after_reinitialization = [A.execute(1.0), A.execute(1.0)]

        np.testing.assert_allclose(output_after_saving_state, output_after_reinitialization)
        np.testing.assert_allclose(original_output, np.array([[[1.0], [1.0]], [[2.0], [2.0]]]))
        np.testing.assert_allclose(output_after_reinitialization, np.array([[[3.0], [3.0]], [[4.0], [4.0]]]))

    def test_reset_state_transfer_mechanism(self):
        A = pnl.TransferMechanism(name='A', integrator_mode=True)

        # Execute A twice
        original_output = [A.execute(1.0), A.execute(1.0)]

        # SAVING STATE  - - - - - - - - - - - - - - - - - - - - - - - - -
        reset_stateful_functions_to = {}

        for attr in A.integrator_function.stateful_attributes:
            reset_stateful_functions_to[attr] = getattr(A.integrator_function, attr)

        # Execute A twice AFTER saving the state so that it continues accumulating.
        # We expect the next two outputs to repeat once we reset the state b/c we will return it to the current state
        output_after_saving_state = [A.execute(1.0), A.execute(1.0)]

        # RESETTING STATE - - - - - - - - - - - - - - - - - - - - - - - -
        A.reset(**reset_stateful_functions_to)

        # We expect these results to match the results from immediately after saving the state
        output_after_reinitialization = [A.execute(1.0), A.execute(1.0)]

        np.testing.assert_allclose(output_after_saving_state, output_after_reinitialization)
        np.testing.assert_allclose(original_output, [np.array([[0.5]]), np.array([[0.75]])])
        np.testing.assert_allclose(output_after_reinitialization, [np.array([[0.875]]), np.array([[0.9375]])])
