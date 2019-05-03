import numpy as np

from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import DriftDiffusionIntegrator, \
    IntegratorFunction
from psyneulink.core.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.process import Process
from psyneulink.core.components.system import System
from psyneulink.core.scheduling.condition import AtTrial, Never

class TestReinitializeValues:

    def test_reinitialize_one_mechanism_default(self):
        A = TransferMechanism(name='A')
        B = TransferMechanism(
            name='B',
            integrator_mode=True,
            integration_rate=0.5
        )
        C = TransferMechanism(name='C')

        abc_process = Process(pathway=[A, B, C])

        abc_system = System(processes=[abc_process])

        C.log.set_log_conditions('value')

        abc_system.run(inputs={A: [1.0]}, num_trials=5)

        # Trial 0: 0.5, Trial 1: 0.75, Trial 2: 0.5, Trial 3: 0.75. Trial 4: 0.875
        assert np.allclose(
            C.log.nparray_dictionary('value')[abc_system.default_execution_id]['value'],
            [
                [np.array([0.5])],
                [np.array([0.5])],
                [np.array([0.5])],
                [np.array([0.5])],
                [np.array([0.5])]
            ]
        )

    def test_reinitialize_one_mechanism_at_trial_2_condition(self):
        A = TransferMechanism(name='A')
        B = TransferMechanism(
            name='B',
            integrator_mode=True,
            integration_rate=0.5
        )
        C = TransferMechanism(name='C')

        abc_process = Process(pathway=[A, B, C])
        abc_system = System(processes=[abc_process])

        # Set reinitialization condition
        B.reinitialize_when = AtTrial(2)

        C.log.set_log_conditions('value')

        abc_system.run(
            inputs={A: [1.0]},
            reinitialize_values={B: [0.]},
            num_trials=5
        )

        # Trial 0: 0.5, Trial 1: 0.75, Trial 2: 0.5, Trial 3: 0.75. Trial 4: 0.875
        assert np.allclose(
            C.log.nparray_dictionary('value')[abc_system.default_execution_id]['value'],
            [
                [np.array([0.5])],
                [np.array([0.75])],
                [np.array([0.5])],
                [np.array([0.75])],
                [np.array([0.875])]
            ]
        )

    def test_reset_state_integrator_mechanism(self):
        A = IntegratorMechanism(name='A', function=DriftDiffusionIntegrator())

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
        A = TransferMechanism(name='A', integrator_mode=True)

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

    def test_save_state_before_simulations(self):

        A = TransferMechanism(
            name='A',
            integrator_mode=True,
            integration_rate=0.2
        )

        B = IntegratorMechanism(name='B', function=DriftDiffusionIntegrator(rate=0.1))
        C = TransferMechanism(name='C')

        P = Process(pathway=[A, B, C])
        S = System(
            processes=[P],
            reinitialize_mechanisms_when=Never()
        )

        S.run(inputs={A: [[1.0], [1.0]]})

        run_1_values = [
            A.parameters.value.get(S),
            B.parameters.value.get(S)[0],
            C.parameters.value.get(S)
        ]

        # "Save state" code from EVCaux

        # Get any values that need to be reinitialized for each run
        reinitialization_values = {}
        for mechanism in S.stateful_mechanisms:
            # "save" the current state of each stateful mechanism by storing the values of each of its stateful
            # attributes in the reinitialization_values dictionary; this gets passed into run and used to call
            # the reinitialize method on each stateful mechanism.
            reinitialization_value = []

            if isinstance(mechanism.function, IntegratorFunction):
                for attr in mechanism.function.stateful_attributes:
                    reinitialization_value.append(getattr(mechanism.function.parameters, attr).get(S))
            elif hasattr(mechanism, "integrator_function"):
                if isinstance(mechanism.integrator_function, IntegratorFunction):
                    for attr in mechanism.integrator_function.stateful_attributes:
                        reinitialization_value.append(getattr(mechanism.integrator_function.parameters, attr).get(S))

            reinitialization_values[mechanism] = reinitialization_value

        # Allow values to continue accumulating so that we can set them back to the saved state
        S.run(inputs={A: [[1.0], [1.0]]})

        run_2_values = [A.parameters.value.get(S),
                        B.parameters.value.get(S)[0],
                        C.parameters.value.get(S)]

        S.run(
            inputs={A: [[1.0], [1.0]]},
            reinitialize_values=reinitialization_values
        )

        run_3_values = [A.parameters.value.get(S),
                        B.parameters.value.get(S)[0],
                        C.parameters.value.get(S)]

        assert np.allclose(run_2_values, run_3_values)
        assert np.allclose(run_1_values, [np.array([[0.36]]), np.array([[0.056]]), np.array([[0.056]])])
        assert np.allclose(run_2_values, [np.array([[0.5904]]), np.array([[0.16384]]), np.array([[0.16384]])])
