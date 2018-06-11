import numpy as np

from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.process import Process
from psyneulink.components.system import System
from psyneulink.scheduling.condition import AtTrial
from psyneulink.components.states.outputstate import OutputState
from psyneulink.components.functions.function import AdaptiveIntegrator, DriftDiffusionIntegrator
from psyneulink.components.mechanisms.processing.integratormechanism import IntegratorMechanism

class TestReinitializeValues:

    def test_reinitialize_one_mechanism(self):
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B',
                              integrator_mode=True,
                              smoothing_factor=0.5)
        print(B.auto_dependent)
        C = TransferMechanism(name='C')

        abc_process = Process(pathway=[A, B, C])

        abc_system = System(processes=[abc_process])

        B.reinitialize_when = AtTrial(2)
        C.log.set_log_conditions('value')

        reinitialize_values = [3.0]
        B.reinitialize(*reinitialize_values)
        reinitialize_values = [5.0]
        B.integrator_function.reinitialize(*reinitialize_values)

        abc_system.run(inputs={A: [1.0]},
                       reinitialize_values={B: [0.]},
                       num_trials=5)

        # Trial 0: 0.5, Trial 1: 0.75, Trial 2: 0.5, Trial 3: 0.75. Trial 4: 0.875
        assert np.allclose(C.log.nparray_dictionary('value')['value'], [[np.array([0.5])],
                                                                        [np.array([0.75])],
                                                                        [np.array([0.5])],
                                                                        [np.array([0.75])],
                                                                        [np.array([0.875])]])

    def test_reset_state_with_mechanism_execute(self):
        A = IntegratorMechanism(name='A',
                                function=DriftDiffusionIntegrator())

        # Execute A twice
        original_output = [A.execute(1.0), A.execute(1.0)]

        # SAVING STATE  - - - - - - - - - - - - - - - - - - - - - - - - -
        reinitialize_values = []
        for attr in A.function_object._reinitialization_attributes:
            reinitialize_values.append(getattr(A.function_object, attr))

        # Execute A twice AFTER saving the state so that it continues accumulating.
        # We expect the next two outputs to repeat once we reset the state b/c we will return it to the current state
        output_after_saving_state = [A.execute(1.0), A.execute(1.0)]

        # RESETTING STATE - - - - - - - - - - - - - - - - - - - - - - - -
        A.reinitialize(*reinitialize_values)

        # We expect these results to match the results from immediately after saving the state
        output_after_reinitialization = [A.execute(1.0), A.execute(1.0)]

        assert np.allclose(output_after_saving_state, output_after_reinitialization)
        assert np.allclose(original_output, [np.array([[1.0]]), np.array([[2.0]])])
        assert np.allclose(output_after_reinitialization, [np.array([[3.0]]), np.array([[4.0]])])
