import numpy as np

from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.process import Process
from psyneulink.components.system import System
from psyneulink.scheduling.condition import AtTrial

class TestReinitializeValues:

    def test_reinitialize_one_mechanism(self):
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B',
                              integrator_mode=True,
                              smoothing_factor=0.5)
        C = TransferMechanism(name='C')

        abc_process = Process(pathway=[A, B, C])

        abc_system = System(processes=[abc_process])

        B.reinitialize_when = AtTrial(2)
        C.log.set_log_conditions('value')

        abc_system.run(inputs={A: [1.0]},
                       reinitialize_values={B: [0.]},
                       num_trials=5)

        # Trial 0: 0.5, Trial 1: 0.75, Trial 2: 0.5, Trial 3: 0.75. Trial 4: 0.875
        assert np.allclose(C.log.nparray_dictionary('value')['value'], [[np.array([0.5])],
                                                                        [np.array([0.75])],
                                                                        [np.array([0.5])],
                                                                        [np.array([0.75])],
                                                                        [np.array([0.875])]])