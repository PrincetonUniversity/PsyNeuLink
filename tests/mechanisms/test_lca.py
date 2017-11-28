import pytest
import numpy as np

from psyneulink.library.mechanisms.processing.transfer.lca import LCA
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.functions.function import Linear
from psyneulink.components.process import Process
from psyneulink.components.system import System

class TestLCA:
    def test_LCA_length_1(self):

        T = TransferMechanism(function=Linear(slope=1.0))
        L = LCA(function=Linear(slope=2.0),
                self_excitation=3.0,
                leak=0.5,
                competition=1.0,   #  competition does not matter because we only have one unit
                time_step_size=0.1)
        P = Process(pathway=[T, L])
        S = System(processes=[P])

        #  - - - - - - - Equations to be executed  - - - - - - -

        # new_transfer_input =
        # previous_transfer_input
        # + (leak * previous_transfer_input_1 + self_excitation * result1 + competition * result2 + outside_input1) * dt
        # + noise

        # result = new_transfer_input*2.0

        # recurrent_matrix = [[3.0]]

        #  - - - - - - - - - - - - - -  - - - - - - - - - - - -

        results=[]
        def record_execution():
            results.append(L.value[0][0])

        S.run(inputs={T: [1.0]},
              num_trials=3,
              call_after_trial=record_execution)

        # - - - - - - - TRIAL 1 - - - - - - -

        # new_transfer_input = 0.0 + ( 0.5 * 0.0 + 3.0 * 0.0 + 0.0 + 1.0)*0.1 + 0.0    =    0.1
        # f(new_transfer_input) = 0.1 * 2.0 = 0.2

        # - - - - - - - TRIAL 2 - - - - - - -

        # new_transfer_input = 0.1 + ( 0.5 * 0.1 + 3.0 * 0.2 + 0.0 + 1.0)*0.1 + 0.0    =    0.265
        # f(new_transfer_input) = 0.265 * 2.0 = 0.53

        # - - - - - - - TRIAL 3 - - - - - - -

        # new_transfer_input = 0.265 + ( 0.5 * 0.265 + 3.0 * 0.53 + 0.0 + 1.0)*0.1 + 0.0    =    0.53725
        # f(new_transfer_input) = 0.53725 * 2.0 = 1.0745

        assert np.allclose(results, [0.2, 0.53, 1.0745])


    def test_LCA_length_2(self):

        T = TransferMechanism(function=Linear(slope=1.0),
                              size=2)
        L = LCA(function=Linear(slope=2.0),
                size=2,
                self_excitation=3.0,
                leak=0.5,
                competition=1.0,
                time_step_size=0.1)
        P = Process(pathway=[T, L])
        S = System(processes=[P])

        #  - - - - - - - Equations to be executed  - - - - - - -

        # new_transfer_input =
        # previous_transfer_input
        # + (leak * previous_transfer_input_1 + self_excitation * result1 + competition * result2 + outside_input1) * dt
        # + noise

        # result = new_transfer_input*2.0

        # recurrent_matrix = [[3.0]]

        #  - - - - - - - - - - - - - -  - - - - - - - - - - - -

        results=[]
        def record_execution():
            results.append(L.value[0])

        S.run(inputs={T: [1.0, 2.0]},
              num_trials=3,
              call_after_trial=record_execution)

        # - - - - - - - TRIAL 1 - - - - - - -

        # new_transfer_input_1 = 0.0 + ( 0.5 * 0.0 + 3.0 * 0.0 - 1.0*0.0 + 1.0)*0.1 + 0.0    =    0.1
        # f(new_transfer_input_1) = 0.1 * 2.0 = 0.2

        # new_transfer_input_2 = 0.0 + ( 0.5 * 0.0 + 3.0 * 0.0 - 1.0*0.0 + 2.0)*0.1 + 0.0    =    0.2
        # f(new_transfer_input_2) = 0.2 * 2.0 = 0.4

        # - - - - - - - TRIAL 2 - - - - - - -

        # new_transfer_input = 0.1 + ( 0.5 * 0.1 + 3.0 * 0.2 - 1.0*0.4 + 1.0)*0.1 + 0.0    =    0.225
        # f(new_transfer_input) = 0.265 * 2.0 = 0.45

        # new_transfer_input_2 = 0.2 + ( 0.5 * 0.2 + 3.0 * 0.4 - 1.0*0.2 + 2.0)*0.1 + 0.0    =    0.51
        # f(new_transfer_input_2) = 0.1 * 2.0 = 1.02

        # - - - - - - - TRIAL 3 - - - - - - -

        # new_transfer_input = 0.225 + ( 0.5 * 0.225 + 3.0 * 0.45 - 1.0*1.02 + 1.0)*0.1 + 0.0    =    0.36925
        # f(new_transfer_input) = 0.36925 * 2.0 = 0.7385

        # new_transfer_input_2 = 0.51 + ( 0.5 * 0.51 + 3.0 * 1.02 - 1.0*0.45 + 2.0)*0.1 + 0.0    =    0.9965
        # f(new_transfer_input_2) = 0.9965 * 2.0 = 1.463

        assert np.allclose(results, [[0.2, 0.4], [0.45, 1.02], [0.7385, 1.993]])
