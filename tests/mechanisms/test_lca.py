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

class TestLCAReinitialize:

    def test_reinitialize_run(self):

        L = LCA(name="L",
                function=Linear,
                initial_value=0.5,
                integrator_mode=True,
                leak=0.1,
                competition=0,
                self_excitation=1.0,
                time_step_size=1.0,
                noise=0.0)
        P = Process(name="P",
                    pathway=[L])
        S = System(name="S",
                   processes=[P])

        assert np.allclose(L.previous_value, 0.5)
        assert np.allclose(L.initial_value, 0.5)
        assert np.allclose(L.integrator_function.initializer, 0.5)

        S.run(inputs={L: 1.0},
              num_trials=2,
              initialize=True,
              initial_values={L: 0.0})

        # Integrator fn: previous_value + (rate*previous_value + new_value)*time_step_size + noise*(time_step_size**0.5)

        # Trial 1    |   variable = 1.0 + 0.0
        # integration: 0.5 + (0.1*0.5 + 1.0)*1.0 + 0.0 = 1.55
        # linear fn: 1.55*1.0 = 1.55
        # Trial 2    |   variable = 1.0 + 1.55
        # integration: 1.55 + (0.1*1.55 + 2.55)*1.0 + 0.0 = 4.255
        #  linear fn: 4.255*1.0 = 4.255
        assert np.allclose(L.previous_value, 4.255)
        assert np.allclose(L.initial_value, 0.5)
        assert np.allclose(L.integrator_function.initializer, 0.5)

        L.integrator_function.reinitialize(0.9)

        assert np.allclose(L.previous_value, 0.9)
        assert np.allclose(L.initial_value, 0.5)
        assert np.allclose(L.integrator_function.initializer, 0.9)
        assert np.allclose(L.value, 4.255)

        L.reinitialize(0.5)

        assert np.allclose(L.previous_value, 0.5)
        assert np.allclose(L.initial_value, 0.5)
        assert np.allclose(L.integrator_function.initializer, 0.5)
        assert np.allclose(L.value, 0.5)

        S.run(inputs={L: 1.0},
              num_trials=2)
        # Trial 3    |   variable = 1.0 + 0.5
        # integration: 0.5 + (0.1*0.5 + 1.5)*1.0 + 0.0 = 2.05
        # linear fn: 2.05*1.0 = 2.05
        # Trial 4    |   variable = 1.0 + 2.05
        # integration: 2.05 + (0.1*2.05 + 3.05)*1.0 + 0.0 = 5.305
        #  linear fn: 5.305*1.0 = 5.305
        assert np.allclose(L.previous_value, 5.305)
        assert np.allclose(L.initial_value, 0.5)
        assert np.allclose(L.integrator_function.initializer, 0.5)

class TestClip:

    def test_clip_float(self):
        L = LCA(clip=[-2.0, 2.0],
                function=Linear,
                integrator_mode=False)
        assert np.allclose(L.execute(3.0), 2.0)
        assert np.allclose(L.execute(-3.0), -2.0)

    def test_clip_array(self):
        L = LCA(default_variable=[[0.0, 0.0, 0.0]],
                clip=[-2.0, 2.0],
                function=Linear,
                integrator_mode=False)
        assert np.allclose(L.execute([3.0, 0.0, -3.0]), [2.0, 0.0, -2.0])

    def test_clip_2d_array(self):
        L = LCA(default_variable=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                clip=[-2.0, 2.0],
                function=Linear,
                integrator_mode=False)
        assert np.allclose(L.execute([[-5.0, -1.0, 5.0], [5.0, -5.0, 1.0], [1.0, 5.0, 5.0]]),
                           [[-2.0, -1.0, 2.0], [2.0, -2.0, 1.0], [1.0, 2.0, 2.0]])