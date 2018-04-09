import numpy as np
import pytest

from psyneulink.components.component import ComponentError
from psyneulink.components.functions.function import FunctionError
from psyneulink.components.functions.function import ConstantIntegrator, Exponential, Linear, Logistic, Reduce, Reinforcement, SoftMax
from psyneulink.components.functions.function import ExponentialDist, GammaDist, NormalDist, UniformDist, WaldDist, UniformToNormalDist
from psyneulink.components.mechanisms.mechanism import MechanismError
from psyneulink.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.globals.keywords import INPUT_LABELS_DICT
from psyneulink.components.process import Process
from psyneulink.components.system import System

class TestMechanismInputLabels:
    def test_dict_of_floats(self):
        input_labels_dict = {"red": 1,
                             "green":0}
        M = ProcessingMechanism(params={INPUT_LABELS_DICT:input_labels_dict})
        P = Process(pathway=[M])
        S = System(processes=[P])

        S.run(inputs=['red', 'green', 'green', 'red'])
        assert np.allclose(S.results, [[[1.]], [[0.]], [[0.]], [[1.]]])

    def test_dict_of_arrays(self):
        input_labels_dict = {"red": [1, 0, 0],
                             "green": [0, 1, 0],
                             "blue": [0, 0, 1]}
        M = ProcessingMechanism(default_variable=[[0, 0, 0]],
                                params={INPUT_LABELS_DICT: input_labels_dict})
        P = Process(pathway=[M])
        S = System(processes=[P])

        S.run(inputs=['red', 'green', 'blue', 'red'])
        assert np.allclose(S.results, [[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]], [[1, 0, 0]]])

    def test_dict_of_2d_arrays(self):
        input_labels_dict = {"red": [[1, 0], [1, 0]],
                             "green": [[0, 1], [0, 1]],
                             "blue": [[0, 1], [1, 0]]}
        M = TransferMechanism(default_variable=[[0, 0], [0, 0]],
                                params={INPUT_LABELS_DICT: input_labels_dict})
        P = Process(pathway=[M])
        S = System(processes=[P])

        S.run(inputs=['red', 'green', 'blue'])
        assert np.allclose(S.results, [[[1, 0], [1, 0]], [[0, 1], [0, 1]], [[0, 1], [1, 0]]])

    def test_dict_of_dicts_1_input_state(self):
        input_labels_dict = {0: {"red": [1, 0],
                                 "green": [0, 1]}}

        M = TransferMechanism(default_variable=[[0, 0]],
                              params={INPUT_LABELS_DICT: input_labels_dict})
        P = Process(pathway=[M])
        S = System(processes=[P])

        S.run(inputs=[['red'], ['green'], ['green']])
        assert np.allclose(S.results, [[[1, 0]], [[0, 1]], [[0, 1]]])

    # class TestMechanismOutputLabels:
    def test_dict_of_dicts(self):
        input_labels_dict = {0: {"red": [1, 0],
                                 "green": [0, 1]},
                             1: {"red": [0, 1],
                                 "green": [1, 0]}}


        M = TransferMechanism(default_variable=[[0, 0], [0, 0]],
                              params={INPUT_LABELS_DICT: input_labels_dict})
        P = Process(pathway=[M])
        S = System(processes=[P])

        S.run(inputs=[['red', 'green'], ['green', 'red'], ['green', 'green']])
        assert np.allclose(S.results, [[[1, 0], [1, 0]], [[0, 1], [0, 1]], [[0, 1], [1, 0]]])

            # class TestTargetLabels:
# class TestMechanismOutputLabels: