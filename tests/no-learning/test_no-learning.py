import logging

import numpy as np

from psyneulink.core.compositions.composition import Composition
from psyneulink.core.components.functions.transferfunctions import Logistic
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.process import Process
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.globals.keywords import SOFT_CLAMP
from psyneulink.core.globals.preferences.basepreferenceset import REPORT_OUTPUT_PREF, VERBOSE_PREF


logger = logging.getLogger(__name__)


class TestNoLearning:

    def test_multilayer(self):
        Input_Layer = TransferMechanism(
            name='Input Layer',
            function=Logistic,
            default_variable=np.zeros((2,)),
        )

        Hidden_Layer_1 = TransferMechanism(
            name='Hidden Layer_1',
            function=Logistic(),
            default_variable=np.zeros((5,)),
        )

        Hidden_Layer_2 = TransferMechanism(
            name='Hidden Layer_2',
            function=Logistic(),
            default_variable=[0, 0, 0, 0],
        )

        Output_Layer = TransferMechanism(
            name='Output Layer',
            function=Logistic,
            default_variable=[0, 0, 0],
        )

        Input_Weights_matrix = (np.arange(2 * 5).reshape((2, 5)) + 1) / (2 * 5)

        # TEST PROCESS.LEARNING WITH:
        # CREATION OF FREE STANDING PROJECTIONS THAT HAVE NO LEARNING (Input_Weights, Middle_Weights and Output_Weights)
        # INLINE CREATION OF PROJECTIONS (Input_Weights, Middle_Weights and Output_Weights)
        # NO EXPLICIT CREATION OF PROJECTIONS (Input_Weights, Middle_Weights and Output_Weights)

        # This projection will be used by the process below by referencing it in the process' pathway;
        #    note: sender and receiver args don't need to be specified
        Input_Weights = MappingProjection(
            name='Input Weights',
            matrix=Input_Weights_matrix,
        )

        c = Composition()
        learning_pathway = c.add_backpropagation_learning_pathway(pathway=[Input_Layer,
                                                                           Input_Weights,
                                                                           Hidden_Layer_1,
                                                                           Hidden_Layer_2,
                                                                           Output_Layer])
        target = learning_pathway.target
        stim_list = {Input_Layer: [[-1, 30]],
                     target: [0, 0, 1]}
        c.run(num_trials=10, inputs=stim_list, clamp_input=SOFT_CLAMP)

        expected_Output_Layer_output = [np.array([0.97988347, 0.97988347, 0.97988347])]

        np.testing.assert_allclose(expected_Output_Layer_output, Output_Layer.get_output_values(c))
