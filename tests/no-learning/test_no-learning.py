import logging
import numpy

from PsyNeuLink.Components.Functions.Function import Logistic
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCMechanism import EVCMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.LearningProjection import LearningProjection
from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
from PsyNeuLink.Components.System import system
from PsyNeuLink.Globals.TimeScale import TimeScale
from PsyNeuLink.scheduling.Scheduler import Scheduler
from PsyNeuLink.scheduling.condition import AfterNCalls

logger = logging.getLogger(__name__)

class TestNoLearning:
    def test_multilayer(self):

        # from PsyNeuLink.Globals.Run import run, construct_inputs

        Input_Layer = TransferMechanism(name='Input Layer',
                               function=Logistic,
                               default_input_value = np.zeros((2,)))

        Hidden_Layer_1 = TransferMechanism(name='Hidden Layer_1',
                                  function=Logistic(),
                                  default_input_value = np.zeros((5,)))

        Hidden_Layer_2 = TransferMechanism(name='Hidden Layer_2',
                                  function=Logistic(),
                                  default_input_value = [0,0,0,0])

        Output_Layer = TransferMechanism(name='Output Layer',
                                function=Logistic,
                                default_input_value = [0,0,0])

        random_weight_matrix = lambda sender, receiver : random_matrix(sender, receiver, .2, -.1)

        Input_Weights_matrix = (np.arange(2*5).reshape((2, 5)) + 1)/(2*5)
        Middle_Weights_matrix = (np.arange(5*4).reshape((5, 4)) + 1)/(5*4)
        Output_Weights_matrix = (np.arange(4*3).reshape((4, 3)) + 1)/(4*3)


        # TEST PROCESS.LEARNING WITH:
        # CREATION OF FREE STANDING PROJECTIONS THAT HAVE NO LEARNING (Input_Weights, Middle_Weights and Output_Weights)
        # INLINE CREATION OF PROJECTIONS (Input_Weights, Middle_Weights and Output_Weights)
        # NO EXPLICIT CREATION OF PROJECTIONS (Input_Weights, Middle_Weights and Output_Weights)

        # This projection will be used by the process below by referencing it in the process' pathway;
        #    note: sender and receiver args don't need to be specified
        Input_Weights = MappingProjection(name='Input Weights',
                                # sender=Input_Layer,
                                # receiver=Hidden_Layer_1,
                                # matrix=(random_weight_matrix, LearningProjection()),
                                # matrix=random_weight_matrix,
                                # matrix=(RANDOM_CONNECTIVITY_MATRIX, LearningProjection()),
                                # matrix=RANDOM_CONNECTIVITY_MATRIX
                                # matrix=FULL_CONNECTIVITY_MATRIX,
                                matrix=Input_Weights_matrix
                                )

        # This projection will be used by the process below by assigning its sender and receiver args
        #    to mechanismss in the pathway
        Middle_Weights = MappingProjection(name='Middle Weights',
                                 sender=Hidden_Layer_1,
                                 receiver=Hidden_Layer_2,
                                 # matrix=(FULL_CONNECTIVITY_MATRIX, LearningProjection())
                                 # matrix=FULL_CONNECTIVITY_MATRIX
                                 matrix=Middle_Weights_matrix
                                 )

        # Commented lines in this projection illustrate variety of ways in which matrix and learning signals can be specified
        Output_Weights = MappingProjection(name='Output Weights',
                                 sender=Hidden_Layer_2,
                                 receiver=Output_Layer,
                                 # matrix=random_weight_matrix,
                                 # matrix=(random_weight_matrix, LEARNING_PROJECTION),
                                 # matrix=(random_weight_matrix, LearningProjection),
                                 # matrix=(random_weight_matrix, LearningProjection()),
                                 # matrix=(RANDOM_CONNECTIVITY_MATRIX),
                                 # matrix=(RANDOM_CONNECTIVITY_MATRIX, LearningProjection),
                                 # matrix=(FULL_CONNECTIVITY_MATRIX, LearningProjection)
                                 # matrix=FULL_CONNECTIVITY_MATRIX
                                 matrix=Output_Weights_matrix
                                 )

        z = process(default_input_value=[0, 0],
                    pathway=[Input_Layer,
                                   # The following reference to Input_Weights is needed to use it in the pathway
                                   #    since it's sender and receiver args are not specified in its declaration above
                                   Input_Weights,
                                   Hidden_Layer_1,
                                   # No projection specification is needed here since the sender arg for Middle_Weights
                                   #    is Hidden_Layer_1 and its receiver arg is Hidden_Layer_2
                                   # Middle_Weights,
                                   Hidden_Layer_2,
                                   # Output_Weights does not need to be listed for the same reason as Middle_Weights
                                   # If Middle_Weights and/or Output_Weights is not declared above, then the process
                                   #    will assign a default for missing projection
                                   # Output_Weights,
                                   Output_Layer],
                    clamp_input=SOFT_CLAMP,
                    target=[0,0,1],
                    prefs={VERBOSE_PREF: False,
                           REPORT_OUTPUT_PREF: True})

        x = system(processes=[z],

        )

        x.reportOutputPref = True
        composition = x

        stim_list = {Input_Layer:[[-1, 30]]}
        # x.show_graph_with_learning()

        term_conds = {TimeScale.TRIAL: AfterNCalls(Output_Layer, 1)}
        sched = Scheduler(system=x)
        x.scheduler = sched

        results = x.run(num_executions=10,
                        # inputs=stim_list,
                        # inputs=[[-1, 30],[2, 10]],
                        # targets=[[0, 0, 1],[0, 0, 1]],
                        inputs=stim_list,
                        termination_conditions=term_conds
        )

        expected_Output_Layer_output = [numpy.array([ 0.8344837, 0.87072018, 0.89997433]), numpy.array(0.8683927358195268), numpy.array(0.0007175454706347559)]

        assert len(expected_Output_Layer_output) == len(Output_Layer.outputValue) == 3
        for i in range(len(expected_Output_Layer_output)):
            assert numpy.allclose(expected_Output_Layer_output[i], Output_Layer.outputValue[i])
