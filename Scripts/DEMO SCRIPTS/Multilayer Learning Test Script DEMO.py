from PsyNeuLink.Components.Functions.Function import Logistic
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCMechanism import EVCMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.System import system
import matplotlib
matplotlib.use('TkAgg')


Input_Layer = TransferMechanism(name='Input Layer',
                       function=Logistic,
                       default_input_value = np.zeros((2,)))

Hidden_Layer_1 = TransferMechanism(name='Hidden Layer_1',
                          function=Logistic(),
                          default_input_value = np.zeros((5,)))

# #region
# Hidden_Layer_1.plot()
# #endregion

Hidden_Layer_2 = TransferMechanism(name='Hidden Layer_2',
                          function=Logistic(),
                          default_input_value = [0,0,0,0])

Output_Layer = TransferMechanism(name='Output Layer',
                        function=Logistic,
                        default_input_value = [0,0,0])


Input_Weights = MappingProjection(name='Input Weights',
                                  # matrix=(np.arange(2*5).reshape((2, 5)) + 1)/(2*5)
                                  # matrix=RANDOM_CONNECTIVITY_MATRIX
                                  matrix=FULL_CONNECTIVITY_MATRIX
                        )

# This projection will be used by the process below by assigning its sender and receiver args
#    to mechanisms in the pathway
Middle_Weights = MappingProjection(name='Middle Weights',
                                   matrix=(np.arange(5*4).reshape((5, 4)) + 1)/(5*4)
                         )

# Commented lines in this projection illustrate variety of ways in which matrix and learning signals can be specified
Output_Weights = MappingProjection(name='Output Weights',
                                   sender=Hidden_Layer_2,
                                   receiver=Output_Layer,
                                   matrix=(np.arange(4*3).reshape((4, 3)) + 1)/(4*3)
                         )

my_process = process(default_input_value=[0, 0],
                     pathway=[Input_Layer,
                                Input_Weights,
                              Hidden_Layer_1,
                                Middle_Weights,
                              Hidden_Layer_2,
                              Output_Layer],
                     clamp_input=SOFT_CLAMP,
                     learning=LEARNING,
                     learning_rate=1.0,
                     target=[0,0,1],
                     prefs={VERBOSE_PREF: False,
                            REPORT_OUTPUT_PREF: True})

#region
# # **************************************************
#
# def print_header():
#     print("\n\n**** TRIAL: ", CentralClock.trial)
#
# def show_target():
#
#     print ('\nOLD WEIGHTS: \n')
#     print ('- Input Weights: \n', Input_Weights.matrix)
#     print ('- Middle Weights: \n', Middle_Weights.matrix)
#     print ('- Output Weights: \n', Output_Weights.matrix)
#     print ('\nSTIMULI:\n\n- Input: {}\n- Target: {}\n'.format(my_system.input,
#                                                               my_system.targetInputStates[0].value))
#     print ('ACTIVITY FROM OLD WEIGHTS: \n')
#     print ('- Middle 1: \n', Hidden_Layer_1.value)
#     print ('- Middle 2: \n', Hidden_Layer_2.value)
#     print ('- Output:\n', Output_Layer.value)
#     # print ('MSE: \n', Output_Layer.output_values[0])
#
# # **************************************************
#endregion

stim_list = {Input_Layer:[[-1, 30]]}
target_list = {Output_Layer:[[0, 0, 1]]}

my_system = system(processes=[my_process],
                   targets=[0, 0, 1],
                   learning_rate=1.0)

my_system.reportOutputPref = True

# my_system.show_graph(direction='LR')
my_system.show_graph_with_learning(direction='LR')

results = my_system.run(num_executions=10,
                        inputs=stim_list,
                        targets=target_list,
                        # call_before_trial=print_header,
                        # call_after_trial=show_target
                        )
