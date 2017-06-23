from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.System import *

# from PsyNeuLink.Globals.Run import run, construct_inputs

Input_Layer = TransferMechanism(name='Input Layer',
                       function=Logistic(),
                       default_input_value = np.zeros((2,)))

Hidden_Layer_1 = TransferMechanism(name='Hidden Layer_1',
                          function=Logistic(),
                          default_input_value = np.zeros((5,)))

Hidden_Layer_2 = TransferMechanism(name='Hidden Layer_2',
                          function=Logistic(),
                          default_input_value = [0,0,0,0])

Output_Layer = TransferMechanism(name='Output Layer',
                        function=Logistic(),
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
                        matrix=Input_Weights_matrix
                        )

# This projection will be used by the process below by assigning its sender and receiver args
#    to mechanismss in the pathway
Middle_Weights = MappingProjection(name='Middle Weights',
                         sender=Hidden_Layer_1,
                         receiver=Hidden_Layer_2,
                         matrix=Middle_Weights_matrix
                         )

# Commented lines in this projection illustrate variety of ways in which matrix and learning signals can be specified
Output_Weights = MappingProjection(name='Output Weights',
                         sender=Hidden_Layer_2,
                         receiver=Output_Layer,
                         matrix=Output_Weights_matrix
                         )

p = process(default_input_value=[0, 0],
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
            learning=LearningProjection,
            target=[0,0,1],
            prefs={VERBOSE_PREF: False,
                   REPORT_OUTPUT_PREF: True})

s = system(processes=[p],
           # controller=EVCMechanism,
           # enable_controller=True,
           # monitor_for_control=[Reward, DDM_PROBABILITY_UPPER_THRESHOLD, (DDM_RESPONSE_TIME, -1, 1)],
           # monitor_for_control=[Input, PROBABILITY_UPPER_THRESHOLD,(RESPONSE_TIME, -1, 1)],
           # monitor_for_control=[MonitoredOutputStatesOption.ALL_OUTPUT_STATES],
           prefs={VERBOSE_PREF: False,
                  REPORT_OUTPUT_PREF: True},
           name='Learning Test System')

def print_header():
    print("\n\n**** TRIAL: ", CentralClock.trial)

def show_target():
    print ('\n\nInput: {}\nTarget: {}\n'.
           format(p.input, p.target))
    print ('\nInput Weights: \n', Input_Weights.matrix)
    print ('Middle Weights: \n', Middle_Weights.matrix)
    print ('Output Weights: \n', Output_Weights.matrix)
    # print ('MSE: \n', Output_Layer.output_values[])

stim_list = {Input_Layer:[[-1, 30],[2, 10]]}

# p.execute()
# s.execute()
s.run(num_executions=10,
      # inputs=stim_list,
      inputs=[[-1, 30],[2, 10]],
      targets=[[0, 0, 1],[0, 0, 2]],
      call_before_trial=print_header,
      call_after_trial=show_target)
