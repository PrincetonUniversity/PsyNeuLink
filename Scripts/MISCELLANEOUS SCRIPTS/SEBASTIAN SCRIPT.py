# from PsyNeuLink.Components.Functions.Function import Logistic, random_matrix
from PsyNeuLink.Components.Functions.Function import Logistic
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
from PsyNeuLink.Components.Projections.TransmissiveProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.System import system

# set up layers

Input_Layer = TransferMechanism(name='Input Layer',
                       function=Logistic(),
                       default_input_value = np.zeros((4,)))

Task_Layer = TransferMechanism(name='Task Layer',
                       function=Logistic(),
                       default_input_value = np.zeros((2,)))

Hidden_Layer = TransferMechanism(name='Hidden Layer',
                          function=Logistic(),
                          default_input_value = np.zeros((5,)))

Output_Layer = TransferMechanism(name='Output Layer',
                        function=Logistic(),
                        default_input_value = [0,0])

random_weight_matrix = lambda sender, receiver : random_matrix(sender, receiver, .2, -.1)

# set up weights

Input_Hidden_Weights = MappingProjection(name='Input-Hidden Weights',
                         sender=Input_Layer,
                         receiver=Hidden_Layer,
                         matrix=random_weight_matrix)

Task_Hidden_Weights = MappingProjection(name='Task-Hidden Weights',
                         sender=Task_Layer,
                         receiver=Hidden_Layer,
                         matrix=random_weight_matrix)

Hidden_Output_Weights = MappingProjection(name='Hidden-Output Weights',
                         sender=Hidden_Layer,
                         receiver=Output_Layer,
                         matrix=random_weight_matrix)

Task_Output_Weights = MappingProjection(name='Task-Output Weights',
                         sender=Task_Layer,
                         receiver=Output_Layer,
                         matrix=random_weight_matrix)

# set up processes

stimulus_process = process(default_input_value=[0, 0, 0, 0],
            pathway=[Input_Layer,
                     Input_Hidden_Weights,
                     Hidden_Layer,
                     Hidden_Output_Weights,
                     Output_Layer],
            learning=LearningProjection,
            target=[0,0],
            prefs={VERBOSE_PREF: False,
                   REPORT_OUTPUT_PREF: True})

taskHidden_process = process(default_input_value=[0, 0],
                             pathway=[Task_Layer,
                                      Task_Hidden_Weights,
                                      Hidden_Layer],
                             learning=LearningProjection)

taskOutput_process = process(default_input_value=[0, 0],
                             pathway=[Task_Layer,
                                      Task_Output_Weights,
                                      Output_Layer],
                             learning=LearningProjection,
                             target=np.zeros((2,)),
                             prefs={VERBOSE_PREF: False,
                                    REPORT_OUTPUT_PREF: False})

# System:
multitaskingNet = system(processes=[stimulus_process, taskHidden_process, taskOutput_process],
                  enable_controller=False,
                  name='Multitasking Test System')