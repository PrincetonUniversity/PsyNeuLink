from PsyNeuLink.Components.Functions.Function import Logistic
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection

Input_Layer = TransferMechanism(name='Input Layer',
                       function=Logistic(),
                       default_variable = np.zeros((2,)))

Hidden_Layer_1 = TransferMechanism(name='Hidden Layer_1',
                          function=Logistic(),
                          default_variable = np.zeros((5,)))

Hidden_Layer_2 = TransferMechanism(name='Hidden Layer_2',
                          function=Logistic(),
                          default_variable = [0,0,0,0])

Output_Layer = TransferMechanism(name='Output Layer',
                        function=Logistic(),
                        default_variable = [0,0,0])

random_weight_matrix = lambda sender, receiver : random_matrix(sender, receiver, .2, -.1)

Input_Weights_matrix = random_weight_matrix
# Input_Weights_matrix = (np.arange(2*5).reshape((2, 5)) + 1)/(2*5)
Middle_Weights_matrix = (np.arange(5*4).reshape((5, 4)) + 1)/(5*4)
Output_Weights_matrix = (np.arange(4*3).reshape((4, 3)) + 1)/(4*3)


Input_Weights = MappingProjection(name='Input Weights',
                        matrix=Input_Weights_matrix)

# Middle_Weights = MappingProjection(name='Middle Weights',
#                          sender=Hidden_Layer_1,
#                          receiver=Hidden_Layer_2,
#                          matrix=Middle_Weights_matrix)
#
# # Commented lines in this projection illustrate variety of ways in which matrix and learning signals can be specified
# Output_Weights = MappingProjection(name='Output Weights',
#                          sender=Hidden_Layer_2,
#                          receiver=Output_Layer,
#                          matrix=Output_Weights_matrix)
#

z = process(default_variable=[0, 0],
            pathway=[Input_Layer,
                           Input_Weights,
                           Hidden_Layer_1,
                           Hidden_Layer_2,
                           Output_Layer],
            learning=LearningProjection,
            target=[0,0,1],
            prefs={VERBOSE_PREF: False,
                   REPORT_OUTPUT_PREF: True})

for i in range(10):

    print("\n\n**** TRIAL: ", i)

    results = z.execute(input=[-1, 30],target=[0, 0, 1])

    print ('\nInput Weights: \n', Input_Weights.matrix)
    print ('Middle Weights: \n', Middle_Weights.matrix)
    print ('Output Weights: \n', Output_Weights.matrix)

    # print ('MSE: \n', Output_Layer.output_values[])

print(results)
