from PsyNeuLink.Globals.Keywords import *

from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import Transfer
from PsyNeuLink.Functions.Projections.Mapping import Mapping
from PsyNeuLink.Functions.Process import Process_Base

Input_Layer = Transfer(name='Input Layer',
                       params={kwExecuteMethod:kwLogistic},
                       default_input_value = [0,0])

Hidden_Layer_1 = Transfer(name='Hidden Layer_1',
                        params={kwExecuteMethod:kwLogistic},
                        default_input_value = [0,0,0,0,0])

Hidden_Layer_2 = Transfer(name='Hidden Layer_2',
                        params={kwExecuteMethod:kwLogistic},
                        default_input_value = [0,0,0,0])

Output_Layer = Transfer(name='Output Layer',
                        params={kwExecuteMethod:kwLogistic},
                        default_input_value = [0,0,0])
# Output_Layer = DDM(name='Output Layer DDM',
#                         # params={kwExecuteMethod:kwLogistic},
#                         default_input_value = [0])

randomized_matrix = lambda sender, receiver, range, offset: ((range * np.random.rand(sender, receiver)) + offset)
random_weight_matrix = lambda sender, receiver : randomized_matrix(sender, receiver, .2, -.1)

Input_Weights = Mapping(name='Input Weights',
                        sender=Input_Layer,
                        receiver=Hidden_Layer_1,
                        # params={kwExecuteMethodParams:{kwMatrix:(kwIdentityMatrix,kwControlSignal)}}
                        # params={kwExecuteMethodParams: {kwMatrix: (kwFullConnectivityMatrix,kwLearningSignal)}}
                        params={kwExecuteMethodParams: {kwMatrix: (random_weight_matrix, kwLearningSignal)}}
                        )

Middle_Weights = Mapping(name='Middle Weights',
                         sender=Hidden_Layer_1,
                         receiver=Hidden_Layer_2,
                         # params={kwExecuteMethodParams:{kwMatrix:kwIdentityMatrix}}
                         # params={kwExecuteMethodParams: {kwMatrix: (kwIdentityMatrix,kwLearningSignal)}}
                         params={kwExecuteMethodParams: {kwMatrix: (kwFullConnectivityMatrix,kwLearningSignal)}}
                         )
Output_Weights = Mapping(name='Output Weights',
                         sender=Hidden_Layer_2,
                         receiver=Output_Layer,
                         # params={kwExecuteMethodParams:{kwMatrix:kwIdentityMatrix}}
                         params={kwExecuteMethodParams: {kwMatrix: (kwFullConnectivityMatrix,kwLearningSignal)}}
                         # params={kwExecuteMethodParams:{kwMatrix:(kwIdentityMatrix,kwControlSignal)}}
                         )

z = Process_Base(default_input_value=[0, 0],
                 # params={kwConfiguration:[Input_Layer, Learned_Weights, Output_Layer]},
                 params={kwConfiguration:[Input_Layer,
                                          # Input_Weights,
                                          Hidden_Layer_1,
                                          # Middle_Weights,
                                          Hidden_Layer_2,
                                          # Output_Weights,
                                          Output_Layer],
                        kwLearning:kwLearningSignal},
                 prefs={kpVerbosePref: PreferenceEntry(False, PreferenceLevel.INSTANCE)})

# z.execute(input=[-1, 30],
#           runtime_params={kwComparatorTarget: [1, 1]})

print ('Input Weights: \n', Input_Weights.matrix)
print ('Middle Weights: \n', Middle_Weights.matrix)
print ('Output Weights: \n', Output_Weights.matrix)

for i in range(10):

    # z.execute([-1, 30])
    z.execute([[-1, 30],[0, 0, 1]])
    # z.execute([[-1, 30],[0]])

    print ('Input Weights: \n', Input_Weights.matrix)
    print ('Middle Weights: \n', Middle_Weights.matrix)
    print ('Output Weights: \n', Output_Weights.matrix)
