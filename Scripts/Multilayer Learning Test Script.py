from PsyNeuLink.Globals.Keywords import *

from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import Transfer
from PsyNeuLink.Functions.Projections.Mapping import Mapping
from PsyNeuLink.Functions.Process import Process_Base

Input_Layer = Transfer(name='Input Layer',
                       params={kwExecuteMethod:kwLogistic},
                       default_input_value = [0,0])

Hidden_Layer = Transfer(name='Hidden Layer',
                        params={kwExecuteMethod:kwLogistic},
                        default_input_value = [0,0])

Output_Layer = Transfer(name='Output Layer',
                        params={kwExecuteMethod:kwLogistic},
                        default_input_value = [0,0])

Input_Weights = Mapping(name='Input Weights',
                                  sender=Input_Layer,
                                  receiver=Hidden_Layer,
                                  # params={kwExecuteMethodParams:{kwMatrix:kwIdentityMatrix}}
                                  params={kwExecuteMethodParams: {kwMatrix: (kwIdentityMatrix,kwLearningSignal)}}
                                  # params={kwExecuteMethodParams:{kwMatrix:(kwIdentityMatrix,kwControlSignal)}}
                                  )

Output_Weights = Mapping(name='Output Weights',
                                  sender=Hidden_Layer,
                                  receiver=Output_Layer,
                                  # params={kwExecuteMethodParams:{kwMatrix:kwIdentityMatrix}}
                                  params={kwExecuteMethodParams: {kwMatrix: (kwIdentityMatrix,kwLearningSignal)}}
                                  # params={kwExecuteMethodParams:{kwMatrix:(kwIdentityMatrix,kwControlSignal)}}
                                  )

z = Process_Base(default_input_value=[0, 0],
                 # params={kwConfiguration:[Input_Layer, Learned_Weights, Output_Layer]},
                 params={kwConfiguration:[Input_Layer, Input_Weights, Hidden_Layer, Output_Weights, Output_Layer]},
                 prefs={kpVerbosePref: PreferenceEntry(True, PreferenceLevel.INSTANCE)})


# Learned_Weights.monitoringMechanism.target = [1,1]
# Learned_Weights.monitoringMechanism.target = [0,0]
# from PsyNeuLink.Functions.Mechanisms.MonitoringMechanisms.Comparator import kwComparatorTarget
# Learned_Weights.monitoringMechanism.paramsCurrent[kwComparatorTarget] = [1,1]

# z.execute(input=[-1, 30],
#           runtime_params={kwComparatorTarget: [1, 1]})

print ('Input Weights: \n', Input_Weights.matrix)
print ('Output Weights: \n', Output_Weights.matrix)

for i in range(10):
    z.execute([[-1, 30],[1, 1]])
    print ('Input Weights: \n', Input_Weights.matrix)
    print ('Output Weights: \n', Output_Weights.matrix)
