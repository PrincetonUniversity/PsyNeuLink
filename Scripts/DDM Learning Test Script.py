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

Output_Layer = DDM(name='Output Layer DDM',
                   params = {kwExecuteMethodParams:{kwDDM_Threshold:0.01},
                             kwMonitorForLearning:kwDDM_Error_Rate},
                   default_input_value = [0])

Input_Weights = Mapping(name='Input Weights',
                                  sender=Input_Layer,
                                  receiver=Hidden_Layer_1,
                                  # params={kwExecuteMethodParams:{kwMatrix:(kwIdentityMatrix,kwControlSignal)}}
                                  params={kwExecuteMethodParams: {kwMatrix: (kwFullConnectivityMatrix,kwLearningSignal)}}
                                  )

Output_Weights = Mapping(name='Output Weights',
                                  sender=Hidden_Layer_1,
                                  receiver=Output_Layer,
                                  # params={kwExecuteMethodParams:{kwMatrix:kwIdentityMatrix}}
                                  params={kwExecuteMethodParams: {kwMatrix: (kwFullConnectivityMatrix,kwLearningSignal)}}
                                  # params={kwExecuteMethodParams:{kwMatrix:(kwIdentityMatrix,kwControlSignal)}}
                                  )

z = Process_Base(default_input_value=[0, 0],
                 # params={kwConfiguration:[Input_Layer, Learned_Weights, Output_Layer]},
                 params={kwConfiguration:[Input_Layer,
                                          Input_Weights,
                                          Hidden_Layer_1,
                                          Output_Weights,
                                          Output_Layer]},
                 prefs={kpVerbosePref: PreferenceEntry(True, PreferenceLevel.INSTANCE)})


# Learned_Weights.monitoringMechanism.target = [1,1]
# Learned_Weights.monitoringMechanism.target = [0,0]
# from PsyNeuLink.Functions.Mechanisms.MonitoringMechanisms.Comparator import kwComparatorTarget
# Learned_Weights.monitoringMechanism.paramsCurrent[kwComparatorTarget] = [1,1]

# z.execute(input=[-1, 30],
#           runtime_params={kwComparatorTarget: [1, 1]})

num_cycles = 10

for i in range(num_cycles):

    # z.execute([[-1, 30],[0, 0, 1, 1]])
    print("\n\n==== {} EXECUTION CYCLE {} ========================================================".
          format(z.name, i))
    z.execute([[-1, 30],[0]])

    print ('Input Weights: \n', Input_Weights.matrix)
    print ('Output Weights: \n', Output_Weights.matrix)
