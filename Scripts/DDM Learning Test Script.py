from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.Transfer import Transfer
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.Mapping import Mapping
from PsyNeuLink.Components.Functions.Function import Logistic

Input_Layer = Transfer(name='Input Layer',
                       function=Logistic(),
                       default_input_value = [0,0])

Hidden_Layer_1 = Transfer(name='Hidden Layer_1',
                          function=Logistic(),
                          default_input_value = [0,0,0,0,0])

Output_Layer = DDM(name='Output Layer DDM',
                   threshold=0.1,
                   params = {MONITOR_FOR_LEARNING:ERROR_RATE},
                   default_input_value = [0])

Input_Weights = Mapping(name='Input Weights',
                                  sender=Input_Layer,
                                  receiver=Hidden_Layer_1,
                                  # params={FUNCTION_PARAMS:{MATRIX:(IDENTITY_MATRIX,CONTROL_SIGNAL)}}
                                  params={FUNCTION_PARAMS: {MATRIX: (FULL_CONNECTIVITY_MATRIX,LEARNING_SIGNAL)}}
                                  )

Output_Weights = Mapping(name='Output Weights',
                                  sender=Hidden_Layer_1,
                                  receiver=Output_Layer,
                                  # params={FUNCTION_PARAMS:{MATRIX:IDENTITY_MATRIX}}
                                  params={FUNCTION_PARAMS: {MATRIX: (FULL_CONNECTIVITY_MATRIX,LEARNING_SIGNAL)}}
                                  # params={FUNCTION_PARAMS:{MATRIX:(IDENTITY_MATRIX,CONTROL_SIGNAL)}}
                                  )

z = process(default_input_value=[0, 0],
            # params={PATHWAY:[Input_Layer, Learned_Weights, Output_Layer]},
            pathway=[Input_Layer,
                           Input_Weights,
                           Hidden_Layer_1,
                           Output_Weights,
                           Output_Layer],
            prefs={kpVerbosePref: PreferenceEntry(True, PreferenceLevel.INSTANCE)})


# Learned_Weights.monitoringMechanism.target = [1,1]
# Learned_Weights.monitoringMechanism.target = [0,0]
# from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.Comparator import COMPARATOR_TARGET
# Learned_Weights.monitoringMechanism.paramsCurrent[COMPARATOR_TARGET] = [1,1]

# z.execute(input=[-1, 30],
#           runtime_params={COMPARATOR_TARGET: [1, 1]})

num_cycles = 10

for i in range(num_cycles):

    # z.execute([[-1, 30],[0, 0, 1, 1]])
    print("\n\n==== {} EXECUTION CYCLE {} ========================================================".
          format(z.name, i))
    z.execute([[-1, 30],[0]])

    print ('Input Weights: \n', Input_Weights.matrix)
    print ('Output Weights: \n', Output_Weights.matrix)
