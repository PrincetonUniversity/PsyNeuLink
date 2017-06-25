from PsyNeuLink.Components.Functions.Function import Logistic
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection

Input_Layer = TransferMechanism(name='Input Layer',
                       function=Logistic(),
                       default_input_value = [0,0])

Hidden_Layer_1 = TransferMechanism(name='Hidden Layer_1',
                          function=Logistic(),
                          default_input_value = [0,0,0,0,0])

Output_Layer = DDM(name='Output Layer DDM',
                   threshold=0.1,
                   params = {MONITOR_FOR_LEARNING:PROBABILITY_LOWER_THRESHOLD},
                   default_input_value = [0])

Input_Weights = MappingProjection(name='Input Weights',
                                  sender=Input_Layer,
                                  receiver=Hidden_Layer_1,
                                  # params={FUNCTION_PARAMS:{MATRIX:(IDENTITY_MATRIX,CONTROL_PROJECTION)}}
                                  params={FUNCTION_PARAMS: {MATRIX: (FULL_CONNECTIVITY_MATRIX,LEARNING_PROJECTION)}}
                                  )

Output_Weights = MappingProjection(name='Output Weights',
                                  sender=Hidden_Layer_1,
                                  receiver=Output_Layer,
                                  # params={FUNCTION_PARAMS:{MATRIX:IDENTITY_MATRIX}}
                                  params={FUNCTION_PARAMS: {MATRIX: (FULL_CONNECTIVITY_MATRIX,LEARNING_PROJECTION)}}
                                  # params={FUNCTION_PARAMS:{MATRIX:(IDENTITY_MATRIX,CONTROL_PROJECTION)}}
                                  )

z = process(default_input_value=[0, 0],
            # params={PATHWAY:[Input_Layer, Learned_Weights, Output_Layer]},
            pathway=[Input_Layer,
                           Input_Weights,
                           Hidden_Layer_1,
                           Output_Weights,
                           Output_Layer],
            prefs={kpVerbosePref: PreferenceEntry(True, PreferenceLevel.INSTANCE)})


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
