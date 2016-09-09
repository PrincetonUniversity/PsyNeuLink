from PsyNeuLink.Globals.Keywords import *

from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import Transfer
from PsyNeuLink.Functions.Projections.Mapping import Mapping
from PsyNeuLink.Functions.Projections.LearningSignal import LearningSignal
from PsyNeuLink.Functions.Process import process
from PsyNeuLink.Functions.Utility import Logistic, LinearMatrix, random_matrix

Input_Layer = Transfer(name='Input Layer',
                       function=Logistic(),
                       default_input_value = np.zeros((2,)))

Hidden_Layer_1 = Transfer(name='Hidden Layer_1',
                          function=Logistic(),
                          default_input_value = np.zeros((5,)))

Hidden_Layer_2 = Transfer(name='Hidden Layer_2',
                          function=Logistic(),
                          default_input_value = [0,0,0,0])

Output_Layer = Transfer(name='Output Layer',
                        function=Logistic(),
                        default_input_value = [0,0,0])

random_weight_matrix = lambda sender, receiver : random_matrix(sender, receiver, .2, -.1)

# TEST PROCESS.LEARNING WITH:
# CREATION OF FREE STANDING PROJECTIONS THAT HAVE NO LEARNING (Input_Weights, Middle_Weights and Output_Weights)
# INLINE CREATION OF PROJECTIONS (Input_Weights, Middle_Weights and Output_Weights)
# NO EXPLICIT CREATION OF PROJECTIONS (Input_Weights, Middle_Weights and Output_Weights)


Input_Weights = Mapping(name='Input Weights',
                        sender=Input_Layer,
                        receiver=Hidden_Layer_1,
                        # matrix=(random_weight_matrix, LearningSignal()),
                        matrix=(FULL_CONNECTIVITY_MATRIX, LearningSignal),
                        # matrix=FULL_CONNECTIVITY_MATRIX
                        # matrix=random_weight_matrix,
                        )

Middle_Weights = Mapping(name='Middle Weights',
                         sender=Hidden_Layer_1,
                         receiver=Hidden_Layer_2,
                        # matrix=(FULL_CONNECTIVITY_MATRIX, LearningSignal())
                        matrix=FULL_CONNECTIVITY_MATRIX
                         )
Output_Weights = Mapping(name='Output Weights',
                         sender=Hidden_Layer_2,
                         receiver=Output_Layer,
                         # params={FUNCTION_PARAMS: {MATRIX: (FULL_CONNECTIVITY_MATRIX, LEARNING_SIGNAL)}}
                         # params={FUNCTION_PARAMS: {MATRIX: FULL_CONNECTIVITY_MATRIX}}
                         matrix = FULL_CONNECTIVITY_MATRIX
                         )

z = process(default_input_value=[0, 0],
            configuration=[Input_Layer,
                           # Input_Weights,
                           Hidden_Layer_1,
                           # Middle_Weights,
                           Hidden_Layer_2,
                           # Output_Weights,
                           Output_Layer],
            learning=LearningSignal,
            prefs={kpVerbosePref: PreferenceEntry(False, PreferenceLevel.INSTANCE)})

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
