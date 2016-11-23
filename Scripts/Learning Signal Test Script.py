from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import Transfer
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
# from PsyNeuLink.Components.Functions.Function import Logistic, random_matrix
from PsyNeuLink.Components.Functions.Function import Logistic

random_weight_matrix = lambda sender, receiver : random_matrix(sender, receiver, .2, -.1)

Input_Layer = Transfer(name='Input Layer',
                       function=Logistic(),
                       default_input_value = [0,0])

Output_Layer = Transfer(name='Output Layer',
                        function=Logistic(),
                        default_input_value = [0,0])

Learned_Weights = MappingProjection(name='Learned Weights',
                          sender=Input_Layer,
                          receiver=Output_Layer,

                          # DEPRECATED:
                          # function=LinearMatrix(matrix=(DEFAULT_MATRIX,LEARNING_SIGNAL)) # FUNCTION NO LONGER A PARAM

                          # THESE ALL WORK:

                          # NOTE: MUST REMOVE FEEDBACK FROM PROCESS INPUT (SEE z.execute BELOW)
                          matrix=random_weight_matrix
                          # params={FUNCTION_PARAMS: {MATRIX: IDENTITY_MATRIX}}
                          # params={FUNCTION_PARAMS: {MATRIX: (IDENTITY_MATRIX,CONTROL_PROJECTION)}}

                          # NOTE: THESE REQUIRE THAT FEEDBACK BE INCLUDED IN PROCESS INPUT:  (SEE z.execute BELOW)
                          # matrix=(DEFAULT_MATRIX, LEARNING_SIGNAL)
                          # matrix=(DEFAULT_MATRIX, LearningSignal)
                          # matrix=(DEFAULT_MATRIX, LEARNING_SIGNAL)
                          # matrix=(DEFAULT_MATRIX, LearningSignal())
                          # matrix=(FULL_CONNECTIVITY_MATRIX, LEARNING_SIGNAL)
                          # matrix=(RANDOM_CONNECTIVITY_MATRIX, LearningSignal())
                          # matrix=(random_weight_matrix, LEARNING_SIGNAL)
                          # params={FUNCTION_PARAMS: {MATRIX: (IDENTITY_MATRIX,LEARNING_SIGNAL)}},
                          # params={FUNCTION_PARAMS: {MATRIX: (IDENTITY_MATRIX,LearningSignal)}}
                          # params={FUNCTION_PARAMS: {MATRIX: (FULL_CONNECTIVITY_MATRIX,LEARNING_SIGNAL)}}
                          # params={FUNCTION_PARAMS: {MATRIX: (random_weight_matrix, LEARNING_SIGNAL)}}
                          )

z = process(name="TEST LEARNER",
            default_input_value=[0, 0],
            pathway=[Input_Layer, Learned_Weights, Output_Layer],
            learning=LEARNING_SIGNAL,
            prefs={VERBOSE_PREF: True,
                   REPORT_OPUTPUT_PREF: True})


# Learned_Weights.monitoringMechanism.target = [1,1]
# Learned_Weights.monitoringMechanism.target = [0,0]
# from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.ComparatorMechanism import TARGET
# Learned_Weights.monitoringMechanism.paramsCurrent[TARGET] = [1,1]

# z.execute(input=[-1, 30],
#           runtime_params={TARGET: [1, 1]})

for i in range(10):

    # WITHOUT FEEDBACK IN INPUT:
    # z.execute([[-1, 30]])
    # WITH FEEDBACK IN INPUT:
    z.execute([[-1, 30],[1, 1]])
    print (Learned_Weights.matrix)
