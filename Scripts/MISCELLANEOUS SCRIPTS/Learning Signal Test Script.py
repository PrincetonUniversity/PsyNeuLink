# from PsyNeuLink.Components.Functions.Function import Logistic, random_matrix
from PsyNeuLink.Components.Functions.Function import Logistic
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection

random_weight_matrix = lambda sender, receiver : random_matrix(sender, receiver, .2, -.1)

Input_Layer = TransferMechanism(name='Input Layer',
                       function=Logistic(),
                       default_variable = [0,0])

Output_Layer = TransferMechanism(name='Output Layer',
                        function=Logistic(),
                        default_variable = [0,0])

Learned_Weights = MappingProjection(name='Learned Weights',
                          sender=Input_Layer,
                          receiver=Output_Layer,

                          # DEPRECATED:
                          # function=LinearMatrix(matrix=(DEFAULT_MATRIX,LEARNING_PROJECTION)) # FUNCTION NO LONGER A PARAM

                          # THESE ALL WORK:

                          # NOTE: MUST REMOVE FEEDBACK FROM PROCESS INPUT (SEE z.execute BELOW)
                          matrix=random_weight_matrix
                          # params={FUNCTION_PARAMS: {MATRIX: IDENTITY_MATRIX}}
                          # params={FUNCTION_PARAMS: {MATRIX: (IDENTITY_MATRIX,CONTROL_PROJECTION)}}

                          # NOTE: THESE REQUIRE THAT FEEDBACK BE INCLUDED IN PROCESS INPUT:  (SEE z.execute BELOW)
                          # matrix=(DEFAULT_MATRIX, LEARNING_PROJECTION)
                          # matrix=(DEFAULT_MATRIX, LearningProjection)
                          # matrix=(DEFAULT_MATRIX, LEARNING_PROJECTION)
                          # matrix=(DEFAULT_MATRIX, LearningProjection())
                          # matrix=(FULL_CONNECTIVITY_MATRIX, LEARNING_PROJECTION)
                          # matrix=(RANDOM_CONNECTIVITY_MATRIX, LearningProjection())
                          # matrix=(random_weight_matrix, LEARNING_PROJECTION)
                          # params={FUNCTION_PARAMS: {MATRIX: (IDENTITY_MATRIX,LEARNING_PROJECTION)}},
                          # params={FUNCTION_PARAMS: {MATRIX: (IDENTITY_MATRIX,LearningProjection)}}
                          # params={FUNCTION_PARAMS: {MATRIX: (FULL_CONNECTIVITY_MATRIX,LEARNING_PROJECTION)}}
                          # params={FUNCTION_PARAMS: {MATRIX: (random_weight_matrix, LEARNING_PROJECTION)}}
                          )

z = process(name="TEST LEARNER",
            default_variable=[0, 0],
            pathway=[Input_Layer, Learned_Weights, Output_Layer],
            learning=LEARNING_PROJECTION,
            prefs={VERBOSE_PREF: True,
                   REPORT_OUTPUT_PREF: True})


# from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ObjectiveMechanisms.ComparatorMechanism \
#                                                                                   import COMPARATOR_TARGET
# Learned_Weights.monitoringMechanism.paramsCurrent[COMPARATOR_TARGET] = [1,1]

# z.execute(input=[-1, 30],
#           runtime_params={COMPARATOR_TARGET: [1, 1]})

for i in range(10):

    # WITHOUT FEEDBACK IN INPUT:
    # z.execute([[-1, 30]])
    # WITH FEEDBACK IN INPUT:
    z.execute([[-1, 30],[1, 1]])
    print (Learned_Weights.matrix)
