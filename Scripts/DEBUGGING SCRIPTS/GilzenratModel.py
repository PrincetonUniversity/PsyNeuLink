import numpy as np
from PsyNeuLink.Components.Functions.Function import Linear, Logistic
from PsyNeuLink.Components.Functions.Function import ModulationParam
from PsyNeuLink.Components.System import system
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism import ObjectiveMechanism
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.TransferMechanisms.RecurrentTransferMechanism \
    import RecurrentTransferMechanism
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.TransferMechanisms.LCA import LCA
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Library.Subsystems.AGT.LCControlMechanism import LCControlMechanism
from PsyNeuLink.Globals.Keywords import FULL_CONNECTIVITY_MATRIX

# target_input = TransferMechanism(name='target_input')
# distractor_input = TransferMechanism(name='distractor_input')
# target_decision = RecurrentTransferMechanism(function=Logistic, name='target_decision')
# distractor_decision = RecurrentTransferMechanism(function=Logistic, name='distractor_decision')
# response = RecurrentTransferMechanism(function=Logistic, name='response')
#
# MappingProjection(sender=target_decision, receiver=distractor_decision)
# target_process = process(pathway=[target_input, target_decision, response])
# distractor_process = process(pathway=[distractor_input, distractor_decision, target_decision])
#
# LC = LCControlMechanism(
#         objective_mechanism=[target_decision],
#         modulated_mechanisms=[target_decision, distractor_decision, response],
#         name='LC')
#
# task = system(processes=[target_process, distractor_process])
#
# task.show()
# task.show_graph()

input_layer = TransferMechanism(size=2,
                                name='INPUT LAYER')
input_weights = np.array([[1, .1],[.1, 1]])
output_weights = np.array([[1], [0]])
decision_layer = RecurrentTransferMechanism(size=2,
                                            # matrix=[[1,-1],[-1,1]],
                                            auto=1,
                                            hetero=-1,
                                            # # GENERATES A FULL CONNECTIVITY MATRIX:
                                            # auto=[[1,1]],
                                            # hetero=[[-1,0],[0,-1]],
                                            # # GENERATES ONLY AN IDENTITY MATRIX, MAYBE SHOULD GENERATE AN ERROR
                                            # auto=[[1]],
                                            # hetero=[[0,-1,],[-1,0]],
                                            function=Logistic,
                                            name='DECISION LAYER')
response = RecurrentTransferMechanism(size=1,
                                      matrix=[[1]],
                                      function=Logistic,
                                      name='RESPONSE')
LC = LCControlMechanism(
        # COMMENTING OUT THE FOLLOWING LINE(S) CAUSES AN ERROR
        objective_mechanism=[ObjectiveMechanism(monitored_output_states=[decision_layer],
                                                input_states=[[0]])],
        # objective_mechanism=[decision_layer],
        modulated_mechanisms=[decision_layer, response],
        name='LC')

# ELICITS WARNING:
decision_process = process(pathway=[input_layer,
                                    input_weights,
                                    decision_layer,
                                    output_weights,
                                    response],
                           name='DECISION PROCESS')


# CAUSES ERROR:
# LC_projection_matrix = MappingProjection(matrix=np.array([[1,0],[0,0]]))

lc_process = process(pathway=[decision_layer,
                              # CAUSES ERROR:
                              # np.array([[1,0],[0,0]]),
                              LC],
                           name='DECISION PROCESS')

# FIX: NEED TO SPECIFY SIZE OF LC OBJECTIVE MECHANISM'S INPUT_STATE AND/OR THE MATRIX OF THE PROJECTION TO IT
# FIX: NEED TO SCHEDULE RESPONSE TO EXECUTE BEFORE LC (TO BE IN THE SAME PHASE AS THE DECISION LAYER WRT TO GAIN MOD)

print(LC.objective_mechanism.input_state.path_afferents[0].matrix)
# THIS DOESN'T WORK: (matrix param remains unchanged
LC.objective_mechanism.input_state.path_afferents[0].matrix = np.array([[1,0],[0,0]])
print(LC.objective_mechanism.input_state.path_afferents[0].matrix)

task = system(processes=[decision_process, lc_process])

task.show()
task.show_graph()
