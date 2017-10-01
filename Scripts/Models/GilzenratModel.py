"""
This implements a model of Locus Coeruleus / Norepinephrine (LC/NE) function described in `Gilzenrat et al. (2002)
<http://www.sciencedirect.com/science/article/pii/S0893608002000552?via%3Dihub>`_, used to simulate behavioral
and electrophysiological data (from LC recordings) in non-human primates.

"""

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
from PsyNeuLink.Globals.Keywords import FULL_CONNECTIVITY_MATRIX, VARIABLE, VALUE, PROJECTIONS


input_layer = TransferMechanism(size=2,
                                name='INPUT LAYER')

# Implement projections from inputs to decision layer with weak cross-talk connections
#    from target and distractor inputs to their competing decision layer units
input_weights = np.array([[1, .1],[.1, 1]])

# Implement self-excitatory (auto) and mutually inhibitory (hetero) connections within the decision layer
decision_layer = RecurrentTransferMechanism(size=2,
                                            auto=1,
                                            hetero=-1,
                                            function=Logistic,
                                            name='DECISION LAYER')

# Implement connection from target but not distractor unit in decision layer to response
output_weights = np.array([[1], [0]])

# Implement response layer with a single, self-excitatory connection
response = RecurrentTransferMechanism(size=1,
                                      function=Logistic,
                                      name='RESPONSE')

# Implement response layer with input_state for ObjectiveMechanism that has a single value
#    and a MappingProjection to it that zeros the contribution of the decision unit in the decision layer
LC = LCControlMechanism(
        objective_mechanism=ObjectiveMechanism(
                                    # monitored_output_states=[decision_layer],
                                    # input_states=[{VARIABLE:[0],
                                    #                PROJECTIONS:np.array([[1],[0]])
                                    #                }],
                                    # MODIFIED 10/1/17 NEW
                                    monitored_output_states=None,
                                    # InputState specification dictionary:
                                    input_states=[{VARIABLE:[0],
                                                   PROJECTIONS:(decision_layer,np.array([[1],[0]]))}],
                                    # InputState specification tuple:
                                    # input_states=[([0],(decision_layer,np.array([[1],[0]])))],
                                    # MODIFIED 10/1/17 END
                                    # # Alternative form of specification:
                                    # monitored_output_states=[(decision_layer, None, None, np.array([[1],[0]]))],
                                    # input_states=[[0]],
                                    name='LC ObjectiveMechanism'
        ),
        modulated_mechanisms=[decision_layer, response],
        name='LC')

# ELICITS WARNING:
decision_process = process(pathway=[input_layer,
                                    input_weights,
                                    decision_layer,
                                    output_weights,
                                    response],
                           name='DECISION PROCESS')

lc_process = process(pathway=[decision_layer,
                              # CAUSES ERROR:
                              # np.array([[1,0],[0,0]]),
                              LC],
                           name='LC PROCESS')

task = system(processes=[decision_process, lc_process])

# This prints information about the System,
# including its execution list indicating the order in which the Mechanisms will execute
# IMPLEMENTATION NOTE:
#  MAY STILL NEED TO SCHEDULE RESPONSE TO EXECUTE BEFORE LC
#  (TO BE MODULATED BY THE GAIN MANIPULATION IN SYNCH WITH THE DECISION LAYER
task.show()

# This displays a diagram of the System
task.show_graph()
