"""
This implements a model of Locus Coeruleus / Norepinephrine (LC/NE) function described in `Gilzenrat et al. (2002)
<http://www.sciencedirect.com/science/article/pii/S0893608002000552?via%3Dihub>`_, used to simulate behavioral
and electrophysiological data (from LC recordings) in non-human primates.

"""

import numpy as np
from psyneulink.components.functions.function import Linear, Logistic
from psyneulink.components.functions.function import ModulationParam
from psyneulink.components.system import system
from psyneulink.components.process import process
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.library.mechanisms.processing.transfer.recurrenttransfermechanism \
    import RecurrentTransferMechanism
from psyneulink.library.mechanisms.processing.transfer.lca import LCA
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.library.subsystems.agt.lccontrolmechanism import LCControlMechanism
from psyneulink.globals.keywords import FULL_CONNECTIVITY_MATRIX, VALUE, PROJECTIONS


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
                                    monitored_output_states=[decision_layer],
                                    input_states=[{VALUE:[0],
                                                   PROJECTIONS:np.array([[1],[0]])
                                                   }],
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
