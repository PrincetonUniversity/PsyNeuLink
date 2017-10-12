"""
This implements a model of Locus Coeruleus / Norepinephrine (LC/NE) function described in `Gilzenrat et al. (2002)
<http://www.sciencedirect.com/science/article/pii/S0893608002000552?via%3Dihub>`_, used to simulate behavioral
and electrophysiological data (from LC recordings) in non-human primates.

"""

import numpy as np

from psyneulink.library.subsystems.agt.gilzenrattransfermechanism import GilzenratTransferMechanism
from psyneulink.components.functions.function import Linear, Logistic, NormalDist
from psyneulink.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.process import Process
from psyneulink.components.system import System
from psyneulink.library.subsystems.agt.lccontrolmechanism import LCControlMechanism

# --------------------------------- Global Variables ----------------------------------------

# Mode ("coherence")
C = 0.55
# Uncorrelated Activity
d = 0.5

# Initial values
# initial_h_of_v = 0.07
initial_h_of_v = 0.2
initial_v = (initial_h_of_v - (1-C)*d)/C
# initial_w = 0.14
initial_w=0.2

# g(t) = G + k*w(t)

# Scaling factor for transforming NE release (u ) to gain (g ) on potentiated units
k = 3.0
# Base level of gain applied to decision and response units
G = 0.5

# numerical integration
time_step_size = 0.02
number_of_trials = int(20/time_step_size)

# noise
standard_deviation = 0.22*(time_step_size**0.5)

# --------------------------------------------------------------------------------------------

input_layer = TransferMechanism(default_variable=np.array([[1,0]]),
                                name='INPUT LAYER')

# Implement projections from inputs to decision layer with weak cross-talk connections
#    from target and distractor inputs to their competing decision layer units
input_weights = np.array([[1, .33],[.33, 1]])


# Implement self-excitatory (auto) and mutually inhibitory (hetero) connections within the decision layer
decision_layer = GilzenratTransferMechanism(size=2,
                                            matrix=np.matrix([[1,0],[0,-1]]),
                                            #auto=1.0,
                                            #hetero=-1.0,
                                            time_step_size=time_step_size,
                                            noise=NormalDist(mean=0.0,standard_dev=standard_deviation).function,
                                            function=Logistic,
                                            name='DECISION LAYER')

# Implement connection from target but not distractor unit in decision layer to response
output_weights = np.array([[1.84], [0]])

# Implement response layer with a single, self-excitatory connection
#To do Markus: specify recurrent self-connrection weight for response unit to 2.00
response = GilzenratTransferMechanism(size=1,
                                      function=Logistic(bias=2),
                                      time_step_size=time_step_size,
                                      noise=NormalDist(mean=0.0,standard_dev=standard_deviation).function,
                                      name='RESPONSE')

# Implement response layer with input_state for ObjectiveMechanism that has a single value
# and a MappingProjection to it that zeros the contribution of the decision unit in the decision layer
LC = LCControlMechanism(time_step_size_FHN=time_step_size,  # integrating step size
                        mode_FHN=C,                         # coherence: set to either .95 or .55
                        uncorrelated_activity_FHN=d,        # Baseline level of intrinsic, uncorrelated LC activity
                        time_constant_v_FHN=0.05,
                        time_constant_w_FHN=5,
                        a_v_FHN=-1.0,
                        b_v_FHN=1.0,
                        c_v_FHN=1.0,
                        d_v_FHN=0.0,
                        e_v_FHN=-1.0,
                        f_v_FHN=1.0,
                        a_w_FHN=1.0,
                        b_w_FHN=-1.0,
                        c_w_FHN=0.0,
                        t_0_FHN=0,
                        initial_v_FHN=initial_v,
                        initial_w_FHN=initial_w,
                        threshold_FHN=0.5,        #Parameter describing shape of the FitzHugh–Nagumo cubic nullcline for the fast excitation variable v
        objective_mechanism=ObjectiveMechanism(
                                    function=Linear,
                                    monitored_output_states=[(decision_layer, None, None, np.array([[0.3],[0.0]]))],
                                    input_states=[[0]],
                                    name='LC ObjectiveMechanism'
        ),
        modulated_mechanisms=[decision_layer, response],
        name='LC')


# ELICITS WARNING:
decision_process = Process(pathway=[input_layer,
                                    input_weights,
                                    decision_layer,
                                    output_weights,
                                    response],
                           name='DECISION PROCESS')


lc_process = Process(pathway=[decision_layer,
                              # CAUSES ERROR:
                              # np.array([[1,0],[0,0]]),
                              LC],
                           name='LC PROCESS')

task = System(processes=[decision_process, lc_process])

# stimulus
stim_list_dict = {input_layer: np.repeat(np.array([[0,0],[1,0]]),10/time_step_size,axis=0)}

def h_v(v,C,d):
    return C*v + (1-C)*d

# Initialize output arrays for plotting
LC_results_v = [h_v(initial_v,C,d)]
LC_results_w = [initial_w]
decision_layer_target = [0.5]

def record_step():
    LC_results_v.append(h_v(LC.value[2][0], C, d))
    LC_results_w.append(LC.value[3][0])
    decision_layer_target.append(decision_layer.value[0][0])

task.run(stim_list_dict, num_trials= number_of_trials, call_after_trial=record_step)

from matplotlib import pyplot as plt
import numpy as np
t = np.arange(0.0, len(LC_results_v), 1.0)
plt.plot(t, LC_results_v)
plt.plot(t, LC_results_w)
plt.plot(t, decision_layer_target)
plt.xlabel(' # of timesteps ')
plt.ylabel('h(V)')
plt.ylim((-0.2,1.2))
plt.show()

# This prints information about the System,
# including its execution list indicating the order in which the Mechanisms will execute
# IMPLEMENTATION NOTE:
#  MAY STILL NEED TO SCHEDULE RESPONSE TO EXECUTE BEFORE LC
#  (TO BE MODULATED BY THE GAIN MANIPULATION IN SYNCH WITH THE DECISION LAYER
task.show()

# This displays a diagram of the System
# task.show_graph()

