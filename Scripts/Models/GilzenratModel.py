"""
This implements a model of Locus Coeruleus / Norepinephrine (LC/NE) function described in `Gilzenrat et al. (2002)
<http://www.sciencedirect.com/science/article/pii/S0893608002000552?via%3Dihub>`_, used to simulate behavioral
and electrophysiological data (from LC recordings) in non-human primates.

"""
import sys
import numpy as np

from psyneulink.library.subsystems.agt.gilzenrattransfermechanism import GilzenratTransferMechanism
from psyneulink.components.functions.function import Linear, Logistic, NormalDist
from psyneulink.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.process import Process
from psyneulink.components.system import System
from psyneulink.library.subsystems.agt.lccontrolmechanism import LCControlMechanism

# --------------------------------- Global Variables ----------------------------------------

# function to set LC coherence high (0.95)
high_C = True
if high_C:
    C = 0.95            # Mode ("coherence")
    initial_hv = 0.07   # Initial value for h(v)
    initial_u=0.14      # initial value u
else:
    C = 0.55            # Mode ("coherence")
    initial_hv = 0.2    # Initial value for h(v)
    initial_u = 0.2     # initial value u

d = 0.5         # Uncorrelated Activity
k = 3.0         # Scaling factor for transforming NE release (u ) to gain (g ) on potentiated units
G = 0.5         # Base level of gain applied to decision and response units
SD = 0.0        # noise determined by standard deviation (SD)
trials = 1000   # number of trials PsyNeuLink runs

initial_v = (initial_hv - (1-C)*d)/C        # get v from h of v
time_step_size = 0.02                       # numerical integration


# --------------------------------------------------------------------------------------------

input_layer = TransferMechanism(size=2,                                 # number of units in input layer
                                initial_value=np.array([[0.0, 0.0]]),   # initial values for both input units
                                name='INPUT LAYER')

# Implement projections from inputs to decision layer with weak cross-talk connections from target and distractor inputs to their competing decision layer units
input_weights = np.array([[1, 1/3],[1/3, 1]])       #1 =  (Input to decision layer) 1/3 = (Crosstalk input to decision layer)


# Implement self-excitatory (auto) and mutually inhibitory (hetero) connections within the decision layer
decision_layer = GilzenratTransferMechanism(size=2,                                 # number of units in decision layer
                                            initial_value=np.array([[0.0, 0.0]]),   # initial values for both decision units
                                            matrix=np.matrix([[1,-1],[-1,1]]),      # 1 = (self recurrent connection for each decision unit), -1 = (mutual inhibition among decision units)
                                            time_step_size=time_step_size,          # 0.02 = Time scale granularity
                                            noise=NormalDist(mean=0.0,standard_dev=SD).function,    #noise with mean = 0 and SD = SD
                                            function=Logistic(bias=0.0),            # 0 = (Bias  input to decision units)
                                            name='DECISION LAYER')

# Implement connection from target but not distractor unit in decision layer to response
output_weights = np.array([[1.84], [0]])

# Implement response layer with a single, self-excitatory connection
response = GilzenratTransferMechanism(size=1,                           # number of units in response layer
                                      initial_value=np.array([[0.0]]),  # initial values for response unit
                                      matrix=np.matrix([[2.0]]),        # 2 = (self recurrent connection for response unit
                                      function=Logistic(bias=-2.0),     # -2 =(Bias input to response unit) need to plug in -2 since Gilzenrat has negative signs in his logistic equation
                                      time_step_size=time_step_size,    # 0.02 = Time scale granularity
                                      noise=NormalDist(mean=0.0,standard_dev=SD).function,
                                      name='RESPONSE')

# Implement response layer with input_state for ObjectiveMechanism that has a single value
# and a MappingProjection to it that zeros the contribution of the decision unit in the decision layer
LC = LCControlMechanism(integration_method="EULER",
                        time_step_size_FHN=time_step_size,  # integrating step size
                        mode_FHN=C,                         # coherence: set to either .95 or .55
                        uncorrelated_activity_FHN=d,        # Baseline level of intrinsic, uncorrelated LC activity
                        time_constant_v_FHN=0.05,           # Time constant of fast LC effect--v # WATCH OUT: this number is WRONG in the Gilzenrat paper there it is 0.5
                        time_constant_w_FHN=5,              # Time constant of slow LC effect--w
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
                        initial_v_FHN=initial_v,            #set initial value v
                        initial_w_FHN=initial_u,            #set initial value u
                        threshold_FHN=0.5,                  #Parameter describing shape of the FitzHugh–Nagumo cubic nullcline for the fast excitation variable v
        objective_mechanism=ObjectiveMechanism(
                                    function=Linear,
                                    monitored_output_states=[(decision_layer,               # from decision layer
                                                              None,                         # Why None?
                                                              None,                         # Why None?
                                                              np.array([[0.3],[0.0]]))],    # 0.3 = Target decision unit to LC; 0.0 = No connection from distraction unit to LC
                                    name='LC ObjectiveMechanism'
        ),
        modulated_mechanisms=[decision_layer, response],    # Modulate gain of decision layer and response layer
        name='LC')

# Implement Process: input layer - input weights - decision layer - output weights - response layer
decision_process = Process(pathway=[input_layer,
                                    input_weights,
                                    decision_layer,
                                    output_weights,
                                    response],
                           name='DECISION PROCESS')

task = System(processes=[decision_process])

# Create stimulus: 500 arrays with zeros in both input units and 500 arrays with 1 in first input unit and 0 in second input unit
stim_list_dict = {input_layer: np.repeat(np.array([[0,0],[1,0]]), 500,axis=0)}

# Function to compute h(v)
def h_v(v,C,d):
    return C*v + (1-C)*d

# Initialize output arrays for plotting
LC_results_v = [h_v(initial_v,C,d)]
LC_results_u = [initial_u]
decision_layer_target = [0.0]
decision_layer_distractor = [0.0]
response_layer = [0.0]

# Recorde results for LC, decision layer and response layer
def record_trial():
    LC_results_v.append(h_v(LC.value[2][0], C, d))
    LC_results_u.append(LC.value[3][0])
    decision_layer_target.append(decision_layer.value[0][0])
    decision_layer_distractor.append(decision_layer.value[0][1])
    response_layer.append(response.value[0][0])
    #show percentage of comlete computation
    current_trial_num = len(LC_results_v)
    if current_trial_num%50 == 0:
        percent = int(round((float(current_trial_num) / trials)*100))
        sys.stdout.write("\r"+ str(percent) +"% complete")
        sys.stdout.flush()

sys.stdout.write("\r0% complete")
sys.stdout.flush()
task.run(stim_list_dict, num_trials= trials, call_after_trial=record_trial)

# Plot results of all units into one figure
from matplotlib import pyplot as plt
import numpy as np

t = np.arange(0.0, 20.02, 0.02)         # Create x axis "t" for plotting
# Plot target unit, distraction unit, response unit, h(v), u
plt.plot(t, decision_layer_target, label="target unit", color = 'green')        # Plot target unit
plt.plot(t,decision_layer_distractor, label="distraction unit", color = 'red')  # Plot distraction unit
plt.plot(t, response_layer, label="response unit", color = 'magenta')           # Plot response unit
plt.plot(t, LC_results_v, label="h(v)", color = 'b')                            # Plot h(v)
plt.plot(t, LC_results_u, label="u", color = 'black')                           # Plot u

plt.xlabel('Time')
plt.ylabel('Activation')
plt.legend(loc='upper left')
plt.xlim((0.0,20.0))
plt.ylim((-0.2, 1.2))
x_values = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
plt.xticks(x_values)
plt.title('GILZENRAT 2002 PsyNeuLink', fontweight='bold')
plt.show()

task.show()

# This displays a diagram of the System
# task.show_graph()

