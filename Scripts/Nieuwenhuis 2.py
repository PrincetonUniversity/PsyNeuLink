
# ****************************************  Nieuwenhuis et al. (2005) model *************************************************


"""
Overview
--------
"The Role of the Locus Coeruleus in Mediating the Attentional Blink: A Neurocomputational Theory",
by Nieuwenhuis et al. (2005).

<https://research.vu.nl/ws/files/2063874/Nieuwenhuis%20Journal%20of%20Experimental%20Psychology%20-%20General%20134(3)-2005%20u.pdf`.

This model seeks to investigate the role of the Locus Coeruleus in mediating the attentional blink. The attentional
blink refers to the temporary impairment in perceiving the 2nd of 2 targets presented in close temporal proximity.

During the attentional blink paradigm, on each trial a list of letters is presented to subjects, colored in black on a
grey background. Additionally, two numbers are presented during each trial and the task is to correctly identify which
two digits between 2-9 were presented. A vast amount of studies showed that the accuracy of identifying both digits
correctly depends on the lag between the two target stimuli. Especially between 200 and 300 ms after T1 onset subjects
accuracy decreases. However, presenting the second target stimulus T2 right after the first target stimulus T1,
subjects performance is as accurate as with lags longer then 400ms between T1 and T2. This model aims to bridge findings
from behavioral psychology and findings from neurophysiology with a neurocomputational theory.

The model by Nieuwenhuis et al. (2005) shows that the findings on the attentional blink paradigm can be explained by
the mechanics of the Locus Ceruleus.

With this model it is possible to simulate that subjects behavior on identifying the second target stimuli T2 accurately
depends on:
    whether T1 was accurately identified
    the lag between T1 and T2
    the mode of the LC

This example illustrates Figure 3 from Nieuwenhuis et al. (2005) paper with Lag 2 and one execution only.
Note that in the Nieuwenhuis et al. (2005) paper the Figure shows the average avtivation over 1000 execution.

The model consists of two networks. A behavioral network, feeding forward information from the input layer,
to the decision layer, to the response layer, and a LC control mechanism, projecting gain to both, the behavioral layer
and the response layer.

COMMENT:
Describe what the LC actually does,i.e, FHN, Euler integration,

COMMENT

Creating Nieuwenhuis et al. (2005)
----------------------------------

After setting global variables, weights and initial values the behavioral network is created with 3 layers,
i.e. INPUT LAYER, DECISION LAYER, and RESPONSE LAYER. The INPUT LAYER is constructed with a TransferMechansim of size 3,
and a Linear function with the default slope set to 1.0 and the intercept set to 0.0.

The DECISION LAYER is implemented with a LCA mechanism where each element is connected to every other element with
mutually inhibitory weights and self-excitation weights, defined in PsyNeuLink as <competition>, <self_excitation>, and
<leak>. <leak> defines the sign of the off-diagonals, here the mutually inhibitory weights. The ordinary differential
equation that describes the change in state with respect to time is implemented in the LCA mechanism with the
<integrator_mode> set to True and setting the <time_step_size>.

The final step is to implement the RESPONSE LAYER:
The RESPONSE LAYER is implemented as the DECISION LAYER with a `LCA` mechanism and the parameters specified as in the
paper. (WATCH OUT !!! In the paper the weight "Mutual inhibition among response units" is not defined, but needs to be
set to 0.0 in order to reproduce the paper)

The weights of the behavioral network are created with two numpy arrays.

The LC is implemented with a `LCControlMechansim`. The `LCControlMechansim` has a FitzHugh–Nagumo system implemented.
All parameters from this system are specified as in the paper. Additionally, the `LCControlMechansim` can only monitor
output states that come from an `ObjectiveMechanism`. Thus, a `ObjectiveMechanism` is created in the
<objective_mechanism> parameter with a <Linear> function set to it's default values and the <monitored_output_states>
parameter set to decision_layer with the weights projecting from T1, T2 and the distraction element to the
`ObjectiveMechanism`. Note that the weights from the distraction unit are set to 0.0 since the paper did not implement
weights from the distraction unit to the LC. The parameters G and k are set inside the `LCControlMechansim`.
This LCControlMechanism projects a gain control signal to the DECISION LAYER and the RESPONSE LAYER.

"""

# Import all dependencies.
# Note: Please import matplotlib before importing any psyneulink dependencies.
from matplotlib import pyplot as plt
import sys
import numpy as np
# from scipy.special import erfinv # need to import this to make us of the UniformToNormalDist function.
from psyneulink.components.functions.function import UniformToNormalDist
import psyneulink as pnl

# --------------------------------- Global Variables ----------------------------------------
# Now, we set the global variables, weights and initial values as in the paper.
# WATCH OUT !!! In the paper the weight "Mutual inhibition among response units" is not defined, but needs to be set to
# 0 in order to reproduce the paper.


SD = 0.15       # noise determined by standard deviation (SD)
a = 0.50        # Parameter describing shape of the FitzHugh–Nagumo cubic nullcline for the fast excitation variable v
d = 0.5         # Uncorrelated Activity
k = 1.5         # Scaling factor for transforming NE release (u ) to gain (g ) on potentiated units
G = 0.5         # Base level of gain applied to decision and response units
dt = 0.02       # time step size
C = 0.90        # LC coherence (see Gilzenrat et al. (2002) on more details on LC coherence

initial_hv = 0.07                     # Initial value for h(v)
initial_w = 0.14                      # initial value u
initial_v = (initial_hv - (1-C)*d)/C  # get initial v from initial h(v)

# Weights:
inpwt = 1.5       # inpwt (Input to decision layer)
crswt = 1/3       # crswt (Crosstalk input to decision layer)
inhwt = 1.0       # inhwt (Mutual inhibition among decision units)
respinhwt = 0     # respinhwt (Mutual inhibition among response units)  !!! WATCH OUT: this parameter is not mentioned
# in the original paper, most likely since it was set 0
decwt = 3.5       # decwt (Target decision unit to response unit)
selfdwt = 2.5     # selfdwt (Self recurrent conn. for each decision unit)
selfrwt = 2.0     # selfrwt (Self recurrent conn. for response unit)
lcwt = 0.3        # lcwt (Target decision unit to LC)
decbias = 1.75    # decbias (Bias input to decision units)
respbias = 1.75   # respbias (Bias input to response units)
tau_v = 0.05    # Time constant for fast LC excitation variable v | NOTE: tau_v is misstated in the Gilzenrat paper(0.5)
tau_u = 5.00    # Time constant for slow LC recovery variable (‘NE release’) u
trials = 1100   # number of trials to reproduce Figure 3 from Nieuwenhuis et al. (2005)

# Create mechanisms ---------------------------------------------------------------------------------------------------

# Input Layer --- [ Target 1, Target 2, Distractor ]

# First, we create the 3 layers of the behavioral network, i.e. INPUT LAYER, DECISION LAYER, and RESPONSE LAYER.
input_layer = pnl.TransferMechanism(size = 3,                      # Number of units in input layer
                                initial_value= [[0.0,0.0,0.0]],     # Initial input values
                                name='INPUT LAYER')                 # Define the name of the layer; this is optional,
                                                                    # but will help you to overview your model later on

# Create Decision Layer  --- [ Target 1, Target 2, Distractor ]
decision_layer = pnl.LCA(size=3,                            # Number of units in input layer
                     initial_value= [[0.0,0.0,0.0]],    # Initial input values
                     time_step_size=dt,                 # Integration step size
                     leak=-1.0,                         # Sets off diagonals to negative values
                     self_excitation=selfdwt,           # Set diagonals to self excitate
                     competition=inhwt,                 # Set off diagonals to inhibit
                     function=pnl.Logistic(bias=decbias),   # Set the Logistic function with bias = decbias
                     # noise=UniformToNormalDist(standard_dev = SD).function, # Set noise with seed generator compatible with MATLAB random seed generator 22 (rsg=22)
                     integrator_mode=True,                                           # Please see https://github.com/jonasrauber/randn-matlab-python for further documentation
                     name='DECISION LAYER')

# decision_layer.set_log_conditions('RESULT')  # Log RESULT of the decision layer
decision_layer.set_log_conditions('value')  # Log value of the decision layer

for output_state in decision_layer.output_states:
    output_state.value *= 0.0                                       # Set initial output values for decision layer to 0

# Create Response Layer  --- [ Target1, Target2 ]
response_layer = pnl.LCA(size=2,                                        # Number of units in input layer
                     initial_value= [[0.0,0.0]],                    # Initial input values
                     time_step_size=dt,                             # Integration step size
                     leak=-1.0,                                     # Sets off diagonals to negative values
                     self_excitation=selfrwt,                       # Set diagonals to self excitate
                     competition=respinhwt,                         # Set off diagonals to inhibit
                     function=pnl.Logistic(bias=respbias),    # Set the Logistic function with bias = decbias
                     # noise=UniformToNormalDist(standard_dev = SD).function, # Set noise with seed generator compatible with MATLAB random seed generator 22 (rsg=22)
                     integrator_mode=True,                                         # Please see https://github.com/jonasrauber/randn-matlab-python for further documentation
                     name='RESPONSE LAYER')

response_layer.set_log_conditions('RESULT')  # Log RESULT of the response layer
for output_state in response_layer.output_states:
    output_state.value *= 0.0                                       # Set initial output values for response layer to 0

# Connect mechanisms --------------------------------------------------------------------------------------------------
# Weight matrix from Input Layer --> Decision Layer
input_weights = np.array([[inpwt, crswt, crswt],                    # Input weights are diagonals, cross weights are off diagonals
                          [crswt, inpwt, crswt],
                          [crswt, crswt, inpwt]])

# Weight matrix from Decision Layer --> Response Layer
output_weights = np.array([[decwt, 0.0],                            # Projection weight from decision layer from T1 and T2 but not distraction unit (row 3 set to all zeros) to response layer
                           [0.0, decwt],                            # Need a 3 by 2 matrix, to project from decision layer with 3 units to response layer with 2 units
                           [0.0, 0.0]])

# The process will connect the layers and weights.
decision_process = pnl.Process(pathway=[input_layer,
                                    input_weights,
                                    decision_layer,
                                    output_weights,
                                    response_layer],
                           name='DECISION PROCESS')

# Abstracted LC to modulate gain --------------------------------------------------------------------

# This LCControlMechanism modulates gain.
LC = pnl.LCControlMechanism(integration_method="EULER",                 # We set the integration method to Euler like in the paper
                        threshold_FHN=a,                            # Here we use the Euler method for integration and we want to set the parameters,
                        uncorrelated_activity_FHN=d,                # for the FitzHugh–Nagumo system.
                        time_step_size_FHN=dt,
                        mode_FHN=C,
                        time_constant_v_FHN=tau_v,
                        time_constant_w_FHN=tau_u,
                        a_v_FHN=-1.0,
                        b_v_FHN=1.0,
                        c_v_FHN=1.0,
                        d_v_FHN=0.0,
                        e_v_FHN=-1.0,
                        f_v_FHN=1.0,
                        a_w_FHN=1.0,
                        b_w_FHN=-1.0,
                        c_w_FHN=0.0,
                        t_0_FHN=0.0,
                        base_level_gain=G,                          # Additionally, we set the parameters k and G to compute the gain equation.
                        scaling_factor_gain=k,
                        initial_v_FHN=initial_v,                    # Initialize v
                        initial_w_FHN=initial_w,                    # Initialize w (WATCH OUT !!!: In the Gilzenrat paper the authors set this parameter to be u, so that one does not think about a small w as if it would represent a weight
                        objective_mechanism= pnl.ObjectiveMechanism(function=pnl.Linear,
                            monitored_output_states=[(decision_layer, # Project the output of T1 and T2 but not the distraction unit of the decision layer to the LC with a linear function.
                            np.array([[lcwt],[lcwt],[0.0]]))],
                            name='Combine values'),
                        modulated_mechanisms=[decision_layer, response_layer],  # Modulate gain of decision & response layers
                        name='LC')

#This is under construction:
LC.loggable_items
LC.set_log_conditions('value')

for output_state in LC.output_states:
	output_state.value *= G + k*initial_w          # Set initial gain to G + k*initial_w, when the System runs the very first time, since the decison layer executes before the LC and hence needs one initial gain value to start with.

# Now, we specify the processes of the System, which in this case is just the decision_process
task = pnl.System(processes=[decision_process])

# Create Stimulus -----------------------------------------------------------------------------------------------------

# In the paper, each period has 100 time steps, so we will create 11 time periods.
# As described in the paper in figure 3, during the first 3 time periods the distractor units are given an input fixed to 1.
# Then T1 gets turned on during time period 4 with an input of 1.
# T2 gets turns on with some lag from T1 onset on, in this example we turn T2 on with Lag 2 and an input of 1
# Between T1 and T2 and after T2 the distractor unit is on.
# We create one array with 3 numbers, one for each input unit and repeat this array 100 times for one time period
# We do this 11 times. T1 is on for time4, T2 is on for time7 to model Lag3
stepSize = 100  # Each stimulus is presented for two units of time which is equivalent to 100 time steps
time1 = np.repeat(np.array([[0,0,1]]), stepSize,axis =0)
time2 = np.repeat(np.array([[0,0,1]]), stepSize,axis =0)
time3 = np.repeat(np.array([[0,0,1]]), stepSize,axis =0)
time4 = np.repeat(np.array([[1,0,0]]), stepSize,axis =0)    # Turn T1 on
time5 = np.repeat(np.array([[0,0,1]]), stepSize,axis =0)
time6 = np.repeat(np.array([[0,1,0]]), stepSize,axis =0)    # Turn T2 on --> example for Lag 2
time7 = np.repeat(np.array([[0,0,1]]), stepSize,axis =0)
time8 = np.repeat(np.array([[0,0,1]]), stepSize,axis =0)
time9 = np.repeat(np.array([[0,0,1]]), stepSize,axis =0)
time10 = np.repeat(np.array([[0,0,1]]), stepSize,axis =0)
time11 = np.repeat(np.array([[0,0,1]]), stepSize,axis =0)

# Concatenate the 11 arrays to one array with 1100 rows and 3 colons.
time = np.concatenate((time1, time2, time3, time4, time5, time6, time7, time8, time9, time10, time11), axis = 0)

# assign inputs to input_layer (Origin Mechanism) for each trial
stim_list_dict = {input_layer:time}

def h_v(v,C,d):
    return C*v + (1-C)*d

# Initialize output arrays for plotting
LC_results_v = [h_v(initial_v,C,d)]
LC_results_w = [initial_w]
decision_layer_target = [0.5]
decision_layer_target2 = [0.5]
decision_layer_distractor = [0.5]
response1 = [0.5]
response2 = [0.5]


# Show percentage while running:
def record_trial():
    current_trial_num = len(LC_results_v)
    if current_trial_num%50 == 0:
        percent = int(round((float(current_trial_num) / trials)*100))
        sys.stdout.write("\r"+ str(percent) +"% complete")
        sys.stdout.flush()
sys.stdout.write("\r0% complete")
sys.stdout.flush()

# run the system
task.run(stim_list_dict, num_trials= 5, call_after_trial=record_trial)

#
# t = np.linspace(0,len(LC_results_v),6)
# plt.plot(t, LC_results_v, label="h(v)")
# plt.plot(t, LC_results_w, label="w")
# plt.plot(t, decision_layer_target, label="target")
# plt.plot(t, decision_layer_target2, label="target2")
#
# plt.plot(t,decision_layer_distractor, label="distractor")
# plt.plot(t, response1, label="response")
# plt.plot(t, response2, label="response2")
# plt.xlabel('Activation')
# plt.ylabel('h(V)')
# plt.legend(loc='upper left')
# plt.ylim((-0.2,1.2))
# # plt.show()

# This prints information about the System,
# including its execution list indicating the order in which the Mechanisms will execute
# IMPLEMENTATION NOTE:
#  MAY STILL NEED TO SCHEDULE RESPONSE TO EXECUTE BEFORE LC
#  (TO BE MODULATED BY THE GAIN MANIPULATION IN SYNCH WITH THE DECISION LAYER
task.show()

# This displays a diagram of the System
# task.show_graph()

LC.log.nparray()


# decision_layer.log.nparray()
print(decision_layer.log.nparray())
