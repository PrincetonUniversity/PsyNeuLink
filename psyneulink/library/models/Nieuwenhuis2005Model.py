# Import all dependencies.
import argparse

import numpy as np
import psyneulink as pnl
# from scipy.special import erfinv # need to import this to make us of the UniformToNormalDist function.

parser = argparse.ArgumentParser()
parser.add_argument('--no-plot', action='store_false', help='Disable plotting', dest='enable_plot')
args = parser.parse_args()

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
initial_v = (initial_hv - (1 - C) * d) / C  # get initial v from initial h(v)

# Weights:
inpwt = 1.5       # inpwt (Input to decision layer)
crswt = 1 / 3       # crswt (Crosstalk input to decision layer)
inhwt = 1.0       # inhwt (Mutual inhibition among decision units)
respinhwt = 0     # respinhwt (Mutual inhibition among response units)  !!! WATCH OUT: this parameter is not mentioned
# in the original paper, most likely since it was set 0
decwt = 3.5       # decwt (Target decision unit to response unit)
selfdwt = 2.5     # selfdwt (Self recurrent conn. for each decision unit)
selfrwt = 2.0     # selfrwt (Self recurrent conn. for response unit)
lcwt = 0.3        # lcwt (Target decision unit to LC)
decbias = 1.75    # decbias (Bias input to decision units)
respbias = 1.75   # respbias (Bias input to response units)
tau_v = 0.05    # Time constant for fast LC excitation variable v
tau_u = 5.00    # Time constant for slow LC recovery variable (‘NE release’) u
trials = 1100   # number of trials to reproduce Figure 3 from Nieuwenhuis et al. (2005)

# Setting seed (if noise is applied)
# np.random.seed(22)    # Set noise with seed generator to 22 to use the UniformToNormalDist function to have a python
# seed compatible with MATLAB random seed generator (rsg=22)
# Please see https://github.com/jonasrauber/randn-matlab-python for further documentation

# Create mechanisms ---------------------------------------------------------------------------------------------------

# Input Layer --- [ Target 1, Target 2, Distractor ]

# First, we create the 3 layers of the behavioral network, i.e. INPUT LAYER, DECISION LAYER, and RESPONSE LAYER.
input_layer = pnl.TransferMechanism(
    size=3,                      # Number of units in input layer
    initial_value=[[0.0, 0.0, 0.0]],     # Initial input values
    name='INPUT LAYER'                  # Define the name of the layer; this is optional,
)                 # but will help you to overview your model later on

# Create Decision Layer  --- [ Target 1, Target 2, Distractor ]
decision_layer = pnl.LCAMechanism(
    size=3,                            # Number of units in input layer
    initial_value=[[0.0, 0.0, 0.0]],    # Initial input values
    time_step_size=dt,                 # Integration step size
    leak=1.0,                         # Sets off diagonals to negative values
    self_excitation=selfdwt,           # Set diagonals to self excitate
    competition=inhwt,                 # Set off diagonals to inhibit
    function=pnl.Logistic(x_0=decbias),   # Set the Logistic function with bias = decbias
    # noise=pnl.UniformToNormalDist(standard_deviation = SD).function, # The UniformToNormalDist function will
    integrator_mode=True,               # set the noise with a seed generator that is compatible with
    name='DECISION LAYER'               # MATLAB random seed generator 22 (rsg=22)
)

# decision_layer.set_log_conditions('RESULT')  # Log RESULT of the decision layer
decision_layer.set_log_conditions('value')  # Log value of the decision layer

for output_port in decision_layer.output_ports:
    output_port.parameters.value.set(output_port.value * 0.0, override=True)  # Set initial output values for decision layer to 0

# Create Response Layer  --- [ Target1, Target2 ]
response_layer = pnl.LCAMechanism(
    size=2,                                        # Number of units in input layer
    initial_value=[[0.0, 0.0]],                    # Initial input values
    time_step_size=dt,                             # Integration step size
    leak=1.0,                                     # Sets off diagonals to negative values
    self_excitation=selfrwt,                       # Set diagonals to self excitate
    competition=respinhwt,                         # Set off diagonals to inhibit
    function=pnl.Logistic(x_0=respbias),          # Set the Logistic function with bias = decbias
    # noise=pnl.UniformToNormalDist(standard_deviation = SD).function,
    integrator_mode=True,
    name='RESPONSE LAYER'
)

response_layer.set_log_conditions('RESULT')     # Log RESULT of the response layer
for output_port in response_layer.output_ports:
    output_port.parameters.value.set(output_port.value * 0.0, override=True)  # Set initial output values for response layer to 0

# Connect mechanisms --------------------------------------------------------------------------------------------------
# Weight matrix from Input Layer --> Decision Layer
input_weights = np.array([
    [inpwt, crswt, crswt],                    # Input weights are diagonals, cross weights are off diagonals
    [crswt, inpwt, crswt],
    [crswt, crswt, inpwt]
])

# Weight matrix from Decision Layer --> Response Layer
output_weights = np.array([
    [decwt, 0.0],  # Weight from T1 and T2 but not distractor unit (row 3 set to all zeros) to response layer
    [0.0, decwt],  # Need a 3 by 2 matrix, to project from decision layer with 3 units to response layer with 2 units
    [0.0, 0.0]
])

decision_pathway = pnl.Pathway(
    pathway=[
        input_layer,
        input_weights,
        decision_layer,
        output_weights,
        response_layer
    ],
    name='DECISION PATHWAY'
)

# Abstracted LC to modulate gain --------------------------------------------------------------------

# This LCControlMechanism modulates gain.
LC = pnl.LCControlMechanism(
    integration_method="EULER",       # We set the integration method to Euler like in the paper
    threshold_FitzHughNagumo=a,                  # Here we use the Euler method for integration and we want to set the parameters,
    uncorrelated_activity_FitzHughNagumo=d,      # for the FitzHugh–Nagumo system.
    time_step_size_FitzHughNagumo=dt,
    mode_FitzHughNagumo=C,
    time_constant_v_FitzHughNagumo=tau_v,
    time_constant_w_FitzHughNagumo=tau_u,
    a_v_FitzHughNagumo=-1.0,
    b_v_FitzHughNagumo=1.0,
    c_v_FitzHughNagumo=1.0,
    d_v_FitzHughNagumo=0.0,
    e_v_FitzHughNagumo=-1.0,
    f_v_FitzHughNagumo=1.0,
    a_w_FitzHughNagumo=1.0,
    b_w_FitzHughNagumo=-1.0,
    c_w_FitzHughNagumo=0.0,
    t_0_FitzHughNagumo=0.0,
    base_level_gain=G,                # Additionally, we set the parameters k and G to compute the gain equation
    scaling_factor_gain=k,
    initial_v_FitzHughNagumo=initial_v,          # Initialize v
    initial_w_FitzHughNagumo=initial_w,          # Initialize w
    objective_mechanism=pnl.ObjectiveMechanism(
        function=pnl.Linear,
        monitor=[(
            decision_layer,  # Project output of T1 and T2 but not distractor from decision layer to LC
            np.array([[lcwt], [lcwt], [0.0]])
        )],
        name='Combine values'
    ),
    modulated_mechanisms=[decision_layer, response_layer],  # Modulate gain of decision & response layers
    name='LC'
)

# Log value of LC
LC.set_log_conditions('value')

# Set initial gain to G + k*initial_w, when the System runs the very first time,
# since the decison layer executes before the LC and hence needs one initial gain value to start with.
for output_port in LC.output_ports:
    output_port.parameters.value.set(output_port.value * (G + k * initial_w), override=True)

task = pnl.Composition()
task.add_linear_processing_pathway(decision_pathway)
task.add_node(LC)

# Create Stimulus -----------------------------------------------------------------------------------------------------

# In the paper, each period has 100 time steps, so we will create 11 time periods.
# As described in the paper in figure 3, during the first 3 time periods input to distractor units is fixed to 1.
# Then T1 gets turned on during time period 4 with an input of 1.
# T2 gets turns on with some lag from T1 onset on, in this example we turn T2 on with Lag 2 and an input of 1
# Between T1 and T2 and after T2 the distractor unit is on.
# We create one array with 3 numbers, one for each input unit and repeat this array 100 times for one time period
# We do this 11 times. T1 is on for time4, T2 is on for time7 to model Lag3
num_time_steps = 100  # Each stimulus is presented for two units of time which is equivalent to 100 time steps
stimulus_T1 = np.repeat(np.array([[0, 0, 1]]), num_time_steps, axis=0)
stimulus_T2 = np.repeat(np.array([[0, 0, 1]]), num_time_steps, axis=0)
stimulus_T3 = np.repeat(np.array([[0, 0, 1]]), num_time_steps, axis=0)
stimulus_T4 = np.repeat(np.array([[1, 0, 0]]), num_time_steps, axis=0)    # Turn T1 on
stimulus_T5 = np.repeat(np.array([[0, 0, 1]]), num_time_steps, axis=0)
stimulus_T6 = np.repeat(np.array([[0, 1, 0]]), num_time_steps, axis=0)    # Turn T2 on --> example for Lag 2
stimulus_T7 = np.repeat(np.array([[0, 0, 1]]), num_time_steps, axis=0)
stimulus_T8 = np.repeat(np.array([[0, 0, 1]]), num_time_steps, axis=0)
stimulus_T9 = np.repeat(np.array([[0, 0, 1]]), num_time_steps, axis=0)
stimulus_T10 = np.repeat(np.array([[0, 0, 1]]), num_time_steps, axis=0)
stimulus_T11 = np.repeat(np.array([[0, 0, 1]]), num_time_steps, axis=0)

# Concatenate the 11 arrays to one array with 1100 rows and 3 columns.
time = np.concatenate((stimulus_T1, stimulus_T2, stimulus_T3, stimulus_T4, stimulus_T5, stimulus_T6,
                       stimulus_T7, stimulus_T8, stimulus_T9, stimulus_T10, stimulus_T11), axis=0)

# assign inputs to input_layer (Origin Mechanism) for each trial
stim_list_dict = {input_layer: time}

# show the system
# task.show_graph()

# run the system
task.run(stim_list_dict, num_trials=trials)


# This displays a diagram of the System
# task.show_graph()

LC_results = LC.log.nparray()[1][1]        # get logged results
LC_results_w = np.zeros([trials])          # get LC_results_w
for i in range(trials):
    LC_results_w[i] = LC_results[5][i + 1][2][0][0]
LC_results_v = np.zeros([trials])          # get LC_results_v
for i in range(trials):
    LC_results_v[i] = LC_results[5][i + 1][1][0][0]


def h_v(v, C, d):                   # Compute h(v)
    return C * v + (1 - C) * d


LC_results_hv = np.zeros([trials])    # get LC_results_hv
for i in range(trials):
    LC_results_hv[i] = h_v(LC_results_v[i], C, d)


if args.enable_plot:
    import matplotlib.pyplot as plt

    # Plot the Figure 3 from the paper
    t = np.linspace(0, trials, trials)            # Create array for x axis with same length then LC_results_v
    fig = plt.figure()                          # Instantiate figure
    ax = plt.gca()                              # Get current axis for plotting
    ax2 = ax.twinx()                            # Create twin axis with a different y-axis on the right side of the figure
    ax.plot(t, LC_results_hv, label="h(v)")      # Plot h(v)
    ax2.plot(t, LC_results_w, label="w", color='red')  # Plot w
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc=2)          # Create legend on one side
    ax.set_xlabel('Time (ms)')                  # Set x axis lable
    ax.set_ylabel('LC Activity')                # Set left y axis label
    ax2.set_ylabel('NE Output')                 # Set right y axis label
    plt.title('Nieuwenhuis 2005 PsyNeuLink Lag 2 without noise', fontweight='bold')  # Set title
    ax.set_ylim((-0.2, 1.0))                     # Set left y axis limits
    ax2.set_ylim((0.0, 0.4))                    # Set right y axis limits
    plt.show(block=not pnl._called_from_pytest)
