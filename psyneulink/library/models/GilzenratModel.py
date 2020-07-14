"""
This implements a model of Locus Coeruleus / Norepinephrine (LC/NE) function described in `Gilzenrat et al. (2002)
<http://www.sciencedirect.com/science/article/pii/S0893608002000552?via%3Dihub>`_, used to simulate behavioral
and electrophysiological data (from LC recordings) in non-human primates.

"""
import argparse
import sys

import numpy as np
import psyneulink as pnl

parser = argparse.ArgumentParser()
parser.add_argument('--no-plot', action='store_false', help='Disable plotting', dest='enable_plot')
parser.add_argument(
    '--noise-stddev',
    type=float,
    help='Standard deviation of Gaussian noise distributions (default: 0.22, Gilzenrat et al. (2002))',
    default=0.22
)
args = parser.parse_args()

# Define Variables ----------------------------------------------------------------------------------------------------

# Weights & Biases:
b_decision = 0.00   # Bias on decision units (not biased)
b_response = 2.00   # Bias on response unit --- NOTE: Gilzenrat has negative signs in his logistic equation
w_XiIi = 1.00       # Connection weight from input units I1 and I2 to respective decision units X1 and X2
w_XiIj = 0.33       # Cross talk weight from input unit to opposite decision unit
w_XiXi = 1.00       # Recurrent self-connection weight for both decision units
w_XiXj = 1.00       # Magnitude of mutually inhibitory weight between decision units
w_X3X1 = 1.84       # Connection weight from target decision unit (X1) to response unit (X3)
w_X3X3 = 2.00       # Recurrent self-connection weight for the response unit
w_vX1 = 0.30        # Connection weight from target decision unit X1 to the abstracted LC

# Other parameters:
a = 0.50        # Parameter describing shape of the FitzHugh–Nagumo cubic nullcline for the fast excitation variable v
d = 0.50        # Baseline level of intrinsic, uncorrelated LC activity
G = 0.50        # Base level of gain applied to decision and response units
k = 3.00        # Scaling factor for transforming NE release (u) to gain (g) on potentiated units
SD = args.noise_stddev  # Standard deviation of Gaussian noise distributions | NOTE: 0.22 in Gilzenrat paper
tau_v = 0.05    # Time constant for fast LC excitation variable v | NOTE: tau_v is misstated in the Gilzenrat paper(0.5)
tau_u = 5.00    # Time constant for slow LC recovery variable (‘NE release’) u
dt = 0.02       # Time step size for numerical integration

# Switch between high C and low C conditions
high_C = True
if high_C:
    C = 0.95            # Mode ("coherence")
    initial_hv = 0.07   # Initial value for h(v)
    initial_u = 0.14      # initial value u
else:
    C = 0.55            # Mode ("coherence")
    initial_hv = 0.2    # Initial value for h(v)
    initial_u = 0.2     # initial value u

initial_v = (initial_hv - (1 - C) * d) / C    # get initial v from initial h(v)

# Create mechanisms ---------------------------------------------------------------------------------------------------

# Input Layer --- [ Target, Distractor ]
input_layer = pnl.TransferMechanism(
    size=2,
    initial_value=np.array([[0.0, 0.0]]),
    name='INPUT LAYER'
)

# Create Decision Layer  --- [ Target, Distractor ]

decision_layer = pnl.LCAMechanism(
    size=2,
    time_step_size=dt,
    leak=1.0,
    self_excitation=w_XiXi,
    competition=w_XiXj,
    #  Recurrent matrix: [  w_XiXi   -w_XiXj ]
    #                    [ -w_XiXj    w_XiXi ]
    function=pnl.Logistic(x_0=b_decision),
    noise=pnl.NormalDist(standard_deviation=SD),
    integrator_mode=True,
    name='DECISION LAYER'
)

# Create Response Layer  --- [ Target ]

response_layer = pnl.LCAMechanism(
    size=1,
    time_step_size=dt,
    leak=1.0,
    self_excitation=w_X3X3,
    #  Recurrent matrix: [w_X3X3]
    #  Competition param does not apply because there is only one unit
    function=pnl.Logistic(x_0=b_response),
    noise=pnl.NormalDist(standard_deviation=SD),
    integrator_mode=True,
    name='RESPONSE'
)

# Connect mechanisms --------------------------------------------------------------------------------------------------

# Weight matrix from Input Layer --> Decision Layer
input_weights = np.array([
    [w_XiIi, w_XiIj],
    [w_XiIj, w_XiIi]
])

# Weight matrix from Decision Layer --> Response Layer
output_weights = np.array([
    [w_X3X1],
    [0.00]
])

decision_pathway = pnl.Pathway(
    pathway=[
        input_layer,
        input_weights,
        decision_layer,
        output_weights,
        response_layer
    ],
    name='DECISION PROCESS'
)

# Monitor decision layer in order to modulate gain --------------------------------------------------------------------

LC = pnl.LCControlMechanism(
    integration_method="EULER",
    threshold_FitzHughNagumo=a,
    uncorrelated_activity_FitzHughNagumo=d,
    base_level_gain=G,
    scaling_factor_gain=k,
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
    initial_v_FitzHughNagumo=initial_v,
    initial_w_FitzHughNagumo=initial_u,
    objective_mechanism=pnl.ObjectiveMechanism(
        function=pnl.Linear,
        monitor=[(
            decision_layer,
            None,
            None,
            np.array([
                [w_vX1],
                [0.0]]
            )
        )],
        name='LC ObjectiveMechanism'
    ),
    modulated_mechanisms=[decision_layer, response_layer],  # Modulate gain of decision & response layers
    name='LC'
)

task = pnl.Composition()
task.add_linear_processing_pathway(decision_pathway)
task.add_node(LC)

# This displays a diagram of the System
# task.show_graph()

# Create Stimulus -----------------------------------------------------------------------------------------------------

# number of trials
trials = 1000

# assign inputs to input_layer (Origin Mechanism) for each trial
stimulus_dictionary = {input_layer: np.repeat(np.array([[0.0, 0.0], [1.0, 0.0]]), 500, axis=0)}
# First 500 trials: target receives input of 0.0, distractor receives input of 0.0
# Second 500 trials: target receives input of 1.0, distractor receives input of 0.0

# Record results & run model ------------------------------------------------------------------------------------------

# Function to compute h(v) from LC's v value


def h_v(v, C, d):
    return C * v + (1 - C) * d


# Initialize output arrays for plotting and storing values
LC_results_h_of_v = [h_v(initial_v, C, d)]
LC_results_u = [initial_u]
decision_layer_target_values = [0.0]
decision_layer_distractor_values = [0.0]
response_layer_values = [0.0]


def record_trial(context):
    # After each trial, store all of the following values:
    LC_results_h_of_v.append(h_v(LC.parameters.value.get(context)[1][0], C, d))
    LC_results_u.append(LC.parameters.value.get(context)[2][0])
    decision_layer_target_values.append(decision_layer.parameters.value.get(context)[0][0])
    decision_layer_distractor_values.append(decision_layer.parameters.value.get(context)[0][1])
    response_layer_values.append(response_layer.parameters.value.get(context)[0][0])

    # Progress bar
    current_trial_num = len(LC_results_h_of_v)
    if current_trial_num % 50 == 0:
        percent = int(round((float(current_trial_num) / trials) * 100))
        sys.stdout.write("\r" + str(percent) + "% complete")
        sys.stdout.flush()


# Initialize progress bar
sys.stdout.write("\r0% complete")
sys.stdout.flush()

# Run the model
print('\nRunning model...')
task.run(
    inputs=stimulus_dictionary,
    num_trials=trials,
    call_after_trial=record_trial
)

# Plot results of all units into one figure ---------------------------------------------------------------------------
if args.enable_plot:
    import matplotlib.pyplot as plt

    print('\nModel run, generating plots...')

    # Create x axis "t" for plotting
    t = np.arange(0.0, 20.02, 0.02)

    # Plot target unit, distraction unit, response unit, h(v), and u using the values that were recorded after each trial
    plt.plot(
        t,
        decision_layer_target_values,
        label="target unit",
        color='green'
    )
    plt.plot(
        t,
        decision_layer_distractor_values,
        label="distraction unit",
        color='red'
    )
    plt.plot(
        t,
        response_layer_values,
        label="response unit",
        color='magenta'
    )
    plt.plot(
        t,
        LC_results_h_of_v,
        label="h(v)",
        color='b'
    )
    plt.plot(
        t,
        LC_results_u,
        label="u",
        color='black'
    )

    plt.xlabel('Time')
    plt.ylabel('Activation')
    plt.legend(loc='upper left')
    plt.xlim((0.0, 20.0))
    plt.ylim((-0.2, 1.2))
    x_values = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    plt.xticks(x_values)
    plt.title('GILZENRAT 2002 PsyNeuLink', fontweight='bold')

    plt.show()

    task.show_graph()
    print('\nPlots generated')
