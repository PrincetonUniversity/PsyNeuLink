import argparse

import numpy as np
import psyneulink as pnl

# This script implements Figure 1 of Botvinick, M. M., Braver, T. S., Barch, D. M., Carter, C. S., & Cohen, J. D. (2001).
# Conflict monitoring and cognitive control. Psychological Review, 108, 624–652.
# http://dx.doi.org/10.1037/0033-295X.108.3.624

# Figure 1 plots the ENERGY computed by a conflict mechanism. It is highest for incongruent trials,
# and similar for congruent and neutral trials.

# DIFFERENCES FROM ORIGINAL FIGURE -----------------------------------------------------------------------------------

# Noise is turned off and for each condition we ran one trial only. A response threshold was not defined. Responses were
# made at the marked * signs in the figure.

# This produces slightly different values compared to the original figure. In the original simulations,
# the run stopped when a response threshold was reached, and the energy values after that point
# were held constant at the value at the response time. This caused the “leveling off” seen in the original figure.
# In contrast, the simulations in this script always run for a fixed number of time steps, so they do not show that
# post-response flattening.

# Throughout the code, there are commented out sections that can be uncommented to more closely approximate the
# original simulations. However, running with those sections uncommented will be considerably slower and also
# not produce the exact same results since we couldn't determine all the original parameters and how noise
# was applied.


parser = argparse.ArgumentParser()
parser.add_argument('--no-plot', action='store_false', help='Disable plotting', dest='enable_plot')
parser.add_argument(
    '--settle-trials', type=int,
    help='Number of trials for composition to initialize and settle (default: %(default)d)', default=500)
parser.add_argument(
    '--stimulus-trials', type=int,
    help='Number of trials for composition to run with the stimulus (default: %(default)d)', default=600)

args = parser.parse_args()

# Define Parameters ---------------------------------------------------------------------------------------------------
RATE = 0.01  # Integration rate
INHIBITION = -2.0  # lateral inhibition
BIAS = 4.0  # In contrast to the paper this is negative, since the logistic function's x_0 already includes a negation
SETTLE_TRIALS = args.settle_trials
STIMULUS_TRIALS = args.stimulus_trials

# Create mechanisms ---------------------------------------------------------------------------------------------------
# Input layers (Linear)
# Note: These layers don't appear in the original paper, they are just a way to feed inputs into the model

# Colors: ('red', 'green', 'black'),
color_input_layer = pnl.ProcessingMechanism(
    input_shapes=3,
    function=pnl.Linear,
    name='COLOR INPUT'
)

# Words: ('RED', 'GREEN', 'XXXX')
word_input_layer = pnl.ProcessingMechanism(
    input_shapes=3,
    function=pnl.Linear,
    name='WORD INPUT'
)

# Tasks: ('Color Naming', 'Word Reading')
task_input_layer = pnl.ProcessingMechanism(
    input_shapes=2,
    function=pnl.Linear,
    name='TASK INPUT'
)

# Hidden layers (Logistic with recurrent connections for inhibition between units)
# Note: These are the layers presented in the original paper. Here, we use RecurrentTransferMechanism to implement
# the lateral inhibition described there. The auto parameters default is 0, which means no self-excitation.

color_hidden_layer = pnl.RecurrentTransferMechanism(
    input_shapes=3,
    function=pnl.Logistic(x_0=BIAS),
    integrator_mode=True,
    hetero=INHIBITION,
    # noise=pnl.NormalDist(mean=0.0, standard_deviation=.01),  # <-- Uncomment to add noise as in the original paper
    integration_rate=RATE,
    name='COLOR HIDDEN'
)

word_hidden_layer = pnl.RecurrentTransferMechanism(
    input_shapes=3,
    function=pnl.Logistic(x_0=BIAS),
    hetero=INHIBITION,
    # noise=pnl.NormalDist(mean=0.0, standard_deviation=.01),  # <-- Uncomment to add noise as in the original paper
    integrator_mode=True,
    integration_rate=RATE,
    name='WORD HIDDEN'
)

task_layer = pnl.RecurrentTransferMechanism(
    input_shapes=2,
    function=pnl.Logistic(),
    hetero=INHIBITION,
    integrator_mode=True,
    integration_rate=RATE,
    name='TASK DEMAND'
)

# Output Layer
#   Response layer, responses: ('red', 'green')
response_layer = pnl.RecurrentTransferMechanism(
    input_shapes=2,
    function=pnl.Logistic(),
    hetero=INHIBITION,
    integrator_mode=True,
    integration_rate=RATE,
    output_ports=[
        pnl.RESULT,
        {pnl.NAME: 'DECISION ENERGY',
         pnl.VARIABLE: (pnl.OWNER_VALUE, 0),
         pnl.FUNCTION:
             pnl.Stability(
                 default_variable=np.array([0.0, 0.0]),
                 metric=pnl.ENERGY,
                 matrix=np.array([[0.0, -4.0],
                                  [-4.0, 0.0]])
             )
         }
    ],
    name='RESPONSE')

# Log -----------------------------------------------------------------------------------------------------------------
response_layer.log.set_log_conditions('DECISION ENERGY')

# Mapping projections--------------------------------------------------------------------------------------------------
color_input_weights = pnl.MappingProjection(
    matrix=np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]]))

word_input_weights = pnl.MappingProjection(
    matrix=np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]]))

task_input_weights = pnl.MappingProjection(
    matrix=np.array([[1.0, 0.0],
                     [0.0, 1.0]]))

color_task_weights = pnl.MappingProjection(
    matrix=np.array([[4.0, 0.0],
                     [4.0, 0.0],
                     [4.0, 0.0]]))

task_color_weights = pnl.MappingProjection(
    matrix=np.array([[4.0, 4.0, 4.0],
                     [0.0, 0.0, 0.0]]))

response_color_weights = pnl.MappingProjection(
    matrix=np.array([[1.5, 0.0, 0.0],
                     [0.0, 1.5, 0.0]]))

response_word_weights = pnl.MappingProjection(
    matrix=np.array([[2.5, 0.0, 0.0],
                     [0.0, 2.5, 0.0]]))

color_response_weights = pnl.MappingProjection(
    matrix=np.array([[1.5, 0.0],
                     [0.0, 1.5],
                     [0.0, 0.0]]))

word_response_weights = pnl.MappingProjection(
    matrix=np.array([[2.5, 0.0],
                     [0.0, 2.5],
                     [0.0, 0.0]]))

word_task_weights = pnl.MappingProjection(
    matrix=np.array([[0.0, 4.0],
                     [0.0, 4.0],
                     [0.0, 4.0]]))

task_word_weights = pnl.MappingProjection(
    matrix=np.array([[0.0, 0.0, 0.0],
                     [4.0, 4.0, 4.0]]))

# Create Pathways -----------------------------------------------------------------------------------------------------
color_response_pathway = [
    color_input_layer,
    color_input_weights,
    color_hidden_layer,
    color_response_weights,
    response_layer,
    response_color_weights,
    color_hidden_layer]

word_response_pathway = [
    word_input_layer,
    word_input_weights,
    word_hidden_layer,
    word_response_weights,
    response_layer,
    response_word_weights,
    word_hidden_layer]

task_color_response_pathway = [
    task_input_layer,
    task_input_weights,
    task_layer,
    task_color_weights,
    color_hidden_layer,
    color_task_weights,
    task_layer]

task_word_response_pathway = [
    task_input_layer,
    task_layer,
    task_word_weights,
    word_hidden_layer,
    word_task_weights,
    task_layer]

# Create Composition -------------------------------------------------------------------------------------------------------
System_Conflict_Monitoring = pnl.Composition(
    pathways=[color_response_pathway,
              word_response_pathway,
              task_color_response_pathway,
              task_word_response_pathway],
    reinitialize_mechanisms_when=pnl.Never(),
    name='CONFLICT MONITORING_SYSTEM')


# This displays a diagram of the Composition (uncomment to see)
# System_Conflict_Monitoring.show_graph(show_dimensions=pnl.ALL)

# Helper function to create a trial dictionary for the trials
def trial_dict(
        red_color,
        green_color,
        neutral_color,
        red_word,
        green_word,
        neutral_word,
        task_color_name,
        task_word_reading):
    """
    Create a trial dictionary for the conflict monitoring model from
    binary inputs for color, word, and task.

    Args:
        red_color: is 1 if red color is present, else 0
        green_color: is 1 if green color is present, else 0
        neutral_color: is 1 if neutral color is present, else 0
        red_word: is 1 if red word is present, else 0
        green_word: is 1 if green word is present, else 0
        neutral_word: is 1 if neutral word is present, else 0
        task_color_name: is 1 if color naming task is present, else 0
        task_word_reading: is 1 if word reading task is present, else 0

    Returns:
        trial_dict: A dictionary mapping input layers to their respective one hot encoded inputs.
    """
    trial = {
        color_input_layer: [red_color, green_color, neutral_color],
        word_input_layer: [red_word, green_word, neutral_word],
        task_input_layer: [task_color_name, task_word_reading],
    }
    return trial


# Define initialization trials separately

# Args: red_color, green color, red_word, green word, CN, WR

# Initialization trial with all inputs set to 0 but task_color_name set to 1
CN_trial_initialize_input = trial_dict(
    0, 0, 0, 0, 0, 0, 1, 0)

# Incongruent trial ('GREEN' written in red)
CN_incongruent_trial_input = trial_dict(
    1, 0, 0, 0, 1, 0, 1, 0)

# Congruent trial ('RED' written in red)
CN_congruent_trial_input = trial_dict(
    1, 0, 0, 1, 0, 0, 1, 0)  # red_color, green color, red_word, green word, CN, WR

# Control trial ('NEUTRAL' written in red)
CN_control_trial_input = trial_dict(
    1, 0, 0, 0, 0, 1, 1, 0)  # red_color, green color, red_word, green word, CN, WR

# Trial sequence
Stimulus = [[CN_trial_initialize_input, CN_congruent_trial_input],
            [CN_trial_initialize_input, CN_incongruent_trial_input],
            [CN_trial_initialize_input, CN_control_trial_input]]


def set_response_weights_to_zero(composition=System_Conflict_Monitoring):
    """
    Set all bidirectional response weights to zero for settling period
    """
    color_response_weights.parameters.matrix.set(
        np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0]
        ]), composition
    )

    word_response_weights.parameters.matrix.set(
        np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0]
        ]), composition
    )

    response_word_weights.parameters.matrix.set(
        np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]), composition
    )

    response_color_weights.parameters.matrix.set(
        np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]), composition
    )


def reset_response_weights(composition=System_Conflict_Monitoring):
    """
    Restore bidirectional response weights to original values after settling period
    """
    color_response_weights.parameters.matrix.set(
        np.array([
            [1.5, 0.0],
            [0.0, 1.5],
            [0.0, 0.0]
        ]), composition
    )
    word_response_weights.parameters.matrix.set(
        np.array([
            [2.5, 0.0],
            [0.0, 2.5],
            [0.0, 0.0]
        ]), composition
    )

    response_color_weights.parameters.matrix.set(
        np.array([
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0]
        ]), composition
    )

    response_word_weights.parameters.matrix.set(
        np.array([
            [2.5, 0.0, 0.0],
            [0.0, 2.5, 0.0]
        ]), composition
    )


# Run the Composition -------------------------------------------------------------------------------------------------
condition = 3

for cond in range(condition):
    # Initialize until settle
    set_response_weights_to_zero(composition=System_Conflict_Monitoring)
    System_Conflict_Monitoring.run(inputs=Stimulus[cond][0], num_trials=SETTLE_TRIALS)  # run System with initial input

    # Run with stimulus
    reset_response_weights(composition=System_Conflict_Monitoring)
    System_Conflict_Monitoring.run(inputs=Stimulus[cond][1], num_trials=STIMULUS_TRIALS)

    # Reinitialize composition after condition was run
    color_hidden_layer.reset([[0, 0, 0]], context=System_Conflict_Monitoring)
    word_hidden_layer.reset([[0, 0, 0]], context=System_Conflict_Monitoring)
    response_layer.reset([[0, 0]], context=System_Conflict_Monitoring)
    task_layer.reset([[0, 0]], context=System_Conflict_Monitoring)

# Results -------------------------------------------------------------------------------------------------------------
r2 = response_layer.log.nparray_dictionary('DECISION ENERGY')[
    System_Conflict_Monitoring.name]  # get logged DECISION_ENERGY dictionary
energy = r2['DECISION ENERGY']  # save logged DECISION_ENERGY

# Plot ----------------------------------------------------------------------------------------------------------------
if args.enable_plot:
    import matplotlib.pyplot as plt

    n = SETTLE_TRIALS + STIMULUS_TRIALS
    plt.figure()
    x = np.arange(0, STIMULUS_TRIALS, 1)  # create x-axis length
    plt.plot(x, energy[n + SETTLE_TRIALS:2 * n], 'b', ls='-')  # plot incongruent condition
    plt.plot(x, energy[2 * n + SETTLE_TRIALS:3 * n], 'g', ls='--')  # plot neutral condition
    plt.plot(x, energy[SETTLE_TRIALS:n], 'r', ls=':')  # plot congruent condition
    plt.ylabel('Energy')  # add ylabel
    plt.xlabel('Cycle')  # add x label
    legend = ['Incongruent', 'Neutral', 'Congruent']
    plt.legend(legend)
    plt.show()

# For more closely mimicking the original figure run the simulation as follows ----------------------------------------
# Note: Also uncomment the noise in the mechanism definitions above
#
# THRESHOLD = .6
# SIMULATIONS_PER_CONDITION = 100
#
# terminate_trial = {
#     pnl.TimeScale.TRIAL: pnl.Or(
#         pnl.Threshold(response_layer, 'value', THRESHOLD, '>=', (0, 0)),
#         pnl.Threshold(response_layer, 'value', THRESHOLD, '>=', (0, 1)),
#     )
# }
#
# c_1 = []
# c_2 = []
# c_3 = []
#
# for cond in range(condition):
#     for _ in range(SIMULATIONS_PER_CONDITION):
#         # Re-create Composition to reset logs
#         System_Conflict_Monitoring = pnl.Composition(
#             pathways=[color_response_pathway,
#                       word_response_pathway,
#                       task_color_response_pathway,
#                       task_word_response_pathway],
#             reinitialize_mechanisms_when=pnl.Never(),
#             name='CONFLICT MONITORING_SYSTEM')
#         # Initialize
#         set_response_weights_to_zero(composition=System_Conflict_Monitoring)
#
#         System_Conflict_Monitoring.run(inputs=Stimulus[cond][0],
#                                        num_trials=SETTLE_TRIALS)  # run System with initial input
#
#         reset_response_weights(composition=System_Conflict_Monitoring)
#
#         System_Conflict_Monitoring.run(inputs=Stimulus[cond][1],
#                                        termination_processing=terminate_trial)
#
#         # Reinitialize composition after condition was run
#         r2 = response_layer.log.nparray_dictionary('DECISION ENERGY')[System_Conflict_Monitoring.name][
#             'DECISION ENERGY']
#
#         n_eff = SETTLE_TRIALS + STIMULUS_TRIALS
#         # fill up to n trials with last value to make plotting easier
#         r2 = np.squeeze(r2)  # shape (n_steps,)
#
#         # Pad with last value if needed
#         need = n_eff - r2.shape[0]
#         if need > 0:
#             r2 = np.concatenate([r2, np.full(need, r2[-1])])
#
#         # Cap length to n_eff
#         r2 = r2[:n_eff]
#
#         # Store
#         if cond == 0:
#             c_1.append(r2)
#         elif cond == 1:
#             c_2.append(r2)
#         elif cond == 2:
#             c_3.append(r2)
#
# if args.enable_plot:
#     import matplotlib.pyplot as plt
#
#     n = SETTLE_TRIALS + STIMULUS_TRIALS
#     plt.figure()
#     x = np.arange(0, n, 1)  # create x-axis length
#     # plot means of conditions
#     plt.plot(x, np.mean(c_1, axis=0), 'r', label='congruent')
#     plt.plot(x, np.mean(c_2, axis=0), 'b', label='incongruent')
#     plt.plot(x, np.mean(c_3, axis=0), 'g', label='neutral')
#     plt.ylabel('Energy')  # add ylabel
#     plt.xlabel('Cycle')  # add x label
#     legend = ['congruent', 'incongruent', 'neutral']
#     plt.legend(legend)
#     plt.show()
