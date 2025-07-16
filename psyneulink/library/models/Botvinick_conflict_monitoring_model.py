import argparse

import numpy as np
import psyneulink as pnl

# This script implements Figure 1 of Botvinick, M. M., Braver, T. S., Barch, D. M., Carter, C. S., & Cohen, J. D. (2001).
# Conflict monitoring and cognitive control. Psychological Review, 108, 624–652.
# http://dx.doi.org/10.1037/0033-295X.108.3.624

# Figure 1 plots the ENERGY computed by a conflict mechanism. It is highest for incongruent trials,
# and similar for congruent and neutral trials.
# Noise is turned off and for each condition we ran one trial only. A response threshold was not defined. Responses were
# made at the marked * signs in the figure.
# Note that this script implements a slightly different Figure than in the original Figure in the paper.
# However, this implementation is identical with a plot we created with an old MATLAB code which was used for the
# conflict monitoring simulations.

parser = argparse.ArgumentParser()
parser.add_argument('--no-plot', action='store_false', help='Disable plotting', dest='enable_plot')
args = parser.parse_args()

# Define Variables ----------------------------------------------------------------------------------------------------

RATE = .01  # As in Cohen-huston text
BIAS = 4.0  # bias 4.0 is -4.0 in the paper see Docs for description. Bbias is positive since Logistic equation has
# minus sing already implemented

# Create mechanisms ---------------------------------------------------------------------------------------------------
# Input layers (linear, is the default function)
#  Color: ('red', 'green', 'grey'),
color_input_layer = pnl.ProcessingMechanism(
    input_shapes=3,
    name='COLOR_INPUT')

#  Word: ('RED','GREEN', 'NEUTRAL')
word_input_layer = pnl.ProcessingMechanism(
    input_shapes=3,
    name='WORD_INPUT')

#  Task: ('color_name', 'word_reading')
task_input_layer = pnl.ProcessingMechanism(
    input_shapes=2,
    name='TASK_INPUT')

# Hidden Layers (logistic)
#  Task
task_layer = pnl.RecurrentTransferMechanism(
    input_shapes=2,
    function=pnl.Logistic(),
    hetero=-2,
    integrator_mode=True,
    integration_rate=RATE,
    name='TASK_HIDDEN')

#  Color
color_hidden_layer = pnl.RecurrentTransferMechanism(
    input_shapes=3,
    function=pnl.Logistic(x_0=BIAS),
    integrator_mode=True,
    hetero=-2,
    integration_rate=RATE,
    name='COLOR_HIDDEN')

#  Word
word_hidden_layer = pnl.RecurrentTransferMechanism(
    input_shapes=3,
    function=pnl.Logistic(x_0=BIAS),
    integrator_mode=True,
    hetero=-2,
    integration_rate=RATE,
    name='WORD_HIDDEN')

# Output Layer
#   Response layer, responses: ('red', 'green')
response_layer = pnl.RecurrentTransferMechanism(
    input_shapes=2,
    function=pnl.Logistic(),
    hetero=-2.0,
    integrator_mode=True,
    integration_rate=RATE,
    output_ports=[
        pnl.RESULT,
        {pnl.NAME: 'DECISION_ENERGY',
         pnl.VARIABLE: (pnl.OWNER_VALUE, 0),
         pnl.FUNCTION: pnl.Stability(
             default_variable=np.array([0.0, 0.0]),
             metric=pnl.ENERGY,
             matrix=np.array([[0.0, -4.0],
                              [-4.0, 0.0]]))}],
    name='RESPONSE')

# Log -----------------------------------------------------------------------------------------------------------------
response_layer.log.set_log_conditions('DECISION_ENERGY')

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

# Control trial ('NEUTRAL' written in grey)
CN_control_trial_input = trial_dict(
    1, 0, 0, 0, 0, 1, 1, 0)  # red_color, green color, red_word, green word, CN, WR

# Trial sequence
Stimulus = [[CN_trial_initialize_input, CN_congruent_trial_input],
            [CN_trial_initialize_input, CN_incongruent_trial_input],
            [CN_trial_initialize_input, CN_control_trial_input]]

# Run the Composition -------------------------------------------------------------------------------------------------
ntrials0 = 500
ntrials = 1000
condition = 3
for cond in range(condition):
    # Initialize
    System_Conflict_Monitoring.run(inputs=Stimulus[cond][0], num_trials=ntrials0)  # run System with initial input
    System_Conflict_Monitoring.run(inputs=Stimulus[cond][1], num_trials=ntrials)  # run System with condition input
    # Reinitialize composition after condition was run
    color_hidden_layer.reset([[0, 0, 0]], context=System_Conflict_Monitoring)
    word_hidden_layer.reset([[0, 0, 0]], context=System_Conflict_Monitoring)
    response_layer.reset([[0, 0]], context=System_Conflict_Monitoring)
    task_layer.reset([[0, 0]], context=System_Conflict_Monitoring)

# Results -------------------------------------------------------------------------------------------------------------
r2 = response_layer.log.nparray_dictionary('DECISION_ENERGY')[
    System_Conflict_Monitoring.name]  # get logged DECISION_ENERGY dictionary
energy = r2['DECISION_ENERGY']  # save logged DECISION_ENERGY

# Plot ----------------------------------------------------------------------------------------------------------------
if args.enable_plot:
    import matplotlib.pyplot as plt

    plt.figure()
    x = np.arange(0, 1500, 1)  # create x-axis length
    plt.plot(x, energy[:1500], 'r')  # plot congruent condition
    plt.plot(x, energy[1500:3000], 'b')  # plot incongruent condition
    plt.plot(x, energy[3000:4500], 'g')  # plot neutral condition
    plt.ylabel('ENERGY')  # add ylabel
    plt.xlabel('cycles')  # add x label
    legend = ['congruent', 'incongruent', 'neutral']
    plt.legend(legend)
    plt.show()
