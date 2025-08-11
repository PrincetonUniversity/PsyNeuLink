import argparse

import numpy as np
import psyneulink as pnl

# This implements the model by Cohen, J. D., & Huston, T. A. (1994). Progress in the use of interactive
# models for understanding attention and performance. In C. Umilta & M. Moscovitch(Eds.),
# AttentionandperformanceXV(pp.453-456). Cam- bridge, MA: MIT Press.
# The model aims to capture top-down effects of selective attention and the bottom-up effects of attentional capture.

parser = argparse.ArgumentParser()
parser.add_argument(
    '--no-plot', action='store_false',
    help='Disable plotting', dest='enable_plot')
parser.add_argument(
    '--threshold', type=float,
    help='Termination threshold for response output (default: %(default)f)', default=0.6)
parser.add_argument(
    '--settle-trials', type=int,
    help='Number of trials for composition to initialize and settle (default: %(default)d)', default=500)
args = parser.parse_args()

# Define Parameters ---------------------------------------------------------------------------------------------------
RATE = 0.01  # Integration rate (called time constant in paper)
INHIBITION = -2.0  # lateral inhibition
BIAS = 4.0  # In contrast to the paper this is negative, since the logistic function's x_0 already includes a negation
THRESHOLD = args.threshold  # default is .5
SETTLE_TRIALS = args.settle_trials  # Not specified in paper; 500 is a reasonable value to allow settling

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
    # noise=pnl.NormalDist(mean=0.0, standard_deviation=.0),
    integration_rate=RATE,
    name='COLOR HIDDEN'
)

word_hidden_layer = pnl.RecurrentTransferMechanism(
    input_shapes=3,
    function=pnl.Logistic(x_0=BIAS),
    hetero=INHIBITION,
    integrator_mode=True,
    # noise=pnl.NormalDist(mean=0.0, standard_deviation=.05),
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

# Responses: ('red', 'green'):
response_layer = pnl.RecurrentTransferMechanism(
    input_shapes=2,
    function=pnl.Logistic(),
    hetero=INHIBITION,
    integrator_mode=True,
    integration_rate=RATE,
    name='RESPONSE'
)

# Log mechanisms ------------------------------------------------------------------------------------------------------
task_layer.set_log_conditions('value')
color_hidden_layer.set_log_conditions('value')
word_hidden_layer.set_log_conditions('value')
response_layer.set_log_conditions('value')

# Create Projections --------------------------------------------------------------------------------------------------

# Input Projections
# Note: Projections from input to hidden layers are identity matrices which are set by PsyNeuLink by default, we
# do not need to explicitly define them here.

# Bidirectional Projections
# To define the bidirectional connections between layers, we create pairs of MappingProjections with appropriate
# weight matrices.

# Task to Color (and vice versa)
task_color_weights = pnl.MappingProjection(
    matrix=np.array([
        [4.0, 4.0, 4.0],
        [0.0, 0.0, 0.0]
    ])
)

color_task_weights = pnl.MappingProjection(
    matrix=np.array([
        [4.0, 0.0],
        [4.0, 0.0],
        [4.0, 0.0]
    ])
)

# Task to Word (and vice versa)
task_word_weights = pnl.MappingProjection(
    matrix=np.array([
        [0.0, 0.0, 0.0],
        [4.0, 4.0, 4.0]
    ])
)

word_task_weights = pnl.MappingProjection(
    matrix=np.array([
        [0.0, 4.0],
        [0.0, 4.0],
        [0.0, 4.0]
    ])
)

# Color to response (and vice versa)
color_response_weights = pnl.MappingProjection(
    matrix=np.array([
        [1.5, 0.0],
        [0.0, 1.5],
        [0.0, 0.0]
    ])
)

response_color_weights = pnl.MappingProjection(
    matrix=np.array([
        [1.5, 0.0, 0.0],
        [0.0, 1.5, 0.0]
    ])
)

# Word to response (and vice versa)
word_response_weights = pnl.MappingProjection(
    matrix=np.array([
        [2.5, 0.0],
        [0.0, 2.5],
        [0.0, 0.0]
    ])
)

response_word_weights = pnl.MappingProjection(
    matrix=np.array([
        [2.5, 0.0, 0.0],
        [0.0, 2.5, 0.0]
    ])
)

# Create Pathways -----------------------------------------------------------------------------------------------------

# Bidirectional pathways between color and response
color_response_pathway_1 = pnl.Pathway(
    pathway=[
        color_input_layer,
        color_hidden_layer,
        color_response_weights,
        response_layer,
    ],
    name='COLOR_RESPONSE_PATHWAY_1'
)

color_response_pathway_2 = pnl.Pathway(
    pathway=[
        response_layer,
        response_color_weights,
        color_hidden_layer
    ],
    name='COLORS_RESPONSE_PATHWAY_2'
)

# Bidirectional pathways between word and response
word_response_pathway_1 = pnl.Pathway(
    pathway=[
        word_input_layer,
        word_hidden_layer,
        word_response_weights,
        response_layer
    ],
    name='WORDS_RESPONSE_PATHWAY_1'
)

word_response_pathway_2 = pnl.Pathway(
    pathway=[
        response_layer,
        response_word_weights,
        word_hidden_layer
    ],
    name='WORDS_RESPONSE_PATHWAY_2'
)

# Bidirectional pathways between task and color
task_color_response_pathway_1 = pnl.Pathway(
    pathway=[
        task_input_layer,
        task_layer,
        task_color_weights,
        color_hidden_layer
    ],
    name='TASK_COLOR_RESPONSE_PATHWAY_1'
)

task_color_response_pathway_2 = pnl.Pathway(
    pathway=[
        color_hidden_layer,
        color_task_weights,
        task_layer])

# Bidirectional pathways between task and word
task_word_response_pathway_1 = pnl.Pathway(
    pathway=[
        task_input_layer,
        task_layer,
        task_word_weights,
        word_hidden_layer])

task_word_response_pathway_2 = pnl.Pathway(
    pathway=[
        word_hidden_layer,
        word_task_weights,
        task_layer])

# Create Composition --------------------------------------------------------------------------------------------------
Bidirectional_Stroop = pnl.Composition(
    pathways=[
        color_response_pathway_1,
        word_response_pathway_1,
        task_color_response_pathway_1,
        task_word_response_pathway_1,
        color_response_pathway_2,
        word_response_pathway_2,
        task_color_response_pathway_2,
        task_word_response_pathway_2
    ],
    reinitialize_mechanisms_when=pnl.Never(),
    name='Bidirectional Stroop Model'
)

input_dict = {color_input_layer: [0, 0, 0],
              word_input_layer: [0, 0, 0],
              task_input_layer: [0, 1]}
print("\n\n\n\n")
print(Bidirectional_Stroop.run(inputs=input_dict))

for node in Bidirectional_Stroop.mechanisms:
    print(node.name, " Value: ", node.get_output_values(Bidirectional_Stroop))

# # LOGGING:
color_hidden_layer.set_log_conditions('value')
word_hidden_layer.set_log_conditions('value')

# Create threshold function -------------------------------------------------------------------------------------------
# Note if either one of the response units crosses threshold, terminate the trial
terminate_trial = {
    pnl.TimeScale.TRIAL: pnl.Or(
        pnl.Threshold(response_layer, 'value', THRESHOLD, '>=', (0, 0)),
        pnl.Threshold(response_layer, 'value', THRESHOLD, '>=', (0, 1)),
    )
}


# Create test trials function -----------------------------------------------------------------------------------------
# a BLUE word input is [1,0] to words_input_layer and GREEN word is [0,1]
# a blue color input is [1,0] to colors_input_layer and green color is [0,1]
# a color-naming trial is [1,0] to task_layer and a word-reading trial is [0,1]


def trial_dict(red_color, green_color, neutral_color, red_word, green_word, neutral_word, CN, WR):
    return {
        color_input_layer: [red_color, green_color, neutral_color],
        word_input_layer: [red_word, green_word, neutral_word],
        task_input_layer: [CN, WR]
    }


# Define initialization trials separately
WR_initialize_input = trial_dict(
    0, 0, 0, 0, 0, 0, 0, 1)
CN_initialize_input = trial_dict(
    0, 0, 0, 0, 0, 0, 1, 0)

CN_incongruent_trial_input = trial_dict(
    1, 0, 0, 0, 1, 0, 1, 0)
CN_congruent_trial_input = trial_dict(
    1, 0, 0, 1, 0, 0, 1, 0)
CN_control_trial_input = trial_dict(
    1, 0, 0, 0, 0, 0, 1, 0)

WR_congruent_trial_input = trial_dict(
    1, 0, 0, 1, 0, 0, 0, 1)
WR_incongruent_trial_input = trial_dict(
    1, 0, 0, 0, 1, 0, 0, 1)
WR_control_trial_input = trial_dict(
    0, 0, 0, 1, 0, 0, 0, 1)

Stimulus_Color_Naming = [
    [CN_initialize_input, CN_control_trial_input],
    [CN_initialize_input, CN_incongruent_trial_input],
    [CN_initialize_input, CN_congruent_trial_input]
]

Stimulus_Word_Reading = [
    [WR_initialize_input, WR_control_trial_input],
    [WR_initialize_input, WR_incongruent_trial_input],
    [WR_initialize_input, WR_control_trial_input]
]

conditions = 3
response_all = []
response_all2 = []


def set_response_weights_to_zero():
    """
    Set all bidirectional response weights to zero for settling period
    """
    color_response_weights.parameters.matrix.set(
        np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0]
        ]), Bidirectional_Stroop
    )

    word_response_weights.parameters.matrix.set(
        np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0]
        ]), Bidirectional_Stroop
    )

    response_word_weights.parameters.matrix.set(
        np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]), Bidirectional_Stroop
    )

    response_color_weights.parameters.matrix.set(
        np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]), Bidirectional_Stroop
    )


def reset_response_weights():
    """
    Restore bidirectional response weights to original values after settling period
    """
    color_response_weights.parameters.matrix.set(
        np.array([
            [1.5, 0.0],
            [0.0, 1.5],
            [0.0, 0.0]
        ]), Bidirectional_Stroop
    )
    word_response_weights.parameters.matrix.set(
        np.array([
            [2.5, 0.0],
            [0.0, 2.5],
            [0.0, 0.0]
        ]), Bidirectional_Stroop)

    response_color_weights.parameters.matrix.set(
        np.array([
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0]
        ]), Bidirectional_Stroop
    )

    response_word_weights.parameters.matrix.set(
        np.array([
            [2.5, 0.0, 0.0],
            [0.0, 2.5, 0.0]
        ]), Bidirectional_Stroop
    )


# Run color naming trials ----------------------------------------------------------------------------------------------
for cond in range(conditions):
    # For the settling period, we set all bidirectional weights to and from the response layer to 0
    set_response_weights_to_zero()

    # Run the model for settling period
    Bidirectional_Stroop.run(inputs=Stimulus_Color_Naming[cond][0], num_trials=SETTLE_TRIALS)

    # After settling, we restore the bidirectional weights to their original values
    reset_response_weights()

    # Run the model for the actual trial until the threshold is reached
    Bidirectional_Stroop.run(inputs=Stimulus_Color_Naming[cond][1], termination_processing=terminate_trial)

    # Store values from run -----------------------------------------------------------------------------------------------
    B_S = Bidirectional_Stroop.name
    r = response_layer.log.nparray_dictionary('value')  # Log response output from special logistic function
    rr = r[B_S]['value']
    n_r = rr.shape[0]
    rrr = rr.reshape(n_r, 2)
    response_all.append(rrr)  # .shape[0])
    response_all2.append(rrr.shape[0])

    # Clear log & reset ----------------------------------------------------------------------------------------
    response_layer.log.clear_entries()
    color_hidden_layer.log.clear_entries()
    word_hidden_layer.log.clear_entries()
    task_layer.log.clear_entries()
    color_hidden_layer.reset([[0, 0, 0]])
    word_hidden_layer.reset([[0, 0, 0]])
    response_layer.reset([[0, 0]])
    task_layer.reset([[0, 0]])
    print('response_all: ', response_all)

# Run word reading trials ----------------------------------------------------------------------------------------------
response_all3 = []
response_all4 = []
for cond in range(conditions):
    set_response_weights_to_zero()
    Bidirectional_Stroop.run(inputs=Stimulus_Word_Reading[cond][0], num_trials=SETTLE_TRIALS)
    reset_response_weights()
    Bidirectional_Stroop.run(inputs=Stimulus_Word_Reading[cond][1], termination_processing=terminate_trial)

    # Store values from run -----------------------------------------------------------------------------------------------
    r2 = response_layer.log.nparray_dictionary('value')  # Log response output from special logistic function
    rr2 = r2[Bidirectional_Stroop.name]['value']
    n_r2 = rr2.shape[0]
    rrr2 = rr2.reshape(n_r2, 2)
    response_all3.append(rrr2)  # .shape[0])
    response_all4.append(rrr2.shape[0])

    # Clear log & reset ----------------------------------------------------------------------------------------
    response_layer.log.clear_entries()
    color_hidden_layer.log.clear_entries()
    word_hidden_layer.log.clear_entries()
    task_layer.log.clear_entries()
    color_hidden_layer.reset([[0, 0, 0]])
    word_hidden_layer.reset([[0, 0, 0]])
    response_layer.reset([[0, 0]])
    task_layer.reset([[0, 0]])
    print('response_all: ', response_all)

if args.enable_plot:
    import matplotlib.pyplot as plt

    # Plot results --------------------------------------------------------------------------------------------------------
    # First, plot response layer activity for whole run
    plt.figure()
    # color naming plot
    plt.plot(response_all[0])
    plt.plot(response_all[1])
    plt.plot(response_all[2])

    # word reading plot
    plt.plot(response_all3[0])
    plt.plot(response_all3[1])
    plt.plot(response_all3[2])
    plt.show(block=not pnl._called_from_pytest)

    # -- Regression --

    # Human data, From Dunbar & MacLeod, 1984:
    # "A Horse Race of a Different Color: Stroop Interference Patterns with Transformed Words"

    rt_control_color_naming = 656
    rt_incongruent_color_naming = 856
    rt_congruent_color_naming = 590

    rt_control_word_reading = 496
    rt_incongruent_word_reading = 518
    rt_congruent_word_reading = 500

    # Arrange human data to match predicted data
    y = np.array([
        rt_control_color_naming,
        rt_incongruent_color_naming,
        rt_congruent_color_naming,
        rt_control_word_reading,
        rt_incongruent_word_reading,
        rt_congruent_word_reading
    ], dtype=float)

    response_all2 = np.asarray(response_all2, dtype=float).ravel()
    response_all4 = np.asarray(response_all4, dtype=float).ravel()

    # Fit scalar slope + intercept
    X = np.concatenate([response_all2, response_all4])  # (6,)
    X_int = np.column_stack([X, np.ones_like(X)])  # (6, 2)
    params, *_ = np.linalg.lstsq(X_int, y + SETTLE_TRIALS, rcond=None)
    slope, intercept = params

    # Predict back in your original form
    reg = response_all2 * slope + intercept - SETTLE_TRIALS
    reg2 = response_all4 * slope + intercept - SETTLE_TRIALS

    plt.figure()

    plt.plot(reg, '-s')  # plot color naming
    plt.plot(reg2, '-or')  # plot word reading
    plt.title('GRAIN MODEL with Bidirectional Weights')
    plt.legend(['Color Naming', 'Word Reading'])
    plt.xticks(np.arange(3), ('Control', 'Conflict', 'Congruent'))
    plt.ylabel('Reaction Time (ms)')
    plt.show(block=not pnl._called_from_pytest)
