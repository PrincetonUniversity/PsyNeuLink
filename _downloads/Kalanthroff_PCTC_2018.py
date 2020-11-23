import argparse

import numpy as np
import psyneulink as pnl

parser = argparse.ArgumentParser()
parser.add_argument('--no-plot', action='store_false', help='Disable plotting', dest='enable_plot')
parser.add_argument('--threshold', type=float, help='Termination threshold for response output (default: %(default)f)', default=0.70)
parser.add_argument('--settle-trials', type=int, help='Number of trials for composition to initialize and settle (default: %(default)d)', default=200)
args = parser.parse_args()

# Implements the Kalanthroff, Davelaar, Henik, Goldfarb & Usher model: Task Conflict and Proactive Control:
# A Computational Theory of the Stroop Task. Psychol Rev. 2018 Jan;125(1):59-82. doi: 10.1037/rev0000083.
# Epub 2017 Oct 16.
# #https://www.ncbi.nlm.nih.gov/pubmed/29035077

# Define Variables ----------------------------------------------------------------------------------------------------
Lambda = 0.03            # PsyNeuLink has Euler integration constant reversed (1-0.97)
pc_high = 0.15           # High proactive control from Figure 6 in Paper
pc_low = 0.025           # Low proactive control from Figure 6 in Paper
pc = pc_low              # Select proactive control
inhibition = -1.3        # Inhibition between units within a layer
inhibition_task = -1.9   # Inhibition between units within task layer
bias = -0.3              # bias input to color feature layer and word feature layer
threshold = args.threshold
settle = args.settle_trials    # Number of trials until Composition settles

# Create mechanisms ---------------------------------------------------------------------------------------------------
# 4 Input layers for color, word, task & bias
colors_input_layer = pnl.TransferMechanism(
    size=2,
    function=pnl.Linear,
    name='COLORS_INPUT'
)

words_input_layer = pnl.TransferMechanism(
    size=2,
    function=pnl.Linear,
    name='WORDS_INPUT'
)

task_input_layer = pnl.TransferMechanism(
    size=2,
    function=pnl.Linear,
    name='PROACTIVE_CONTROL'
)

bias_input = pnl.TransferMechanism(
    size=2,
    function=pnl.Linear,
    name='BIAS'
)

# Built python function to ensure that the logistic function outputs 0 when input is <= 0


def my_special_Logistic(variable):
    maxi = variable - 0.0180
    output = np.fmax([0], maxi)
    return output

# Built python function that takes output of special logistic function and computes conflict by multiplying
# output both task units with each over times 500


def my_conflict_function(variable):
    maxi = variable - 0.0180
    new = np.fmax([0], maxi)
    out = [new[0] * new[1] * 500]
    return out


# Create color feature layer, word feature layer, task demand layer and response layer
color_feature_layer = pnl.RecurrentTransferMechanism(
    size=2,                     # Define unit size
    function=pnl.Logistic(gain=4, x_0=1),       # to 4 & bias to 1
    integrator_mode=True,       # Set IntegratorFunction mode to True
    integration_rate=Lambda,    # smoothing factor ==  integration rate
    hetero=inhibition,          # Inhibition among units within a layer
    output_ports=[{                          # Create new OutputPort by applying
        pnl.NAME: 'SPECIAL_LOGISTIC',         # the "my_special_Logistic" function
        pnl.VARIABLE: (pnl.OWNER_VALUE, 0),
        pnl.FUNCTION: my_special_Logistic
    }],
    name='COLOR_LAYER')

# The word_feature_layer is set up as the color_feature_layer
word_feature_layer = pnl.RecurrentTransferMechanism(
    size=2,                     # Define unit size
    function=pnl.Logistic(gain=4, x_0=1),            # to 4 & bias to 1
    integrator_mode=True,   # Set IntegratorFunction mode to True
    integration_rate=Lambda,  # smoothing factor ==  integration rate
    hetero=inhibition,      # Inhibition among units within a layer
    output_ports=[{              # Create new OutputPort by applying
        pnl.NAME: 'SPECIAL_LOGISTIC',        # the "my_special_Logistic" function
        pnl.VARIABLE: (pnl.OWNER_VALUE, 0),
        pnl.FUNCTION: my_special_Logistic
    }],
    name='WORD_LAYER')


# The response_layer is set up as the color_feature_layer & the word_feature_layer
response_layer = pnl.RecurrentTransferMechanism(
    size=2,                         # Define unit size
    function=pnl.Logistic(gain=4, x_0=1),           # to 4 & bias to 1
    integrator_mode=True,           # Set IntegratorFunction mode to True
    integration_rate=Lambda,        # smoothing factor ==  integration rate
    hetero=inhibition,              # Inhibition among units within a layer
    output_ports=[{           # Create new OutputPort by applying
        pnl.NAME: 'SPECIAL_LOGISTIC',        # the "my_special_Logistic" function
        pnl.VARIABLE: (pnl.OWNER_VALUE, 0),
        pnl.FUNCTION: my_special_Logistic
    }],
    name='RESPONSE_LAYER'
)

# The task_demand_layer is set up as the color_feature_layer but with a different python function on it's OutputPort
# and a differnet inhibition weight on the hetero
task_demand_layer = pnl.RecurrentTransferMechanism(
    size=2,                      # Define unit size
    function=pnl.Logistic(gain=4, x_0=1),            # to 4 & bias to 1
    integrator_mode=True,   # Set IntegratorFunction mode to True
    integration_rate=Lambda,  # smoothing factor ==  integration rate
    hetero=inhibition_task,  # Inhibition among units within a layer
    output_ports=[               # Create new OutputPort by applying
        {
            pnl.NAME: 'SPECIAL_LOGISTIC',        # the "my_conflict_function" function
            pnl.VARIABLE: (pnl.OWNER_VALUE, 0),
            pnl.FUNCTION: my_special_Logistic
        },
        {
            pnl.NAME: 'CONFLICT',
            pnl.VARIABLE: (pnl.OWNER_VALUE, 0),
            pnl.FUNCTION: my_conflict_function
        }
    ],
    name='TASK_LAYER'
)


# Log mechanisms ------------------------------------------------------------------------------------------------------
color_feature_layer.set_log_conditions('SPECIAL_LOGISTIC')       # Log output of my_special_Logistic function
word_feature_layer.set_log_conditions('SPECIAL_LOGISTIC')        # Log output of my_special_Logistic function
response_layer.set_log_conditions('SPECIAL_LOGISTIC')            # Log output of my_special_Logistic function
task_demand_layer.set_log_conditions('SPECIAL_LOGISTIC')         # Log output of my_special_Logistic function

task_demand_layer.set_log_conditions('CONFLICT')                 # Log outout of my_conflict_function function

# Connect mechanisms --------------------------------------------------------------------------------------------------
color_input_weights = pnl.MappingProjection(
    matrix=np.array([
        [0.0, 0.0],     # response layer projections are set to all
        [0.0, 0.0]      # zero for initialization period first
    ])
)
word_input_weights = pnl.MappingProjection(
    matrix=np.array([
        [0.0, 0.0],
        [0.0, 0.0]
    ])
)
color_task_weights = pnl.MappingProjection(
    matrix=np.array([
        [2.0, 0.0],     # color to task projection
        [2.0, 0.0]
    ])
)
word_task_weights = pnl.MappingProjection(
    matrix=np.array([
        [0.0, 2.0],     # word to task projection
        [0.0, 2.0]
    ])
)
task_color_weights = pnl.MappingProjection(
    matrix=np.array([
        [1.0, 1.0],     # task to color projection
        [0.0, 0.0]
    ])
)
task_word_weights = pnl.MappingProjection(
    matrix=np.array([
        [0.0, 0.0],     # task to word projection
        [1.0, 1.0]
    ])
)
color_response_weights = pnl.MappingProjection(
    matrix=np.array([
        [2.0, 0.0],     # color to response projection
        [0.0, 2.0]
    ])
)
word_response_weights = pnl.MappingProjection(
    matrix=np.array([
        [2.5, 0.0],     # word to response projection
        [0.0, 2.5]
    ])
)
task_input_weights = pnl.MappingProjection(
    matrix=np.array([
        [1.0, 0.0],     # proactive control to task
        [0.0, 1.0]
    ])
)

# to send a control signal from the task demand layer to the response layer,
# set matrix to -1 to reduce response layer activation
# specify the sender of the projection which is the second OutputPort the task demand layer
# specify the receiver of the projection
task_conflict_to_response_weights = pnl.MappingProjection(
    matrix=np.array([[-1.0, -1.0]]),
    sender=task_demand_layer.output_ports[1],
    receiver=response_layer
)

# Create pathways -----------------------------------------------------------------------------------------------------
color_response_process = pnl.Pathway(pathway=[
    colors_input_layer,
    color_input_weights,
    color_feature_layer,
    color_response_weights,
    response_layer
],
    name='COLORS_RESPONSE_PROCESS'
)

word_response_process = pnl.Pathway(
    pathway=[words_input_layer,
             word_input_weights,
             word_feature_layer,
             word_response_weights,
             response_layer
             ],
    name='WORDS_RESPONSE_PROCESS'
)

task_color_process = pnl.Pathway(
    pathway=[task_input_layer,
             task_input_weights,
             task_demand_layer,
             task_color_weights,
             color_feature_layer,
             color_task_weights,
             task_demand_layer
             ],
    name='TASK_COLOR_PROCESS'
)

task_word_process = pnl.Pathway(
    pathway=[task_input_layer,
             task_demand_layer,
             task_word_weights,
             word_feature_layer,
             word_task_weights,
             task_demand_layer
             ],
    name='TASK_WORD_PROCESS'
)

bias_color_process = pnl.Pathway(
    pathway=[bias_input, color_feature_layer],
    name='BIAS_COLOR'
)

bias_word_process = pnl.Pathway(
    pathway=[bias_input, word_feature_layer],
    name='WORD_COLOR'
)

conflict_process = pnl.Pathway(
    pathway=[
        task_demand_layer,
        task_conflict_to_response_weights,
        response_layer
    ],
    name='CONFLICT_PROCESS'
)

# Create Composition --------------------------------------------------------------------------------------------------
PCTC = pnl.Composition(
    pathways=[
        word_response_process,
        color_response_process,
        task_color_process,
        task_word_process,
        bias_word_process,
        bias_color_process,
        task_word_process,
        conflict_process
    ],
    reinitialize_mechanisms_when=pnl.Never(),
    name='PCTC_MODEL')

# Create threshold function -------------------------------------------------------------------------------------------

def pass_threshold(response_layer, thresh, context):
    results1 = response_layer.get_output_values(context)[0][0]  # red response
    results2 = response_layer.get_output_values(context)[0][1]  # green response
    # print(results1)
    # print(results2)
    if results1 >= thresh or results2 >= thresh:
        return True
    return False


terminate_trial = {
    pnl.TimeScale.TRIAL: pnl.While(pass_threshold, response_layer, threshold)
}

# Create test trials function -----------------------------------------------------------------------------------------
# a BLUE word input is [1,0] to words_input_layer and GREEN word is [0,1]
# a blue color input is [1,0] to colors_input_layer and green color is [0,1]
# a color-naming trial is [1,0] to task_layer and a word-reading trial is [0,1]


def trial_dict(blue_color, green_color, blue_word, green_word, PC_CN, PC_WR, bias):
    trialdict = {
        colors_input_layer: [blue_color, green_color],
        words_input_layer: [blue_word, green_word],
        task_input_layer: [PC_CN, PC_WR],
        bias_input: [bias, bias]
    }
    return trialdict


initialize_input = trial_dict(1.0, 0.0, 1.0, 0.0, pc, 0.0, bias)

# Run congruent trial -------------------------------------------------------------------------------------------------
congruent_input = trial_dict(1.0, 0.0, 1.0, 0.0, pc, 0.0, bias)  # specify congruent trial input
# run Composition to settle for 200 trials with congruent stimuli input
PCTC.run(inputs=initialize_input, num_trials=settle)

color_input_weights.parameters.matrix.set(
    np.array([
        [1.0, 0.0],      # set color input projections to 1 on the diagonals to e.g.
        [0.0, 1.0]
    ]),     # send a green color input to the green unit of the color layer
    PCTC
)
word_input_weights.parameters.matrix.set(
    np.array([
        [1.0, 0.0],      # the same for word input projections
        [0.0, 1.0]
    ]),
    PCTC
)

# run Composition with congruent stimulus input until threshold in of of the response layer units is reached
PCTC.run(inputs=congruent_input, termination_processing=terminate_trial)

# Store values from run -----------------------------------------------------------------------------------------------
t = task_demand_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')    # Log task output from special logistic function
tt = t[PCTC.name]['SPECIAL_LOGISTIC']
n_con = tt.shape[0]
ttt_cong = tt.reshape(n_con, 2)
conflict_con = ttt_cong[200:, 0] * ttt_cong[200:, 1] * 100             # Compute conflict for plotting (as in MATLAB code)

c = color_feature_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')  # Log color output from special logistic function
cc = c[PCTC.name]['SPECIAL_LOGISTIC']
ccc_cong = cc.reshape(n_con, 2)
w = word_feature_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')   # Log word output from special logistic function
ww = w[PCTC.name]['SPECIAL_LOGISTIC']
www_cong = ww.reshape(n_con, 2)
r = response_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')       # Log response output from special logistic function
rr = r[PCTC.name]['SPECIAL_LOGISTIC']
rrr_cong = rr.reshape(n_con, 2)

# Clear log & reset --------------------------------------------------------------------------------------------
response_layer.log.clear_entries()
color_feature_layer.log.clear_entries()
word_feature_layer.log.clear_entries()
task_demand_layer.log.clear_entries()

color_feature_layer.reset([[0, 0]], context=PCTC)
word_feature_layer.reset([[0, 0]], context=PCTC)
response_layer.reset([[0, 0]], context=PCTC)
task_demand_layer.reset([[0, 0]], context=PCTC)

# Run neutral trials --------------------------------------------------------------------------------------------------
# Set input projections back to 0 for settling period
color_input_weights.parameters.matrix.set(
    np.array([
        [0.0, 0.0],
        [0.0, 0.0]
    ]),
    PCTC
)
word_input_weights.parameters.matrix.set(
    np.array([
        [0.0, 0.0],
        [0.0, 0.0]
    ]),
    PCTC
)

neutral_input = trial_dict(1.0, 0.0, 0.0, 0.0, pc, 0.0, bias)  # create neutral stimuli input
# run Compositoin to settle for 200 trials with neutral stimuli input
PCTC.run(inputs=initialize_input, num_trials=settle)

color_input_weights.parameters.matrix.set(
    np.array([
        [1.0, 0.0],      # Set input projections to 1 for stimulus presentation period
        [0.0, 1.0]
    ]),
    PCTC
)
word_input_weights.parameters.matrix.set(
    np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ]),
    PCTC
)

# run Composition with neutral stimulus input until threshold in of of the response layer units is reached
PCTC.run(inputs=neutral_input, termination_processing=terminate_trial)

# Store values from neutral run ---------------------------------------------------------------------------------------
t = task_demand_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')
tt = t[PCTC.name]['SPECIAL_LOGISTIC']
n_neutral = tt.shape[0]
ttt_neutral = tt.reshape(n_neutral, 2)
conflict_neutral = ttt_neutral[200:, 0] * ttt_neutral[200:, 1] * 100

c = color_feature_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')
cc = c[PCTC.name]['SPECIAL_LOGISTIC']
ccc_neutral = cc.reshape(n_neutral, 2)
w = word_feature_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')
ww = w[PCTC.name]['SPECIAL_LOGISTIC']
www_neutral = ww.reshape(n_neutral, 2)
r = response_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')
rr = r[PCTC.name]['SPECIAL_LOGISTIC']
rrr_neutral = rr.reshape(n_neutral, 2)
# Clear log & reset --------------------------------------------------------------------------------------------

response_layer.log.clear_entries()
color_feature_layer.log.clear_entries()
word_feature_layer.log.clear_entries()
task_demand_layer.log.clear_entries()

color_feature_layer.reset([[0, 0]], context=PCTC)
word_feature_layer.reset([[0, 0]], context=PCTC)
response_layer.reset([[0, 0]], context=PCTC)
task_demand_layer.reset([[0, 0]], context=PCTC)

# Run incongruent trials ----------------------------------------------------------------------------------------------
# Set input projections back to 0 for settling period
color_input_weights.parameters.matrix.set(
    np.array([
        [0.0, 0.0],
        [0.0, 0.0]
    ]),
    PCTC
)
word_input_weights.parameters.matrix.set(
    np.array([
        [0.0, 0.0],
        [0.0, 0.0]
    ]),
    PCTC
)

incongruent_input = trial_dict(1.0, 0.0, 0.0, 1.0, pc, 0.0, bias)
PCTC.run(inputs=initialize_input, num_trials=settle)

color_input_weights.parameters.matrix.set(
    np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ]),
    PCTC
)
word_input_weights.parameters.matrix.set(
    np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ]),
    PCTC
)

PCTC.run(inputs=incongruent_input, termination_processing=terminate_trial)

# Store values from neutral run ---------------------------------------------------------------------------------------

t = task_demand_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')
tt = t[PCTC.name]['SPECIAL_LOGISTIC']
n_incon = tt.shape[0]
ttt_incong = tt.reshape(n_incon, 2)
conflict_incon = ttt_incong[200:, 0] * ttt_incong[200:, 1] * 100

c = color_feature_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')
cc = c[PCTC.name]['SPECIAL_LOGISTIC']
ccc_incong = cc.reshape(n_incon, 2)
w = word_feature_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')
ww = w[PCTC.name]['SPECIAL_LOGISTIC']
www_incong = ww.reshape(n_incon, 2)
r = response_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')
rr = r[PCTC.name]['SPECIAL_LOGISTIC']
rrr_incong = rr.reshape(n_incon, 2)

# Plotting ------------------------------------------------------------------------------------------------------------
if args.enable_plot:
    import matplotlib.pyplot as plt

    # Set up plot structure
    fig, axes = plt.subplots(nrows=3, ncols=4, sharey=True, sharex=True)
    axes[0, 0].set_ylabel('Congruent')
    axes[1, 0].set_ylabel('Neutral')
    axes[2, 0].set_ylabel('Incongruent')

    axes[0, 0].set_title('Task demand units', fontsize=9)
    axes[0, 1].set_title('Response units', fontsize=9)
    axes[0, 2].set_title('Color feature map', fontsize=9)
    axes[0, 3].set_title('Word feature map', fontsize=9)
    plt.setp(
        axes,
        xticks=[0, 400, 780],
        yticks=[0, 0.4, 0.79],
        yticklabels=['0', '0.4', '0.8'],
        xticklabels=['0', '400', '800']
    )

    # Plot congruent output --------------------------
    axes[0, 0].plot(ttt_cong[settle:, 0], 'c')
    axes[0, 0].plot(ttt_cong[settle:, 1], 'k')
    axes[0, 0].plot(conflict_con, 'r')

    axes[0, 1].plot(rrr_cong[settle:, 0], 'b')
    axes[0, 1].plot(rrr_cong[settle:, 1], 'g')
    axes[0, 1].plot([0, n_con - settle], [threshold, threshold], 'k')
    axes[0, 2].plot(ccc_cong[settle:, 0], 'b')
    axes[0, 2].plot(ccc_cong[settle:, 1], 'g')

    axes[0, 3].plot(www_cong[settle:, 0], 'b')
    axes[0, 3].plot(www_cong[settle:, 1], 'g')

    # Plot neutral output --------------------------
    axes[1, 0].plot(ttt_neutral[settle:, 0], 'c')
    axes[1, 0].plot(ttt_neutral[settle:, 1], 'k')
    axes[1, 0].plot(conflict_neutral, 'r')

    axes[1, 1].plot(rrr_neutral[settle:, 0], 'b')
    axes[1, 1].plot(rrr_neutral[settle:, 1], 'g')
    axes[1, 1].plot([0, n_neutral - settle], [threshold, threshold], 'k')
    axes[1, 2].plot(ccc_neutral[settle:, 0], 'b')
    axes[1, 2].plot(ccc_neutral[settle:, 1], 'g')

    axes[1, 3].plot(www_neutral[settle:, 0], 'b')
    axes[1, 3].plot(www_neutral[settle:, 1], 'g')

    # Plot incongruent output --------------------------
    axes[2, 0].plot(ttt_incong[settle:, 0], 'c')
    axes[2, 0].plot(ttt_incong[settle:, 1], 'k')
    axes[2, 0].plot(conflict_incon, 'r')

    axes[2, 1].plot(rrr_incong[settle:, 0], 'b')
    axes[2, 1].plot(rrr_incong[settle:, 1], 'g')
    axes[2, 1].plot([0, n_incon - settle], [threshold, threshold], 'k')
    axes[2, 2].plot(ccc_incong[settle:, 0], 'b')
    axes[2, 2].plot(ccc_incong[settle:, 1], 'g')

    axes[2, 3].plot(www_incong[settle:, 0], 'b')
    axes[2, 3].plot(www_incong[settle:, 1], 'g')

    plt.show(block=not pnl._called_from_pytest)
