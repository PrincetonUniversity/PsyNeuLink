import functools
import numpy as np
import psyneulink as pnl

input_layer = pnl.TransferMechanism(
    size=2,
    name='Input Layer'
)

def decision_variable_to_one_hot(x):
    """Generate "one-hot" 1d array designating selected action from DDM's scalar decision variable
    (used to generate value of OutputState for action_selection Mechanism"""
    if x > 0:
        return [1,0]
    else:
        return [0,1]

# Takes sum of input layer elements as external component of drift rate
# Notes:
#    - drift_rate parameter in constructor for DDM is the "internally modulated" component of the drift_rate;
#    - arguments to DDM's function (BogaczEtAl) are specified as CONTROL, so that their values will be determined
#        by the EVCControlMechanism of the System to which the action_selection Mechanism is assigned (see below)
action_selection = pnl.DDM(
    function=pnl.BogaczEtAl(
        drift_rate=pnl.CONTROL,
        threshold=pnl.CONTROL,
        starting_point=pnl.CONTROL,
        noise=pnl.CONTROL,
    ),
    output_states=[{pnl.NAME: 'ACTION VECTOR',
                    pnl.INDEX: 0,
                    pnl.CALCULATE: decision_variable_to_one_hot}],
    name='DDM'
)

# Specifies Reinforcement as the learning function for all projections generated for the Process
#    (in this case, the one between the input_layer and action_selection Mechanisms).
p = pnl.Process(
    default_variable=[0, 0],
    pathway=[input_layer, action_selection],
    learning=pnl.LearningProjection(learning_function=pnl.Reinforcement(learning_rate=0.05)),
    target=0
)

# Prints initial weight matrix for the Projection from the input_layer to the action_selection Mechanism
print('reward prediction weights: \n', action_selection.input_state.path_afferents[0].matrix)

# Specify reward values for corresponding one-hot coded actions specified by action_selection.output_state.value
reward_values = [10, 0]

# Used by System to generate a reward on each trial based on the outcome of the action_selection (DDM) Mechanism
def reward():
    return [reward_values[int(np.nonzero(action_selection.output_state.value)[0])]]


# Used by *call_before_trial* and *call_after_trial* to generate printouts.
# Note:  should be replaced by use of logging functionality that has now been implemented.
def print_header(system):
    print("\n\n**** Time: ", system.scheduler_processing.clock.simple_time)
def show_weights():
    print('Reward prediction weights: \n', action_selection.input_state.path_afferents[0].matrix)
    print(
        '\nAction selected:  {}; predicted reward: {}'.format(
            np.nonzero(action_selection.output_state.value)[0][0],
            action_selection.output_state.value[np.nonzero(action_selection.output_state.value)][0]
        )
    )

# Input stimuli and corresponding targets (rewards) for run of the System.
# Note:  this list contains two sets of stimuli and corresponding rewards;  they will be used in sequence,
#        and the sequence will be recycled for as many trials as specified by the *num_trials* argument
#        in the call the the System's run method see below)
input_list = {input_layer: [[1, 0],[0, 1]]}

s = pnl.System(
        processes=[p],
        targets=[0],
        controller=pnl.EVCControlMechanism
)

# # Shows graph of system (learning components are in orange)
# s.show_graph(show_learning=pnl.ALL, show_dimensions=True)

# Run System.
# Note: *targets* is specified as the reward() function (see above).
s.run(
    num_trials=10,
    inputs=input_list,
    targets=reward,
    call_before_trial=functools.partial(print_header, s),
    call_after_trial=show_weights
)
