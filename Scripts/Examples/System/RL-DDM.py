import functools
import numpy as np
import psyneulink as pnl

# Seed random number generators for consistency in testing
# seed = 0
# random.seed(seed)
# np.random.seed(seed)

# CONSTRUCTION:
import psyneulink.core.components.functions.distributionfunctions
import psyneulink.core.components.functions.learningfunctions

input_layer = pnl.TransferMechanism(
    size=2,
    name='Input Layer'
)

# Takes sum of input layer elements as external component of drift rate
# Notes:
#    - drift_rate parameter in constructor for DDM is the "internally modulated" component of the drift_rate;
#    - arguments to DDM's function (DriftDiffusionAnalytical) are specified as CONTROL, so that their values will be determined
#        by the EVCControlMechanism of the System to which the action_selection Mechanism is assigned (see below)
#    - the input_format argument specifies that the input to the DDM should be one-hot encoded two-element array
#    - the output_ports argument specifies use of the DECISION_VARIABLE_ARRAY OutputPort, which encodes the
#        response in the same format as the ARRAY input_format/.
action_selection = pnl.DDM(
        input_format=pnl.ARRAY,
        function=psyneulink.core.components.functions.distributionfunctions.DriftDiffusionAnalytical(
                drift_rate=pnl.CONTROL,
                threshold=pnl.CONTROL,
                starting_point=pnl.CONTROL,
                noise=pnl.CONTROL,
        ),
        output_ports=[pnl.SELECTED_INPUT_ARRAY],
        name='DDM'
)

# Construct Process
# Notes:
#    The np.array specifies the matrix used as the Mapping Projection from input_layer to action_selection,
#        which insures the left element of the input favors the left action (positive value of DDM decision variable),
#        and the right element favors the right action (negative value of DDM decision variable)
#    The learning argument specifies Reinforcement as the learning function for the Projection
p = pnl.Process(
    default_variable=[0, 0],
    # pathway=[input_layer, np.array([[1],[-1]]), action_selection],
    pathway=[input_layer, pnl.IDENTITY_MATRIX, action_selection],
    learning=pnl.LearningProjection(learning_function=psyneulink.core.components.functions.learningfunctions
                                    .Reinforcement(learning_rate=0.5)),
    target=0
)

s = pnl.System(
    processes=[p],
    controller=pnl.EVCControlMechanism(control_signals=(pnl.LEARNING_RATE, p.learning_mechanisms[1]))
)

# EXECUTION:

# Prints initial weight matrix for the Projection from the input_layer to the action_selection Mechanism
print('reward prediction weights: \n', action_selection.input_port.path_afferents[0].matrix)


# Used by *call_before_trial* and *call_after_trial* to generate printouts.
# Note:  should be replaced by use of logging functionality that has now been implemented.
def print_header(system):
    print("\n\n**** Time: ", system.scheduler.get_clock(system).simple_time)


def show_weights(context=None):
    print('\nReward prediction weights: \n', action_selection.input_port.path_afferents[0].get_mod_matrix(context))
    # print(
    #     '\nAction selected:  {}; predicted reward: {}'.format(
    #         np.nonzero(action_selection.output_port.value)[0][0],
    #         action_selection.output_port.value[np.nonzero(action_selection.output_port.value)][0]
    #     )
    # )
    comparator = action_selection.output_port.efferents[0].receiver.owner
    learn_mech = action_selection.output_port.efferents[1].receiver.owner
    print('\nact_sel_in_state variable:  {} '
          '\nact_sel_in_state value:     {} '
          '\naction_selection variable:  {} '
          '\naction_selection output:    {} '
          '\ncomparator sample:          {} '
          '\ncomparator target:          {} '
          '\nlearning mech act in:       {} '
          '\nlearning mech act out:      {} '
          '\nlearning mech error in:     {} '
          '\nlearning mech error out:    {} '
          '\nlearning mech learning_sig: {} '
          # '\npredicted reward:           {} '
        .format(
            action_selection.input_ports[0].parameters.variable.get(context),
            action_selection.input_ports[0].parameters.value.get(context),
            action_selection.parameters.variable.get(context),
            action_selection.output_port.parameters.value.get(context),
            comparator.input_ports[pnl.SAMPLE].parameters.value.get(context),
            comparator.input_ports[pnl.TARGET].parameters.value.get(context),
            learn_mech.input_ports[pnl.ACTIVATION_INPUT].parameters.value.get(context),
            learn_mech.input_ports[pnl.ACTIVATION_OUTPUT].parameters.value.get(context),
            learn_mech.input_ports[pnl.ERROR_SIGNAL].parameters.value.get(context),
            learn_mech.output_ports[pnl.ERROR_SIGNAL].parameters.value.get(context),
            learn_mech.output_ports[pnl.LEARNING_SIGNAL].parameters.value.get(context),
            # action_selection.output_port.value[np.nonzero(action_selection.output_port.value)][0]
    ))


# Specify reward values associated with each action (corresponding to elements of esaction_selection.output_port.value)
# reward_values = [10, 0]
reward_values = [0, 10]

# Used by System to generate a reward on each trial based on the outcome of the action_selection (DDM) Mechanism
def reward(context=None):
    """Return the reward associated with the selected action"""
    selected_action = action_selection.output_port.parameters.value.get(context)
    if not any(selected_action):
        # Deal with initialization, during which action_selection.output_port.value may == [0,0]
        selected_action = np.array([1,0])
    return [reward_values[int(np.nonzero(selected_action)[0])]]


# Input stimuli for run of the System.
# Notes:
#    - for illustrative purposes, this list contains two sets of stimuli;
#        they will be used in sequence, and the sequence will be recycled for as many trials as specified by the
#        *num_trials* argument in the call the System's run method see below)
#    - rewards are specified by the reward function above
input_list = {input_layer: [[1, 1],[1, 1]]}

# # Shows graph of system (learning components are in orange)
# s.show_graph(show_learning=pnl.ALL, show_control=True, show_dimensions=True)
# s.show_graph(show_learning=pnl.ALL, show_mechanism_structure=True)
# s.show_graph(show_mechanism_structure=True)

# Run System.
# Note: *targets* is specified as the reward() function (see above).
# s.run(
#     num_trials=10,
#     inputs=input_list,
#     targets=reward,
#     call_before_trial=functools.partial(print_header, s),
#     call_after_trial=show_weights
# )

s.show_graph(show_learning=True)