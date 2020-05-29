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
#  - drift_rate parameter in constructor for DDM is the "internally modulated" component of the drift_rate;
#  - arguments to DDM's function (DriftDiffusionAnalytical) are specified as CONTROL, so their values are determined
#      by the OptimizationControlMechanism of the Composition to which the action_selection Mechanism is assigned
#  - the input_format argument specifies that the input to the DDM should be one-hot encoded two-element array
#  - the output_ports argument specifies use of the DECISION_VARIABLE_ARRAY OutputPort, which encodes the
#      response in the same format as the ARRAY input_format/.
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

comp = pnl.Composition()
p = comp.add_reinforcement_learning_pathway(pathway=[input_layer, action_selection],
                                            learning_rate=0.5)
comp.add_controller(
    pnl.OptimizationControlMechanism(
    # pnl.ControlMechanism(
        # objective_mechanism=True,
        objective_mechanism=pnl.ObjectiveMechanism(monitor=action_selection),
        control_signals=(pnl.LEARNING_RATE,
                         p.learning_components[pnl.LEARNING_MECHANISMS])))

# EXECUTION:

# Prints initial weight matrix for the Projection from the input_layer to the action_selection Mechanism
print('reward prediction weights: \n', action_selection.input_port.path_afferents[0].matrix)


# Used by *call_before_trial* and *call_after_trial* to generate printouts.
# Note:  should be replaced by use of logging functionality that has now been implemented.
def print_header(comp):
    print("\n\n**** Time: ", comp.scheduler.get_clock(comp).simple_time)


def show_weights(context=None):
    print('\nReward prediction weights: \n', action_selection.input_port.path_afferents[0].get_mod_matrix(context))
    # print(
    #     '\nAction selected:  {}; predicted reward: {}'.format(
    #         np.nonzero(action_selection.output_port.value)[0][0],
    #         action_selection.output_port.value[np.nonzero(action_selection.output_port.value)][0]
    #     )
    # )
    target = p.target
    comparator = p.learning_objective
    learn_mech = p.learning_components[pnl.LEARNING_MECHANISMS]
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
          '\npredicted reward:           {} '
          '\nreward:                     {} '
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
            action_selection.output_port.value[np.nonzero(action_selection.output_port.value)][0],
            target.parameters.value.get(context)
    ))


input_list = [[1,1],[1,1]]
# Specify reward values associated with each action (corresponding to elements of action_selection.output_port.value)
# reward_values = [10, 0]
reward_values = [0, 10]

def generate_inputs_and_targets(trial_number):
    """Get the input and target for the current trial"""

    # Cycle through input_list:
    inputs = {input_layer:input_list[trial_number%len(input_list)]}

    # Get target as rewarded value for selected action
    selected_action = action_selection.output_port.value
    if not any(selected_action):
        # Deal with initialization, during which action_selection.output_port.value may == [0,0]
        selected_action = np.array([1,0])
    # Get reward value for selected action
    reward = [reward_values[int(np.nonzero(selected_action)[0])]]
    targets = {p.target: reward}

    return {
        "inputs": inputs,
        "targets": targets
    }

# Run Composition.
comp.learn(
    num_trials=20,
    inputs=generate_inputs_and_targets,
    call_before_trial=functools.partial(print_header, comp),
    call_after_trial=show_weights
)

comp.show_graph(show_learning=True, show_controller=pnl.ALL)