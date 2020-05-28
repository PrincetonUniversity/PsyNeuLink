import functools
import numpy as np
import psyneulink as pnl
import psyneulink.core.components.functions.transferfunctions

input_layer = pnl.TransferMechanism(
    size=3,
    name='Input Layer'
)

action_selection = pnl.TransferMechanism(
        size=3,
        function=psyneulink.core.components.functions.transferfunctions.SoftMax(
                output=pnl.ALL,
                gain=1.0),
        output_ports={pnl.NAME: 'SELECTED ACTION',
                       pnl.VARIABLE:[(pnl.INPUT_PORT_VARIABLES, 0), (pnl.OWNER_VALUE, 0)],
                       pnl.FUNCTION: psyneulink.core.components.functions.selectionfunctions.OneHot(mode=pnl.PROB).function},
    # output_ports={pnl.NAME: "SOFT_MAX",
    #                pnl.VARIABLE: (pnl.OWNER_VALUE,0),
    #                pnl.FUNCTION: pnl.SoftMax(output=pnl.PROB,gain=1.0)},
    name='Action Selection'
)

p = pnl.Pathway(
    pathway=([input_layer, action_selection], pnl.Reinforcement),
)


actions = ['left', 'middle', 'right']
reward_values = [10, 0, 0]
first_reward = 0


# Must initialize reward (won't be used, but needed for declaration of lambda function)
action_selection.output_port.value = [0, 0, 1]
# Get reward value for selected action)


def reward(context=None):
    """Return the reward associated with the selected action"""
    return [reward_values[int(np.nonzero(action_selection.output_port.parameters.value.get(context))[0])]]


def print_header(comp):
    print("\n\n**** Time: ", comp.scheduler.get_clock(comp).simple_time)


def show_weights(comp):
    comparator = action_selection.output_port.efferents[0].receiver.owner
    learn_mech = action_selection.output_port.efferents[1].receiver.owner
    print(
        '\n'
        '\naction_selection value:     {} '
        '\naction_selection output:    {} '
        '\ncomparator sample:          {} '
        '\ncomparator target:          {} '
        '\nlearning mech act in:       {} '
        '\nlearning mech act out:      {} '
        '\nlearning mech error in:     {} '
        '\nlearning mech error out:    {} '
        '\nlearning mech learning_sig: {} '
        '\npredicted reward:           {} '.format(
            action_selection.parameters.value.get(comp),
            action_selection.output_port.parameters.value.get(comp),
            comparator.input_ports[pnl.SAMPLE].parameters.value.get(comp),
            comparator.input_ports[pnl.TARGET].parameters.value.get(comp),
            learn_mech.input_ports[pnl.ACTIVATION_INPUT].parameters.value.get(comp),
            learn_mech.input_ports[pnl.ACTIVATION_OUTPUT].parameters.value.get(comp),
            learn_mech.input_ports[pnl.ERROR_SIGNAL].parameters.value.get(comp),
            learn_mech.output_ports[pnl.ERROR_SIGNAL].parameters.value.get(comp),
            learn_mech.output_ports[pnl.LEARNING_SIGNAL].parameters.value.get(comp),
            action_selection.output_port.parameters.value.get(comp)[np.nonzero(action_selection.output_port.parameters.value.get(comp))][0]
        )
    )

input_list = {input_layer: [[1, 1, 1]]}

c = pnl.Composition(pathways=[p])
print('reward prediction weights: \n', action_selection.input_port.path_afferents[0].matrix)
print('target_mechanism weights: \n', action_selection.output_port.efferents[0].matrix)

c.show_graph(show_learning=pnl.ALL)

c.learn(
    num_trials=10,
    inputs=input_list,
    # FIX: PROPER FORMAT FOR ASSIGNING TARGET AS FUNCTION?
    targets={action_selection:reward},
    call_before_trial=functools.partial(print_header, c),
    call_after_trial=functools.partial(show_weights, c)
)
