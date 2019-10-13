import functools
import numpy as np
import psyneulink as pnl
import psyneulink.core.components.functions.learningfunctions
import psyneulink.core.components.functions.transferfunctions

input_layer = pnl.TransferMechanism(
    size=3,
    name='Input Layer'
)

action_selection = pnl.TransferMechanism(
    size=3,
    function=psyneulink.core.components.functions.transferfunctions.SoftMax(
        output=pnl.PROB,
        gain=1.0
    ),
    name='Action Selection'
)

p = pnl.Process(
    default_variable=[0, 0, 0],
    pathway=[input_layer, action_selection],
    learning=pnl.LearningProjection(learning_function=psyneulink.core.components.functions.learningfunctions
                                    .Reinforcement(learning_rate=0.05)),
    target=0
)

print('reward prediction weights: \n', action_selection.input_port.path_afferents[0].matrix)
print('target_mechanism weights: \n', action_selection.output_port.efferents[0].matrix)

actions = ['left', 'middle', 'right']
reward_values = [10, 0, 0]
first_reward = 0

# Must initialize reward (won't be used, but needed for declaration of lambda function)
action_selection.output_port.value = [0, 0, 1]
# Get reward value for selected action)


def reward(context=None):
    """Return the reward associated with the selected action"""
    return [reward_values[int(np.nonzero(action_selection.parameters.output_port.value.get(context))[0])]]


def print_header(system):
    print("\n\n**** Time: ", system.scheduler.get_clock(system).simple_time)


def show_weights(system):
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
            action_selection.parameters.value.get(system),
            action_selection.output_port.parameters.value.get(system),
            comparator.input_ports[pnl.SAMPLE].parameters.value.get(system),
            comparator.input_ports[pnl.TARGET].parameters.value.get(system),
            learn_mech.input_ports[pnl.ACTIVATION_INPUT].parameters.value.get(system),
            learn_mech.input_ports[pnl.ACTIVATION_OUTPUT].parameters.value.get(system),
            learn_mech.input_ports[pnl.ERROR_SIGNAL].parameters.value.get(system),
            learn_mech.output_ports[pnl.ERROR_SIGNAL].parameters.value.get(system),
            learn_mech.output_ports[pnl.LEARNING_SIGNAL].parameters.value.get(system),
            action_selection.output_port.parameters.value.get(system)[np.nonzero(action_selection.output_port.parameters.value.get(system))][0]
        )
    )

p.run(
    num_trials=10,
    inputs=[[[1, 1, 1]]],
    targets=reward,
    call_after_trial=functools.partial(show_weights, p)
)

input_list = {input_layer: [[1, 1, 1]]}

s = pnl.System(
    processes=[p],
    targets=[0]
)

s.show_graph(show_learning=pnl.ALL)

s.run(
    num_trials=10,
    inputs=input_list,
    targets=reward,
    call_before_trial=functools.partial(print_header, s),
    call_after_trial=functools.partial(show_weights, s)
)
