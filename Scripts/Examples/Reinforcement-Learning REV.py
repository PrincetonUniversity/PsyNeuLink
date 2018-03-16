import functools
import numpy as np
import psyneulink as pnl

input_layer = pnl.TransferMechanism(
    size=3,
    name='Input Layer'
)

action_selection = pnl.TransferMechanism(
        size=3,
        function=pnl.SoftMax(
                output=pnl.ALL,
                gain=1.0),
        output_states={pnl.NAME: 'SELECTED ACTION',
                       pnl.VARIABLE:[(pnl.INPUT_STATE_VARIABLES, 0), (pnl.OWNER_VALUE, 0)],
                       pnl.FUNCTION: pnl.OneHot(mode=pnl.PROB).function},
    # output_states={pnl.NAME: "SOFT_MAX",
    #                pnl.VARIABLE: (pnl.OWNER_VALUE,0),
    #                pnl.FUNCTION: pnl.SoftMax(output=pnl.PROB,gain=1.0)},
    name='Action Selection'
)

p = pnl.Process(
    default_variable=[0, 0, 0],
    pathway=[input_layer, action_selection],
    learning=pnl.LearningProjection(learning_function=pnl.Reinforcement(learning_rate=0.05)),
    target=0
)

print('reward prediction weights: \n', action_selection.input_state.path_afferents[0].matrix)
print('target_mechanism weights: \n', action_selection.output_state.efferents[0].matrix)

actions = ['left', 'middle', 'right']
reward_values = [10, 0, 0]
first_reward = 0

# Must initialize reward (won't be used, but needed for declaration of lambda function)
action_selection.output_state.value = [0, 0, 1]
# Get reward value for selected action)


def reward():
    """Return the reward associated with the selected action"""
    return [reward_values[int(np.nonzero(action_selection.output_state.value)[0])]]


def print_header(system):
    print("\n\n**** Time: ", system.scheduler_processing.clock.simple_time)


def show_weights():
    # print('Reward prediction weights: \n', action_selection.input_state.path_afferents[0].matrix)
    # print(
    #     '\nAction selected:  {}; predicted reward: {}'.format(
    #         np.nonzero(action_selection.output_state.value)[0][0],
    #         action_selection.output_state.value[np.nonzero(action_selection.output_state.value)][0]
    #     )
    assert True
    comparator = action_selection.output_state.efferents[0].receiver.owner
    learn_mech = action_selection.output_state.efferents[1].receiver.owner
    print('\n'
          '\naction_selection value:     {} '
          '\naction_selection output:    {} '
          '\ncomparator sample:          {} '
          '\ncomparator target:          {} '
          '\nlearning mech act in:       {} '
          '\nlearning mech act out:      {} '
          '\nlearning mech error in:     {} '
          '\nlearning mech error out:    {} '
          '\nlearning mech learning_sig: {} '
          '\npredicted reward:           {} '.
        format(
            action_selection.value,
            action_selection.output_state.value,
            comparator.input_states[pnl.SAMPLE].value,
            comparator.input_states[pnl.TARGET].value,
            learn_mech.input_states[pnl.ACTIVATION_INPUT].value,
            learn_mech.input_states[pnl.ACTIVATION_OUTPUT].value,
            learn_mech.input_states[pnl.ERROR_SIGNAL].value,
            learn_mech.output_states[pnl.ERROR_SIGNAL].value,
            learn_mech.output_states[pnl.LEARNING_SIGNAL].value,
            action_selection.output_state.value[np.nonzero(action_selection.output_state.value)][0])
    )

p.run(
    num_trials=10,
    inputs=[[[1, 1, 1]]],
    targets=reward,
    call_after_trial=show_weights
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
    call_after_trial=show_weights
)
