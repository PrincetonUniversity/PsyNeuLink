import functools
import numpy as np
import psyneulink as pnl

input_layer = pnl.TransferMechanism(
    default_variable=[0, 0, 0],
    name='Input Layer'
)

action_selection = pnl.TransferMechanism(
    default_variable=[0, 0, 0],
    function=pnl.SoftMax(
        output=pnl.PROB,
        gain=1.0
    ),
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
reward_values = [10, 10, 10]
first_reward = 0

# Must initialize reward (won't be used, but needed for declaration of lambda function)
action_selection.output_state.value = [0, 0, 1]
# Get reward value for selected action)


def reward():
    return [reward_values[int(np.nonzero(action_selection.output_state.value)[0])]]


def print_header(system):
    print("\n\n**** TRIAL: ", system.scheduler_processing.times[pnl.TimeScale.RUN][pnl.TimeScale.TRIAL])


def show_weights():
    print('Reward prediction weights: \n', action_selection.input_state.path_afferents[0].matrix)
    print(
        '\nAction selected:  {}; predicted reward: {}'.format(
            np.nonzero(action_selection.output_state.value)[0][0],
            action_selection.output_state.value[np.nonzero(action_selection.output_state.value)][0]
        )
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

s.run(
    num_trials=10,
    inputs=input_list,
    targets=reward,
    call_before_trial=functools.partial(print_header, s),
    call_after_trial=show_weights
)
