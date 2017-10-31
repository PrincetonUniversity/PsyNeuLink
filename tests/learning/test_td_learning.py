import matplotlib.pyplot as plt

import numpy as np

from PsyNeuLink import CentralClock, LearningProjection, SoftMax, \
    TransferMechanism, process, system
from PsyNeuLink.Components.Functions.Function import TDLearning
from PsyNeuLink.Globals.Keywords import PROB, SAMPLE, TARGET


def test_td_learning():
    sample = TransferMechanism(
            default_variable=[[0]],
            name=SAMPLE
    )

    target = TransferMechanism(
            default_variable=[[0]],
            name=TARGET
    )

    action_selection = TransferMechanism(
            default_variable=[[0]],
            function=SoftMax(
                    output=PROB,
                    gain=1.0,
            ),
            size=3,
            name='Action Selection'
    )

    samples = [[0]] * 60
    # samples = [[0]] * 5
    samples[41] = [1]
    # samples[2] = [1]
    samples = np.array(samples, ndmin=2)

    targets = [[0]] * 60
    # targets = [[0]] * 5
    targets[53] = [1]
    # targets[4] = [1]

    p = process(
            default_variable=[[0]],
            size=1,
            pathway=[sample, action_selection],
            learning=LearningProjection(
                    learning_function=TDLearning(learning_rate=0.3),
            ),
            target=targets
    )

    action_selection_values = np.zeros((50, 60))

    print(action_selection.value)

    timestep = 0
    trial = 0

    def print_header():
        nonlocal trial
        if CentralClock.trial == 0:
            print("\n\n== == == == TRIAL {} == == == ==".format(trial + 1))
        print("\n\n**** TIMESTEP: {}".format(CentralClock.trial))

    def show_weights():
        nonlocal timestep
        nonlocal trial
        # if timestep < 120:
        delta_vals[trial][timestep] = s.mechanisms[2].value
        if timestep < 60:
            action_selection_values[trial][timestep] = \
            action_selection.value[0][0]
            timestep += 1
            if CentralClock.trial == 59:
            # if CentralClock.trial == 4:
                print("restarting timesteps...")
                timestep = 0
                trial += 1


        print('Reward prediction weights: \n',
              action_selection.input_state.path_afferents[0].matrix)
        print("\nAction selection value: {}".format(
                action_selection.value[0][0]))

    input_list = {
        sample: samples,
    }

    target_list = {
        action_selection: targets
    }

    s = system(processes=[p])

    print(s.mechanisms)

    s.mechanisms[2].max_time_steps = 60

    delta_vals = np.zeros((50, 60))
    s.show()

    # for i in range(50):
    for i in range(5):
        results = s.run(
                num_trials=60,
                # num_trials=5,
                inputs=input_list,
                targets=target_list,
                learning=True,
                call_before_trial=print_header,
                call_after_trial=show_weights
        )
        s.mechanisms[2].reset()

    plt.plot(delta_vals[0], "-o", label="Trial 1")
    plt.plot(delta_vals[1], "-1", label="Trial 2")
    plt.plot(delta_vals[2], "-2", label="Trial 3")
    plt.plot(delta_vals[3], "-8", label="Trial 4")
    plt.plot(delta_vals[4], "-s", label="Trial 5")
    # plt.plot(delta_vals[29][34:], "-p", label="Trial 30")
    # plt.plot(delta_vals[39], "-*", label="Trial 40")
    # plt.plot(delta_vals[49], "-D", label="Trial 50")
    plt.xlabel("Timestep")
    plt.ylabel("∂")
    plt.legend()
    plt.xlim(xmin=35)
    plt.xticks()
    plt.show()

    print(delta_vals)


if __name__ == '__main__':
    test_td_learning()
