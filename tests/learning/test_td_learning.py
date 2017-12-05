import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import seaborn

from psyneulink import LearningProjection, Process, SAMPLE, System, \
    TransferMechanism, PROB, MappingProjection, HOLLOW_MATRIX, IDENTITY_MATRIX
from psyneulink.components.functions.function import TDLearning, SoftMax
from psyneulink.scheduling.timescale import CentralClock


def test_td_learning(capsys):
    with capsys.disabled():
        sample = TransferMechanism(
                default_variable=np.zeros(60),
                name=SAMPLE,
        )

        action_selection = TransferMechanism(
                default_variable=np.zeros(60),
                name='Action Selection',
        )

        stimulus_onset = 42
        reward_delivery = 54

        samples = np.zeros(60)
        samples[stimulus_onset:] = 1
        samples = np.tile(samples, (100, 1))

        targets = np.zeros(60)
        targets[reward_delivery] = 1
        targets = np.tile(targets, (100, 1))
        # no reward given every 15 trials to simulate a wrong response
        targets[14][reward_delivery] = 0
        targets[29][reward_delivery] = 0
        targets[44][reward_delivery] = 0
        targets[59][reward_delivery] = 0
        targets[74][reward_delivery] = 0
        targets[89][reward_delivery] = 0

        projection = MappingProjection(sender=sample,
                                       receiver=action_selection,
                                       matrix=np.zeros((60, 60)))

        learning_projection = LearningProjection(learning_function=TDLearning(learning_rate=0.3))

        p = Process(
                default_variable=np.zeros(60),
                pathway=[sample, action_selection],
                learning=learning_projection,
                size=60,
                target=np.zeros(60)
        )
        trial = 0

        def print_header():
            print("\n\n*** EPISODE: {}".format(CentralClock.trial))

        def show_weights():
            nonlocal trial
            # if timestep < 120:
            delta_vals[trial] = s.mechanisms[2].value
            trial += 1

            print('Reward prediction weights: \n',
                  np.diag(action_selection.input_state.path_afferents[0].matrix))
            print("\nAction selection value: {}".format(
                    action_selection.value[0][0]))

        input_list = {
            sample: samples
        }

        target_list = {
            action_selection: targets
        }

        s = System(processes=[p])

        delta_vals = np.zeros((100, 60))

        results = s.run(
                num_trials=100,
                inputs=input_list,
                targets=target_list,
                learning=True,
                call_before_trial=print_header,
                call_after_trial=show_weights
        )

        plt.plot(delta_vals[0], "-o", label="Trial 1")
        plt.plot(delta_vals[29], "-8", label="Trial 30")
        plt.plot(delta_vals[49], "-s", label="Trial 50")
        plt.title("Montague et. al. (1996) -- Figure 5A")
        plt.xlabel("Timestep")
        plt.ylabel("∂")
        plt.legend()
        plt.xlim(xmin=35)
        plt.xticks()
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_vals, y_vals = np.meshgrid(np.arange(100), np.arange(40, 60, step=1))
        ax.plot_surface(x_vals, y_vals, delta_vals[:, 40:60].transpose())
        ax.invert_yaxis()
        ax.set_xlabel("Trial")
        ax.set_ylabel("Timestep")
        ax.set_zlabel("∂")
        ax.set_title("Montague et. al. (1996) -- Figure 5B")
        plt.show()


def test_td_learning_response_extinction():
    sample = TransferMechanism(
            default_variable=np.zeros(60),
            name=SAMPLE
    )

    action_selection = TransferMechanism(
            default_variable=np.zeros(60),
            name='Action Selection'
    )

    stimulus_onset = 42
    reward_delivery = 54

    samples = np.zeros(60)
    samples[stimulus_onset:] = 1
    samples = np.tile(samples, (150, 1))

    targets = np.zeros(60)
    targets[reward_delivery] = 1
    targets = np.tile(targets, (150, 1))

    # stop delivering reward after trial 70
    for i in range(71, 150):
        targets[i][reward_delivery] = 0

    print("targets = {}".format(targets))

    projection = MappingProjection(sender=sample,
                                   receiver=action_selection,
                                   matrix=np.full((60, 60), 0.01))

    learning_projection = LearningProjection(learning_function=TDLearning(learning_rate=0.3))

    p = Process(
            default_variable=np.zeros(60),
            pathway=[sample, action_selection],
            learning=learning_projection,
            size=60,
            target=np.zeros(60)
    )

    trial = 0

    def print_header():
        print("\n\n*** EPISODE: {}".format(CentralClock.trial))

    def show_weights():
        nonlocal trial
        delta_vals[trial] = s.mechanisms[2].value
        trial += 1

        print('Reward prediction weights: \n',
              np.diag(action_selection.input_state.path_afferents[0].matrix))
        print("\nAction selection value: {}".format(
                action_selection.value[0][0]))

    input_list = {
        sample: samples
    }

    target_list = {
        action_selection: targets
    }

    s = System(processes=[p])

    delta_vals = np.zeros((150, 60))
    trial = 0

    def store_delta_vals():
        nonlocal trial
        delta_vals[trial] = s.mechanisms[2].value
        trial += 1

    results = s.run(
            num_trials=150,
            inputs=input_list,
            targets=target_list,
            learning=True,
            call_after_trial=store_delta_vals
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_vals, y_vals = np.meshgrid(np.arange(150), np.arange(40, 60, step=1))
    ax.plot_surface(x_vals, y_vals, delta_vals[:, 40:60].transpose())
    ax.invert_yaxis()
    ax.set_xlabel("Trial")
    ax.set_ylabel("Timestep")
    ax.set_zlabel("∂")
    ax.set_title("Montague et. al. (1996) -- Figure 5C")
    plt.show()
