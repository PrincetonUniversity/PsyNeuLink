import matplotlib.pyplot as plt
import numpy as np
import seaborn

from psyneulink import LearningProjection, Process, SAMPLE, System, \
    TransferMechanism, PROB
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

        # samples = [[0]] * 60
        samples = np.zeros(60)
        # samples[41] = [1]
        samples[42:60] = 1

        targets = np.zeros(60)
        targets[54] = 1
        # targets[14][54] = 0
        # targets[29][54] = 0

        p = Process(
                default_variable=np.zeros(60),
                pathway=[sample, action_selection],
                learning=LearningProjection(
                        learning_function=TDLearning(learning_rate=0.3),
                ),
                size=60,
                target=np.zeros(60)
        )
        print(action_selection.value)

        timestep = 0
        trial = 0

        def print_header():
            print("\n\n*** EPISODE: {}".format(CentralClock.trial))

        def show_weights():
            nonlocal timestep
            nonlocal trial
            # if timestep < 120:
            delta_vals[trial] = s.mechanisms[2].value
            trial += 1

            print('Reward prediction weights: \n',
                  action_selection.input_state.path_afferents[0].matrix)
            print("\nAction selection value: {}".format(
                    action_selection.value[0][0]))

        input_list = {
            sample: samples,
        }

        target_list = {
            action_selection: [targets]
        }

        s = System(processes=[p], targets=np.zeros(60))
        # s.show_graph(show_learning=True)

        print(s.mechanisms)

        delta_vals = np.zeros((60, 60))

        # for i in range(50):
        results = s.run(
                num_trials=60,
                inputs=input_list,
                targets=target_list,
                learning=True,
                call_before_trial=print_header,
                call_after_trial=show_weights
        )

        plt.plot(delta_vals[0], "-o", label="Trial 1")
        # plt.plot(delta_vals[4], "-1", label="Trial 5")
        # plt.plot(delta_vals[9], "-2", label="Trial 10")
        plt.plot(delta_vals[29], "-8", label="Trial 30")
        plt.plot(delta_vals[49], "-s", label="Trial 50")
        # plt.plot(delta_vals[29][34:], "-p", label="Trial 30")
        # plt.plot(delta_vals[39], "-*", label="Trial 40")
        # plt.plot(delta_vals[49], "-D", label="Trial 50")
        plt.vlines(41, 0, 1, linestyles='dashed', label="Stimulus onset")
        plt.vlines(53, 0, 1, linestyles='dashed', colors='b', label="Reward onset")
        plt.title("Montague et. al. (1996) -- Figure 5A")
        plt.xlabel("Timestep")
        plt.ylabel("∂")
        plt.legend()
        plt.xlim(xmin=35)
        plt.xticks()
        plt.show()

        print(delta_vals)
