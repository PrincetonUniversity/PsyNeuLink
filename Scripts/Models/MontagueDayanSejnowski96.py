"""
This implements a model of mesolimbic dopamine cell activity during monkey
conditioning as found in `Montague, Dayan, and Sejnowski (1996) in PsyNeuLink
<http://www.jneurosci.org/content/jneuro/16/5/1936.full.pdf>`_
"""
import matplotlib.pyplot as plt
import numpy as np
import psyneulink as pnl

# needed for 3d plotting in full_experiment and response_extinction
from mpl_toolkits import mplot3d


def model_training():
    """
    This creates the plot for figure 5A in the Montague paper. Figure 5A is
    a 'plot of ∂(t) over time for three trials during training (1, 30, and 50).'
    """
    sample = pnl.TransferMechanism(
        default_variable=np.zeros(60),
        name=pnl.SAMPLE
    )

    action_selection = pnl.TransferMechanism(
        default_variable=np.zeros(60),
        function=pnl.Linear(slope=1.0, intercept=0.01),
        name='Action Selection'
    )

    stimulus_onset = 41
    reward_delivery = 54

    samples = np.zeros(60)
    samples[stimulus_onset:] = 1
    samples = np.tile(samples, (120, 1))

    targets = np.zeros(60)
    targets[reward_delivery] = 1
    targets = np.tile(targets, (120, 1))

    # no reward given every 15 trials to simulate a wrong response
    targets[14][reward_delivery] = 0
    targets[29][reward_delivery] = 0
    targets[44][reward_delivery] = 0
    targets[59][reward_delivery] = 0
    targets[74][reward_delivery] = 0
    targets[89][reward_delivery] = 0

    pnl.MappingProjection(
        sender=sample,
        receiver=action_selection,
        matrix=np.full((60, 60), 0.0)
    )
    learning_projection = pnl.LearningProjection(
        learning_function=pnl.TDLearning(learning_rate=0.3)
    )

    p = pnl.Process(
        default_variable=np.zeros(60),
        pathway=[sample, action_selection],
        learning=learning_projection,
        size=60,
        target=np.zeros(60)
    )
    trial = 0

    def print_header():
        nonlocal trial
        print("\n\n*** EPISODE: {}".format(trial))

    def store_delta_vals():
        nonlocal trial
        delta_vals[trial] = s.mechanisms[2].value
        trial += 1

        print('Delta values: \n{0}'.format(s.mechanisms[2].value))
        

    input_list = {
        sample: samples
    }

    target_list = {
        action_selection: targets
    }

    s = pnl.System(processes=[p])

    delta_vals = np.zeros((120, 60))

    s.run(
        num_trials=120,
        inputs=input_list,
        targets=target_list,
        learning=True,
        call_before_trial=print_header,
        call_after_trial=store_delta_vals
    )
    with plt.style.context('seaborn'):
        plt.plot(delta_vals[0], "-o", label="Trial 1")
        plt.plot(delta_vals[29], "-s", label="Trial 30")
        plt.plot(delta_vals[49], "-o", label="Trial 50")
        plt.title("Montague et. al. (1996) -- Figure 5A")
        plt.xlabel("Timestep")
        plt.ylabel("∂")
        plt.legend()
        plt.xlim(xmin=35)
        plt.xticks()
        plt.show()


def model_training_full_experiment():
    """
    This creates the plot for figure 5B in the Montague paper. Figure 5B shows
    the 'entire time course of model responses (trials 1-150).' The setup is
    the same as in Figure 5A, except that training begins at trial 10.
    """
    sample = pnl.TransferMechanism(
        default_variable=np.zeros(60),
        name=pnl.SAMPLE
    )

    action_selection = pnl.TransferMechanism(
        default_variable=np.zeros(60),
        function=pnl.Linear(slope=1.0, intercept=1.0),
        name='Action Selection'
    )

    stimulus_onset = 41
    reward_delivery = 54

    samples = np.zeros(60)
    samples[stimulus_onset:] = 1
    samples = np.tile(samples, (120, 1))

    targets = np.zeros(60)
    targets[reward_delivery] = 1
    targets = np.tile(targets, (120, 1))

    # training begins at trial 11
    # no reward given every 15 trials to simulate a wrong response
    no_reward_trials = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 29, 44, 59, 74,
                        89, 104, 119]
    for t in no_reward_trials:
        targets[t][reward_delivery] = 0

    pnl.MappingProjection(
        sender=sample,
        receiver=action_selection,
        matrix=np.zeros((60, 60))
    )

    learning_projection = pnl.LearningProjection(
        learning_function=pnl.TDLearning(learning_rate=0.3)
    )

    p = pnl.Process(
        default_variable=np.zeros(60),
        pathway=[sample, action_selection],
        learning=learning_projection,
        size=60,
        target=np.zeros(60)
    )
    trial = 0

    def print_header():
        nonlocal trial
        print("\n\n*** EPISODE: {}".format(trial))

    def store_delta_vals():
        nonlocal trial
        delta_vals[trial] = s.mechanisms[2].value
        trial += 1

    input_list = {
        sample: samples
    }

    target_list = {
        action_selection: targets
    }

    s = pnl.System(processes=[p])

    delta_vals = np.zeros((120, 60))

    s.run(
        num_trials=120,
        inputs=input_list,
        targets=target_list,
        learning=True,
        call_before_trial=print_header,
        call_after_trial=store_delta_vals
    )
    with plt.style.context('seaborn'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_vals, y_vals = np.meshgrid(np.arange(120), np.arange(40, 60, step=1))
        ax.plot_surface(x_vals, y_vals, delta_vals[:, 40:60].transpose())
        ax.invert_yaxis()
        ax.set_xlabel("Trial")
        ax.set_ylabel("Timestep")
        ax.set_zlabel("∂")
        ax.set_title("Montague et. al. (1996) -- Figure 5B")
        plt.show()


def model_training_response_extinction():
    """
    This creates the plot for Figure 5C in the Montague paper. Figure 5C shows
    'extinction of response to the sensory cue.' The setup is the same as
    Figure 5A, except that reward delivery stops at trial 70
    """
    sample = pnl.TransferMechanism(
        default_variable=np.zeros(60),
        name=pnl.SAMPLE
    )

    action_selection = pnl.TransferMechanism(
        default_variable=np.zeros(60),
        function=pnl.Linear(slope=1.0, intercept=1.0),
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

    pnl.MappingProjection(
        sender=sample,
        receiver=action_selection,
        matrix=np.zeros((60, 60))
    )

    learning_projection = pnl.LearningProjection(
        learning_function=pnl.TDLearning(learning_rate=0.3)
    )

    p = pnl.Process(
        default_variable=np.zeros(60),
        pathway=[sample, action_selection],
        learning=learning_projection,
        size=60,
        target=np.zeros(60)
    )

    trial = 0

    def print_header():
        nonlocal trial
        print("\n\n*** EPISODE: {}".format(trial))

    input_list = {
        sample: samples
    }

    target_list = {
        action_selection: targets
    }

    s = pnl.System(processes=[p])

    delta_vals = np.zeros((150, 60))
    trial = 0

    def store_delta_vals():
        nonlocal trial
        delta_vals[trial] = s.mechanisms[2].value
        trial += 1

    s.run(
        num_trials=150,
        inputs=input_list,
        targets=target_list,
        learning=True,
        call_before_trial=print_header,
        call_after_trial=store_delta_vals
    )
    with plt.style.context('seaborn'):
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


if __name__ == '__main__':
    model_training()
    model_training_full_experiment()
    model_training_response_extinction()
