"""
This implements a model of mesolimbic dopamine cell activity during monkey
conditioning as found in `Montague, Dayan, and Sejnowski (1996) in PsyNeuLink
<http://www.jneurosci.org/content/jneuro/16/5/1936.full.pdf>`_
"""
import argparse

import numpy as np
import psyneulink as pnl

all_figures = ['5a', '5b', '5c']

parser = argparse.ArgumentParser()
parser.add_argument('--no-plot', action='store_false', help='Disable plotting', dest='enable_plot')
parser.add_argument('--figure', nargs='+', help='Figure(s) to plot (default=all)', choices=all_figures, default=all_figures)
args = parser.parse_args()

if args.enable_plot:
    import matplotlib.pyplot as plt


def build_stimulus_dictionary(sample_mechanism, target_mechanism, no_reward_trials):

    stimulus_onset = 41
    reward_delivery = 54

    samples = []
    targets = []
    for trial in range(120):
        target = [0.] * 60
        target[reward_delivery] = 1.
        if trial in no_reward_trials:
            target[reward_delivery] = 0.
        targets.append(target)

        sample = [0.] * 60
        for i in range(stimulus_onset, 60):
            sample[i] = 1.
        samples.append(sample)

    return {sample_mechanism: samples,
            target_mechanism: targets}


def build_stimulus_dictionary_figure_5c(sample_mechanism, target_mechanism):

    stimulus_onset = 42
    reward_delivery = 54

    # build input dictionary
    samples = []
    targets = []
    for trial in range(150):
        target = [0.] * 60
        target[reward_delivery] = 1.
        if trial > 70:
            target[reward_delivery] = 0.
        targets.append(target)

        sample = [0.] * 60
        for i in range(stimulus_onset, 60):
            sample[i] = 1.
        samples.append(sample)

    return {sample_mechanism: samples,
            target_mechanism: targets}


def figure_5a():
    """
    This creates the plot for figure 5A in the Montague paper. Figure 5A is
    a 'plot of ∂(t) over time for three trials during training (1, 30, and 50).'
    """

    # Create Processing Components
    sample_mechanism = pnl.TransferMechanism(default_variable=np.zeros(60),
                                             name=pnl.SAMPLE)

    action_selection = pnl.TransferMechanism(default_variable=np.zeros(60),
                                             function=pnl.Linear(slope=1.0,
                                                                 intercept=0.01),
                                             name='Action Selection')

    sample_to_action_selection = pnl.MappingProjection(sender=sample_mechanism,
                                                       receiver=action_selection,
                                                       matrix=np.zeros((60, 60)))
    # Create Composition
    composition_name = 'TD_Learning_Figure_5A'
    comp = pnl.Composition(name=composition_name)

    # Add Processing Components to the Composition
    pathway = [sample_mechanism, sample_to_action_selection, action_selection]

    # Add Learning Components to the Composition
    learning_related_components = comp.add_td_learning_pathway(pathway, learning_rate=0.3).learning_components

    # Unpack Relevant Learning Components
    prediction_error_mechanism = learning_related_components[pnl.OBJECTIVE_MECHANISM]
    target_mechanism = learning_related_components[pnl.TARGET_MECHANISM]

    # Create Log
    prediction_error_mechanism.log.set_log_conditions(pnl.VALUE)

    # Create Stimulus Dictionary
    no_reward_trials = {14, 29, 44, 59, 74, 89}
    inputs = build_stimulus_dictionary(sample_mechanism, target_mechanism, no_reward_trials)

    # Run Composition
    comp.learn(inputs=inputs)

    if args.enable_plot:
        # Get Delta Values from Log
        delta_vals = prediction_error_mechanism.log.nparray_dictionary()[composition_name][pnl.VALUE]

        # Plot Delta Values form trials 1, 30, and 50
        with plt.style.context('seaborn'):
            plt.plot(delta_vals[0][0], "-o", label="Trial 1")
            plt.plot(delta_vals[29][0], "-s", label="Trial 30")
            plt.plot(delta_vals[49][0], "-o", label="Trial 50")
            plt.title("Montague et. al. (1996) -- Figure 5A")
            plt.xlabel("Timestep")
            plt.ylabel("∂")
            plt.legend()
            plt.xlim(xmin=35)
            plt.xticks()
            plt.show()

    return comp


def figure_5b():
    """
    This creates the plot for figure 5B in the Montague paper. Figure 5B shows
    the 'entire time course of model responses (trials 1-150).' The setup is
    the same as in Figure 5A, except that training begins at trial 10.
    """

    # Create Processing Components
    sample_mechanism = pnl.TransferMechanism(default_variable=np.zeros(60),
                                             name=pnl.SAMPLE)

    action_selection = pnl.TransferMechanism(default_variable=np.zeros(60),
                                             function=pnl.Linear(slope=1.0,
                                                                 intercept=1.0),
                                             name='Action Selection')

    sample_to_action_selection = pnl.MappingProjection(sender=sample_mechanism,
                                                       receiver=action_selection,
                                                       matrix=np.zeros((60, 60)))
    # Create Composition
    composition_name = 'TD_Learning_Figure_5B'
    comp = pnl.Composition(name=composition_name)

    # Add Processing Components to the Composition
    pathway = [sample_mechanism, sample_to_action_selection, action_selection]

    # Add Learning Components to the Composition
    learning_related_components = comp.add_td_learning_pathway(pathway, learning_rate=0.3).learning_components

    # Unpack Relevant Learning Components
    prediction_error_mechanism = learning_related_components[pnl.OBJECTIVE_MECHANISM]
    target_mechanism = learning_related_components[pnl.TARGET_MECHANISM]

    # Create Log
    prediction_error_mechanism.log.set_log_conditions(pnl.VALUE)

    # Create Stimulus Dictionary
    no_reward_trials = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 29, 44, 59, 74,
                        89, 104, 119}
    inputs = build_stimulus_dictionary(sample_mechanism, target_mechanism, no_reward_trials)

    # Run Composition
    comp.learn(inputs=inputs)

    if args.enable_plot:
        # Get Delta Values from Log
        delta_vals = prediction_error_mechanism.log.nparray_dictionary()[composition_name][pnl.VALUE]

        with plt.style.context('seaborn'):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x_vals, y_vals = np.meshgrid(np.arange(120), np.arange(40, 60, step=1))
            d_vals = np.array([d[0][40:60] for d in delta_vals]).transpose()
            ax.plot_surface(x_vals, y_vals, d_vals)
            ax.set_xlabel("Trial")
            ax.set_ylabel("Timestep")
            ax.set_zlabel("∂")
            ax.set_ylim(y_vals.max(), y_vals.min())
            ax.set_xlim(0, 120)
            ax.set_zlim(-1, 1)
            ax.set_title("Montague et. al. (1996) -- Figure 5B")
            plt.show()

    return comp


def figure_5c():
    """
    This creates the plot for Figure 5C in the Montague paper. Figure 5C shows
    'extinction of response to the sensory cue.' The setup is the same as
    Figure 5A, except that reward delivery stops at trial 70
    """

    # Create Processing Components
    sample_mechanism = pnl.TransferMechanism(default_variable=np.zeros(60),
                                             name=pnl.SAMPLE)

    action_selection = pnl.TransferMechanism(default_variable=np.zeros(60),
                                             function=pnl.Linear(slope=1.0,
                                                                 intercept=1.0),
                                             name='Action Selection')

    sample_to_action_selection = pnl.MappingProjection(sender=sample_mechanism,
                                                       receiver=action_selection,
                                                       matrix=np.zeros((60, 60)))
    # Create Composition
    composition_name = 'TD_Learning_Figure_5C'
    comp = pnl.Composition(name=composition_name)

    # Add Processing Components to the Composition
    pathway = [sample_mechanism, sample_to_action_selection, action_selection]

    # Add Learning Components to the Composition
    learning_related_components = comp.add_td_learning_pathway(pathway, learning_rate=0.3).learning_components

    # Unpack Relevant Learning Components
    prediction_error_mechanism = learning_related_components[pnl.OBJECTIVE_MECHANISM]
    target_mechanism = learning_related_components[pnl.TARGET_MECHANISM]

    # Create Log
    prediction_error_mechanism.log.set_log_conditions(pnl.VALUE)

    # Create Stimulus Dictionary
    inputs = build_stimulus_dictionary_figure_5c(sample_mechanism, target_mechanism)

    # Run Composition
    comp.learn(inputs=inputs)

    if args.enable_plot:
        # Get Delta Values from Log
        delta_vals = prediction_error_mechanism.log.nparray_dictionary()[composition_name][pnl.VALUE]

        with plt.style.context('seaborn'):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x_vals, y_vals = np.meshgrid(np.arange(150), np.arange(40, 60, step=1))
            d_vals = np.array([d[0][40:60] for d in delta_vals]).transpose()
            ax.plot_surface(x_vals, y_vals, d_vals)
            ax.set_ylim(y_vals.max(), y_vals.min())
            ax.set_xlim(0, 140)
            ax.set_zlim(-1, 1)
            ax.set_xlabel("Trial")
            ax.set_ylabel("Timestep")
            ax.set_zlabel("∂")
            ax.set_title("Montague et. al. (1996) -- Figure 5C")
            plt.show()

    return comp


if '5a' in args.figure:
    comp_5a = figure_5a()

if '5b' in args.figure:
    comp_5b = figure_5b()

if '5c' in args.figure:
    comp_5c = figure_5c()
