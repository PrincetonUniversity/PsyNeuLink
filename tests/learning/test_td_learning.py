import numpy as np

from PsyNeuLink import TransferMechanism, SoftMax, process, \
    LearningProjection, CentralClock, system
from PsyNeuLink.Components.Functions.Function import Reinforcement, TDLearning
from PsyNeuLink.Globals.Keywords import SAMPLE, TARGET, PROB
from PsyNeuLink.Library.Mechanisms.AdaptiveMechanisms.LearningMechanisms\
    .TDLearningMechanism import \
    TDLearningMechanism


def test_td_learning():
    sample = TransferMechanism(
        default_variable=np.zeros(60),
        name=SAMPLE
    )

    target = TransferMechanism(
        default_variable=np.zeros(60),
        name=TARGET
    )

    action_selection = TransferMechanism(
        default_variable=np.zeros(60),
        function=SoftMax(
            output=PROB,
            gain=1.0
        ),
        name='Action Selection'
    )

    p = process(
        default_variable=np.zeros(60),
        size=60,
        pathway=[sample, action_selection],
        learning=LearningProjection(sender=TDLearningMechanism())
    )

    def print_header():
        print("\n\n**** TRIAL: {}".format(CentralClock.trial))

    def show_weights():
        print('Reward prediction weights: \n',
              action_selection.input_state.path_afferents[0].matrix)
        print("\nAction selected: {}; predicted reward: {}".format(
            np.nonzero(action_selection.value)[0][0],
            action_selection.value[np.nonzero(action_selection.value)[0][0]]))

    samples = np.zeros((60, 1))
    samples[40] = 1

    targets = np.zeros((60, 1))
    targets[53] = 1

    input_list = {
        sample: samples
    }

    s = system(processes=[p])

    results = s.run(
        num_trials=50,
        inputs=input_list,
        targets=targets,
        call_before_trial=print_header,
        call_after_trial=show_weights,
    )

    print(results)
