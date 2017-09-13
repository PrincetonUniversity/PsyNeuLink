import pprint

import numpy as np

from PsyNeuLink import system
from PsyNeuLink.Components.Functions.Function import TDLearning, \
    TransferFunction, IdentityTransform
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism \
    import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.ModulatoryProjections \
    .LearningProjection import LearningProjection
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection \
    import MappingProjection
from PsyNeuLink.Globals.Keywords import CURRENT_STATE, FUNCTION, MATRIX
from PsyNeuLink.Scheduling.TimeScale import CentralClock


def test_td_learning():
    reward = [[None, None, None, None, 0, None],
              [None, None, None, 0, None, 100],
              [None, None, None, 0, None, None],
              [None, 0, 0, None, 0, None],
              [0, None, None, 0, None, 100],
              [None, 0, None, None, 0, 100]]

    reward = np.array(reward, dtype=float)

    t = TDLearning(reward=reward, discount_factor=0.8, goal_state=5,
                   initial_iterations=10, initial_state=2)
    print(t)
    for i in range(len(reward)):
        t.paramsCurrent[CURRENT_STATE] = i
        for _ in range(100):
            weights = t.function()
    print("weights = {}".format(weights))


def test_td_learning_2():
    reward = [
        [None, 0, None, None, 0, None, None, None, None, None, None, None, None,
         None, None, None],
        [0, None, None, None, None, None, None, None, None, None, None, None,
         None, None, None, None],
        [None, None, None, 0, None, None, 0, None, None, None, None, None, None,
         None, None, None],
        [None, None, 0, None, None, None, None, 0, None, None, None, None, None,
         None, None, None],
        [0, None, None, None, None, None, None, None, 0, None, None, None, None,
         None, None, None],
        [None, None, None, None, None, None, 0, None, None, 0, None, None, None,
         None, None, None],
        [None, None, 0, None, None, 0, None, None, None, None, None, None, None,
         None, None, None],
        [None, None, None, 0, None, None, None, None, None, None, None, 0, None,
         None, None, None],
        [None, None, None, None, 0, None, None, None, None, 0, None, None, 0,
         None, None, None],
        [None, None, None, None, None, 0, None, None, 0, None, 0, None, None, 0,
         None, None],
        [None, None, None, None, None, None, None, None, None, 0, None, None,
         None, None, 0, None],
        [None, None, None, None, None, None, None, 0, None, None, None, None,
         None, None, None, 100],
        [None, None, None, None, None, None, None, None, 0, None, None, None,
         None, None, None, None],
        [None, None, None, None, None, None, None, None, None, 0, None, None,
         None, None, 0, None],
        [None, None, None, None, None, None, None, None, None, None, 0, None,
         None, 0, None, None],
        [None, None, None, None, None, None, None, None, None, None, None, 0,
         None, None, None, 100]]

    reward = np.array(reward, dtype=float)
    t = TDLearning(reward=reward, discount_factor=0.8, goal_state=15,
                   initial_state=2)
    print(t)
    for i in range(len(reward)):
        t.paramsCurrent[CURRENT_STATE] = i
        for _ in range(100):
            weights = t.function()
    print("weights = {}".format(weights))


def test_q_matrix():
    reward = [[None, None, None, None, 0, None],
              [None, None, None, 0, None, 100],
              [None, None, None, 0, None, None],
              [None, 0, 0, None, 0, None],
              [0, None, None, 0, None, 100],
              [None, 0, None, None, 0, 100]]
    reward = np.array(reward, dtype=float)

    final_q_matrix = [[0, 0, 0, 0, 396, 0],
                      [0, 0, 0, 316, 0, 496],
                      [0, 0, 0, 316, 0, 0],
                      [0, 396, 252, 0, 396, 0],
                      [316, 0, 0, 316, 0, 496],
                      [0, 396, 0, 0, 396, 496]]

    final_q_matrix = np.array(final_q_matrix, dtype=np.int32)

    td_learning_function = TDLearning(reward=reward, discount_factor=0.8,
                                      goal_state=5)

    pp = pprint.PrettyPrinter()
    pp.pprint(final_q_matrix)
    pp.pprint(td_learning_function.q_matrix)

    assert np.array_equal(final_q_matrix, td_learning_function.q_matrix)


def test_mechanism_integration():
    reward = [[None, None, None, None, 0, None],
              [None, None, None, 0, None, 100],
              [None, None, None, 0, None, None],
              [None, 0, 0, None, 0, None],
              [0, None, None, 0, None, 100],
              [None, 0, None, None, 0, 100]]
    reward = np.array(reward, dtype=float)

    print("Num dimensions: {}".format(reward.ndim))
    print("Is reward instance of np.ndarray? {}".format(isinstance(reward, np.ndarray)))

    input_layer = TransferMechanism(
        default_variable=np.ones(reward.shape),
        name='Input Layer'
    )

    action_selection = TransferMechanism(
        default_variable=np.ones(reward.shape),
        name='Action Selection'
    )

    identity_transform_func = IdentityTransform(
        default_variable=np.ones(reward.shape), matrix=np.ones(reward.shape))

    mapping_proj_params = {
        FUNCTION: identity_transform_func
    }

    identity_mapping_projection = MappingProjection(sender=input_layer,
                                                    receiver=action_selection,
                                                    matrix=reward,
                                                    params=mapping_proj_params)

    td_learning = TDLearning(reward=reward, discount_factor=0.8, goal_state=5)

    p = process(
        default_variable=np.ones((6, 1, 6)),
        pathway=[input_layer, identity_mapping_projection, action_selection],
        learning=LearningProjection(
            learning_function=td_learning)
    )

    def print_header():
        print("\n\n**** TRIAL: {}".format(CentralClock.trial))

    def show_weights():
        print("Reward prediction weights: {}".format(
            action_selection.input_states[0].path_afferents[0].matrix))

    input_list = {input_layer: np.ones((3, 6, 6))}



    s = system(
        processes=[p],
    )
    # size of inputs must = size of targets
    results = s.run(
        input_list,
        num_trials=10,
        targets=reward,
        call_before_trial=print_header,
        call_after_trial=show_weights
    )

    print(results)
