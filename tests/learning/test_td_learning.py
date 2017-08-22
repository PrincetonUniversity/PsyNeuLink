import pprint

import numpy as np

from PsyNeuLink import CURRENT_STATE
from PsyNeuLink.Components.Functions.Function import TDLearning


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
