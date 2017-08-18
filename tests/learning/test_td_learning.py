import pprint

import numpy as np
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.AdaptiveMechanism \
    import \
    AdaptiveMechanism_Base

from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms import \
    AdaptiveMechanism
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanisms \
    .LearningMechanism import \
    LearningMechanism

from PsyNeuLink.Components.Functions.Function import TDLearning
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanisms \
    .TDLearningMechanism import TDLearningMechanism


def test_td_learning():
    reward = [[None, None, None, None, 0, None],
              [None, None, None, 0, None, 100],
              [None, None, None, 0, None, None],
              [None, 0, 0, None, 0, None],
              [0, None, None, 0, None, 100],
              [None, 0, None, None, 0, 100]]

    reward = np.array(reward, dtype=float)

    t = TDLearning(reward=reward, discount_factor=0.8, goal_state=5, initial_iterations=10)
    print(t)
    learning_mech = TDLearningMechanism([[0, 0], [0, 0], [0, 0]], function=t)
    learning_mech.function = t
    learning_mech.run([0])

    print(learning_mech.value)


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
