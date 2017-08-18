from typing import List, Union

import numpy as np
from PsyNeuLink.Globals.Keywords import INITIALIZING

from PsyNeuLink import TDLEARNING_FUNCTION
from PsyNeuLink.Components.Component import ComponentError
from PsyNeuLink.Components.Functions.Function import Function_Base, Function
from PsyNeuLink.Components.Mechanisms import Mechanism


class TDLearning(Function):
    componentName = TDLEARNING_FUNCTION
    variableClassDefault = [[[0, 0], [0, 0]]]
    default_learning_rate = 0.05
    default_reward = 0.5
    default_discount_factor = 0.5
    default_initial_weights = [[1, 1]]
    paramClassDefaults = Function_Base.paramClassDefaults

    def __init__(self,
                 default_variable=None,
                 reward: Union[List, np.ndarray] = None,
                 learning_rate: float = default_learning_rate,
                 discount_factor: float = default_discount_factor,
                 states: Union[List, np.ndarray] = None,
                 initial_weights: Union[List, np.ndarray] = None,
                 goal_state: int = 0,
                 initial_state: int = None,
                 initial_action: int = None,
                 params=None,
                 owner: Mechanism = None,
                 prefs=None,
                 context=componentName + INITIALIZING):
        """

        Parameters
        ----------
        default_variable

        learning_rate: float: default 0.05

        reward: 2d np.ndarray or List: default None
            2-dimensional Numpy array or List representing the reward at each
            state for taking an action. States are represented by rows and
            actions are represented by columns. Invalid actions in a state are
            represented with `None`. All valid actions should have a numerical
            value.
        discount_factor: float: default 0.5
            the discount factor or gamma value in the Q-learning algorithm. Must
            be between 0 and 1 inclusive. Values closer to 1 will prioritize
            future states. Values closer to 0 will prioritize more immediate
            states.
        initial_weights: 2d np.ndarray or List: default None
            initial weights given to each action
        goal_state: int: default None
            the goal state to reach represented as an integer. Must be within the
            range 0 - `len(reward)`
        initial_state: int: default None
            the state to start in represented as an integer. Must be within the
            range 0 - `len(reward)`
        initial_action: int: default None
            first action to take from the `initial_state`. Must be within the
            range 0 - `len(reward[initial_state])` and must correspond to a valid
            action in `initial_state`
        params
        owner
        prefs
        context
        """
        # TODO: is it helpful to have a states argument that specifies which
        # states to go to
        if default_variable is None and reward is None:
            raise ComponentError("Reward matrix must be specified in either the"
                                 "variable argument or the reward argument.")

        if not initial_state:
            initial_state = np.random.randint(0, len(reward))

        if learning_rate > 1 or learning_rate < 0:
            raise ComponentError(
                "learning_rate must be between 0 and 1 inclusive")

        if not initial_weights:
            initial_weights = np.ones(len(reward[initial_state]))

        params = self._assign_args_to_param_dicts(learning_rate=learning_rate,
                                                  reward=reward,
                                                  discount_factor=discount_factor,
                                                  initial_weights=initial_weights,
                                                  initial_state=initial_state,
                                                  goal_state=goal_state,
                                                  owner=owner,
                                                  params=params)
        self._name = "TDLearning"

        super().__init__(default_variable,
                         param_defaults=params,
                         prefs=prefs,
                         context=context)

        self.learning_rate = learning_rate
        self.reward = reward
        self.discount_factor = discount_factor
        self.states = states
        self.functionOutputType = None
        self.initial_weights = initial_weights
        self.weights = initial_weights
        self.time_step = 0
        self.q_matrix = np.zeros(reward.shape)
        self.goal_state = goal_state
        self.current_state = initial_state
        self.current_action = initial_action

        while self.current_state != self.goal_state:
            possible_next_states = np.where(
                self.reward[self.current_state] is not None)
            for state in possible_next_states:
                for a in range(len(self.q_matrix[state])):
                    self.q_matrix[self.current_state][a] = \
                        self.reward[self.current_state][
                            a] + self.discount_factor * np.max(
                            self.q_matrix[state])

    def _validate_variable(self, variable, context=None):
        super()._validate_variable(variable, context)

        # TODO: get rid of magic number
        if not np.atleast_2d(self.reward):
            raise ComponentError("State action matrix is not 2d")

        if len(self.weights) != len(self.reward[0]):
            raise ComponentError("The length of the initial weight matrix is "
                                 "not the same as the number of actions in the "
                                 "state action matrix.")

    def function(self,
                 variable=None,
                 params=None,
                 context=None):
        self._check_args(variable=variable, params=params, context=context)

        # if self.time_step >= len(self.state_action_matrix):
        #     return self.weights
        if self.current_state == self.goal_state:
            return self.weights

        # TODO: change this to highest q-value of next state
        self.current_action = np.argmax(self.q_matrix[self.current_state])

        self.weights = self.weights + self.learning_rate * (
            self.discount_factor *
            self.q_matrix[self.current_state][self.current_action] -
            self.reward[self.current_state][self.current_action]) * np.gradient(
            self.q_matrix)

        self.time_step += 1
        return [self.weights - self.initial_weights, params['error_source']]
