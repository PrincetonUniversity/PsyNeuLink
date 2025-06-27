from KeysAndDoors import KeysAndDoors
from collections import namedtuple
Action = namedtuple("Action", "dx dy open pickup")
class KeysAndDoorsEnv:
    def __init__(
        self,
        coherence=.95,
        discount_rate=.95,
        step_cost=-1,
        target_reward=100,
        grid= None
    ):

        self.env = KeysAndDoors(
        grid=grid,
        coherence=coherence,
        discount_rate=discount_rate,
        step_cost=step_cost,
        target_reward=target_reward
        )

        self.current_state = None
        self.current_obs = None
        self.last_action = None

    # resets the environment and returns the first observation
    def reset(self):
        # Setting initial state
        self.current_state = list(self.env.initial_state_dist())[0]

        # Get init observation
        self.current_obs = list(self.env.observation_dist(Action(0, 0, False, False), self.current_state))[0]

        return self.current_obs

    # Agent takes an action
    # Should return reward, ending_condition, and the observation
    def step(self, dx, dy, open, pickup):
        action = Action(dx, dy, open, pickup)
        next_state = list(self.env.next_state_dist(self.current_state, action))[0]
        observation = list(self.env.observation_dist(Action(dx, dy, open, pickup), next_state))[0]
        self.current_obs = observation
        reward = self.env.reward(self.current_state, action, next_state)
        ending_condition = self.env.is_absorbing(next_state)
        self.current_state = next_state
        self.last_action = action
        return observation, reward, ending_condition

    def render(self):
        print(self.env.state_string(self.current_state))
        print(self.last_action)