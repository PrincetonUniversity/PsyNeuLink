from KeysAndDoors import KeysAndDoors
from collections import namedtuple
Action = namedtuple("Action", "right left up down open pickup read")
class KeysAndDoorsEnv:
    def __init__(
        self,
        coherence=.95,
        discount_rate=.95,
        step_cost=-1,
        target_reward=100,
        heaven_reward=50,
        hell_reward=-50,
        grid= None,
        certainty_thresh=0.9
    ):

        self.env = KeysAndDoors(
        grid=grid,
        coherence=coherence,
        discount_rate=discount_rate,
        step_cost=step_cost,
        target_reward=target_reward,
        heaven_reward=heaven_reward,
        hell_reward=hell_reward
        )

        self.current_state = None
        self.current_obs = None
        self.last_action = None
        self.certainty_thresh = certainty_thresh

    # Translate current observations to a format that the model can understand
    def translate(self, old_obs):
        door_states = []
        for door in old_obs[3]:
            if door == None: door_states.append(0)
            if door == 'R': door_states.append(1)
            if door == 'G': door_states.append(2)
            if door == 'B': door_states.append(3)
            if door == 'D': door_states.append(4)
            if door == 'O': door_states.append(5)
        key_states = []
        for key in old_obs[4]:
            if key == True: key_states.append(1);
            if key == False: key_states.append(0);
        holding_key = 0
        if old_obs[5] == True: holding_key = 1
        key_color = 0
        if old_obs[6] == 'r': key_color = 1
        if old_obs[6] == 'g': key_color = 2
        if old_obs[6] == 'b': key_color = 3

        if len(key_states) == 0:
            key_states = -1
        if len(door_states) == 0:
            door_states = -1

        heaven = 0
        if old_obs[7] == 'h': heaven = 1
        if old_obs[7] == 'j': heaven = 2

        certainty = 0
        if old_obs[8] >= self.certainty_thresh: certainty = 1
        observation = (old_obs[0], old_obs[1], door_states, key_states, holding_key, key_color, heaven, certainty)
        return observation


    # resets the environment and returns the first observation
    def reset(self):
        # Setting initial state
        self.current_state = list(self.env.initial_state_dist())[0]

        # Get init observation
        self.current_obs = list(self.env.observation_dist(Action(0, 0, 0, 0, False, False, False), self.current_state))[0]

        observation = self.translate(self.current_obs)

        return observation

    # Agent takes an action
    # Should return reward, ending_condition, and the observation
    def step(self, right, left, up, down, open, pickup, read):
        action = Action(right, left, up, down, open, pickup, read)
        next_state = list(self.env.next_state_dist(self.current_state, action))[0]
        observation = list(self.env.observation_dist(Action(right, left, up, down, open, pickup, read), next_state))[0]
        self.current_obs = observation
        reward = self.env.reward(self.current_state, action, next_state)
        ending_condition = self.env.is_absorbing(next_state)
        self.current_state = next_state
        self.last_action = action
        new_observation = self.translate(observation)
        return new_observation, reward, ending_condition

    def render(self):
        print(self.env.state_string(self.current_state))
        print(self.last_action)