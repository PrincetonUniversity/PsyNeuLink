import numpy as np
from psyneulink import *

from gym_forager.envs.forager_env import ForagerEnv

num_trials = 4
env = ForagerEnv()
reward = 0
done = False

# These should probably be replaced by reference to ForagerEnv constants:
obs_len = 3
obs_coords = 2
action_len = 2
player_idx = 0
player_obs_start_idx = player_idx * obs_len
player_value_idx = player_idx * obs_len + obs_coords
player_coord_slice = slice(player_obs_start_idx,player_value_idx)
predator_idx = 1
predator_obs_start_idx = predator_idx * obs_len
predator_value_idx = predator_idx * obs_len + obs_coords
predator_coord_slice = slice(predator_obs_start_idx,predator_value_idx)
prey_idx = 2
prey_obs_start_idx = prey_idx * obs_len
prey_value_idx = prey_idx * obs_len + obs_coords
prey_coord_slice = slice(prey_obs_start_idx,prey_value_idx)

player_len = prey_len = predator_len = obs_coords
dist = Distance(metric=EUCLIDEAN)
PREDATOR = 0
PREY = 1
ATTEND = 0
DISATTEND = 200
UNDECIDED = 0

player_obs = ProcessingMechanism(size=prey_len, function=GaussianDistort, name="PLAYER OBS")
prey_obs = ProcessingMechanism(size=prey_len, function=GaussianDistort, name="PREY OBS")
predator_obs = TransferMechanism(size=predator_len, function=GaussianDistort, name="PREDATOR OBS")
# For future use:
values = TransferMechanism(size=3, name="AGENT VALUES")
reward = TransferMechanism(name="REWARD")


# Sample function (that takes the place of eventual OCM evaluate method)
def choose_closer_agent_function(variable):
    if variable is None:
        return [0,0]
    player_coord = variable[0]
    predator_coord = variable[1]
    prey_coord = variable[2]
    dist_to_predator = dist([player_coord, predator_coord])
    dist_to_prey = dist([player_coord, prey_coord])
    denom = dist_to_predator + dist_to_prey
    if denom != 0:
        normalized_dist_to_predator = dist_to_predator/denom
        normalized_dist_to_prey = dist_to_prey/denom
        if normalized_dist_to_predator < normalized_dist_to_prey:
            return [PREDATOR]
        else:
            return [PREY]
    return [-1]

# Sample function (that takes the place of eventual OCM function)
def control_allocation_function(variable):

    closest_agent = variable[0]

    if closest_agent == PREDATOR:
        return [[ATTEND],[DISATTEND]]
    elif closest_agent == PREY:
        return [[DISATTEND],[ATTEND]]
    else:
        return [[UNDECIDED],[UNDECIDED]]

# Use ComparatorMechanism to compute direction of action as difference of coordinates between player and prey:
# note: unitization is done in main loop, to allow compilation of LinearCombination function) (TBI)
greedy_action_mech = ComparatorMechanism(name='ACTION',sample=player_obs,target=prey_obs)

Panicky_control_mech = ControlMechanism(objective_mechanism=ObjectiveMechanism(function=choose_closer_agent_function,
                                                                               monitored_output_states=[player_obs,
                                                                                                        predator_obs,
                                                                                                        prey_obs]),
                                        function = control_allocation_function,
                                        control_signals=[(VARIANCE,predator_obs), (VARIANCE,prey_obs)]
)

agent_comp = Composition(name='PANICKY CONTROL COMPOSITION')
# NOTE: THE ASSIGNMENT OF required_roles BELOW SHOULD NOT BE NEEDED;
#       CIRCUMVENTS A BUG IN ROLE ASSIGNMENTS CURRENTLY BEING FIXED
agent_comp.add_c_node(player_obs, required_roles=CNodeRole.ORIGIN)
agent_comp.add_c_node(prey_obs, required_roles=CNodeRole.ORIGIN)
agent_comp.add_c_node(predator_obs, required_roles=CNodeRole.ORIGIN)
agent_comp.add_c_node(greedy_action_mech, required_roles=CNodeRole.TERMINAL)
agent_comp.add_c_node((Panicky_control_mech))


def main():
    for _ in range(num_trials):
        observation = env.reset()
        while True:
            run_results = agent_comp.run(inputs={
                player_obs:[observation[player_coord_slice]],
                predator_obs:[observation[predator_coord_slice]],
                prey_obs:[observation[prey_coord_slice]],
                # NOTE: ASSIGNMENT OF INPUT TO Panicky_control_mech IS TO CIRCUMVENT A BUG IN ROLE ASSIGNMENTS
                #       (INCLUDING MISASSIGNMENT OF ControlMechanism AS ORIGIN MECH) THAT IS CURRENTLY BEING FIXED
                Panicky_control_mech:[0]
            })
            action= np.where(run_results[0]==0,0,run_results[0]/np.abs(run_results[0]))
            observation, reward, done, _ = env.step(action)
            if done:
                break

if __name__ == "__main__":
    main()
