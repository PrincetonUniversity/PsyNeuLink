import numpy as np
from psyneulink import *

from gym_forager.envs.forager_env import ForagerEnv

num_trials = 4
env = ForagerEnv()
reward = 0
done = False

# These should probably be replaced by reference to ForagerEnv constants:
obs_len = 2
action_len = 2
player_coord_idx = slice(0,2)
predator_coord_idx = slice(3,5)
prey_coord_idx = slice(6,8)
player_value_idx = 2
predator_value_idx = 5
prey_value_idx = 8

player_len = prey_len = predator_len = obs_len

player = ProcessingMechanism(size=prey_len, function=GaussianDistort(variance=1000), name="PLAYER OBS")
prey = ProcessingMechanism(size=prey_len, function=GaussianDistort(variance=1000), name="PREY OBS")

# For future use:
values = TransferMechanism(size=3, name="AGENT VALUES")
reward = TransferMechanism(name="REWARD")

# Use ComparatorMechanism to compute direction of action as difference of coordinates between player and prey:
# note: unitization is done in main loop, to allow compilation of LinearCombination function) (TBI)
greedy_action_mech = ComparatorMechanism(name='MOTOR OUTPUT',sample=player,target=prey)

agent_comp = Composition(name='PREDATOR-PREY COMPOSITION')
agent_comp.add_c_node(player)
agent_comp.add_c_node(prey)
agent_comp.add_c_node(greedy_action_mech)

# Projections to greedy_action_mech were created by assignments of sample and target args in its constructor,
#  so just add them to the Composition).
for projection in greedy_action_mech.projections:
    agent_comp.add_projection(projection)

def main():
    for _ in range(num_trials):
        observation = env.reset()
        while True:
            run_results = agent_comp.run(inputs={
                player:[observation[player_coord_idx]],
                prey:[observation[prey_coord_idx]],
            },
            # bin_execute='LLVM'
            )
            action= np.where(run_results[0]==0,0,run_results[0]/np.abs(run_results[0]))
            observation, reward, done, _ = env.step(action)
            if done:
                break

if __name__ == "__main__":
    main()
