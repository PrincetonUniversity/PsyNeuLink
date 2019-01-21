import timeit
import numpy as np
from psyneulink import *

from gym_forager.envs.forager_env import ForagerEnv

# Runtime Switches:
PNL=True
# RENDER = True
PERCEPT_DISTORT = False
PNL_COMPILE = False

# *********************************************************************************************************************
# *********************************************** CONSTANTS ***********************************************************
# *********************************************************************************************************************

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


# *********************************************************************************************************************
# **************************************  MECHANISMS AND COMPOSITION  *************************************************
# *********************************************************************************************************************

if PERCEPT_DISTORT:
    player = ProcessingMechanism(size=prey_len, function=GaussianDistort(variance=0), name="PLAYER OBS")
    prey = ProcessingMechanism(size=prey_len, function=GaussianDistort(variance=0), name="PREY OBS")
else:
    player = TransferMechanism(size=prey_len, name="PLAYER OBS")
    prey = TransferMechanism(size=prey_len, name="PREY OBS")

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

# agent_comp.show_graph()


# *********************************************************************************************************************
# ******************************************   RUN SIMULATION  ********************************************************
# *********************************************************************************************************************

# num_trials = 4
agent_comp.env = ForagerEnv()
first_observation = agent_comp.env.reset()
def get_next_input(env, result):
    action = np.where(0. in result[0], 0, result[0] / np.abs(result[0]))
    observation = env.step(action)[0]
    # print("observation = ", observation)
    return {player: [observation[player_coord_idx]],
            prey: [observation[prey_coord_idx]]}
def main():
    reward = 0
    done = False
    steps = 100
    start_time = timeit.default_timer()
    run_results = agent_comp.run(inputs=get_next_input,
                                 num_trials=steps,
                                 # bin_execute=BIN_EXECUTE
                                 )
    stop_time = timeit.default_timer()
    total_time = stop_time -start_time
    print("Total Time = ", total_time)
    print(total_time/steps, " seconds per step")
    print(steps/total_time, " steps per second")

if __name__ == "__main__":
    main()
