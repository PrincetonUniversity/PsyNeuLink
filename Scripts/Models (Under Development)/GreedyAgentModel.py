import timeit

import numpy as np
from psyneulink import *

from gym_forager.envs.forager_env import ForagerEnv

# Runtime Switches:
PNL=True
RENDER = False
PERCEPT_DISTORT = False
PNL_COMPILE = True

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

agent_comp.add_node(player)
agent_comp.add_node(prey)
agent_comp.add_node(greedy_action_mech)

# agent_comp.show_graph()


# *********************************************************************************************************************
# ******************************************   RUN SIMULATION  ********************************************************
# *********************************************************************************************************************

num_trials = 100

def main():
    env = ForagerEnv()
    reward = 0
    done = False
    if RENDER:
        env.render()  # If visualization is desired
    else:
        print("Running simulation...")
    steps = 0
    start_time = timeit.default_timer()
    for _ in range(num_trials):
        observation = env.reset()
        while True:
            if PNL:
                if PNL_COMPILE:
                    BIN_EXECUTE = 'LLVMRun'
                else:
                    BIN_EXECUTE = 'Python'
                run_results = agent_comp.run(inputs={player:[observation[player_coord_idx]],
                                                     prey:[observation[prey_coord_idx]],
                                                     },
                                             bin_execute=BIN_EXECUTE
                                             )
                run_results[0] = np.array(run_results[0])
                action= np.where(run_results[0] == 0, 0, run_results[0] / np.abs(run_results[0]))
            else:
                run_results = observation[prey_coord_idx] - observation[player_coord_idx]
                action= np.where(run_results == 0, 0, run_results / np.abs(run_results))

            observation, reward, done, _ = env.step(action)
            steps +=1
            if done:
                break
    stop_time = timeit.default_timer()
    print(f'{steps / (stop_time - start_time):.1f} steps/second, {steps} total steps in '
          f'{stop_time - start_time:.2f} seconds')
    if RENDER:
        env.render()  # If visualization is desired

if __name__ == "__main__":
    main()
