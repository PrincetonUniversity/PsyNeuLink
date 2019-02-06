import numpy as np
from psyneulink import *
from gym_forager.envs.forager_env import ForagerEnv

# Runtime Switches:
RENDER = True
PERCEPT_DISTORT = False
PNL_COMPILE = False

# *********************************************************************************************************************
# *********************************************** CONSTANTS ***********************************************************
# *********************************************************************************************************************

# These should probably be replaced by reference to ForagerEnv constants:
obs_len = 2
action_len = 2
player_coord_idx = slice(0, 2)
predator_coord_idx = slice(3, 5)
prey_coord_idx = slice(6, 8)
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
greedy_action_mech = ComparatorMechanism(name='MOTOR OUTPUT', sample=player, target=prey)

agent_comp = Composition(name='PREDATOR-PREY COMPOSITION')
agent_comp.add_node(player)
agent_comp.add_node(prey)
agent_comp.add_node(greedy_action_mech)

agent_comp.env = ForagerEnv() # NEW: ForagerEnv must be stored in an attribute on the Composition

def main():

    # NEW: get_next_input interactively returns a new input from the ForagerEnv
    #      (rather than specifying a pre-determined list of input values)
    def get_next_input(env, result):
        action = np.where(result[0] == 0, 0, result[0] / np.abs(result[0]))
        env_step = env.step(action)
        observation = env_step[0]
        done = env_step[2]
        if not done:
            # NEW: This function MUST return a dictionary of input values for a single trial for each INPUT node
            return {player: [observation[player_coord_idx]],
                    prey: [observation[prey_coord_idx]]}
        return done

    if RENDER:
        agent_comp.env.render()

    BIN_EXECUTE = 'Python'
    if PNL_COMPILE:
        BIN_EXECUTE = 'LLVM'

    max_steps = 100 # maximum number of steps before agent quits
    num_games = 3
    for i in range(3):
        agent_comp.run(inputs=get_next_input,     # specify the NAME of a fn in lieu of an inputs dict
                       num_trials=max_steps,      # maximum number of steps before agent quits
                       bin_execute=BIN_EXECUTE)

if __name__ == "__main__":
    main()
