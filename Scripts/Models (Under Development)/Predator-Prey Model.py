
import timeit

import numpy as np
from psyneulink import *

from gym_forager.envs.forager_env import ForagerEnv

# Runtime Switches:
RENDER = False
PNL_COMPILE = False
RUN = True
SHOW_GRAPH = False

# *********************************************************************************************************************
# *********************************************** CONSTANTS ***********************************************************
# *********************************************************************************************************************

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

# *********************************************************************************************************************
# **************************************  MECHANISMS AND COMPOSITION  *************************************************
# *********************************************************************************************************************

# Perceptual Mechanisms
player_obs = ProcessingMechanism(size=prey_len, function=GaussianDistort, name="PLAYER OBS")
prey_obs = ProcessingMechanism(size=prey_len, function=GaussianDistort, name="PREY OBS")
predator_obs = TransferMechanism(size=predator_len, function=GaussianDistort, name="PREDATOR OBS")
# Value and Reward Mechanisms (not yet used;  for future use)
values = TransferMechanism(size=3, name="AGENT VALUES")
reward = TransferMechanism(name="REWARD")

# Action Mechanism
#    Use ComparatorMechanism to compute direction of action as difference of coordinates between player and prey:
#    note: unitization is done in main loop, to allow compilation of LinearCombination function) (TBI)
greedy_action_mech = ComparatorMechanism(name='ACTION',sample=player_obs,target=prey_obs)

# Create Composition
agent_comp = Composition(name='PREDATOR-PREY COMPOSITION')
agent_comp.add_node(player_obs)
agent_comp.add_node(predator_obs)
agent_comp.add_node(prey_obs)
agent_comp.add_node(greedy_action_mech)


# ControlMechanism

#   function for ObjectiveMechanism
dist = Distance(metric=EUCLIDEAN)
def dist_diff_fct(variable):
    # Get difference in distance of player to predator vs. prey
    if variable is None:
        return 0
    player_coord = variable[0]
    predator_coord = variable[1]
    prey_coord = variable[2]
    dist_to_predator = dist([player_coord, predator_coord])
    dist_to_prey = dist([player_coord, prey_coord])
    return dist_to_predator - dist_to_prey

ocm = OptimizationControlMechanism(features={SHADOW_INPUTS: [player_obs, predator_obs, prey_obs]},
                                   agent_rep=agent_comp,
                                   function=GridSearch(direction=MINIMIZE,
                                                       save_values=True),

                                   objective_mechanism=ObjectiveMechanism(function=dist_diff_fct,
                                                                          monitor=[player_obs,
                                                                                   predator_obs,
                                                                                   prey_obs]),
                                   control_signals=[ControlSignal(modulates=(VARIANCE, player_obs),
                                                                  # allocation_samples=[0, 1, 10, 100]),
                                                                  # allocation_samples=[0, 10, 100]),
                                                                  # allocation_samples=[10, 1]),
                                                                  allocation_samples=[0, 10]),
                                                    ControlSignal(modulates=(VARIANCE, predator_obs),
                                                                  # allocation_samples=[0, 1, 10, 100]),
                                                                  # allocation_samples=[0, 10, 100]),
                                                                  # allocation_samples=[10, 1]),
                                                                  allocation_samples=[0, 10]),
                                                    ControlSignal(modulates=(VARIANCE, prey_obs),
                                                                  # allocation_samples=[0, 1, 10, 100]),
                                                                  # allocation_samples=[0, 10, 100]),
                                                                  # allocation_samples=[10, 1]),
                                                                  allocation_samples=[0, 10]),
                                                    ],
                                   )
agent_comp.add_controller(ocm)
agent_comp.enable_model_based_optimizer = True

if SHOW_GRAPH:
    # agent_comp.show_graph(show_mechanism_structure='ALL')
    agent_comp.show_graph(show_controller=True, show_mechanism_structure=True)


# *********************************************************************************************************************
# ******************************************   RUN SIMULATION  ********************************************************
# *********************************************************************************************************************

num_trials = 4

def main():
    env = ForagerEnv()
    reward = 0
    done = False

    def my_print():
        print(ocm.net_outcome)

    if RENDER:
        env.render()  # If visualization is desired
    else:
        print("Running simulation...")
    steps = 0
    start_time = timeit.default_timer()
    for _ in range(num_trials):
        observation = env.reset()
        while True:
            if PNL_COMPILE:
                BIN_EXECUTE = 'LLVM'
            else:
                BIN_EXECUTE = 'Python'
            run_results = agent_comp.run(inputs={player_obs:[observation[player_coord_slice]],
                                                 predator_obs:[observation[predator_coord_slice]],
                                                 prey_obs:[observation[prey_coord_slice]],
                                                 },
                                         call_after_trial=my_print,
                                         bin_execute=BIN_EXECUTE
                                         )

            action = np.where(run_results[0] == 0, 0, run_results[0] / np.abs(run_results[0]))
            # action = np.squeeze(np.where(greedy_action_mech.value==0,0,
            #                              greedy_action_mech.value[0]/np.abs(greedy_action_mech.value[0])))
            observation, reward, done, _ = env.step(action)
            print('OCM ControlSignals:')
            print('\n\tOutcome: {}\n\tPlayer OBS: {}\n\tPredator OBS: {}\n\tPrey OBS: {}'.
                  format(ocm._objective_mechanism.value,
                         ocm.control_signals[0].value,
                         ocm.control_signals[1].value,
                         ocm.control_signals[2].value))
            for sample, value in zip(ocm.saved_samples, ocm.saved_values):
                print('\n\t\tSample: {} Value: {}'.format(sample, value))
            print('\n\tOutcome: {}\n\tPlayer OBS: {}\n\tPredator OBS: {}\n\tPrey OBS: {}'.
                  format(ocm._objective_mechanism.value,
                         ocm.control_signals[0].value,
                         ocm.control_signals[1].value,
                         ocm.control_signals[2].value))
            if done:
                break
    stop_time = timeit.default_timer()
    print(f'{steps / (stop_time - start_time):.1f} steps/second, {steps} total steps in '
          f'{stop_time - start_time:.2f} seconds')
    if RENDER:
        env.render()  # If visualization is desired

if RUN:
    if __name__ == "__main__":
        main()
