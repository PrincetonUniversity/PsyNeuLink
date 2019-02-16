import timeit
import numpy as np
from psyneulink import *

from gym_forager.envs.forager_env import ForagerEnv

# *********************************************************************************************************************
# *********************************************** CONSTANTS ***********************************************************
# *********************************************************************************************************************

# Runtime Switches:
RENDER = False
PNL_COMPILE = False
RUN = False
SHOW_GRAPH = True


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


# **********************************************************************************************************************
# **************************************  CREATE COMPOSITION ***********************************************************
# **********************************************************************************************************************

# **************************************  PROCESSING MECHANISMS ********************************************************

# Perceptual Mechanisms
player_percept = ProcessingMechanism(size=prey_len, function=GaussianDistort, name="PLAYER PERCEPT")
predator_percept = TransferMechanism(size=predator_len, function=GaussianDistort, name="PREDATOR PERCEPT")
prey_percept = ProcessingMechanism(size=prey_len, function=GaussianDistort, name="PREY PERCEPT")

# Value and Reward Mechanisms (not yet used;  for future use)
values = TransferMechanism(size=3, name="AGENT VALUES")
reward = TransferMechanism(name="REWARD")

env = ForagerEnv()
policy = double_dqn(env=env, policy_net='trained net')

# Action Mechanism
#    Use ddqn's eval function to compute action for a given observation
#    note: unitization is done in main loop, to allow compilation of LinearCombination function in ObjectiveMech) (TBI)
action = ProcessingMechanism(function=policy.eval)

# ************************************** BASIC COMPOSITION *************************************************************

agent_comp = Composition(name='PREDATOR-PREY COMPOSITION')
agent_comp.add_node(player_percept)
agent_comp.add_node(prey_percept)
agent_comp.add_node(predator_percept)
agent_comp.add_node(action)

# **************************************  CONOTROL APPRATUS ************************************************************

difference = Distance(metric=COSINE)
#   function for ObjectiveMechanism

def objective_function(variable):
    '''Return difference between optimal and actual actions'''
    player_coord = variable[0]
    predator_coord = variable[1]
    prey_coord = variable[2]
    actual_action = variable[3]
    optimal_action = policy.eval(observation=[player_coord, prey_coord, predator_coord])
    return 1-difference(optimal_action, actual_action)

# ocm = OptimizationControlMechanism(features={SHADOW_EXTERNAL_INPUTS: [player_obs, predator_obs, prey_obs]},
ocm = OptimizationControlMechanism(features=[player_percept.input_state,
                                             predator_percept.input_state,
                                             prey_percept.input_state],
                                   agent_rep=agent_comp, # Use Composition itself (i.e., fully "model-based" evaluation)
                                   function=GridSearch(direction=MAXIMIZE, save_values=True),
                                   objective_mechanism=ObjectiveMechanism(function=objective_function,
                                                                          monitor=[player_percept,
                                                                                   predator_percept,
                                                                                   prey_percept, action]),
                                   control_signals=[ControlSignal(projections=(VARIANCE,player_percept),
                                                                  # allocation_samples=[0, 1, 10, 100]),
                                                                  # allocation_samples=[0, 10, 100]),
                                                                  # allocation_samples=[10, 1]),
                                                                  allocation_samples=[0, 100],
                                                                  intensity_cost_function=Exponential(rate=-.1,
                                                                                                      bias=5)),
                                                    ControlSignal(projections=(VARIANCE,predator_percept),
                                                                  # allocation_samples=[0, 1, 10, 100]),
                                                                  # allocation_samples=[0, 10, 100]),
                                                                  # allocation_samples=[10, 1]),
                                                                  allocation_samples=[0, 100],
                                                                  intensity_cost_function=Exponential(rate=-.1,
                                                                                                      bias=5)),
                                                    ControlSignal(projections=(VARIANCE,prey_percept),
                                                                  # allocation_samples=[0, 1, 10, 100]),
                                                                  # allocation_samples=[0, 10, 100]),
                                                                  # allocation_samples=[10, 1]),
                                                                  allocation_samples=[0, 100],
                                                                  intensity_cost_function=Exponential(rate=-.1,
                                                                                                      bias=5))])
# Add controller to Composition
agent_comp.add_model_based_optimizer(ocm)
agent_comp.enable_model_based_optimizer = True

if SHOW_GRAPH:
    # agent_comp.show_graph(show_mechanism_structure='ALL')
    agent_comp.show_graph()


# *********************************************************************************************************************
# ******************************************   RUN SIMULATION  ********************************************************
# *********************************************************************************************************************

num_trials = 1

def main():
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
            if PNL_COMPILE:
                BIN_EXECUTE = 'LLVM'
            else:
                BIN_EXECUTE = 'Python'
            run_results = agent_comp.run(inputs={player_percept:[observation[player_coord_slice]],
                                                 predator_percept:[observation[predator_coord_slice]],
                                                 prey_percept:[observation[prey_coord_slice]],
                                                 },
                                         bin_execute=BIN_EXECUTE
                                         )
            action = np.where(run_results[0]==0,0,run_results[0]/np.abs(run_results[0]))
            # action = np.squeeze(np.where(greedy_action_mech.value==0,0,
            #                              greedy_action_mech.value[0]/np.abs(greedy_action_mech.value[0])))
            observation, reward, done, _ = env.step(action)
            print('\nStep: ', steps)
            print('Outcome: {}'.format(ocm.objective_mechanism.value))
            print('OCM ControlSignals:')
            print('\tPlayer OBS: {}\n\tPredator OBS: {}\n\tPrey OBS: {}'.
                  format(ocm.control_signals[0].value,
                         ocm.control_signals[1].value,
                         ocm.control_signals[2].value))
            print('OCM ControlSignal Costs:')
            print('\tPlayer OBS: {}\n\tPredator OBS: {}\n\tPrey OBS: {}'.
                  format(ocm.control_signals[0].cost,
                         ocm.control_signals[1].cost,
                         ocm.control_signals[2].cost))
            print('SIMULATION (PREP FOR NEXT TRIAL):')
            for sample, value in zip(ocm.saved_samples, ocm.saved_values):
                print('\t\tSample: {} Value: {}'.format(sample, value))
            steps += 1
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
