import timeit
import numpy as np
from psyneulink import *

from gym_forager.envs.forager_env import ForagerEnv

# Runtime Switches:
RENDER = False
PNL_COMPILE = False
PERCEPTUAL_DISTORT = False
RUN = False
SHOW_GRAPH = True


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

# Input Mechanisms
player_input = ProcessingMechanism(size=prey_len, name="PLAYER INPUT")
prey_input = ProcessingMechanism(size=prey_len, name="PREY INPUT")
predator_input = TransferMechanism(size=predator_len, name="PREDATOR INPUT")

# Perceptual Mechanisms
if PERCEPTUAL_DISTORT:
    player_obs = ProcessingMechanism(size=prey_len, function=GaussianDistort, name="PLAYER OBS")
    prey_obs = ProcessingMechanism(size=prey_len, function=GaussianDistort, name="PREY OBS")
    predator_obs = TransferMechanism(size=predator_len, function=GaussianDistort, name="PREDATOR OBS")
else:
    player_obs = ProcessingMechanism(size=prey_len, name="PLAYER OBS")
    prey_obs = ProcessingMechanism(size=prey_len, name="PREY OBS")
    predator_obs = TransferMechanism(size=predator_len, name="PREDATOR OBS")


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
# agent_comp.add_linear_processing_pathway([player_input, player_obs])
# agent_comp.add_linear_processing_pathway([predator_input, predator_obs])
# agent_comp.add_linear_processing_pathway([prey_input, prey_obs])
agent_comp.add_node(greedy_action_mech)

# ControlMechanism

#   function for ObjectiveMechanism
def diff_fct(variable):
    # Get difference in distance of player to predator vs. prey
    if variable is None:
        return 0
    player_coord = variable[0]
    player_percept = variable[1]
    predator_coord = variable[2]
    predator_percept = variable[3]
    prey_coord = variable[4]
    prey_percept = variable[5]
    player_diff = np.sum(np.abs(player_percept - player_coord))
    predator_diff = np.sum(np.abs(predator_percept - predator_coord))
    prey_diff = np.sum(np.abs(prey_percept - prey_coord))
    # return - (np.sum(player_diff) + np.sum(predator_diff))
    return -(np.sum(player_diff))

def test_fct(variable):
    if variable is None:
        return 0
    return variable[1] - variable[0]

if PERCEPTUAL_DISTORT:
    CTL_PARAM = VARIANCE
else:
    CTL_PARAM = SLOPE
# agent_comp._analyze_graph()
ocm = OptimizationControlMechanism(features=[player_obs.input_port, predator_obs.input_port, prey_obs.input_port],
                                   agent_rep=agent_comp,
                                   function=GridSearch(direction=MAXIMIZE,
                                                       save_values=True),
                                   objective_mechanism=ObjectiveMechanism(
                                           # function=diff_fct,
                                           function=test_fct,
                                           monitor=[player_obs,
                                                    player_obs.input_port,
                                                    predator_obs,
                                                    predator_obs.input_port,
                                                    prey_obs,
                                                    prey_obs.input_port
                                                    ]
                                           # monitored_output_ports=[player_input, player_obs,
                                           #                          predator_input, predator_obs,
                                           #                          prey_input, prey_obs
                                           #                          ]
                                           # monitored_output_ports=[agent_comp.input_CIM_ports[
                                           #                              player_obs.input_port][1],
                                           #                          player_obs,
                                           #                          agent_comp.input_CIM_ports[
                                           #                              predator_obs.input_port][1],
                                           #                          predator_obs,
                                           #                          agent_comp.input_CIM_ports[
                                           #                              prey_obs.input_port][1],
                                           #                          prey_obs
                                           #                          ]
                                   ),
                                   control_signals=[ControlSignal(modulates=(CTL_PARAM, player_obs),
                                                                  # allocation_samples=[0, 1, 10, 100]),
                                                                  # allocation_samples=[0, 10, 100]),
                                                                  # allocation_samples=[10, 1]),
                                                                  allocation_samples=[0, 100],
                                                                  intensity_cost_function=Exponential(rate=-.1,
                                                                                                      bias=5),
                                                                  ),
                                                    ControlSignal(modulates=(CTL_PARAM, predator_obs),
                                                                  # allocation_samples=[0, 1, 10, 100]),
                                                                  # allocation_samples=[0, 10, 100]),
                                                                  # allocation_samples=[10, 1]),
                                                                  allocation_samples=[0, 100],
                                                                  intensity_cost_function=Exponential(rate=-.1,
                                                                                                      bias=5),
                                                                  ),
                                                    ControlSignal(modulates=(CTL_PARAM, prey_obs),
                                                                  # allocation_samples=[0, 1, 10, 100]),
                                                                  # allocation_samples=[0, 10, 100]),
                                                                  # allocation_samples=[10, 1]),
                                                                  allocation_samples=[0, 100],
                                                                  intensity_cost_function=Exponential(rate=-.1,
                                                                                                      bias=5),
                                                                  ),
                                                    ],
                                   )
agent_comp.add_model_based_optimizer(ocm)
agent_comp.enable_model_based_optimizer = True

full_comp = Composition(name='FULL_COMPOSITION')
full_comp.add_node(agent_comp)
full_comp.add_node(player_input)
full_comp.add_node(predator_input)
full_comp.add_node(prey_input)

# full_comp.add_projection(sender=player_input, receiver=player_obs)
# full_comp.add_projection(sender=predator_input, receiver=predator_obs)
# full_comp.add_projection(sender=prey_input, receiver=prey_obs)

full_comp.add_linear_processing_pathway([player_input,player_obs])
full_comp.add_linear_processing_pathway([predator_input,predator_obs])
full_comp.add_linear_processing_pathway([prey_input,prey_obs])

if SHOW_GRAPH:
    # agent_comp.show_graph(show_mechanism_structure='ALL')
    # agent_comp.show_graph(show_controller=True)
    full_comp.show_graph(show_controller=True)


# *********************************************************************************************************************
# ******************************************   RUN SIMULATION  ********************************************************
# *********************************************************************************************************************

num_trials = 1

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
            if PNL_COMPILE:
                BIN_EXECUTE = 'LLVM'
            else:
                BIN_EXECUTE = 'Python'

            run_results = full_comp.run(inputs={player_input:[observation[player_coord_slice]],
                                                 predator_input:[observation[predator_coord_slice]],
                                                 prey_input:[observation[prey_coord_slice]],
                                                 },
                                         bin_execute=BIN_EXECUTE
                                         )
            action = np.where(run_results[0] == 0, 0, run_results[0] / np.abs(run_results[0]))
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
