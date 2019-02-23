import timeit
import numpy as np
from psyneulink import *

from double_dqn import DoubleDQNAgent

from gym_forager.envs.forager_env import ForagerEnv

# *********************************************************************************************************************
# *********************************************** CONSTANTS ***********************************************************
# *********************************************************************************************************************

# Runtime switches:
MPI_IMPLEMENTATION = True
RENDER = False
PNL_COMPILE = False
RUN = True
SHOW_GRAPH = False
# MODEL_PATH = '/Users/jdc/Dropbox (Princeton)/Documents (DropBox)/Python/double-dqn/models/trained_models/policy_net_trained_0.99_20190214-1651.pt'
MODEL_PATH = '../../../double-dqn/models/trained_models/policy_net_trained_0.99_20190214-1651.pt'


# Control costs
COST_RATE = -.05
COST_BIAS = -3
ALLOCATION_SAMPLES = [0]


# These should probably be replaced by reference to ForagerEnv constants:
obs_len = 2
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

# ************************************** DOUBLE_DQN AGENT **************************************************************

# ddqn_agent = DoubleDQNAgent(env=env, model_load_path='', eval_mode=True)
ddqn_optimal_agent = DoubleDQNAgent(model_load_path=MODEL_PATH,
                                    eval_mode=True,
                                    render=False
                                    )
ddqn_actual_agent = DoubleDQNAgent(model_load_path=MODEL_PATH,
                                   eval_mode=True,
                                   render=False
                                   )

veridical_state = None
perceptual_state = None

def new_episode():
    global veridical_state
    global perceptual_state
    veridical_observation = ddqn_optimal_agent.env.reset()
    actual_observation = ddqn_optimal_agent.env.reset()
    g=GaussianDistort()
    veridical_state = ddqn_optimal_agent.buffer.next(veridical_observation, is_new_episode=True)

    # Initial
    ddqn_actual_agent.buffer = ddqn_optimal_agent.buffer
    perceptual_state = veridical_state


def get_optimal_action(observation):
    # Get new state based on observation:
    veridical_state = ddqn_optimal_agent.buffer.next(np.array(observation))
    # optimal_action = np.array(ddqn_agent._io_map(ddqn_agent._select_action(veridical_state).item()))
    optimal_action = ddqn_optimal_agent._select_action(veridical_state)
    optimal_action_item = optimal_action.item()
    mapped_action = ddqn_optimal_agent._io_map(optimal_action_item)
    optimal_action = np.array(mapped_action)
    print(f'\n\nOPTIMAL OBSERVATION: {observation}'
          f'\nOPTIMAL ACTION: {optimal_action}'
          f'\nOPTIMAL ACTION_ITEM: {optimal_action_item}'
          f'\nMAPPED ACTION: {mapped_action}'
          f'\nOPTIMAL ACTION FROM FUNCTION: {optimal_action}')
    return optimal_action

# **************************************  PROCESSING MECHANISMS ********************************************************

# Perceptual Mechanisms
player_percept = ProcessingMechanism(size=prey_len, function=GaussianDistort, name="PLAYER PERCEPT")
predator_percept = ProcessingMechanism(size=predator_len, function=GaussianDistort, name="PREDATOR PERCEPT")
prey_percept = ProcessingMechanism(size=prey_len, function=GaussianDistort, name="PREY PERCEPT")

# Mechanism used to encode optimal action from call to Run
optimal_action_mech = ProcessingMechanism(size=action_len, name="OPTIMAL ACTION")

def get_action(variable=[[0,0],[0,0],[0,0]]):
    global perceptual_state
    # Convert variable to observation:
    observation = variable.reshape(6,)
    # Get new state based on observation:
    perceptual_state = ddqn_agent.buffer.next(observation)
    # action = np.array(ddqn_agent._io_map(ddqn_agent._select_action(perceptual_state).item()))
    selected_action = ddqn_agent._select_action(perceptual_state)
    selected_action_item = selected_action.item()
    mapped_action = ddqn_agent._io_map(selected_action_item)
    action = np.array(mapped_action)
    print(f'\n\nACTUAL OBSERVATION: {observation}'
          f'\nSELECTED ACTION: {selected_action}'
          f'\nSELECTED ACTION_ITEM: {selected_action_item}'
          f'\nMAPPED ACTION: {mapped_action}'
          f'\nACTUAL ACTION FROM FUNCTION: {action}')
    return action

# Action Mechanism
#    Use ddqn's eval function to compute action for a given observation
#    note: unitization is done in main loop, to allow compilation of LinearCombination function in ObjectiveMech) (TBI)
action_mech = ProcessingMechanism(default_variable=[[0,0],[0,0],[0,0]], function=get_action, name='ACTION')

# ************************************** BASIC COMPOSITION *************************************************************

agent_comp = Composition(name='PREDATOR-PREY COMPOSITION')
agent_comp.add_nodes([player_percept, predator_percept, prey_percept, optimal_action_mech])
agent_comp.add_node(action_mech, required_roles=[NodeRole.OUTPUT])

a = MappingProjection(sender=player_percept, receiver=action_mech.input_states[0])
b = MappingProjection(sender=predator_percept, receiver=action_mech.input_states[1])
c = MappingProjection(sender=prey_percept, receiver=action_mech.input_states[2])
agent_comp.add_projection(a)
agent_comp.add_projection(b)
agent_comp.add_projection(c)


# **************************************  CONOTROL APPRATUS ************************************************************

difference = Distance(metric=DIFFERENCE)
#   function for ObjectiveMechanism

def objective_function(variable):
    '''Return difference between optimal and actual actions'''
    actual_action = variable[0]
    optimal_action = variable[1]
    similarity = 1-difference([optimal_action, actual_action])/4
    return similarity

ocm = OptimizationControlMechanism(features={SHADOW_INPUTS:[player_percept, predator_percept, prey_percept,
                                                            optimal_action_mech]},
                                   agent_rep=agent_comp, # Use Composition itself (i.e., fully "model-based" evaluation)
                                   function=GridSearch(direction=MAXIMIZE, save_values=True),
                                   objective_mechanism=ObjectiveMechanism(function=objective_function,
                                                                          monitor=[action_mech, optimal_action_mech]),
                                   control_signals=[ControlSignal(projections=(VARIANCE,player_percept),
                                                                  allocation_samples=ALLOCATION_SAMPLES,
                                                                  intensity_cost_function=Exponential(rate=COST_RATE,
                                                                                                      bias=COST_BIAS)),
                                                    ControlSignal(projections=(VARIANCE,predator_percept),
                                                                  allocation_samples=ALLOCATION_SAMPLES,
                                                                  intensity_cost_function=Exponential(rate=COST_RATE,
                                                                                                      bias=COST_BIAS)),
                                                    ControlSignal(projections=(VARIANCE,prey_percept),
                                                                  allocation_samples=ALLOCATION_SAMPLES,
                                                                  intensity_cost_function=Exponential(rate=COST_RATE,
                                                                                                      bias=COST_BIAS))])
# Add controller to Composition
agent_comp.add_model_based_optimizer(ocm)
agent_comp.enable_model_based_optimizer = True
agent_comp.model_based_optimizer_mode = BEFORE

if SHOW_GRAPH:
    # agent_comp.show_graph(show_mechanism_structure='ALL')
    agent_comp.show_graph(show_control=True)


# *********************************************************************************************************************
# ******************************************   RUN SIMULATION  ********************************************************
# *********************************************************************************************************************

num_episodes = 1

def main():
    reward = 0
    done = False
    if RENDER:
        ddqn_agent.env.render()  # If visualization is desired
    else:
        print("Running simulation...")
    steps = 0
    start_time = timeit.default_timer()
    new_episode()
    for _ in range(num_episodes):
        observation = ddqn_agent.env.reset()
        optimal_action=[0,0]
        while True:
            execution_id = 'TEST'
            if PNL_COMPILE:
                BIN_EXECUTE = 'LLVM'
            else:
                BIN_EXECUTE = 'Python'
            optimal_action = get_optimal_action(observation)
            run_results = agent_comp.run(inputs={player_percept:[observation[player_coord_slice]],
                                                 predator_percept:[observation[predator_coord_slice]],
                                                 prey_percept:[observation[prey_coord_slice]],
                                                 optimal_action_mech:optimal_action},
                                         execution_id=execution_id,
                                         bin_execute=BIN_EXECUTE,
                                         )
            action = np.where(run_results[0]==0,0,run_results[0]/np.abs(run_results[0]))

            def print_controller():
                print('SIMULATION:')
                for sample, value in zip(ocm.saved_samples, ocm.saved_values):
                    print(f'\t\tSample: {sample} Value: {value}')
                print('OCM Allocation:\n\t{}'.
                      format(repr(list(np.squeeze(ocm.parameters.control_allocation.get(execution_id))))))

            print('\n**********************\nSTEP: ', steps)

            print(f'OUTER LOOP OPTIMAL ACTION:{optimal_action}')
            print(f'OUTER LOOP RUN RESULTS:{run_results}')
            print(f'OUTER LOOP ACTION:{action}')

            if agent_comp.model_based_optimizer_mode is BEFORE:
                print_controller()

            print('Observations:'
                  f'\n\tPlayer:\n\t\tveridical: {player_percept.parameters.variable.get(execution_id)}'
                  f'\n\t\tperceived: {player_percept.parameters.value.get(execution_id)}'
                  f'\n\tPredator:\n\t\tveridical: {predator_percept.parameters.variable.get(execution_id)}'
                  f'\n\t\tperceived: {predator_percept.parameters.value.get(execution_id)}'
                  f'\n\tPrey:\n\t\tveridical: {prey_percept.parameters.variable.get(execution_id)}'
                  f'\n\t\tperceived: {prey_percept.parameters.value.get(execution_id)}'
                  f'\nActions:\n\tActual: {action}\n\tOptimal: {optimal_action}'
                  f'\nOutcome:\n\t{ocm.objective_mechanism.parameters.value.get(execution_id)}'
                  f'\nOCM ControlSignals:'
                  f'\n\tPlayer:\t\t{ocm.control_signals[0].parameters.value.get(execution_id)}'
                  f'\n\tPredator\t{ocm.control_signals[1].parameters.value.get(execution_id)}'
                  f'\n\tPrey:\t\t{ocm.control_signals[2].parameters.value.get(execution_id)}'
                  f'\nOCM ControlSignal Costs:'
                  f'\n\tPlayer:\t\t{ocm.control_signals[0].parameters.cost.get(execution_id)}'
                  f'\n\tPredator:\t{ocm.control_signals[1].parameters.cost.get(execution_id)}'
                  f'\n\tPrey:\t\t{ocm.control_signals[2].parameters.cost.get(execution_id)}')

            if agent_comp.model_based_optimizer_mode is AFTER:
                print_controller()


            # Get observation for next iteration (based on action taken on this one)
            observation, reward, done, _ = ddqn_agent.env.step(action)
            steps += 1
            if done:
                break
    stop_time = timeit.default_timer()
    print(f'{steps / (stop_time - start_time):.1f} steps/second, {steps} total steps in '
          f'{stop_time - start_time:.2f} seconds')
    if RENDER:
        ddqn_agent.env.render(close=True)  # If visualization is desired

if RUN:
    if __name__ == "__main__":
        main()
