import timeit
import numpy as np
from psyneulink import *
from double_dqn import DoubleDQNAgent

# *********************************************************************************************************************
# *********************************************** CONSTANTS ***********************************************************
# *********************************************************************************************************************

# Runtime switches:
MPI_IMPLEMENTATION = True
RENDER = True
PNL_COMPILE = False
RUN = True
SHOW_GRAPH = False
MODEL_PATH = '../../../double-dqn/models/trained_models/policy_net_trained_0.99_20190214-1651.pt'

# Switch for determining actual action taken in each step
OPTIMAL_ACTION = 'OPTIMAL_ACTION'
AGENT_ACTION = 'AGENT_ACTION'
ACTION = AGENT_ACTION

# Verbosity levels for console printout
ACTION_REPORTING = 3
SIMULATION_REPORTING = 2
STANDARD_REPORTING = 1
VERBOSE = ACTION_REPORTING

# ControlSignal parameters
COST_RATE = -.05
COST_BIAS = -5
ALLOCATION_SAMPLES = [0, 500]


# Environment coordinates
# (these should probably be replaced by reference to ForagerEnv constants)
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
ddqn_agent = DoubleDQNAgent(model_load_path=MODEL_PATH,
                            eval_mode=True,
                            # render=False
                            )

def new_episode():
    # Start new episode with veridical state

    initial_observation = ddqn_agent.env.reset()

    # Initialize both ports to verdical state based on first observation
    perceptual_state = veridical_state = ddqn_agent.buffer.next(initial_observation, is_new_episode=True)

def get_optimal_action(observation):
    # Get new state based on observation:
    veridical_state = ddqn_agent.buffer.next(np.array(observation))
    optimal_action = np.array(ddqn_agent._io_map(ddqn_agent._select_action(veridical_state).item()))
    if VERBOSE >= ACTION_REPORTING:
        print(f'\n\nOPTIMAL OBSERVATION: {observation}'
              f'\nVERIDICAL STATE: {veridical_state.reshape(12,)}'
              f'\nOPTIMAL ACTION: {optimal_action}')
    return optimal_action

# **************************************  PROCESSING MECHANISMS ********************************************************

# Perceptual Mechanisms
player_percept = ProcessingMechanism(size=prey_len, function=GaussianDistort(), name="PLAYER PERCEPT")
predator_percept = ProcessingMechanism(size=predator_len, function=GaussianDistort(), name="PREDATOR PERCEPT")
prey_percept = ProcessingMechanism(size=prey_len, function=GaussianDistort(), name="PREY PERCEPT")

# Mechanism used to encode optimal action from call to Run
optimal_action_mech = ProcessingMechanism(size=action_len, name="OPTIMAL ACTION")

actual_agent_frame_buffer = None

def get_action(variable=[[0,0],[0,0],[0,0]]):
    global actual_agent_frame_buffer
    # Convert variable to observation:
    observation = variable.reshape(6,)

    # Get new state
    # - first cache initial state of buffer
    buffer_cache = ddqn_agent.buffer.buffer.copy()
    # - then get new state based on current observation
    perceptual_state = ddqn_agent.buffer.next(observation)
    # Save frame buffer in case needed to restore buffer to state following perceptual observation
    actual_agent_frame_buffer = ddqn_agent.buffer.buffer
    # - finally, restore frame buffer to initial state for use by next simulation or actual action
    ddqn_agent.buffer.buffer = buffer_cache

    # Get and return action
    action = np.array(ddqn_agent._io_map(ddqn_agent._select_action(perceptual_state).item()))
    if VERBOSE >= ACTION_REPORTING:
        print(f'\n\nACTUAL OBSERVATION: {observation}'
              f'\nACTUAL PERCEPTUAL STATE: {perceptual_state.reshape(12,)}'
              f'\nACTUAL ACTION FROM FUNCTION: {action}')
    return action

# Action Mechanism
#    Use ddqn's eval function to compute action for a given observation
#    note: unitization is done in main loop, to allow compilation of LinearCombination function in ObjectiveMech) (TBI)
action_mech = ProcessingMechanism(default_variable=[[0,0],[0,0],[0,0]],
                                  function=get_action, name='ACTION',
                                  output_ports='agent action')

# ************************************** BASIC COMPOSITION *************************************************************

agent_comp = Composition(name='PREDATOR-PREY COMPOSITION')
agent_comp.add_nodes([player_percept, predator_percept, prey_percept, optimal_action_mech])
agent_comp.add_node(action_mech, required_roles=[NodeRole.OUTPUT])

a = MappingProjection(sender=player_percept, receiver=action_mech.input_ports[0])
b = MappingProjection(sender=predator_percept, receiver=action_mech.input_ports[1])
c = MappingProjection(sender=prey_percept, receiver=action_mech.input_ports[2])
agent_comp.add_projections([a,b,c])



# **************************************  CONOTROL APPRATUS ************************************************************

difference = Distance(metric=DIFFERENCE)
#   function for ObjectiveMechanism

def objective_function(variable):
    """Return difference between optimal and actual actions"""
    actual_action = variable[0]
    optimal_action = variable[1]
    similarity = 1 - difference([optimal_action, actual_action]) / 4
    return similarity

ocm = OptimizationControlMechanism(name='EVC',
                                   features={SHADOW_INPUTS:[player_percept, predator_percept, prey_percept,
                                                            optimal_action_mech]},
                                   agent_rep=agent_comp, # Use Composition itself (i.e., fully "model-based" evaluation)
                                   function=GridSearch(direction=MAXIMIZE, save_values=True),
                                   objective_mechanism=ObjectiveMechanism(name='OBJECTIVE MECHANISM',
                                                                          function=objective_function,
                                                                          monitor=[action_mech, optimal_action_mech]),
                                   # compute_reconfiguration_cost=Distance(metric=EUCLIDEAN, normalize=True),
                                   control_signals=[ControlSignal(modulates=(VARIANCE, player_percept),
                                                                  allocation_samples=ALLOCATION_SAMPLES,
                                                                  intensity_cost_function=Exponential(rate=COST_RATE,
                                                                                                      bias=COST_BIAS)),
                                                    ControlSignal(modulates=(VARIANCE, predator_percept),
                                                                  allocation_samples=ALLOCATION_SAMPLES,
                                                                  intensity_cost_function=Exponential(rate=COST_RATE,
                                                                                                      bias=COST_BIAS)),
                                                    ControlSignal(modulates=(VARIANCE, prey_percept),
                                                                  allocation_samples=ALLOCATION_SAMPLES,
                                                                  intensity_cost_function=Exponential(rate=COST_RATE,
                                                                                                      bias=COST_BIAS))])
# Add controller to Composition
agent_comp.add_controller(ocm)
agent_comp.enable_controller = True
agent_comp.controller_mode = BEFORE
# agent_comp.controller_condition=All(AtRun(0), AtTrial(0))

if SHOW_GRAPH:
    # agent_comp.show_graph()
    agent_comp.show_graph(show_controller=True, show_cim=True)
    # agent_comp.show_graph(show_controller=True, show_node_structure=True, show_cim=True)
    # agent_comp.show_graph(show_controller=True,
    #                       show_cim=True,
    #                       show_node_structure=ALL,
    #                       show_headers=True,
    #                       )


# *********************************************************************************************************************
# ******************************************   RUN SIMULATION  ********************************************************
# *********************************************************************************************************************

num_episodes = 1

def main():

    if RENDER:
        ddqn_agent.env.render()  # If visualization is desired
    else:
        print('\nRunning simulation... ')

    steps = 0
    start_time = timeit.default_timer()
    for _ in range(num_episodes):
        observation = ddqn_agent.env.reset()
        new_episode()
        while True:
            context = 'TEST'
            if PNL_COMPILE:
                BIN_EXECUTE = 'LLVM'
            else:
                BIN_EXECUTE = 'Python'

            if VERBOSE >= STANDARD_REPORTING:
                print(f'\nSTEP: {steps} ************************************************')

            # Cache frame buffer
            trial_start_buffer = ddqn_agent.buffer.buffer.copy()
            # Get optimal action based on observation
            optimal_action = get_optimal_action(observation)
            # Save frame buffer after optimal action
            optimal_agent_frame_buffer = ddqn_agent.buffer.buffer
            # Restore initial state of frame buffer (for use by Composition)
            ddqn_agent.buffer.buffer = trial_start_buffer

            if VERBOSE >= ACTION_REPORTING:
                print(f'\nOUTER LOOP OPTIMAL ACTION:{optimal_action}')

            # Get agent's action based on perceptual distortion of observation (and application of control)
            run_results = agent_comp.run(inputs={player_percept:[observation[player_coord_slice]],
                                                 predator_percept:[observation[predator_coord_slice]],
                                                 prey_percept:[observation[prey_coord_slice]],
                                                 optimal_action_mech:optimal_action},
                                         context=context,
                                         bin_execute=BIN_EXECUTE,
                                         )
            agent_action = np.where(run_results[0] == 0, 0, run_results[0] / np.abs(run_results[0]))
            
            def print_controller():
                if VERBOSE >= SIMULATION_REPORTING:
                    print('\nSIMULATION RESULTS:')
                    for sample, value in zip(ocm.saved_samples, ocm.saved_values):
                        print(f'\t\tSample: {sample} Value: {value}')
                # print(f'\nOCM Allocation:\n\t{repr(list(np.squeeze(ocm.parameters.control_allocation.get(context))))})
                print(f'\nOCM:'
                      f'\n\tControlSignals:'
                      f'\n\t\tPlayer:\t\t{ocm.control_signals[0].parameters.value.get(context)}'
                      f'\n\t\tPredator\t{ocm.control_signals[1].parameters.value.get(context)}'
                      f'\n\t\tPrey:\t\t{ocm.control_signals[2].parameters.value.get(context)}'
                      f'\n\n\tControlSignal Costs:'
                      f'\n\t\tPlayer:\t\t{ocm.control_signals[0].parameters.cost.get(context)}'
                      f'\n\t\tPredator:\t{ocm.control_signals[1].parameters.cost.get(context)}'
                      f'\n\t\tPrey:\t\t{ocm.control_signals[2].parameters.cost.get(context)}')

            if VERBOSE >= ACTION_REPORTING:
                print(f'OUTER LOOP RUN RESULTS:{run_results}')
                print(f'OUTER LOOP AGENT ACTION:{agent_action}')

            if VERBOSE >= STANDARD_REPORTING:
                if agent_comp.controller_mode is BEFORE:
                    print_controller()
                print(f'\nObservations:'
                      f'\n\tPlayer:\n\t\tveridical: {player_percept.parameters.variable.get(context)}'
                      f'\n\t\tperceived: {player_percept.parameters.value.get(context)}'
                      f'\n\tPredator:\n\t\tveridical: {predator_percept.parameters.variable.get(context)}'
                      f'\n\t\tperceived: {predator_percept.parameters.value.get(context)}'
                      f'\n\tPrey:\n\t\tveridical: {prey_percept.parameters.variable.get(context)}'
                      f'\n\t\tperceived: {prey_percept.parameters.value.get(context)}'
                      f'\n\nActions:\n\tAgent: {agent_action}\n\tOptimal: {optimal_action}'
                      f'\n\nOutcome:\n\t{ocm.objective_mechanism.parameters.value.get(context)}'
                      )
                if agent_comp.controller_mode is AFTER:
                    print_controller()

            # Restore frame buffer to state after optimal action taken (at beginning of trial)
            # This is so that agent's action's can be compared to optimal ones on a trial-by-trial basis
            ddqn_agent.buffer.buffer = optimal_agent_frame_buffer
            # # The following allows accumulation of agent's errors (assumes simulations are run before actual action)
            # ddqn_agent.buffer.buffer = actual_agent_frame_buffer

            if ACTION is OPTIMAL_ACTION:
                action = optimal_action
            elif ACTION is AGENT_ACTION:
                action = agent_action
            else:
                assert False, "Must choose either OPTIMAL_ACTION or AGENT_ACTION"

            # Get observation for next iteration based on optimal action taken in this one
            observation, reward, done, _ = ddqn_agent.env.step(action)

            print(f'\nAction Taken (using {ACTION}): {action}')


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
