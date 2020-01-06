import sys
import os
import timeit
import numpy as np
import argparse

from psyneulink import *
from double_dqn import DoubleDQNAgent

from gym_forager.envs.forager_env import ForagerEnv

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int,
        default=int.from_bytes(os.urandom(4), byteorder="big"),
        help='Random seed, seed from os.urandom if unspecified.')
args = parser.parse_args()

# Set the global seed for PsyNeuLink
SEED = args.seed
from psyneulink.core.globals.utilities import set_global_seed
set_global_seed(SEED)
np.random.seed(SEED+1)
gym_forager_env = ForagerEnv(obs_type='egocentric', incl_values=False, frameskip=2)
gym_forager_env.seed(SEED+2)

# *********************************************************************************************************************
# *********************************************** CONSTANTS ***********************************************************
# *********************************************************************************************************************

# Runtime switches:
MPI_IMPLEMENTATION = True
RENDER = False
PNL_COMPILE = False
RUN = True
SHOW_GRAPH = False
MODEL_PATH = '../../../double-dqn/models/trained_models/policy_net_trained_0.99_20190214-1651.pt'

# Switch for determining actual action taken in each step
OPTIMAL_ACTION = 'OPTIMAL_ACTION'
AGENT_ACTION = 'AGENT_ACTION'
ACTION = AGENT_ACTION

# Verbosity levels for console printout
ACTION_REPORTING = 2
STANDARD_REPORTING = 1
VERBOSE = STANDARD_REPORTING


# ControlSignal parameters
COST_RATE = -.8 # -0.05
COST_BIAS = 1
# COST_RATE = 0#-0.0015
# COST_BIAS = 0
ALLOCATION_SAMPLES = [0, 100, 200, 300, 400, 500]
ALLOCATION_SAMPLES_PREY = [0, 100, 200, 300, 400, 500] # [0, 500]
ALLOCATION_SAMPLES_PREDATOR = [0, 100, 200, 300, 400, 500] # [0, 500]
ALLOCATION_SAMPLES_PLAYER = [0]


# Condition for executing controller
new_episode_flag = True
def get_new_episode_flag():
    return new_episode_flag

CONTROLLER_CONDITION = Condition(func=get_new_episode_flag) # tells schedule when to run OCM
# FEATURE_FUNCTION = Buffer(history=3)
FEATURE_FUNCTION = AdaptiveIntegrator(rate=0.5)


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

# Return True if predator is present (i.e., all its coordinates are >= 0), else return False
def get_trial_type(observation):
    return all(coord >= 0 for coord in observation[predator_coord_slice])

# **********************************************************************************************************************
# **************************************  CREATE COMPOSITION ***********************************************************
# **********************************************************************************************************************

# ************************************** DOUBLE_DQN AGENT **************************************************************

# ddqn_agent = DoubleDQNAgent(env=env, model_load_path='', eval_mode=True)
ddqn_agent = DoubleDQNAgent(model_load_path=MODEL_PATH,
                            eval_mode=True,
                            save_frames=False,
                            # render=False
                            env=gym_forager_env
                            )

def new_episode():
    # Start new episode with veridical state

    global new_episode_flag
    initial_observation = ddqn_agent.env.reset()
    print(f"initial_observation = {initial_observation}")
    new_episode_flag = True

    # Initialize both states to verdical state based on first observation
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

# Mechanism used to encode trialtype from environment
prey_pred_trial_input_mech = ProcessingMechanism(name="PREY PREDATOR TRIAL")
single_prey_trial_input_mech = ProcessingMechanism(name="SINGLE PREY TRIAL")
double_prey_trial_input_mech = ProcessingMechanism(name="DOUBLE PREY TRIAL")


# Mechanism used to encode and reward from environment
reward_input_mech = ProcessingMechanism(name="REWARD INPUT")

# Function used by action_mech to generate action from trained DQN
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
# agent_comp.add_nodes([player_percept, predator_percept, prey_percept, prey_pred_trial_input_mech, reward_input_mech],
#                      required_roles=NodeRole.INPUT)
agent_comp.add_nodes([player_percept, predator_percept, prey_percept, prey_pred_trial_input_mech, single_prey_trial_input_mech, double_prey_trial_input_mech, reward_input_mech])
agent_comp.add_node(action_mech, required_roles=[NodeRole.OUTPUT])

a = MappingProjection(sender=player_percept, receiver=action_mech.input_ports[0])
b = MappingProjection(sender=predator_percept, receiver=action_mech.input_ports[1])
c = MappingProjection(sender=prey_percept, receiver=action_mech.input_ports[2])
agent_comp.add_projections([a,b,c])


# **************************************  CONOTROL APPARATUS ***********************************************************

ocm = OptimizationControlMechanism(name='EVC',
                                   features=[prey_pred_trial_input_mech, single_prey_trial_input_mech, double_prey_trial_input_mech],
                                   # feature_function=FEATURE_FUNCTION,
                                   agent_rep=RegressionCFA(
                                           update_weights=BayesGLM(mu_0=-0.0, sigma_0=0.0001),
                                           prediction_terms=[PV.F, PV.C, PV.COST]
                                   ),
                                   function=GridSearch(direction=MAXIMIZE, save_values=True),

                                   objective_mechanism=ObjectiveMechanism(name='OBJECTIVE MECHANISM',
                                                                          monitor=[reward_input_mech]),
                                   control_signals=[ControlSignal(projections=(VARIANCE,player_percept),
                                                                  allocation_samples=ALLOCATION_SAMPLES_PLAYER,
                                                                  intensity_cost_function=Exponential(rate=COST_RATE,
                                                                                                      bias=COST_BIAS)),
                                                    ControlSignal(projections=(VARIANCE,predator_percept),
                                                                  allocation_samples=ALLOCATION_SAMPLES_PREDATOR,
                                                                  intensity_cost_function=Exponential(rate=COST_RATE,
                                                                                                      bias=COST_BIAS)),
                                                    ControlSignal(projections=(VARIANCE,prey_percept),
                                                                  allocation_samples=ALLOCATION_SAMPLES_PREY,
                                                                  intensity_cost_function=Exponential(rate=COST_RATE,
                                                                                                      bias=COST_BIAS))])
# Add controller to Composition
# agent_comp.add_node(ocm)
agent_comp.add_controller(ocm)
agent_comp.enable_controller = True
agent_comp.controller_mode = BEFORE
agent_comp.controller_condition=CONTROLLER_CONDITION # can also specify this condition on the node if the ocm is added as a node
                                                    # agent_comp,scheduler_processing.add_condition((com, CONTROLLER_CONDITION))

if SHOW_GRAPH:
    # agent_comp.show_graph()
    agent_comp.show_graph(show_controller=True, show_cim=True)
    # agent_comp.show_graph(show_model_based_optimizer=True, show_node_structure=True, show_cim=True)
    # agent_comp.show_graph(show_controller=True,
    #                       show_cim=True,
    #                       show_node_structure=ALL,
    #                       show_headers=True,
    #                       )


# Wrap the entire composition inside another composition so we can perform
# parameter optimization.
opt_comp = Composition(name='outer_opt_comp')
opt_comp.add_node(agent_comp)

# *********************************************************************************************************************
# ******************************************   RUN SIMULATION  ********************************************************
# *********************************************************************************************************************

num_episodes = 20
outcome_log = []
reward_log = []
predator_control_log = []
prey_control_log = []

# The context/execution id to use for all the runs
context = Context()

def print_controller():
    print(f'\nOCM:'
          f'\n\tControlSignals:'
          f'\n\t\tPlayer:\t\t{ocm.control_signals[0].parameters.value.get(context)}'
          f'\n\t\tPredator\t{ocm.control_signals[1].parameters.value.get(context)}'
          f'\n\t\tPrey:\t\t{ocm.control_signals[2].parameters.value.get(context)}'
          f'\n\n\tControlSignal Costs:'
          f'\n\t\tPlayer:\t\t{ocm.control_signals[0].parameters.cost.get(context)}'
          f'\n\t\tPredator:\t{ocm.control_signals[1].parameters.cost.get(context)}'
          f'\n\t\tPrey:\t\t{ocm.control_signals[2].parameters.cost.get(context)}')


def input_generator():

    global new_episode_flag

    if RENDER:
        ddqn_agent.env.render()  # If visualization is desired
    else:
        print('\nRunning simulation... ')

    reward = 0
    steps = 0
    start_time = timeit.default_timer()
    for episode_i in range(num_episodes):
        trialType = 0
        prey_pred_trialType = 0
        single_prey_trialType = 0
        double_prey_trialType = 0

        ddqn_agent.env.trialType = trialType  # 0 is single prey, 1 is two prey, 2 is prey & predator
        observation = ddqn_agent.env.reset()
        new_episode()
        while True:

            if VERBOSE >= STANDARD_REPORTING:
                print(f'\nEPISODE {episode_i}, STEP: {steps} ************************************************')

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

            # Yield the next input the agent composition. Since this generator is
            # passed to the outert optimization composition, it must generate
            # an input dictionary keyed by the inner agent composition node.
            yield {
                    agent_comp: {
                        player_percept:[observation[player_coord_slice]],
                        predator_percept:[observation[predator_coord_slice]],
                        prey_percept:[observation[prey_coord_slice]],
                        prey_pred_trial_input_mech:[prey_pred_trialType],
                        single_prey_trial_input_mech: [single_prey_trialType],
                        double_prey_trial_input_mech: [double_prey_trialType],
                        reward_input_mech:[reward]
                        }
                    }

            # Get agent's action based on perceptual distortion of observation (and application of control)
            run_results = opt_comp.results[-1]
            agent_action = np.where(run_results[0]==0,0,run_results[0]/np.abs(run_results[0]))

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

            outcome_log.append(ocm.objective_mechanism.parameters.value.get(context))

            # Restore frame buffer to state after optimal action taken (at beginning of trial)
            # This is so that agent's action's can be compared to optimal ones on a trial-by-trial basis
            ddqn_agent.buffer.buffer = optimal_agent_frame_buffer
            # # The following allows accumulation of agent's errors (assumes simulations are run before actual action)
            # ddqn_agent.buffer.buffer = actual_agent_frame_buffer

            # if ACTION is OPTIMAL_ACTION:
            #     action = optimal_action
            # elif ACTION is AGENT_ACTION:
            #     action = agent_action
            # else:
            #     assert False, "Must choose either OPTIMAL_ACTION or AGENT_ACTION"
            action = agent_action

            # Get observation for next iteration based on optimal action taken in this one
            observation, reward, done, _ = ddqn_agent.env.step(action)

            print(f'\nAction Taken (using {ACTION}): {action}')

            new_episode_flag = False
            steps += 1
            if done:
                break

        predator_control_log.append(ocm.control_signals[1].parameters.value.get(context))
        prey_control_log.append(ocm.control_signals[2].parameters.value.get(context))
        reward_log.append(reward)

    stop_time = timeit.default_timer()
    print(f'{steps / (stop_time - start_time):.1f} steps/second, {steps} total steps in '
          f'{stop_time - start_time:.2f} seconds')

    outcome_mean = np.mean(np.asarray(outcome_log))
    reward_mean = np.mean(np.asarray(reward_log))
    print(f'\nTotal Outcome: {outcome_mean}')
    print(f'\nTotal Reward: {reward_mean}')
    print('predator control log')
    print(predator_control_log)
    print('prey control log')
    print(prey_control_log)
    predator_control_mean = np.mean(np.asarray(predator_control_log))
    print(f'\npredator control MEAN: {predator_control_mean}')
    prey_control_mean = np.mean(np.asarray(prey_control_log))
    print(f'\nprey control MEAN: {prey_control_mean}')

    if RENDER:
        ddqn_agent.env.render(close=True)  # If visualization is desired

def run_games(cost_rate):
    ocm.control_signals[0].parameters.intensity_cost_function.get(context).parameters.rate.set(cost_rate, context)
    ocm.control_signals[0].parameters.intensity_cost_function.get(context).parameters.rate.set(cost_rate, context)
    ocm.control_signals[0].parameters.intensity_cost_function.get(context).parameters.rate.set(cost_rate, context)

    # Run num_episodes games to completion.
    opt_comp.run(inputs=input_generator,
                   bin_execute='LLVM' if PNL_COMPILE else 'Python',
                   context=context)

    loss = np.abs(np.mean(np.asarray(predator_control_log[-20:])) - 500) + np.mean(np.asarray(prey_control_log[-20:]))
    print(f"Loss = {loss}")

    return loss

def main(cost_rate):
    return run_games(cost_rate)

if __name__ == "__main__":
    main(COST_RATE)

