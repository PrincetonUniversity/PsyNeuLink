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
#args = parser.parse_args()

SEED = int.from_bytes(os.urandom(4), byteorder="big")

from psyneulink.core.globals.utilities import set_global_seed
set_global_seed(SEED)
np.random.seed(SEED+1)

# *********************************************************************************************************************
# *********************************************** CONSTANTS ***********************************************************
# *********************************************************************************************************************

# Runtime switches:
MPI_IMPLEMENTATION = True
RENDER = False
PNL_COMPILE = False
RUN = True
SHOW_GRAPH = False
MODEL_PATH = '../../../../double-dqn/models/trained_models/policy_net_trained_0.99_20190214-1651.pt'

# Switch for determining actual action taken in each step
OPTIMAL_ACTION = 'OPTIMAL_ACTION'
AGENT_ACTION = 'AGENT_ACTION'
ACTION = AGENT_ACTION

# Verbosity levels for console printout
ACTION_REPORTING = 2
STANDARD_REPORTING = 1
VERBOSE = 0


# ControlSignal parameters
COST_RATE = -.05 # -0.05
COST_BIAS = 1
# COST_RATE = 0#-0.0015
# COST_BIAS = 0
ALLOCATION_SAMPLES = [0, 100, 200, 300, 400, 500]
ALLOCATION_SAMPLES_PREY = [0, 100, 200, 300, 400, 500] # [0, 500]
ALLOCATION_SAMPLES_PREDATOR = [0, 100, 200, 300, 400, 500] # [0, 500]
ALLOCATION_SAMPLES_PLAYER = [0]

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

NUM_EPISODES = 100

# **********************************************************************************************************************
# **************************************  CREATE COMPOSITION ***********************************************************
# **********************************************************************************************************************

# ************************************** DOUBLE_DQN AGENT **************************************************************

class PredatorPreySimulator:

    def __init__(self):

        self.seed = int.from_bytes(os.urandom(4), byteorder="big")

        from psyneulink.core.globals.utilities import set_global_seed
        set_global_seed(self.seed)
        np.random.seed(self.seed+1)

        # Setup a Gym Forager environment for the game
        self.gym_forager_env = ForagerEnv(obs_type='egocentric', incl_values=False, frameskip=2)
        self.gym_forager_env.seed(self.seed+2)

        # Setup an instance of the double DQN agent for determining optimal actions
        self.ddqn_agent = DoubleDQNAgent(model_load_path=MODEL_PATH,
                                         eval_mode=True,
                                         save_frames=False,
                                         render=RENDER,
                                         env=self.gym_forager_env)

        # Setup the PsyNeuLink composition
        self._setup_composition()

    # Helper function for getting the optimal action from the double DQN
    def _get_optimal_action(self, observation):
        # Get new state based on observation:
        veridical_state = self.ddqn_agent.buffer.next(np.array(observation))
        optimal_action = np.array(self.ddqn_agent._io_map(self.ddqn_agent._select_action(veridical_state).item()))
        if VERBOSE >= ACTION_REPORTING:
            print(f'\n\nOPTIMAL OBSERVATION: {observation}'
                  f'\nVERIDICAL STATE: {veridical_state.reshape(12, )}'
                  f'\nOPTIMAL ACTION: {optimal_action}')
        return optimal_action

    def _setup_composition(self):

        def get_new_episode_flag():
            return self.new_episode_flag

        # Condition for executing controller, execute on a new episode.
        self.new_episode_flag = True
        self.CONTROLLER_CONDITION = Condition(func=get_new_episode_flag)  # tells schedule when to run OCM

        # **************************************  PROCESSING MECHANISMS ********************************************************

        # Perceptual Mechanisms
        self.player_percept = ProcessingMechanism(size=prey_len, function=GaussianDistort(), name="PLAYER PERCEPT")
        self.predator_percept = ProcessingMechanism(size=predator_len, function=GaussianDistort(), name="PREDATOR PERCEPT")
        self.prey_percept = ProcessingMechanism(size=prey_len, function=GaussianDistort(), name="PREY PERCEPT")

        # Mechanism used to encode trialtype from environment
        self.prey_pred_trial_input_mech = ProcessingMechanism(name="PREY PREDATOR TRIAL")
        self.single_prey_trial_input_mech = ProcessingMechanism(name="SINGLE PREY TRIAL")
        self.double_prey_trial_input_mech = ProcessingMechanism(name="DOUBLE PREY TRIAL")

        # Mechanism used to encode a reward from environment
        self.reward_input_mech = ProcessingMechanism(name="REWARD INPUT")

        # Function used by action_mech to generate action from trained DQN
        def get_action(variable=[[0, 0], [0, 0], [0, 0]]):
            # Convert variable to observation:
            observation = variable.reshape(6, )

            # Get new state
            # - first cache initial state of buffer
            buffer_cache = self.ddqn_agent.buffer.buffer.copy()

            # - then get new state based on current observation
            perceptual_state = self.ddqn_agent.buffer.next(observation)

            # - finally, restore frame buffer to initial state for use by next simulation or actual action
            self.ddqn_agent.buffer.buffer = buffer_cache

            # Get and return action
            action = np.array(self.ddqn_agent._io_map(self.ddqn_agent._select_action(perceptual_state).item()))
            if VERBOSE >= ACTION_REPORTING:
                print(f'\n\nACTUAL OBSERVATION: {observation}'
                      f'\nACTUAL PERCEPTUAL STATE: {perceptual_state.reshape(12, )}'
                      f'\nACTUAL ACTION FROM FUNCTION: {action}')
            return action

        # Action Mechanism
        #    Use ddqn's eval function to compute action for a given observation
        #    note: unitization is done in main loop, to allow compilation of LinearCombination function in ObjectiveMech) (TBI)
        self.action_mech = ProcessingMechanism(default_variable=[[0,0],[0,0],[0,0]],
                                          function=get_action, name='ACTION',
                                          output_ports='agent action')

        # ************************************** BASIC COMPOSITION *************************************************************

        self.agent_comp = Composition(name='PREDATOR-PREY COMPOSITION')

        self.agent_comp.add_nodes([self.player_percept, self.predator_percept,
                              self.prey_percept, self.prey_pred_trial_input_mech,
                              self.single_prey_trial_input_mech, self.double_prey_trial_input_mech,
                              self.reward_input_mech])
        self.agent_comp.add_node(self.action_mech, required_roles=[NodeRole.OUTPUT])

        a = MappingProjection(sender=self.player_percept, receiver=self.action_mech.input_ports[0])
        b = MappingProjection(sender=self.predator_percept, receiver=self.action_mech.input_ports[1])
        c = MappingProjection(sender=self.prey_percept, receiver=self.action_mech.input_ports[2])
        self.agent_comp.add_projections([a,b,c])


        # **************************************  CONOTROL APPARATUS ***********************************************************
        self.ocm = OptimizationControlMechanism(name='EVC',
                                                state_features=[self.prey_pred_trial_input_mech, self.single_prey_trial_input_mech, self.double_prey_trial_input_mech],
                                                # state_feature_function=FEATURE_FUNCTION,
                                                agent_rep=RegressionCFA(
                               update_weights=BayesGLM(mu_0=-0.0, sigma_0=0.0001),
                               prediction_terms=[PV.F, PV.C, PV.COST]
                       ),
                                                function=GridSearch(direction=MAXIMIZE, save_values=True),

                                                objective_mechanism=ObjectiveMechanism(name='OBJECTIVE MECHANISM',
                                                              monitor=[self.reward_input_mech]),
                                                control_signals=[ControlSignal(projections=(VARIANCE,self.player_percept),
                                                      allocation_samples=ALLOCATION_SAMPLES_PLAYER,
                                                      intensity_cost_function=Exponential(rate=COST_RATE,
                                                                                          bias=COST_BIAS)),
                                        ControlSignal(projections=(VARIANCE,self.predator_percept),
                                                      allocation_samples=ALLOCATION_SAMPLES_PREDATOR,
                                                      intensity_cost_function=Exponential(rate=COST_RATE,
                                                                                          bias=COST_BIAS)),
                                        ControlSignal(projections=(VARIANCE,self.prey_percept),
                                                      allocation_samples=ALLOCATION_SAMPLES_PREY,
                                                      intensity_cost_function=Exponential(rate=COST_RATE,
                                                                                          bias=COST_BIAS))])
        # Add controller to Composition
        # agent_comp.add_node(ocm)
        self.agent_comp.add_controller(self.ocm)
        self.agent_comp.enable_controller = True
        self.agent_comp.controller_mode = BEFORE
        self.agent_comp.controller_condition=self.CONTROLLER_CONDITION # can also specify this condition on the node if the ocm is added as a node
                                                            # agent_comp,scheduler_processing.add_condition((com, CONTROLLER_CONDITION))

        if SHOW_GRAPH:
            self.agent_comp.show_graph(show_controller=True, show_cim=True)

        # Wrap the entire composition inside another composition so we can perform
        # parameter optimization.
        self.opt_comp = Composition(name='outer_opt_comp')
        self.opt_comp.add_node(self.agent_comp)

    def make_input_generator(self, num_episodes=100):

        self.outcome_log = []
        self.reward_log = []
        self.predator_control_log = []
        self.prey_control_log = []

        # The context/execution id to use for all the runs
        self.context = Context()

        # Helper function to print controller details
        def print_controller():
            print(f'\nOCM:'
                  f'\n\tControlSignals:'
                  f'\n\t\tPlayer:\t\t{self.ocm.control_signals[0].parameters.value.get(self.context)}'
                  f'\n\t\tPredator\t{self.ocm.control_signals[1].parameters.value.get(self.context)}'
                  f'\n\t\tPrey:\t\t{self.ocm.control_signals[2].parameters.value.get(self.context)}'
                  f'\n\n\tControlSignal Costs:'
                  f'\n\t\tPlayer:\t\t{self.ocm.control_signals[0].parameters.cost.get(self.context)}'
                  f'\n\t\tPredator:\t{self.ocm.control_signals[1].parameters.cost.get(self.context)}'
                  f'\n\t\tPrey:\t\t{self.ocm.control_signals[2].parameters.cost.get(self.context)}')

        # The input generator function
        def input_generator():

            if RENDER:
                self.ddqn_agent.env.render()  # If visualization is desired
            else:
                print('\nRunning simulation... ')

            reward = 0
            steps = 0
            start_time = timeit.default_timer()
            for episode_i in range(num_episodes):
                trialType = 2
                prey_pred_trialType = 0
                single_prey_trialType = 0
                double_prey_trialType = 0
                print(f'EPISODE {episode_i}')

                self.ddqn_agent.env.trialType = trialType  # 0 is single prey, 1 is two prey, 2 is prey & predator

                # Start a new episode by resetting the enviroment
                observation = self.ddqn_agent.env.reset()

                # Set the new episode flag, controller condition depends on this.
                self.new_episode_flag = True

                while True:

                    if VERBOSE >= STANDARD_REPORTING:
                        print(f'\nEPISODE {episode_i}, STEP: {steps} ************************************************')

                    # Cache frame buffer
                    trial_start_buffer = self.ddqn_agent.buffer.buffer.copy()
                    # Get optimal action based on observation
                    optimal_action = self._get_optimal_action(observation)
                    # Save frame buffer after optimal action
                    optimal_agent_frame_buffer = self.ddqn_agent.buffer.buffer
                    # Restore initial state of frame buffer (for use by Composition)
                    self.ddqn_agent.buffer.buffer = trial_start_buffer

                    if VERBOSE >= ACTION_REPORTING:
                        print(f'\nOUTER LOOP OPTIMAL ACTION:{optimal_action}')

                    # Yield the next input the agent composition. Since this generator is
                    # passed to the outert optimization composition, it must generate
                    # an input dictionary keyed by the inner agent composition node.
                    yield {
                            self.agent_comp: {
                                self.player_percept:[observation[player_coord_slice]],
                                self.predator_percept:[observation[predator_coord_slice]],
                                self.prey_percept:[observation[prey_coord_slice]],
                                self.prey_pred_trial_input_mech:[prey_pred_trialType],
                                self.single_prey_trial_input_mech: [single_prey_trialType],
                                self.double_prey_trial_input_mech: [double_prey_trialType],
                                self.reward_input_mech: [reward]
                                }
                            }

                    # Get agent's action based on perceptual distortion of observation (and application of control)
                    run_results = self.opt_comp.results[-1]
                    agent_action = np.where(run_results[0]==0,0,run_results[0]/np.abs(run_results[0]))

                    if VERBOSE >= ACTION_REPORTING:
                        print(f'OUTER LOOP RUN RESULTS:{run_results}')
                        print(f'OUTER LOOP AGENT ACTION:{agent_action}')

                    if VERBOSE >= STANDARD_REPORTING:
                        if self.agent_comp.controller_mode is BEFORE:
                            print_controller()
                        print(f'\nObservations:'
                              f'\n\tPlayer:\n\t\tveridical: {self.player_percept.parameters.variable.get(self.context)}'
                              f'\n\t\tperceived: {self.player_percept.parameters.value.get(self.context)}'
                              f'\n\tPredator:\n\t\tveridical: {self.predator_percept.parameters.variable.get(self.context)}'
                              f'\n\t\tperceived: {self.predator_percept.parameters.value.get(self.context)}'
                              f'\n\tPrey:\n\t\tveridical: {self.prey_percept.parameters.variable.get(self.context)}'
                              f'\n\t\tperceived: {self.prey_percept.parameters.value.get(self.context)}'
                              f'\n\nActions:\n\tAgent: {agent_action}\n\tOptimal: {optimal_action}'
                              f'\n\nOutcome:\n\t{self.ocm.objective_mechanism.parameters.value.get(self.context)}'
                              )
                        if self.agent_comp.controller_mode is AFTER:
                            print_controller()

                    self.outcome_log.append(self.ocm.objective_mechanism.parameters.value.get(self.context))

                    # Restore frame buffer to state after optimal action taken (at beginning of trial)
                    # This is so that agent's action's can be compared to optimal ones on a trial-by-trial basis
                    self.ddqn_agent.buffer.buffer = optimal_agent_frame_buffer

                    # if ACTION is OPTIMAL_ACTION:
                    #     action = optimal_action
                    # elif ACTION is AGENT_ACTION:
                    #     action = agent_action
                    # else:
                    #     assert False, "Must choose either OPTIMAL_ACTION or AGENT_ACTION"
                    action = agent_action

                    # Get observation for next iteration based on optimal action taken in this one
                    observation, reward, done, _ = self.ddqn_agent.env.step(action)

                    if VERBOSE >= STANDARD_REPORTING:
                        print(f'\nAction Taken (using {ACTION}): {action}')

                    self.new_episode_flag = False
                    steps += 1
                    if done:
                        break

                self.predator_control_log.append(self.ocm.control_signals[1].parameters.value.get(self.context))
                self.prey_control_log.append(self.ocm.control_signals[2].parameters.value.get(self.context))
                self.reward_log.append(reward)

            stop_time = timeit.default_timer()
            print(f'{steps / (stop_time - start_time):.1f} steps/second, {steps} total steps in '
                  f'{stop_time - start_time:.2f} seconds')

            outcome_mean = np.mean(np.asarray(self.outcome_log))
            reward_mean = np.mean(np.asarray(self.reward_log))
            print(f'\nTotal Outcome: {outcome_mean}')
            print(f'\nTotal Reward: {reward_mean}')
            print('predator control log')
            print(self.predator_control_log)
            print('prey control log')
            print(self.prey_control_log)
            predator_control_mean = np.mean(np.asarray(self.predator_control_log))
            print(f'\npredator control MEAN: {predator_control_mean}')
            prey_control_mean = np.mean(np.asarray(self.prey_control_log))
            print(f'\nprey control MEAN: {prey_control_mean}')

            if RENDER:
                self.ddqn_agent.env.render(close=True)  # If visualization is desired

        # Return the generator instantiation function.
        return input_generator

    def run_games(self, cost_rate):

        # Setup data generator.
        input_gen = self.make_input_generator(NUM_EPISODES)

        self.ocm.control_signals[0].parameters.intensity_cost_function.get(self.context).parameters.rate.set(cost_rate, self.context)
        self.ocm.control_signals[0].parameters.intensity_cost_function.get(self.context).parameters.rate.set(cost_rate, self.context)
        self.ocm.control_signals[0].parameters.intensity_cost_function.get(self.context).parameters.rate.set(cost_rate, self.context)

        # Run num_episodes games to completion.
        self.opt_comp.run(inputs=input_gen,
                       bin_execute='LLVM' if PNL_COMPILE else 'Python',
                       context=self.context)

        loss = np.abs(np.mean(np.asarray(self.predator_control_log[-20:])) - 500) + np.mean(np.asarray(self.prey_control_log[-20:]))
        print(f"Loss = {loss}")

        return loss

def run_games(cost_rate):
    return PredatorPreySimulator().run_games(cost_rate)

def run_search():

    from dask.distributed import Client, LocalCluster
    import joblib
    import hypertunity as ht

    #client = Client(scheduler_file='scheduler.json')
    client = Client()
    print(client)

    domain = ht.Domain({
                    "cost_rate": set([-.8])
    })

    # with joblib.parallel_backend('dask'):
    #     with joblib.Parallel() as parallel:
    #         print("Doing the work ... ")
    #         results = parallel(joblib.delayed(run_games)(*domain.sample().as_namedtuple()) for s in range(1))
    #
    # print(results)
    run_games(-.8)

if __name__ == "__main__":
    run_search()

