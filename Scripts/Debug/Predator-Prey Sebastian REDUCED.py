from psyneulink import *

# *********************************************************************************************************************
# *********************************************** CONSTANTS ***********************************************************
# *********************************************************************************************************************

# Runtime switches:
MPI_IMPLEMENTATION = True
PNL_COMPILE = False
RUN = True
SHOW_GRAPH = False
MODEL_PATH = '../../../double-dqn/models/trained_models/policy_net_trained_0.99_20190214-1651.pt'

# Verbosity levels for console printout
ACTION_REPORTING = 2
STANDARD_REPORTING = 1
VERBOSE = ACTION_REPORTING

# ControlSignal parameters
COST_RATE = -.05
COST_BIAS = 1
ALLOCATION_SAMPLES = [0, 500]

# Condition for executing controller
new_episode_flag = True
def get_new_episode_flag():
    return new_episode_flag

CONTROLLER_CONDITION = Condition(func=get_new_episode_flag) # tells schedule when to run OCM
FEATURE_FUNCTION = AdaptiveIntegrator(rate=0.5)

# **********************************************************************************************************************
# **************************************  CREATE COMPOSITION ***********************************************************
# **********************************************************************************************************************

# **************************************  PROCESSING MECHANISMS ********************************************************

# Perceptual Mechanisms
player_percept = ProcessingMechanism(size=2, function=GaussianDistort(), name="PLAYER PERCEPT")
predator_percept = ProcessingMechanism(size=2, function=GaussianDistort(), name="PREDATOR PERCEPT")
prey_percept = ProcessingMechanism(size=2, function=GaussianDistort(), name="PREY PERCEPT")

# Mechanism used to encode trialtype from environment
trial_type_input_mech = ProcessingMechanism(name="TRIAL TYPE INPUT")

# Mechanism used to encode and reward from environment
reward_input_mech = ProcessingMechanism(name="REWARD INPUT")

# Action Mechanism
# action_mech = ProcessingMechanism(default_variable=[[0,0],[0,0],[0,0]],
#                                   function=get_action, name='ACTION',
#                                   output_ports='agent action')
action_mech = ComparatorMechanism(name='ACTION',sample=player_percept,target=predator_percept)

# ************************************** BASIC COMPOSITION *************************************************************

agent_comp = Composition(name='PREDATOR-PREY COMPOSITION')
agent_comp.add_nodes([player_percept, predator_percept, prey_percept, trial_type_input_mech, reward_input_mech])
agent_comp.add_node(action_mech, required_roles=[NodeRole.OUTPUT])

# **************************************  CONOTROL APPARATUS ***********************************************************

ocm = OptimizationControlMechanism(name='EVC',
                                   features=[trial_type_input_mech],
                                   # feature_function=FEATURE_FUNCTION,
                                   agent_rep=RegressionCFA(
                                           name='RegressionCFA',
                                           update_weights=BayesGLM(mu_0=0.5, sigma_0=0.1),
                                           prediction_terms=[PV.F, PV.C, PV.COST]
                                   ),
                                   function=GridSearch(direction=MAXIMIZE, save_values=True),

                                   objective_mechanism=ObjectiveMechanism(name='OBJECTIVE MECHANISM',
                                                                          monitor=[reward_input_mech]),
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
agent_comp.controller_condition=CONTROLLER_CONDITION

if SHOW_GRAPH:
    agent_comp.show_graph(show_controller=True, show_cim=True)

# *********************************************************************************************************************
# ******************************************   RUN SIMULATION  ********************************************************
# *********************************************************************************************************************

def main():

    context = 'TEST'
    if PNL_COMPILE:
        BIN_EXECUTE = 'LLVM'
    else:
        BIN_EXECUTE = 'Python'

    # Get agent's action based on perceptual distortion of observation (and application of control)
    run_results = agent_comp.run(inputs={player_percept:[[ 219., -123.], [ 199., 135.]],
                                         predator_percept:[[-200., -108.], [-190., 223.]],
                                         prey_percept:[[-282.,  125.], [260., -182.]],
                                         trial_type_input_mech:[[2],[2]],
                                         reward_input_mech:[[0],[1]]},
                                 context=context,
                                 bin_execute=BIN_EXECUTE,
# <<<<<<< HEAD
                                 animate={'show_controller':True,
                                          'show_cim':True}
                                 )
# # =======
#                                  )
    print(run_results)
# >>>>>>> 431b1188c0625f981643db64d2db3226eed4ddb2

if RUN:
    if __name__ == "__main__":
        main()

