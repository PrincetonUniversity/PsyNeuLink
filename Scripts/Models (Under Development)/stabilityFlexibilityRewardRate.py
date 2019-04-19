import numpy as np
import psyneulink as pnl


# computeReward Rate
# Inputs: trialInformation[0, 1, 2, 3, 4]
# trialInformation[0] - Task Dimension : [0, 1] or [1, 0]
# trialInformation[1] - Stimulus Dimension: Congruent {[1, 1] or [-1, -1]} // Incongruent {[-1, 1] or [1, -1]}
# trialInformation[2] - Upper Threshold: Probability of DDM choosing upper bound
# trialInformation[3] - Lower Threshold: Probability of DDM choosing lower bound
# trialInformation[4] - Reaction Time: Response time of DDM

def computeRewardRate(trialInformation):

    # constants
    reward = 1
    interTrialInterval = 0.7

    # unloading variable contents
    taskInputs = trialInformation[0]
    stimulusInputs = trialInformation[1]
    upperThreshold = trialInformation[2]
    lowerThreshold = trialInformation[3]
    reactionTime = trialInformation[4]

    # print("Task: ", taskInputs)
    # print("Stimulus: ", stimulusInputs)
    # print("Reaction Time: ", reactionTime)

    # keep track of accuracy
    accuracy = []

    # Beginning of accuracy computation
    colorTrial = (taskInputs[0] == 1)
    motionTrial = (taskInputs[1] == 1)

    # Based on the task dimension information, decide which response is "correct"
    # Obtain accuracy probability from DDM thresholds in "correct" direction
    if colorTrial:
        if stimulusInputs[0] == 1:
            accuracy.append(upperThreshold)
        elif stimulusInputs[0] == -1:
            accuracy.append(lowerThreshold)

    if motionTrial:
        if stimulusInputs[1] == 1:
            accuracy.append(upperThreshold)
        elif stimulusInputs[1] == -1:
            accuracy.append(lowerThreshold)

    # Accounts for initialization runs that have no variable input
    if len(accuracy) == 0:
        accuracy = [0]

    rewardRate = reward * accuracy[0] / (interTrialInterval + reactionTime)

    print("Accuracy: ", accuracy[0])
    print("Reward Rate: ", rewardRate[0])
    print("*********************************************************")
    print()

    return [[rewardRate]]


# Constants as defined in Musslick et al. 2018

tau = 0.9               # Time Constant
DRIFT = 1               # Drift Rate
STARTING_POINT = 0.0    # Starting Point
THRESHOLD = 0.0475      # Threshold
NOISE = 0.04            # Noise
T0 = 0.2                # T0


# Origin Node Inputs
taskTrain = [[1, 0], [1, 0], [1, 0]]
stimulusTrain = [[1, -1], [1, -1], [1, -1]]
runs = len(taskTrain)


# Task Layer: [Color, Motion]
# Origin Node
taskLayer = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                  size=2,
                                  function=pnl.Linear(slope=1, intercept=0),
                                  output_states=[pnl.RESULT],
                                  name='Task Input [I1, I2]')

# Stimulus Layer: [Color Stimulus, Motion Stimulus]
# Origin Node
stimulusInfo = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                     size=2,
                                     function=pnl.Linear(slope=1, intercept=0),
                                     output_states=[pnl.RESULT],
                                     name="Stimulus Input [S1, S2]")

# Activation Layer: [Color Activation, Motion Activation]
# Recurrent: Self Excitation, Mutual Inhibition
# Controlled: Gain Parameter
activation = pnl.RecurrentTransferMechanism(default_variable=[[0.0, 0.0]],
                                            function=pnl.Logistic(gain=1.0),
                                            matrix=[[1.0, -1.0],
                                                    [-1.0, 1.0]],
                                            integrator_mode=True,
                                            integrator_function=pnl.AdaptiveIntegrator(rate=tau),
                                            initial_value=np.array([[0.0, 0.0]]),
                                            output_states=[pnl.RESULT],
                                            name='Task Activations [Activity1, Activity2]')

# Hadamard product of Activation and Stimulus Information
nonAutomatic = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                     size=2,
                                     function=pnl.Linear(slope=1, intercept= 0),
                                     input_states=pnl.InputState(combine=pnl.PRODUCT),
                                     output_states=[pnl.RESULT],
                                     name='Non-Automatic Component [S1*Activity1, S2*Activity2]')

# Summation of nonAutomatic and Automatic Components
ddmCombination = pnl.TransferMechanism(size=1,
                                       function=pnl.Linear(slope=1, intercept=0),
                                       input_states=pnl.InputState(combine=pnl.SUM),
                                       output_states=[pnl.RESULT],
                                       name="Drift = (S1 + S2) + (S1*Activity1 + S2*Activity2)")


decisionMaker = pnl.DDM(function=pnl.DriftDiffusionAnalytical(drift_rate=DRIFT,
                                                              starting_point=STARTING_POINT,
                                                              threshold=THRESHOLD,
                                                              noise=NOISE,
                                                              t0=T0),
                        output_states=[pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME,
                                       pnl.PROBABILITY_UPPER_THRESHOLD,
                                       pnl.PROBABILITY_LOWER_THRESHOLD],
                        name='DDM')

taskLayer.set_log_conditions([pnl.RESULT])
stimulusInfo.set_log_conditions([pnl.RESULT])
activation.set_log_conditions([pnl.RESULT, "mod_gain"])
nonAutomatic.set_log_conditions([pnl.RESULT])
ddmCombination.set_log_conditions([pnl.RESULT])
decisionMaker.set_log_conditions([pnl.PROBABILITY_UPPER_THRESHOLD, pnl.PROBABILITY_LOWER_THRESHOLD,
                                  pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME])

# Composition Creation

stabilityFlexibility = pnl.Composition()

# Add each mechanism into the stabilityFlexibility composition as a node
stabilityFlexibility.add_node(taskLayer)
stabilityFlexibility.add_node(activation)
stabilityFlexibility.add_node(nonAutomatic)
stabilityFlexibility.add_node(stimulusInfo)
stabilityFlexibility.add_node(ddmCombination)
stabilityFlexibility.add_node(decisionMaker)

# Projection connections between mechanisms
stabilityFlexibility.add_projection(sender=taskLayer, receiver=activation)
stabilityFlexibility.add_projection(sender=activation, receiver=nonAutomatic)
stabilityFlexibility.add_projection(sender=stimulusInfo, receiver=nonAutomatic)
stabilityFlexibility.add_projection(sender=stimulusInfo, receiver=ddmCombination)
stabilityFlexibility.add_projection(sender=nonAutomatic, receiver=ddmCombination)
stabilityFlexibility.add_projection(sender=ddmCombination, receiver=decisionMaker)

# Beginning of Controller

# Grid Search Range
searchRange = pnl.SampleSpec(start=0.1, stop=0.3, num=3)

# Modulate the GAIN parameter from activation layer
# Initalize cost function as 0
signal = pnl.ControlSignal(projections=[(pnl.GAIN, activation)],
                           function=pnl.Linear,
                           variable=1.0,
                           intensity_cost_function=pnl.Linear(slope=0.0),
                           allocation_samples=searchRange)

# Use the computeRewardRate function to obtain selection values
# Pass in 5 arguments whenever computeRewardRate is called
objectiveMechanism = pnl.ObjectiveMechanism(monitor=[taskLayer, stimulusInfo,
                                                     (pnl.PROBABILITY_UPPER_THRESHOLD, decisionMaker),
                                                     (pnl.PROBABILITY_LOWER_THRESHOLD, decisionMaker),
                                                     (pnl.RESPONSE_TIME, decisionMaker)],
                                            function=pnl.RewardRateIntegrator,
                                            name="Controller Objective Mechanism")

#  Sets trial history for simulations over specified signal search parameters
metaController = pnl.OptimizationControlMechanism(agent_rep=stabilityFlexibility,
                                                  features=[taskLayer.input_state, stimulusInfo.input_state],
                                                  feature_function=pnl.Buffer(history=3),
                                                  name="Controller",
                                                  objective_mechanism=objectiveMechanism,
                                                  function=pnl.GridSearch(),
                                                  control_signals=[signal])

# Enable Controllers
stabilityFlexibility.add_controller(metaController)
# stabilityFlexibility.enable_controller = True
stabilityFlexibility.controller_mode = pnl.BEFORE

# I actually have no idea what this does but if it's not here none of the information gets loaded into simulations
for i in range(1, len(stabilityFlexibility.controller.input_states)):
    stabilityFlexibility.controller.input_states[i].function.reinitialize()

# END OF COMPOSITION CONSTRUCTION

# TESTING
print("Beginning of Run")
print("--------------------------------------------------------------------------")
print()
# stabilityFlexibility.show_graph(show_model_based_optimizer=True)
# stabilityFlexibility.show_graph(show_model_based_optimizer=True, show_node_structure=True)

inputs = {taskLayer: taskTrain, stimulusInfo: stimulusTrain}
stabilityFlexibility.run(inputs)

# Print Statements to check behavior
print(stabilityFlexibility.results)
print()
activation.log.print_entries()
print()
print(activation.mod_gain)



