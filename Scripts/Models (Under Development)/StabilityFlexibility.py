import numpy as np
import psyneulink as pnl
import time


# computeAccuracy(trialInformation)
# Inputs: trialInformation[0, 1, 2, 3]
# trialInformation[0] - Task Dimension : [0, 1] or [1, 0]
# trialInformation[1] - Stimulus Dimension: Congruent {[1, 1] or [-1, -1]} // Incongruent {[-1, 1] or [1, -1]}
# trialInformation[2] - Upper Threshold: Probability of DDM choosing upper bound
# trialInformation[3] - Lower Threshold: Probability of DDM choosing lower bound

def computeAccuracy(trialInformation):

    # Unload contents of trialInformation
    # Origin Node Inputs
    taskInputs = trialInformation[0]
    stimulusInputs = trialInformation[1]

    # DDM Outputs
    upperThreshold = trialInformation[2]
    lowerThreshold = trialInformation[3]

    # Keep Track of Accuracy
    accuracy = []

    # Beginning of Accuracy Calculation
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

    print("Accuracy: ", accuracy[0])
    print()

    return [accuracy]

def accuracy_integrator(variable, self=None, execution_id=None):

    if self.context.initialization_status is pnl.ContextFlags.INITIALIZING:
        return [[0.]]
    try:
        previous_value = self.parameters.value.history[execution_id][0]
    except IndexError:
        previous_value = [[0.]]

    # Unload contents of trialInformation

    # Origin Node Inputs
    taskInputs, stimulusInputs, upperThreshold, lowerThreshold = variable

    # Keep Track of Accuracy
    accuracy = 0

    # Beginning of Accuracy Calculation
    colorTrial = (taskInputs[0] == 1)
    motionTrial = (taskInputs[1] == 1)

    # Based on the task dimension information, decide which response is "correct"
    # Obtain accuracy probability from DDM thresholds in "correct" direction
    if colorTrial:
        if stimulusInputs[0] == 1:
            accuracy = upperThreshold[0]
        elif stimulusInputs[0] == -1:
            accuracy = lowerThreshold[0]

    if motionTrial:
        if stimulusInputs[1] == 1:
            accuracy = upperThreshold[0]
        elif stimulusInputs[1] == -1:
            accuracy = lowerThreshold[0]

    adjusted_value = accuracy + previous_value

    print("Accuracy: ", accuracy)
    print("Previous Value: ", previous_value)

    return self.convert_output_type(adjusted_value)

# BEGIN: Composition Construction

# Constants as defined in Musslick et al. 2018
tau = 0.9               # Time Constant
DRIFT = 1               # Drift Rate
STARTING_POINT = 0.0    # Starting Point
THRESHOLD = 0.0475      # Threshold
NOISE = 0.5            # Noise
T0 = 0.2                # T0
congruentWeight = 1

# Task Layer: [Color, Motion] {0, 1} Mutually Exclusive
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

congruenceWeighting = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                          size = 2,
                                          function=pnl.Linear(slope=congruentWeight, intercept= 0),
                                          name = 'Congruence * Automatic Component')

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
                                            name='Task Activations [Act 1, Act 2]')

# Hadamard product of Activation and Stimulus Information
nonAutomaticComponent = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                              size=2,
                                              function=pnl.Linear(slope=1, intercept=0),
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
nonAutomaticComponent.set_log_conditions([pnl.RESULT])
ddmCombination.set_log_conditions([pnl.RESULT])
decisionMaker.set_log_conditions([pnl.PROBABILITY_UPPER_THRESHOLD, pnl.PROBABILITY_LOWER_THRESHOLD,
                                  pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME])


# Composition Creation

stabilityFlexibility = pnl.Composition()

# Node Creation
stabilityFlexibility.add_node(taskLayer)
stabilityFlexibility.add_node(activation)
stabilityFlexibility.add_node(congruenceWeighting)
stabilityFlexibility.add_node(nonAutomaticComponent)
stabilityFlexibility.add_node(stimulusInfo)
stabilityFlexibility.add_node(ddmCombination)
stabilityFlexibility.add_node(decisionMaker)

# Projection Creation
stabilityFlexibility.add_projection(sender=taskLayer, receiver=activation)
stabilityFlexibility.add_projection(sender=activation, receiver=nonAutomaticComponent)
stabilityFlexibility.add_projection(sender=stimulusInfo, receiver=nonAutomaticComponent)
stabilityFlexibility.add_projection(sender=stimulusInfo, receiver=congruenceWeighting)
stabilityFlexibility.add_projection(sender=congruenceWeighting, receiver=ddmCombination)
stabilityFlexibility.add_projection(sender=nonAutomaticComponent, receiver=ddmCombination)
stabilityFlexibility.add_projection(sender=ddmCombination, receiver=decisionMaker)


# Beginning of Controller

# Grid Search Range
searchRange = pnl.SampleSpec(start=1.0, stop=1.9, num=10)

# Modulate the GAIN parameter from activation layer
# Initalize cost function as 0
signal = pnl.ControlSignal(projections=[(pnl.GAIN, activation)],
                           function=pnl.Linear,
                           variable=1.0,
                           intensity_cost_function=pnl.Linear(slope=0.0),
                           allocation_samples=searchRange)

# Use the computeAccuracy function to obtain selection values
# Pass in 4 arguments whenever computeRewardRate is called
objectiveMechanism = pnl.ObjectiveMechanism(monitor=[taskLayer, stimulusInfo,
                                                     (pnl.PROBABILITY_UPPER_THRESHOLD, decisionMaker),
                                                     (pnl.PROBABILITY_LOWER_THRESHOLD, decisionMaker)],
                                            function=accuracy_integrator,
                                            name="Controller Objective Mechanism")
objectiveMechanism.set_log_conditions(items=pnl.VALUE)

#  Sets trial history for simulations over specified signal search parameters
metaController = pnl.OptimizationControlMechanism(agent_rep=stabilityFlexibility,
                                                  features=[taskLayer.input_state, stimulusInfo.input_state],
                                                  feature_function=pnl.Buffer(history=5),
                                                  name="Controller",
                                                  objective_mechanism=objectiveMechanism,
                                                  function=pnl.GridSearch(),
                                                  control_signals=[signal])

stabilityFlexibility.add_controller(metaController)
stabilityFlexibility.enable_controller = True
stabilityFlexibility.controller_mode = pnl.BEFORE

for i in range(1, len(stabilityFlexibility.controller.input_states)):
    stabilityFlexibility.controller.input_states[i].function.reinitialize()

# END OF COMPOSITION CONSTRUCTION


# TESTING

print("Beginning of Run")
print("--------------------------------------------------------------------------")
print()

# stabilityFlexibility.show_graph()
# stabilityFlexibility.show_graph(show_controller=True)
# stabilityFlexibility.show_graph(show_model_based_optimizer=True, show_node_structure=True)

# Origin Node Inputs
taskTrain = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
             [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
stimulusTrain = [[1, -1], [1, -1], [1, -1], [1, -1], [1, -1],
                 [1, -1], [1, -1], [1, -1], [1, -1], [1, -1],]


# eid2 = 50
# times = [time.time()]
# def sanityCheck():
#     if stabilityFlexibility.scheduler_processing.clocks[eid2].time.trial % 5 == 0:
#         times.append(time.time())
#         print(times[-1] - times[-2])

inputs = {taskLayer: taskTrain, stimulusInfo: stimulusTrain}

#print(activation.parameters)
activation.parameters.value.retain_old_simulation_data = True


stabilityFlexibility.run(inputs)
#stabilityFlexibility.show_graph(show_controller=True)
# stabilityFlexibility.run(inputs, call_after_trial=sanityCheck, execution_id=eid2)
print(stabilityFlexibility.results)
# stabilityFlexibility.scheduler_processing.clocks[eid2]


activation.log.print_entries()
print()
decisionMaker.log.print_entries()
print(pnl.BEFORE)
print()
objectiveMechanism.log.print_entries()


