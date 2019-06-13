import numpy as np
import psyneulink as pnl
import sys
import csv
from random import randint

# if len(sys.argv) < 2:
#     print("Error: Need File Name")

# fileName = sys.argv[1]
# fileName = '1.csv'

def readFile(fileName):

    fileContent = []

    # reading csv file
    with open(fileName, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            fileContent.append(row)

        # cast each string into an int
        for i in range(0, len(fileContent)):
            for j in range(0, len(fileContent[0])):
                fileContent[i][j] = int(fileContent[i][j])

    return fileContent


def getTaskTrain(dataFile):
    taskTrain = []
    trials = len(dataFile)
    for i in range(0, trials):
        task = [0, 0]
        if dataFile[i][0] == 1:
            task = [1, 0]
        elif dataFile[i][0] == 2:
            task = [0, 1]

        taskTrain.append(task)

    return taskTrain

def getStimulusTrain():
    a = [-1, 1]
    b = [1, -1]

    stimulusTrain = []
    for i in range(0, 260):
        x = randint(0, 1)
        if x == 0:
            stimulusTrain.append(a)
        elif x == 1:
            stimulusTrain.append(b)
    return stimulusTrain


def insertCues(tasks, stimuli):
    trials = len(tasks)
    newTasks = []
    newStimuli = []
    for i in range(0, trials):
        newTasks.append(tasks[i])
        newTasks.append(tasks[i])

        newStimuli.append([0, 0])
        newStimuli.append(stimuli[i])

    return newTasks, newStimuli


def extractValues(outputLog):
    decisionVariable = []
    probabilityUpper = []
    probabilityLower = []
    responseTime = []

    DECISION_VARIABLE = outputLog[1][1][4]
    PROBABILITY_LOWER_THRESHOLD = outputLog[1][1][5]
    PROBABILITY_UPPER_THRESHOLD = outputLog[1][1][6]
    RESPONSE_TIME = outputLog[1][1][7]

    for j in range(1, len(PROBABILITY_LOWER_THRESHOLD)):
        decision = DECISION_VARIABLE[j]
        trialUpper = PROBABILITY_UPPER_THRESHOLD[j]
        trialLower = PROBABILITY_LOWER_THRESHOLD[j]
        reaction = RESPONSE_TIME[j]

        decisionVariable.append(decision[0])
        probabilityUpper.append(trialUpper[0])
        probabilityLower.append(trialLower[0])
        responseTime.append(reaction[0])

    return probabilityUpper, probabilityLower


def computeAccuracy(variable):

    taskInputs = variable[0]
    stimulusInputs = variable[1]
    upperThreshold = variable[2]
    lowerThreshold = variable[3]

    accuracy = []
    for i in range(0, len(taskInputs)):

        colorTrial = (taskInputs[i][0] > 0)
        motionTrial = (taskInputs[i][1] > 0)

        # during color trials

        if colorTrial:
            # if the correct answer is the upper threshold
            if stimulusInputs[i][0] > 0:
                accuracy.append(upperThreshold[i])
                # print('Color Trial: 1')

            # if the correct answer is the lower threshold
            elif stimulusInputs[i][0] < 0:
                accuracy.append(lowerThreshold[i])
                # print('Color Trial: -1')

        if motionTrial:
            # if the correct answer is the upper threshold
            if stimulusInputs[i][1] > 0:
                accuracy.append(upperThreshold[i])
                # print('Motion Trial: 1')

            # if the correct answer is the lower threshold
            elif stimulusInputs[i][1] < 0:
                accuracy.append(lowerThreshold[i])
                # print('Motion Trial: -1')

    return accuracy



##### BEGIN STABILITY FLEXIBILITY MODEL CONSTRUCTION

def runStabilityFlexibility(tasks, stimuli, gain):

    integrationConstant = 0.8 # time constant
    DRIFT = 0.25 # Drift Rate
    STARTING_POINT = 0.0 # Starting Point
    THRESHOLD = 0.05 # Threshold
    NOISE = 0.1 # Noise,
    T0 = 0.2 # T0
    wa = 0.2
    g = gain


    # first element is color task attendance, second element is motion task attendance
    inputLayer = pnl.TransferMechanism(#default_variable=[[0.0, 0.0]],
                                       size=2,
                                       function=pnl.Linear(slope=1, intercept=0),
                                       output_states = [pnl.RESULT],
                                       name='Input')
    inputLayer.set_log_conditions([pnl.RESULT])

    # Recurrent Transfer Mechanism that models the recurrence in the activation between the two stimulus and action
    # dimensions. Positive self excitation and negative opposite inhibition with an integrator rate = tau
    # Modulated variable in simulations is the GAIN variable of this mechanism
    activation = pnl.RecurrentTransferMechanism(default_variable=[[0.0, 0.0]],
                                                function=pnl.Logistic(gain=g),
                                                matrix=[[1.0, -1.0],
                                                        [-1.0, 1.0]],
                                                integrator_mode = True,
                                                integrator_function=pnl.AdaptiveIntegrator(rate=integrationConstant),
                                                initial_value=np.array([[0.0, 0.0]]),
                                                output_states = [pnl.RESULT],
                                                name = 'Activity')

    activation.set_log_conditions([pnl.RESULT, "mod_gain"])


    stimulusInfo = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                         size = 2,
                                         function = pnl.Linear(slope=1, intercept=0),
                                         output_states = [pnl.RESULT],
                                         name = "Stimulus Info")

    stimulusInfo.set_log_conditions([pnl.RESULT])

    congruenceWeighting = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                              size = 2,
                                              function=pnl.Linear(slope=wa, intercept= 0),
                                              name = 'Congruence * Automatic Component')


    controlledElement = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                              size = 2,
                                              function=pnl.Linear(slope=1, intercept= 0),
                                              input_states=pnl.InputState(combine=pnl.PRODUCT),
                                              output_states = [pnl.RESULT],
                                              name = 'Stimulus Info * Activity')

    controlledElement.set_log_conditions([pnl.RESULT])

    ddmCombination = pnl.TransferMechanism(size = 1,
                                           function = pnl.Linear(slope=1, intercept=0),
                                           output_states = [pnl.RESULT],
                                           name = "DDM Integrator")

    ddmCombination.set_log_conditions([pnl.RESULT])

    decisionMaker = pnl.DDM(function=pnl.DriftDiffusionAnalytical(drift_rate = DRIFT,
                                                                     starting_point = STARTING_POINT,
                                                                     threshold = THRESHOLD,
                                                                     noise = NOISE,
                                                                     t0 = T0),
                                                                     output_states = [pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME,
                                                                                      pnl.PROBABILITY_UPPER_THRESHOLD, pnl.PROBABILITY_LOWER_THRESHOLD],
                                                                     name='DDM')

    decisionMaker.set_log_conditions([pnl.PROBABILITY_UPPER_THRESHOLD, pnl.PROBABILITY_LOWER_THRESHOLD,
                                pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME])

    ########### Composition

    stabilityFlexibility = pnl.Composition()

    ### NODE CREATION

    stabilityFlexibility.add_node(inputLayer)
    stabilityFlexibility.add_node(activation)
    stabilityFlexibility.add_node(congruenceWeighting)
    stabilityFlexibility.add_node(controlledElement)
    stabilityFlexibility.add_node(stimulusInfo)
    stabilityFlexibility.add_node(ddmCombination)
    stabilityFlexibility.add_node(decisionMaker)


    stabilityFlexibility.add_projection(sender = inputLayer, receiver = activation)
    stabilityFlexibility.add_projection(sender = activation, receiver = controlledElement)
    stabilityFlexibility.add_projection(sender = stimulusInfo, receiver = congruenceWeighting)
    stabilityFlexibility.add_projection(sender = stimulusInfo, receiver = controlledElement)
    stabilityFlexibility.add_projection(sender = congruenceWeighting, receiver = ddmCombination)
    stabilityFlexibility.add_projection(sender = controlledElement, receiver = ddmCombination)
    stabilityFlexibility.add_projection(sender = ddmCombination, receiver = decisionMaker)

    runs = len(tasks)
    inputs = {inputLayer: tasks, stimulusInfo: stimuli}

    stabilityFlexibility.run(inputs)

    decisions = decisionMaker.log.nparray()
    upper, lower = extractValues(decisions)
    modelResults = [tasks, stimuli, upper, lower]
    accuracies = computeAccuracy(modelResults)

    activations = activation.log.nparray()
    activity1 = []
    activity2 = []
    for i in range(0, runs):
        activity1.append(activations[1][1][4][i+1][0])
        activity2.append(activations[1][1][4][i+1][1])

    stabilityFlexibility.show_graph()

    return accuracies, activity1, activity2



# TESTING INPUTS

## NO SWITCHING

# Observe the Effect of Gain on Congruent Trials
# taskTrain = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
#              [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]
#
# stimulusTrain = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
#                  [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]


## Observe the Effect of Gain on Incongruent Trials
# taskTrain = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
#              [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]
#
# stimulusTrain = [[1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1],
#                  [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1]]

## SWITCHING

# CONGRUENT TRIALS ONLY
# taskTrain = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
#              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
#
# stimulusTrain = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
#                  [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]

## INCONGRUENT TRIALS ONLY
# taskTrain = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
#              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
#
# stimulusTrain = [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1],
#                  [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1]]

## DIFFERENT AMOUNTS OF SWITCHING

# # 100%  Task Switch
# taskTrain = [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1],
#              [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]
#
# stimulusTrain = [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1],
#                  [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]

## 50%  Task Switch
# taskTrain = [[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0],
#              [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1]]
#
# stimulusTrain = [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1],
#                  [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]

# # 20% Task Switch
# taskTrain = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
#               [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
#
# stimulusTrain = [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1],
#                  [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]

# ## 10% Switch Rate
taskTrain = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
             [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]

stimulusTrain = [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1],
                 [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]

## 5 Switches Switch Rate
# taskTrain = [[1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0],
#              [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0]]
#
# stimulusTrain = [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1],
#                  [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]

## 2 switches
# taskTrain = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1],
#              [0, 1], [0, 1], [1, 0], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0]]
#
# stimulusTrain = [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1],
#                  [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
#
taskTrain, stimulusTrain = insertCues(taskTrain, stimulusTrain)

gainsToTest = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

# dataFile = readFile(fileName)
# taskTrain = getTaskTrain(dataFile)[0:5]
# stimulusTrain = getStimulusTrain()[0:5]

# taskTrain, stimulusTrain = insertCues(taskTrain, stimulusTrain)

# taskTrain = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1],
#              [0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]
#
# stimulusTrain = [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1],
#                  [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]




gains = {}
totalAccuracies = []
for g in gainsToTest:

    acc, act1, act2 = runStabilityFlexibility(taskTrain, stimulusTrain, g)

    totalAccuracy = np.mean(acc)
    totalAccuracies.append(totalAccuracy)

    gains[g] = {"totalAccuracies": acc,
                "unitActivity1": act1,
                "unitActivity2": act2}

optimalGain = gainsToTest[np.argmax(totalAccuracies)]

# print("File Name:", fileName)
# condition = dataFile[0][14]
# print("Condition:", condition)

print(optimalGain)
print(totalAccuracies)

print(gains)
print()




