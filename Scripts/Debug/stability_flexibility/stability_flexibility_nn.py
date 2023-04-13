import psyneulink as pnl
import numpy as np
import random
import pytest
import pandas as pd


# Define function to generate a counterbalanced trial sequence with a specified switch trial frequency
def generate_trial_sequence(n, frequency, seed: int = None):

    # Compute trial number
    nTotalTrials = n
    switchFrequency = frequency

    nSwitchTrials = int(nTotalTrials * switchFrequency)
    nRepeatTrials = int(nTotalTrials - nSwitchTrials)

    # Determine task transitions
    transitions = [1] * nSwitchTrials + [0] * nRepeatTrials
    rng = np.random.RandomState(seed)
    order = rng.permutation(list(range(nTotalTrials)))
    transitions[:] = [transitions[i] for i in order]

    # Determine stimuli with 50% congruent trials
    stimuli = (
        [[1, 0, 1, 0]] * int(nSwitchTrials / 4)
        + [[1, 0, 0, 1]] * int(nSwitchTrials / 4)
        + [[0, 1, 0, 1]] * int(nSwitchTrials / 4)
        + [[0, 1, 1, 0]] * int(nSwitchTrials / 4)
        + [[1, 0, 1, 0]] * int(nRepeatTrials / 4)
        + [[1, 0, 0, 1]] * int(nRepeatTrials / 4)
        + [[0, 1, 0, 1]] * int(nRepeatTrials / 4)
        + [[0, 1, 1, 0]] * int(nRepeatTrials / 4)
    )
    stimuli[:] = [stimuli[i] for i in order]

    # Determine cue-stimulus intervals
    CSI = (
        [500] * int(nSwitchTrials / 8)
        + [500] * int(nSwitchTrials / 8)
        + [500] * int(nSwitchTrials / 8)
        + [500] * int(nSwitchTrials / 8)
        + [500] * int(nSwitchTrials / 8)
        + [500] * int(nSwitchTrials / 8)
        + [500] * int(nSwitchTrials / 8)
        + [500] * int(nSwitchTrials / 8)
        + [500] * int(nRepeatTrials / 8)
        + [500] * int(nRepeatTrials / 8)
        + [500] * int(nRepeatTrials / 8)
        + [500] * int(nRepeatTrials / 8)
        + [500] * int(nRepeatTrials / 8)
        + [500] * int(nRepeatTrials / 8)
        + [500] * int(nRepeatTrials / 8)
        + [500] * int(nRepeatTrials / 8)
    )
    CSI[:] = [CSI[i] for i in order]

    # Set the task order
    tasks = [[1, 0]] * (nTotalTrials + 1)
    for i in list(range(nTotalTrials)):
        if transitions[i] == 0:
            tasks[i + 1] = tasks[i]
        if transitions[i] == 1:
            if tasks[i] == [1, 0]:
                tasks[i + 1] = [0, 1]
            if tasks[i] == [0, 1]:
                tasks[i + 1] = [1, 0]
    tasks = tasks[1:]

    # Determine correct response based on stimulus and task input
    # First, determine which stimulus input is task-relevant
    relevantInput = np.repeat(tasks, 2, axis=1) * stimuli
    relevantIndex = np.argwhere(relevantInput == 1)[:,1]

    # If index of relevant input is 0 or 2 then correct response is 1, else -1
    correctResponse = np.where(np.logical_or(relevantIndex == 0, relevantIndex == 2), 1, -1)

    return tasks, stimuli, CSI, correctResponse


# Stability-Flexibility Model
def make_stab_flex(
    gain=3.0,
    leak=3.0,
    competition=2.0,
    lca_time_step_size=0.01,
    non_decision_time=0.2,
    stim_hidden_wt=1.5,
    starting_value=0.0,
    threshold=0.1,
    ddm_noise=0.1,
    lca_noise=0.0,
    hidden_resp_wt=2.0,
    ddm_time_step_size=0.01,
    rng_seed=None,
):

    GAIN = gain
    LEAK = leak
    COMP = competition
    STIM_HIDDEN_WT = stim_hidden_wt  # Stimulus Input to Hidden Unit Connection Weight

    STARTING_POINT = starting_value  # Starting Point
    THRESHOLD = threshold  # Threshold
    NOISE = ddm_noise  # Noise
    HIDDEN_RESP_WT = hidden_resp_wt  # Stimulus-Response Mapping Weight
    NON_DECISION_TIME = non_decision_time

    # Task Input: [Parity, Magnitude] {0, 1} Mutually Exclusive
    # Origin Node
    taskInput = pnl.TransferMechanism(name="Task Input", size=2)    # Note default function is linear

    # Stimulus Input: [Odd, Even, Small, Large] {0, 1}
    # Origin Node
    stimulusInput = pnl.TransferMechanism(name="Stimulus Input", size=4)

    # Cue-To-Stimulus Interval Input
    # Origin Node
    cueInterval = pnl.TransferMechanism(name="Cue-Stimulus Interval", size=1)

    # Correct Response Info {1, -1}
    # Origin Node
    correctResponseInfo = pnl.TransferMechanism(name="Correct Response Info", size=1)

    # Control Units: [Parity Activation, Magnitude Activation]
    controlModule = pnl.LCAMechanism(
        name="Task Activations [C1, C2]",
        size=2,
        function=pnl.Logistic(gain=GAIN),
        leak=LEAK,
        competition=COMP,
        self_excitation=0,
        noise=lca_noise,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=1200,
        time_step_size=lca_time_step_size
    )

    # Control Mechanism Setting Cue-To-Stimulus Interval
    csiController = pnl.ControlMechanism(
        monitor_for_control=cueInterval,
        control_signals=[(pnl.TERMINATION_THRESHOLD, controlModule)],
        modulation=pnl.OVERRIDE,
    )

    # Stimulus Input to Hidden Weighting
    stimulusWeighting = pnl.TransferMechanism(
        name="Stimulus Input to Hidden Weighting",
        size=4,
        function=pnl.Linear(slope=STIM_HIDDEN_WT, intercept=0),
    )

    # Hidden Units [Odd, Even, Small, Large]
    hiddenLayer = pnl.TransferMechanism(
        name="Hidden Units",
        size=4,
        function=pnl.Logistic(gain=1, bias=-4),
        input_ports=pnl.InputPort(combine=pnl.SUM)
    )

    # Hidden to Response Weighting
    hiddenWeighting = pnl.TransferMechanism(
        name="Hidden Unit to Response Weighting",
        size=4,
        function=pnl.Linear(slope=HIDDEN_RESP_WT, intercept=0)
    )

    # Response Units [Left, Right]
    responseLayer = pnl.TransferMechanism(
        name="Response Units",
        size=2,
        function=pnl.Logistic(gain=1),
        input_ports=pnl.InputPort(combine=pnl.SUM)
    )

    # Difference in activation of response units
    ddmCombination = pnl.TransferMechanism(
        name="Drift",
        size=1,
        input_ports=pnl.InputPort(combine=pnl.SUM)
    )

    # Ensure upper boundary of DDM is always correct response by multiplying DDM input by correctResponseInfo
    ddmRecodeDrift = pnl.TransferMechanism(
        name="Recoded Drift = Drift * correctResponseInfo",
        size=1,
        input_ports=pnl.InputPort(combine=pnl.PRODUCT)
    )

    # Decision Module
    decisionMaker = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=STARTING_POINT,
            threshold=THRESHOLD,
            noise=NOISE,
            time_step_size=ddm_time_step_size,
            non_decision_time=NON_DECISION_TIME,
        ),
        reset_stateful_function_when=pnl.AtTrialStart(),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )

    taskInput.set_log_conditions([pnl.RESULT])
    stimulusInput.set_log_conditions([pnl.RESULT])
    cueInterval.set_log_conditions([pnl.RESULT])
    correctResponseInfo.set_log_conditions([pnl.RESULT])
    controlModule.set_log_conditions([pnl.RESULT, "termination_threshold"])
    stimulusWeighting.set_log_conditions([pnl.RESULT])
    hiddenLayer.set_log_conditions([pnl.RESULT])
    hiddenWeighting.set_log_conditions([pnl.RESULT])
    responseLayer.set_log_conditions([pnl.RESULT])
    ddmCombination.set_log_conditions([pnl.RESULT])
    ddmRecodeDrift.set_log_conditions([pnl.RESULT])
    decisionMaker.set_log_conditions([pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME])

    # Composition Creation
    stabilityFlexibility = pnl.Composition()

    # Node Creation
    stabilityFlexibility.add_node(taskInput)
    stabilityFlexibility.add_node(stimulusInput)
    stabilityFlexibility.add_node(cueInterval)
    stabilityFlexibility.add_node(correctResponseInfo)
    stabilityFlexibility.add_node(controlModule)
    stabilityFlexibility.add_node(csiController)
    stabilityFlexibility.add_node(stimulusWeighting)
    stabilityFlexibility.add_node(hiddenLayer)
    stabilityFlexibility.add_node(hiddenWeighting)
    stabilityFlexibility.add_node(responseLayer)
    stabilityFlexibility.add_node(ddmCombination)
    stabilityFlexibility.add_node(ddmRecodeDrift)
    stabilityFlexibility.add_node(decisionMaker)

    # Projection Creation
    stabilityFlexibility.add_projection(sender=taskInput, receiver=controlModule,
                                        projection=pnl.MappingProjection(matrix=np.array([[1, 0], [0, 1]]))
                                        )
    stabilityFlexibility.add_projection(sender=stimulusInput, receiver=stimulusWeighting,
                                        projection=pnl.MappingProjection(matrix=np.array([[1, 0, 0, 0],
                                                                                          [0, 1, 0, 0],
                                                                                          [0, 0, 1, 0],
                                                                                          [0, 0, 0, 1]]))
                                        )
    stabilityFlexibility.add_projection(sender=controlModule, receiver=hiddenLayer,
                                        projection=pnl.MappingProjection(matrix=np.array([[4, 4, 0, 0],
                                                                                          [0, 0, 4, 4]]))
                                        )
    stabilityFlexibility.add_projection(sender=stimulusWeighting, receiver=hiddenLayer,
                                        projection=pnl.MappingProjection(matrix=np.array([[1, -1, 0, 0],
                                                                                          [-1, 1, 0, 0],
                                                                                          [0, 0, 1, -1],
                                                                                          [0, 0, -1, 1]]))
                                        )
    stabilityFlexibility.add_projection(sender=hiddenLayer, receiver=hiddenWeighting,
                                        projection=pnl.MappingProjection(matrix=np.array([[1, 0, 0, 0],
                                                                                          [0, 1, 0, 0],
                                                                                          [0, 0, 1, 0],
                                                                                          [0, 0, 0, 1]]))
                                        )
    stabilityFlexibility.add_projection(sender=hiddenWeighting, receiver=responseLayer,
                                        projection=pnl.MappingProjection(matrix=np.array([[1, -1],
                                                                                          [-1, 1],
                                                                                          [1, -1],
                                                                                          [-1, 1]]))
                                        )
    stabilityFlexibility.add_projection(sender=responseLayer, receiver=ddmCombination,
                                        projection=pnl.MappingProjection(matrix=np.array([[1], [-1]]))
                                        )
    stabilityFlexibility.add_projection(sender=ddmCombination, receiver=ddmRecodeDrift)
    stabilityFlexibility.add_projection(sender=correctResponseInfo, receiver=ddmRecodeDrift)
    stabilityFlexibility.add_projection(sender=ddmRecodeDrift, receiver=decisionMaker)

    # Hot-fix currently necessary to allow control module and DDM to execute in parallel in compiled mode
    # We need two gates in order to output both values (decision and response) from the ddm
    decisionGate = pnl.ProcessingMechanism(size=1, name="DECISION_GATE")
    stabilityFlexibility.add_node(decisionGate)

    responseGate = pnl.ProcessingMechanism(size=1, name="RESPONSE_GATE")
    stabilityFlexibility.add_node(responseGate)

    stabilityFlexibility.add_projection(
        sender=decisionMaker.output_ports[0], receiver=decisionGate
    )
    stabilityFlexibility.add_projection(
        sender=decisionMaker.output_ports[1], receiver=responseGate
    )

    # Sets scheduler conditions, so that the gates are not executed (and hence the composition doesn't finish) until decisionMaker is finished
    stabilityFlexibility.scheduler.add_condition(
        decisionGate, pnl.WhenFinished(decisionMaker)
    )
    stabilityFlexibility.scheduler.add_condition(
        responseGate, pnl.WhenFinished(decisionMaker)
    )

    return stabilityFlexibility


def run_stab_flex(
    taskTrain,
    stimulusTrain,
    cueTrain,
    correctResponse,
    gain=3.0,
    leak=3.0,
    competition=2.0,
    lca_time_step_size=0.01,
    non_decision_time=0.2,
    stim_hidden_wt=1.5,
    starting_value=0.0,
    threshold=0.1,
    ddm_noise=0.1,
    lca_noise=0.0,
    short_csi=None,
    delta_csi=None,
    hidden_resp_wt=2.0,
    ddm_time_step_size=0.01,
    rng_seed=None,
):

    # If the user has specified a short_csi and delta_csi as parameters, modify cueTrain
    # such that its min is replaced with short_csi and its max (short_csi + delta_csi)
    if delta_csi and short_csi:
        csi_params = np.zeros(cueTrain.shape)
        csi_params[cueTrain == np.min(cueTrain)] = short_csi
        csi_params[cueTrain == np.max(cueTrain)] = short_csi + delta_csi
    else:
        csi_params = cueTrain

    stabilityFlexibility = make_stab_flex(
        gain=gain,
        leak=leak,
        competition=competition,
        lca_time_step_size=lca_time_step_size,
        non_decision_time=non_decision_time,
        stim_hidden_wt=stim_hidden_wt,
        starting_value=starting_value,
        threshold=threshold,
        ddm_noise=ddm_noise,
        lca_noise=lca_noise,
        hidden_resp_wt=hidden_resp_wt,
        ddm_time_step_size=ddm_time_step_size,
        rng_seed=rng_seed,
    )

    taskInput = stabilityFlexibility.nodes["Task Input"]
    stimulusInput = stabilityFlexibility.nodes["Stimulus Input"]
    cueInterval = stabilityFlexibility.nodes["Cue-Stimulus Interval"]
    correctInfo = stabilityFlexibility.nodes["Correct Response Info"]

    inputs = {
        taskInput: taskTrain,
        stimulusInput: stimulusTrain,
        cueInterval: csi_params,
        correctInfo: correctResponse
    }

    stabilityFlexibility.run(inputs)

    return stabilityFlexibility


# taskInput.log.print_entries()
# stimulusInput.log.print_entries()
# cueInterval.log.print_entries()
# correctResponseInfo.log.print_entries()
# controlModule.log.print_entries()
# stimulusWeighting.log.print_entries()
# hiddenLayer.log.print_entries()
# hiddenWeighting.log.print_entries()
# responseLayer.log.print_entries()
# ddmCombination.log.print_entries()
# ddmRecodeDrift.log.print_entries()
# decisionMaker.log.print_entries()

if __name__ == "__main__":

    taskTrain, stimulusTrain, cueTrain, correctResponse = generate_trial_sequence(240, 0.5)
    taskTrain = taskTrain[0:3]
    stimulusTrain = stimulusTrain[0:3]
    cueTrain = cueTrain[0:3]
    correctResponse = correctResponse[0:3]

    comp = run_stab_flex(
        taskTrain,
        stimulusTrain,
        cueTrain,
        correctResponse,
        ddm_time_step_size=0.01,
        lca_time_step_size=0.01,
    )
    print(comp.results)