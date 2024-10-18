import psyneulink as pnl
import numpy as np


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
        [[1, 1]] * int(nSwitchTrials / 4)
        + [[1, -1]] * int(nSwitchTrials / 4)
        + [[-1, -1]] * int(nSwitchTrials / 4)
        + [[-1, 1]] * int(nSwitchTrials / 4)
        + [[1, 1]] * int(nRepeatTrials / 4)
        + [[1, -1]] * int(nRepeatTrials / 4)
        + [[-1, -1]] * int(nRepeatTrials / 4)
        + [[-1, 1]] * int(nRepeatTrials / 4)
    )
    stimuli[:] = [stimuli[i] for i in order]

    # stimuli[:] = [[1, 1]] * nTotalTrials

    # Determine cue-stimulus intervals
    CSI = (
        [1200] * int(nSwitchTrials / 8)
        + [1200] * int(nSwitchTrials / 8)
        + [1200] * int(nSwitchTrials / 8)
        + [1200] * int(nSwitchTrials / 8)
        + [1200] * int(nSwitchTrials / 8)
        + [1200] * int(nSwitchTrials / 8)
        + [1200] * int(nSwitchTrials / 8)
        + [1200] * int(nSwitchTrials / 8)
        + [1200] * int(nRepeatTrials / 8)
        + [1200] * int(nRepeatTrials / 8)
        + [1200] * int(nRepeatTrials / 8)
        + [1200] * int(nRepeatTrials / 8)
        + [1200] * int(nRepeatTrials / 8)
        + [1200] * int(nRepeatTrials / 8)
        + [1200] * int(nRepeatTrials / 8)
        + [1200] * int(nRepeatTrials / 8)
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
    correctResponse = np.sum(np.multiply(tasks, stimuli), axis=1)

    # # Check whether combinations of transitions, stimuli and CSIs are counterbalanced

    # # This is used later to check whether trials are counterbalanced
    # stimuli_type = [1] * int(nSwitchTrials/4) + [2] * int(nSwitchTrials/4) + [3] * int(nSwitchTrials/4) + [4] * int(nSwitchTrials/4) + \
    #           [1] * int(nRepeatTrials/4) + [2] * int(nRepeatTrials/4) + [3] * int(nRepeatTrials/4) + [4] * int(nRepeatTrials/4)
    # stimuli_type[:] = [stimuli_type[i] for i in order]

    # Trials = pd.DataFrame({'TrialType': transitions,
    #                        'Stimuli': stimuli_type,
    #                        'CSI': CSI
    #                        }, columns= ['TrialType', 'Stimuli', 'CSI'])
    #
    # trial_counts = Trials.pivot_table(index=['TrialType', 'Stimuli', 'CSI'], aggfunc='size')
    # print (trial_counts)

    return tasks, stimuli, CSI, correctResponse


# Stability-Flexibility Model
def make_stab_flex(
    gain=3.0,
    leak=3.0,
    competition=2.0,
    lca_time_step_size=0.01,
    non_decision_time=0.2,
    automaticity=0.01,
    starting_value=0.0,
    threshold=0.1,
    ddm_noise=0.1,
    lca_noise=0.0,
    scale=0.2,
    ddm_time_step_size=0.01,
    rng_seed=None,
):

    GAIN = gain
    LEAK = leak
    COMP = competition
    AUTOMATICITY = automaticity  # Automaticity Weight

    STARTING_POINT = starting_value  # Starting Point
    THRESHOLD = threshold  # Threshold
    NOISE = ddm_noise  # Noise
    SCALE = scale  # Scales DDM inputs so threshold can be set to 1
    NON_DECISION_TIME = non_decision_time

    # Task Layer: [Color, Motion] {0, 1} Mutually Exclusive
    # Origin Node
    taskLayer = pnl.TransferMechanism(
        size=2,
        function=pnl.Linear(slope=1, intercept=0),
        output_ports=[pnl.RESULT],
        name="Task Input [I1, I2]",
    )

    # Stimulus Layer: [Color Stimulus, Motion Stimulus]
    # Origin Node
    stimulusInfo = pnl.TransferMechanism(
        size=2,
        function=pnl.Linear(slope=1, intercept=0),
        output_ports=[pnl.RESULT],
        name="Stimulus Input [S1, S2]",
    )

    # Cue-To-Stimulus Interval Layer
    # Origin Node
    cueInterval = pnl.TransferMechanism(
        size=1,
        function=pnl.Linear(slope=1, intercept=0),
        output_ports=[pnl.RESULT],
        name="Cue-Stimulus Interval",
    )

    # Correct Response Info
    # Origin Node
    correctResponseInfo = pnl.TransferMechanism(
        size=1,
        function=pnl.Linear(slope=1, intercept=0),
        output_ports=[pnl.RESULT],
        name="Correct Response Info",
    )

    # Control Module Layer: [Color Activation, Motion Activation]
    controlModule = pnl.LCAMechanism(
        size=2,
        function=pnl.Logistic(gain=GAIN),
        leak=LEAK,
        competition=COMP,
        self_excitation=0,
        noise=lca_noise,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=1200,
        time_step_size=lca_time_step_size,
        name="Task Activations [Act1, Act2]",
    )

    # Control Mechanism Setting Cue-To-Stimulus Interval
    csiController = pnl.ControlMechanism(
        monitor_for_control=cueInterval,
        control_signals=[(pnl.TERMINATION_THRESHOLD, controlModule)],
        modulation=pnl.OVERRIDE,
    )

    # Hadamard product of controlModule and Stimulus Information
    nonAutomaticComponent = pnl.TransferMechanism(
        size=2,
        function=pnl.Linear(slope=1, intercept=0),
        input_ports=pnl.InputPort(combine=pnl.PRODUCT),
        output_ports=[pnl.RESULT],
        name="Non-Automatic Component [S1*Act1, S2*Act2]",
    )

    # Multiply Stimulus Input by the automaticity weight
    congruenceWeighting = pnl.TransferMechanism(
        size=2,
        function=pnl.Linear(slope=AUTOMATICITY, intercept=0),
        output_ports=[pnl.RESULT],
        name="Automaticity-weighted Stimulus Input [w*S1, w*S2]",
    )

    # Summation of nonAutomatic and Automatic Components
    ddmCombination = pnl.TransferMechanism(
        size=1,
        function=pnl.Linear(slope=1, intercept=0),
        input_ports=pnl.InputPort(combine=pnl.SUM),
        output_ports=[pnl.RESULT],
        name="Drift = (w*S1 + w*S2) + (S1*Act1 + S2*Act2)",
    )

    # Ensure upper boundary of DDM is always correct response by multiplying DDM input by correctResponseInfo
    ddmRecodeDrift = pnl.TransferMechanism(
        size=1,
        function=pnl.Linear(slope=1, intercept=0),
        input_ports=pnl.InputPort(combine=pnl.PRODUCT),
        output_ports=[pnl.RESULT],
        name="Recoded Drift = Drift * correctResponseInfo",
    )

    # Scale DDM inputs
    ddmInputScale = pnl.TransferMechanism(
        size=1,
        function=pnl.Linear(slope=SCALE, intercept=0),
        output_ports=[pnl.RESULT],
        name="Scaled DDM Input",
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

    taskLayer.set_log_conditions([pnl.RESULT])
    stimulusInfo.set_log_conditions([pnl.RESULT])
    cueInterval.set_log_conditions([pnl.RESULT])
    correctResponseInfo.set_log_conditions([pnl.RESULT])
    controlModule.set_log_conditions([pnl.RESULT, "termination_threshold"])
    nonAutomaticComponent.set_log_conditions([pnl.RESULT])
    congruenceWeighting.set_log_conditions([pnl.RESULT])
    ddmCombination.set_log_conditions([pnl.RESULT])
    ddmRecodeDrift.set_log_conditions([pnl.RESULT])
    ddmInputScale.set_log_conditions([pnl.RESULT])
    decisionMaker.set_log_conditions([pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME])

    # Composition Creation
    stabilityFlexibility = pnl.Composition()

    # Node Creation
    stabilityFlexibility.add_node(taskLayer)
    stabilityFlexibility.add_node(stimulusInfo)
    stabilityFlexibility.add_node(cueInterval)
    stabilityFlexibility.add_node(correctResponseInfo)
    stabilityFlexibility.add_node(controlModule)
    stabilityFlexibility.add_node(csiController)
    stabilityFlexibility.add_node(nonAutomaticComponent)
    stabilityFlexibility.add_node(congruenceWeighting)
    stabilityFlexibility.add_node(ddmCombination)
    stabilityFlexibility.add_node(ddmRecodeDrift)
    stabilityFlexibility.add_node(ddmInputScale)
    stabilityFlexibility.add_node(decisionMaker)

    # Projection Creation
    stabilityFlexibility.add_projection(sender=taskLayer, receiver=controlModule)
    stabilityFlexibility.add_projection(
        sender=controlModule, receiver=nonAutomaticComponent
    )
    stabilityFlexibility.add_projection(
        sender=stimulusInfo, receiver=nonAutomaticComponent
    )
    stabilityFlexibility.add_projection(
        sender=stimulusInfo, receiver=congruenceWeighting
    )
    stabilityFlexibility.add_projection(
        sender=nonAutomaticComponent, receiver=ddmCombination
    )
    stabilityFlexibility.add_projection(
        sender=congruenceWeighting, receiver=ddmCombination
    )
    stabilityFlexibility.add_projection(sender=ddmCombination, receiver=ddmRecodeDrift)
    stabilityFlexibility.add_projection(
        sender=correctResponseInfo, receiver=ddmRecodeDrift
    )
    stabilityFlexibility.add_projection(sender=ddmRecodeDrift, receiver=ddmInputScale)
    stabilityFlexibility.add_projection(sender=ddmInputScale, receiver=decisionMaker)

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
    gain=3.0,
    leak=3.0,
    competition=2.0,
    lca_time_step_size=0.01,
    non_decision_time=0.2,
    automaticity=0.01,
    starting_value=0.0,
    threshold=0.1,
    ddm_noise=0.1,
    lca_noise=0.0,
    short_csi=None,
    delta_csi=None,
    scale=0.2,
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
        automaticity=automaticity,
        starting_value=starting_value,
        threshold=threshold,
        ddm_noise=ddm_noise,
        lca_noise=lca_noise,
        scale=scale,
        ddm_time_step_size=ddm_time_step_size,
        rng_seed=rng_seed,
    )

    taskLayer = stabilityFlexibility.nodes["Task Input [I1, I2]"]
    stimulusInfo = stabilityFlexibility.nodes["Stimulus Input [S1, S2]"]
    cueInterval = stabilityFlexibility.nodes["Cue-Stimulus Interval"]

    inputs = {
        taskLayer: taskTrain,
        stimulusInfo: stimulusTrain,
        cueInterval: csi_params,
    }

    stabilityFlexibility.run(inputs)

    return stabilityFlexibility


# taskLayer.log.print_entries()
# stimulusInfo.log.print_entries()
# cueInterval.log.print_entries()
# correctResponseInfo.log.print_entries()
# controlModule.log.print_entries()
# nonAutomaticComponent.log.print_entries()
# congruenceWeighting.log.print_entries()
# ddmCombination.log.print_entries()
# ddmRecodeDrift.log.print_entries()
# ddmInputScale.log.print_entries()
# decisionMaker.log.print_entries()

if __name__ == "__main__":

    taskTrain, stimulusTrain, cueTrain, switch = generate_trial_sequence(240, 0.5)
    taskTrain = taskTrain[0:3]
    stimulusTrain = stimulusTrain[0:3]
    cueTrain = cueTrain[0:3]

    comp = run_stab_flex(
        taskTrain,
        stimulusTrain,
        cueTrain,
        ddm_time_step_size=0.01,
        lca_time_step_size=0.01,
    )
    print(comp.results)
