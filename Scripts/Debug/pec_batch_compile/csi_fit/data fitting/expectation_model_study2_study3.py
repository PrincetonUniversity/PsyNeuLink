import psyneulink as pnl
import numpy as np

# Define function to generate a counterbalanced trial sequence with a specified switch trial frequency
def generate_mixed_task_sequence(n: int = 512, switch_frequency: float = 0.5, incongruence_frequency: float = 0.5, seed: int = None):

    # Compute trial number
    nTotalTrials = n
    switchFrequency = switch_frequency
    incongruenceFrequency = incongruence_frequency

    nSwitchTrials = int(nTotalTrials * switchFrequency)
    nRepeatTrials = int(nTotalTrials - nSwitchTrials)

    # Determine task transitions
    transitions = [1] * nSwitchTrials + [0] * nRepeatTrials
    rng = np.random.RandomState(seed)
    order = rng.permutation(list(range(nTotalTrials)))
    transitions[:] = [transitions[i] for i in order]

    # Determine stimuli with 50% congruent trials
    stimuli = (
            [[1, 0, 1, 0]] * int(nSwitchTrials * ((1 - incongruenceFrequency) / 2))
            + [[1, 0, 0, 1]] * int(nSwitchTrials * (incongruenceFrequency / 2))
            + [[0, 1, 0, 1]] * int(nSwitchTrials * ((1 - incongruenceFrequency) / 2))
            + [[0, 1, 1, 0]] * int(nSwitchTrials * (incongruenceFrequency / 2))
            + [[1, 0, 1, 0]] * int(nRepeatTrials * ((1 - incongruenceFrequency) / 2))
            + [[1, 0, 0, 1]] * int(nRepeatTrials * (incongruenceFrequency / 2))
            + [[0, 1, 0, 1]] * int(nRepeatTrials * ((1 - incongruenceFrequency) / 2))
            + [[0, 1, 1, 0]] * int(nRepeatTrials * (incongruenceFrequency / 2))
    )
    stimuli[:] = [stimuli[i] for i in order]

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
    correctResponse = [[np.array(correctResponse[i])] for i in range(n)]

    stimuli_tasks = np.concatenate((stimuli, tasks), axis=1)
    stimuli_tasks_correctResponse = np.concatenate((stimuli_tasks, correctResponse), axis=1)

    return tasks, stimuli, correctResponse, stimuli_tasks_correctResponse


# Stability-Flexibility Model
def make_stab_flex(
    gain=10.0,
    leak=7.0,
    competition=3.0,
    lca_time_step_size=0.01,
    non_decision_time=0.3,
    starting_value=0.0,
    threshold=0.06,
    threshold_collapse=0.0,
    ddm_noise=0.1,
    lca_noise=0.0,
    iti=0,
    csi_repeat=0,
    csi_switch=0,
    ddm_time_step_size=0.01,
    rng_seed=None,
):

    GAIN = gain
    LEAK = leak
    COMP = competition
    STARTING_POINT = starting_value  # Starting Point
    THRESHOLD = threshold  # Threshold
    THRESHOLD_COLLAPSE = threshold_collapse # Increment per timestep with which threshold collapses
    NOISE = ddm_noise  # Noise
    NON_DECISION_TIME = non_decision_time
    ITI = iti   # Inter-trial interval in number of LCA executions, during which controlExecution decays
    CSI_REPEAT = csi_repeat   # Repeat trial surrogate CSI in number of LCA executions, during which controlExecution processes the task cue
    CSI_SWITCH = csi_switch # Additional switch trial surrogate CSI, which gets summed with the repeat trial CSI

    # Origin Node [S1, S2, S3, S4]
    stimulusInput = pnl.ProcessingMechanism(
        name="Stimulus Input",
        input_shapes=4,
        function=pnl.Linear(intercept=0, slope=1)
    )

    # Origin Node [T1, T2]
    # Must be a TransferMechanism so we can reset it at the start of each trial
    # That way its output port will be 0 until it executes for the first time on a trial
    # This allows the control units to process an input of 0 during the ITI
    taskInput = pnl.TransferMechanism(
        name="Task Input",
        input_shapes=2,
        function=pnl.Linear(intercept=0, slope=1),
        integrator_mode=True,
        integration_rate=1,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=0,
        execute_until_finished=False,
        reset_stateful_function_when=pnl.AtTrialStart()
    )

    # Origin Node [CSI]
    # This mechanism represents the duration of the CSI per trial, with its intercept representing repeat trial CSI,
    # and intercept + slope representing the switch trial CSI. We use a control mechanism to override
    # controlExecution's termination_threshold parameter (which actually implements the CSI) with cueStimulusInterval's value
    cueStimulusInterval = pnl.ProcessingMechanism(
        name="Cue Stimulus Interval",
        input_shapes=1,
        function=pnl.Linear(intercept=CSI_REPEAT, slope=CSI_SWITCH)
    )

    # Origin Node [correctResponse]
    correctResponse = pnl.ProcessingMechanism(
        name="Correct Response",
        input_shapes=1
    )

    # Control Units: [Task 1 Activation, Task 2 Activation]
    # Note that execute_until_finished needs to be False so it always executes once per pass.
    # The termination_threshold parameter of controlExecution indicates the pass on which the
    # surrogate CSI has finished.
    # Note that the first X passes correspond to the ITI, and the next Y passes correspond to the CSI.
    # So if the ITI is 1 s (100 executions) and the CSI is 0.5 s (50 executions), then the CSI is over on pass 150
    controlExecution = pnl.LCAMechanism(
        name="Task Activations [C1, C2]",
        input_shapes=2,
        function=pnl.Logistic(gain=GAIN),
        leak=LEAK,
        competition=COMP,
        self_excitation=0.0,
        noise=lca_noise,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=1,
        time_step_size=lca_time_step_size,
        execute_until_finished=False,
    )

    # Control mechanism that adjusts controlExecutions's termination_threshold parameter to implement the CSI
    # Adding ITI to the intercept here ensures that the termination_threshold of controlExecution is equal
    # to the length of the ITI plus the length of the CSI.
    csiOverride = pnl.ControlMechanism(
        name="CSI Override",
        function=pnl.Linear(intercept=ITI, slope=1),
        monitor_for_control=cueStimulusInterval,
        control_signals=[("termination_threshold", controlExecution)],
        modulation=pnl.OVERRIDE
    )

    # A custom function that integrates stimulus inputs, control inputs, and correct response info into a drift rate
    # In compiled mode, we must use this function that does not rely on non-local variables and is difficult to read
    def drift_rate_fct(input):
        return ((1 / (1 + np.exp(-(1 * (1 / (1 + np.exp(-(1 * (input[0][0] - input[0][1]) + 4 * input[0][4] - 4)))) - 1 * (1 / (1 + np.exp(-(1 * (input[0][1] - input[0][0]) + 4 * input[0][4] - 4)))) + 1 * (1 / (1 + np.exp(-(1 * (input[0][2] - input[0][3]) + 4 * input[0][5] - 4)))) - 1 * (1 / (1 + np.exp(-(1 * (input[0][3] - input[0][2]) + 4 * input[0][5] - 4)))))))) - (1 / (1 + np.exp(-(1 * -(1 / (1 + np.exp(-(1 * (input[0][0] - input[0][1]) + 4 * input[0][4] - 4)))) + 1 * (1 / (1 + np.exp(-(1 * (input[0][1] - input[0][0]) + 4 * input[0][4] - 4)))) - 1 * (1 / (1 + np.exp(-(1 * (input[0][2] - input[0][3]) + 4 * input[0][5] - 4)))) + 1 * (1 / (1 + np.exp(-(1 * (input[0][3] - input[0][2]) + 4 * input[0][5] - 4))))))))) * input[0][6]

    driftRate = pnl.ProcessingMechanism(name="Drift Rate Value",
                                        function=drift_rate_fct,
                                        input_ports=[{pnl.NAME: 'Drift Rate Input Port', pnl.INPUT_SHAPES: 7}])

    # Decision Module
    decisionMaker = pnl.DDM(
        name="DDM",
        function=pnl.DriftDiffusionIntegrator(
            starting_value=STARTING_POINT,
            threshold=THRESHOLD,
            noise=NOISE,
            time_step_size=ddm_time_step_size,
            non_decision_time=NON_DECISION_TIME,
        ),
        reset_stateful_function_when=pnl.AtTrialStart(),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        execute_until_finished=False
    )

    # Mechanism that represents within-trial changes in threshold
    thresholdMechanism = pnl.TransferMechanism(
        name="Threshold Mechanism",
        input_shapes=1,
        default_variable=0,
        integrator_function=pnl.SimpleIntegrator(rate=1, offset=THRESHOLD_COLLAPSE),
        function=pnl.Linear(intercept=THRESHOLD, slope=1),
        integrator_mode=True,
        termination_measure=pnl.TimeScale.TRIAL,
        execute_until_finished=False,
        reset_stateful_function_when=pnl.AtTrialStart()
    )

    # Control mechanism to modulate threshold per execution of the DDM
    thresholdOverride = pnl.ControlMechanism(
        monitor_for_control=thresholdMechanism,
        control_signals=[(pnl.THRESHOLD, decisionMaker)],
        modulation=pnl.OVERRIDE
    )

    # Set logging conditions
    taskInput.set_log_conditions([pnl.RESULT])
    stimulusInput.set_log_conditions(["value"])
    controlExecution.set_log_conditions([pnl.RESULT])
    decisionMaker.set_log_conditions(["value"])
    thresholdMechanism.set_log_conditions(["value"])

    # Composition Creation
    stabilityFlexibility = pnl.Composition()

    # Node Creation
    stabilityFlexibility.add_node(stimulusInput)
    stabilityFlexibility.add_node(taskInput)
    stabilityFlexibility.add_node(cueStimulusInterval)
    stabilityFlexibility.add_node(csiOverride)
    stabilityFlexibility.add_node(correctResponse)
    stabilityFlexibility.add_node(controlExecution)
    stabilityFlexibility.add_node(driftRate)
    stabilityFlexibility.add_node(thresholdMechanism)
    stabilityFlexibility.add_node(thresholdOverride)
    stabilityFlexibility.add_node(decisionMaker)

    # Projection Creation
    stabilityFlexibility.add_projection(sender=stimulusInput, receiver=driftRate,
                                        projection=pnl.MappingProjection(matrix=np.array([[1, 0, 0, 0, 0, 0, 0],
                                                                                          [0, 1, 0, 0, 0, 0, 0],
                                                                                          [0, 0, 1, 0, 0, 0, 0],
                                                                                          [0, 0, 0, 1, 0, 0, 0]]))
                                        )

    stabilityFlexibility.add_projection(sender=taskInput, receiver=controlExecution,
                                        projection=pnl.MappingProjection(matrix=np.array([[1, 0],
                                                                                          [0, 1]]))
                                        )

    stabilityFlexibility.add_projection(sender=correctResponse, receiver=driftRate,
                                        projection=pnl.MappingProjection(matrix=np.array([[0, 0, 0, 0, 0, 0, 1]]))
                                        )

    stabilityFlexibility.add_projection(sender=controlExecution, receiver=driftRate,
                                        projection=pnl.MappingProjection(matrix=np.array([[0, 0, 0, 0, 1, 0, 0,],
                                                                                          [0, 0, 0, 0, 0, 1, 0,]]))
                                        )

    stabilityFlexibility.add_projection(sender=driftRate, receiver=decisionMaker)

    # We add several scheduling conditions to ensure things execute at the appropriate moment
    # and not more than necessary

    # The mechanisms related to CSI, correct response, and stimulus input only need to execute on the first pass
    stabilityFlexibility.scheduler.add_condition(
        cueStimulusInterval, pnl.AtPass(0)
    )

    stabilityFlexibility.scheduler.add_condition(
        csiOverride, pnl.AtPass(0)
    )

    stabilityFlexibility.scheduler.add_condition(
        stimulusInput, pnl.AtPass(0)
    )

    stabilityFlexibility.scheduler.add_condition(
        correctResponse, pnl.AtPass(0)
    )

    # We schedule taskInput to only execute on the first pass after the ITI.
    # If ITI is zero, it executes on the first pass (i.e., pass zero). If ITI is 1 s, it executes on pass 100.
    stabilityFlexibility.scheduler.add_condition(
        taskInput, pnl.AtPass(ITI)
    )

    # We schedule controlExecution to always execute, this means controlExecution integrates an input of 0
    # until the pass where taskInput executes (ITI decay), and then starts integrating input from taskInput
    stabilityFlexibility.scheduler.add_condition(
        controlExecution, pnl.Always()
    )

    # # We schedule driftRate, thresholdMechanism, thresholdOverride, and decisionMaker, to start once
    # the ITI and CSI are both over, which is indicated by the controlExecutions's is_finished flag.
    stabilityFlexibility.scheduler.add_condition(
        driftRate, pnl.WhenFinished(controlExecution)
    )

    stabilityFlexibility.scheduler.add_condition(
        thresholdMechanism, pnl.WhenFinished(controlExecution)
    )

    stabilityFlexibility.scheduler.add_condition(
        thresholdOverride, pnl.WhenFinished(controlExecution)
    )

    stabilityFlexibility.scheduler.add_condition(
        decisionMaker, pnl.WhenFinished(controlExecution)
    )

    # To prevent the composition from finishing after one DDM execution,
    # we need two gates that only execute when DDM is finished
    decisionGate = pnl.ProcessingMechanism(input_shapes=1, name="DECISION_GATE")
    stabilityFlexibility.add_node(decisionGate)

    responseGate = pnl.ProcessingMechanism(input_shapes=1, name="RESPONSE_GATE")
    stabilityFlexibility.add_node(responseGate)

    stabilityFlexibility.add_projection(
        sender=decisionMaker.output_ports[0], receiver=decisionGate
    )
    stabilityFlexibility.add_projection(
        sender=decisionMaker.output_ports[1], receiver=responseGate
    )

    # Sets scheduler conditions, so the gates do not execute (and composition doesn't finish) until DDM is finished
    stabilityFlexibility.scheduler.add_condition(
        decisionGate, pnl.WhenFinished(decisionMaker)
    )
    stabilityFlexibility.scheduler.add_condition(
        responseGate, pnl.WhenFinished(decisionMaker)
    )

    # Lastly, so that the surrogate CSI is added to the overall response time of the composition, we add a projection
    # from cueStimulusInterval to responseGate (which represents RT), making sure to scale the former's value to RT.
    stabilityFlexibility.add_projection(sender=cueStimulusInterval, receiver=responseGate,
                                        projection=pnl.MappingProjection(matrix=np.array([ddm_time_step_size]))
                                        )

    return stabilityFlexibility


def run_stab_flex(
    all_inputs,
    gain=10.0,
    leak=7.0,
    competition=3.0,
    lca_time_step_size=0.01,
    non_decision_time=0.3,
    starting_value=0.0,
    threshold=0.06,
    threshold_collapse=-0.001,
    ddm_noise=0.1,
    lca_noise=0.0,
    iti=0,
    csi_repeat=0,
    csi_switch=0,
    ddm_time_step_size=0.01,
    rng_seed=None,
):

    stabilityFlexibility = make_stab_flex(
        gain=gain,
        leak=leak,
        competition=competition,
        lca_time_step_size=lca_time_step_size,
        non_decision_time=non_decision_time,
        starting_value=starting_value,
        threshold=threshold,
        threshold_collapse=threshold_collapse,
        ddm_noise=ddm_noise,
        lca_noise=lca_noise,
        ddm_time_step_size=ddm_time_step_size,
        iti=iti,
        csi_repeat=csi_repeat,
        csi_switch=csi_switch,
        rng_seed=rng_seed,
    )

    cueStimulusInterval = stabilityFlexibility.nodes["Cue Stimulus Interval"]
    stimulusInput = stabilityFlexibility.nodes["Stimulus Input"]
    taskInput = stabilityFlexibility.nodes["Task Input"]
    correctResponse = stabilityFlexibility.nodes["Correct Response"]
    controlExecution = stabilityFlexibility.nodes["Task Activations [C1, C2]"]
    decisionMaker = stabilityFlexibility.nodes["DDM"]
    thresholdMechanism = stabilityFlexibility.nodes["Threshold Mechanism"]

    inputs = {
        stimulusInput: stimulusSequence,
        taskInput: taskSequence,
        correctResponse: correctResponseSequence,
        cueStimulusInterval: csiSequence,
    }

    # stabilityFlexibility.run(inputs)
    stabilityFlexibility.run(inputs, execution_mode=pnl.ExecutionMode.LLVMRun)

    # controlExecution.log.print_entries()
    # taskInput.log.print_entries()
    # stimulusInput.log.print_entries()
    # decisionMaker.log.print_entries()
    # thresholdMechanism.log.print_entries()


    return stabilityFlexibility


if __name__ == "__main__":
    from psyneulink.core.globals.utilities import set_global_seed
    set_global_seed(0)

    num_trials = 512

    taskSequence, stimulusSequence, correctResponseSequence, combinedSequence = generate_mixed_task_sequence(num_trials, 0.5, 0.5, seed=0)

    itiSequence = [0 for i in taskSequence]
    csiSequence = [0 if taskSequence[i] == taskSequence[i-1] else 1 for i in range(num_trials)]

    stimulusSequence = stimulusSequence[0:4]
    taskSequence = taskSequence[0:4]
    correctResponseSequence = correctResponseSequence[0:4]
    csiSequence = csiSequence[0:4]

    inputs = {
        "stimulusInput": stimulusSequence,
        "taskInput": taskSequence,
        "correctResponse": correctResponseSequence,
        "cueStimulusInterval": csiSequence
    }

    comp = run_stab_flex(inputs, iti=10, csi_repeat=10, csi_switch=10, threshold_collapse=-0.001)
    print(comp.results)