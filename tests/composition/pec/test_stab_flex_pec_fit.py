import psyneulink as pnl

import optuna

from psyneulink.core.components.functions.nonstateful.fitfunctions import (
    PECOptimizationFunction,
)

import numpy as np
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
        input_shapes=2,
        function=pnl.Linear(slope=1, intercept=0),
        output_ports=[pnl.RESULT],
        name="Task Input [I1, I2]",
    )

    # Stimulus Layer: [Color Stimulus, Motion Stimulus]
    # Origin Node
    stimulusInfo = pnl.TransferMechanism(
        input_shapes=2,
        function=pnl.Linear(slope=1, intercept=0),
        output_ports=[pnl.RESULT],
        name="Stimulus Input [S1, S2]",
    )

    # Cue-To-Stimulus Interval Layer
    # Origin Node
    cueInterval = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Linear(slope=1, intercept=0),
        output_ports=[pnl.RESULT],
        name="Cue-Stimulus Interval",
    )

    # Correct Response Info
    # Origin Node
    correctResponseInfo = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Linear(slope=1, intercept=0),
        output_ports=[pnl.RESULT],
        name="Correct Response Info",
    )

    # Control Module Layer: [Color Activation, Motion Activation]
    controlModule = pnl.LCAMechanism(
        input_shapes=2,
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
        input_shapes=2,
        function=pnl.Linear(slope=1, intercept=0),
        input_ports=pnl.InputPort(combine=pnl.PRODUCT),
        output_ports=[pnl.RESULT],
        name="Non-Automatic Component [S1*Act1, S2*Act2]",
    )

    # Multiply Stimulus Input by the automaticity weight
    congruenceWeighting = pnl.TransferMechanism(
        input_shapes=2,
        function=pnl.Linear(slope=AUTOMATICITY, intercept=0),
        output_ports=[pnl.RESULT],
        name="Automaticity-weighted Stimulus Input [w*S1, w*S2]",
    )

    # Summation of nonAutomatic and Automatic Components
    ddmCombination = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Linear(slope=1, intercept=0),
        input_ports=pnl.InputPort(combine=pnl.SUM),
        output_ports=[pnl.RESULT],
        name="Drift = (w*S1 + w*S2) + (S1*Act1 + S2*Act2)",
    )

    # Ensure upper boundary of DDM is always correct response by multiplying DDM input by correctResponseInfo
    ddmRecodeDrift = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Linear(slope=1, intercept=0),
        input_ports=pnl.InputPort(combine=pnl.PRODUCT),
        output_ports=[pnl.RESULT],
        name="Recoded Drift = Drift * correctResponseInfo",
    )

    # Scale DDM inputs
    ddmInputScale = pnl.TransferMechanism(
        input_shapes=1,
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

    # Sets scheduler conditions, so that the gates are not executed (and hence the composition doesn't finish) until decisionMaker is finished
    stabilityFlexibility.scheduler.add_condition(
        decisionGate, pnl.WhenFinished(decisionMaker)
    )
    stabilityFlexibility.scheduler.add_condition(
        responseGate, pnl.WhenFinished(decisionMaker)
    )

    return stabilityFlexibility

def get_node(comp, name):
    """
    Get the node from the composition with the given name. The name needs to match from the beginning, but it
    can have any numeric suffix after the name.
    """
    for node in comp.nodes:
        if node.name.startswith(name):
            return node
    return None


def make_input_dict(stab_flex_comp, taskTrain, stimulusTrain, cueTrain, correctResponse):
    inputs = {
        get_node(stab_flex_comp, "Task Input [I1, I2]"): [[np.array(v)] for v in taskTrain],
        get_node(stab_flex_comp, "Stimulus Input [S1, S2]"): [[np.array(v)] for v in stimulusTrain],
        get_node(stab_flex_comp, "Cue-Stimulus Interval"): [[np.array(v)] for v in cueTrain],
        get_node(stab_flex_comp, "Correct Response Info"): [[np.array(v)] for v in correctResponse]
    }

    return inputs

def run_stab_flex_cond(
        taskTrain,
        stimulusTrain,
        cueTrain,
        correctResponse,
        num_trials,
        **kwargs):
    """
    Create a stability flexibility composition and run it with the given parameters. Return the composition and the
    results as a pandas DataFrame. If any of the parameters are a list, then that parameter is assumed to be trial-wise
    and the length of the list should be the number of trials. A control mechanism will be added to the composition to
    override the parameter with the value from the input.
    """

    # Remove any parameters that are trial-wise from the kwargs, these values will be passed in as inputs
    # to the composition.
    cond_params = {name: value for name, value in kwargs.items()
                   if isinstance(value, list) or isinstance(value, np.ndarray)}

    # Remove the trial-wise parameters from the kwargs
    kwargs = {name: value for name, value in kwargs.items() if name not in cond_params}

    # Make a stability flexibility composition
    comp = make_stab_flex(**kwargs)

    inputs = make_input_dict(comp, taskTrain, stimulusTrain, cueTrain, correctResponse)

    # A dict to map keyword arg name to the corresponding mechanism in the composition
    param_map = {
        "gain": ("gain", comp.nodes["Task Activations [Act1, Act2]"]), # Gain
        "automaticity": ("slope", comp.nodes["Automaticity-weighted Stimulus Input [w*S1, w*S2]"]), # Automaticity
        "threshold": ("threshold", comp.nodes["DDM"]), # Threshold
        "non_decision_time": ("non_decision_time", comp.nodes["DECISION_GATE"]), # Non-decision time
    }
    # Go through the parameters and check if any are trial-wise, if so, add a control mechanism to override the value on
    # trial-by-trial basis with the value from the input.
    pec_mechs = {}
    for (name, value) in cond_params.items():

        if len(value) != num_trials:
            raise ValueError("Length of trial-wise parameter must be equal to the number of trials.")

        pec_mechs[name] = pnl.ControlMechanism(name=f"{name}_control",
                                               control_signals=param_map[name],
                                               modulation=pnl.OVERRIDE)
        comp.add_node(pec_mechs[name])
        inputs[pec_mechs[name]] = [[np.array([value[i]])] for i in range(num_trials)]

    comp.run(inputs, execution_mode=pnl.ExecutionMode.LLVMRun)

    df = pd.DataFrame(
        np.squeeze(np.array(comp.results))[:, 1:], columns=["decision", "response_time"]
    )
    df["decision"] = df["decision"].astype("category")

    # Add the trial-wise parameters to the DataFrame as well.
    for name in pec_mechs.keys():
        df[name] = cond_params[name]

    assert len(comp.input_ports) > 0

    return comp, df

def test_stab_flex_cond_fit():
    from psyneulink.core.globals.utilities import set_global_seed

    # # Let's make things reproducible
    pnl_seed = 42
    set_global_seed(pnl_seed)
    trial_seq_seed = 43

    # High-level parameters the impact performance of the test
    num_trials = 75
    time_step_size = 0.01
    num_estimates = 100

    sf_params = dict(
        gain=3.0,
        leak=3.0,
        competition=2.0,
        lca_time_step_size=time_step_size,
        non_decision_time=0.2,
        automaticity=0.01,
        starting_value=0.0,
        threshold=0.1,
        ddm_noise=0.1,
        lca_noise=0.0,
        scale=0.2,
        ddm_time_step_size=time_step_size,
    )

    # Generate some sample data to run the model on
    taskTrain, stimulusTrain, cueTrain, correctResponse = generate_trial_sequence(240, 0.5, seed=trial_seq_seed)
    taskTrain = taskTrain[0:num_trials]
    stimulusTrain = stimulusTrain[0:num_trials]
    cueTrain = cueTrain[0:num_trials]
    correctResponse = correctResponse[0:num_trials]

    # CSI is in terms of time steps, we need to scale by ten because original code
    # was set to run with timestep size of 0.001
    cueTrain = [c / 10.0 for c in cueTrain]

    # We will generate a dataset that comprises two different conditions. Each condition will have a different threshold.
    # Randomly select which trials will be in each condition uniformly.
    rng = np.random.default_rng(12345)
    threshold = rng.choice([0.3, 0.7], size=num_trials, replace=True)

    # Run
    _, data_to_fit = run_stab_flex_cond(
        taskTrain,
        stimulusTrain,
        cueTrain,
        correctResponse,
        num_trials,
        **{**sf_params, 'threshold': threshold}
    )

    # Turn our trial-wise threshold into a condition
    data_to_fit['condition'] = np.where(data_to_fit['threshold'] == 0.3, 'threshold=0.3', 'threshold=0.7')
    data_to_fit.drop(columns=['threshold'], inplace=True)

    # %%
    # Create a parameter estimation composition to fit the data we just generated and hopefully recover the
    # parameters of the composition.
    comp = make_stab_flex(**sf_params)

    controlModule = get_node(comp, "Task Activations [Act1, Act2]")
    congruenceWeighting = get_node(comp, "Automaticity-weighted Stimulus Input [w*S1, w*S2]")
    decisionMaker = get_node(comp, "DDM")
    decisionGate = get_node(comp, "DECISION_GATE")
    responseGate = get_node(comp, "RESPONSE_GATE")

    fit_parameters = {
        ("gain", controlModule): np.linspace(1.0, 10.0, 1000),  # Gain
        ("slope", congruenceWeighting): np.linspace(0.0, 0.1, 1000),  # Automaticity
        ("threshold", decisionMaker): np.linspace(0.01, 0.5, 1000),  # Threshold
        ("non_decision_time", decisionMaker): np.linspace(0.1, 0.4, 1000),  # Threshold
    }

    pec = pnl.ParameterEstimationComposition(
        name="pec",
        nodes=comp,
        parameters=fit_parameters,
        depends_on={("threshold", decisionMaker): 'condition'},
        outcome_variables=[
            decisionGate.output_ports[0],
            responseGate.output_ports[0],
        ],
        data=data_to_fit,
        optimization_function=PECOptimizationFunction(
            method=optuna.samplers.RandomSampler, max_iterations=10
        ),
        num_estimates=num_estimates,
        initial_seed=42,
    )

    pec.controller.parameters.comp_execution_mode.set("LLVM")
    pec.controller.function.parameters.save_values.set(True)

    inputs = make_input_dict(comp, taskTrain, stimulusTrain, cueTrain, correctResponse)

    pec.run(inputs=inputs)
    optimal_parameters = pec.optimized_parameter_values

    # These aren't the recovered parameters, we are doing too few trials and too few estimates to get the correct
    # results.
    expected_results = {
        'Task Activations [Act1, Act2]-1.gain': 2.3965500000000004,
        'Automaticity-weighted Stimulus Input [w*S1, w*S2]-1.slope': 0.0058000000000000005,
        'DDM-1.threshold[threshold=0.7]': 0.43483,
        'DDM-1.threshold[threshold=0.3]': 0.30449,
        'DDM-1.non_decision_time': 0.3124
    }

    for key, value in expected_results.items():
        np.testing.assert_allclose(optimal_parameters[key], value, rtol=1e-6)
