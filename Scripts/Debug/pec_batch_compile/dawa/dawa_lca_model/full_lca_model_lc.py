import psyneulink as pnl
import numpy as np

#Define function to generate a counterbalanced trial sequence
def conflict_task_sequence(n: int = 512, incongruence_frequency: float = 0.5, seed: int = None):
    rng = np.random.RandomState(seed)

    # Half red, half blue
    n_red = n // 2
    n_blue = n - n_red

    # Split each into congruent/incongruent
    n_red_incon  = int(n_red * incongruence_frequency)
    n_red_con    = n_red - n_red_incon
    n_blue_incon = int(n_blue * incongruence_frequency)
    n_blue_con   = n_blue - n_blue_incon

    #[Red, Blue, Left, Right]
    red_con    = [[1, 0, 1, 0]] * n_red_con     # Red target, left location (congruent)
    red_incon  = [[1, 0, 0, 1]] * n_red_incon   # Red target, right location (incongruent)
    blue_con   = [[0, 1, 0, 1]] * n_blue_con    # Blue target, right location (congruent)
    blue_incon = [[0, 1, 1, 0]] * n_blue_incon  # Blue target, left location (incongruent)

    stimuli = red_con + red_incon + blue_con + blue_incon
    rng.shuffle(stimuli)
    stimuli = np.array(stimuli)

    # Task is always color discrimination
    tasks = np.tile([1, 0], (n, 1))

    return tasks, stimuli

def make_lca_model(
# Control LCA,
    c_bias=0.0,
    c_gain=100,
    c_leak=1,
    c_competition=1,
    c_w=4.0,
    # Stimulus LCA
    s_bias=-0.45,
    s_gain=5.0,
    s_leak=8.0,
    s_competition=8.0,
    # Decision LCA
    d_bias=-0.45,
    d_gain=5.0,
    d_leak=8.0,
    d_competition=8.0,
    # Response LCA
    r_bias=-0.45,
    r_gain=5.0,
    r_leak=8.0,
    r_competition=8.0,
    r_threshold=0.3,
    r_noise=0.0,
    d_noise=0.1,
    # Time gate
    non_decision_time=0,
    time_step_size=0.01,
    # Weights
    w1=1.0,
    w2=1.0,
    sdr_bias=0.0,
    lc_base_gain=5.0,
    lc_scaling=1.5,
    lc_mode=0.5,
    lc_input=0.25,
    lc_threshold=0.5,
    rng_seed=None,
    c_noise=0.0,
    s_noise=0.0,
):
    """Build the LC/LCA network; each ``*_noise`` is a Gaussian standard deviation.

    Control state carries across trials; stimulus, decision and response state
    reset at trial start. Noise does not change these reset conditions.
    """

    taskInput = pnl.ProcessingMechanism(
        name="Task Input",
        input_shapes=2,
    )

    stimulusInput = pnl.ProcessingMechanism(
        name="Stimulus Input",
        input_shapes=4,
    )

    WeightedColorInput = pnl.ProcessingMechanism(
        name="Weighted Color Input",
        input_shapes=2,
        function=pnl.Linear(intercept=0, slope=w1)
    )

    WeightedLocationInput = pnl.ProcessingMechanism(
        name="Weighted Location Input",
        input_shapes=2,
        function=pnl.Linear(intercept=0, slope=w2)
    )


    controlExecution = pnl.LCAMechanism(
        name="Control Units\n[Color, Location]",
        input_shapes=2,
        function=pnl.Logistic(gain=c_gain, bias=c_bias),
        leak=c_leak,
        competition=c_competition,
        self_excitation=0,
        # Keep the original numeric zero when disabled: a NormalDist, even
        # with zero variance, opts into the compiler's stochastic LCA adapter.
        noise=pnl.NormalDist(mean=0.0, standard_deviation=c_noise) if c_noise else 0,
        time_step_size=time_step_size,
        termination_measure=pnl.TimeScale.TRIAL,
        execute_until_finished=False,
        #reset_stateful_function_when=pnl.AtTrialStart(),
        termination_threshold=0,
    )

    stimulusLayer = pnl.LCAMechanism(
        name="Stimulus Units\n[Red, Blue, Left, Right]",
        input_shapes=4,
        function=pnl.Logistic(gain=s_gain, bias=s_bias),
        matrix=[[0, -s_competition, 0, 0],
                [-s_competition, 0, 0, 0],
                [0, 0, 0, -s_competition],
                [0, 0, -s_competition, 0]],
        leak=s_leak,
        competition=s_competition,
        self_excitation=0,
        noise=pnl.NormalDist(mean=0.0, standard_deviation=s_noise),
        time_step_size=time_step_size,
        termination_measure=pnl.TimeScale.TRIAL,
        execute_until_finished=False,
        termination_threshold=0,
        reset_stateful_function_when=pnl.AtTrialStart()
    )

    WeightedColorStimulus = pnl.ProcessingMechanism(
        name="Weighted Color Stimulus",
        input_shapes=2,
        function=pnl.Linear(intercept=0, slope=w1)
    )

    WeightedLocationStimulus = pnl.ProcessingMechanism(
        name="Weighted Location Stimulus",
        input_shapes=2,
        function=pnl.Linear(intercept=0, slope=w2)
    )

    decisionLayer = pnl.LCAMechanism(
        name="Decision Units\n[Left, Right]",
        input_shapes=2,
        function=pnl.Logistic(gain=d_gain, bias=d_bias),
        leak=d_leak,
        competition=d_competition,
        self_excitation=0,
        noise=pnl.NormalDist(mean=0.0, standard_deviation=d_noise),
        time_step_size=time_step_size,
        termination_measure=pnl.TimeScale.TRIAL,
        output_ports=[pnl.RESULT, pnl.ENERGY],
        execute_until_finished=False,
        termination_threshold=0,
        reset_stateful_function_when=pnl.AtTrialStart()
    )

    responseLayer = pnl.LCAMechanism(
        name="Response Units\n[Left, Right]",
        input_shapes=2,
        function=pnl.Logistic(gain=r_gain, bias=r_bias),
        termination_threshold=r_threshold,
        leak=r_leak,
        competition=r_competition,
        self_excitation=0,
        noise=pnl.NormalDist(mean=0.0, standard_deviation=r_noise),
        time_step_size=time_step_size,
        execute_until_finished=False,
        reset_stateful_function_when=pnl.AtTrialStart(),
        output_ports=[pnl.RESULT, pnl.DECISION_TIME, pnl.DECISION_INDEX],
    )

    decisionGate = pnl.ProcessingMechanism(
        name="DECISION_GATE",
        input_shapes=1,
    )

    timeGate = pnl.ProcessingMechanism(
        name="RT_GATE",
        input_shapes=1,
        function=pnl.Linear(slope=1.0, intercept=non_decision_time),
    )

    biasMechanism = pnl.ProcessingMechanism(
         name="Bias Mechanism",
         input_shapes = 1,
         function = pnl.Linear(slope=0, intercept=sdr_bias),
    )

    biasOverride = pnl.ControlMechanism(
        monitor_for_control=biasMechanism,
        control_signals=[(pnl.BIAS, stimulusLayer),
                         (pnl.BIAS, decisionLayer),
                         (pnl.BIAS, responseLayer)],
        modulation = pnl.OVERRIDE
    )


    w1Mechanism = pnl.ProcessingMechanism(
        name="w1 Mechanism",
        input_shapes = 1,
        function = pnl.Linear(slope=0, intercept=w1),
    )

    w1Override = pnl.ControlMechanism(
        monitor_for_control=w1Mechanism,
        control_signals=[(pnl.SLOPE, WeightedColorInput),
                         (pnl.SLOPE, WeightedColorStimulus)],
        modulation = pnl.OVERRIDE
    )

    w2Mechanism = pnl.ProcessingMechanism(
        name="w2 Mechanism",
        input_shapes=1,
        function=pnl.Linear(slope=0, intercept=w2),
    )

    w2Override = pnl.ControlMechanism(
        monitor_for_control=w2Mechanism,
        control_signals=[(pnl.SLOPE, WeightedLocationInput),
                         (pnl.SLOPE, WeightedLocationStimulus)],
        modulation=pnl.OVERRIDE
    )


    comp = pnl.Composition()
    comp.add_nodes([taskInput, stimulusInput, controlExecution, stimulusLayer, decisionLayer, responseLayer, decisionGate, timeGate,
                    biasMechanism, biasOverride,
                    WeightedColorInput, WeightedLocationInput, WeightedColorStimulus, WeightedLocationStimulus,
                    w1Mechanism, w1Override, w2Mechanism, w2Override
    ])

    comp.add_projection(sender=taskInput, receiver=controlExecution)

    comp.add_projection(
        sender=stimulusInput,
        receiver=WeightedColorInput,
        projection=pnl.MappingProjection(
            matrix=np.array([
                [1, 0],
                [0, 1],
                [0, 0],
                [0, 0],
            ])
        ),
    )

    comp.add_projection(
        sender=stimulusInput,
        receiver=WeightedLocationInput,
        projection=pnl.MappingProjection(
            matrix=np.array([
                [0, 0],
                [0, 0],
                [1, 0],
                [0, 1],
            ])
        ),
    )


    comp.add_projection(
        sender=controlExecution,
        receiver=stimulusLayer,
        projection=pnl.MappingProjection(
            matrix=np.array([
                [c_w, c_w, 0,   0],  # Color control → Red & Blue
                [0,   0,   c_w, c_w] # Location control → Left & Right
            ])
        ),
    )

    comp.add_projection(
        sender=controlExecution,
        receiver=decisionLayer,
        projection=pnl.MappingProjection(
            matrix=np.array([
                [c_w, c_w],  # Control unit 1 → Left, Right
                [c_w, c_w],  # Control unit 2 → Left, Right
            ])
        ),
    )

    comp.add_projection(
        sender=controlExecution,
        receiver=responseLayer,
        projection=pnl.MappingProjection(
            matrix=np.array([
                [c_w, c_w],  # Control unit 1 → Left, Right
                [c_w, c_w],  # Control unit 2 → Left, Right
            ])
        ),
    )

    comp.add_projection(
        sender=WeightedColorInput,
        receiver=stimulusLayer,
        projection=pnl.MappingProjection(
            matrix=np.array([
                [1, -1, 0, 0],
                [-1, 1, 0, 0],
            ])
        ),
    )

    comp.add_projection(
        sender=WeightedLocationInput,
        receiver=stimulusLayer,
        projection=pnl.MappingProjection(
            matrix=np.array([
                [0, 0, 1, -1],
                [0, 0, -1, 1],
            ])
        ),
    )

    comp.add_projection(
        sender=stimulusLayer,
        receiver=WeightedColorStimulus,
        projection=pnl.MappingProjection(
            matrix=np.array([
                [1, 0],
                [0, 1],
                [0, 0],
                [0, 0],
            ])
        ),
    )

    comp.add_projection(
        sender=stimulusLayer,
        receiver=WeightedLocationStimulus,
        projection=pnl.MappingProjection(
            matrix=np.array([
                [0, 0],
                [0, 0],
                [1, 0],
                [0, 1],
            ])
        ),
    )

    comp.add_projection(
        sender=WeightedColorStimulus,
        receiver=decisionLayer,
        projection=pnl.MappingProjection(
            matrix=np.array([
                [1, -1],
                [-1, 1],
            ])
        ),
    )

    comp.add_projection(
        sender=WeightedLocationStimulus,
        receiver=decisionLayer,
        projection=pnl.MappingProjection(
            matrix=np.array([
                [1, -1],
                [-1, 1],
            ])
        ),
    )

    comp.add_projection(
        sender=decisionLayer,
        receiver=responseLayer,
        projection=pnl.MappingProjection(
            matrix=np.array([
                [1, -1],  # Decision Left → Response Left
                [-1,  1], # Decision Right → Response Right
            ])
        ),
    )
    # responseLayer output port order: 0=RESULT, 1=DECISION_TIME, 2=DECISION_INDEX
    comp.add_projection(sender=responseLayer.output_ports[1], receiver=timeGate)  # DECISION_TIME
    comp.add_projection(sender=responseLayer.output_ports[2], receiver=decisionGate)  # DECISION_INDEX

    comp.scheduler.add_condition(decisionGate, pnl.WhenFinished(responseLayer))
    comp.scheduler.add_condition(timeGate, pnl.WhenFinished(responseLayer))

    comp.scheduler.add_condition(biasMechanism, pnl.AtPass(0))
    comp.scheduler.add_condition(biasOverride, pnl.AtPass(0))


    comp.scheduler.add_condition(w1Mechanism, pnl.AtPass(0))
    comp.scheduler.add_condition(w1Override, pnl.AtPass(0))

    comp.scheduler.add_condition(w2Mechanism, pnl.AtPass(0))
    comp.scheduler.add_condition(w2Override, pnl.AtPass(0))


    lc_drive = pnl.ObjectiveMechanism(
        name="LC Monitor",
        monitor=[decisionLayer.output_ports[pnl.RESULT]],
        function=pnl.Linear(slope=lc_input, intercept=0.0),
    )

    lc = pnl.TransferMechanism(
        name="LC",
        input_shapes=1,
       integrator_mode=True,
       integrator_function=pnl.FitzHughNagumoIntegrator(
           integration_method="EULER",
           time_step_size=0.02,
           mode=lc_mode,
           uncorrelated_activity=0.5,
           time_constant_v=0.05,
           time_constant_w=5.0,
           a_v=-1.0, b_v=1.0, c_v=1.0,
           d_v=0.0, e_v=-1.0, f_v=1.0,
           a_w=1.0, b_w=-1.0, c_w=0.0,
           threshold=lc_threshold,
       ),
       termination_measure=pnl.TimeScale.PASS,
       termination_threshold=10,
       execute_until_finished=True,
       output_ports=[{
           pnl.NAME: "NE_OUTPUT",
           pnl.VARIABLE: (pnl.OWNER_VALUE, 1)
       }],
        function=pnl.Linear(intercept=lc_base_gain, slope=lc_scaling),
        reset_stateful_function_when=pnl.AtTrialStart(),
    )

    comp.add_nodes([lc_drive, lc])
    comp.add_projection(sender=lc_drive, receiver=lc)

    lc_control = pnl.ControlMechanism(
        name="LC Control",
        monitor_for_control=lc,
        control_signals=[(pnl.GAIN, stimulusLayer),
                         (pnl.GAIN, decisionLayer),
                         (pnl.GAIN, responseLayer)],
        modulation=pnl.OVERRIDE,
    )



    comp.add_nodes([lc_control])

    comp.scheduler.add_condition(lc_drive, pnl.Always())
    comp.scheduler.add_condition(lc, pnl.Always())
    comp.scheduler.add_condition(lc_control, pnl.Always())

    # Each LCA advances one integration step per call. Bias and weight
    # controllers publish once per trial and hold their values, so processing
    # must not wait for those controllers to execute again on subsequent passes.
    # Preserve the graph's execution order while allowing integration to
    # continue until the response reaches threshold and the output gates run.
    for mechanism in (
        controlExecution, stimulusLayer, decisionLayer, responseLayer,
        WeightedColorInput, WeightedLocationInput,
        WeightedColorStimulus, WeightedLocationStimulus,
    ):
        comp.scheduler.add_condition(mechanism, pnl.Always())

    #comp.show_graph(show_learning=pnl.ALL)
    return comp


def run_lca_model(
        tasks,
        stimuli,
        # Control LCA,
        c_gain=10,
        c_leak=7,
        c_competition=3,
        c_bias=0,
        c_w=4,
        # Stimulus LCA
        s_bias=-0.45,
        s_gain=5,
        s_leak=8,
        s_competition=8,
        # Decision LCA
        d_bias=-0.45,
        d_gain=5,
        d_leak=8,
        d_competition=8,
        d_noise=0.0,
        # Response LCA
        r_bias=-0.45,
        r_gain=5,
        r_leak=8,
        r_competition=8,
        r_threshold=0.0,
        r_noise=0.1,
        # Time gate
        non_decision_time=0.0,
        time_step_size=0.01,
        # Weights
        w1=1.0,
        w2=1.2,
        sdr_bias=-0.45,
        lc_base_gain=5.0,
        lc_scaling=1.0,
        lc_mode=0.9,
        lc_input=0.3,
        lc_threshold=0.5,
        rng_seed=None,
        c_noise=0.0,
        s_noise=0.0,
):

    comp = make_lca_model(
        c_bias=c_bias,
        c_gain=c_gain,
        c_leak=c_leak,
        c_competition=c_competition,
        c_w=c_w,
        c_noise=c_noise,
        s_bias=s_bias,
        s_gain=s_gain,
        s_leak=s_leak,
        s_competition=s_competition,
        s_noise=s_noise,
        d_bias=d_bias,
        d_gain=d_gain,
        d_leak=d_leak,
        d_competition=d_competition,
        d_noise=d_noise,
        r_bias=r_bias,
        r_gain=r_gain,
        r_leak=r_leak,
        r_competition=r_competition,
        r_threshold=r_threshold,
        r_noise=r_noise,
        non_decision_time=non_decision_time,
        time_step_size=time_step_size,
        w1=w1,
        w2=w2,
        sdr_bias=sdr_bias,
        lc_base_gain=lc_base_gain,
        lc_scaling=lc_scaling,
        lc_mode=lc_mode,
        lc_input=lc_input,
        lc_threshold=lc_threshold,
        rng_seed=rng_seed,
    )

    taskInput = comp.nodes["Task Input"]
    stimulusInput = comp.nodes["Stimulus Input"]
    controlExecution = comp.nodes["Control Units\n[Color, Location]"]
    stimulusLayer = comp.nodes["Stimulus Units\n[Red, Blue, Left, Right]"]
    decisionLayer = comp.nodes["Decision Units\n[Left, Right]"]
    responseLayer = comp.nodes["Response Units\n[Left, Right]"]
    lc = comp.nodes["LC"]
  #  lc_control = comp.nodes["LC Control"]

    # Log values for all three layers
    controlExecution.set_log_conditions("value")
    stimulusLayer.set_log_conditions("value")
    decisionLayer.set_log_conditions("value")
    responseLayer.set_log_conditions("value")
    responseLayer.set_log_conditions("num_executions_before_finished")

    # SANITY: also log *inputs* to Stimulus and Decision (called 'variable' in PNL)
    stimulusLayer.set_log_conditions("variable")
    decisionLayer.set_log_conditions("variable")

    # Run the whole block in a single call
    comp.run(
        {
            taskInput: tasks,
            stimulusInput: stimuli,
        },
        execution_mode=pnl.ExecutionMode.LLVMRun
    )

    return comp



if __name__ == "__main__":
    from psyneulink.core.globals.utilities import set_global_seed
    set_global_seed(0)

    tasks, stimuli = conflict_task_sequence(
        n=512, incongruence_frequency=0.5, seed=3
    )
    tasks = tasks[:5]
    stimuli = stimuli[:5]
    print("Full input sequence (first 5):")
    print(np.concatenate((stimuli, tasks), axis=1))

    comp = run_lca_model(tasks, stimuli)

    print(comp.results)

    decision_times = []
    decision_indices = []

    for trial_out in comp.results:
        # trial_out is a (2,1) array: [DECISION_GATE, RT_GATE]
        di = int(np.array(trial_out[0]).ravel()[0])  # DECISION_GATE (index)
        dt = float(np.array(trial_out[1]).ravel()[0])  # RT_GATE (time)

        decision_indices.append(di)
        decision_times.append(dt)

    print("Stimuli (Red,Blue,Left,Right):")
    print(stimuli[:len(decision_indices)])

    print("Decision times (seconds):")
    print(decision_times)

    print("Decisions (0=Left, 1=Right):")
    print(decision_indices)
