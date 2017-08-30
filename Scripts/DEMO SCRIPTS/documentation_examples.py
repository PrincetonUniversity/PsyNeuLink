import numpy as np
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms \
    .IntegratorMechanism import \
    IntegratorMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms \
    .RecurrentTransferMechanism import \
    RecurrentTransferMechanism
from PsyNeuLink.Scheduling.condition import EveryNCalls, Any, AfterNCalls, EveryNPasses

from PsyNeuLink import ModulationParam
from PsyNeuLink.Components.Functions.Function import AdaptiveIntegrator, \
    BogaczEtAl, DriftDiffusionIntegrator, Linear, Logistic, PROB, SoftMax
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.GatingMechanism \
    .GatingMechanism import \
    GatingMechanism
from PsyNeuLink.Components.Process import PARAMETER_STATE_PARAMS, \
    RANDOM_CONNECTIVITY_MATRIX, process
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection \
    import MappingProjection
from PsyNeuLink.Components.States.ModulatorySignals import ControlSignal
from PsyNeuLink.Components.System import system
from PsyNeuLink.Globals.Keywords import ENABLED, GAIN, NAME, INDEX, CALCULATE, \
    INTERCEPT, MECHANISM, MODULATION, GATE
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.IntegratorMechanisms import DDM, \
    NOISE, THRESHOLD, TimeScale
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms import \
    ComparatorMechanism
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.TransferMechanisms.TransferMechanism \
    import \
    MEAN, RESULT, TransferMechanism, VARIANCE, TRANSFER_OUTPUT
from PsyNeuLink.Scheduling.Scheduler import Scheduler


def intro():
    # Simple Configurations
    # TypeError: object of type 'int' has no len() --> Transfer_DEFAULT_BIAS
    # is set
    # to 0, which sets the variable argument --> needs to somehow not be an
    # int or
    # need to change line 1257 in Component.py (_instantiate_defaults())
    input_layer = TransferMechanism(size=5)
    hidden_layer = TransferMechanism(size=2, function=Logistic)
    output_layer = TransferMechanism(size=5, function=Logistic)
    my_encoder = process(pathway=[input_layer, hidden_layer, output_layer])

    output_layer.execute([0, 2.5, 10.9, 2, 7.6])

    my_encoder.run([0, 2.5, 10.9, 2, 7.6])

    my_projection_1 = MappingProjection(matrix=(0.2 * np.random.rand(2, 5)))
    my_encoder = process(
        pathway=[input_layer, my_projection_1, hidden_layer, output_layer])

    my_encoder_2 = process(
        pathway=[input_layer, (0.2 * np.random.rand(2, 5)) + -0.1])

    # recurrent projection
    my_encoder_3 = process(
        pathway=[input_layer, hidden_layer, output_layer, hidden_layer])

    # More elaborate configurations
    my_encoder_4 = process(
        pathway=[input_layer, hidden_layer, output_layer], learning=ENABLED)
    my_encoder_4.run(
        input=[[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 1, 0, 0],
               [0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1]],
        target=[[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]])

    # System
    colors_input_layer = TransferMechanism(
        size=2, function=Logistic, name='COLORS INPUT')
    words_input_layer = TransferMechanism(
        size=2, function=Logistic, name='WORDS INPUT')
    differencing_weights = np.array([[1], [-1]])
    output_layer = TransferMechanism(size=1, name='OUTPUT')
    decision_mech = DDM(name='DECISION')
    colors_process = process(
        pathway=[colors_input_layer, differencing_weights, output_layer])
    words_process = process(
        pathway=[words_input_layer, differencing_weights, output_layer])
    decision_process = process(pathway=[output_layer, decision_mech])
    my_simple_Stroop = system(
        processes=[colors_process, words_process, decision_process])

    my_simple_Stroop.show_graph(output_fmt='pdf')


def processes():
    # specification of mechanisms in a pathway
    mechanism_1 = TransferMechanism()
    mechanism_2 = DDM()
    some_params = {PARAMETER_STATE_PARAMS: {THRESHOLD: 2, NOISE: 0.1}}
    my_process = process(
        pathway=[mechanism_1, TransferMechanism, (mechanism_2, some_params)])

    # default projection specification
    mechanism_3 = DDM()
    my_process_2 = process(pathway=[mechanism_1, mechanism_2, mechanism_3])

    # inline projection specification using an existing projection
    projection_A = MappingProjection()
    my_process_3 = process(
        pathway=[mechanism_1, projection_A, mechanism_2, mechanism_3])

    # inline projection specification using a keyword
    my_process_4 = process(
        pathway=[mechanism_1, RANDOM_CONNECTIVITY_MATRIX, mechanism_2,
                 mechanism_3])

    # stand-alone projection specification
    projection_A = MappingProjection(sender=mechanism_1, receiver=mechanism_2)
    my_process_5 = process(pathway=[mechanism_1, mechanism_2, mechanism_3])

    # process that implements learning
    mechanism_1 = TransferMechanism(function=Logistic)
    mechanism_2 = TransferMechanism(function=Logistic)
    mechanism_3 = TransferMechanism(function=Logistic)


def mechanisms():
    # Creating a mechanism
    my_mech = TransferMechanism(input_states=['MY_INPUT'],
                                output_states=[RESULT, MEAN, VARIANCE])

    # Structure
    # Function
    my_mechanism = TransferMechanism(function=Logistic(gain=1.0, bias=-4))

    # region TransferMechanism
    my_linear_transfer_mechanism = TransferMechanism(function=Linear)
    my_logistic_transfer_mechanism = TransferMechanism(
        function=Logistic(gain=1.0, bias=-4))
    # class arguments section doesn't word wrap
    # endregion

    # region IntegratorMechanism
    my_time_averaging_mechanism = IntegratorMechanism(
        function=AdaptiveIntegrator(rate=0.5))
    # endregion

    # region DDM
    # Structure
    my_DDM = DDM(function=BogaczEtAl)
    # FIXME: Value of threshold param for BogaczEtAl must be a float
    # my_DDM = DDM(
    #     function=BogaczEtAl(drift_rate=0.2, threshold=(1.0,
    # ControlProjection)))

    # DDM Parameters
    my_DDM_BogaczEtAl = DDM(
        function=BogaczEtAl(drift_rate=3.0, starting_point=1.0, threshold=30.0,
                            noise=1.5, t0=2.0),
        time_scale=TimeScale.TRIAL,
        name='MY_DDM_BogaczEtAl')
    # FIXME: Undefined function 'ddmSim' for input arguments of type 'int64'
    # my_DDM_NavarroAndFuss = DDM(
    #     function=NavarroAndFuss(drift_rate=3.0, starting_point=1.0,
    #                             threshold=30.0, noise=1.5, t0=2.0),
    #     time_scale=TimeScale.TRIAL, name='MY_DDM_NavarroAndFuss')
    my_DDM_TimeStep = DDM(
        function=DriftDiffusionIntegrator(noise=0.5,
                                          time_step_size=1.0, initializer=0.0),
        time_scale=TimeScale.TIME_STEP,
        name="My_DDM_TimeStep")
    # endregion

    # region ObjectiveMechanism
    my_action_select_mech = TransferMechanism(default_variable=[0, 0, 0],
                                              function=SoftMax(output=PROB))
    my_reward_mech = TransferMechanism(default_variable=[0])
    # FIXME: ObjectiveMechanism has no `default_variable` keyword
    # FIXME: PROGRAM ERROR: call to State._parse_state_spec() for OutputState
    #  of ObjectiveMechanism should have returned dict or State, but returned
    #  str instead
    # my_objective_mech = ObjectiveMechanism(monitored_values=[
    #     my_action_select_mech,
    #     my_reward_mech])

    # my_objective_mech = ObjectiveMechanism(
    #     monitored_values=[my_action_select_mech, my_reward_mech],
    #     function=LinearCombination(weights=[[-1], [1]]))
    # endregion

    # region ComparatorMechanism
    # FIXME: TypeError: object of type 'int' has no len()
    my_action_select_mech = TransferMechanism(function=SoftMax(output=PROB))
    my_reward_mech = TransferMechanism(default_variable=[0])
    # FIXME: ObjectiveMechanismError: "PROGRAM ERROR: call to
    # State._parse_state_spec() for OutputState of ComparatorMechanism-1
    # should have returned dict or State, but returned <class 'str'> instead"
    my_comparator_mech = ComparatorMechanism(sample=my_action_select_mech,
                                             target=my_reward_mech,
                                             input_states=[[0], [0]])
    # which function keyword argument description is correct?
    # endregion


def states():
    # region ParameterState
    # FIXME: noise should be a numeric value
    my_mechanism = RecurrentTransferMechanism(size=5,
                                              function=Logistic(
                                                  gain=0.5,
                                                  bias=(1.0,
                                                        ControlSignal.ControlSignal(
                                                            modulation=ModulationParam.ADDITIVE))))

    # my_mapping_projection = MappingProjection(sender=my_input_mechanism,
    # receiver=my_output_mechanism, matrix=(RANDOM_CONNECTIVITY_MATRIX,
    # LearningSignal))


    # FIXME: AttributeError: 'RecurrentTransferMechanism' object has no
    # attribute '_prefs'
    # my_mechanism = RecurrentTransferMechanism(size=5, params={NOISE: 5,
    #                                                           'size':
    #
    # ControlSignal,
    #                                                           FUNCTION:
    #                                                               Logistic,
    #
    # FUNCTION_PARAMS: {
    #                                                               GAIN: (0.5,
    #
    # ControlSignal),
    #                                                               BIAS: (1.0,
    #
    # ControlSignal.ControlSignal(
    #
    #   modulation=ModulationParam.ADDITIVE))}})

    # endregion

    # region OutputStates
    my_mech = TransferMechanism(default_variable=[0, 0], function=Logistic(),
                                output_states=[TRANSFER_OUTPUT.RESULT,
                                               TRANSFER_OUTPUT.MEAN,
                                               TRANSFER_OUTPUT.VARIANCE])

    my_mech = DDM(function=BogaczEtAl(), output_states=[DDM.DECISION_VARIABLE,
                                                        DDM.PROB_UPPER_THRESHOLD,
                                                        {
                                                            NAME: 'DECISION '
                                                                  'ENTROPY',
                                                            INDEX: 2,
                                                            CALCULATE:
                                                                Entropy().function}])
    # endregion

    # region ControlSignals
    my_mech = TransferMechanism(function=Logistic(bias=(1.0, ControlSignal)))
    # need clarification on Modulate the parameter of a Mechanism’s function

    my_mech = TransferMechanism(function=Logistic(gain=(
        1.0, ControlSignal.ControlSignal(modulation=ModulationParam.ADDITIVE))))

    my_mech_A = TransferMechanism(function=Logistic)
    my_mech_B = TransferMechanism(function=Linear, output_states=[RESULT, MEAN])

    process_a = process(pathway=[my_mech_A])
    process_b = process(pathway=[my_mech_B])

    my_system = system(processes=[process_a, process_b],
                       monitor_for_control=[my_mech_A.output_states[RESULT],
                                            my_mech_B.output_states[MEAN]],
                       control_signals=[(GAIN, my_mech_A), {NAME: INTERCEPT,
                                                            MECHANISM:
                                                                my_mech_B,
                                                            MODULATION:
                                                                ModulationParam.ADDITIVE}],
                       name='My Test System')
    # endregion

    # region GatingSignal
    my_mechanism_a = TransferMechanism()
    my_mechanism_b = TransferMechanism()
    my_gating_mechanism = GatingMechanism(
        gating_signals=[my_mechanism_a, my_mechanism_b.output_state])

    my_input_layer = TransferMechanism(size=3)
    my_hidden_layer = TransferMechanism(size=5)
    my_output_layer = TransferMechanism(size=2)
    my_gating_mechanism = GatingMechanism(gating_signals=[
        {'GATE_ALL': [my_input_layer, my_hidden_layer, my_output_layer]}],
        modulation=ModulationParam.ADDITIVE)

    # Should this be list or tuple?
    my_gating_mechanism = GatingMechanism(gating_signals=[
        {NAME: 'GATING_SIGNAL_A', GATE: my_input_layer,
         MODULATION: ModulationParam.ADDITIVE},
        {NAME: 'GATING_SIGNAL_B', GATE: [my_hidden_layer, my_output_layer]}])
    # endregion


def scheduler():
    # basic phasing in a linear process
    A = TransferMechanism(function=Linear(), name='A')
    B = TransferMechanism(function=Linear(), name='B')
    C = TransferMechanism(function=Linear(), name='C')

    # p = process(pathway=[A, B, C], name='p')
    # s = system(processes=[p], name='s')
    #
    # sched = Scheduler(system=s)
    # sched.add_condition(B, EveryNCalls(A, 2))
    # sched.add_condition(C, EveryNCalls(B, 3))
    #
    # output = list(sched.run())
    # print(output)

    # alternate basic phasing in a linear process
    # p = process(pathway=[A, B], name='p')
    # s = system(processes=[p], name='s')
    #
    # sched = Scheduler(system=s)
    # sched.add_condition(A, Any(AtPass(0), EveryNCalls(B, 2)))
    # sched.add_condition(B, Any(EveryNCalls(A, 1), EveryNCalls(B, 1)))
    #
    # termination_conds = {ts: None for ts in TimeScale}
    # termination_conds[TimeScale.TRIAL] = AfterNCalls(B, 4,
    #                                                  time_scale=TimeScale.TRIAL)
    # output = list(sched.run())
    # print(output)
    # A = TransferMechanism(function=Linear(), name='A')
    # B = TransferMechanism(function=Linear(), name='B')
    #
    # p = process(
    #     pathway=[A, B],
    #     name='p',
    # )
    # s = system(
    #     processes=[p],
    #     name='s',
    # )
    # sched = Scheduler(system=s)
    #
    # sched.add_condition(A, Any(AtPass(0), EveryNCalls(B, 2)))
    # sched.add_condition(B, Any(EveryNCalls(A, 1), EveryNCalls(B, 1)))
    #
    # termination_conds = {ts: None for ts in TimeScale}
    # termination_conds[TimeScale.TRIAL] = AfterNCalls(B, 4,
    #                                                  time_scale=TimeScale.TRIAL)
    # output = list(sched.run(termination_conds=termination_conds))
    # print(output)

    # basic phasing in two processes
    p = process(pathway=[A, C], name='p')
    q = process(pathway=[B, C], name='q')
    s = system(processes=[p, q], name='s')

    sched = Scheduler(system=s)
    sched.add_condition(A, EveryNPasses(1))
    sched.add_condition(B, EveryNCalls(A, 2))
    sched.add_condition(C, Any(AfterNCalls(A, 3), AfterNCalls(B, 3)))

    termination_conds = {ts: None for ts in TimeScale}
    termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 4, time_scale=TimeScale.TRIAL)

    output = list(sched.run(termination_conds=termination_conds))
    print(output)


if __name__ == '__main__':
    # processes()
    # mechanisms()
    # states()
    processes()
    scheduler()
