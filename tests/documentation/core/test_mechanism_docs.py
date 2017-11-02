import psyneulink as pnl


def test_mechanisms():
    my_mech = pnl.TransferMechanism(input_states=['MY_INPUT'],
                                    output_states=[pnl.RESULT, pnl.MEAN,
                                                   pnl.VARIANCE])
    my_mech = pnl.TransferMechanism(params={pnl.INPUT_STATES: ['MY_INPUT'],
                                            pnl.OUTPUT_STATES: [pnl.RESULT,
                                                                pnl.MEAN,
                                                                pnl.VARIANCE]})

    my_mechanism = pnl.TransferMechanism(
            function=pnl.Logistic(gain=1.0, bias=-4))
    my_mechanism = pnl.TransferMechanism(function=pnl.Logistic, params={
        pnl.FUNCTION_PARAMS: {pnl.GAIN: 1.0, pnl.BIAS: -4.0}})


def test_transfer_mechanism():
    my_linear_transfer_mechanism = pnl.TransferMechanism(function=pnl.Linear)
    my_logistic_transfer_mechanism = pnl.TransferMechanism(
            function=pnl.Logistic(gain=1.0, bias=-4))


def test_integrator_mechanism():
    my_time_averaging_mechanism = pnl.IntegratorMechanism(
            function=pnl.AdaptiveIntegrator(rate=0.5))


def test_objective_mechanism():
    my_action_select_mech = pnl.TransferMechanism(default_variable=[0, 0, 0],
                                                  function=pnl.SoftMax(
                                                          output=pnl.PROB),
                                                  name='Action Selection')

    my_reward_mech = pnl.TransferMechanism(default_variable=[0],
                                           name='Reward')

    my_objective_mech = pnl.ObjectiveMechanism(
            input_states=[my_action_select_mech.output_state,
                          my_reward_mech.output_state])

    # FIXME: StateError: fewer InputStates than number of items in variable
    my_objective_mech = pnl.ObjectiveMechanism(default_variable=[[0], [0]],
                                               input_states=[
                                                   my_action_select_mech,
                                                   my_reward_mech])

    my_objective_mech = pnl.ObjectiveMechanism(default_variable=[[0], [0]],
                                               input_states=[(
                                                   my_action_select_mech, -1,
                                                   1),
                                                   my_reward_mech])

    # idk what this is supposed to be doing
    my_objective_mech = pnl.ObjectiveMechanism(
            inputt_states=[my_reward_mech, {pnl.MECHANISM: Decision,
                                            pnl.OUTPUT_STATES: [
                                                pnl.PROBABILITY_UPPER_THRESHOLD,
                                                (pnl.RESPONSE_TIME, 1, -1)]}])


def test_control_mechanism():
    my_transfer_mech_A = pnl.TransferMechanism(name="Transfer Mech A")
    my_DDM = pnl.DDM(name="DDM")
    my_transfer_mech_B = pnl.TransferMechanism(function=pnl.Logistic,
                                               name="Transfer Mech B")
    # how should we refer to OutputStates??
    my_control_mech = pnl.ControlMechanism(
            objective_mechanism=pnl.ObjectiveMechanism(
                    input_states=[(my_transfer_mech_A, 2, 1, None),
                                  my_DDM.output_states[pnl.RESPONSE_TIME]],
                    function=pnl.LinearCombination(operation=pnl.PRODUCT)),
            control_signals=[(pnl.THRESHOLD, my_DDM),
                             (pnl.GAIN, my_transfer_mech_B)])

    # this is the only example that's broken -- the rest work
    my_control_mech = pnl.ControlMechanism(
            objective_mechanism=[
                pnl.MonitoredOutputStateTuple(
                        output_state=my_transfer_mech_A,
                        weight=2, exponent=1, matrix=None),
                my_DDM.output_states[pnl.RESPONSE_TIME]],
            control_signals=[(pnl.THRESHOLD, my_DDM),
                             (pnl.GAIN, my_transfer_mech_B)])

    my_obj_mech = pnl.ObjectiveMechanism(
            input_states=[(my_transfer_mech_A, 2, 1),
                          my_DDM.output_states[pnl.RESPONSE_TIME]],
            function=pnl.LinearCombination(operation=pnl.PRODUCT))

    my_control_mech = pnl.ControlMechanism(
            objective_mechanism=my_obj_mech,
            control_signals=[(pnl.THRESHOLD, my_DDM),
                             (pnl.GAIN, my_transfer_mech_B)])
