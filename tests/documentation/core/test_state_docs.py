# -*- coding: utf-8 -*-
# @Author: allieluu
# @Date:   2017-10-31 19:01:05
# @Last Modified by:   allieluu
# @Last Modified time: 2017-11-01 11:12:48
import psyneulink as pnl


def test_state_docs():
    # get examples of mechanisms that can be used with GatingSignals/Mechanisms
    pass


def test_parameter_state_docs():
    my_mechanism = pnl.RecurrentTransferMechanism(size=5,
                                                  noise=pnl.ControlSignal,
                                                  function=pnl.Logistic(
                                                          gain=(0.5,
                                                                pnl.ControlSignal),
                                                          bias=(1.0,
                                                                pnl.ControlSignal(
                                                                        modulation=pnl.ModulationParam.ADDITIVE))))
    # TODO: get example input/output mechanisms
    # my_mapping_projection = MappingProjection(sender=my_input_mechanism,
    #                                           receiver=my_output_mechanism,
    # matrix=(RANDOM_CONNECTIVITY_MATRIX, LearningSignal))

    my_mechanism = pnl.RecurrentTransferMechanism(
            size=5,
            params={pnl.NOISE: 5,
                    'size': pnl.ControlSignal,
                    pnl.FUNCTION: pnl.Logistic,
                    pnl.FUNCTION_PARAMS: {pnl.GAIN: (0.5, pnl.ControlSignal),
                                          pnl.BIAS: (1.0, pnl.ControlSignal(
                                                  modulation=pnl.ModulationParam.ADDITIVE))}})


def test_output_state_docs():
    my_mech = pnl.TransferMechanism(default_variable=[0, 0],
                                    function=pnl.Logistic(),
                                    output_states=[pnl.TRANSFER_OUTPUT.RESULT,
                                                   pnl.TRANSFER_OUTPUT.MEAN,
                                                   pnl.TRANSFER_OUTPUT.VARIANCE])

    my_mech = pnl.DDM(function=pnl.BogaczEtAl(),
                      output_states=[pnl.DDM_OUTPUT.DECISION_VARIABLE,
                                     pnl.DDM_OUTPUT.PROBABILITY_UPPER_THRESHOLD,
                                     {pnl.NAME: 'DECISION ENTROPY',
                                      pnl.INDEX: 2,
                                      pnl.CALCULATE: pnl.Entropy().function}])
    # TODO: figure out what this entropy thing is
    decision_entropy_output_state = pnl.OutputState(name='DECISION ENTROPY',
                                                    index=2,
                                                    calculate=pnl.Entropy().function)

    my_mech = pnl.DDM(function=pnl.BogaczEtAl(),
                      output_states=[pnl.DDM_OUTPUT.DECISION_VARIABLE,
                                     pnl.DDM_OUTPUT.PROBABILITY_UPPER_THRESHOLD,
                                     decision_entropy_output_state])

    my_mech = pnl.DDM(function=pnl.BogaczEtAl(),
                      output_states=[pnl.DDM_OUTPUT.DECISION_VARIABLE,
                                     pnl.DDM_OUTPUT.PROBABILITY_UPPER_THRESHOLD])
    my_mech.add_states(decision_entropy_output_state)


def test_control_signal_docs():
    my_mech = pnl.TransferMechanism(function=pnl.Logistic(bias=(1.0,
                                                                pnl.ControlSignal)))

    my_mech = pnl.TransferMechanism(function=pnl.Logistic(
            gain=(
                1.0,
                pnl.ControlSignal(modulation=pnl.ModulationParam.ADDITIVE))))

    my_mech_a = pnl.TransferMechanism(function=pnl.Logistic)
    my_mech_b = pnl.TransferMechanism(function=pnl.Linear,
                                      output_states=[pnl.RESULT, pnl.MEAN])
    process_a = pnl.Process(pathway=[my_mech_a])
    process_b = pnl.Process(pathway=[my_mech_b])

    my_system = pnl.System(processes=[process_a, process_b],
                           monitor_for_control=[
                               my_mech_a.output_states[pnl.RESULT],
                               my_mech_b.output_states[pnl.MEAN]],
                           control_signals=[
                               (pnl.GAIN, my_mech_a),
                               {
                                   pnl.NAME: pnl.INTERCEPT,
                                   pnl.MECHANISM: my_mech_b,
                                   pnl.MODULATION: pnl.ModulationParam.ADDITIVE
                               }],
                           name="My Test System"
                           )
