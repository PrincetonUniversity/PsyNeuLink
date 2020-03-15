import pytest
import numpy as np
import psyneulink as pnl
import psyneulink.core.components.functions.transferfunctions


def test_control_mechanism_assignment():
    """ControlMechanism assignment/replacement,  monitor_for_control, and control_signal specifications"""

    T1 = pnl.TransferMechanism(size=3, name='T-1')
    T2 = pnl.TransferMechanism(function=psyneulink.core.components.functions.transferfunctions.Logistic, output_ports=[{pnl.NAME: 'O-1'}], name='T-2')
    T3 = pnl.TransferMechanism(function=psyneulink.core.components.functions.transferfunctions.Logistic, name='T-3')
    T4 = pnl.TransferMechanism(function=psyneulink.core.components.functions.transferfunctions.Logistic, name='T-4')
    P = pnl.Process(pathway=[T1, T2, T3, T4])
    S = pnl.System(processes=P,
                   # controller=pnl.EVCControlMechanism,
                   controller=pnl.EVCControlMechanism(
                           control_signals=[(pnl.GAIN, T2)],
                           monitor_for_control=T4
                   ),
                   enable_controller=True,
                   # Test for use of 4-item tuple with matrix in monitor_for_control specification
                   monitor_for_control=[(T1, None, None, np.ones((3,1))),
                                        ('O-1', 1, -1)],
                   control_signals=[(pnl.GAIN, T3)]
                   )
    assert len(S.controller.objective_mechanism.monitor)==3
    assert len(S.control_signals)==2

    # Test for avoiding duplicate assignment of monitor and control_signals
    C1 = pnl.EVCControlMechanism(name='C-1',
                                 objective_mechanism = [(T1, None, None, np.ones((3,1)))],
                                 control_signals=[(pnl.GAIN, T3)]
                                 )

    # Test direct assignment
    S.controller = C1
    assert len(C1.monitored_output_ports)==2
    assert len(S.control_signals)==3
    assert S.controller.name == 'C-1'


    # Test for adding a monitored_output_port and control_signal
    C2 = pnl.EVCControlMechanism(name='C-2',
                                 objective_mechanism = [T3.output_ports[pnl.RESULT]],
                                 control_signals=[(pnl.GAIN, T4)])
    # Test use of assign_as_controller method
    C2.assign_as_controller(S)
    assert len(C2.monitored_output_ports)==3
    assert len(S.control_signals)==4
    assert S.controller.name == 'C-2'


# def test_control_mechanism_assignment_additional():
#     """Tests "free-standing" specifications of monitor_for_control and ControlSignal (i.e., outside of a list)"""
#     T_1 = pnl.TransferMechanism(name='T_1')
#     T_2 = pnl.TransferMechanism(name='T_2')
#     T_3 = pnl.TransferMechanism(name='T_3')
#     S = pnl.sys([T_1, T_2, T_3],
#                 controller=pnl.EVCControlMechanism(
#                         control_signals=(pnl.SLOPE, T_1),
#                         monitor_for_control=[T_1],
#                         objective_mechanism=[T_2]),
#                 monitor_for_control=T_3,
#                 control_signals=(pnl.SLOPE, T_2),
#                 enable_controller=True)
#     assert S.controller.objective_mechanism.input_port[0].path_afferents[0].sender.owner == T_1
#     assert S.controller.objective_mechanism.input_port[1].path_afferents[0].sender.owner == T_2
#     assert S.controller.objective_mechanism.input_port[2].path_afferents[0].sender.owner == T_3
#     assert T_1.parameter_ports[pnl.SLOPE].mod_afferents[0].sender.owner == S.controller
#     assert T_2.parameter_ports[pnl.SLOPE].mod_afferents[0].sender.owner == S.controller



def test_control_mechanism_assignment_additional():
    """Tests "free-standing" specifications of monitor_for_control and ControlSignal (i.e., outside of a list)"""
    T_1 = pnl.TransferMechanism(name='T_1')
    T_2 = pnl.TransferMechanism(name='T_2')
    S = pnl.sys([T_1,T_2],
                controller=pnl.EVCControlMechanism(control_signals=(pnl.SLOPE, T_1)),
                monitor_for_control=[T_1],
                control_signals=(pnl.SLOPE, T_2),
                enable_controller=True)
    assert S.controller.objective_mechanism.input_port.path_afferents[0].sender.owner == T_1
    assert T_1.parameter_ports[pnl.SLOPE].mod_afferents[0].sender.owner == S.controller
    assert T_2.parameter_ports[pnl.SLOPE].mod_afferents[0].sender.owner == S.controller

# def test_prediction_mechanism_assignment():
#     """Tests prediction mechanism assignment and more tests for ObjectiveMechanism and ControlSignal assignments"""
#
#     T1 = pnl.TransferMechanism(name='T1')
#     T2 = pnl.TransferMechanism(name='T2')
#     T3 = pnl.TransferMechanism(name='T3')
#
#     S = pnl.sys([T1, T2, T3],
#                 # controller=pnl.EVCControlMechanism(name='EVC',
#                 controller=pnl.EVCControlMechanism(name='EVC',
#                                                    prediction_mechanisms=(pnl.PredictionMechanism,
#                                                                           {pnl.FUNCTION: pnl.INPUT_SEQUENCE,
#                                                                            pnl.RATE: 1,
#                                                                            pnl.WINDOW_SIZE: 3,
#                                                                            }),
#                                                    monitor_for_control=[T1],
#                                                    objective_mechanism=[T2]
#                                                    ),
#                 control_signals=pnl.ControlSignal(allocation_samples=[1, 5, 10],
#                                                   modulates=(pnl.SLOPE, T1)),
#                 monitor_for_control=T3,
#                 enable_controller=True
#                 )
#     assert len(S.controller.objective_mechanism.input_ports)==3
#
#     S.recordSimulationPref = True
#
#     input_dict = {T1:[1,2,3,4]}
#     results = S.run(inputs=input_dict)
#     assert results == [[[1.]], [[2.]], [[15.]], [[20.]]]
#     assert S.simulation_results ==  [[[1.]], [[5.]], [[10.]],
#                                     [[1.]], [[2.]], [[5.]], [[10.]], [[10.]], [[20.]],
#                                     [[1.]], [[2.]], [[3.]], [[5.]], [[10.]], [[15.]], [[10.]], [[20.]], [[30.]],
#                                     [[2.]], [[3.]], [[4.]], [[10.]], [[15.]], [[20.]], [[20.]], [[30.]], [[40.]]]

# def test_prediction_mechanism_filter_function():
#     """Tests prediction mechanism assignment and more tests for ObjectiveMechanism and ControlSignal assignments"""
#
#     f = lambda x: [x[0]*7]
#     T = pnl.TransferMechanism(name='T')
#
#     S = pnl.sys(T,
#                 controller=pnl.EVCControlMechanism(name='EVC',
#                                                    prediction_mechanisms=(pnl.PredictionMechanism,
#                                                                           {pnl.FUNCTION: pnl.INPUT_SEQUENCE,
#                                                                            pnl.RATE: 1,
#                                                                            pnl.WINDOW_SIZE: 3,
#                                                                            pnl.FILTER_FUNCTION: f
#                                                                            }),
#                                                    objective_mechanism=[T]
#                                                    ),
#                 control_signals=pnl.ControlSignal(allocation_samples=[1, 5, 10],
#                                                   modulates=(pnl.SLOPE, T)),
#                 enable_controller=True
#                 )
#
#     S.recordSimulationPref = True
#     input_dict = {T: [1, 2, 3, 4]}
#     results = S.run(inputs=input_dict)
#     expected_results = [[[1.0]], [[2.0]], [[3.0]], [[4.0]]]
#     expected_sim_results = [[[1.]], [[5.]], [[10.]],    # before EVC | [1]
#                             [[7.]], [[35.]], [[70.]],   # [1, 2]
#                             [[7.]], [[35.]], [[70.]],   # [1, 2, 3]
#                             [[14.]], [[70.]], [[140.]]] # [2, 3, 4]
#
#     np.testing.assert_allclose(results, expected_results, atol=1e-08, err_msg='Failed on results')
#     np.testing.assert_allclose(S.simulation_results, expected_sim_results, atol=1e-08, err_msg='Failed on results')
