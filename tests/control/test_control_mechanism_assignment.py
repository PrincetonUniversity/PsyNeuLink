import pytest
import numpy as np
import psyneulink as pnl


def test_control_mechanism_assignment():
    '''ControlMechanism assignment/replacement,  monitor_for_control, and control_signal specifications'''

    T1 = pnl.TransferMechanism(size=3, name='T-1')
    T2 = pnl.TransferMechanism(function=pnl.Logistic, output_states=[{pnl.NAME: 'O-1'}], name='T-2')
    T3 = pnl.TransferMechanism(function=pnl.Logistic, name='T-3')
    T4 = pnl.TransferMechanism(function=pnl.Logistic, name='T-4')
    P = pnl.Process(pathway=[T1, T2, T3, T4])
    S = pnl.System(processes=P,
                   # controller=pnl.EVCControlMechanism,
                   controller=pnl.EVCControlMechanism(
                           control_signals=[(pnl.GAIN, T2)]
                   ),
                   enable_controller=True,
                   # Test for use of 4-item tuple with matrix in monitor_for_control specification
                   monitor_for_control=[(T1, None, None, np.ones((3,1))),
                                        ('O-1', 1, -1)],
                   control_signals=[(pnl.GAIN, T3)]
                   )
    assert len(S.controller.objective_mechanism.monitored_output_states)==2
    assert len(S.control_signals)==2

    # Test for avoiding duplicate assignment of monitored_output_states and control_signals
    C1 = pnl.EVCControlMechanism(name='C-1',
                                 objective_mechanism = [(T1, None, None, np.ones((3,1)))],
                                 control_signals=[(pnl.GAIN, T3)]
                                 )

    # Test direct assignment
    S.controller = C1
    assert len(C1.monitored_output_states)==2
    assert len(S.control_signals)==3
    assert S.controller.name == 'C-1'


    # Test for adding a monitored_output_state and control_signal
    C2 = pnl.EVCControlMechanism(name='C-2',
                                 objective_mechanism = [T3.output_states[pnl.RESULTS]],
                                 control_signals=[(pnl.GAIN, T4)])
    # Test use of assign_as_controller method
    C2.assign_as_controller(S)
    assert len(C2.monitored_output_states)==3
    assert len(S.control_signals)==4
    assert S.controller.name == 'C-2'


