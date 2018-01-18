import pytest
import numpy as np
import psyneulink as pnl


def test_EVC():
    '''ControlMechanism replacement and monitor_for_control specifications'''

    T1 = pnl.TransferMechanism(size=3)
    T2 = pnl.TransferMechanism(output_states=[{pnl.NAME: 'O-1'}])
    T3 = pnl.TransferMechanism()
    P = pnl.Process(pathway=[T1, T2, T3])
    S = pnl.System(processes=P,
                   controller=pnl.EVCControlMechanism,
                   enable_controller=True,
                   # Test for use of 4-item tuple with matrix in monitor_for_control specification
                   monitor_for_control=[(T1, None, None, np.ones((3,1))),
                                        ('O-1', 1, -1)]
                   )
    assert len(S.controller.objective_mechanism.monitored_output_states)==2

    # Test for duplicate assignment of a monitored_output_state
    C1 = pnl.EVCControlMechanism(name='C1',
                                objective_mechanism = [(T1, None, None, np.ones((3,1)))])

    # Test for adding a monitored_output_state
    C2 = pnl.EVCControlMechanism(name='C2',
                                 objective_mechanism = [T3.output_states[pnl.RESULTS]])

    # Test direct assignment
    S.controller = C1
    assert len(C1.monitored_output_states)==2
    assert S.controller.name == 'C1'

    # Test use of assign_as_controller method
    C2.assign_as_controller(S)
    assert len(C2.monitored_output_states)==3
    assert S.controller.name == 'C2'

