import pytest

import doctest
import psyneulink as pnl

def test_port_docs():
    # get examples of mechanisms that can be used with GatingSignals/Mechanisms
    pass


def test_parameter_port_docs():
    fail, total = doctest.testmod(pnl.core.components.ports.parameterport, globs={})

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total),
                    pytrace=False)


def test_output_port_docs():
    fail, total = doctest.testmod(pnl.core.components.ports.outputport)

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total))


def test_control_signal_docs():
    fail, total = doctest.testmod(pnl.core.components.ports.modulatorysignals.controlsignal)

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total))


def test_gating_signal_docs():
    fail, total = doctest.testmod(pnl.core.components.ports.modulatorysignals.gatingsignal)

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total))
