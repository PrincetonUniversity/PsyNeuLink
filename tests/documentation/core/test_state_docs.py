import pytest

import psyneulink as pnl
import doctest


def test_state_docs():
    # get examples of mechanisms that can be used with GatingSignals/Mechanisms
    pass


def test_parameter_state_docs():
    fail, total = doctest.testmod(pnl.components.states.parameterstate)

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total),
                    pytrace=False)


def test_output_state_docs():
    fail, total = doctest.testmod(pnl.components.states.outputstate)

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total))


def test_control_signal_docs():
    fail, total = doctest.testmod(pnl.components.states.modulatorysignals.controlsignal)

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total))


def test_gating_signal_docs():
    fail, total = doctest.testmod(pnl.components.states.modulatorysignals.gatingsignal)

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total))
