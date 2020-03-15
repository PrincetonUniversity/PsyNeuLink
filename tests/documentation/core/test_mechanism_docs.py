import pytest

import doctest
import psyneulink as pnl


def test_mechanisms():
    fail, total = doctest.testmod(pnl.core.components.mechanisms.mechanism)
    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total),
                    pytrace=False)


def test_transfer_mechanism():
    fail, total = doctest.testmod(
            pnl.core.components.mechanisms.processing.transfermechanism)
    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total),
                    pytrace=False)


def test_integrator_mechanism():
    fail, total = doctest.testmod(
            pnl.core.components.mechanisms.processing.integratormechanism)
    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total),
                    pytrace=False)


# FAILS: AttributeError: 'InputPort' object has no attribute '_name'
def test_objective_mechanism():
    fail, total = doctest.testmod(
            pnl.core.components.mechanisms.processing.objectivemechanism)
    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total),
                    pytrace=False)


def test_control_mechanism():
    fail, total = doctest.testmod(
            pnl.core.components.mechanisms.modulatory.control.controlmechanism)
    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total),
                    pytrace=False)
