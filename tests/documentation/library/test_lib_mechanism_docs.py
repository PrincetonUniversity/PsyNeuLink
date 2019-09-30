import doctest

import os
import pytest

import psyneulink as pnl


def test_ddm_docs():
    # FIXME: Does this run outside of the test directory?
    # os.chdir('../../Matlab/DDMFunctions')
    # print("current dir = {}".format(os.getcwd()))
    # ALSO FIXME: ValueError cannot convert float NaN integer
    fail, total = doctest.testmod(
        pnl.library.components.mechanisms.processing.integrator.ddm
    )

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total),
                    pytrace=False)


def test_comparator_mechanism_docs():
    fail, total = doctest.testmod(
        pnl.library.components.mechanisms.processing.objective.comparatormechanism
    )

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total),
                    pytrace=False)


def test_lc_control_mechanism_docs():
    fail, test = doctest.testmod(
        pnl.library.components.mechanisms.modulatory.control.agt.lccontrolmechanism,
        optionflags=doctest.REPORT_NDIFF
    )

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, test),
                    pytrace=False)


def test_evc_control_mechanism_docs():
    fail, test = doctest.testmod(
        pnl.library.components.mechanisms.modulatory.control.evc.evccontrolmechanism,
        optionflags=doctest.REPORT_NDIFF
    )

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, test),
                    pytrace=False)
