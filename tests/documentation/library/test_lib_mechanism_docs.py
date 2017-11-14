import doctest

import pytest

import psyneulink as pnl


def test_lc_mechanism_docs():
    fail, total = doctest.testmod(
            pnl.library.mechanisms.adaptive.control.lcmechanism)

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total),
                    pytrace=False)


def test_ddm_docs():
    fail, total = doctest.testmod(
            pnl.library.mechanisms.processing.integrator.ddm)

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total),
                    pytrace=False)


def test_comparator_mechanism_docs():
    fail, total = doctest.testmod(
            pnl.library.mechanisms.processing.objective.comparatormechanism)

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total),
                    pytrace=False)
