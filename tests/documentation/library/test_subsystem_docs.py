import doctest

import pytest

import psyneulink as pnl


def test_lc_control_mechanism_docs():
    fail, test = doctest.testmod(pnl.library.subsystems.agt.lccontrolmechanism,
                                 optionflags=doctest.REPORT_NDIFF)

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, test),
                    pytrace=False)


def test_evc_control_mechanism_docs():
    fail, test = doctest.testmod(pnl.library.subsystems.evc.evccontrolmechanism,
                                 optionflags=doctest.REPORT_NDIFF)

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, test),
                    pytrace=False)
