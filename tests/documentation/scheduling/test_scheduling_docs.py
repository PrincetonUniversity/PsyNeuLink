import doctest

import pytest

import psyneulink as pnl


def test_scheduler_docs():
    fail, test = doctest.testmod(pnl.core.scheduling.scheduler,
                                 optionflags=doctest.REPORT_NDIFF)

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, test))
