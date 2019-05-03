import doctest
import psyneulink as pnl
import pytest


def test_log_docs():
    fail, total = doctest.testmod(pnl.core.globals.log)

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total),
                    pytrace=False)
