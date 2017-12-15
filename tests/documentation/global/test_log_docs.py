import pytest
import doctest
import psyneulink as pnl

def test_log_docs():
    fail, total = doctest.testmod(pnl.globals.log)

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total),
                    pytrace=False)
