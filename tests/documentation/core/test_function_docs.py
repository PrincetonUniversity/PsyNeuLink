import pytest

import doctest
import psyneulink as pnl


def test_function_docs():
    fail, total = doctest.testmod(pnl.core.components.functions.function)

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total),
                    pytrace=False)
