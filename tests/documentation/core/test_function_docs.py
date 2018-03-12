import pytest

import psyneulink as pnl
import doctest


def test_function_docs():
    fail, total = doctest.testmod(pnl.components.functions.function)

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total),
                    pytrace=False)