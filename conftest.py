import pytest
import random
# import time
import numpy as np

from psyneulink.core.llvm import ptx_enabled
from psyneulink import clear_registry, primary_registries

# def pytest_addoption(parser):
#     parser.addoption(
#         '--pnl-seed',
#         action='store',
#         default=int(time.time() * 256),
#         help='the seed to use for each test'
#     )

mark_stress_tests = 'stress'

marks_default_skip = [mark_stress_tests]


# skip stress tests by default, add command option to include
# http://blog.devork.be/2009/12/skipping-slow-test-by-default-in-pytest.html
def pytest_addoption(parser):
    parser.addoption('--{0}'.format(mark_stress_tests), action='store_true', default=False, help='Run {0} tests (long)'.format(mark_stress_tests))

def pytest_runtest_setup(item):
    import doctest

    for m in marks_default_skip:
        if m in item.keywords and not item.config.getvalue(m):
            pytest.skip('{0} tests not requested'.format(m))

    if 'cuda' in item.keywords and not ptx_enabled:
            pytest.skip('PTX engine not enabled/available')

    doctest.ELLIPSIS_MARKER = "[...]"


def pytest_runtest_call(item):
    # seed = int(item.config.getoption('--pnl-seed'))
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    from psyneulink.core.globals.utilities import set_global_seed
    set_global_seed(seed)



def pytest_runtest_teardown(item):
    for registry in primary_registries:
        # Clear Registry to have a stable reference for indexed suffixes of default names
        clear_registry(registry)

    from psyneulink.core import llvm as pnlvm
    pnlvm.cleanup()


@pytest.helpers.register
def expand_np_ndarray(arr):
    # this will fail on an input containing a float (not np.ndarray)
    try:
        iter(arr)
    except TypeError:
        return arr.tolist()

    results_list = []
    for elem in arr:
        try:
            iter(elem)
        except TypeError:
            elem = [elem]

        for nested_elem in elem:
            nested_elem = nested_elem.tolist()
            try:
                iter(nested_elem)
            except TypeError:
                nested_elem = [nested_elem]
            results_list.extend(nested_elem)
    return results_list
