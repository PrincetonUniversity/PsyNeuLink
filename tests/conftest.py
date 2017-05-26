import pytest
import random
# import time
import numpy as np

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
    for m in marks_default_skip:
        if getattr(item.obj, m, None) and not item.config.getvalue(m):
            pytest.skip('{0} tests not requested'.format(m))


def pytest_runtest_call(item):
    # seed = int(item.config.getoption('--pnl-seed'))
    seed = 0
    random.seed(seed)
    np.random.seed(seed)


@pytest.helpers.register
def expand_np_ndarray(arr):
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
