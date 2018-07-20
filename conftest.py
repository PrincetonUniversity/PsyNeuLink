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
    import doctest

    for m in marks_default_skip:
        if getattr(item.obj, m, None) and not item.config.getvalue(m):
            pytest.skip('{0} tests not requested'.format(m))

    doctest.ELLIPSIS_MARKER = "[...]"


def pytest_runtest_call(item):
    # seed = int(item.config.getoption('--pnl-seed'))
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

def pytest_runtest_teardown(item):
    from psyneulink import clear_registry
    from psyneulink.components.functions.function import FunctionRegistry
    from psyneulink.components.mechanisms.adaptive.control.controlmechanism import ControlMechanismRegistry
    from psyneulink.components.mechanisms.adaptive.gating.gatingmechanism import GatingMechanismRegistry
    from psyneulink.components.mechanisms.mechanism import MechanismRegistry
    from psyneulink.components.projections.projection import ProjectionRegistry
    from psyneulink.components.states.state import StateRegistry
    from psyneulink.components.system import SystemRegistry
    from psyneulink.components.component import DeferredInitRegistry
    from psyneulink.components.process import ProcessRegistry
    from psyneulink.globals.preferences.preferenceset import PreferenceSetRegistry

    # Clear Registry to have a stable reference for indexed suffixes of default names
    clear_registry(FunctionRegistry)
    clear_registry(ControlMechanismRegistry)
    clear_registry(GatingMechanismRegistry)
    clear_registry(MechanismRegistry)
    clear_registry(ProjectionRegistry)
    clear_registry(StateRegistry)
    clear_registry(SystemRegistry)
    clear_registry(DeferredInitRegistry)
    clear_registry(ProcessRegistry)
    clear_registry(PreferenceSetRegistry)


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
