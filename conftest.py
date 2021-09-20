import doctest
import psyneulink
import pytest
import numpy as np


from psyneulink import clear_registry, primary_registries
from psyneulink.core import llvm as pnlvm
from psyneulink.core.globals.utilities import set_global_seed


try:
    import torch
    pytorch_available = True
except ImportError:
    pytorch_available = False

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
        if m in item.keywords and not item.config.getvalue(m):
            pytest.skip('{0} tests not requested'.format(m))

    if 'cuda' in item.keywords and not pnlvm.ptx_enabled:
            pytest.skip('PTX engine not enabled/available')

    if 'pytorch' in item.keywords and not pytorch_available:
            pytest.skip('pytorch not available')

    doctest.ELLIPSIS_MARKER = "[...]"

def pytest_generate_tests(metafunc):
    mech_and_func_modes = ['Python',
                           pytest.param('LLVM', marks=pytest.mark.llvm),
                           pytest.param('PTX', marks=[pytest.mark.llvm,
                                                      pytest.mark.cuda])
                          ]

    if "func_mode" in metafunc.fixturenames:
        metafunc.parametrize("func_mode", mech_and_func_modes)

    if "mech_mode" in metafunc.fixturenames:
        metafunc.parametrize("mech_mode", mech_and_func_modes)

    if "comp_mode_no_llvm" in metafunc.fixturenames:
        modes = [m for m in get_comp_execution_modes()
                 if m.values[0] is not pnlvm.ExecutionMode.LLVM]
        metafunc.parametrize("comp_mode", modes)

    elif "comp_mode" in metafunc.fixturenames:
        metafunc.parametrize("comp_mode", get_comp_execution_modes())

    if "autodiff_mode" in metafunc.fixturenames:
        auto_modes = [pnlvm.ExecutionMode.Python,
                      pytest.param(pnlvm.ExecutionMode.LLVMRun, marks=pytest.mark.llvm)]
        metafunc.parametrize("autodiff_mode", auto_modes)

def pytest_runtest_call(item):
    # seed = int(item.config.getoption('--pnl-seed'))
    seed = 0
    np.random.seed(seed)
    set_global_seed(seed)

    if 'pytorch' in item.keywords:
        assert pytorch_available
        torch.manual_seed(seed)


def pytest_runtest_teardown(item):
    for registry in primary_registries:
        # Clear Registry to have a stable reference for indexed suffixes of default names
        clear_registry(registry)

    pnlvm.cleanup()

@pytest.fixture
def comp_mode_no_llvm():
    # dummy fixture to allow 'comp_mode' filtering
    pass

@pytest.helpers.register
def get_comp_execution_modes():
    return [pytest.param(pnlvm.ExecutionMode.Python),
            pytest.param(pnlvm.ExecutionMode.LLVM, marks=pytest.mark.llvm),
            pytest.param(pnlvm.ExecutionMode.LLVMExec, marks=pytest.mark.llvm),
            pytest.param(pnlvm.ExecutionMode.LLVMRun, marks=pytest.mark.llvm),
            pytest.param(pnlvm.ExecutionMode.PTXExec, marks=[pytest.mark.llvm, pytest.mark.cuda]),
            pytest.param(pnlvm.ExecutionMode.PTXRun, marks=[pytest.mark.llvm,  pytest.mark.cuda])
           ]

@pytest.helpers.register
def cuda_param(val):
    return pytest.param(val, marks=[pytest.mark.llvm, pytest.mark.cuda])

@pytest.helpers.register
def get_func_execution(func, func_mode):
    if func_mode == 'LLVM':
        return pnlvm.execution.FuncExecution(func).execute
    elif func_mode == 'PTX':
        return pnlvm.execution.FuncExecution(func).cuda_execute
    elif func_mode == 'Python':
        return func.function
    else:
        assert False, "Unknown function mode: {}".format(func_mode)

@pytest.helpers.register
def get_mech_execution(mech, mech_mode):
    if mech_mode == 'LLVM':
        return pnlvm.execution.MechExecution(mech).execute
    elif mech_mode == 'PTX':
        return pnlvm.execution.MechExecution(mech).cuda_execute
    elif mech_mode == 'Python':
        def mech_wrapper(x):
            mech.execute(x)
            return mech.output_values
        return mech_wrapper
    else:
        assert False, "Unknown mechanism mode: {}".format(mech_mode)

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


# flag when run from pytest
# https://docs.pytest.org/en/stable/example/simple.html#detect-if-running-from-within-a-pytest-run
def pytest_configure(config):
    psyneulink._called_from_pytest = True
