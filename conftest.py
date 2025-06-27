import contextlib
import doctest
import inspect
import io
import itertools
import numpy as np
import pytest
import re
import types

import graph_scheduler as gs
import psyneulink
from psyneulink import clear_registry, primary_registries, torch_available
from psyneulink.core import llvm as pnlvm
from psyneulink.core.globals.utilities import is_numeric, set_global_seed

try:
    import torch
except ImportError:
    pass
else:
    # Check that torch is usable if installed
    assert torch_available, "Torch module is available, but not usable by PNL"

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

    parser.addoption('--fp-precision', action='store', default='fp64', choices=['fp32', 'fp64'],
                     help='Set default fp precision for the runtime compiler. Default: fp64')

def pytest_runtest_setup(item):
    # Check that all 'cuda' tests are also marked 'llvm'
    assert 'llvm' in item.keywords or 'cuda' not in item.keywords

    # It the item is a parametrized function. It has a 'callspec' attribute.
    # Convert any dict arguments to an unmutable MappingProxyType.
    if hasattr(item, 'callspec'):
        for k, v in item.callspec.params.items():
            if isinstance(v, dict):
                item.callspec.params[k] = types.MappingProxyType(v)

    for m in marks_default_skip:
        if m in item.keywords and not item.config.getvalue(m):
            pytest.skip('{0} tests not requested'.format(m))

    if 'llvm' in item.keywords and 'llvm_not_implemented' in item.keywords:
        pytest.skip('LLVM implementation not available')

    if 'cuda' in item.keywords and not pnlvm.ptx_enabled:
        pytest.skip('PTX engine not enabled/available')

    if 'pytorch' in item.keywords and not torch_available:
        pytest.skip('pytorch not available')

    doctest.ELLIPSIS_MARKER = "[...]"

def pytest_generate_tests(metafunc):
    mech_and_func_modes = ['Python',
                           pytest.param('LLVM', marks=pytest.mark.llvm),
                           pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])
                          ]

    if "func_mode" in metafunc.fixturenames:
        metafunc.parametrize("func_mode", mech_and_func_modes)

    if "mech_mode" in metafunc.fixturenames:
        metafunc.parametrize("mech_mode", mech_and_func_modes)

    if "comp_mode_no_per_node" in metafunc.fixturenames:
        modes = [m for m in get_comp_execution_modes()
                 if m.values[0] is not pnlvm.ExecutionMode._LLVMPerNode]
        metafunc.parametrize("comp_mode", modes)

    elif "comp_mode" in metafunc.fixturenames:
        metafunc.parametrize("comp_mode", get_comp_execution_modes())

    if "autodiff_mode" in metafunc.fixturenames:
        auto_modes = [
            # pnlvm.ExecutionMode.Python,
            pytest.param(pnlvm.ExecutionMode.PyTorch, marks=pytest.mark.pytorch),
            pytest.param(pnlvm.ExecutionMode.LLVMRun, marks=pytest.mark.llvm)
        ]
        metafunc.parametrize("autodiff_mode", auto_modes)


_old_register_prefix = None

# Collection hooks
def pytest_sessionstart(session):
    """Initialize session with the right floating point precision and component name prefix."""

    precision = session.config.getvalue("--fp-precision")
    if precision == 'fp64':
        pnlvm.LLVMBuilderContext.default_float_ty = pnlvm.ir.DoubleType()
    elif precision == 'fp32':
        pnlvm.LLVMBuilderContext.default_float_ty = pnlvm.ir.FloatType()
    else:
        assert False, "Unsupported precision parameter: {}".format(precision)

    global _old_register_prefix
    _old_register_prefix = psyneulink.core.globals.registry._register_auto_name_prefix
    psyneulink.core.globals.registry._register_auto_name_prefix = "__pnl_pytest_"

def pytest_collection_finish(session):
    """Restore component prefix at the end of test collection."""
    psyneulink.core.globals.registry._register_auto_name_prefix = _old_register_prefix

# Runtest hooks
def pytest_runtest_call(item):
    # seed = int(item.config.getoption('--pnl-seed'))
    seed = 0
    np.random.seed(seed)
    set_global_seed(seed)

    if 'pytorch' in item.keywords:
        assert torch_available
        torch.manual_seed(seed)


def pytest_runtest_teardown(item):
    for registry in primary_registries:
        # Clear Registry to have a stable reference for indexed suffixes of default names
        clear_registry(registry)

    gs.utilities.cached_hashable_graph_function.cache_clear()

    # Skip running the leak checker if the test is marked xfail.
    # XFAIL tests catch exceptions that references call frames
    # including PNL objects that would be reported as leaks.
    # Hopefully, there are no leaky codepaths that are only hit
    # in xfail tests.
    # The same applies to test failures
    skip_cleanup_check = ("xfail" in item.keywords) or item.session.testsfailed > 0

    # Only run the llvm leak checker on llvm tests
    pnlvm.cleanup("llvm" in item.keywords and not skip_cleanup_check)

@pytest.fixture
def comp_mode_no_per_node():
    # dummy fixture to allow 'comp_mode' filtering
    pass

@pytest.fixture
def benchmark(benchmark):

    orig_class = type(benchmark)

    class _FirstBench(orig_class):
        def __call__(self, f, *args, **kwargs):
            res = []
            # Compute the first result if benchmark is enabled
            if self.enabled:
                res.append(f(*args, **kwargs))

            res.append(orig_class.__call__(self, f, *args, **kwargs))
            return res[0]

    benchmark.__class__ = _FirstBench

    return benchmark

@pytest.helpers.register
def llvm_current_fp_precision():
    float_ty = pnlvm.LLVMBuilderContext.get_current().float_ty
    if float_ty == pnlvm.ir.DoubleType():
        return 'fp64'
    elif float_ty == pnlvm.ir.FloatType():
        return 'fp32'
    else:
        assert False, "Unknown floating point type: {}".format(float_ty)

@pytest.helpers.register
def get_comp_execution_modes():
    return [pytest.param(pnlvm.ExecutionMode.Python),
            pytest.param(pnlvm.ExecutionMode._LLVMPerNode, marks=pytest.mark.llvm),
            pytest.param(pnlvm.ExecutionMode._LLVMExec, marks=pytest.mark.llvm),
            pytest.param(pnlvm.ExecutionMode.LLVMRun, marks=pytest.mark.llvm),
            pytest.param(pnlvm.ExecutionMode.PTXRun, marks=[pytest.mark.llvm,  pytest.mark.cuda])
           ]

@pytest.helpers.register
def get_comp_and_ocm_execution_modes():

    # The first part converts composition execution mode to (comp_mod, ocm_mode) pair.
    # All comp_mode-s other than Python set ocm_mode to None, which is invalid and will
    # fail assertion if executed in Python mode, ExecutionMode.Python sets ocm_mode to 'Python'.
    return [pytest.param(x.values[0], 'Python' if x.values[0] is pnlvm.ExecutionMode.Python else 'None', id=str(x.values[0]), marks=x.marks) for x in get_comp_execution_modes()] + \
           [pytest.param(pnlvm.ExecutionMode.Python, 'LLVM', id='Python-LLVM', marks=pytest.mark.llvm),
            pytest.param(pnlvm.ExecutionMode.Python, 'PTX', id='Python-PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])]

@pytest.helpers.register
def cuda_param(val):
    return pytest.param(val, marks=[pytest.mark.llvm, pytest.mark.cuda])

@pytest.helpers.register
def get_func_execution(func, func_mode, *, tags:frozenset=frozenset(), member='function'):
    if func_mode == 'LLVM':
        return pnlvm.execution.FuncExecution(func, tags=tags).execute

    elif func_mode == 'PTX':
        return pnlvm.execution.FuncExecution(func, tags=tags).cuda_execute

    elif func_mode == 'Python':
        return getattr(func, member)
    else:
        assert False, "Unknown function mode: {}".format(func_mode)

@pytest.helpers.register
def get_mech_execution(mech, mech_mode, *, tags:frozenset=frozenset(), member='execute'):
    if mech_mode == 'LLVM':
        return pnlvm.execution.MechExecution(mech, tags=tags).execute

    elif mech_mode == 'PTX':
        return pnlvm.execution.MechExecution(mech, tags=tags).cuda_execute

    elif mech_mode == 'Python':
        def mech_wrapper(x):
            getattr(mech, member)(x)
            return mech.output_values

        return mech_wrapper
    else:
        assert False, "Unknown mechanism mode: {}".format(mech_mode)

@pytest.helpers.register
def numpy_uses_avx512():

    try:
        # numpy >= 1.26 can return config info in a dictionary
        config = np.show_config(mode="dicts")

    except TypeError:
        # Numpy >=1.21 < 1.26 doesn't support 'mode' argument and
        # prints CPU extensions in one line per category:
        # baseline = ...
        # found = ...
        # not found = ...
        out = io.StringIO()

        with contextlib.redirect_stdout(out):
            np.show_config()

        return re.search('  found = .*AVX512.*', out.getvalue()) is not None
    else:
        return any(ext.startswith("AVX512") for ext in config['SIMD Extensions']['found'])

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

@pytest.helpers.register
def power_set(s):
    """Set of all potential subsets."""

    vals = list(s)
    return (c for l in range(len(vals) + 1) for c in itertools.combinations(vals, l))


def patch_parameter_set_value_numeric_check():
    orig_parameter_set_value = psyneulink.core.globals.parameters.Parameter._set_value

    def check_numeric_set_value(self, value, **kwargs):
        assert isinstance(value, np.ndarray) or not is_numeric(value), (
            f'{self._owner._owner}.{self.name} is being set to a numeric value.'
            f' It must first be wrapped in a numpy array:\n\t{value}\n\t{type(value)}'
        )

        return orig_parameter_set_value(self, value, **kwargs)

    psyneulink.core.globals.parameters.Parameter._set_value = check_numeric_set_value


# flag when run from pytest
# https://docs.pytest.org/en/stable/example/simple.html#detect-if-running-from-within-a-pytest-run
def pytest_configure(config):
    psyneulink._called_from_pytest = True

    patch_parameter_set_value_numeric_check()


@pytest.helpers.register
def get_all_subclasses(
    type_=psyneulink.core.components.component.Component,
    module=psyneulink,
    exclude_type=None,
    include_abstract=True,
    sort=True,
):
    classes = []

    for item in module.__all__:
        cls_ = getattr(module, item)

        if (
            inspect.isclass(cls_)
            and issubclass(cls_, type_)
            and (include_abstract or not inspect.isabstract(cls_))
        ):
            if exclude_type is None or not issubclass(cls_, exclude_type):
                classes.append(cls_)

    if sort:
        classes.sort(key=lambda x: x.__name__)

    return classes
