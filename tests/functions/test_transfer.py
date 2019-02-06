
import numpy as np
import psyneulink.core.llvm as pnlvm
import psyneulink.core.components.functions.transferfunctions as Functions
import psyneulink.core.globals.keywords as kw
import pytest

SIZE=500
test_var = np.random.rand(SIZE)
test_matrix = np.random.rand(SIZE, SIZE)
test_matrix_s = np.random.rand(SIZE, SIZE // 4)
test_matrix_l = np.random.rand(SIZE, 3 * SIZE)

RAND1 = np.random.rand()
RAND2 = np.random.rand()
RAND3 = np.random.rand()

softmax_helper = RAND1 * test_var
softmax_helper = softmax_helper - np.max(softmax_helper)
softmax_helper = np.exp(softmax_helper) / np.sum(np.exp(softmax_helper))

test_data = [
    (Functions.Linear, test_var, {'slope':RAND1, 'intercept':RAND2}, None, test_var * RAND1 + RAND2),
    (Functions.Exponential, test_var, {'scale':RAND1, 'rate':RAND2}, None, RAND1 * np.exp(RAND2 * test_var)),
    (Functions.Logistic, test_var, {'gain':RAND1, 'x_0':RAND2, 'offset':RAND3}, None, 1 / (1 + np.exp(-(RAND1 * (test_var - RAND2)) + RAND3))),
    (Functions.ReLU, test_var, {'gain':RAND1, 'bias':RAND2, 'leak':RAND3}, None, np.maximum(RAND1 * (test_var - RAND2), RAND2, RAND3 * (test_var - RAND2))),
    (Functions.SoftMax, test_var, {'gain':RAND1, 'per_item': False}, None, softmax_helper),
    (Functions.SoftMax, test_var, {'gain':RAND1, 'params':{kw.OUTPUT_TYPE:kw.MAX_VAL}, 'per_item': False}, None, np.where(softmax_helper == np.max(softmax_helper), np.max(softmax_helper), 0)),
    (Functions.SoftMax, test_var, {'gain':RAND1, 'params':{kw.OUTPUT_TYPE:kw.MAX_INDICATOR}, 'per_item': False}, None, np.where(softmax_helper == np.max(softmax_helper), 1, 0)),
    ### Skip probabilistic since it has no-deterministic result ###
    (Functions.LinearMatrix, test_var.tolist(), {'matrix':test_matrix.tolist()}, None, np.dot(test_var, test_matrix)),
    (Functions.LinearMatrix, test_var.tolist(), {'matrix':test_matrix_l.tolist()}, None, np.dot(test_var, test_matrix_l)),
    (Functions.LinearMatrix, test_var.tolist(), {'matrix':test_matrix_s.tolist()}, None, np.dot(test_var, test_matrix_s)),
]

# use list, naming function produces ugly names
names = [
    "LINEAR",
    "EXPONENTIAL",
    "LOGISTIC",
    "RELU",
    "SOFT_MAX ALL",
    "SOFT_MAX MAX_VAL",
    "SOFT_MAX MAX_INDICATOR",
    "LINEAR_MATRIX SQUARE",
    "LINEAR_MATRIX WIDE",
    "LINEAR_MATRIX TALL",
]

@pytest.mark.function
@pytest.mark.transfer_function
@pytest.mark.parametrize("func, variable, params, fail, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_basic(func, variable, params, fail, expected, benchmark):
    f = func(default_variable=variable, **params)
    benchmark.group = "TransferFunction " + func.componentName;
    res = benchmark(f.function, variable)
    assert np.allclose(res, expected)


@pytest.mark.llvm
@pytest.mark.function
@pytest.mark.transfer_function
@pytest.mark.parametrize("func, variable, params, fail, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_llvm(func, variable, params, fail, expected, benchmark):
    f = func(default_variable=variable, **params)
    benchmark.group = "TransferFunction " + func.componentName;
    m = pnlvm.execution.FuncExecution(f)
    res = benchmark(m.execute, variable)
    assert np.allclose(res, expected)

@pytest.mark.llvm
@pytest.mark.cuda
@pytest.mark.function
@pytest.mark.transfer_function
@pytest.mark.parametrize("func, variable, params, fail, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_ptx_cuda(func, variable, params, fail, expected, benchmark):
    f = func(default_variable=variable, **params)
    benchmark.group = "TransferFunction " + func.componentName;
    m = pnlvm.execution.FuncExecution(f)
    res = benchmark(m.execute, variable)
    assert np.allclose(res, expected)
