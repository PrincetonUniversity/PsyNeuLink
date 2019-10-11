import numpy as np
import pytest

from psyneulink.core.components.functions.statefulfunctions.memoryfunctions import ContentAddressableMemory
from psyneulink.library.components.mechanisms.processing.integrator.episodicmemorymechanism import EpisodicMemoryMechanism
import psyneulink.core.llvm as pnlvm

np.random.seed(0)
CONTENT_SIZE=10
ASSOC_SIZE=10
test_var = np.random.rand(2, CONTENT_SIZE)
test_initializer = {tuple(test_var[0]): test_var[1]}


test_data = [
    (test_var, ContentAddressableMemory, {'default_variable':test_var}, [[
       0.5488135039273248, 0.7151893663724195, 0.6027633760716439, 0.5448831829968969, 0.4236547993389047, 0.6458941130666561, 0.4375872112626925, 0.8917730007820798, 0.9636627605010293, 0.3834415188257777], [
       0.7917250380826646, 0.5288949197529045, 0.5680445610939323, 0.925596638292661, 0.07103605819788694, 0.08712929970154071, 0.02021839744032572, 0.832619845547938, 0.7781567509498505, 0.8700121482468192 ]]),
    (test_var, ContentAddressableMemory, {'default_variable':test_var, 'retrieval_prob':0.5},
       [[ 0. for i in range(CONTENT_SIZE) ],[ 0. for i in range(ASSOC_SIZE) ]]),
    (test_var, ContentAddressableMemory, {'default_variable':test_var, 'storage_prob':0.1},
       [[ 0. for i in range(CONTENT_SIZE) ],[ 0. for i in range(ASSOC_SIZE) ]]),
    (test_var, ContentAddressableMemory, {'default_variable':test_var, 'retrieval_prob':0.9, 'storage_prob':0.9}, [[
       0.5488135039273248, 0.7151893663724195, 0.6027633760716439, 0.5448831829968969, 0.4236547993389047, 0.6458941130666561, 0.4375872112626925, 0.8917730007820798, 0.9636627605010293, 0.3834415188257777], [
       0.7917250380826646, 0.5288949197529045, 0.5680445610939323, 0.925596638292661, 0.07103605819788694, 0.08712929970154071, 0.02021839744032572, 0.832619845547938, 0.7781567509498505, 0.8700121482468192 ]]),
]

# use list, naming function produces ugly names
names = [
    "ContentAddressableMemory",
    "ContentAddressableMemory Random Retrieval",
    "ContentAddressableMemory Random Storage",
    "ContentAddressableMemory Random Retrieval-Storage",
]

@pytest.mark.function
@pytest.mark.memory_function
@pytest.mark.parametrize("variable, func, params, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_basic(variable, func, params, expected, benchmark):
    f= func(seed=0, **params)
    m = EpisodicMemoryMechanism(content_size=len(variable[0]), assoc_size=len(variable[1]), function=f)
    m.execute(variable)
    m.execute(variable)
    res = [s.value for s in m.output_ports]
    assert np.allclose(res[0], expected[0])
    assert np.allclose(res[1], expected[1])
    benchmark(m.execute, variable)


@pytest.mark.llvm
@pytest.mark.function
@pytest.mark.memory_function
@pytest.mark.parametrize("variable, func, params, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_llvm(variable, func, params, expected, benchmark):
    f= func(seed=0, **params)
    m = EpisodicMemoryMechanism(content_size=len(variable[0]), assoc_size=len(variable[1]), function=f)
    e = pnlvm.execution.MechExecution(m)
    e.execute(variable)
    res = e.execute(variable)
    assert np.allclose(res[0], expected[0])
    assert np.allclose(res[1], expected[1])
    benchmark(e.execute, variable)


@pytest.mark.llvm
@pytest.mark.cuda
@pytest.mark.function
@pytest.mark.memory_function
@pytest.mark.parametrize("variable, func, params, expected", test_data, ids=names)
@pytest.mark.benchmark
def test_ptx_cuda(variable, func, params, expected, benchmark):
    f= func(seed=0, **params)
    m = EpisodicMemoryMechanism(content_size=len(variable[0]), assoc_size=len(variable[1]), function=f)
    e = pnlvm.execution.MechExecution(m)
    e.cuda_execute(variable)
    res = e.cuda_execute(variable)
    assert np.allclose(res[0], expected[0])
    assert np.allclose(res[1], expected[1])
    benchmark(e.cuda_execute, variable)
