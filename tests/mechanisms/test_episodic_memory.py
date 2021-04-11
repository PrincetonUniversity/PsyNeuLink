import numpy as np
import pytest

from psyneulink.core.components.functions.statefulfunctions.memoryfunctions import DictionaryMemory, ContentAddressableMemory
from psyneulink.library.components.mechanisms.processing.integrator.episodicmemorymechanism import EpisodicMemoryMechanism
import psyneulink.core.llvm as pnlvm

np.random.seed(0)

# TEST WITH DictionaryMemory ****************************************************************************************

CONTENT_SIZE=10
ASSOC_SIZE=10
test_var = np.random.rand(2, CONTENT_SIZE)
test_initializer = {tuple(test_var[0]): test_var[1]}

test_data = [
    (test_var, DictionaryMemory, {'default_variable':test_var}, [[
       0.5488135039273248, 0.7151893663724195, 0.6027633760716439, 0.5448831829968969, 0.4236547993389047,
        0.6458941130666561, 0.4375872112626925, 0.8917730007820798, 0.9636627605010293, 0.3834415188257777],
        [0.7917250380826646, 0.5288949197529045, 0.5680445610939323, 0.925596638292661, 0.07103605819788694,
         0.08712929970154071, 0.02021839744032572, 0.832619845547938, 0.7781567509498505, 0.8700121482468192 ]]),
    (test_var, DictionaryMemory, {'default_variable':test_var, 'retrieval_prob':0.5},
     [[ 0. for i in range(CONTENT_SIZE) ],[ 0. for i in range(ASSOC_SIZE) ]]),
    (test_var, DictionaryMemory, {'default_variable':test_var, 'storage_prob':0.1},
     [[ 0. for i in range(CONTENT_SIZE) ],[ 0. for i in range(ASSOC_SIZE) ]]),
    (test_var, DictionaryMemory, {'default_variable':test_var, 'retrieval_prob':0.9, 'storage_prob':0.9}, [[
       0.5488135039273248, 0.7151893663724195, 0.6027633760716439, 0.5448831829968969, 0.4236547993389047,
        0.6458941130666561, 0.4375872112626925, 0.8917730007820798, 0.9636627605010293, 0.3834415188257777],
        [0.7917250380826646, 0.5288949197529045, 0.5680445610939323, 0.925596638292661, 0.07103605819788694,
         0.08712929970154071, 0.02021839744032572, 0.832619845547938, 0.7781567509498505, 0.8700121482468192 ]]),
]

# use list, naming function produces ugly names
names = [
    "DictionaryMemory",
    "DictionaryMemory Random Retrieval",
    "DictionaryMemory Random Storage",
    "DictionaryMemory Random Retrieval-Storage",
]

# Test using DictionaryMemory
@pytest.mark.function
@pytest.mark.memory_function
@pytest.mark.benchmark
@pytest.mark.parametrize('variable, func, params, expected', test_data, ids=names)
def test_with_dictionary_memory(variable, func, params, expected, benchmark, mech_mode):
    f = func(seed=0, **params)
    m = EpisodicMemoryMechanism(content_size=len(variable[0]), assoc_size=len(variable[1]), function=f)
    if mech_mode == 'Python':
        def EX(variable):
            m.execute(variable)
            return m.output_values
    elif mech_mode == 'LLVM':
        EX = pnlvm.execution.MechExecution(m).execute
    elif mech_mode == 'PTX':
        EX = pnlvm.execution.MechExecution(m).cuda_execute

    EX(variable)
    res = EX(variable)
    assert np.allclose(res[0], expected[0])
    assert np.allclose(res[1], expected[1])
    if benchmark.enabled:
        benchmark(EX, variable)

# TEST WITH ContentAddressableMemory ***********************************************************************************

# Note:  ContentAddressableMemory has not yet been compiled for use with LLVM

# use list, naming function produces ugly names

SIZE = 3

test_data = [
    (
        # name
        "ContentAddressableMemory Default Variable Init",
        # func
        DictionaryMemory,
        # func_params
        {'default_variable':np.random.rand(2, SIZE)},
        # mech_params
        {},
        # test_var
        np.random.rand(2, SIZE),
        # expected
        # FIX:
        [[0.5488135039273248, 0.7151893663724195, 0.6027633760716439, 0.5448831829968969, 0.4236547993389047,
          0.6458941130666561, 0.4375872112626925, 0.8917730007820798, 0.9636627605010293, 0.3834415188257777],
         [0.7917250380826646, 0.5288949197529045, 0.5680445610939323, 0.925596638292661, 0.07103605819788694,
          0.08712929970154071, 0.02021839744032572, 0.832619845547938, 0.7781567509498505, 0.8700121482468192 ]]
    ),
    (
        "ContentAddressableMemory Size Init",
        DictionaryMemory,
        {'size':[SIZE]},
        {},
        np.random.rand(3),
        # FIX:
        [[0.5488135039273248, 0.7151893663724195, 0.6027633760716439, 0.5448831829968969, 0.4236547993389047,
          0.6458941130666561, 0.4375872112626925, 0.8917730007820798, 0.9636627605010293, 0.3834415188257777],
         [0.7917250380826646, 0.5288949197529045, 0.5680445610939323, 0.925596638292661, 0.07103605819788694,
          0.08712929970154071, 0.02021839744032572, 0.832619845547938, 0.7781567509498505, 0.8700121482468192 ]]
    ),
    (
        "ContentAddressableMemory InputPort Names",
        # FIX:
        # OTHER DATA
    ),
    (
        "ContentAddressableMemory OutputPort Names",
        # FIX:
        # OTHER DATA
    ),
    (
        "ContentAddressableMemory OutputPort Names",
        # FIX:
        # OTHER DATA
    ),
    (
        "ContentAddressableMemory Initializer Regular Fields",
        # FIX:
        # OTHER DATA
        # np.array([[np.array([1,1,1]), np.array([2,1, 3]), np.array([4, 5, 6])],
        # #        # [list([10]), list([20, 30]), list([40, 50, 60])],
        # #        [np.array([11,1,1]), np.array([22, 33,1]), np.array([44, 55, 66])]])
    ),
    (
        "ContentAddressableMemory Initializer Ragged Fields",
        # FIX:
        # OTHER DATA
        # np.array([[np.array([1]), np.array([2, 3]), np.array([4, 5, 6])],
        #        [list([10]), list([20, 30]), list([40, 50, 60])],
        #        [np.array([11]), np.array([22, 33]), np.array([44, 55, 66])]])
        #
    ),
    (
        "ContentAddressableMemory Initializer Ndimensional Fields",
        # FIX:
        # OTHER DATA
        [[[1],[[2],[3,4]],[4]],[[1],[[2],[3,4]],[4]]]
        [[[1,2,3],[4]],[[1],[[2],[3,4]],[4]]]
    ),
    (
        "ContentAddressableMemory Random Retrieval",
        # FIX:
        # OTHER DATA
    ),
    (
        "ContentAddressableMemory Random Storage",
        # FIX:
        # OTHER DATA
    ),
    (
        "ContentAddressableMemory Random Retrieval-Storage",
        # FIX:
        # OTHER DATA
    ),
    (
        "ContentAddressableMemory Weighted Retrieval",
        # FIX:
        # OTHER DATA
    ),
    (
        "ContentAddressableMemory Duplicates Retrieval",
        # FIX:
        # OTHER DATA
    ),
]

# Allows names to be with each test_data set
names = [test_data[i][0] for i in range(len(test_data))]

@pytest.mark.parametrize('name, func, func_params, mech_params, test_var, expected', test_data, ids=names)
def test_with_contentaddressablememory(name, func, func_params, mech_params, expected, mech_mode):
    f = func(seed=0, **func_params)
    EpisodicMemoryMechanism(function=f, **mech_params)
    em = EpisodicMemoryMechanism(**mech_params)
    if mech_mode == 'Python':
        def EX(variable):
            em.execute(variable)
            return em.output_values
    elif mech_mode == 'LLVM':
        pass
    elif mech_mode == 'PTX':
        pass

    EX(test_var)
    res = EX(test_var)
    assert np.allclose(res, expected)

# TEST FAILURE:
[ [[1,2,3], [4]], [[1,2,3], [[1],[4]]] ]
