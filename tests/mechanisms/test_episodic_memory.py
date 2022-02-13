import numpy as np
import pytest

import psyneulink.core.llvm as pnlvm
from psyneulink.core.components.functions.function import FunctionError
from psyneulink.core.components.functions.stateful.memoryfunctions import DictionaryMemory, \
    ContentAddressableMemory
from psyneulink.library.components.mechanisms.processing.integrator.episodicmemorymechanism import \
    EpisodicMemoryMechanism, EpisodicMemoryMechanismError

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

@pytest.mark.function
@pytest.mark.memory_function
@pytest.mark.benchmark
@pytest.mark.parametrize('variable, func, params, expected', test_data, ids=names)
def test_with_dictionary_memory(variable, func, params, expected, benchmark, mech_mode):
    f = func(seed=0, **params)
    m = EpisodicMemoryMechanism(content_size=len(variable[0]), assoc_size=len(variable[1]), function=f)
    EX = pytest.helpers.get_mech_execution(m, mech_mode)

    EX(variable)
    res = EX(variable)
    assert np.allclose(res[0], expected[0])
    assert np.allclose(res[1], expected[1])
    if benchmark.enabled:
        benchmark(EX, variable)


# TEST WITH ContentAddressableMemory ***********************************************************************************
# Note:  ContentAddressableMemory has not yet been compiled for use with LLVM or PTX, so those are dummy tests for now
test_data = [
    (
        # name
        "ContentAddressableMemory Default",
        # func
        ContentAddressableMemory,
        # func_params
        {},
        # mech_params
        {},
        # test_var
        [[10.,10.]],
        # expected input_port names
        ['FIELD_0_INPUT'],
        # expected output_port names
        ['RETREIVED_FIELD_0'],
        # expected output
        [[0,0]]
    ),
    (
        # name
        "ContentAddressableMemory Func Default Variable Mech Size Init",
        # func
        ContentAddressableMemory,
        # func_params
        {'default_variable': [[0,0],[0,0],[0,0,0]]},
        # mech_params
        {'size':[2,2,3]},
        # test_var
        [[10.,10.],[20., 30.],[40., 50., 60.]],
        # expected input_port names
        ['FIELD_0_INPUT', 'FIELD_1_INPUT', 'FIELD_2_INPUT'],
        # expected output_port names
        ['RETREIVED_FIELD_0', 'RETREIVED_FIELD_1', 'RETREIVED_FIELD_2'],
        # expected output
        [[0,0],[0,0],[0,0,0]]
    ),
    (
        "ContentAddressableMemory Func Default Variable Mech Default Var Init",
        ContentAddressableMemory,
        {'default_variable': [[0],[0,0],[0,0,0]]},
        {'default_variable': [[0],[0,0],[0,0,0]]},
        [[10.],[20., 30.],[40., 50., 60.]],
        ['FIELD_0_INPUT', 'FIELD_1_INPUT', 'FIELD_2_INPUT'],
        ['RETREIVED_FIELD_0', 'RETREIVED_FIELD_1', 'RETREIVED_FIELD_2'],
        [[0],[0,0],[0,0,0]]
    ),
    (
        "ContentAddressableMemory Func Initializer (ragged) Mech Size Init",
        ContentAddressableMemory,
        {'initializer':np.array([[np.array([1]), np.array([2, 3]), np.array([4, 5, 6])],
                                 [list([10]), list([20, 30]), list([40, 50, 60])],
                                 [np.array([11]), np.array([22, 33]), np.array([44, 55, 66])]], dtype=object)},
        {'size':[1,2,3]},
        [[10.],[20., 30.],[40., 50., 60.]],
        ['FIELD_0_INPUT', 'FIELD_1_INPUT', 'FIELD_2_INPUT'],
        ['RETREIVED_FIELD_0', 'RETREIVED_FIELD_1', 'RETREIVED_FIELD_2'],
        # [[10.],[20., 30.],[40., 50., 60.]]
        [[1], [2,3], [4,5,6]] # <- distance = 0 to [[10.],[20., 30.],[40., 50., 60.]]
    ),
    (
        "ContentAddressableMemory Func Initializer (ragged) Mech Default Variable Init",
        ContentAddressableMemory,
        {'initializer':np.array([[np.array([1]), np.array([2, 3]), np.array([4, 5, 6])],
                                 [[12], [20, 55], [40, 50, 60]],
                                 [np.array([15]), np.array([22, 37]), np.array([44, 55, 66])]], dtype=object)},
        {'default_variable': [[0],[0,0],[0,0,0]], 'input_ports':['hello','world','goodbye']},
        [[10.],[20., 30.],[40., 50., 60.]],
        ['hello', 'world', 'goodbye'],
        ['RETREIVED_hello', 'RETREIVED_world', 'RETREIVED_goodbye'],
        [[1.],[2., 3.],[4., 5., 6.]]
    ),
    (
        "ContentAddressableMemory Func Initializer (regular 2d) Mech Size Init",
        ContentAddressableMemory,
        {'initializer':np.array([[np.array([1,2]), np.array([3,4]), np.array([5, 6])],
                                 [[10,20], [30,40], [50,60]],
                                 [np.array([11,12]), np.array([22, 23]), np.array([34, 35])]])},
        {'size':[2,2,2]},
        [[11,13], [22,23], [34, 35]],
        ['FIELD_0_INPUT', 'FIELD_1_INPUT', 'FIELD_2_INPUT'],
        ['RETREIVED_FIELD_0', 'RETREIVED_FIELD_1', 'RETREIVED_FIELD_2'],
        [[11,12], [22,23], [34, 35]],
    ),
    (
        "ContentAddressableMemory Func Initializer (regular 2d) Mech Default Variable Init",
        ContentAddressableMemory,
        {'initializer':np.array([[np.array([1,2]), np.array([3,4]), np.array([5, 6])],
                                 [[10,20], [30,40], [50,60]],
                                 [np.array([11,12]), np.array([22, 23]), np.array([34, 35])]]),
         'equidistant_entries_select':'newest',
         'duplicate_entries_allowed':True
         },
        {'default_variable':[[0,0],[0,0],[0,0]]},
        [[10,20], [30,40], [50, 60]],
        ['FIELD_0_INPUT', 'FIELD_1_INPUT', 'FIELD_2_INPUT'],
        ['RETREIVED_FIELD_0', 'RETREIVED_FIELD_1', 'RETREIVED_FIELD_2'],
        [[10,20], [30,40], [50, 60]],
    ),
    (
        "ContentAddressableMemory Func Mech default_variable Init",
        ContentAddressableMemory,
        {},
        {'default_variable':[[10,20], [30,40]],
         'input_ports':['FIRST','SECOND']},
        [[10,20], [30,40]],
        ['FIRST', 'SECOND'],
        ['RETREIVED_FIRST', 'RETREIVED_SECOND'],
        [[0,0], [0,0]],
    ),
    (
        "ContentAddressableMemory Func Mech Memory Init",
        ContentAddressableMemory,
        {},
        {'memory':[[[10,20],[30,40]],
                   [[11,12],[22, 23]]],
         'input_ports':['FIRST','SECOND']},
        [[10,20], [30,40]],
        ['FIRST', 'SECOND'],
        ['RETREIVED_FIRST', 'RETREIVED_SECOND'],
        [[10,20], [30,40]],
    ),
    (
        "ContentAddressableMemory Func Mech Memory Init Enforce Shape",
        ContentAddressableMemory,
        {},
        {'memory':[[11,12],[22, 23]], # <- memory incorrect shape, but should be cast by function._enforce_memory_shape
         'input_ports':['FIRST','SECOND']},
        [[10,20], [30,40]],
        ['FIRST', 'SECOND'],
        ['RETREIVED_FIRST', 'RETREIVED_SECOND'],
        [[11,12],[22, 23]],
    )
]

# Allows names to be with each test_data set
names = [test_data[i][0] for i in range(len(test_data))]

@pytest.mark.parametrize('name, func, func_params, mech_params, test_var,'
                         'input_port_names, output_port_names, expected_output', test_data, ids=names)
def test_with_contentaddressablememory(name, func, func_params, mech_params, test_var,
                                       input_port_names, output_port_names, expected_output, mech_mode):
    f = func(seed=0, **func_params)
    # EpisodicMemoryMechanism(function=f, **mech_params)
    em = EpisodicMemoryMechanism(function=f, **mech_params)
    assert em.input_ports.names == input_port_names
    assert em.output_ports.names == output_port_names

    if mech_mode != 'Python':
        pytest.skip("PTX not yet implemented for ContentAddressableMemory")

    EX = pytest.helpers.get_mech_execution(em, mech_mode)


    # EX(test_var)
    actual_output = EX(test_var)
    for i,j in zip(actual_output,expected_output):
        np.testing.assert_allclose(i, j, atol=1e-08)

def test_contentaddressable_memory_warnings_and_errors():

    # both memory arg of Mechanism and initializer for its function are specified
    text = "The 'memory' argument specified for EpisodicMemoryMechanism-0 will override the specification " \
            "for the 'initializer' argument of its function"
    with pytest.warns(UserWarning, match=text):
        em = EpisodicMemoryMechanism(
            memory = [[[1,2,3],[4,5,6]]],
            function=ContentAddressableMemory(initializer = [[[10,10,10],[4,5,6]]])
        )

    # default_value doesn't match shape of initializer for function
    with pytest.raises(EpisodicMemoryMechanismError) as error_text:
        em = EpisodicMemoryMechanism(default_variable = [[1,2,3],[4,5,6],[7,8,9]],
                                     function=ContentAddressableMemory(initializer=[[[10,10,10],[4,5,6]]])
                                     )
    assert "Shape of 'variable' for EpisodicMemoryMechanism-1 ((3, 3)) does not match the shape of entries ((2, 3)) " \
           "in the memory of its function" in str(error_text.value)

    # default_value doesn't match shape of entry in memory arg
    with pytest.raises(EpisodicMemoryMechanismError) as error_text:
        em = EpisodicMemoryMechanism(default_variable = [[1,2,3],[4,5,6]],
                                     memory = [[[1,2,3],[4,5,6],[7,8,9]]],
                                     function=ContentAddressableMemory(initializer=[[[10,10,10],[4,5,6]]])
                                     )
    assert "Shape of 'variable' for EpisodicMemoryMechanism-2 ((2, 3)) does not match the " \
           "shape of entries ((3, 3)) in specification of its 'memory' argument" in str(error_text.value)

    # Initializer with >2d regular array
    with pytest.raises(FunctionError) as error_text:
        f = ContentAddressableMemory(initializer=[[[[1],[0],[1]], [[1],[0],[0]], [[0],[1],[1]]],
                                                  [[[0],[1],[0]], [[0],[1],[1]], [[1],[1],[0]]]])
        em = EpisodicMemoryMechanism(size = [1,1,1], function=f)
        em.execute([[[0],[1],[0]], [[0],[1],[1]], [[1],[1],[0]]])
    assert 'Attempt to store and/or retrieve an entry in ContentAddressableMemory ' \
           '([[[1]\n  [0]\n  [1]]\n\n [[1]\n  [0]\n  [0]]\n\n [[0]\n  [1]\n  [1]]]) ' \
           'that has more than 2 dimensions (3);  try flattening innermost ones.' in str(error_text.value)
