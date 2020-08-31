import numpy as np
import pytest

import psyneulink.core.components.functions.statefulfunctions.memoryfunctions as Functions
import psyneulink.core.llvm as pnlvm
from psyneulink import *

np.random.seed(0)
SIZE=10
test_var = np.random.rand(2, SIZE)
test_initializer = np.array([[test_var[0], test_var[1]]])
test_noise_arr = np.random.rand(SIZE)

RAND1 = np.random.random(1)
RAND2 = np.random.random()

test_data = [
# Default initializer does not work
#    (Functions.Buffer, test_var, {'rate':RAND1}, [[0.0],[0.0]]),
    (Functions.Buffer, test_var[0], {'history':512, 'rate':RAND1, 'initializer':[test_var[0]]}, [[0.03841128, 0.05005587, 0.04218721, 0.0381362 , 0.02965146, 0.04520592, 0.03062659, 0.0624149 , 0.06744644, 0.02683695],[0.14519169, 0.18920736, 0.15946443, 0.1441519 , 0.11208025, 0.17087491, 0.11576615, 0.23592355, 0.25494239, 0.10144161]]),
    (Functions.ContentAddressableMemory, test_var, {'rate':RAND1}, [[
       0.5488135039273248, 0.7151893663724195, 0.6027633760716439, 0.5448831829968969, 0.4236547993389047, 0.6458941130666561, 0.4375872112626925, 0.8917730007820798, 0.9636627605010293, 0.3834415188257777], [
       0.7917250380826646, 0.5288949197529045, 0.5680445610939323, 0.925596638292661, 0.07103605819788694, 0.08712929970154071, 0.02021839744032572, 0.832619845547938, 0.7781567509498505, 0.8700121482468192 ]]),
    (Functions.ContentAddressableMemory, test_var, {'rate':RAND1, 'retrieval_prob':0.5},
       [[ 0. for i in range(SIZE) ],[ 0. for i in range(SIZE) ]]),
    (Functions.ContentAddressableMemory, test_var, {'rate':RAND1, 'storage_prob':0.1},
       [[ 0. for i in range(SIZE) ],[ 0. for i in range(SIZE) ]]),
    (Functions.ContentAddressableMemory, test_var, {'rate':RAND1, 'retrieval_prob':0.9, 'storage_prob':0.9}, [[
       0.5488135039273248, 0.7151893663724195, 0.6027633760716439, 0.5448831829968969, 0.4236547993389047, 0.6458941130666561, 0.4375872112626925, 0.8917730007820798, 0.9636627605010293, 0.3834415188257777], [
       0.7917250380826646, 0.5288949197529045, 0.5680445610939323, 0.925596638292661, 0.07103605819788694, 0.08712929970154071, 0.02021839744032572, 0.832619845547938, 0.7781567509498505, 0.8700121482468192 ]]),
    (Functions.ContentAddressableMemory, test_var, {'initializer':test_initializer, 'rate':RAND1}, [[
       0.5488135039273248, 0.7151893663724195, 0.6027633760716439, 0.5448831829968969, 0.4236547993389047, 0.6458941130666561, 0.4375872112626925, 0.8917730007820798, 0.9636627605010293, 0.3834415188257777], [
       0.7917250380826646, 0.5288949197529045, 0.5680445610939323, 0.925596638292661, 0.07103605819788694, 0.08712929970154071, 0.02021839744032572, 0.832619845547938, 0.7781567509498505, 0.8700121482468192 ]]),
# Disable noise tests for now as they trigger failure in ContentAddressableMemory lookup
#    (Functions.ContentAddressableMemory, test_var, {'rate':RAND1, 'noise':RAND2}, [[
#       0.79172504, 0.52889492, 0.56804456, 0.92559664, 0.07103606, 0.0871293 , 0.0202184 , 0.83261985, 0.77815675, 0.87001215 ],[
#       1.3230471933615413, 1.4894230558066361, 1.3769970655058605, 1.3191168724311135, 1.1978884887731214, 1.4201278025008728, 1.2118209006969092, 1.6660066902162964, 1.737896449935246, 1.1576752082599944
#]]),
#    (Functions.ContentAddressableMemory, test_var, {'rate':RAND1, 'noise':[RAND2], 'retrieval_prob':0.5},
#       [[ 0. for i in range(SIZE) ],[ 0. for i in range(SIZE) ]]),
#    (Functions.ContentAddressableMemory, test_var, {'rate':RAND1, 'noise':RAND2, 'storage_prob':0.5},
#       [[ 0. for i in range(SIZE) ],[ 0. for i in range(SIZE) ]]),
#    (Functions.ContentAddressableMemory, test_var, {'initializer':test_initializer, 'rate':RAND1, 'noise':RAND2}, [[
#       0.79172504, 0.52889492, 0.56804456, 0.92559664, 0.07103606, 0.0871293 , 0.0202184 , 0.83261985, 0.77815675, 0.87001215 ],[
#       1.3230471933615413, 1.4894230558066361, 1.3769970655058605, 1.3191168724311135, 1.1978884887731214, 1.4201278025008728, 1.2118209006969092, 1.6660066902162964, 1.737896449935246, 1.1576752082599944
#]]),
]

# use list, naming function produces ugly names
names = [
    "Buffer",
#    "Buffer Initializer",
    "ContentAddressableMemory",
    "ContentAddressableMemory Random Retrieval",
    "ContentAddressableMemory Random Storage",
    "ContentAddressableMemory Random Retrieval-Storage",
    "ContentAddressableMemory Initializer",
#    "ContentAddressableMemory Noise",
#    "ContentAddressableMemory Noise Random Retrieval",
#    "ContentAddressableMemory Noise Random Storage",
#    "ContentAddressableMemory Initializer Noise",
]

@pytest.mark.function
@pytest.mark.memory_function
@pytest.mark.benchmark
@pytest.mark.parametrize("func, variable, params, expected", test_data, ids=names)
@pytest.mark.parametrize('mode', ['Python',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
def test_basic(func, variable, params, expected, benchmark, mode):
    if func is Functions.Buffer and mode != 'Python':
        pytest.skip("Not implemented")

    f = func(default_variable=variable, **params)
    benchmark.group = func.componentName
    if mode == 'Python':
        EX = f.function
    elif mode == 'LLVM':
        e = pnlvm.execution.FuncExecution(f)
        EX = e.execute
    elif mode == 'PTX':
        e = pnlvm.execution.FuncExecution(f)
        EX = e.cuda_execute
    EX(variable)
    res = EX(variable)
    assert np.allclose(res[0], expected[0])
    assert np.allclose(res[1], expected[1])
    if benchmark.enabled:
        benchmark(f, variable)


# Test of ContentAddressableMemory without LLVM:
def test_ContentAddressableMemory_with_initializer_and_key_size_same_as_val_size():

    stimuli = {'A': [[1,2,3],[4,5,6]],
               'B': [[8,9,10],[11,12,13]],
               'C': [[1,2,3],[11,12,13]],
               'D': [[1,2,3],[21,22,23]],
               'E': [[9,8,4],[11,12,13]],
               'F': [[10,10,30],[40,50,60]],
               }

    em = EpisodicMemoryMechanism(
            content_size=3,
            assoc_size=3,
            function = ContentAddressableMemory(
                    seed=2,
                    initializer=np.array([stimuli['F'], stimuli['F']], dtype=object),
                    duplicate_keys=True,
                    equidistant_keys_select=RANDOM)
    )

    retrieved_keys=[]
    for key in sorted(stimuli.keys()):
        retrieved = [i for i in em.execute(stimuli[key])]
        retrieved_key = [k for k,v in stimuli.items() if v == retrieved] or [None]
        retrieved_keys.append(retrieved_key)
    assert retrieved_keys == [['F'], ['A'], ['A'], ['C'], ['B'], ['F']]

    # Run again to test re-initialization and random retrieval
    em.function.reset(np.array([stimuli['A'], stimuli['F']]))
    retrieved_keys=[]
    for key in sorted(stimuli.keys()):
        retrieved = [i for i in em.execute(stimuli[key])]
        retrieved_key = [k for k,v in stimuli.items() if v == retrieved] or [None]
        retrieved_keys.append(retrieved_key)
    assert retrieved_keys == [['A'], ['A'], ['A'], ['A'], ['B'], ['F']]

    stim = 'C'
    em.function.equidistant_keys_select = OLDEST
    retrieved = [i for i in em.function.get_memory(stimuli[stim][0])]
    retrieved_key = [k for k,v in stimuli.items() if v == retrieved] or [None]
    assert retrieved_key == ['A']

    em.function.equidistant_keys_select = NEWEST
    retrieved = [i for i in em.function.get_memory(stimuli[stim][0])]
    retrieved_key = [k for k,v in stimuli.items() if v == retrieved] or [None]
    assert retrieved_key == ['D']

    # Test that after allowing dups, warning is issued and memory with zeros is returned
    em.function.duplicate_keys = False
    stim = 'A'

    text = r'More than one item matched key \(\[1 2 3\]\) in memory for ContentAddressableMemory'
    with pytest.warns(UserWarning, match=text):
        retrieved = em.execute(stimuli[stim])

    retrieved_key = [k for k,v in stimuli.items() if v==list(retrieved)] or [None]
    assert retrieved_key == [None]
    assert retrieved[0] == [0, 0, 0]
    assert retrieved[1] == [0, 0, 0]

def test_ContentAddressableMemory_with_initializer_and_key_size_diff_from_val_size():

    stimuli = {'A': [[1,2,3],[4,5,6,7]],
               'B': [[8,9,10],[11,12,13,14]],
               'C': [[1,2,3],[11,12,13,14]],
               'D': [[1,2,3],[21,22,23,24]],
               'E': [[9,8,4],[11,12,13,14]],
               'F': [[10,10,30],[40,50,60,70]],
               }

    em = EpisodicMemoryMechanism(
            content_size=3,
            assoc_size=4,
            function = ContentAddressableMemory(
                    initializer=np.array([stimuli['F'], stimuli['F']], dtype=object),
                    duplicate_keys=True,
                    equidistant_keys_select=RANDOM)
    )

    retrieved_keys=[]
    for key in sorted(stimuli.keys()):
        print(key)
        retrieved = [i for i in em.execute(stimuli[key])]
        retrieved_key = [k for k,v in stimuli.items() if v == retrieved] or [None]
        retrieved_keys.append(retrieved_key)
    assert retrieved_keys == [['F'], ['A'], ['A'], ['A'], ['B'], ['F']]

    stim = 'C'
    em.function.equidistant_keys_select = OLDEST
    retrieved = [i for i in em.function.get_memory(stimuli[stim][0])]
    retrieved_key = [k for k,v in stimuli.items() if v == retrieved] or [None]
    assert retrieved_key == ['A']

    em.function.equidistant_keys_select = NEWEST
    retrieved = [i for i in em.function.get_memory(stimuli[stim][0])]
    retrieved_key = [k for k,v in stimuli.items() if v == retrieved] or [None]
    assert retrieved_key == ['D']

    # Test that after allowing dups, warning is issued and memory with zeros is returned
    em.function.duplicate_keys = False
    stim = 'A'

    text = r'More than one item matched key \(\[1 2 3\]\) in memory for ContentAddressableMemory'
    with pytest.warns(UserWarning, match=text):
        retrieved = em.execute(stimuli[stim])

    retrieved_key = [k for k,v in stimuli.items() if v==list(retrieved)] or [None]
    assert retrieved_key == [None]
    assert retrieved[0] == [0, 0, 0]
    assert retrieved[1] == [0, 0, 0, 0]

def test_ContentAddressableMemory_without_initializer_and_key_size_same_as_val_size():

    stimuli = {'A': [[1,2,3],[4,5,6]],
               'B': [[8,9,10],[11,12,13]],
               'C': [[1,2,3],[11,12,13]],
               'D': [[1,2,3],[21,22,23]],
               'E': [[9,8,4],[11,12,13]],
               'F': [[10,10,30],[40,50,60]],
               }

    em = EpisodicMemoryMechanism(
            content_size=3,
            assoc_size=3,
            function = ContentAddressableMemory(
                    duplicate_keys=True,
                    equidistant_keys_select=RANDOM)
    )

    retrieved_keys=[]
    for key in sorted(stimuli.keys()):
        retrieved = [i for i in em.execute(stimuli[key])]
        retrieved_key = [k for k,v in stimuli.items() if v == retrieved] or [None]
        retrieved_keys.append(retrieved_key)
    assert retrieved_keys == [[None], ['A'], ['A'], ['C'], ['B'], ['D']]

    stim = 'C'
    em.function.equidistant_keys_select = OLDEST
    retrieved = [i for i in em.function.get_memory(stimuli[stim][0])]
    retrieved_key = [k for k,v in stimuli.items() if v == retrieved] or [None]
    assert retrieved_key == ['A']

    em.function.equidistant_keys_select = NEWEST
    retrieved = [i for i in em.function.get_memory(stimuli[stim][0])]
    retrieved_key = [k for k,v in stimuli.items() if v == retrieved] or [None]
    assert retrieved_key == ['D']

    # Test that after allowing dups, warning is issued and memory with zeros is returned
    em.function.duplicate_keys = False
    stim = 'A'

    text = r'More than one item matched key \(\[1 2 3\]\) in memory for ContentAddressableMemory'
    with pytest.warns(UserWarning, match=text):
        retrieved = em.execute(stimuli[stim])

    retrieved_key = [k for k,v in stimuli.items() if v==list(retrieved)] or [None]
    assert retrieved_key == [None]
    assert retrieved[0] == [0, 0, 0]
    assert retrieved[1] == [0, 0, 0]

def test_ContentAddressableMemory_without_initializer_and_key_size_diff_from_val_size():

    stimuli = {'A': [[1,2,3],[4,5,6,7]],
               'B': [[8,9,10],[11,12,13,14]],
               'C': [[1,2,3],[11,12,13,14]],
               'D': [[1,2,3],[21,22,23,24]],
               'E': [[9,8,4],[11,12,13,14]],
               'F': [[10,10,30],[40,50,60,70]],
               }

    em = EpisodicMemoryMechanism(
            content_size=3,
            assoc_size=4,
            function = ContentAddressableMemory(
                    duplicate_keys=True,
                    equidistant_keys_select=RANDOM)
    )

    retrieved_keys=[]
    for key in sorted(stimuli.keys()):
        retrieved = [i for i in em.execute(stimuli[key])]
        retrieved_key = [k for k,v in stimuli.items() if v == retrieved] or [None]
        retrieved_keys.append(retrieved_key)
    assert retrieved_keys == [[None], ['A'], ['A'], ['C'], ['B'], ['D']]

    stim = 'C'
    em.function.equidistant_keys_select = OLDEST
    retrieved = [i for i in em.function.get_memory(stimuli[stim][0])]
    retrieved_key = [k for k,v in stimuli.items() if v == retrieved] or [None]
    assert retrieved_key == ['A']

    em.function.equidistant_keys_select = NEWEST
    retrieved = [i for i in em.function.get_memory(stimuli[stim][0])]
    retrieved_key = [k for k,v in stimuli.items() if v == retrieved] or [None]
    assert retrieved_key == ['D']

    # Test that after allowing dups, warning is issued and memory with zeros is returned
    em.function.duplicate_keys = False
    stim = 'A'

    text = r'More than one item matched key \(\[1 2 3\]\) in memory for ContentAddressableMemory'
    with pytest.warns(UserWarning, match=text):
        retrieved = em.execute(stimuli[stim])

    retrieved_key = [k for k,v in stimuli.items() if v==list(retrieved)] or [None]
    assert retrieved_key == [None]
    assert retrieved[0] == [0, 0, 0]
    assert retrieved[1] == [0, 0, 0, 0]


def test_ContentAddressableMemory_without_assoc():

    stimuli = {'A': [[1,2,3]],
               'B': [[8,9,10]],
               'C': [[1,2,3]],
               'D': [[1,2,3]],
               'E': [[9,8,4]],
               'F': [[10,10,30]],
               }

    em = EpisodicMemoryMechanism(
            name='EPISODIC MEMORY MECH',
            content_size=3,
            function = ContentAddressableMemory(
                    # initializer=np.array([stimuli['F'], stimuli['F']], dtype=object),
                    duplicate_keys=True,
                    equidistant_keys_select=RANDOM,
                    retrieval_prob = 1.0
            )
    )

    for key in sorted(stimuli.keys()):
        print(f'\nCurrent memory: \n{em.memory}\n')
        retrieved = [i for i in em.execute(stimuli[key])]
        retrieved_key = [k for k,v in stimuli.items() if v == retrieved] or [None]
        print(f'\nExecuted with stimulus {key}: {stimuli[key]};'
              f'\nRetrieved memory {retrieved_key[0]}: \n\t{retrieved}')

    retrieved_keys=[]
    for key in sorted(stimuli.keys()):
        retrieved = [i for i in em.execute(stimuli[key])]
        retrieved_key = [k for k,v in stimuli.items() if v == retrieved] or [None]
        retrieved_keys.append(retrieved_key)

    assert retrieved_keys == [['A', 'C', 'D'], ['B'], ['A', 'C', 'D'], ['A', 'C', 'D'], ['E'], ['F']]


def test_ContentAddressableMemory_with_duplicate_entry_in_initializer_warning():

    regexp = r'Attempt to initialize memory of ContentAddressableMemory with an entry \([[1 2 3]'
    with pytest.warns(UserWarning, match=regexp):
        em = EpisodicMemoryMechanism(
                name='EPISODIC MEMORY MECH',
                content_size=3,
                assoc_size=3,
                function = ContentAddressableMemory(
                        initializer=np.array([[[1,2,3], [4,5,6]],
                                              [[1,2,3], [7,8,9]]]),
                        duplicate_keys=False,
                        equidistant_keys_select=RANDOM,
                        retrieval_prob = 1.0
                )
        )
    assert np.allclose(em.memory, np.array([[[1, 2, 3], [4, 5, 6]]]))

def test_ContentAddressableMemory_add_and_delete_from_memory():

    em = ContentAddressableMemory(
            initializer=[[[1,2,3], [4,5,6]],
                         [[7,8,9], [10,11,12]]],
            duplicate_keys=True,
            equidistant_keys_select=RANDOM,
            retrieval_prob = 1.0,
            storage_prob = 1.0
    )
    em.add_to_memory([[[10,20,30],[40,50,60]],
                     [[11,21,31],[41,51,61]]])

    expected_memory = [[[ 1,  2,  3],[ 4,  5,  6]],
                       [[ 7,  8,  9],[10, 11, 12]],
                       [[10, 20, 30],[40, 50, 60]],
                       [[11, 21, 31],[41, 51, 61]]]
    assert np.allclose(em.memory, expected_memory)

    em.delete_from_memory([[[1,2,3],[4,5,6]]])
    expected_memory = [[[ 7,  8,  9],[10, 11, 12]],
                       [[10, 20, 30],[40, 50, 60]],
                       [[11, 21, 31],[41, 51, 61]]]
    assert np.allclose(em.memory, expected_memory)

    # Test adding and deleting a single memory
    em.add_to_memory([[1,2,3],[100,101,102]])
    expected_memory = [[[ 7,  8,  9],[10, 11, 12]],
                       [[10, 20, 30],[40, 50, 60]],
                       [[11, 21, 31],[41, 51, 61]],
                       [[ 1,  2,  3],[100,101,102]]]
    assert np.allclose(em.memory, expected_memory)

    em.delete_from_memory([[1,2,3],[100,101,102]])
    expected_memory = [[[ 7,  8,  9],[10, 11, 12]],
                       [[10, 20, 30],[40, 50, 60]],
                       [[11, 21, 31],[41, 51, 61]]]
    assert np.allclose(em.memory, expected_memory)

    # Test adding memory with different size value
    em.add_to_memory([[1,2,3],[100,101,102,103]])
    expected_memory = [[[ 7,  8,  9],[10, 11, 12]],
                       [[10, 20, 30],[40, 50, 60]],
                       [[11, 21, 31],[41, 51, 61]],
                       [[ 1,  2,  3],[100,101,102,103]]]
    for m,e in zip(em.memory,expected_memory):
        for i,j in zip(m,e):
            assert np.allclose(i,j)

    # Test adding memory with different size value as np.array
    em.add_to_memory(np.array([[1,2,3],[200,201,202,203]], dtype=object))
    expected_memory = [[[ 7,  8,  9],[10, 11, 12]],
                       [[10, 20, 30],[40, 50, 60]],
                       [[11, 21, 31],[41, 51, 61]],
                       [[ 1,  2,  3],[100,101,102,103]],
                       [[ 1,  2,  3],[200,201,202,203]]]
    for m,e in zip(em.memory,expected_memory):
        for i,j in zip(m,e):
            assert np.allclose(i,j)

    # Test error for illegal key:
    with pytest.raises(FunctionError) as error_text:
        em.add_to_memory(np.array([[1,2],[200,201,202,203]], dtype=object))
    assert "Length of 'key'" in str(error_text.value) and "must be same as others in the dict" in str(error_text.value)


def test_ContentAddressableMemory_overwrite_mode():

    em = ContentAddressableMemory(
            initializer=[[[1,2,3], [4,5,6]],
                         [[7,8,9], [10,11,12]]],
            duplicate_keys=True,
            equidistant_keys_select=RANDOM,
            retrieval_prob = 1.0,
            storage_prob = 1.0
    )

    em.duplicate_keys = OVERWRITE

    # Add new memory
    retreived = em.execute([[7,8,10], [100,110,120]])
    assert np.allclose(list(retreived), [[7,8,9],[10,11,12]])
    expected_memory = [[[ 1,  2,  3],[4, 5, 6]],
                       [[7,8,9], [10,11,12]],
                       [[7,8,10], [100,110,120]]]
    assert np.allclose(em.memory, expected_memory)

    # Overwrite old memory
    retreived = em.execute([[7,8,9], [100,110,120]])
    assert np.allclose(list(retreived), [[7,8,9],[10,11,12]])
    expected_memory = [[[ 1,  2,  3],[4, 5, 6]],
                       [[7,8,9], [100,110,120]],
                       [[7,8,10], [100,110,120]]]
    assert np.allclose(em.memory, expected_memory)

    # Allow entry of memory with duplicate key
    em.duplicate_keys = True
    retreived = em.execute([[7,8,9], [200,210,220]])
    assert np.allclose(list(retreived), [[7,8,9],[100,110,120]])
    expected_memory = [[[ 1,  2,  3],[4, 5, 6]],
                       [[7,8,9], [100,110,120]],
                       [[7,8,10], [100,110,120]],
                       [[7,8,9], [200,210,220]]]
    assert np.allclose(em.memory, expected_memory)

    # Attempt to overwrite with two matches should generate error
    em.duplicate_keys = OVERWRITE
    with pytest.raises(FunctionError) as error_text:
        em.execute([[7,8,9], [200,210,220]])
    assert ('Attempt to store item' in str(error_text.value)
            and 'with \'duplicate_keys\'=\'OVERWRITE\'' in str(error_text.value))


def test_ContentAddressableMemory_max_entries():

    em = ContentAddressableMemory(
            initializer=[[[1,2,3], [4,5,6]],
                         [[7,8,9], [10,11,12]],
                         [[1,2,3], [100,101,102]]],
            duplicate_keys=True,
            equidistant_keys_select=RANDOM,
            retrieval_prob = 1.0,
            storage_prob = 1.0,
            max_entries = 4
    )
    em.add_to_memory([[[10,20,30],[40,50,60]],
                    [[11,21,31],[41,51,61]],
                    [[12,22,32],[42,52,62]]])
    expected_memory = [[[1,2,3], [100,101,102]],
                       [[10,20,30],[40,50,60]],
                       [[11,21,31],[41,51,61]],
                       [[12,22,32],[42,52,62]]]
    assert np.allclose(em.memory, expected_memory)


@pytest.mark.parametrize(
    'param_name',
    [
        'distance_function',
        'selection_function',
    ]
)
def test_ContentAddressableMemory_unique_functions(param_name):
    a = ContentAddressableMemory()
    b = ContentAddressableMemory()

    assert (
        getattr(a.parameters, param_name).get()
        is not getattr(b.parameters, param_name).get()
    )

    assert (
        getattr(a.defaults, param_name)
        is not getattr(b.defaults, param_name)
    )
