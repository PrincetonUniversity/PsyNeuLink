from itertools import combinations

import numpy as np
import pytest

import psyneulink.core.components.functions.stateful.memoryfunctions as Functions
from psyneulink import *
from psyneulink.core.globals.utilities import _SeededPhilox, convert_all_elements_to_np_array

# **********************************************************************************************************************
# OMINBUS TEST *********************************************************************************************************
# **********************************************************************************************************************

#region

module_seed = 0
np.random.seed(0)
SIZE=10
test_var = np.random.rand(2, SIZE)
test_initializer = np.array([[test_var[0] * 5, test_var[1] * 4]])
test_noise_arr = np.random.rand(SIZE)

RAND1 = np.random.random(1)
RAND2 = np.random.random()

# Use different size for Philox case,
# to easily detect mixups
philox_var = np.random.rand(2, SIZE - 1)
philox_initializer = np.array([[philox_var[0] * 5, philox_var[1] * 4]])

test_data = [
# Default initializer does not work
#    (Functions.Buffer, test_var, {'rate':RAND1}, [[0.0],[0.0]]),
    pytest.param(Functions.Buffer, test_var[0], {'history':512, 'rate':RAND1, 'initializer':[test_var[0]]},
                 # TODO: Why is the first result using rate^2 ?
                 [test_var[0] * RAND1 * RAND1, test_var[0] * RAND1],
                 marks=pytest.mark.llvm_not_implemented,
                 id="Buffer"),

    # Tests using Mersenne-Twister as function PRNG
    pytest.param(Functions.DictionaryMemory, test_var, {'seed': module_seed},
                 [test_var[0], test_var[1]],
                 id="DictionaryMemory"),
    pytest.param(Functions.DictionaryMemory, test_var, {'rate':RAND1, 'seed': module_seed},
                 [test_var[0] * RAND1, test_var[1]],
                 id="DictionaryMemory Rate"),
    pytest.param(Functions.DictionaryMemory, test_var, {'initializer':test_initializer, 'seed': module_seed},
                 [test_var[0], test_var[1]],
                 id="DictionaryMemory Initializer"),
    pytest.param(Functions.DictionaryMemory, test_var, {'retrieval_prob':0.1, 'seed': module_seed},
                 np.zeros_like(test_var),
                 id="DictionaryMemory Low Retrieval"),
    pytest.param(Functions.DictionaryMemory, test_var, {'storage_prob':0.1, 'seed': module_seed},
                 np.zeros_like(test_var),
                 id="DictionaryMemory Low Storage"),
    pytest.param(Functions.DictionaryMemory, test_var, {'retrieval_prob':0.9, 'storage_prob':0.9, 'seed': module_seed},
                 [test_var[0], test_var[1]],
                 id="DictionaryMemory High Storage/Retrieve"),
    pytest.param(Functions.DictionaryMemory, test_var, {'noise':RAND2},
                 [test_var[0] + RAND2, test_var[1]],
                 id="DictionaryMemory NoiseScalar"),
    pytest.param(Functions.DictionaryMemory, test_var, {'noise':RAND2, 'rate':RAND1},
                 [test_var[0] * RAND1 + RAND2, test_var[1]],
                 id="DictionaryMemory Rate NoiseScalar"),
    pytest.param(Functions.DictionaryMemory, test_var, {'noise':[RAND2]},
                 [test_var[0] + RAND2, test_var[1]],
                 id="DictionaryMemory NoiseVec1"),
    pytest.param(Functions.DictionaryMemory, test_var, {'noise':test_var / 2},
                 [test_var[0] + test_var[0] / 2, test_var[1]],
                 id="DictionaryMemory NoiseVecN"),

    # ContentAddressableMemory
    pytest.param(Functions.ContentAddressableMemory, test_var, {'rate':RAND1, 'retrieval_prob':0.1, 'seed': module_seed},
                 np.zeros_like(test_var),
                 marks=pytest.mark.llvm_not_implemented,
                 id="ContentAddressableMemory Low Retrieval"),
    pytest.param(Functions.ContentAddressableMemory, test_var, {'rate':RAND1, 'storage_prob':0.1, 'seed': module_seed},
                 np.zeros_like(test_var),
                 marks=pytest.mark.llvm_not_implemented,
                 id="ContentAddressableMemory Low Storage"),
    pytest.param(Functions.ContentAddressableMemory, test_var, {'rate':RAND1, 'retrieval_prob':0.9, 'storage_prob':0.9, 'seed': module_seed},
                 [test_var[0], test_var[1]],
                 marks=pytest.mark.llvm_not_implemented,
                 id="ContentAddressableMemory High Storage/Retrieval"),
    pytest.param(Functions.ContentAddressableMemory, test_var, {'initializer':test_initializer, 'rate':RAND1, 'seed': module_seed},
                 [test_var[0], test_var[1]],
                 marks=pytest.mark.llvm_not_implemented,
                 id="ContentAddressableMemory Initializer"),

    # Tests using philox var
    pytest.param(Functions.DictionaryMemory, philox_var, {'seed': module_seed},
                 [philox_var[0], philox_var[1]],
                 id="DictionaryMemory Philox"),
    pytest.param(Functions.DictionaryMemory, philox_var, {'rate':RAND1, 'seed': module_seed},
                 [philox_var[0] * RAND1, philox_var[1]],
                 id="DictionaryMemory Rate Philox"),
    pytest.param(Functions.DictionaryMemory, philox_var, {'initializer':philox_initializer, 'seed': module_seed},
                 [philox_var[0], philox_var[1]],
                 id="DictionaryMemory Initializer Philox"),
    pytest.param(Functions.DictionaryMemory, philox_var, {'retrieval_prob':0.01, 'seed': module_seed},
                 np.zeros_like(philox_var),
                 id="DictionaryMemory Low Retrieval Philox"),
    pytest.param(Functions.DictionaryMemory, philox_var, {'storage_prob':0.01, 'seed': module_seed},
                 np.zeros_like(philox_var),
                 id="DictionaryMemory Low Storage Philox"),
    pytest.param(Functions.DictionaryMemory, philox_var, {'retrieval_prob':0.98, 'storage_prob':0.98, 'seed': module_seed},
                 [philox_var[0], philox_var[1]],
                 id="DictionaryMemory High Storage/Retrieve Philox"),
    pytest.param(Functions.DictionaryMemory, philox_var, {'noise':RAND2},
                 [philox_var[0] + RAND2, philox_var[1]],
                 id="DictionaryMemory NoiseScalar Philox"),
    pytest.param(Functions.DictionaryMemory, philox_var, {'noise':RAND2, 'rate':RAND1},
                 [philox_var[0] * RAND1 + RAND2, philox_var[1]],
                 id="DictionaryMemory Rate NoiseScalar Philox"),
    pytest.param(Functions.DictionaryMemory, philox_var, {'noise':[RAND2]},
                 [philox_var[0] + RAND2, philox_var[1]],
                 id="DictionaryMemory NoiseVec1 Philox"),
    pytest.param(Functions.DictionaryMemory, philox_var, {'noise':philox_var / 2},
                 [philox_var[0] + philox_var[0] / 2, philox_var[1]],
                 id="DictionaryMemory NoiseVecN Philox"),

    # ContentAddressableMemory
    pytest.param(Functions.ContentAddressableMemory, philox_var, {'rate':RAND1, 'retrieval_prob':0.1, 'seed': module_seed},
                 np.zeros_like(philox_var),
                 marks=pytest.mark.llvm_not_implemented,
                 id="ContentAddressableMemory Low Retrieval Philox"),
    pytest.param(Functions.ContentAddressableMemory, philox_var, {'rate':RAND1, 'storage_prob':0.01, 'seed': module_seed},
                 np.zeros_like(philox_var),
                 marks=pytest.mark.llvm_not_implemented,
                 id="ContentAddressableMemory Low Storage Philox"),
    pytest.param(Functions.ContentAddressableMemory, philox_var, {'rate':RAND1, 'retrieval_prob':0.98, 'storage_prob':0.98, 'seed': module_seed},
                 [philox_var[0], philox_var[1]],
                 marks=pytest.mark.llvm_not_implemented,
                 id="ContentAddressableMemory High Storage/Retrieval Philox"),
    pytest.param(Functions.ContentAddressableMemory, philox_var, {'initializer':philox_initializer, 'rate':RAND1, 'seed': module_seed},
                 [philox_var[0], philox_var[1]],
                 marks=pytest.mark.llvm_not_implemented,
                 id="ContentAddressableMemory Initializer Philox"),
]

@pytest.mark.function
@pytest.mark.memory_function
@pytest.mark.benchmark
@pytest.mark.parametrize("func, variable, params, expected", test_data)
def test_basic(func, variable, params, expected, benchmark, func_mode):
    benchmark.group = func.componentName
    f = func(default_variable=variable, **params)
    if variable is philox_var:
        f.parameters.random_state.set(_SeededPhilox([module_seed]))

    EX = pytest.helpers.get_func_execution(f, func_mode)

    EX(variable)

    # Store value * 4 with a duplicate key
    # This insertion should be ignored unless the function allows
    # "duplicate_keys"
    if len(variable) == 2:
        EX([variable[0], variable[1] * 4])

    res = benchmark(EX, variable)

    # This still needs to use "allclose" as the key gets manipulated before
    # storage in some subtests. The rounding in that calculation might not
    # match the one done for expected values above.
    np.testing.assert_allclose(res[0], expected[0])
    np.testing.assert_allclose(res[1], expected[1])

#endregion

# **********************************************************************************************************************
# TEST Dictionary Memory ***********************************************************************************************
# **********************************************************************************************************************

#region

class TestDictionaryMemory:

    # standard numpy comparison methods don't work well with ragged arrays
    @staticmethod
    def _get_retrieved_key(stimuli, retrieved_value):
        # assumes as in tests below that at most one stimulus key will match
        for k, v in stimuli.items():
            v = convert_all_elements_to_np_array(v)
            if len(v) != len(retrieved_value):
                continue

            for i in range(len(v)):
                if not np.array_equal(v[i], retrieved_value[i]):
                    break
            else:
                return [k]

        return [None]

    # Test of DictionaryMemory without LLVM:
    def test_DictionaryMemory_with_initializer_and_key_size_same_as_val_size(self):

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
                function = DictionaryMemory(
                        seed=2,
                        initializer=np.array([stimuli['F'], stimuli['F']], dtype=object),
                        duplicate_keys=True,
                        equidistant_keys_select=RANDOM)
        )

        retrieved_keys=[]
        for key in sorted(stimuli.keys()):
            retrieved = em.execute(stimuli[key])
            retrieved_key = [k for k, v in stimuli.items() if np.array_equal(v, retrieved)] or [None]
            retrieved_keys.append(retrieved_key)
        assert retrieved_keys == [['F'], ['A'], ['A'], ['C'], ['B'], ['F']]

        # Run again to test re-initialization and random retrieval
        em.function.reset(np.array([stimuli['A'], stimuli['F']], dtype=object))
        retrieved_keys=[]
        for key in sorted(stimuli.keys()):
            retrieved = em.execute(stimuli[key])
            retrieved_key = [k for k, v in stimuli.items() if np.array_equal(v, retrieved)] or [None]
            retrieved_keys.append(retrieved_key)
        assert retrieved_keys == [['A'], ['A'], ['A'], ['A'], ['B'], ['F']]

        stim = 'C'
        em.function.equidistant_keys_select = OLDEST
        retrieved = em.function.get_memory(stimuli[stim][0])
        retrieved_key = [k for k, v in stimuli.items() if np.array_equal(v, retrieved)] or [None]
        assert retrieved_key == ['A']

        em.function.equidistant_keys_select = NEWEST
        retrieved = em.function.get_memory(stimuli[stim][0])
        retrieved_key = [k for k, v in stimuli.items() if np.array_equal(v, retrieved)] or [None]
        assert retrieved_key == ['D']

        # Test that after allowing dups, warning is issued and memory with zeros is returned
        em.function.duplicate_keys = False
        stim = 'A'

        text = r'More than one item matched key \(\[1. 2. 3.\]\) in memory for DictionaryMemory'
        with pytest.warns(UserWarning, match=text):
            retrieved = em.execute(stimuli[stim])

        retrieved_key = [k for k, v in stimuli.items() if np.array_equal(v, retrieved)] or [None]
        assert retrieved_key == [None]
        np.testing.assert_array_equal(retrieved[0], [0, 0, 0])
        np.testing.assert_array_equal(retrieved[1], [0, 0, 0])

    def test_DictionaryMemory_with_initializer_and_key_size_diff_from_val_size(self):

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
                function = DictionaryMemory(
                        initializer=np.array([stimuli['F'], stimuli['F']], dtype=object),
                        duplicate_keys=True,
                        equidistant_keys_select=RANDOM,
                        seed=module_seed)
        )

        retrieved_keys=[]
        for key in sorted(stimuli.keys()):
            print(key)
            retrieved = [i for i in em.execute(stimuli[key])]
            retrieved_keys.append(TestDictionaryMemory._get_retrieved_key(stimuli, retrieved))
        assert retrieved_keys == [['F'], ['A'], ['A'], ['A'], ['B'], ['F']]

        stim = 'C'
        em.function.equidistant_keys_select = OLDEST
        retrieved = [i for i in em.function.get_memory(stimuli[stim][0])]
        assert TestDictionaryMemory._get_retrieved_key(stimuli, retrieved) == ['A']

        em.function.equidistant_keys_select = NEWEST
        retrieved = [i for i in em.function.get_memory(stimuli[stim][0])]
        assert TestDictionaryMemory._get_retrieved_key(stimuli, retrieved) == ['D']

        # Test that after allowing dups, warning is issued and memory with zeros is returned
        em.function.duplicate_keys = False
        stim = 'A'

        text = r'More than one item matched key \(\[1. 2. 3.\]\) in memory for DictionaryMemory'
        with pytest.warns(UserWarning, match=text):
            retrieved = em.execute(stimuli[stim])

        retrieved_key = [k for k, v in stimuli.items() if np.array_equal(v, retrieved)] or [None]
        assert retrieved_key == [None]
        np.testing.assert_array_equal(retrieved[0], [0, 0, 0])
        np.testing.assert_array_equal(retrieved[1], [0, 0, 0, 0])

    # def test_DictionaryMemory_without_initializer_in_composition():
    #
    #     content = TransferMechanism(input_shapes=5)
    #     assoc = TransferMechanism(input_shapes=3)
    #     content_out = TransferMechanism(input_shapes=5)
    #     assoc_out = TransferMechanism(input_shapes=3)
    #
    #     # Episodic Memory, Decision and Control
    #     em = EpisodicMemoryMechanism(name='EM',
    #                                  content_size=5, assoc_size=3)
    #     comp = Composition()
    #     comp.add_nodes([content, assoc, content_out, assoc_out, em])
    #     comp.add_projection(MappingProjection(), content, em.input_ports[KEY_INPUT])
    #     comp.add_projection(MappingProjection(), assoc, em.input_ports[VALUE_INPUT])
    #     comp.add_projection(MappingProjection(), em.output_ports[KEY_OUTPUT], content_out)
    #     comp.add_projection(MappingProjection(), em.output_ports[VALUE_OUTPUT], assoc_out)
    #
    #     comp.run(inputs={content:[1,2,3,4,5],
    #                      assoc:[6,7,8]})

    def test_DictionaryMemory_without_initializer_and_key_size_same_as_val_size(self):

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
                function = DictionaryMemory(
                        duplicate_keys=True,
                        equidistant_keys_select=RANDOM,
                        seed=module_seed)
        )

        retrieved_keys=[]
        for key in sorted(stimuli.keys()):
            retrieved = [i for i in em.execute(stimuli[key])]
            retrieved_key = [k for k, v in stimuli.items() if np.array_equal(v, retrieved)] or [None]
            retrieved_keys.append(retrieved_key)
        assert retrieved_keys == [[None], ['A'], ['A'], ['C'], ['B'], ['D']]

        stim = 'C'
        em.function.equidistant_keys_select = OLDEST
        retrieved = [i for i in em.function.get_memory(stimuli[stim][0])]
        retrieved_key = [k for k, v in stimuli.items() if np.array_equal(v, retrieved)] or [None]
        assert retrieved_key == ['A']

        em.function.equidistant_keys_select = NEWEST
        retrieved = [i for i in em.function.get_memory(stimuli[stim][0])]
        retrieved_key = [k for k, v in stimuli.items() if np.array_equal(v, retrieved)] or [None]
        assert retrieved_key == ['D']

        # Test that after allowing dups, warning is issued and memory with zeros is returned
        em.function.duplicate_keys = False
        stim = 'A'

        text = r'More than one item matched key \(\[1. 2. 3.\]\) in memory for DictionaryMemory'
        with pytest.warns(UserWarning, match=text):
            retrieved = em.execute(stimuli[stim])

        retrieved_key = [k for k, v in stimuli.items() if np.array_equal(v, retrieved)] or [None]
        assert retrieved_key == [None]
        np.testing.assert_array_equal(retrieved[0], [0, 0, 0])
        np.testing.assert_array_equal(retrieved[1], [0, 0, 0])

    def test_DictionaryMemory_without_initializer_and_key_size_diff_from_val_size(self):

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
                function = DictionaryMemory(
                        duplicate_keys=True,
                        equidistant_keys_select=RANDOM,
                        seed=module_seed)
        )

        retrieved_keys=[]
        for key in sorted(stimuli.keys()):
            retrieved = [i for i in em.execute(stimuli[key])]
            retrieved_keys.append(TestDictionaryMemory._get_retrieved_key(stimuli, retrieved))
        assert retrieved_keys == [[None], ['A'], ['A'], ['C'], ['B'], ['D']]

        stim = 'C'
        em.function.equidistant_keys_select = OLDEST
        retrieved = [i for i in em.function.get_memory(stimuli[stim][0])]
        assert TestDictionaryMemory._get_retrieved_key(stimuli, retrieved) == ['A']

        em.function.equidistant_keys_select = NEWEST
        retrieved = [i for i in em.function.get_memory(stimuli[stim][0])]
        assert TestDictionaryMemory._get_retrieved_key(stimuli, retrieved) == ['D']

        # Test that after allowing dups, warning is issued and memory with zeros is returned
        em.function.duplicate_keys = False
        stim = 'A'

        text = r'More than one item matched key \(\[1. 2. 3.\]\) in memory for DictionaryMemory'
        with pytest.warns(UserWarning, match=text):
            retrieved = em.execute(stimuli[stim])

        retrieved_key = [k for k, v in stimuli.items() if np.array_equal(v, retrieved)] or [None]
        assert retrieved_key == [None]
        np.testing.assert_array_equal(retrieved[0], [0, 0, 0])
        np.testing.assert_array_equal(retrieved[1], [0, 0, 0, 0])

    def test_DictionaryMemory_without_assoc(self):

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
                function = DictionaryMemory(
                        # initializer=np.array([stimuli['F'], stimuli['F']], dtype=object),
                        duplicate_keys=True,
                        equidistant_keys_select=RANDOM,
                        retrieval_prob = 1.0,
                        seed=module_seed,
                )
        )

        for key in sorted(stimuli.keys()):
            print(f'\nCurrent memory: \n{em.memory}\n')
            retrieved = [i for i in em.execute(stimuli[key])]
            retrieved_key = [k for k, v in stimuli.items() if np.array_equal(v, retrieved)] or [None]
            print(f'\nExecuted with stimulus {key}: {stimuli[key]};'
                  f'\nRetrieved memory {retrieved_key[0]}: \n\t{retrieved}')

        retrieved_keys=[]
        for key in sorted(stimuli.keys()):
            retrieved = [i for i in em.execute(stimuli[key])]
            retrieved_key = [k for k, v in stimuli.items() if np.array_equal(v, retrieved)] or [None]
            retrieved_keys.append(retrieved_key)

        assert retrieved_keys == [['A', 'C', 'D'], ['B'], ['A', 'C', 'D'], ['A', 'C', 'D'], ['E'], ['F']]

    def test_DictionaryMemory_with_duplicate_entry_in_initializer_warning(self):

        regexp = r'Attempt to initialize memory of DictionaryMemory with an entry \(\[\[1 2 3\]'
        with pytest.warns(UserWarning, match=regexp):
            em = EpisodicMemoryMechanism(
                    name='EPISODIC MEMORY MECH',
                    content_size=3,
                    assoc_size=3,
                    function = DictionaryMemory(
                            initializer=np.array([[[1,2,3], [4,5,6]],
                                                  [[1,2,3], [7,8,9]]]),
                            duplicate_keys=False,
                            equidistant_keys_select=RANDOM,
                            retrieval_prob = 1.0,
                            seed=module_seed,
                    )
            )
        np.testing.assert_allclose(em.memory, np.array([[[1, 2, 3], [4, 5, 6]]]))

    def test_DictionaryMemory_add_and_delete_from_memory(self):

        em = DictionaryMemory(
                initializer=[[[1,2,3], [4,5,6]],
                             [[7,8,9], [10,11,12]]],
                duplicate_keys=True,
                equidistant_keys_select=RANDOM,
                retrieval_prob = 1.0,
                storage_prob = 1.0,
                seed=module_seed,
        )
        em.add_to_memory([[[10,20,30],[40,50,60]],
                         [[11,21,31],[41,51,61]]])

        expected_memory = [[[ 1,  2,  3],[ 4,  5,  6]],
                           [[ 7,  8,  9],[10, 11, 12]],
                           [[10, 20, 30],[40, 50, 60]],
                           [[11, 21, 31],[41, 51, 61]]]
        np.testing.assert_allclose(em.memory, expected_memory)

        em.delete_from_memory([[[1,2,3],[4,5,6]]])
        expected_memory = [[[ 7,  8,  9],[10, 11, 12]],
                           [[10, 20, 30],[40, 50, 60]],
                           [[11, 21, 31],[41, 51, 61]]]
        np.testing.assert_allclose(em.memory, expected_memory)

        # Test adding and deleting a single memory
        em.add_to_memory([[1,2,3],[100,101,102]])
        expected_memory = [[[ 7,  8,  9],[10, 11, 12]],
                           [[10, 20, 30],[40, 50, 60]],
                           [[11, 21, 31],[41, 51, 61]],
                           [[ 1,  2,  3],[100,101,102]]]
        np.testing.assert_allclose(em.memory, expected_memory)

        em.delete_from_memory([[1,2,3],[100,101,102]])
        expected_memory = [[[ 7,  8,  9],[10, 11, 12]],
                           [[10, 20, 30],[40, 50, 60]],
                           [[11, 21, 31],[41, 51, 61]]]
        np.testing.assert_allclose(em.memory, expected_memory)

        # Test adding memory with different size value
        em.add_to_memory([[1,2,3],[100,101,102,103]])
        expected_memory = [[[ 7,  8,  9],[10, 11, 12]],
                           [[10, 20, 30],[40, 50, 60]],
                           [[11, 21, 31],[41, 51, 61]],
                           [[ 1,  2,  3],[100,101,102,103]]]
        for m,e in zip(em.memory,expected_memory):
            for i,j in zip(m,e):
                np.testing.assert_allclose(i,j)

        # Test adding memory with different size value as np.array
        em.add_to_memory(np.array([[1,2,3],[200,201,202,203]], dtype=object))
        expected_memory = [[[ 7,  8,  9],[10, 11, 12]],
                           [[10, 20, 30],[40, 50, 60]],
                           [[11, 21, 31],[41, 51, 61]],
                           [[ 1,  2,  3],[100,101,102,103]],
                           [[ 1,  2,  3],[200,201,202,203]]]
        for m,e in zip(em.memory,expected_memory):
            for i,j in zip(m,e):
                np.testing.assert_allclose(i,j)

        # Test error for illegal key:
        with pytest.raises(FunctionError) as error_text:
            em.add_to_memory(np.array([[1,2],[200,201,202,203]], dtype=object))
        assert "Length of 'key'" in str(error_text.value) and "must be same as others in the dict" in str(error_text.value)

    def test_DictionaryMemory_overwrite_mode(self):

        em = DictionaryMemory(
                initializer=[[[1,2,3], [4,5,6]],
                             [[7,8,9], [10,11,12]]],
                duplicate_keys=True,
                equidistant_keys_select=RANDOM,
                retrieval_prob = 1.0,
                storage_prob = 1.0,
                seed=module_seed,
        )

        em.duplicate_keys = OVERWRITE

        # Add new memory
        retreived = em.execute([[7,8,10], [100,110,120]])
        np.testing.assert_allclose(list(retreived), [[7,8,9],[10,11,12]])
        expected_memory = [[[ 1,  2,  3],[4, 5, 6]],
                           [[7,8,9], [10,11,12]],
                           [[7,8,10], [100,110,120]]]
        np.testing.assert_allclose(em.memory, expected_memory)

        # Overwrite old memory
        retreived = em.execute([[7,8,9], [100,110,120]])
        np.testing.assert_allclose(list(retreived), [[7,8,9],[10,11,12]])
        expected_memory = [[[ 1,  2,  3],[4, 5, 6]],
                           [[7,8,9], [100,110,120]],
                           [[7,8,10], [100,110,120]]]
        np.testing.assert_allclose(em.memory, expected_memory)

        # Allow entry of memory with duplicate key
        em.duplicate_keys = True
        retreived = em.execute([[7,8,9], [200,210,220]])
        np.testing.assert_allclose(list(retreived), [[7,8,9],[100,110,120]])
        expected_memory = [[[ 1,  2,  3],[4, 5, 6]],
                           [[7,8,9], [100,110,120]],
                           [[7,8,10], [100,110,120]],
                           [[7,8,9], [200,210,220]]]
        np.testing.assert_allclose(em.memory, expected_memory)

        # Attempt to overwrite with two matches should generate error
        em.duplicate_keys = OVERWRITE
        with pytest.raises(FunctionError) as error_text:
            em.execute([[7,8,9], [200,210,220]])
        assert ('Attempt to store item' in str(error_text.value)
                and 'with \'duplicate_keys\'=\'OVERWRITE\'' in str(error_text.value))

    def test_DictionaryMemory_max_entries(self):

        em = DictionaryMemory(
                initializer=[[[1,2,3], [4,5,6]],
                             [[7,8,9], [10,11,12]],
                             [[1,2,3], [100,101,102]]],
                duplicate_keys=True,
                equidistant_keys_select=RANDOM,
                retrieval_prob = 1.0,
                storage_prob = 1.0,
                max_entries = 4,
                seed=module_seed,
        )
        em.add_to_memory([[[10,20,30],[40,50,60]],
                        [[11,21,31],[41,51,61]],
                        [[12,22,32],[42,52,62]]])
        expected_memory = [[[1,2,3], [100,101,102]],
                           [[10,20,30],[40,50,60]],
                           [[11,21,31],[41,51,61]],
                           [[12,22,32],[42,52,62]]]
        np.testing.assert_allclose(em.memory, expected_memory)

    @pytest.mark.parametrize(
        'param_name',
        [
            'distance_function',
            'selection_function',
        ]
    )
    def test_DictionaryMemory_unique_functions(self, param_name):
        a = DictionaryMemory()
        b = DictionaryMemory()

        assert (
            getattr(a.parameters, param_name).get()
            is not getattr(b.parameters, param_name).get()
        )

        assert (
            getattr(a.defaults, param_name)
            is not getattr(b.defaults, param_name)
        )
#endregion

# **********************************************************************************************************************
# TEST ContentAddressableMemory ****************************************************************************************
# **********************************************************************************************************************

def retrieve_label_helper(retrieved, stimuli):
    return [k for k,v in stimuli.items()
            if all(all(a)
                   for a in np.equal(np.array(retrieved, dtype=object),
                                     np.array(v, dtype=object),
                                     dtype=object))] or [None]

#region
class TestContentAddressableMemory:

    # Note:  this warning is issued because the default distance_function is Distance(metric=COSINE)
    #        if the default is changed, this warning may not occur
    distance_warning_msg = "always produce a distance of 0 (since angle of scalars is not defined)."

    test_vars = [
        # initializer:      expected_result (as list):
        (1,                 [[[1.]]], distance_warning_msg),
        ([1],               [[[1.]]], distance_warning_msg),
        ([1,1],             [[[1., 1.]]], None),
        ([[1,1]],           [[[1., 1.]]], None),
        ([[[1,1]]],         [[[1., 1.]]], None),
        ([[1,1],[2,2,2]],   [[[1., 1.],[2., 2., 2.]]], None),
        ([[[1,1],[2,2,2]]], [[[1., 1.],[2., 2., 2.]]], None)
    ]

    @pytest.mark.parametrize('initializer, expected_memory, warning_msg', test_vars)
    def test_ContentAddressableMemory_allowable_initializer_shapes(self, initializer, expected_memory, warning_msg):
        if warning_msg:
            with pytest.warns(UserWarning) as warning:
                c = ContentAddressableMemory(initializer=initializer)
            assert warning_msg in str(warning[0].message)
        else:
            c = ContentAddressableMemory(initializer=initializer)

        # Some test cases return dtype=object for rugged arrays.
        # There's no np.testing function that handles the case correctly
        expected_memory = convert_all_elements_to_np_array(expected_memory)
        assert len(c.memory) == len(expected_memory)
        for m, e in zip(c.memory, expected_memory):
            assert len(m) == len(e)
            for x, y in zip(m, e):
                np.testing.assert_array_equal(x, y)

    def test_ContentAddressableMemory_simple_distances(self):

        stimuli = np.array([[[1,2,3],[4,5,6]],
                            [[1,2,5],[4,5,8]],
                            [[1,2,10],[4,5,10]]
                            ])

        c = ContentAddressableMemory(
            initializer=stimuli,
            storage_prob=0,
            distance_function=Distance(metric=COSINE),
            seed=module_seed,
        )

        # Test distance (for retrieved item) and distances_by_field
        retrieved = c([[1, 2, 4], [4, 5, 9]])
        np.testing.assert_equal(retrieved, [[1, 2, 5], [4, 5, 8]])
        assert c.distance == Distance(metric=COSINE)([retrieved,[[1,2,4],[4,5,9]]])
        np.testing.assert_allclose(c.distances_by_field, [0.00397616, 0.00160159], rtol=1e-5, atol=1e-8)

        # Test distance_field_weights as scalar
        c.distance_field_weights=[2.5]
        retrieved = c([[1, 2, 4], [4, 5, 9]])
        np.testing.assert_equal(retrieved, [[1, 2, 5], [4, 5, 8]])
        assert c.distance == 2.5 * Distance(metric=COSINE)([retrieved,[[1,2,4],[4,5,9]]])
        np.testing.assert_allclose(c.distances_by_field, [2.5 * 0.00397616, 2.5 * 0.00160159], rtol=1e-5, atol=1e-8)

        # Test with 0 as one field weight
        c.distance_field_weights=[1,0]
        retrieved = c([[1, 2, 3], [4, 5, 10]])
        np.testing.assert_equal(retrieved, [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_equal(c.distances_by_field, [0.0, 0.0])

        # Test with 0 as the other field weight
        c.distance_field_weights=[0,1]
        retrieved = c([[1, 2, 3], [4, 5, 10]])
        np.testing.assert_equal(retrieved, [[1, 2, 10], [4, 5, 10]])
        np.testing.assert_equal(c.distances_by_field, [0.0, 0.0])

        # Test with 0 as both field weights (equvialent to setting retrieval_prob=0, so should return 0's)
        c.distance_field_weights=[0,0]
        retrieved = c([[1, 2, 3], [4, 5, 10]])
        np.testing.assert_equal(retrieved, [[0, 0, 0], [0, 0, 0]])
        np.testing.assert_equal(c.distances_by_field, [0.0, 0.0])

        # Test with None as field weight
        c.distance_field_weights=[None,1]
        retrieved = c([[1, 2, 3], [4, 5, 10]])
        np.testing.assert_equal(retrieved, [[1, 2, 10], [4, 5, 10]])
        np.testing.assert_equal(c.distances_by_field, [None, 0.0])

        c.distance_field_weights=[1, None]
        retrieved = c([[1, 2, 3], [4, 5, 10]])
        np.testing.assert_equal(retrieved, [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_equal(c.distances_by_field, [0.0, None])

        # Test with [] as field weight
        c.distance_field_weights=[[],1]
        retrieved = c([[1, 2, 3], [4, 5, 10]])
        np.testing.assert_equal(retrieved, [[1, 2, 10], [4, 5, 10]])
        np.testing.assert_equal(c.distances_by_field, [None, 0.0])

        c.distance_field_weights=[1, []]
        retrieved = c([[1, 2, 3], [4, 5, 10]])
        np.testing.assert_equal(retrieved, [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_equal(c.distances_by_field, [0.0, None])

    # FIX: COULD CONDENSE THESE TESTS BY PARAMETERIZING FIELD-WEIGHTS AND ALSO INCLUDE DISTANCE METRIC AS A PARAM
    def test_ContentAddressableMemory_parametric_distances(self):

        stimuli = np.array([[[1,2,3],[4,5,6]],
                            [[7,8,9],[10,11,12]],
                            [[13,14,15],[16,17,18]]])

        c = ContentAddressableMemory(
            initializer=stimuli,
            storage_prob=0,
            distance_function=Distance(metric=COSINE),
            seed=module_seed,
        )

        pairs = list(combinations(range(0,3),2))
        # Distances between all stimuli
        distances = [Distance(metric=COSINE)([stimuli[i],stimuli[j]]) for i, j in pairs]
        c_distances = []
        # for i,j in pairs:

        # Test distances with evenly weighted fields
        retrieved = c(stimuli[0])
        np.testing.assert_equal(retrieved, stimuli[0])
        np.testing.assert_allclose(c.distances_to_entries, [0, distances[0], distances[1]], rtol=1e-5, atol=1e-8)

        retrieved = c(stimuli[1])
        np.testing.assert_equal(retrieved, stimuli[1])
        np.testing.assert_allclose(c.distances_to_entries, [distances[0], 0, distances[2]], rtol=1e-5, atol=1e-8)

        retrieved = c(stimuli[2])
        np.testing.assert_equal(retrieved, stimuli[2])
        np.testing.assert_allclose(c.distances_to_entries, [distances[1], distances[2], 0], rtol=1e-5, atol=1e-8)

        # Test distances using distance_field_weights
        field_weights = [np.array([[1],[0]]), np.array([[0],[1]])]
        for fw in field_weights:
            c.distance_field_weights = fw
            distances = []
            for k in range(2):
                if fw[k]:
                    distances.append([Distance(metric=COSINE)([stimuli[i][k], stimuli[j][k]]) * fw[k]
                                      for i, j in pairs])
            distances = np.array(distances)
            distances = np.squeeze(np.sum(distances, axis=0) / len([f for f in fw if f]))

            retrieved = c(stimuli[0])
            np.testing.assert_equal(retrieved, stimuli[0])
            np.testing.assert_allclose(c.distances_to_entries, [0, distances[0], distances[1]], rtol=1e-5, atol=1e-8)

            retrieved = c(stimuli[1])
            np.testing.assert_equal(retrieved, stimuli[1])
            np.testing.assert_allclose(c.distances_to_entries, [distances[0], 0, distances[2]], rtol=1e-5, atol=1e-8)

            retrieved = c(stimuli[2])
            np.testing.assert_equal(retrieved, stimuli[2])
            np.testing.assert_allclose(c.distances_to_entries, [distances[1], distances[2], 0], rtol=1e-5, atol=1e-8)

        # Test distances_by_fields
        c.distance_field_weights=[1,1]
        stim = [[8,9,10],[11,12,13]]
        retrieved = c(stim)
        np.testing.assert_equal(retrieved, [[7, 8, 9], [10, 11, 12]])
        distances_by_field = [Distance(metric=COSINE)([retrieved[i], stim[i]]) for i in range(2)]
        np.testing.assert_equal(c.distances_by_field, distances_by_field)

    # Test of ContentAddressableMemory without LLVM:
    def test_ContentAddressableMemory_with_initializer_and_equal_field_sizes(self):

        stimuli = {'A': [[1,2,3],[4,5,6]],
                   'B': [[8,9,10],[11,12,13]],
                   'C': [[1,2,3],[11,12,13]],
                   'D': [[1,2,3],[21,22,23]],
                   'E': [[9,8,4],[11,12,13]],
                   'F': [[10,10,30],[40,50,60]],
                   }

        c = ContentAddressableMemory(
            seed=2,
            initializer=np.array([stimuli['F'], stimuli['F']], dtype=object),
            distance_function=Distance(metric=COSINE),
            duplicate_entries_allowed=True,
            equidistant_entries_select=RANDOM
        )

        retrieved_labels=[]
        sorted_labels = sorted(stimuli.keys())
        for label in sorted_labels:
            retrieved = [i for i in c(stimuli[label])]
            # Get label of retrieved item
            retrieved_label = retrieve_label_helper(retrieved, stimuli)
            # Get distances of retrieved entry to all other entries and assert it has the minimum distance
            distances = [Distance(metric=COSINE)([retrieved,stimuli[k]]) for k in sorted_labels]
            min_idx = distances.index(min(distances))
            assert retrieved_label == [sorted_labels[min_idx]]
            retrieved_labels.append(retrieved_label)
        assert retrieved_labels == [['F'], ['A'], ['F'], ['C'], ['B'], ['F']]

        # Run again to test re-initialization and random retrieval
        c.reset(np.array([stimuli['A'], stimuli['F']], dtype=object))
        retrieved_labels=[]
        for label in sorted(stimuli.keys()):
            retrieved = [i for i in c(stimuli[label])]
            retrieved_label = retrieve_label_helper(retrieved, stimuli)
            # Get distances of retrieved entry to all other entries and assert it has the minimum distance
            distances = [Distance(metric=COSINE)([retrieved,stimuli[k]]) for k in sorted_labels]
            min_idx = distances.index(min(distances))
            assert retrieved_label == [sorted_labels[min_idx]]
            retrieved_labels.append(retrieved_label)
            Distance(metric=COSINE)([retrieved,stimuli['A']])
        assert retrieved_labels == [['A'], ['A'], ['F'], ['C'], ['B'], ['F']]

        # Test  restricting retrieval to only 1st field (which has duplicate values) and selecting for OLDEST
        c.distance_field_weights = [1,0]
        stim = 'C' # Has same 1st field as A (older) and D (newer)

        c.equidistant_entries_select = OLDEST  # Should return A
        retrieved = c.get_memory(stimuli[stim])
        retrieved_label = [k for k, v in stimuli.items()
                           if np.all([vi == retrieved[i] for i, vi in enumerate(v)])] or [None]
        assert retrieved_label == ['A']

        c.equidistant_entries_select = NEWEST  # Should return D
        retrieved = c.get_memory(stimuli[stim])
        retrieved_label = retrieve_label_helper(retrieved, stimuli)
        assert retrieved_label == ['D']

        # Test that after allowing dups and now disallowing them, warning is issued and memory with zeros is returned
        c.duplicate_entries_allowed = False
        stim = 'A'
        text = "More than one entry matched cue"
        with pytest.warns(UserWarning, match=text):
            retrieved = c(stimuli[stim])
        retrieved_label = retrieve_label_helper(retrieved, stimuli)
        assert retrieved_label == [None]
        np.testing.assert_equal(retrieved, [[0, 0, 0], [0, 0, 0]])

    def test_ContentAddressableMemory_with_initializer_and_diff_field_sizes(self):

        stimuli = {'A': np.array([[1.,2.,3.],[4.,5.,6.,7.]], dtype=object),
                   'B': np.array([[8.,9.,10.],[11.,12.,13.,14.]], dtype=object),
                   'C': np.array([[1.,2.,3.],[11.,12.,13.,14.]], dtype=object),
                   'D': np.array([[1.,2.,3.],[21.,22.,23.,24.]], dtype=object),
                   'E': np.array([[9.,8.,4.],[11.,12.,13.,14.]], dtype=object),
                   'F': np.array([[10.,10.,30.],[40.,50.,60.,70.]], dtype=object),
                   }

        c = ContentAddressableMemory(
            initializer=np.array([stimuli['F'], stimuli['F']], dtype=object),
            duplicate_entries_allowed=True,
            equidistant_entries_select=RANDOM,
            seed=module_seed,
        )

        # Run again to test re-initialization and random retrieval
        c.reset(np.array([stimuli['A'], stimuli['F']], dtype=object))
        retrieved_labels=[]
        for key in sorted(stimuli.keys()):
            retrieved = c(stimuli[key])
            retrieved_label = retrieve_label_helper(retrieved, stimuli)
            retrieved_labels.append(retrieved_label)
        assert retrieved_labels == [['A'], ['A'], ['F'], ['C'], ['B'], ['F']]

        c.distance_field_weights = [1,0]
        stim = 'C'
        c.equidistant_entries_select = OLDEST
        retrieved = c.get_memory(stimuli[stim])
        retrieved_label = retrieve_label_helper(retrieved, stimuli)
        retrieved_labels.append(retrieved_label)
        assert retrieved_label == ['A']

        c.equidistant_entries_select = NEWEST
        retrieved = c.get_memory(stimuli[stim])
        retrieved_label = retrieve_label_helper(retrieved, stimuli)
        assert retrieved_label == ['D']

        # Test that after allowing dups, warning is issued and memory with zeros is returned
        c.duplicate_entries_allowed = False
        stim = 'A'

        text = r'More than one entry matched cue'
        with pytest.warns(UserWarning, match=text):
            retrieved = c(stimuli[stim])

        retrieved_label = retrieve_label_helper(retrieved, stimuli)
        assert retrieved_label == [None]

        expected = convert_all_elements_to_np_array([[0, 0, 0], [0, 0, 0, 0]])

        # There's no np.testing function that handles the rugged arrays correctly
        assert len(retrieved) == len(expected)
        for m, e in zip(retrieved, expected):
            assert len(m) == len(e)
            for x, y in zip(m, e):
                np.testing.assert_array_equal(x, y)

    def test_ContentAddressableMemory_without_initializer_and_equal_field_sizes(self):

        stimuli = {'A': [[1,2,3],[4,5,6]],
                   'B': [[8,9,10],[11,12,13]],
                   'C': [[1,2,3],[11,12,13]],
                   'D': [[1,2,3],[21,22,23]],
                   'E': [[9,8,4],[11,12,13]],
                   'F': [[10,10,30],[40,50,60]],
                   }

        c = ContentAddressableMemory(
            distance_function=Distance(metric=COSINE),
            duplicate_entries_allowed=True,
            equidistant_entries_select=RANDOM,
            seed=module_seed,
        )

        retrieved_labels=[]
        sorted_labels = sorted(stimuli.keys())
        for label in sorted_labels:
            retrieved = [i for i in c(stimuli[label])]
            retrieved_label = retrieve_label_helper(retrieved, stimuli)
            retrieved_labels.append(retrieved_label)
        assert retrieved_labels == [[None], ['A'], ['A'], ['C'], ['B'], ['A']]

        stim = 'C'
        c.distance_field_weights = [1,0]
        c.equidistant_entries_select = OLDEST
        retrieved = [i for i in c.get_memory(stimuli[stim])]
        retrieved_label = retrieve_label_helper(retrieved, stimuli)
        assert retrieved_label == ['A']

        c.equidistant_entries_select = NEWEST
        retrieved = [i for i in c.get_memory(stimuli[stim])]
        retrieved_label = retrieve_label_helper(retrieved, stimuli)
        assert retrieved_label == ['D']

        # Test that after allowing dups, warning is issued and memory with zeros is returned
        c.duplicate_entries_allowed = False
        stim = 'A'

        text = "More than one entry matched cue"
        with pytest.warns(UserWarning, match=text):
            retrieved = c.execute(stimuli[stim])

        retrieved_label = retrieve_label_helper(retrieved, stimuli)
        assert retrieved_label == [None]
        expected = np.array([np.array([0,0,0]),np.array([0,0,0])])
        assert all(all(x) for x in np.equal(expected,retrieved, dtype=object))

    def test_ContentAddressableMemory_without_initializer_and_diff_field_sizes(self):

        stimuli = {'A': np.array([[1,2,3],[4,5,6,7]], dtype=object),
                   'B': np.array([[8,9,10],[11,12,13,14]], dtype=object),
                   'C': np.array([[1,2,3],[11,12,13,14]], dtype=object),
                   'D': np.array([[1,2,3],[21,22,23,24]], dtype=object),
                   'E': np.array([[9,8,4],[11,12,13,14]], dtype=object),
                   'F': np.array([[10,10,30],[40,50,60,70]], dtype=object),
                   }

        c = ContentAddressableMemory(
            duplicate_entries_allowed=True,
            equidistant_entries_select=RANDOM,
            distance_field_weights=[1,0],
            seed=module_seed,
        )

        retrieved_labels=[]
        for key in sorted(stimuli.keys()):
            retrieved = c(stimuli[key])
            retrieved_label = retrieve_label_helper(retrieved, stimuli)
            retrieved_labels.append(retrieved_label)
        assert retrieved_labels == [[None], ['A'], ['A'], ['C'], ['B'], ['D']]

        stim = 'C'
        c.equidistant_entries_select = OLDEST
        retrieved = c.get_memory(stimuli[stim])
        retrieved_label = retrieve_label_helper(retrieved, stimuli)
        assert retrieved_label == ['A']

        c.equidistant_entries_select = NEWEST
        retrieved = c.get_memory(stimuli[stim])
        retrieved_label = retrieve_label_helper(retrieved, stimuli)
        assert retrieved_label == ['D']

        # Test that after allowing dups, warning is issued and memory with zeros is returned
        c.duplicate_entries_allowed = False
        stim = 'A'

        text = "More than one entry matched cue"
        with pytest.warns(UserWarning, match=text):
            retrieved = c(stimuli[stim])

        retrieved_label = retrieve_label_helper(retrieved, stimuli)
        assert retrieved_label == [None]

        expected = convert_all_elements_to_np_array([[0, 0, 0], [0, 0, 0, 0]])

        # There's no np.testing function that handles the rugged arrays correctly
        assert len(retrieved) == len(expected)
        for m, e in zip(retrieved, expected):
            assert len(m) == len(e)
            for x, y in zip(m, e):
                np.testing.assert_array_equal(x, y)

    def test_ContentAddressableMemory_with_duplicate_entry_in_initializer_warning(self):

        regexp = r'Attempt to initialize memory of ContentAddressableMemory with an entry \(\[\[1 2 3\]'
        with pytest.warns(UserWarning, match=regexp):
            c = ContentAddressableMemory(
                initializer=np.array([[[1,2,3], [4,5,6]],
                                      [[1,2,3], [7,8,9]]]),
                duplicate_entries_allowed=False,
                distance_field_weights=[1,0],
                equidistant_entries_select=RANDOM,
                retrieval_prob = 1.0,
                seed=module_seed,
            )
        np.testing.assert_allclose(c.memory, np.array([[[1, 2, 3], [4, 5, 6]]]))

    def test_ContentAddressableMemory_add_and_delete_from_memory(self):

        c = ContentAddressableMemory(
            initializer=[[[1,2,3], [4,5,6]],
                         [[7,8,9], [10,11,12]]],
            duplicate_entries_allowed=True,
            equidistant_entries_select=RANDOM,
            retrieval_prob = 1.0,
            storage_prob = 1.0,
            seed=module_seed,
        )
        c.add_to_memory([[[10,20,30],[40,50,60]],
                         [[11,21,31],[41,51,61]]])

        expected_memory = [[[ 1,  2,  3],[ 4,  5,  6]],
                           [[ 7,  8,  9],[10, 11, 12]],
                           [[10, 20, 30],[40, 50, 60]],
                           [[11, 21, 31],[41, 51, 61]]]
        np.testing.assert_allclose(c.memory, expected_memory)

        c.delete_from_memory([[[1,2,3],[4,5,6]]])
        expected_memory = [[[ 7,  8,  9],[10, 11, 12]],
                           [[10, 20, 30],[40, 50, 60]],
                           [[11, 21, 31],[41, 51, 61]]]
        np.testing.assert_allclose(c.memory, expected_memory)

        # Test adding and deleting a single memory
        c.add_to_memory([[1,2,3],[100,101,102]])
        expected_memory = [[[ 7,  8,  9],[10, 11, 12]],
                           [[10, 20, 30],[40, 50, 60]],
                           [[11, 21, 31],[41, 51, 61]],
                           [[ 1,  2,  3],[100,101,102]]]
        np.testing.assert_allclose(c.memory, expected_memory)

        c.delete_from_memory([[1,2,3],[100,101,102]])
        expected_memory = [[[ 7,  8,  9],[10, 11, 12]],
                           [[10, 20, 30],[40, 50, 60]],
                           [[11, 21, 31],[41, 51, 61]]]
        np.testing.assert_allclose(c.memory, expected_memory)

        # Test adding memory with different size value
        with pytest.raises(FunctionError) as error_text:
            c.add_to_memory(np.array([[1,2,3],[100,101,102,103]], dtype=object))
        assert "Field 1 of entry ([array([1, 2, 3]) array([100, 101, 102, 103])]) has incorrect shape ((4,)) " \
               "for memory of 'ContentAddressableMemory Function-0';  should be: (3,)." in str(error_text.value)

        # Test adding memory in first field of np.ndarray with wrong size:
        with pytest.raises(FunctionError) as error_text:
            c.add_to_memory(np.array([[1,2],[200,201,202,203]], dtype=object))
        assert "Field 0 of entry ([array([1, 2]) array([200, 201, 202, 203])]) has incorrect shape ((2,)) " \
               "for memory of 'ContentAddressableMemory Function-0';  should be: (3,)." in str(error_text.value)

        # Test adding memory in second field of np.ndarray with wrong size:
        with pytest.raises(FunctionError) as error_text:
            c.add_to_memory(np.array([[1,2,3],[200,201,202,203]], dtype=object))
        assert "Field 1 of entry ([array([1, 2, 3]) array([200, 201, 202, 203])]) has incorrect shape ((4,)) " \
               "for memory of 'ContentAddressableMemory Function-0';  should be: (3,)." in str(error_text.value)

    def test_ContentAddressableMemory_duplicate_entries(self):

        c = ContentAddressableMemory(
            initializer=[[[1,2,3], [4,5,6]],
                         [[7,8,9], [10,11,12]],
                         [[7,8,9], [10,11,12]]],
            duplicate_entries_allowed=False,
            seed=module_seed,
        )

        expected_memory = [[[ 1,  2,  3],[ 4,  5,  6]],
                           [[ 7,  8,  9],[10, 11, 12]]]
        np.testing.assert_allclose(c.memory, expected_memory)

        c.add_to_memory([[ 1,  2,  3],[ 4,  5,  6]])
        np.testing.assert_allclose(c.memory, expected_memory)

        c.execute([[ 1,  2,  3],[ 4,  5,  6]])
        np.testing.assert_allclose(c.memory, expected_memory)

        c.duplicate_threshold = 0  # <- Low threshold allows new entry to be considered distinct
        c.add_to_memory([[ 1,  2,  3],[ 4,  5,  7]])
        expected_memory = [[[ 1,  2,  3],[ 4,  5,  6]],
                           [[ 7,  8,  9],[10, 11, 12]],
                           [[ 1,  2,  3],[ 4,  5,  7]]]
        np.testing.assert_allclose(c.memory, expected_memory)

        c.duplicate_threshold = .1  # <- Higher threshold means new entry is considered duplicate
        c.add_to_memory([[ 1,  2,  3],[ 4,  5,  8]])
        expected_memory = [[[ 1,  2,  3],[ 4,  5,  6]],
                           [[ 7,  8,  9],[10, 11, 12]],
                           [[ 1,  2,  3],[ 4,  5,  7]]]
        np.testing.assert_allclose(c.memory, expected_memory)

        c = ContentAddressableMemory(
            initializer=[[[1,2,3], [4,5,6]],
                         [[7,8,9], [10,11,12]],
                         [[7,8,9], [10,11,12]]],
            duplicate_entries_allowed=True,
            seed=module_seed,
        )
        expected_memory = [[[ 1,  2,  3],[ 4,  5,  6]],
                           [[ 7,  8,  9],[10, 11, 12]],
                           [[7,8,9], [10,11,12]]]
        np.testing.assert_allclose(c.memory, expected_memory)
        c.add_to_memory([[ 1,  2,  3],[ 4,  5,  6]])
        expected_memory = [[[ 1,  2,  3],[ 4,  5,  6]],
                           [[ 7,  8,  9],[10, 11, 12]],
                           [[7,8,9], [10,11,12]],
                           [[ 1,  2,  3],[ 4,  5,  6]]]
        np.testing.assert_allclose(c.memory, expected_memory)

    def test_ContentAddressableMemory_weighted_retrieval(self):

        c = ContentAddressableMemory(
            initializer=[[[1,2], [4,5,6]],
                         [[7,8], [10,11,12]]],
            duplicate_entries_allowed=False,
            storage_prob=0.0,
            selection_function=SoftMax,
            seed=module_seed,
        )

        result = c([[1,2],[4,5,6]])
        expected = np.array([[4.06045099, 5.06045099], [7.06045099, 8.06045099, 9.06045099]], dtype=object)
        c.selection_type = 'weighted',
        assert not any(np.testing.assert_allclose(e,r,atol=1e-8) for e,r in zip(expected, result))

    def test_ContentAddressableMemory_overwrite_mode(self):

        c = ContentAddressableMemory(
                initializer=[[[1,2,3], [4,5,6]],
                             [[7,8,9], [10,11,12]]],
                distance_field_weights=[1,0],
                duplicate_entries_allowed=OVERWRITE,
                equidistant_entries_select=RANDOM,
                retrieval_prob = 1.0,
                storage_prob = 1.0,
                seed=module_seed,
        )

        # Add new memory
        retreived = c([[10,11,12], [100,110,120]])
        np.testing.assert_allclose(list(retreived), [[7,8,9], [10,11,12]])
        expected_memory = [[[1,2,3], [4,5,6]],
                           [[7,8,9], [10,11,12]],
                           [[10,11,12], [100,110,120]]]
        np.testing.assert_allclose(c.memory, expected_memory)

        # Overwrite old memory
        retreived = c([[7,8,9], [200,201,202]])
        np.testing.assert_allclose(list(retreived), [[7,8,9], [10,11,12]])
        expected_memory = [[[1,2,3], [4,5,6]],
                           [[7,8,9], [200,201,202]],
                           [[10,11,12], [100,110,120]]]
        np.testing.assert_allclose(c.memory, expected_memory)

        # Allow entry duplicate of memory with
        c.duplicate_entries_allowed = True
        retreived = c([[7,8,9], [300,310,320]])
        np.testing.assert_allclose(list(retreived), [[7,8,9],[200,201,202]])
        expected_memory = [[[1,2,3],[4,5,6]],
                           [[7,8,9], [200,201,202]],
                           [[10,11,12], [100,110,120]],
                           [[7,8,9], [300,310,320]]]
        np.testing.assert_allclose(c.memory, expected_memory)

        # Attempt to overwrite with two matches should generate error
        c.duplicate_entries_allowed = OVERWRITE
        with pytest.raises(FunctionError) as error_text:
            c.execute([[7,8,9], [100,110,120]])
        assert ('Attempt to store item' in str(error_text.value)
                and 'with \'duplicate_entries_allowed\'=\'OVERWRITE\'' in str(error_text.value))
        with pytest.raises(FunctionError) as error_text:
            c.execute([[7,8,9], [300,310,320]])
        assert ('Attempt to store item' in str(error_text.value)
                and 'with \'duplicate_entries_allowed\'=\'OVERWRITE\'' in str(error_text.value))

    def test_ContentAddressableMemory_max_entries(self):

        c = ContentAddressableMemory(
                initializer=[[[1,2,3], [4,5,6]],
                             [[7,8,9], [10,11,12]],
                             [[1,2,3], [100,101,102]]],
                duplicate_entries_allowed=True,
                equidistant_entries_select=RANDOM,
                retrieval_prob = 1.0,
                storage_prob = 1.0,
                max_entries = 4,
                seed=module_seed,
        )
        c.add_to_memory([[[10,20,30],[40,50,60]],
                        [[11,21,31],[41,51,61]],
                        [[12,22,32],[42,52,62]]])
        expected_memory = [[[1,2,3], [100,101,102]],
                           [[10,20,30],[40,50,60]],
                           [[11,21,31],[41,51,61]],
                           [[12,22,32],[42,52,62]]]
        np.testing.assert_allclose(c.memory, expected_memory)

    def test_ContentAddressableMemory_errors_and_warnings(self):

        # Test constructor warnings and errors

        text = "angle of scalars is not defined."
        with pytest.warns(UserWarning, match=text):
            clear_registry(FunctionRegistry)
            c = ContentAddressableMemory(
                default_variable=np.array([[0],[1,2]],dtype=object),
                distance_function=Distance(metric=COSINE),
                seed=module_seed,
            )

        with pytest.raises(ParameterError) as error_text:
            clear_registry(FunctionRegistry)
            c = ContentAddressableMemory(retrieval_prob=32)
        assert 'Value (32) assigned to parameter \'retrieval_prob\' of (ContentAddressableMemory ' \
               'ContentAddressableMemory Function-0).parameters is not valid: ' \
               'must be a float in the interval [0,1].' in str(error_text.value)

        with pytest.raises(ParameterError) as error_text:
            clear_registry(FunctionRegistry)
            c = ContentAddressableMemory(storage_prob=-1)
        assert 'Value (-1) assigned to parameter \'storage_prob\' of (ContentAddressableMemory ' \
               'ContentAddressableMemory Function-0).parameters is not valid: ' \
               'must be a float in the interval [0,1].' in str(error_text.value)

        with pytest.raises(ParameterError) as error_text:
            clear_registry(FunctionRegistry)
            c = ContentAddressableMemory(initializer=[[1,1]],
                                         distance_field_weights=[[1]])
        assert 'Value ([[1]]) assigned to parameter \'distance_field_weights\' of (ContentAddressableMemory ' \
               'ContentAddressableMemory Function-0).parameters is not valid: ' \
               'must be a scalar or list or 1d array of scalars' in str(error_text.value)

        with pytest.raises(ParameterError) as error_text:
            clear_registry(FunctionRegistry)
            c = ContentAddressableMemory(initializer=[[1,1]],
                                         distance_field_weights=[1,2])
        assert 'Value ([1 2]) assigned to parameter \'distance_field_weights\' of (ContentAddressableMemory ' \
               'ContentAddressableMemory Function-0).parameters is not valid: ' \
               'length (2) must be same as number of fields in entries of initializer (1)' in str(error_text.value)

        with pytest.raises(ParameterError) as error_text:
            clear_registry(FunctionRegistry)
            c = ContentAddressableMemory(equidistant_entries_select='HELLO')
        assert "parameters is not valid: must be random or oldest or newest."\
               in str(error_text.value)

        with pytest.raises(ParameterError) as error_text:
            clear_registry(FunctionRegistry)
            c = ContentAddressableMemory(duplicate_entries_allowed='HELLO')
        assert "parameters is not valid: must be a bool or 'OVERWRITE'."\
               in str(error_text.value)

        with pytest.raises(FunctionError) as error_text:
            clear_registry(FunctionRegistry)
            c = ContentAddressableMemory(distance_function=LinearCombination)
        assert "Value returned by 'distance_function' (LinearCombination) specified for ContentAddressableMemory " \
               "must return a scalar if 'distance_field_weights' is not specified or is homogenous " \
               "(i.e., all elements are the same." in str(error_text.value)

        # Test parameter assignment Parameter errors and warnings

        with pytest.raises(ParameterError) as error_text:
            clear_registry(FunctionRegistry)
            c.parameters.retrieval_prob = 2
        assert "Value (2) assigned to parameter 'retrieval_prob' of (ContentAddressableMemory " \
               "ContentAddressableMemory Function-0).parameters is not valid: " \
               "must be a float in the interval [0,1]." in str(error_text.value)

        with pytest.raises(ParameterError) as error_text:
            clear_registry(FunctionRegistry)
            c = ContentAddressableMemory(retrieval_prob=32)
        assert "Value (32) assigned to parameter 'retrieval_prob' of (ContentAddressableMemory " \
               "ContentAddressableMemory Function-0).parameters is not valid: " \
               "must be a float in the interval [0,1]." in str(error_text.value)

        with pytest.raises(ParameterError) as error_text:
            clear_registry(FunctionRegistry)
            c = ContentAddressableMemory(storage_prob=-1)
        assert f"Value (-1) assigned to parameter 'storage_prob' of (ContentAddressableMemory " \
               f"ContentAddressableMemory Function-0).parameters is not valid: " \
               f"must be a float in the interval [0,1]." in str(error_text.value)

        text = ("All weights in the 'distance_fields_weights' Parameter of ContentAddressableMemory Function-0 are "
                "set to '0', no retrieval will occur (equivalent to setting 'retrieval_prob=0.0'.")
        with pytest.warns(UserWarning) as warning:
            clear_registry(FunctionRegistry)
            c = ContentAddressableMemory(initializer=[[1,2],[1,2]],
                                         distance_field_weights=[0,0])
            assert any(text in item.message.args[0] for item in warning.list)

        # Test storage and retrieval Function errors
        with pytest.raises(FunctionError) as error_text:
            clear_registry(FunctionRegistry)
            c = ContentAddressableMemory(initializer=[[1,1],[2,2]])
            c([1,1])
        assert 'Attempt to store and/or retrieve entry in ContentAddressableMemory ([[1 1]]) ' \
               'that has an incorrect number of fields (1; should be 2).' in str(error_text.value)

        clear_registry(FunctionRegistry)
        c = ContentAddressableMemory()
        c([[1,2,3],[4,5,6]])

        with pytest.raises(FunctionError) as error_text:
            c([[[1,2,3],[4,5,6]]])
        assert 'Attempt to store and/or retrieve an entry in ContentAddressableMemory ([[[1 2 3]\n  [4 5 6]]]) that ' \
               'has more than 2 dimensions (3);  try flattening innermost ones.' in str(error_text.value)

        with pytest.raises(FunctionError) as error_text:
            c([[1,2,3],[4,5],[6,7]])
        assert ('Attempt to store and/or retrieve entry in ContentAddressableMemory' in str(error_text.value)
                and 'that has an incorrect number of fields' in str(error_text.value))

        with pytest.raises(FunctionError) as error_text:
            c([[1,2,3],[4,5,6,7]])
        assert "Field 1 of entry ([array([1, 2, 3]) array([4, 5, 6, 7])]) has incorrect shape ((4,)) for memory of " \
               "'ContentAddressableMemory Function-0';  should be: (3,)." in str(error_text.value)

        with pytest.raises(FunctionError) as error_text:
            c.duplicate_entries_allowed = True
            c([[1,2,3],[4,5,6]])
            c.duplicate_entries_allowed = OVERWRITE
            c([[1,2,3],[4,5,6]])
        assert "Attempt to store item ([[1. 2. 3.]\n [4. 5. 6.]]) in ContentAddressableMemory Function-0 with " \
               "'duplicate_entries_allowed'='OVERWRITE' when there is more than one matching entry in its memory; " \
               "'duplicate_entries_allowed' may have previously been set to 'True'" in str(error_text.value)

        with pytest.raises(FunctionError) as error_text:
            clear_registry(FunctionRegistry)
            c.add_to_memory([[[1,2,3],[4,5,6]],
                             [[8,9,10],[11,12,13,14]]])
        assert ("has incorrect shape ((2, 3)) for memory of 'ContentAddressableMemory Function-0';  should be: (3,)."
                in str(error_text.value))

        with pytest.raises(FunctionError) as error_text:
            clear_registry(FunctionRegistry)
            c.add_to_memory([1,2,3])
        assert 'Attempt to store and/or retrieve entry in ContentAddressableMemory ([1 2 3]) ' \
               'that has an incorrect number of fields (3; should be 2).' in str(error_text.value)

        with pytest.raises(FunctionError) as error_text:
            clear_registry(FunctionRegistry)
            c.add_to_memory([[[1]]])
        assert 'Attempt to store and/or retrieve entry in ContentAddressableMemory ([[1]]) ' \
               'that has an incorrect number of fields (1; should be 2).' in str(error_text.value)

        with pytest.raises(FunctionError) as error_text:
            clear_registry(FunctionRegistry)
            c.add_to_memory(1)
        assert "The 'memories' arg for add_to_memory method of must be a list or array containing 1d or 2d arrays " \
               "(was scalar)." in str(error_text.value)

        with pytest.raises(FunctionError) as error_text:
            clear_registry(FunctionRegistry)
            c.add_to_memory([[[[1,2]]]])
        assert "The 'memories' arg for add_to_memory method of must be a list or array containing 1d or 2d arrays " \
               "(was 4d)." in str(error_text.value)

        text = (f"Selection function (SoftMax Function) specified for ContentAddressableMemory Function-0 returns "
                f"more than one item (3) while 'duplicate_entries_allowed'==True. If a weighted sum of entries is intended, "
                f"set 'duplicate_entries_allowed'==False and use a selection function that returns a weighted sum "
                f"(e.g., SoftMax with 'output='ALL').")
        with pytest.warns(UserWarning) as warning:
            clear_registry(FunctionRegistry)
            c = ContentAddressableMemory(initializer=[[1,2],[1,2,3]],
                                         distance_field_weights=[1,0],
                                         duplicate_entries_allowed=True,
                                         selection_function=SoftMax)
            assert any(text in item.message.args[0] for item in warning.list)


    @pytest.mark.parametrize(
        'param_name',
        [
            'distance_function',
            'selection_function',
        ]
    )
    def test_ContentAddressableMemory_unique_functions(self, param_name):
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

    #

        # (
        #     "ContentAddressableMemory Initializer Ndimensional Fields",
        #     # FIX:
        #     # OTHER DATA
        #     [[[1],[[2],[3,4]],[4]],[[1],[[2],[3,4]],[4]]]
        #     [[[1,2,3],[4]],[[1],[[2],[3,4]],[4]]]
        # ),
        # FIX: THESE SHOULD BE IN MemoryFunctions TEST for ContentAddressableMemory
        # (
        #     "ContentAddressableMemory Random Retrieval",
        #     # FIX:
        #     # OTHER DATA
        # ),
        # (
        #     "ContentAddressableMemory Random Storage",
        #     # FIX:
        #     # OTHER DATA
        # ),
        # (
        #     "ContentAddressableMemory Random Retrieval-Storage",
        #     # FIX:
        #     # OTHER DATA
        # ),
        # (
        #     "ContentAddressableMemory Weighted Retrieval",
        #     # FIX:
        #     # OTHER DATA
        # ),
        # (
        #     "ContentAddressableMemory Duplicates Retrieval",
        #     # FIX:
        #     # OTHER DATA
        # ),


        # # Initializer with >2d regular array
        # with pytest.raises(FunctionError) as error_text:
        #     f = ContentAddressableMemory(initializer=[[[[1,0],[1,0],[1,0]], [[1,0],[1,0],[1,0]], [[1,0],[1,0],[1,0]]],
        #                                               [[[0,1],[0,1],[0,1]], [[0,1],[0,0],[1,0]], [[0,1],[0,1],[0,1]]]])
        #     em = EpisodicMemoryMechanism(input_shapes = [1,1,1], function=f)
        #     em.execute([[[0,1],[0,1],[0,1]], [[0,1],[0,0],[1,0]], [[0,1],[0,1],[0,1]]])
        # assert 'Attempt to store and/or retrieve an entry in ContentAddressableMemory that has more than 2 dimensions (' \
        #        '3);  try flattening innermost ones.' in str(error_text.value)
        #
        # # Initializer with >2d ragged array
        # with pytest.raises(FunctionError) as error_text:
        #     f = ContentAddressableMemory(initializer=[ [[1,2,3], [4]], [[1,2,3], [[1],[4]]] ])
        #     em = EpisodicMemoryMechanism(input_shapes = [1,1,1], function=f)
        #     em.execute([[[0,1],[0,1],[0,1]], [[0,1],[0,0],[1,0]], [[0,1],[0,1],[0,1]]])
        # assert 'Attempt to store and/or retrieve an entry in ContentAddressableMemory that has more than 2 dimensions (' \
        #        '3);  try flattening innermost ones.' in str(error_text.value)

        # [ [[1,2,3], [4]], [[1,2,3], [[1],[4]]] ]

#endregion
