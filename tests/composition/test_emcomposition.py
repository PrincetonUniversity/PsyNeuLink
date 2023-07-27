import logging
import timeit as timeit
import os
import numpy as np

import pytest

import psyneulink as pnl

from psyneulink.core.components.functions.nonstateful.transferfunctions import Logistic
from psyneulink.core.components.functions.nonstateful.learningfunctions import BackPropagation
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.globals import Context
from psyneulink.core.globals.keywords import AUTO, CONTROL
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.library.compositions.emcomposition import EMComposition, EMCompositionError
from psyneulink.core.compositions.report import ReportOutput

module_seed = 0
np.random.seed(0)

logger = logging.getLogger(__name__)


# All tests are set to run. If you need to skip certain tests,
# see http://doc.pytest.org/en/latest/skipping.html

# Unit tests for functions of EMComposition class that are new (not in Composition)
# or override functions in Composition

def _single_learn_results(composition, *args, **kwargs):
    composition.learn(*args, **kwargs)
    return composition.learning_results

@pytest.mark.pytorch
@pytest.mark.acconstructor
class TestConstruction:

    def test_two_calls_no_args(self):
        comp = EMComposition()
        comp_2 = EMComposition()
        assert isinstance(comp, EMComposition)
        assert isinstance(comp_2, EMComposition)

    def test_pytorch_representation(self):
        comp = EMComposition()
        assert comp.pytorch_representation is None

    def test_report_prefs(self):
        comp = EMComposition()
        assert comp.input_CIM.reportOutputPref == ReportOutput.OFF
        assert comp.output_CIM.reportOutputPref == ReportOutput.OFF

# memory_template - NUMBER OF ENTRIES AND FORMAT:
# 1) single entry
#    √ memory_template is tuple - 0
#    √ memory_template is list with zeros and same lengths - 1, 2.x
#    √ memory_template is list with zeros and different lengths - 3,4,6,7,11
#    √ memory_template is list with non-zeros and same lengths - 9
#    √ memory_template is np.ndarray with zeros and different lengths - 9.1, 10
#    - memory_template is np.ndarray with zeros and same lengths - FIX
#    - memory_template is np.ndarray with zeros and different lengths FIX
# 2) multiple entries
#    √ memory_template is partial entries with zeros and different lengths
#    √ memory_template is partial entries with non-zeros and different lengths - 13,14,15,16
#    - memory_template is full entries with zeros and different lengths
#    - memory_template is full entries with non-zeros and different lengths - 17

# memory_fill
# √ single value - 10-17
# √ tuple of values (random) 0.2

# field_weights
# √ single value - 5.x
# √ multiple values all same - 6,7,12-14
# √ multiple values different - 8-11, 15-17

# TODO:
# field names

# normalize_memory - True/False

# Execution:
# retrieval:
# 1) concatenation - True/False
# 2) normalization - True/False
# 3) field_weights - same / different
# softmax_gain - None/float/function
# memory_decay_rate - None/AUTO/float
# storage_probability - None, float
# learn_weights - True/False

    # FIX: ADD WARNING TESTS
    # FIX: ADD ERROR TESTS
    test_structure_data = [
        # NOTE: None => use default value (i.e., don't specify in constructor, rather than forcing None as value of arg)
        # ------------------ SPECS ---------------------------------------------   ------- EXPECTED -------------------
        #   memory_template       memory_fill   field_wts cncat_ky nmlze sm_gain   repeat  #fields #keys #vals  concat
        (0,    (2,3),                  None,      None,    None,    None,  None,    False,    2,     1,   1,    False,),
        (0.1,  (2,3),                   .1,       None,    None,    None,  None,    False,    2,     1,   1,    False,),
        (0.2,  (2,3),                 (0,.1),     None,    None,    None,  None,    False,    2,     1,   1,    False,),
        (0.3,  (4,2,3),                 .1,       None,    None,    None,  None,    False,    2,     1,   1,    False,),
        (1,    [[0,0],[0,0]],          None,      None,    None,    None,  None,    False,    2,     1,   1,    False,),
        (1.1,  [[0,0],[0,0]],          None,      [1,1],   None,    None,  None,    False,    2,     2,   0,    False,),
        (2,    [[0,0],[0,0],[0,0]],    None,      None,    None,    None,  None,    False,    3,     2,   1,    False,),
        (2.1,  [[0,0],[0,0],[0,0]],    None,      None,    None,    None,   1.5,    False,    3,     2,   1,    False,),
        (2.2,  [[0,0],[0,0],[0,0]],    None,      None,    None,    None, CONTROL,  False,    3,     2,   1,    False,),
        (3,    [[0,0,0],[0,0]],        None,      None,    None,    None,  None,    False,    2,     1,   1,    False,),
        (4,    [[0,0,0],[0],[0,0]],    None,      None,    None,    None,  None,    False,    3,     2,   1,    False,),
        (5,    [[0,0],[0,0],[0,0]],    None,       1,      None,    None,  None,    False,    3,     3,   0,    False,),
        (5.1,  [[0,0],[0,0],[0,0]],    None,       1,      None,    None,   0.1,    False,    3,     3,   0,    False,),
        (5.2,  [[0,0],[0,0],[0,0]],    None,       1,      None,    None, CONTROL,  False,    3,     3,   0,    False,),
        (6,    [[0,0,0],[0],[0,0]],    None,    [1,1,1],   False,   None,  None,    False,    3,     3,   0,    False,),
        (7,    [[0,0,0],[0],[0,0]],    None,    [1,1,1],   True,    None,  None,    False,    3,     3,   0,    True,),
        (7.1,  [[0,0,0],[0],[0,0]],    None,    [1,1,1],   True ,   False, None,    False,    3,     3,   0,    False,),
        (8,    [[0,0],[0,0],[0,0]],    None,    [1,2,0],   None,    None,  None,    False,    3,     2,   1,    False,),
        (8.1,  [[0,0],[0,0],[0,0]],    None,    [1,2,0],   True,    None,  None,    False,    3,     2,   1,    False,),
        (9,    [[0,1],[0,0],[0,0]],    None,    [1,2,0],   None,    None,  None,    [0,1],    3,     2,   1,    False,),
        (9.1,  [[0,1],[0,0,0],[0,0]],  None,    [1,2,0],   None,    None,  None,    [0,1],    3,     2,   1,    False,),
        (10,   [[0,1],[0,0,0],[0,0]],    .1,    [1,2,0],   None,    None,  None,    [0,1],    3,     2,   1,    False,),
        (11,   [[0,0],[0,0,0],[0,0]],    .1,    [1,2,0],   None,    None,  None,    False,    3,     2,   1,    False,),
        (12,   [[[0,0],[0,0],[0,0]],   # two entries specified, fields all same length, both entries have all 0's
                [[0,0],[0,0],[0,0]]],    .1,    [1,1,1],   None,    None,  None,      2,      3,     3,   0,    False,),
        (12.1, [[[0,0],[0,0,0],[0,0]], # two entries specified, fields have different lenghts, entries all have 0's
                [[0,0],[0,0,0],[0,0]]],  .1,    [1,1,0],   None,    None,  None,      2,      3,     2,   1,    False,),
        (12.2,  [[[0,0],[0,0,0],[0,0]], # two entries specified, first has 0's
                [[0,2],[0,0,0],[0,0]]],  .1,    [1,1,0],   None,    None,  None,      2,      3,     2,   1,    False,),
        (12.3, [[[0,1],[0,0,0],[0,0]], # two entries specified, fields have same weights
                [[0,2],[0,0,0],[0,0]]],  .1,    [1,1,0],   None,    None,  None,      2,      3,     2,   1,    False),
        (13,   [[[0,1],[0,0,0],[0,0]], # two entries specified, fields have same weights, but conccatenate_keys is False
                [[0,2],[0,0,0],[0,0]]],  .1,    [1,1,0],   True,    None,  None,      2,      3,     2,   1,    True),
        (14,   [[[0,1],[0,0,0],[0,0]], # two entries specified, all fields are keys
                [[0,2],[0,0,0],[0,0]]],  .1,    [1,1,1],   None,    None,  None,      2,      3,     3,   0,    False),
        (15,   [[[0,1],[0,0,0],[0,0]], # two entries specified; fields have different weights, constant memory_fill
                [[0,2],[0,0,0],[0,0]]],  .1,    [1,2,0],   None,    None,  None,      2,      3,     2,   1,    False),
        (15.1, [[[0,1],[0,0,0],[0,0]], # two entries specified; fields have different weights, random memory_fill
                [[0,2],[0,0,0],[0,0]]], (0,.1), [1,2,0],   None,    None,  None,      2,      3,     2,   1,    False),
        (16,   [[[0,1],[0,0,0],[0,0]], # three enrtries specified
                [[0,2],[0,0,0],[0,0]],
                [[0,3],[0,0,0],[0,0]]],  .1,     [1,2,0],   None,    None,  None,     3,      3,     2,   1,    False),
        (17,   [[[0,1],[0,0,0],[0,0]], # all four enrtries allowed by memory_capacity specified
                [[0,2],[0,0,0],[0,0]],
                [[0,3],[0,0,0],[0,0]],
                [[0,4],[0,0,0],[0,0]]],  .1,     [1,2,0],   None,    None,  None,      4,      3,     2,   1,    False),
    ]
    args_names = "test_num, memory_template, memory_fill, field_weights, concatenate_keys, normalize_memories, " \
                 "softmax_gain, repeat, num_fields, num_keys, num_values, concatenate_node"
    @pytest.mark.parametrize(args_names,
                             test_structure_data,
                             ids=[x[0] for x in test_structure_data]
                             )
    @pytest.mark.benchmark
    def test_structure(self,
                       test_num,
                       memory_template,
                       memory_fill,
                       field_weights,
                       concatenate_keys,
                       normalize_memories,
                       softmax_gain,
                       repeat,
                       num_fields,
                       num_keys,
                       num_values,
                       concatenate_node,
                       benchmark):
        """Note: weight matrices used for memory are validated by using em.memory, since its getter uses thos matrices
        """

        params = {'memory_template': memory_template}
        # Add explicit argument specifications (to avoid forcing to None in constructor)
        if isinstance(memory_template, tuple) and len(memory_template) == 3:
            # Assign for tests below, but allow it to be inferred in constructor
            memory_capacity = memory_template[0]
        else:
            memory_capacity = 4
            # Specify it explicitly
            params.update({'memory_capacity': memory_capacity})
        if memory_fill is not None:
            params.update({'memory_fill': memory_fill})
        if field_weights is not None:
            params.update({'field_weights': field_weights})
        if concatenate_keys is not None:
            params.update({'concatenate_keys': concatenate_keys})
        if normalize_memories is not None:
            params.update({'normalize_memories': normalize_memories})
        if softmax_gain is not None:
            params.update({'softmax_gain': softmax_gain})

        em = EMComposition(**params)

        # Validate basic structure
        assert len(em.memory) == memory_capacity
        assert len(em.memory[0]) == num_fields
        assert len(em.field_weights) == num_fields
        assert len(em.field_weights) == num_keys + num_values

        # Validate memory_template
        # If tuple spec, ensure that all fields have the same length
        if isinstance(memory_template, tuple):
            if len(memory_template) == 3:
                # If 3-item tuple, ensure that memory_capacity == number of entries specified in first item
                assert len(em.memory) == memory_template[0]
            field_idx = 1 if len(memory_template) == 2 else 2
            assert all(len(em.memory[j][i]) == memory_template[field_idx]
                       for i in range(num_fields) for j in range(memory_capacity))
        # If list or array spec, ensure that all fields have the same length as those in the specified memory_template
        else:
            # memory_template has all zeros, so all fields should be empty
            if not repeat:
                assert all(len(em.memory[j][i]) == len(memory_template[i])
                       for i in range(num_fields) for j in range(memory_capacity))
            # memory_template is a single specified entry:
            elif repeat and isinstance(repeat, list):
                assert all(len(em.memory[k][j]) == len(memory_template[j])
                           for j in range(num_fields) for k in range(memory_capacity))
            # memory_template is multiple entries, so need outer dimension on em.memory for test
            else:
                # ensure all specified entries have correct number of fields
                assert all(len(em.memory[k][j]) == len(memory_template[k][j])
                       for j in range(num_fields) for k in range(repeat))
                # ensure all repeated entries have correct number of fields
                assert all(len(em.memory[k][j]) == len(memory_template[0][j])
                       for j in range(num_fields) for k in range(repeat,memory_capacity))

        # Validate node structure
        assert len(em.key_input_nodes) == num_keys
        assert len(em.value_input_nodes) == num_values
        assert isinstance(em.concatenate_keys_node, Mechanism) == concatenate_node
        if em.concatenate_keys:
            assert em.retrieval_gating_nodes == []
            assert bool(softmax_gain in {None, CONTROL}) == bool(len(em.softmax_control_nodes))
        else:
            if num_keys > 1:
                assert len(em.retrieval_gating_nodes) == num_keys
            else:
                assert em.retrieval_gating_nodes == []
            if softmax_gain in {None, CONTROL}:
                assert len(em.softmax_control_nodes) == num_keys
            else:
                assert em.softmax_control_nodes == []
        assert len(em.retrieval_nodes) == num_fields

        def test_memory_fill(start, memory_fill):
            memory_fill = memory_fill or 0
            for k in range(start, memory_capacity):
                for j in range(num_fields):
                    for i in range(len(em.memory[k][j])):
                        elem = em.memory[k][j][i]
                        # Random fill
                        if isinstance(memory_fill, tuple):
                            assert isinstance(elem, float) and (elem >= memory_fill[0] and elem <= memory_fill[1])
                        # Constant fill
                        else:
                            assert elem == memory_fill

        # Validate specified entries and memory_memory_fill
        # If memory_template is all zeros, ensure that all fields are filled with zeros or memory_fill
        if not repeat:
            test_memory_fill(start=0, memory_fill=memory_fill)

        if isinstance(repeat,list):  # Single entry specification and repeat = item repeated for all entries
            for j in range(num_fields):
                for i in range(len(em.memory[0][j])):
                    np.testing.assert_allclose(em.memory[0][j][i], em.memory[-1][j][i])
            np.testing.assert_allclose(em.memory[-1][0], np.array(repeat,dtype=object).astype(float))
        elif repeat and repeat < memory_capacity:  # Multi-entry specification and repeat = number entries; remainder
            test_memory_fill(start=repeat, memory_fill=memory_fill)


@pytest.mark.pytorch
class TestExecution:

    # TEST:
    # 0: 3 entries that fill memory; no decay, one key, high softmax gain, no storage, inputs has only key (no value)
    # 1: 3 entries that fill memory; no decay, one key, high softmax gain, no storage, inputs has key & value
    # 2:   same as 1 but different value (that should be ignored)
    # 3:   same as 2 but has extra entry filled with random values (which changes retrieval)
    # 4:   same as 3 but uses both fields as keys (no values)
    # 5:   same as 4 but no concatenation of keys (confirms that results are similar w/ and w/o concatenation)
    # 6:   same as 5, but different field_weights
    # 7:   store + no decay
    # 8:   store + default decay (should be AUTO
    # 9:   store + explicit AUTO decay
    # 10:  store + numerical decay

    test_execution_data = [
        # NOTE: None => use default value (i.e., don't specify in constructor, rather than forcing None as value of arg)
        # ---------------------------------------- SPECS -----------------------------------  ----- EXPECTED ---------
        #   memory_template         mem    mem  mem  fld   concat  nlz  sm   str    inputs        expected_retrieval
        #                           fill   cap decay wts    keys       gain  prob
        # ----------------------------------------------------------------------------------  ------------------------
        (0, [[[1,2,3],[4,5,6]],
             [[1,2,5],[4,5,8]],
             [[1,2,10],[4,5,10]]],  None,   3,  0, [1,0],  None, None,  100,  0, [[[1, 2, 3]]], [[1., 2., 3.16585899],
                                                                                                 [4., 5., 6.16540637]]),
        (1, [[[1,2,3],[4,5,6]],
             [[1,2,5],[4,5,8]],
             [[1,2,10],[4,5,10]]],  None,   3,  0, [1,0],  None, None,  100,  0, [[[1, 2, 3]],
                                                                                  [[4, 5, 6]]], [[1., 2., 3.16585899],
                                                                                                 [4., 5., 6.16540637]]),
        (2, [[[1,2,3],[4,5,6]],
             [[1,2,5],[4,5,8]],
             [[1,2,10],[4,5,10]]],  None,   3,  0, [1,0],  None, None,  100,  0, [[[1, 2, 3]],
                                                                                  [[4, 5, 8]]], [[1., 2., 3.16585899],
                                                                                                 [4., 5., 6.16540637]]),
        (3, [[[1,2,3],[4,5,6]],
             [[1,2,5],[4,5,8]],
             [[1,2,10],[4,5,10]]], (0,.01), 4,  0, [1,0],  None, None,  100,  0, [[[1, 2, 3]],
                                                                                   [[4, 5, 8]]], [[0.99998628,
                                                                                                   1.99997247,
                                                                                                   3.1658154 ],
                                                                                                  [3.99994492,
                                                                                                   4.99993115,
                                                                                                   6.16532141]]),
        (4, [[[1,2,3],[4,5,6]],     # Equal field_weights (but not concatenated)
             [[1,2,5],[4,5,8]],
             [[1,2,10],[4,5,10]]], (0,.01), 4,  0, [1,1],  None, None,  100,  0, [[[1, 2, 3]],
                                                                                   [[4, 5, 6]]], [[0.99637453,
                                                                                                   1.99272658,
                                                                                                   3.44135342],
                                                                                                  [3.98544898,
                                                                                                   4.9818115,
                                                                                                   6.38099054]]
         ),
        (5, [[[1,2,3],[4,5,6]],     # Equal field_weights with concatenation
             [[1,2,5],[4,5,8]],
             [[1,2,10],[4,5,10]]], (0,.01), 4,  0, [1,1],  True, None,  100,  0, [[[1, 2, 4]],
                                                                                  [[4, 5, 6]]], [[0.99992176,
                                                                                                  1.99984303,
                                                                                                  3.80435538],
                                                                                                 [3.99968598,
                                                                                                  4.99960748,
                                                                                                  6.79677593]]),
        (6, [[[1,2,3],[4,5,6]],        # Unequal field_weights
             [[1,2,5],[4,5,8]],
             [[1,2,10],[4,5,10]]], (0,.01), 4,  0, [9,1],  None, None,  100,  0, [[[1, 2, 3]],
                                                                                  [[4, 5, 6]]], [[0.99926393,
                                                                                                  1.99852329,
                                                                                                  3.220923],
                                                                                                 [3.99704573,
                                                                                                  4.99630722,
                                                                                                  6.20845524]]),
        (7, [[[1,2,3],[4,5,6]],        # Store + no decay
             [[1,2,5],[4,5,8]],
             [[1,2,10],[4,5,10]]], (0,.01), 4,  0, [9,1],  None, None,  100,  1, [[[1, 2, 3]],
                                                                                  [[4, 5, 6]]], [[0.99926393,
                                                                                                  1.99852329,
                                                                                                  3.220923],
                                                                                                 [3.99704573,
                                                                                                  4.99630722,
                                                                                                  6.20845524]]),
        (8, [[[1,2,3],[4,5,6]],        # Store + default decay (should be AUTO)
             [[1,2,5],[4,5,8]],
             [[1,2,10],[4,5,10]]], (0,.01), 4, None, [9,1],  None, None,  100,  1, [[[1, 2, 3]],
                                                                                    [[4, 5, 6]]], [[0.99926393,
                                                                                                    1.99852329,
                                                                                                    3.220923],
                                                                                                   [3.99704573,
                                                                                                    4.99630722,
                                                                                                    6.20845524]]),
        (9, [[[1,2,3],[4,5,6]],        # Store + explicit AUTO decay
             [[1,2,5],[4,5,8]],
             [[1,2,10],[4,5,10]]], (0,.01), 4, AUTO, [9,1],  None, None,  100,  1, [[[1, 2, 3]],
                                                                                    [[4, 5, 6]]], [[0.99926393,
                                                                                                    1.99852329,
                                                                                                    3.220923],
                                                                                                   [3.99704573,
                                                                                                    4.99630722,
                                                                                                    6.20845524]]),
        (10, [[[1,2,3],[4,5,6]],        # Store + numerical decay
              [[1,2,5],[4,5,8]],
              [[1,2,10],[4,5,10]]], (0,.01), 4, .1, [9,1],  None, None,  100,  1, [[[1, 2, 3]],
                                                                                   [[4, 5, 6]]], [[0.99926393,
                                                                                                   1.99852329,
                                                                                                   3.220923],
                                                                                                  [3.99704573,
                                                                                                   4.99630722,
                                                                                                   6.20845524]]),
        (11, [[[1,2,3],[4,5,6]],    # Same as 10, but with equal weights and concatenate keys
              [[1,2,5],[4,5,8]],
              [[1,2,10],[4,5,10]]], (0,.01), 4, .1, [1,1],  True, None,  100,  1, [[[1, 2, 3]],
                                                                                   [[4, 5, 6]]], [[0.9999599,
                                                                                                   1.99991955,
                                                                                                   3.32054524],
                                                                                                  [3.99983906,
                                                                                                   4.99979883,
                                                                                                   6.3201438]]),
]

    args_names = "test_num, memory_template, memory_fill, memory_capacity, memory_decay_rate, field_weights, " \
                 "concatenate_keys, normalize_memories, softmax_gain, storage_prob, inputs, expected_retrieval"
    @pytest.mark.parametrize(args_names,
                             test_execution_data,
                             ids=[x[0] for x in test_execution_data])
    @pytest.mark.composition
    @pytest.mark.benchmark
    def test_execution(self,
                       comp_mode,
                       test_num,
                       memory_template,
                       memory_capacity,
                       memory_fill,
                       memory_decay_rate,
                       field_weights,
                       concatenate_keys,
                       normalize_memories,
                       softmax_gain,
                       storage_prob,
                       inputs,
                       expected_retrieval):

        if comp_mode != pnl.ExecutionMode.Python:
            pytest.skip('Compilation not yet support for Composition.import.')

        params = {'memory_template': memory_template,
                  'memory_capacity': memory_capacity,
                  }
        # Add explicit argument specifications only for args that are not None
        # (to avoid forcing to None in constructor)
        if memory_fill is not None:
            params.update({'memory_fill': memory_fill})
        if memory_decay_rate is not None:
            params.update({'memory_decay_rate': memory_decay_rate})
        if field_weights is not None:
            params.update({'field_weights': field_weights})
        if concatenate_keys is not None:
            params.update({'concatenate_keys': concatenate_keys})
        if normalize_memories is not None:
            params.update({'normalize_memories': normalize_memories})
        if softmax_gain is not None:
            params.update({'softmax_gain': softmax_gain})
        if storage_prob is not None:
            params.update({'storage_prob': storage_prob})

        em = EMComposition(**params)

        # Construct inputs
        input_nodes = em.key_input_nodes + em.value_input_nodes
        inputs = {input_nodes[i]:inputs[i] for i in range(len(inputs))}

        # Validate any specified initial memories
        np.testing.assert_equal(np.array(em.memory_template[:len(memory_template)]), np.array(memory_template))

        # Execute and validate results
        retrieved = em.run(inputs=inputs, execution_mode=comp_mode)
        np.testing.assert_allclose(retrieved, expected_retrieval)

        # Validate that sum of weighted softmax distributions in retrieval_weighting_node itself sums to 1
        np.testing.assert_allclose(np.sum(em.retrieval_weighting_node.value), 1.0, atol=1e-15)

        # Validate that sum of its output ports also sums to 1
        np.testing.assert_allclose(np.sum([port.value for port in em.retrieval_weighting_node.output_ports]),
                                   1.0, atol=1e-15)

        # Validate storage
        if storage_prob:
            np.testing.assert_array_equal(em.memory[-1],[[1,2,3],[4,5,6]])

            if memory_decay_rate in {None, AUTO}:
                np.testing.assert_array_equal(np.array(memory_template) * (1 / memory_capacity), em.memory[:3])
            elif memory_decay_rate:
                np.testing.assert_array_equal(np.array(memory_template) * memory_decay_rate, em.memory[:3])
            else:
                np.testing.assert_array_equal(memory_template, em.memory[:3])

        elif len(memory_template) < memory_capacity:
            if isinstance(memory_fill, tuple):
                for field in em.memory[-1]:
                    assert all((memory_fill[0] <= elem <= memory_fill[1]) for elem in field)
            else:
                memory_fill = memory_fill or 0
                assert all(elem == memory_fill for elem in em.memory[-1])


# *****************************************************************************************************************
# *************************************  FROM AutodiffComposition  ************************************************
# *****************************************************************************************************************

@pytest.mark.skip(reason="no pytorch representation of EMComposition yet")
@pytest.mark.pytorch
@pytest.mark.composition
def test_autodiff_forward(autodiff_mode):
    """FIX: SHOULD IMPLEMENT CORRESPONDING TESTS FROM AutodiffComposition"""
    pass


@pytest.mark.skip(reason="no pytorch representation of EMComposition yet")
@pytest.mark.pytorch
@pytest.mark.accorrectness
@pytest.mark.composition
class TestTrainingCorrectness:
    """FIX: SHOULD IMPLEMENT CORRESPONDING TESTS FROM AutodiffComposition"""
    pass


@pytest.mark.skip(reason="no pytorch representation of EMComposition yet")
@pytest.mark.pytorch
@pytest.mark.acidenticalness
class TestTrainingIdenticalness():
    """FIX: SHOULD IMPLEMENT CORRESPONDING TESTS FROM AutodiffComposition"""
    pass


@pytest.mark.skip(reason="no pytorch representation of EMComposition yet")
@pytest.mark.pytorch
@pytest.mark.acmisc
@pytest.mark.composition
class TestMiscTrainingFunctionality:
    """FIX: SHOULD IMPLEMENT CORRESPONDING TESTS FROM AutodiffComposition"""
    pass

@pytest.mark.skip(reason="no pytorch representation of EMComposition yet")
@pytest.mark.pytorch
@pytest.mark.actime
class TestTrainingTime:
    """FIX: SHOULD IMPLEMENT CORRESPONDING TESTS FROM AutodiffComposition"""
    pass


@pytest.mark.skip(reason="no pytorch representation of EMComposition yet")
@pytest.mark.pytorch
def test_autodiff_saveload(tmp_path):
    """FIX: SHOULD IMPLEMENT CORRESPONDING TESTS FROM AutodiffComposition"""
    pass


@pytest.mark.skip(reason="no pytorch representation of EMComposition yet")
@pytest.mark.pytorch
@pytest.mark.aclogging
class TestACLogging:
    """FIX: SHOULD IMPLEMENT CORRESPONDING TESTS FROM AutodiffComposition"""
    pass


@pytest.mark.skip(reason="no pytorch representation of EMComposition yet")
@pytest.mark.pytorch
@pytest.mark.acnested
@pytest.mark.composition
class TestNested:
    """FIX: SHOULD IMPLEMENT CORRESPONDING TESTS FROM AutodiffComposition"""
    pass


@pytest.mark.skip(reason="no pytorch representation of EMComposition yet")
@pytest.mark.pytorch
class TestBatching:
    """FIX: SHOULD IMPLEMENT CORRESPONDING TESTS FROM AutodiffComposition"""
    pass
