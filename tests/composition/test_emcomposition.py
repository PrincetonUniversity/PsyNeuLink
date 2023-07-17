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
from psyneulink.core.globals.keywords import TRAINING_SET, Loss
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.library.compositions.emcomposition import EMComposition, EMCompositionError
from psyneulink.core.compositions.report import ReportOutput

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
class TestACConstructor:

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

# TEST PARAMETERS:
# memory_template - NUMBER OF ENTRIES AND FORMAT:
# 1) single entry
#    - memory_template is tuple
#    - memory_template is list with zeros and same lengths
#    - memory_template is list with zeros and different lengths
#    - memory_template is list with non-zeros and same lengths
#    - memory_template is np.ndarray with zeros and different lengths
#    - memory_template is np.ndarray with zeros and same lengths
#    - memory_template is np.ndarray with zeros and different lengths
# 2) multiple entries
#    - memory_template is partial entries with zeros and different lengths
#    - memory_template is partial entries with non-zeros and differnt lengths
#    - memory_template is full entries with zeros and differnt lengths
#    - memory_template is full entries with non-zeros and different lengths

# memory_fill
# 1) single value
# 2) tuple of values (random)

# field_weights
# 1) single value
# 2) multiple values all same
# 3) multiple values different

# field names

# concatenate_keys - True/False

# normalize_memory - True/False

# Execution:
# retrieval:
# 1) concatenation - True/False
# 2) normalization - True/False
# 3) field_weights - same / different
# softmax_gain - None/float/function
# memory_decay - True/False
# storage_probability - None, float
# learn_weights - True/False

    # FIX: ADD SoftMax TESTS:
    # FIX: ADD WARNING TESTS
    # FIX: ADD ERROR TESTS
    # FIX: rtrv_wt_node TESTS
    test_data = [
        # memory_template, field_weights, concatenate_keys, normalize_memory, repeat (entries),
        # num_fields, num_keys, num_values, concatenate_node, retrieval_weighting_nodes,
        # ------------------ SPECS -----------------------------------------   ------------ EXPECTED -------------------
        #  memory_template    memory_fill  field_wts concat_key  normalize  repeat  #field  #key  #val   concat  rtrv_wt
        (0,  (2,3),                 None,    None,      None,      None,    False,     2,     1,   1,    False,  False),
        (1,  [[0,0],[0,0]],         None,    None,      None,      None,    False,     2,     1,   1,    False,  False),
        (2,  [[0,0],[0,0],[0,0]],   None,    None,      None,      None,    False,     3,     2,   1,    True,   False),
        (3,  [[0,0,0],[0,0]],       None,    None,      None,      None,    False,     2,     1,   1,    False,  False),
        (4,  [[0,0,0],[0],[0,0]],   None,    None,      None,      None,    False,     3,     2,   1,    True,   False),
        (5,  [[0,0],[0,0],[0,0]],   None,     1,        None,      None,    False,     3,     3,   0,    True,   False),
        (6,  [[0,0,0],[0],[0,0]],   None,  [1,1,1],     None,      None,    False,     3,     3,   0,    True,   False),
        (7,  [[0,0],[0,0],[0,0]],   None,  [1,2,0],     None,      None,    False,     3,     2,   1,    False,  False),
        (8,  [[0,1],[0,0,0],[0,0]], None,  [1,2,0],     None,      None,    [0,1],     3,     2,   1,    False,  False),
        (9,  [[0,1],[0,0,0],[0,0]],   .1,  [1,2,0],     None,      None,    [0,1],     3,     2,   1,    False,  False),
        (10, [[0,0],[0,0,0],[0,0]],   .1,  [1,2,0],     None,      None,    False,     3,     2,   1,    False,  False),
        (11, [[[0,1],[0,0,0],[0,0]],  # two entries specified, fields have same lengths
              [[0,2],[0,0,0],[0,0]]], .1,  [1,1,0],     None,      None,      2,       3,     2,   1,    True,   False),
        (12, [[[0,1],[0,0,0],[0,0]],  # two entries specified; fields have different lengths
              [[0,2],[0,0,0],[0,0]]], .1,  [1,2,0],     None,      None,      2,       3,     2,   1,    False,  False),
        (13, [[[0,1],[0,0,0],[0,0]],  # three enrtries specified
              [[0,2],[0,0,0],[0,0]],
              [[0,3],[0,0,0],[0,0]]], .1,  [1,2,0],     None,      None,      3,       3,     2,   1,    False,  False),
        (14, [[[0,1],[0,0,0],[0,0]],  # all four enrtries allowed by memory_capacity specified
              [[0,2],[0,0,0],[0,0]],
              [[0,3],[0,0,0],[0,0]],
              [[0,4],[0,0,0],[0,0]]], .1,  [1,2,0],     None,      None,      4,       3,     2,   1,    False,  False),
    ]
    args_names = "test_num, memory_template, memory_fill, field_weights, concatenate_keys, normalize_memories, repeat,"\
                 " num_fields, num_keys, num_values, concatenate_node, retrieval_weighting_nodes"
    @pytest.mark.parametrize(args_names,
                             test_data,
                             ids=[x[0] for x in test_data]
                             )
    @pytest.mark.benchmark
    def test_structure(self,
                       test_num,
                       memory_template,
                       memory_fill,
                       field_weights,
                       concatenate_keys,
                       normalize_memories,
                       repeat,
                       num_fields,
                       num_keys,
                       num_values,
                       concatenate_node,
                       retrieval_weighting_nodes,
                       benchmark):
        """Note: weight matrices used for memory are validated by using em.memory, since its getter uses thos matrices
        """
        memory_capacity = 4
        params = {'memory_template': memory_template,
                  'memory_capacity': memory_capacity,
                  }
        if memory_fill is not None:
            params.update({'memory_fill': memory_fill})
        if field_weights is not None:
            params.update({'field_weights': field_weights})
        if concatenate_keys is not None:
            params.update({'concatenate_keys': concatenate_keys})
        if normalize_memories is not None:
            params.update({'normalize_memories': normalize_memories})

        em = EMComposition(**params)

        # Validate basic structure
        assert len(em.memory) == memory_capacity
        assert len(em.memory[0]) == num_fields
        assert len(em.field_weights) == num_fields
        assert len(em.field_weights) == num_keys + num_values

        # Validate memory_template
        # If tuple spec, ensure that all fields have the same length
        if isinstance(memory_template, tuple):
            assert all(len(em.memory[j][i]) == memory_template[1]
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
        assert len(em.retrieval_nodes) == num_keys + num_values
        # assert len(em.retrieval_weighting_nodes) == retrieval_weighting_nodes
        assert len(em.retrieval_nodes) == num_fields
        assert isinstance(em.concatenate_keys_node, Mechanism) == concatenate_node
        # assert em.retrieval_weighting_node == retrieval_weighting_nodes

        # Validate specified entries and memory_memory_fill
        # If memory_template is all zeros, ensure that all fields are empty
        if not repeat and memory_fill:
            # Random fill
            if isinstance(memory_fill, tuple):
                # elem = em.memory[-1][0][0]
                # assert isinstance(elem, float) and (elem >= memory_fill[0] and elem <= memory_fill[1])
                for k in range(memory_capacity):
                    for j in range(num_fields):
                        for i in range(len(em.memory[k][j])):
                            elem = em.memory[k][j][i]
                            assert isinstance(elem, float) and (elem >= memory_fill[0] and elem <= memory_fill[1])
            # Constant fill
            else:
                for k in range(memory_capacity):
                    for j in range(num_fields):
                        for i in range(len(em.memory[k][j])):
                            assert em.memory[k][j][i] == memory_fill
        if isinstance(repeat,list):  # Single entry specification and repeat = item repeated for all entries
            for j in range(num_fields):
                for i in range(len(em.memory[0][j])):
                    np.testing.assert_allclose(em.memory[0][j][i], em.memory[-1][j][i])
            np.testing.assert_allclose(em.memory[-1][0], np.array(repeat,dtype=object).astype(float))
        elif repeat:  # Multi-entry specification and repeat = number entries; remainder should be memory_fill
            for k in range(repeat, memory_capacity):
                for j in range(num_fields):
                    for i in range(len(em.memory[0][j])):
                        # All items after number specified should be identical and populated with memory_fill
                        np.testing.assert_allclose(em.memory[k][j][i], em.memory[-1][j][i])
            if repeat < memory_capacity:
                memory_fill = memory_fill or 0
                # Check that all non-specified entries are populated with memory_fill
                for entry in em.memory[repeat:]:
                    for field in entry:
                        for item in field:
                            assert item == memory_fill
                for field in em.memory[-1]:
                    for item in field:
                        assert item == memory_fill
        elif memory_fill:
            assert em.memory[-1][0][0] == memory_fill


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
