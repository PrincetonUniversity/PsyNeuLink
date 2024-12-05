import numpy as np

import pytest

import psyneulink as pnl

from psyneulink.core.globals.keywords import AUTO, CONTROL
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.library.compositions.emcomposition import EMComposition, EMCompositionError

# All tests are set to run. If you need to skip certain tests,
# see http://doc.pytest.org/en/latest/skipping.html

# Unit tests for functions of EMComposition class that are new (not in Composition or AutodiffComposition)
# or override functions in those classes
#
# TODO:
#     FIX: ADD WARNING TESTS
#     FIX: ADD ERROR TESTS
#     FIX: ADD TESTS FOR LEARNING COMPONENTS in TestStructure
#     FIX: ADD TESTS FOR ACTUAL CALL TO learn() FOR LEARNING in TestExecution
#     FIX: ENABLE TESTS FOR LEARNING ONCE CONCATENATION IS IMPLEMENTED FOR THAT

@pytest.mark.pytorch
@pytest.mark.autodiff_constructor
class TestConstruction:

    def test_two_calls_no_args(self):
        comp = EMComposition()
        comp_2 = EMComposition()
        assert isinstance(comp, EMComposition)
        assert isinstance(comp_2, EMComposition)

    # def test_pytorch_representation(self):
    #     comp = EMComposition()
    #     assert comp.pytorch_representation is None

    # def test_report_prefs(self):
    #     comp = EMComposition()
    #     assert comp.input_CIM.reportOutputPref == ReportOutput.OFF
    #     assert comp.output_CIM.reportOutputPref == ReportOutput.OFF

    test_structure_data = [
        # NOTE: None => use default value (i.e., don't specify in constructor, rather than forcing None as value of arg)
        # ------------------ SPECS ---------------------------------------------   ------- EXPECTED -------------------
        #   memory_template       memory_fill   field_wts cncat_qy nmlze  sm_gain  repeat  #fields #keys #vals  concat
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
        (8,    [[0,0],[0,0],[0,0]],    None,   [1,2,None], None,    None,  None,    False,    3,     2,   1,    False,),
        (8.1,  [[0,0],[0,0],[0,0]],    None,   [1,2,None], True,    None,  None,    False,    3,     2,   1,    False,),
        (8.2,  [[0,0],[0,0],[0,0]],    None,   [1,1,None], True,    None,  None,    False,    3,     2,   1,    True,),
        (8.3,  [[0,0],[0,0],[0,0]],    None,    [1,1,0],   True,    None,  None,    False,    3,     3,   0,    False,),
        (8.4,  [[0,0],[0,0],[0,0]],    None,    [0,0,0],   True,    None,  None,    False,    3,     3,   0,    True,),
        (9,    [[0,1],[0,0],[0,0]],    None,   [1,2,None], None,    None,  None,    [0,1],    3,     2,   1,    False,),
        (9.1,  [[0,1],[0,0,0],[0,0]],  None,   [1,2,None], None,    None,  None,    [0,1],    3,     2,   1,    False,),
        (10,   [[0,1],[0,0,0],[0,0]],    .1,   [1,2,None], None,    None,  None,    [0,1],    3,     2,   1,    False,),
        (11,   [[0,0],[0,0,0],[0,0]],    .1,   [1,2,None], None,    None,  None,    False,    3,     2,   1,    False,),
        (12,   [[[0,0],[0,0],[0,0]],   # two entries specified, fields all same length, both entries have all 0's
                [[0,0],[0,0],[0,0]]],    .1,    [1,1,1],   None,    None,  None,      2,      3,     3,   0,    False,),
        (12.1, [[[0,0],[0,0,0],[0,0]], # two entries specified, fields have different lenghts, entries all have 0's
                [[0,0],[0,0,0],[0,0]]],  .1,   [1,1,None], None,    None,  None,      2,      3,     2,   1,    False,),
        (12.2,  [[[0,0],[0,0,0],[0,0]], # two entries specified, first has 0's
                [[0,2],[0,0,0],[0,0]]],  .1,   [1,1,None], None,    None,  None,      2,      3,     2,   1,    False,),
        (12.3, [[[0,1],[0,0,0],[0,0]], # two entries specified, fields have same weights, but concatenate is False
                [[0,2],[0,0,0],[0,0]]],  .1,   [1,1,None], None,    None,  None,      2,      3,     2,   1,    False),
        (13,   [[[0,1],[0,0,0],[0,0]], # two entries specified, fields have same weights, and concatenate_queries is True
                [[0,2],[0,0,0],[0,0]]],  .1,   [1,1,None], True,    None,  None,      2,      3,     2,   1,    True),
        (14,   [[[0,1],[0,0,0],[0,0]], # two entries specified, all fields are keys
                [[0,2],[0,0,0],[0,0]]],  .1,    [1,1,1],   None,    None,  None,      2,      3,     3,   0,    False),
        (15,   [[[0,1],[0,0,0],[0,0]], # two entries specified; fields have different weights, constant memory_fill
                [[0,2],[0,0,0],[0,0]]],  .1,   [1,2,None], None,    None,  None,      2,      3,     2,   1,    False),
        (15.1, [[[0,1],[0,0,0],[0,0]], # two entries specified; fields have different weights, random memory_fill
                [[0,2],[0,0,0],[0,0]]], (0,.1),[1,2,None], None,    None,  None,      2,      3,     2,   1,    False),
        (16,   [[[0,1],[0,0,0],[0,0]], # three entries specified
                [[0,2],[0,0,0],[0,0]],
                [[0,3],[0,0,0],[0,0]]],  .1,    [1,2,None], None,    None,  None,     3,      3,     2,   1,    False),
        (17,   [[[0,1],[0,0,0],[0,0]], # all four entries allowed by memory_capacity specified
                [[0,2],[0,0,0],[0,0]],
                [[0,3],[0,0,0],[0,0]],
                [[0,4],[0,0,0],[0,0]]],  .1,    [1,2,None], None,    None,  None,      4,      3,     2,   1,    False),
    ]
    args_names = "test_num, memory_template, memory_fill, field_weights, concatenate_queries, normalize_memories, " \
                 "softmax_gain, repeat, num_fields, num_keys, num_values, concatenate_node"
    @pytest.mark.parametrize(args_names,
                             test_structure_data,
                             ids=[x[0] for x in test_structure_data]
                             )
    @pytest.mark.parametrize('enable_learning', [False, True], ids=['no_learning','learning'])
    def test_structure(self,
                       test_num,
                       enable_learning,
                       memory_template,
                       memory_fill,
                       field_weights,
                       concatenate_queries,
                       normalize_memories,
                       softmax_gain,
                       repeat,
                       num_fields,
                       num_keys,
                       num_values,
                       concatenate_node):
        """Note: weight matrices used for memory are validated by using em.memory, since its getter uses those matrices
        """

        # Restrict testing of learning configurations (which are much larger) to select tests
        if enable_learning and test_num not in {2, 2.2, 4, 8}:
            pytest.skip('Limit tests of learning to subset of parametrizations (for efficiency)')

        params = {'memory_template': memory_template,
                  'enable_learning': enable_learning}
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
        if concatenate_queries is not None:
            params.update({'concatenate_queries': concatenate_queries})
            # FIX: DELETE THE FOLLOWING ONCE CONCATENATION IS IMPLEMENTED FOR LEARNING
            params.update({'enable_learning': False})
        if normalize_memories is not None:
            params.update({'normalize_memories': normalize_memories})
        if softmax_gain is not None:
            params.update({'softmax_gain': softmax_gain})

        em = EMComposition(**params)
        assert np.hstack(np.array(em.memory, dtype=object).flatten()).size < 30

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
        assert len(em.query_input_nodes) == num_keys
        assert len(em.value_input_nodes) == num_values
        assert isinstance(em.concatenate_queries_node, Mechanism) == concatenate_node
        if em.concatenate_queries:
            assert em.field_weight_nodes == []
            assert bool(softmax_gain == CONTROL) == bool(em.softmax_gain_control_node)
        else:
            if num_keys > 1:
                assert len(em.field_weight_nodes) == num_keys
            else:
                assert em.field_weight_nodes == []
            if softmax_gain == CONTROL:
                assert em.softmax_gain_control_node
        assert len(em.retrieved_nodes) == num_fields

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

    @pytest.mark.parametrize("softmax_choice, expected",
                             [(pnl.WEIGHTED_AVG, [[0.93016008, 0.1, 0.16983992]]),
                              (pnl.ARG_MAX, [[1, .1, .1]]),
                              (pnl.PROBABILISTIC, [[1, .1, .1]]), # NOTE: actual stochasticity not tested here
                             ])
    def test_softmax_choice(self, softmax_choice, expected):
        em = EMComposition(memory_template=[[[1,.1,.1]], [[1,.1,.1]], [[.1,.1,1]]],
                           softmax_choice=softmax_choice,
                           enable_learning=False)
        result = em.run(inputs={em.query_input_nodes[0]:[[1,0,0]]})

        np.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize("softmax_choice", [pnl.ARG_MAX, pnl.PROBABILISTIC])
    def test_softmax_choice_error(self, softmax_choice):
        em = EMComposition(memory_template=[[[1, .1, .1]], [[.1, 1, .1]], [[.1, .1, 1]]])
        msg = (f"The ARG_MAX and PROBABILISTIC options for the 'softmax_choice' arg "
               f"of '{em.name}' cannot be used during learning; change to WEIGHTED_AVG.")

        with pytest.raises(EMCompositionError, match=msg):
            em.parameters.softmax_choice.set(softmax_choice)
            em.learn()

        for softmax_choice in [pnl.ARG_MAX, pnl.PROBABILISTIC]:
            with pytest.warns(UserWarning) as warning:
                em = EMComposition(softmax_choice=softmax_choice, enable_learning=True)
                warning_msg = (f"The 'softmax_choice' arg of '{em.name}' is set to '{softmax_choice}' with "
                               f"'enable_learning' set to True; this will generate an error if its "
                               f"'learn' method is called. Set 'softmax_choice' to WEIGHTED_AVG before learning.")
            assert warning_msg in str(warning[0].message)

    def test_fields_arg(self):

        em = EMComposition(memory_template=(5,1),
                           memory_capacity=1,
                           normalize_field_weights=False,
                           fields={'A': (1.2, 3.4, True),
                                   'B': (None, False, True),
                                   'C': (0, True, True),
                                   'D': (7.8, False, True),
                                   'E': (5.6, True, True)})
        assert em.num_fields == 5
        assert em.num_keys == 4
        assert (em.field_weights == [1.2, None, 0, 7.8, 5.6]).all()
        assert (em.learn_field_weights == [3.4, False, True, False, True]).all()
        np.testing.assert_allclose(em.target_fields, [True, True, True, True, True])

        # # Test wrong number of entries
        with pytest.raises(EMCompositionError) as error_text:
            EMComposition(memory_template=(3,1), memory_capacity=1, fields={'A': (1.2, 3.4)})
        assert error_text.value.error_value == (f"The number of entries (1) in the dict specified in the 'fields' arg "
                                                f"of 'EM_Composition' does not match the number of fields in its "
                                                f"memory (3).")
        # Test dual specification of fields and corresponding args and learning specified for value field
        with pytest.warns(UserWarning) as warning:
            EMComposition(memory_template=(2,1),
                          memory_capacity=1,
                          fields={'A': (1.2, 3.4, True),
                                  'B': (None, True, True)},
                          field_weights=[10, 11.0])
        warning_msg_1 = (f"The 'fields' arg for 'EM_Composition' was specified, so any of the 'field_names', "
                         f"'field_weights',  'learn_field_weights' or 'target_fields' args will be ignored.")
        warning_msg_2 = (f"Learning was specified for field 'B' in the 'learn_field_weights' arg for "
                         f"'EM_Composition', but it is not allowed for value fields; it will be ignored.")
        assert warning_msg_1 in str(warning[0].message)
        assert warning_msg_2 in str(warning[1].message)



    field_names = ['KEY A','VALUE A', 'KEY B','KEY VALUE','VALUE LEARN']
    field_weights = [1, None, 2, 0, None]
    learn_field_weights = [True, False, .01, False, False]
    target_fields = [True, False, False, True, True]
    dict_subdict = {}
    for i, fn in enumerate(field_names):
        dict_subdict[fn] = {pnl.FIELD_WEIGHT: field_weights[i],
                            pnl.LEARN_FIELD_WEIGHT: learn_field_weights[i],
                            pnl.TARGET_FIELD: target_fields[i]}
    dict_tuple = {fn:(fw,lfw,tf) for fn,fw,lfw,tf in zip(field_names,
                                                         field_weights,
                                                         learn_field_weights,
                                                         target_fields)}
    test_field_map_and_args_assignment_data = [
        ('args', None, field_names, field_weights, learn_field_weights, target_fields),
        ('dict-subdict', dict_subdict, None, None, None, None),
        ('dict-tuple', dict_tuple, None, None, None, None)]
    field_arg_names = "format, fields, field_names, field_weights, learn_field_weights, target_fields"

    @pytest.mark.parametrize(field_arg_names, test_field_map_and_args_assignment_data,
                             ids=[x[0] for x in test_field_map_and_args_assignment_data])
    def test_field_args_and_map_assignments(self,
                                            format,
                                            fields,
                                            field_names,
                                            field_weights,
                                            learn_field_weights,
                                            target_fields):
        # individual args
        em = EMComposition(memory_template=(5,2),
                           memory_capacity=2,
                           fields=fields,
                           field_names=field_names,
                           field_weights=field_weights,
                           learn_field_weights=learn_field_weights,
                           target_fields=target_fields,
                           learning_rate=0.5)
        assert em.num_fields == 5
        assert em.num_keys == 3
        for actual, expected in zip(em.field_weights, [0.33333333, None, 0.66666667, 0, None]):
            if expected is None:
                assert actual is None
            else:
                np.testing.assert_allclose(actual, expected)

        # Validate targets for target_fields
        np.testing.assert_allclose(em.target_fields, [True, False, False, True, True])
        learning_components = em.infer_backpropagation_learning_pathways(pnl.ExecutionMode.PyTorch)
        assert len(learning_components) == 3
        assert 'TARGET for KEY A [RETRIEVED]' in learning_components[0].name
        assert 'TARGET for KEY VALUE [RETRIEVED]' in learning_components[1].name
        assert 'TARGET for VALUE LEARN [RETRIEVED]' in learning_components[2].name

        # Validate learning specs for field weights
        # Presence or absence of field weight components based on keys vs. values:
        assert ['KEY A [WEIGHT]' in node.name for node in em.nodes]
        assert ['KEY B [WEIGHT]' in node.name for node in em.nodes]
        assert ['KEY VALUE [WEIGHT]' in node.name for node in em.nodes]
        assert not any('VALUE A [WEIGHT]' in node.name for node in em.nodes)
        assert not any('VALUE LEARN [WEIGHT]' in node.name for node in em.nodes)
        assert not any('WEIGHT to WEIGHTED MATCH for VALUE A' in proj.name for proj in em.projections)
        assert not any('WEIGHT to WEIGHTED MATCH for VALUE LEARN' in proj.name for proj in em.projections)
        # Learnability and learning rate for field weights
        # FIX: ONCE LEARNING IS FULLY IMPLEMENTED FOR FIELD WEIGHTS, VALIDATE THAT:
        #      KEY A USES COMPOSITION DEFAULT LEARNING RATE OF .5
        #      KEY B USES INDIVIDUALLY ASSIGNED LEARNING RATE OF .01
        assert em.learn_field_weights == [True, False, .01, False, False]
        assert em.projections['WEIGHT to WEIGHTED MATCH for KEY A'].learnable
        assert em.projections['WEIGHT to WEIGHTED MATCH for KEY B'].learnable
        assert not em.projections['WEIGHT to WEIGHTED MATCH for KEY VALUE'].learnable

        # Validate _field_index_map
        assert em._field_index_map[[k for k in em._field_index_map.keys()
                                    if ('MappingProjection from KEY A [QUERY][OutputPort-0] to STORE[InputPort-0]')
                                    in k.name][0]]==0
        assert em._field_index_map[[k for k in em._field_index_map.keys() if 'KEY A [QUERY]' in k.name][0]]==0
        assert em._field_index_map[[k for k in em._field_index_map.keys() if 'KEY A [MATCH to KEYS]' in k.name][0]]==0
        assert em._field_index_map[[k for k in em._field_index_map.keys() if 'KEY A [WEIGHTED MATCH]' in k.name][0]]==0
        assert em._field_index_map[[k for k in em._field_index_map.keys() if 'KEY A [RETRIEVED]' in k.name][0]]==0
        assert em._field_index_map[[k for k in em._field_index_map.keys() if 'MEMORY FOR KEY A [RETRIEVE KEY]'
                                    in k.name][0]]==0
        assert em._field_index_map[[k for k in em._field_index_map.keys() if 'VALUE A [VALUE]' in k.name][0]] == 1
        assert em._field_index_map[[k for k in em._field_index_map.keys() if
                                    ('VALUE A [VALUE][OutputPort-0] to STORE[InputPort-1]') in k.name][0]] == 1
        assert em._field_index_map[[k for k in em._field_index_map.keys() if 'VALUE A [RETRIEVED]' in k.name][0]] == 1
        assert em._field_index_map[[k for k in em._field_index_map.keys()
                                    if 'MEMORY FOR VALUE A' in k.name][0]] == 1
        assert em._field_index_map[[k for k in em._field_index_map.keys() if 'KEY B [QUERY]' in k.name][0]] == 2
        assert em._field_index_map[[k for k in em._field_index_map.keys()
                                    if ('KEY B [QUERY][OutputPort-0] to STORE[InputPort-2]') in k.name][0]] == 2
        assert em._field_index_map[[k for k in em._field_index_map.keys() if 'KEY B [RETRIEVED]' in k.name][0]] == 2
        assert (em._field_index_map[[k for k in em._field_index_map.keys()
                                     if 'MEMORY FOR KEY B [RETRIEVE KEY]' in k.name][0]] == 2)
        assert em._field_index_map[[k for k in em._field_index_map.keys() if 'KEY VALUE [QUERY]' in k.name][0]] == 3
        assert em._field_index_map[[k for k in em._field_index_map.keys()
                                    if 'KEY VALUE [QUERY][OutputPort-0] to STORE[InputPort-3]' in k.name][0]] == 3
        assert em._field_index_map[[k for k in em._field_index_map.keys() if 'KEY VALUE [RETRIEVED]' in k.name][0]] == 3
        assert em._field_index_map[[k for k in em._field_index_map.keys()
                                    if 'MEMORY FOR KEY VALUE [RETRIEVE KEY]' in k.name][0]] == 3
        assert em._field_index_map[[k for k in em._field_index_map.keys() if 'VALUE LEARN [VALUE]' in k.name][0]] == 4
        assert em._field_index_map[[k for k in em._field_index_map.keys()
                                    if 'VALUE LEARN [VALUE][OutputPort-0] to STORE[InputPort-4]' in k.name][0]] == 4
        assert (em._field_index_map[[k for k in em._field_index_map.keys()
                                     if 'VALUE LEARN [RETRIEVED]' in k.name][0]] == 4)
        assert em._field_index_map[[k for k in em._field_index_map.keys() if 'VALUE LEARN [VALUE]' in k.name][0]] == 4
        assert em._field_index_map[[k for k in em._field_index_map.keys()
                                    if 'MEMORY FOR VALUE LEARN [RETRIEVE VALUE]' in k.name][0]] == 4
        assert (em._field_index_map[[k for k in em._field_index_map.keys()
                                     if 'MEMORY for KEY A [KEY]' in k.name][0]] == 0)
        assert em._field_index_map[[k for k in em._field_index_map.keys()
                                    if 'MATCH to WEIGHTED MATCH for KEY A' in k.name][0]] == 0
        assert em._field_index_map[[k for k in em._field_index_map.keys()
                                    if 'WEIGHTED MATCH for KEY A to COMBINE MATCHES' in k.name][0]] == 0
        assert em._field_index_map[[k for k in em._field_index_map.keys() if 'KEY B [MATCH to KEYS]' in k.name][0]] == 2
        assert em._field_index_map[[k for k in em._field_index_map.keys()
                                    if 'MEMORY for KEY B [KEY]' in k.name][0]] == 2
        assert em._field_index_map[[k for k in em._field_index_map.keys()
                                    if 'MATCH to WEIGHTED MATCH for KEY B' in k.name][0]] == 2
        assert (em._field_index_map[[k for k in em._field_index_map.keys()
                                     if 'KEY B [WEIGHTED MATCH]' in k.name][0]] == 2)
        assert em._field_index_map[[k for k in em._field_index_map.keys()
                                    if 'WEIGHTED MATCH for KEY B to COMBINE MATCHES' in k.name][0]] == 2
        assert (em._field_index_map[[k for k in em._field_index_map.keys()
                                     if 'KEY VALUE [MATCH to KEYS]' in k.name][0]] == 3)
        assert em._field_index_map[[k for k in em._field_index_map.keys() if
                                    'MEMORY for KEY VALUE [KEY]' in k.name][0]] == 3
        assert em._field_index_map[[k for k in em._field_index_map.keys()
                                    if 'MATCH to WEIGHTED MATCH for KEY VALUE' in k.name][0]] == 3
        assert (em._field_index_map[[k for k in em._field_index_map.keys()
                                     if 'KEY VALUE [WEIGHTED MATCH]' in k.name][0]] == 3)
        assert em._field_index_map[[k for k in em._field_index_map.keys()
                                    if 'WEIGHTED MATCH for KEY VALUE to COMBINE MATCHES' in k.name][0]] == 3
        assert em._field_index_map[[k for k in em._field_index_map.keys() if 'KEY B [WEIGHT]' in k.name][0]] == 2
        assert em._field_index_map[[k for k in em._field_index_map.keys() if 'KEY VALUE [WEIGHT]' in k.name][0]] == 3
        assert em._field_index_map[[k for k in em._field_index_map.keys()
                                    if 'WEIGHT to WEIGHTED MATCH for KEY VALUE' in k.name][0]] == 3
        assert em._field_index_map[[k for k in em._field_index_map.keys()
                                    if 'WEIGHT to WEIGHTED MATCH for KEY A' in k.name][0]] == 0
        assert em._field_index_map[[k for k in em._field_index_map.keys()
                                    if 'WEIGHT to WEIGHTED MATCH for KEY B' in k.name][0]] == 2

    @pytest.mark.parametrize('field_weight_1', ([None], [0], [1]),  ids=['None', '0', '1'])
    @pytest.mark.parametrize('field_weight_2', ([None], [0], [1]),  ids=['None', '0', '1'])
    @pytest.mark.parametrize('field_weight_3', ([None], [0], [1]),  ids=['None', '0', '1'])
    def test_order_fields_in_memory(self, field_weight_1, field_weight_2, field_weight_3):
        """Test that order of keys and values doesn't matter"""

        # pytest.skip(<UNECESSARY TESTS>>)

        def construct_em(field_weights):
            return pnl.EMComposition(memory_template=[[[5,0], [5], [5,0,3]], [[20,0], [20], [20,1,199]]],
                                     memory_capacity=4,
                                     field_weights=field_weights)

        field_weights = field_weight_1 + field_weight_2 + field_weight_3
        em = None

        if all([fw is None for fw in field_weights]):
            with pytest.raises(EMCompositionError) as error_text:
                construct_em(field_weights)
            assert ("The entries in 'field_weights' arg for EM_Composition can't all be 'None' "
                    "since that will preclude the construction of any keys." in str(error_text.value))

        elif not any(field_weights):
            with pytest.warns(UserWarning) as warning:
                em = construct_em(field_weights)
            warning_msg = ("All of the entries in the 'field_weights' arg for EM_Composition "
                           "are either None or set to 0; this will result in no retrievals "
                           "unless/until one or more of them are changed to a positive value.")
            assert warning_msg in str(warning[0].message)

        elif any([fw == 0 for fw in field_weights]):
            with pytest.warns(UserWarning) as warning:
                em = construct_em(field_weights)
            warning_msg = ("Some of the entries in the 'field_weights' arg for EM_Composition are set to 0; those "
                           "fields will be ignored during retrieval unless/until they are changed to a positive value.")
            assert warning_msg in str(warning[0].message)

        else:
            em = construct_em(field_weights)

        if em:
            for field_weight, field in zip(field_weights, em.fields):
                # Validate proper field-type assignments
                if field_weight is None:
                    assert field.type == pnl.FieldType.VALUE
                else:
                    assert field.type == pnl.FieldType.KEY
                # Validate alignment of field with memory
                assert len(field.memories[0]) == [2,1,3][field.index]



@pytest.mark.pytorch
class TestExecution:

    test_execution_data = [
        # NOTE: None => use default value (i.e., don't specify in constructor, rather than forcing None as value of arg)
        # ---------------------------------------- SPECS -----------------------------------  ----- EXPECTED ---------
        #   memory_template         mem    mem  mem  fld   concat  nlz  sm   str    inputs        expected_retrieval
        #                           fill   cap decay wts    keys   mem gain  prob
        # ----------------------------------------------------------------------------------  ------------------------
        # (0, [[[1,2,3],[4,6]],
        #      [[1,2,5],[4,8]],
        #      [[1,2,10],[4,10]]],    None,   3,  0, [1,None], None, None,  100,  0, [[[1, 2, 3]]], [[1., 2., 3.16585899],
        #                                                                                            [4., 6.16540637]]),
        # (1, [[[1,2,3],[4,6]],
        #      [[1,2,5],[4,8]],
        #      [[1,2,10],[4,10]]],  None,   3,  0, [1,None], None, None,  100,  0,   [[1, 2, 3],
        #                                                                             [4, 6]],      [[1., 2., 3.16585899],
        #                                                                                            [4., 6.16540637]]),
        # (2, [[[1,2,3],[4,6]],
        #      [[1,2,5],[4,8]],
        #      [[1,2,10],[4,10]]],  None,   3,  0, [1,None], None, None,  100,  0,   [[1, 2, 3],
        #                                                                             [4, 8]],     [[1., 2., 3.16585899],
        #                                                                                           [4., 6.16540637]]),
        # (3, [[[1,2,3],[4,6]],
        #      [[1,2,5],[4,8]],
        #      [[1,2,10],[4,10]]], (0,.01), 4,  0, [1,None],  None, None,  100,  0, [[1, 2, 3],
        #                                                                            [4, 8]],      [[0.99998628,
        #                                                                                            1.99997247,
        #                                                                                            3.1658154 ],
        #                                                                                           [3.99994492,
        #                                                                                            6.16532141]]),
        # (4, [[[1,2,3],[4,6]],     # Equal field_weights (but not concatenated)
        #      [[1,2,5],[4,6]],
        #      [[1,2,10],[4,6]]], (0,.01), 4,  0, [1,1],  None, None,  100,  0, [[1, 2, 3],
        #                                                                            [4, 6]],     [[0.99750462,
        #                                                                                           1.99499376,
        #                                                                                           3.51623568],
        #                                                                                          [3.98998465,
        #                                                                                           5.9849743]]
        #  ),
        # (5, [[[1,2,3],[4,6]],     # Equal field_weights with concatenation
        #      [[1,2,5],[4,8]],
        #      [[1,2,10],[4,10]]], (0,.01), 4,  0, [1,1],  True, None,  100,  0, [[1, 2, 4],
        #                                                                           [4, 6]],      [[0.99898504,
        #                                                                                           1.99796378,
        #                                                                                           4.00175037],
        #                                                                                          [3.99592639,
        #                                                                                           6.97406456]]),
        # (6, [[[1,2,3],[4,6]],        # Unequal field_weights
        #      [[1,2,5],[4,8]],
        #      [[1,2,10],[4,10]]], (0,.01), 4,  0, [9,1],  None, None,  100,  0, [[1, 2, 3],
        #                                                                           [4, 6]],      [[0.99996025,
        #                                                                                           1.99992024,
        #                                                                                           3.19317783],
        #                                                                                          [3.99984044,
        #                                                                                           6.19219795]]),
        # (7, [[[1,2,3],[4,6]],        # Store + no decay
        #      [[1,2,5],[4,8]],
        #      [[1,2,10],[4,10]]], (0,.01), 4,  0, [9,1],  None, None,  100,  1, [[1, 2, 3],
        #                                                                           [4, 6]],      [[0.99996025,
        #                                                                                           1.99992024,
        #                                                                                           3.19317783],
        #                                                                                          [3.99984044,
        #                                                                                           6.19219795]]),
        # (8, [[[1,2,3],[4,6]],        # Store + default decay (should be AUTO)
        #      [[1,2,5],[4,8]],
        #      [[1,2,10],[4,10]]], (0,.01), 4, None, [9,1],  None, None,  100,  1,[[1, 2, 3],
        #                                                                             [4, 6]],   [[0.99996025,
        #                                                                                          1.99992024,
        #                                                                                          3.19317783],
        #                                                                                          [3.99984044,
        #                                                                                           6.19219795]]),
        # (9, [[[1,2,3],[4,6]],        # Store + explicit AUTO decay
        #      [[1,2,5],[4,8]],
        #      [[1,2,10],[4,10]]], (0,.01), 4, AUTO, [9,1],  None, None,  100,  1, [[1, 2, 3],
        #                                                                           [4, 6]],      [[0.99996025,
        #                                                                                           1.99992024,
        #                                                                                           3.19317783],
        #                                                                                          [3.99984044,
        #                                                                                           6.19219795]]),
        (10, [[[1,2,3],[4,6]],        # Store + numerical decay
              [[1,2,5],[4,8]],
              [[1,2,10],[4,10]]], (0,.01), 4, .1, [9,1],  None, None,  100,  1, [[1, 2, 3],
                                                                                 [4, 6]],       [[0.99996025,
                                                                                                  1.99992024,
                                                                                                  3.19317783],
                                                                                                 [3.99984044,
                                                                                                  6.19219795]]),
        (11, [[[1,2,3],[4,6]],    # Same as 10, but with equal weights and concatenate keys
              [[1,2,5],[4,8]],
              [[1,2,10],[4,10]]], (0,.01), 4, .1, [1,1],  True, None,  100,  1, [[1, 2, 3],
                                                                                 [4, 6]],       [[0.99922544,
                                                                                                  1.99844608,
                                                                                                  3.38989346],
                                                                                                 [3.99689126,
                                                                                                  6.38682264]]),

        (12, [[[1],[2],[3]],    # Scalar keys - exact match  (this tests use of L0 for retreieval in MEMORY matrix)
              [[10],[0],[100]]], (0,.01), 3, 0, [1,1,None], None, None, pnl.ARG_MAX, 1, [[10],[0],[100]],
                                                                                                   [[10],[0],[100]]),

        (13, [[[1],[2],[3]],    # Scalar keys - close match  (this tests use of L0 for retreieval in MEMORY matrix
              [[10],[0],[100]]], (0,.01), 3, 0, [1,1,None], None, None, pnl.ARG_MAX, 1, [[2],[3],[4]], [[1],[2],[3]]),
]

    args_names = "test_num, memory_template, memory_fill, memory_capacity, memory_decay_rate, field_weights, " \
                 "concatenate_queries, normalize_memories, softmax_gain, storage_prob, inputs, expected_retrieval"
    @pytest.mark.parametrize(args_names,
                             test_execution_data,
                             ids=[x[0] for x in test_execution_data])
    @pytest.mark.parametrize('learn_field_weights', [False, True], ids=['no_learning','learning'])
    @pytest.mark.composition
    @pytest.mark.parametrize('exec_mode', [pnl.ExecutionMode.Python, pnl.ExecutionMode.PyTorch],
                             ids=['Python','PyTorch'])
    def test_simple_execution_without_learning(self,
                                               exec_mode,
                                               learn_field_weights,
                                               test_num,
                                               memory_template,
                                               memory_capacity,
                                               memory_fill,
                                               memory_decay_rate,
                                               field_weights,
                                               concatenate_queries,
                                               normalize_memories,
                                               softmax_gain,
                                               storage_prob,
                                               inputs,
                                               expected_retrieval):

        # # if comp_mode not in {pnl.ExecutionMode.Python, pnl.ExecutionMode.PyTorch}:
        # #     pytest.skip('Execution of EMComposition not yet supported for LLVM Mode.')

        # Restrict testing of learning configurations (which are much larger) to select tests
        if learn_field_weights and test_num not in {10}:
            pytest.skip('Limit tests of learning to subset of parametrizations (for efficiency)')

        params = {'memory_template': memory_template,
                  'memory_capacity': memory_capacity,
                  'learn_field_weights': learn_field_weights,
                  }
        # Add explicit argument specifications only for args that are not None
        # (to avoid forcing to None in constructor)
        if memory_fill is not None:
            params.update({'memory_fill': memory_fill})
        if memory_decay_rate is not None:
            params.update({'memory_decay_rate': memory_decay_rate})
        if field_weights is not None:
            params.update({'field_weights': field_weights})
        if concatenate_queries is not None:
            params.update({'concatenate_queries': concatenate_queries})
            # FIX: DELETE THE FOLLOWING ONCE CONCATENATION IS IMPLEMENTED FOR LEARNING
            params.update({'learn_field_weights': False})
        if normalize_memories is not None:
            params.update({'normalize_memories': normalize_memories})
        if softmax_gain is not None:
            if softmax_gain == pnl.ARG_MAX:
                params.update({'softmax_choice': softmax_gain})
                params.update({'softmax_gain': 100})
            else:
                params.update({'softmax_gain': softmax_gain})
        if storage_prob is not None:
            params.update({'storage_prob': storage_prob})
        params.update({'softmax_threshold': None})
        # FIX: ADD TESTS FOR VALIDATION USING SOFTMAX_THRESHOLD

        em = EMComposition(**params)

        # Construct inputs
        input_nodes = em.query_input_nodes + em.value_input_nodes
        inputs = {input_nodes[i]:inputs[i] for i in range(len(inputs))}

        # Validate any specified initial memories
        for i in range(len(memory_template)):
            for j in range(len(memory_template[i])):
                np.testing.assert_equal(em.memory_template[i][j], memory_template[i][j])

        # Execute and validate results
        retrieved = em.run(inputs=inputs, execution_mode=exec_mode)
        for retrieved, expected in zip(retrieved, expected_retrieval):
            np.testing.assert_allclose(retrieved, expected)

        # Validate that sum of weighted softmax distributions in field_weight_node itself sums to 1
        np.testing.assert_allclose(np.sum(em.softmax_node.value), 1.0, atol=1e-15)

        # Validate that sum of its output ports also sums to 1
        np.testing.assert_allclose(np.sum([port.value for port in em.softmax_node.output_ports]),
                                   1.0, atol=1e-15)

        # Validate storage
        if storage_prob:
            # for actual, expected in zip(em.memory[-1], [[1,2,3],[4,6]]):
            for actual, expected in zip(em.memory[-1], list(inputs.values())):
                np.testing.assert_array_equal(actual, expected)

            if memory_decay_rate in {None, AUTO}:
                for expected, actual in zip(memory_template, em.memory[:3]):
                    for expected_item, actual_item in zip(expected,actual):
                        np.testing.assert_array_equal(np.array(expected_item)  * (1 / memory_capacity), actual_item)
            elif memory_decay_rate:
                for expected, actual in zip(memory_template, em.memory[:3]):
                    for expected_item, actual_item in zip(expected,actual):
                        np.testing.assert_array_equal(np.array(expected_item) * memory_decay_rate, actual_item)
            else:
                for actual, expected in zip(em.memory[:3], memory_template):
                    for actual_item, expected_item in zip(actual, expected):
                        np.testing.assert_array_equal(actual_item, expected_item)

        elif len(memory_template) < memory_capacity:
            if isinstance(memory_fill, tuple):
                for field in em.memory[-1]:
                    assert all((memory_fill[0] <= elem <= memory_fill[1]) for elem in field)
            else:
                memory_fill = memory_fill or 0
                assert all(elem == memory_fill for elem in em.memory[-1])

    @pytest.mark.parametrize('test_field_weights_0_vs_None_data',
                             (([[[5], [0], [10]],       # 1d memory template
                                [[0], [5], [10]],
                                [[0.1], [0.1], [10]],
                                [[0.1], [0.1], [10]]],
                               [[5], [5], [10]],        # 1d query
                               pnl.L0),                 # 1d retrieval operation
                              ([[[5,0], [0,5], [10,10]],   # 2d memory template
                                [[0,5], [5,0], [10,10]],
                                [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]],
                                [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]],
                                [[5,0], [5,0], [10,10]],  # 2d query
                               pnl.DOT_PRODUCT),          # 2d retrieval operation
                               ),
                             ids=['1d', '2d'])
    @pytest.mark.parametrize('field_weights', [[.75, .25, 0], [.75, .25, None]], ids=['0','None'])
    @pytest.mark.parametrize('softmax_choice', [pnl.MAX_VAL, pnl.ARG_MAX], ids=['MAX_VAL','ARG_MAX'])
    @pytest.mark.parametrize('exec_mode', [pnl.ExecutionMode.Python,
                                           pnl.ExecutionMode.PyTorch,
                                           # pnl.ExecutionMode.LLVM
                                           ],
                             ids=['Python',
                                  'PyTorch',
                                  # 'LLVM'
                                  ])
    @pytest.mark.composition
    def test_assign_field_weights_and_0_vs_None(self,
                                                field_weights,
                                                softmax_choice,
                                                test_field_weights_0_vs_None_data,
                                                exec_mode):
        memory_template = test_field_weights_0_vs_None_data[0]
        query = test_field_weights_0_vs_None_data[1]
        operation = test_field_weights_0_vs_None_data[2]

        em = pnl.EMComposition(memory_template=memory_template,
                               memory_capacity=4,
                               memory_decay_rate= 0,
                               learn_field_weights = False,
                               softmax_choice=softmax_choice,
                               field_weights=field_weights,
                               field_names=['A','B','C'])
        # Confirm initial weight assignments (that favor A)
        assert em.nodes['A [WEIGHT]'].input_port.defaults.variable == [.75]
        assert em.nodes['B [WEIGHT]'].input_port.defaults.variable == [.25]
        if field_weights[2] == 0:
            assert 'C [QUERY]' in em.nodes.names
            assert len(em.field_weight_nodes) == 3
            assert em.nodes['C [WEIGHT]'].input_port.defaults.variable == [0]
        elif field_weights[2] is None:
            assert 'C [VALUE]' in em.nodes.names
            assert len(em.field_weight_nodes) == 2
            assert 'C [WEIGHT]' not in em.nodes.names

        # Confirm use of L0 for retrieval since keys for A and B are scalars
        assert em.projections['MEMORY for A [KEY]'].function.operation == operation
        assert em.projections['MEMORY for B [KEY]'].function.operation == operation
        if field_weights[2] == 0:
            assert em.projections['MEMORY for C [KEY]'].function.operation == operation

        A = em.nodes['A [QUERY]']
        B = em.nodes['B [QUERY]']
        C = em.nodes['C [QUERY]' if field_weights[2] == 0 else 'C [VALUE]']

        # Note:  The input matches both fields A and B
        test_input = {A: [query[0]],
                      B: [query[1]],
                      C: [query[2]]}
        result = em.run(test_input, execution_mode=exec_mode)
        # Note: field_weights favors A
        if softmax_choice == pnl.MAX_VAL:
            if operation == pnl.L0:
                expected = [[1.70381182], [0.], [3.40762364]]
            else:
                expected = [[1.56081243, 0.0], [0.0, 1.56081243], [3.12162487, 3.12162487]]
        else:
            expected = memory_template[0]
        np.testing.assert_allclose(result, expected)

        # Change fields weights to favor C
        if field_weights[2] is None:
            with pytest.raises(EMCompositionError) as error_text:
                em.field_weights = np.array([0,0,1])
            assert error_text.value.error_value == (f"Field 'C' of 'EM_Composition' was originally assigned "
                                                    f"as a value node (i.e., with a field_weight = None); "
                                                    f"this cannot be changed after construction. If you want to "
                                                    f"change it to a key field, you must re-construct the "
                                                    f"EMComposition using a scalar for its field in the "
                                                    f"`field_weights` arg (including 0.")
        else:
            em.field_weights = np.array([0,0,1])
            # Ensure weights got changed
            assert em.nodes['A [WEIGHT]'].input_port.defaults.variable == [0]
            assert em.nodes['B [WEIGHT]'].input_port.defaults.variable == [0]
            assert em.nodes['C [WEIGHT]'].input_port.defaults.variable == [1]
            # Note:  The input matches both fields A and B;
            test_input = {em.nodes['A [QUERY]']: [query[0]],
                          em.nodes['B [QUERY]']: [query[1]],
                          em.nodes['C [QUERY]']: [query[2]]}
            result = em.run(test_input, execution_mode=exec_mode)
            #  If the weights change DIDN'T get used, it should favor field A and return [5,0,10] as the best match
            #  If weights change DID get used, it should favor field B and return [0,5,10] as the best match
            if softmax_choice == pnl.MAX_VAL:
                if operation == pnl.L0:
                    expected = [[2.525], [2.525], [10]]
                else:
                    expected = [[2.525, 1.275], [2.525, 1.275], [7.525, 7.525]]
            else:
                expected = memory_template[0]
            np.testing.assert_allclose(result, expected)

            #  Change weights back and confirm that it now favors A
            em.field_weights = [0,1,0]
            result = em.run(test_input, execution_mode=exec_mode)
            if softmax_choice == pnl.MAX_VAL:
                if operation == pnl.L0:
                    expected = [[3.33333333], [5], [10]]
                else:
                    expected = [[3.33333333, 1.66666667], [5, 0], [10, 10]]
            else:
                expected = memory_template[1]
            np.testing.assert_allclose(result, expected)


    @pytest.mark.composition
    @pytest.mark.parametrize('exec_mode', [pnl.ExecutionMode.Python, pnl.ExecutionMode.PyTorch])
    @pytest.mark.parametrize('concatenate', [True, False], ids=['concatenate', 'no_concatenate'])
    @pytest.mark.parametrize('use_storage_node', [True, False], ids=['use_storage_node', 'no_storage_node'])
    @pytest.mark.parametrize('learning', [True, False], ids=['learning', 'no_learning'])
    def test_multiple_trials_concatenation_and_storage_node(self, exec_mode, concatenate, use_storage_node, learning):
        """Test with and without learning (learning is tested only for using_storage_node and no concatenation)"""

        # if comp_mode != pnl.ExecutionMode.Python:
        #     pytest.skip('Execution of EMComposition not yet supported for LLVM Mode.')

        em = EMComposition(memory_template=(2,3),
                           field_weights=[1,1],
                           memory_capacity=4,
                           softmax_gain=100,
                           memory_fill=(0,.001),
                           concatenate_queries=concatenate,
                           # learn_field_weights=learning,
                           learn_field_weights=False,
                           enable_learning=True,
                           use_storage_node=use_storage_node)

        inputs = [[[[1,2,3]],[[4,5,6]],[[10,20,30]],[[40,50,60]],[[100,200,300]],[[400,500,600]]],
                  [[[1,2,5]],[[4,5,8]],[[11,21,31]],[[41,51,61]],[[111,222,333]],[[444,555,666]]],
                  [[[1,2,10]],[[4,5,10]]],[[[51,52,53]],[[81,82,83]],[[777,888,999]],[[1111,2222,3333]]]]

        expected_memory = [[[0.15625, 0.3125,  0.46875], [0.171875, 0.328125, 0.484375]],
                           [[400., 500., 600.], [444., 555., 666.]],
                           [[2.5, 3.125, 3.75 ], [2.5625, 3.1875, 3.8125]],
                           [[25., 50., 75.], [27.75, 55.5,  83.25]]]

        input_nodes = em.query_input_nodes + em.value_input_nodes
        inputs = {input_nodes[i]:inputs[i] for
                  i in range(len(input_nodes))}
        em.run(inputs=inputs, execution_mode=exec_mode)
        np.testing.assert_equal(em.memory, expected_memory)

        if use_storage_node:
            # Only test learning if using storage_node, as this is required for learning
            if concatenate:
                with pytest.raises(EMCompositionError) as error:
                    em.learn(inputs=inputs, execution_mode=exec_mode)
                assert "EMComposition does not support learning with 'concatenate_queries'=True." in str(error.value)

            else:
                # if exec_mode == pnl.ExecutionMode.Python:
                #     # FIX: Not sure why Python mode reverses last two rows/entries (dict issue?)
                expected_memory = [[[0.15625, 0.3125,  0.46875], [0.171875, 0.328125, 0.484375]],
                                   [[400., 500., 600.], [444., 555., 666.]],
                                   [[25., 50., 75.], [27.75, 55.5,  83.25]],
                                   [[2.5, 3.125, 3.75 ], [2.5625, 3.1875, 3.8125]]]
                em.learn(inputs=inputs, execution_mode=exec_mode)
                np.testing.assert_equal(em.memory, expected_memory)

    @pytest.mark.composition
    def test_backpropagation_of_error_in_learning(self):
        """This test is based on the EGO CSW Model"""

        import torch
        torch.manual_seed(0)
        state_input_layer = pnl.ProcessingMechanism(name='STATE', input_shapes=11)
        previous_state_layer = pnl.ProcessingMechanism(name='PREVIOUS STATE', input_shapes=11)
        context_layer = pnl.TransferMechanism(name='CONTEXT',
                                          input_shapes=11,
                                          function=pnl.Tanh,
                                          integrator_mode=True,
                                          integration_rate=.69)
        em = EMComposition(name='EM',
                           memory_template=[[0] * 11, [0] * 11, [0] * 11],  # context
                           memory_fill=(0,.0001),
                           memory_capacity=50,
                           memory_decay_rate=0,
                           softmax_gain=10,
                           softmax_threshold=.001,
                           fields = {'STATE': {pnl.FIELD_WEIGHT: None,
                                               pnl.LEARN_FIELD_WEIGHT: False,
                                               pnl.TARGET_FIELD: True},
                                     'PREVIOUS_STATE': {pnl.FIELD_WEIGHT:.5,
                                                        pnl.LEARN_FIELD_WEIGHT: False,
                                                        pnl.TARGET_FIELD: False},
                                     'CONTEXT': {pnl.FIELD_WEIGHT:.5,
                                                 pnl.LEARN_FIELD_WEIGHT: False,
                                                 pnl.TARGET_FIELD: False}},
                           normalize_field_weights=True,
                           normalize_memories=False,
                           concatenate_queries=False,
                           enable_learning=True,
                           learning_rate=.5,
                           device=pnl.CPU
                           )
        prediction_layer = pnl.ProcessingMechanism(name='PREDICTION', input_shapes=11)

        QUERY = ' [QUERY]'
        VALUE = ' [VALUE]'
        RETRIEVED = ' [RETRIEVED]'

        # Pathways
        state_to_previous_state_pathway = [state_input_layer,
                                           pnl.MappingProjection(matrix=pnl.IDENTITY_MATRIX,
                                                             learnable=False),
                                           previous_state_layer]
        state_to_context_pathway = [state_input_layer,
                                    pnl.MappingProjection(matrix=pnl.IDENTITY_MATRIX,
                                                      learnable=False),
                                    context_layer]
        state_to_em_pathway = [state_input_layer,
                               pnl.MappingProjection(sender=state_input_layer,
                                                 receiver=em.nodes['STATE' + VALUE],
                                                 matrix=pnl.IDENTITY_MATRIX,
                                                 learnable=False),
                               em]
        previous_state_to_em_pathway = [previous_state_layer,
                                        pnl.MappingProjection(sender=previous_state_layer,
                                                          receiver=em.nodes['PREVIOUS_STATE' + QUERY],
                                                          matrix=pnl.IDENTITY_MATRIX,
                                                          learnable=False),
                                        em]
        context_learning_pathway = [context_layer,
                                    pnl.MappingProjection(sender=context_layer,
                                                      matrix=pnl.IDENTITY_MATRIX,
                                                      receiver=em.nodes['CONTEXT' + QUERY],
                                                      learnable=True),
                                    em,
                                    pnl.MappingProjection(sender=em.nodes['STATE' + RETRIEVED],
                                                      receiver=prediction_layer,
                                                      matrix=pnl.IDENTITY_MATRIX,
                                                      learnable=False),
                                    prediction_layer]

        # Composition
        EGO = pnl.AutodiffComposition([state_to_previous_state_pathway,
                                        state_to_context_pathway,
                                        state_to_em_pathway,
                                        previous_state_to_em_pathway,
                                        context_learning_pathway],
                                       learning_rate=.5,
                                       loss_spec=pnl.Loss.BINARY_CROSS_ENTROPY,
                                       device=pnl.CPU)

        learning_components = EGO.infer_backpropagation_learning_pathways(pnl.ExecutionMode.PyTorch)
        assert len(learning_components) == 1
        assert learning_components[0].name == 'TARGET for PREDICTION'
        EGO.add_projection(pnl.MappingProjection(sender=state_input_layer,
                                                  receiver=learning_components[0],
                                                  learnable=False))

        EGO.scheduler.add_condition(em, pnl.BeforeNodes(previous_state_layer, context_layer))

        INPUTS = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]

        result = EGO.learn(inputs={'STATE':INPUTS}, learning_rate=.5, execution_mode=pnl.ExecutionMode.PyTorch)
        expected = [[ 0.00000000e+00,  1.35476414e-03,  1.13669378e-03,  2.20434260e-03,  6.61008388e-04, 9.88672202e-01,
                      6.52088276e-04,  1.74149507e-03,  1.09769133e-03,  2.47971436e-03,  0.00000000e+00],
                    [ 0.00000000e+00, -6.75284069e-02, -1.28930436e-03, -2.10726610e-01, -1.41050716e-03, -5.92286989e-01,
                     -2.75196416e-03, -2.21010605e-03, -7.14369243e-03, -2.05167374e-02,  0.00000000e+00],
                    [ 0.00000000e+00,  1.18578255e-03,  1.29393181e-03,  1.35476414e-03,  1.13669378e-03, 2.20434260e-03,
                      6.61008388e-04,  9.88672202e-01,  6.52088276e-04,  2.83918640e-03,  0.00000000e+00]]
        np.testing.assert_allclose(result, expected)

        # Plot (for during debugging):
        # TARGETS = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        #            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
        #
        # fig, axes = plt.subplots(3, 1, figsize=(5, 12))
        # axes[0].imshow(EGO.projections[7].parameters.matrix.get(EGO.name), interpolation=None)
        # axes[1].plot((1 - np.abs(EGO.results[1:50,2]-TARGETS[:49])).sum(-1))
        # axes[1].set_xlabel('Stimuli')
        # axes[1].set_ylabel('loss_spec')
        # axes[2].plot( (EGO.results[1:50,2]*TARGETS[:49]).sum(-1) )
        # axes[2].set_xlabel('Stimuli')
        # axes[2].set_ylabel('Correct Logit')
        # plt.suptitle(f"Blocked Training")
        # plt.show()
