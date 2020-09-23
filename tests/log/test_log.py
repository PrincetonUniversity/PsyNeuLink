import numpy as np
import psyneulink as pnl
import pytest

from collections import OrderedDict

import psyneulink.core.components.functions.transferfunctions
from psyneulink.core.globals.keywords import ALLOCATION_SAMPLES, PROJECTIONS

class TestLog:

    def test_log(self):

        T_1 = pnl.TransferMechanism(name='log_test_T_1', size=2)
        T_2 = pnl.TransferMechanism(name='log_test_T_2', size=2)
        PS = pnl.Composition(name='log_test_PS', pathways=[T_1, T_2])
        PJ = T_2.path_afferents[0]

        assert T_1.loggable_items == {
            'clip': 'OFF',
            'termination_threshold': 'OFF',
            'execute_until_finished': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_execute_until_finished': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_max_executions_before_finished': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_num_executions_before_finished': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'InputPort-0': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_executions_before_finished': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'OFF',
            'mod_offset': 'OFF',
            'mod_rate': 'OFF',
            'noise': 'OFF',
            'num_executions_before_finished': 'OFF',
            'termination_measure_value': 'OFF',
            'RESULT': 'OFF',
            'value': 'OFF',
            'variable': 'OFF'
        }
        assert T_2.loggable_items == {
            'InputPort-0': 'OFF',
            'RESULT': 'OFF',
            'clip': 'OFF',
            'termination_threshold': 'OFF',
            'execute_until_finished': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_execute_until_finished': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_max_executions_before_finished': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_num_executions_before_finished': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_executions_before_finished': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'OFF',
            'mod_offset': 'OFF',
            'mod_rate': 'OFF',
            'noise': 'OFF',
            'num_executions_before_finished': 'OFF',
            'termination_measure_value': 'OFF',
            'value': 'OFF',
            'variable': 'OFF'
        }
        assert PJ.loggable_items == {
            'execute_until_finished': 'OFF',
            'exponent': 'OFF',
            'func_bounds': 'OFF',
            'func_execute_until_finished': 'OFF',
            'func_has_initializers': 'OFF',
            'func_matrix': 'OFF',
            'func_max_executions_before_finished': 'OFF',
            'func_num_executions_before_finished': 'OFF',
            'func_value': 'OFF',
            'has_initializers': 'OFF',
            'func_variable': 'OFF',
            'matrix': 'OFF',
            'max_executions_before_finished': 'OFF',
            'mod_matrix': 'OFF',
            'num_executions_before_finished': 'OFF',
            'value': 'OFF',
            'variable': 'OFF',
            'weight': 'OFF'
        }

        T_1.set_log_conditions('mod_noise')
        T_1.set_log_conditions(pnl.RESULT)
        PJ.set_log_conditions('mod_matrix')

        assert T_1.loggable_items == {
            'InputPort-0': 'OFF',
            'RESULT': 'EXECUTION',
            'clip': 'OFF',
            'termination_threshold': 'OFF',
            'execute_until_finished': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_execute_until_finished': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_max_executions_before_finished': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_num_executions_before_finished': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_executions_before_finished': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'EXECUTION',
            'mod_slope': 'OFF',
            'mod_offset': 'OFF',
            'mod_rate': 'OFF',
            'noise': 'OFF',
            'num_executions_before_finished': 'OFF',
            'termination_measure_value': 'OFF',
            'value': 'OFF',
            'variable': 'OFF'
        }
        assert T_2.loggable_items == {
            'InputPort-0': 'OFF',
            'RESULT': 'OFF',
            'clip': 'OFF',
            'termination_threshold': 'OFF',
            'execute_until_finished': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_execute_until_finished': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_max_executions_before_finished': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_num_executions_before_finished': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_executions_before_finished': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'OFF',
            'mod_offset': 'OFF',
            'mod_rate': 'OFF',
            'noise': 'OFF',
            'num_executions_before_finished': 'OFF',
            'termination_measure_value': 'OFF',
            'value': 'OFF',
            'variable': 'OFF'
        }
        assert PJ.loggable_items == {
            'execute_until_finished': 'OFF',
            'exponent': 'OFF',
            'func_bounds': 'OFF',
            'func_execute_until_finished': 'OFF',
            'func_has_initializers': 'OFF',
            'func_matrix': 'OFF',
            'func_max_executions_before_finished': 'OFF',
            'func_num_executions_before_finished': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'has_initializers': 'OFF',
            'matrix': 'OFF',
            'max_executions_before_finished': 'OFF',
            'mod_matrix': 'EXECUTION',
            'num_executions_before_finished': 'OFF',
            'value': 'OFF',
            'variable': 'OFF',
            'weight': 'OFF'
        }

        PS.run(inputs={T_1:[0,0]})
        PS.run(inputs={T_1:[1,2]})
        PS.run(inputs={T_1:[3,4]})

        assert T_1.logged_items == {'RESULT': 'EXECUTION', 'mod_noise': 'EXECUTION'}
        assert PJ.logged_items == {'mod_matrix': 'EXECUTION'}

        T_1.log.print_entries(contexts=PS)
        # assert T_1.log.print_entries() ==
        # test_log.py::TestLog::test_log
        # Log for log_test_T_1:
        # Logged Item:   Time       Context                    Value
        # 'RESULT'       0:0:0:0   'PROCESSING, COMPOSI...   [0. 0.]
        # 'RESULT'       1:0:0:0   'PROCESSING, COMPOSI...   [1. 2.]
        # 'RESULT'       2:0:0:0   'PROCESSING, COMPOSI...   [3. 4.]
        # 'mod_noise'    0:0:0:0   'PROCESSING, COMPOSI...   [0.]
        # 'mod_noise'    1:0:0:0   'PROCESSING, COMPOSI...   [0.]
        # 'mod_noise'    2:0:0:0   'PROCESSING, COMPOSI...   [0.]

        T_1_csv = T_1.log.csv(entries=['mod_noise', 'RESULT'], owner_name=False, quotes=None)
        print(T_1_csv)
        assert T_1_csv == \
            "'Execution Context', 'Data'\n" \
            + f"'{PS.default_execution_id}', \'Run\', \'Trial\', \'Pass\', \'Time_step\', \'mod_noise\', \'RESULT\'\n" \
            + ", 0, 0, 0, 0, 0.0, 0.0 0.0\n" \
            + ", 1, 0, 0, 0, 0.0, 1.0 2.0\n" \
            + ", 2, 0, 0, 0, 0.0, 3.0 4.0\n"

        assert PJ.log.csv(entries='mod_matrix', owner_name=True, quotes=True) == \
            "'Execution Context', 'Data'\n" \
            + f"'{PS.default_execution_id}', \'Run\', \'Trial\', \'Pass\', \'Time_step\', " \
              "\'MappingProjection from log_test_T_1[RESULT] to log_test_T_2[InputPort-0][mod_matrix]\'\n" \
            + ", \'0\', \'0\', \'0\', \'1\', \'1.0 0.0\' \'0.0 1.0\'\n" \
            + ", \'1\', \'0\', \'0\', \'1\', \'1.0 0.0\' \'0.0 1.0\'\n" \
            + ", \'2\', \'0\', \'0\', \'1\', \'1.0 0.0\' \'0.0 1.0\'\n"

        result = T_1.log.nparray(entries=['mod_noise', 'RESULT'], header=False, owner_name=True)
        assert result[0] == PS.default_execution_id

        expected = [
            [[0], [1], [2]],
            [[0.], [0.], [0.]],
            [[0.], [0.], [0.]],
            [[0.], [0.], [0.]],
            [[0.0], [0.0], [0.0]],
            [[0., 0.], [1., 2.], [3., 4.]],
        ]
        assert result[1][0] == expected

    def test_log_initialization(self):
        T = pnl.TransferMechanism(
                prefs={pnl.LOG_PREF: pnl.PreferenceEntry(pnl.LogCondition.INITIALIZATION, pnl.PreferenceLevel.INSTANCE)}
        )
        assert T.logged_items == {'value': 'INITIALIZATION'}

    def test_log_dictionary_without_time(self):

        T1 = pnl.TransferMechanism(name='log_test_T1',
                                    size=2)
        T2 = pnl.TransferMechanism(name='log_test_T2',
                                    size=2)
        PS = pnl.Composition(name='log_test_PS', pathways=[T1, T2])
        PJ = T2.path_afferents[0]

        assert T1.loggable_items == {
            'InputPort-0': 'OFF',
            'RESULT': 'OFF',
            'clip': 'OFF',
            'termination_threshold': 'OFF',
            'execute_until_finished': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_execute_until_finished': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_max_executions_before_finished': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_num_executions_before_finished': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_executions_before_finished': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'OFF',
            'mod_offset': 'OFF',
            'mod_rate': 'OFF',
            'noise': 'OFF',
            'num_executions_before_finished': 'OFF',
            'termination_measure_value': 'OFF',
            'value': 'OFF',
            'variable': 'OFF'
        }
        assert T2.loggable_items == {
            'InputPort-0': 'OFF',
            'RESULT': 'OFF',
            'clip': 'OFF',
            'termination_threshold': 'OFF',
            'execute_until_finished': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_execute_until_finished': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_max_executions_before_finished': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_num_executions_before_finished': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_executions_before_finished': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'OFF',
            'mod_offset': 'OFF',
            'mod_rate': 'OFF',
            'num_executions_before_finished': 'OFF',
            'noise': 'OFF',
            'termination_measure_value': 'OFF',
            'value': 'OFF',
            'variable': 'OFF'
        }
        assert PJ.loggable_items == {
            'execute_until_finished': 'OFF',
            'exponent': 'OFF',
            'func_bounds': 'OFF',
            'func_execute_until_finished': 'OFF',
            'func_has_initializers': 'OFF',
            'func_matrix': 'OFF',
            'func_max_executions_before_finished': 'OFF',
            'func_num_executions_before_finished': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'has_initializers': 'OFF',
            'matrix': 'OFF',
            'max_executions_before_finished': 'OFF',
            'mod_matrix': 'OFF',
            'num_executions_before_finished': 'OFF',
            'value': 'OFF',
            'variable': 'OFF',
            'weight': 'OFF'
        }

        T1.set_log_conditions('mod_slope')
        T1.set_log_conditions(pnl.RESULT)
        T1.set_log_conditions(pnl.VALUE)
        PJ.set_log_conditions('mod_matrix')
        T2.set_log_conditions('mod_slope')
        T2.set_log_conditions(pnl.RESULT)
        T2.set_log_conditions(pnl.VALUE)

        assert T1.loggable_items == {
            'InputPort-0': 'OFF',
            'RESULT': 'EXECUTION',
            'clip': 'OFF',
            'termination_threshold': 'OFF',
            'execute_until_finished': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_execute_until_finished': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_max_executions_before_finished': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_num_executions_before_finished': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_executions_before_finished': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'EXECUTION',
            'mod_offset': 'OFF',
            'mod_rate': 'OFF',
            'noise': 'OFF',
            'num_executions_before_finished': 'OFF',
            'termination_measure_value': 'OFF',
            'value': 'EXECUTION',
            'variable': 'OFF'
        }
        assert T2.loggable_items == {
            'InputPort-0': 'OFF',
            'RESULT': 'EXECUTION',
            'clip': 'OFF',
            'termination_threshold': 'OFF',
            'execute_until_finished': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_execute_until_finished': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_max_executions_before_finished': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_num_executions_before_finished': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_executions_before_finished': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'EXECUTION',
            'mod_offset': 'OFF',
            'mod_rate': 'OFF',
            'noise': 'OFF',
            'num_executions_before_finished': 'OFF',
            'termination_measure_value': 'OFF',
            'value': 'EXECUTION',
            'variable': 'OFF'
        }
        assert PJ.loggable_items == {
            'execute_until_finished': 'OFF',
            'exponent': 'OFF',
            'func_bounds': 'OFF',
            'func_execute_until_finished': 'OFF',
            'func_has_initializers': 'OFF',
            'func_matrix': 'OFF',
            'func_max_executions_before_finished': 'OFF',
            'func_num_executions_before_finished': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'has_initializers': 'OFF',
            'matrix': 'OFF',
            'max_executions_before_finished': 'OFF',
            'mod_matrix': 'EXECUTION',
            'num_executions_before_finished': 'OFF',
            'value': 'OFF',
            'variable': 'OFF',
            'weight': 'OFF'
        }

        PS.run(inputs={T1:[1.0, 2.0]})
        PS.run(inputs={T1:[3.0, 4.0]})
        PS.run(inputs={T1:[5.0, 6.0]})

        assert T1.logged_items == {'RESULT': 'EXECUTION',
                                   'mod_slope': 'EXECUTION',
                                   'value': 'EXECUTION'}
        assert T2.logged_items == {'RESULT': 'EXECUTION',
                                   'mod_slope': 'EXECUTION',
                                   'value': 'EXECUTION'}
        assert PJ.logged_items == {'mod_matrix': 'EXECUTION'}

        log_dict_T1 = T1.log.nparray_dictionary(entries=['value', 'mod_slope', 'RESULT'])

        expected_values_T1 = [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]
        expected_slopes_T1 = [[1.0], [1.0], [1.0]]
        expected_results_T1 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

        assert list(log_dict_T1.keys()) == [PS.default_execution_id]

        assert np.allclose(expected_values_T1, log_dict_T1[PS.default_execution_id]['value'])
        assert np.allclose(expected_slopes_T1, log_dict_T1[PS.default_execution_id]['mod_slope'])
        assert np.allclose(expected_results_T1, log_dict_T1[PS.default_execution_id]['RESULT'])

        assert list(log_dict_T1[PS.default_execution_id].keys()) == \
               ['Run', 'Trial', 'Pass', 'Time_step', 'value', 'mod_slope', 'RESULT']

        log_dict_T1_reorder = T1.log.nparray_dictionary(entries=['mod_slope', 'value', 'RESULT'])

        assert list(log_dict_T1_reorder[PS.default_execution_id].keys()) == \
               ['Run', 'Trial', 'Pass', 'Time_step', 'mod_slope', 'value', 'RESULT']

    def test_run_resets(self):
        import psyneulink as pnl
        T1 = pnl.TransferMechanism(name='log_test_T1',
                                   size=2)
        T2 = pnl.TransferMechanism(name='log_test_T2',
                                   size=2)
        COMP = pnl.Composition(name='COMP', pathways=[T1, T2])
        T1.set_log_conditions('mod_slope')
        T2.set_log_conditions('value')
        COMP.run(inputs={T1: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]})

        log_array_T1 = T1.log.nparray()
        log_array_T2 = T2.log.nparray()
        assert log_array_T1.shape == log_array_T1.shape == (2,2)
        assert log_array_T1[0][0] == log_array_T2[0][0] == 'Execution Context'
        assert log_array_T1[0][1] == log_array_T1[0][1] == 'COMP'
        assert log_array_T2[1][1][4][1:4] == [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]

        COMP.run(inputs={T1: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]})

        log_array_T1_second_run = T1.log.nparray()
        log_array_T2_second_run = T2.log.nparray()
        assert log_array_T2_second_run[1][1][4][1:4] == [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]

    def test_log_dictionary_with_time(self):

        T1 = pnl.TransferMechanism(name='log_test_T1',
                                   size=2)
        T2 = pnl.TransferMechanism(name='log_test_T2',
                                   function=psyneulink.core.components.functions.transferfunctions.Linear(slope=2.0),
                                   size=2)
        COMP = pnl.Composition(name='log_test_COMP', pathways=[T1, T2])

        assert T1.loggable_items == {
            'InputPort-0': 'OFF',
            'RESULT': 'OFF',
            'clip': 'OFF',
            'termination_threshold': 'OFF',
            'execute_until_finished': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_execute_until_finished': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_max_executions_before_finished': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_num_executions_before_finished': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_executions_before_finished': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'OFF',
            'mod_offset': 'OFF',
            'mod_rate': 'OFF',
            'noise': 'OFF',
            'num_executions_before_finished': 'OFF',
            'termination_measure_value': 'OFF',
            'value': 'OFF',
            'variable': 'OFF'
        }
        assert T2.loggable_items == {
            'InputPort-0': 'OFF',
            'RESULT': 'OFF',
            'clip': 'OFF',
            'termination_threshold': 'OFF',
            'execute_until_finished': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_execute_until_finished': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_max_executions_before_finished': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_num_executions_before_finished': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_executions_before_finished': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'OFF',
            'mod_offset': 'OFF',
            'mod_rate': 'OFF',
            'noise': 'OFF',
            'num_executions_before_finished': 'OFF',
            'termination_measure_value': 'OFF',
            'value': 'OFF',
            'variable': 'OFF'
        }

        T1.set_log_conditions('mod_slope')
        T1.set_log_conditions(pnl.RESULT)
        T1.set_log_conditions(pnl.VALUE)

        assert T1.loggable_items == {
            'execute_until_finished': 'OFF',
            'InputPort-0': 'OFF',
            'RESULT': 'EXECUTION',
            'clip': 'OFF',
            'termination_threshold': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_execute_until_finished': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_max_executions_before_finished': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_num_executions_before_finished': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_executions_before_finished': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'EXECUTION',
            'mod_offset': 'OFF',
            'mod_rate': 'OFF',
            'noise': 'OFF',
            'num_executions_before_finished': 'OFF',
            'termination_measure_value': 'OFF',
            'value': 'EXECUTION',
            'variable': 'OFF'
        }

        T2.set_log_conditions('mod_slope')
        T2.set_log_conditions(pnl.RESULT)
        T2.set_log_conditions(pnl.VALUE)

        assert T2.loggable_items == {
            'InputPort-0': 'OFF',
            'RESULT': 'EXECUTION',
            'clip': 'OFF',
            'termination_threshold': 'OFF',
            'execute_until_finished': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_execute_until_finished': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_max_executions_before_finished': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_num_executions_before_finished': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_executions_before_finished': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'EXECUTION',
            'mod_offset': 'OFF',
            'mod_rate': 'OFF',
            'noise': 'OFF',
            'num_executions_before_finished': 'OFF',
            'termination_measure_value': 'OFF',
            'value': 'EXECUTION',
            'variable': 'OFF'
        }

        # RUN ZERO  |  TRIALS ZERO, ONE, TWO ----------------------------------

        COMP.run(inputs={T1: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]})

        assert T1.logged_items == {'RESULT': 'EXECUTION',
                                   'mod_slope': 'EXECUTION',
                                   'value': 'EXECUTION'}
        assert T2.logged_items == {'RESULT': 'EXECUTION',
                                   'mod_slope': 'EXECUTION',
                                   'value': 'EXECUTION'}

        # T1 log after zero-th run -------------------------------------------

        log_dict_T1 = T1.log.nparray_dictionary(entries=['value', 'mod_slope', 'RESULT'])

        expected_run_T1 = [[0], [0], [0]]
        expected_trial_T1 = [[0], [1], [2]]
        expected_time_step_T1 = [[0], [0], [0]]
        expected_values_T1 = [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]
        expected_slopes_T1 = [[1.0], [1.0], [1.0]]
        expected_results_T1 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

        assert list(log_dict_T1.keys()) == [COMP.default_execution_id]
        sys_log_dict = log_dict_T1[COMP.default_execution_id]

        assert np.allclose(expected_run_T1, sys_log_dict['Run'])
        assert np.allclose(expected_trial_T1, sys_log_dict['Trial'])
        assert np.allclose(expected_time_step_T1, sys_log_dict['Time_step'])
        assert np.allclose(expected_values_T1, sys_log_dict['value'])
        assert np.allclose(expected_slopes_T1, sys_log_dict['mod_slope'])
        assert np.allclose(expected_results_T1, sys_log_dict['RESULT'])

        # T2 log after zero-th run --------------------------------------------

        log_dict_T2 = T2.log.nparray_dictionary(entries=['value', 'mod_slope', 'RESULT'])

        expected_run_T2 = [[0], [0], [0]]
        expected_trial_T2 = [[0], [1], [2]]
        expected_time_step_T2 = [[1], [1], [1]]
        expected_values_T2 = [[[2.0, 4.0]], [[6.0, 8.0]], [[10.0, 12.0]]]
        expected_slopes_T2 = [[2.0], [2.0], [2.0]]
        expected_results_T2 = [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]

        assert list(log_dict_T2.keys()) == [COMP.default_execution_id]
        sys_log_dict = log_dict_T2[COMP.default_execution_id]

        assert np.allclose(expected_run_T2, sys_log_dict['Run'])
        assert np.allclose(expected_trial_T2, sys_log_dict['Trial'])
        assert np.allclose(expected_time_step_T2, sys_log_dict['Time_step'])
        assert np.allclose(expected_values_T2, sys_log_dict['value'])
        assert np.allclose(expected_slopes_T2, sys_log_dict['mod_slope'])
        assert np.allclose(expected_results_T2, sys_log_dict['RESULT'])

        # RUN ONE  |  TRIALS ZERO, ONE, TWO -------------------------------------

        COMP.run(inputs={T1: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]})

        # T1 log after first run -------------------------------------------

        log_dict_T1 = T1.log.nparray_dictionary(entries=['value', 'mod_slope', 'RESULT'])

        assert list(log_dict_T1.keys()) == [COMP.default_execution_id]
        sys_log_dict = log_dict_T1[COMP.default_execution_id]

        # expected_run_T1_2 = [[1], [1], [1]]
        expected_run_T1_2 = [[0], [0], [0]] + expected_run_T1
        expected_trial_T1_2 = [[0], [1], [2]] + expected_trial_T1
        expected_time_step_T1_2 = [[0], [0], [0]] + expected_time_step_T1
        expected_values_T1_2 = expected_values_T1 + expected_values_T1
        expected_slopes_T1_2 = expected_slopes_T1 + expected_slopes_T1
        expected_results_T1_2 = expected_results_T1 + expected_results_T1

        # assert np.allclose(expected_run_T1_2, sys_log_dict['Run'])
        # assert np.allclose(expected_trial_T1_2, sys_log_dict['Trial'])
        # assert np.allclose(expected_time_step_T1_2, sys_log_dict['Time_step'])
        assert np.allclose(expected_values_T1_2, sys_log_dict['value'])
        assert np.allclose(expected_slopes_T1_2, sys_log_dict['mod_slope'])
        assert np.allclose(expected_results_T1_2, sys_log_dict['RESULT'])

        # T2 log after first run -------------------------------------------

        log_dict_T2_2 = T2.log.nparray_dictionary(entries=['value', 'mod_slope', 'RESULT'])

        assert list(log_dict_T2_2.keys()) == [COMP.default_execution_id]
        sys_log_dict = log_dict_T2_2[COMP.default_execution_id]

        expected_run_T2_2 = [[0], [0], [0]] + expected_run_T2
        expected_trial_T2_2 = [[0], [1], [2]] + expected_trial_T2
        expected_time_step_T2_2 = [[1], [1], [1]] + expected_time_step_T2
        expected_values_T2_2 = [[[2.0, 4.0]], [[6.0, 8.0]], [[10.0, 12.0]]] + expected_values_T2
        expected_slopes_T2_2 = [[2.0], [2.0], [2.0]] + expected_slopes_T2
        expected_results_T2_2 = [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]] + expected_results_T2

        # assert np.allclose(expected_run_T2_2, sys_log_dict['Run'])
        # assert np.allclose(expected_trial_T2_2, sys_log_dict['Trial'])
        # assert np.allclose(expected_time_step_T2_2, sys_log_dict['Time_step'])
        assert np.allclose(expected_values_T2_2, sys_log_dict['value'])
        assert np.allclose(expected_slopes_T2_2, sys_log_dict['mod_slope'])
        assert np.allclose(expected_results_T2_2, sys_log_dict['RESULT'])

    def test_log_dictionary_with_scheduler(self):
        T1 = pnl.TransferMechanism(name='log_test_T1',
                                   integrator_mode=True,
                                   integration_rate=0.5)
        T2 = pnl.TransferMechanism(name='log_test_T2',
                                   function=psyneulink.core.components.functions.transferfunctions.Linear(slope=6.0))
        COMP = pnl.Composition(name='log_test_COMP', pathways=[T1, T2])

        def pass_threshold(mech, thresh):
            results = mech.output_ports[0].parameters.value.get(COMP)
            for val in results:
                if abs(val) >= thresh:
                    return True
            return False

        terminate_trial = {
            pnl.TimeScale.TRIAL: pnl.While(pass_threshold, T2, 5.0)
        }

        T1.set_log_conditions(pnl.VALUE)
        T1.set_log_conditions('mod_slope')
        T1.set_log_conditions(pnl.RESULT)
        T2.set_log_conditions(pnl.VALUE)
        T2.set_log_conditions('mod_slope')

        COMP.run(inputs={T1: [[1.0]]}, termination_processing=terminate_trial)

        log_dict_T1 = T1.log.nparray_dictionary(entries=['RESULT', 'mod_slope', 'value'])
        log_dict_T2 = T2.log.nparray_dictionary(entries=['value', 'mod_slope'])

        assert list(log_dict_T1.keys()) == [COMP.default_execution_id]
        sys_log_dict = log_dict_T1[COMP.default_execution_id]

        # Check order of keys (must match order of specification)
        assert list(sys_log_dict.keys()) == ['Run', 'Trial', 'Pass', 'Time_step', 'RESULT', 'mod_slope', 'value']
        assert list(log_dict_T2[COMP.default_execution_id].keys()) == ['Run', 'Trial', 'Pass', 'Time_step', 'value',
                                                                    'mod_slope']

        # Check values T1
        assert np.allclose(sys_log_dict["Run"], [[0], [0], [0]])
        assert np.allclose(sys_log_dict["Trial"], [[0], [0], [0]])
        assert np.allclose(sys_log_dict["Time_step"], [[0], [0], [0]])
        assert np.allclose(sys_log_dict["RESULT"], [[0.5], [0.75], [0.875]])
        assert np.allclose(sys_log_dict["value"], [[[0.5]], [[0.75]], [[0.875]]])
        assert np.allclose(sys_log_dict["mod_slope"], [[1], [1], [1]])

        # Check values T2
        assert np.allclose(log_dict_T2[COMP.default_execution_id]["Run"], [[0], [0], [0]])
        assert np.allclose(log_dict_T2[COMP.default_execution_id]["Trial"], [[0], [0], [0]])
        assert np.allclose(log_dict_T2[COMP.default_execution_id]["Time_step"], [[1], [1], [1]])
        assert np.allclose(log_dict_T2[COMP.default_execution_id]["value"], [[[3]], [[4.5]], [[5.25]]])
        assert np.allclose(log_dict_T2[COMP.default_execution_id]["mod_slope"], [[6], [6], [6]])

    def test_log_array_with_scheduler(self):
        T1 = pnl.TransferMechanism(name='log_test_T1',
                                   integrator_mode=True,
                                   integration_rate=0.5)
        T2 = pnl.TransferMechanism(name='log_test_T2',
                                   function=psyneulink.core.components.functions.transferfunctions.Linear(slope=6.0))
        COMP = pnl.Composition(name='log_test_COMP', pathways=[T1, T2])

        def pass_threshold(mech, thresh):
            results = mech.output_ports[0].parameters.value.get(COMP)
            for val in results:
                if abs(val) >= thresh:
                    return True
            return False

        terminate_trial = {
            pnl.TimeScale.TRIAL: pnl.While(pass_threshold, T2, 5.0)
        }

        T1.set_log_conditions(pnl.VALUE)
        T1.set_log_conditions('mod_slope')
        T1.set_log_conditions(pnl.RESULT)
        T2.set_log_conditions(pnl.VALUE)
        T2.set_log_conditions('mod_slope')

        COMP.run(inputs={T1: [[1.0]]}, termination_processing=terminate_trial)

        log_array_T1 = T1.log.nparray(entries=['RESULT', 'mod_slope', 'value'])
        log_array_T2 = T2.log.nparray(entries=['value', 'mod_slope'])

        context_results = [pnl.Log.context_header, COMP.default_execution_id]
        # Check values
        run_results = [["Run"], [0], [0], [0]]
        trial_results = [["Trial"], [0], [0], [0]]
        pass_results = [["Pass"], [0], [1], [2]]
        time_step_results = [["Time_step"], [0], [0], [0]]
        results_results = ["RESULT", [0.5], [0.75], [0.875]]
        slope_results = ["mod_slope", [1], [1], [1]]
        value_results = ["value", [[0.5]], [[0.75]], [[0.875]]]

        for i in range(2):
            assert log_array_T1[0][i] == context_results[i]

        assert log_array_T1[1][0] == pnl.Log.data_header
        data_array = log_array_T1[1][1]
        j = 0
        for i in range(4):
            assert data_array[j][i] == run_results[i]
            assert data_array[j + 1][i] == trial_results[i]
            assert data_array[j + 2][i] == pass_results[i]
            assert data_array[j + 3][i] == time_step_results[i]
            assert data_array[j + 4][i] == results_results[i]
            assert data_array[j + 5][i] == slope_results[i]
            assert data_array[j + 6][i] == value_results[i]

        # Check values
        run_results = [["Run"], [0], [0], [0]]
        trial_results = [["Trial"], [0], [0], [0]]
        pass_results = [["Pass"], [0], [1], [2]]
        time_step_results = [["Time_step"], [1], [1], [1]]
        value_results = ["value", [[3]], [[4.5]], [[5.25]]]
        slope_results = ["mod_slope", [6], [6], [6]]

        for i in range(2):
            assert log_array_T1[0][i] == context_results[i]

        assert log_array_T2[1][0] == pnl.Log.data_header
        data_array = log_array_T2[1][1]
        j = 0
        for i in range(4):
            assert data_array[j][i] == run_results[i]
            assert data_array[j + 1][i] == trial_results[i]
            assert data_array[j + 2][i] == pass_results[i]
            assert data_array[j + 3][i] == time_step_results[i]
            assert data_array[j + 4][i] == value_results[i]
            assert data_array[j + 5][i] == slope_results[i]

    def test_log_dictionary_with_scheduler_many_time_step_increments(self):
        T1 = pnl.TransferMechanism(name='log_test_T1',
                                   integrator_mode=True,
                                   integration_rate=0.05)
        COMP = pnl.Composition(name='log_test_COMP', pathways=[T1])

        def pass_threshold(mech, thresh):
            results = mech.output_ports[0].parameters.value.get(COMP)
            for val in results:
                if abs(val) >= thresh:
                    return True
            return False

        terminate_trial = {
            pnl.TimeScale.TRIAL: pnl.While(pass_threshold, T1, 0.95)
        }

        T1.set_log_conditions(pnl.VALUE)

        COMP.run(inputs={T1: [[1.0]]}, termination_processing=terminate_trial)

        log_dict_T1 = T1.log.nparray_dictionary(entries=['value'])

        assert list(log_dict_T1.keys()) == [COMP.default_execution_id]
        sys_log_dict = log_dict_T1[COMP.default_execution_id]

        # Check order of keys (must match order of specification)
        assert list(sys_log_dict.keys()) == ['Run', 'Trial', 'Pass', 'Time_step', 'value']

        # # Check values T1
        assert len(sys_log_dict["Run"]) == 59
        assert np.allclose(sys_log_dict["Pass"][30], 30)
        assert np.allclose(sys_log_dict["Time_step"][30], 0)
        assert abs(sys_log_dict["value"][58]) >= 0.95
        assert abs(sys_log_dict["value"][57]) < 0.95

    def test_log_csv_multiple_contexts(self):
        A = pnl.TransferMechanism(name='A')
        B = pnl.TransferMechanism(name='B')
        C = pnl.TransferMechanism(name='C')

        C.set_log_conditions(pnl.VALUE)

        X = pnl.Composition(name='comp X')
        Y = pnl.Composition(name='comp Y')

        X.add_linear_processing_pathway([A, C])
        Y.add_linear_processing_pathway([B, C])

        # running with manual contexts for consistent output
        # because output is sorted by context
        X.run(inputs={A: 1}, context='comp X')
        Y.run(inputs={B: 2}, context='comp Y')

        expected_str = "'Execution Context', 'Data'\n" \
            + "'comp X', 'Run', 'Trial', 'Pass', 'Time_step', 'value'\n" \
            + ", '0', '0', '0', '1', '1.0'\n" \
            + "'comp Y', 'Run', 'Trial', 'Pass', 'Time_step', 'value'\n" \
            + ", '0', '0', '0', '1', '2.0'\n"
        assert C.log.csv() == expected_str

        print()
        print()

    @pytest.mark.parametrize(
            'scheduler_conditions, multi_run', [
                (False, False),
                (True, False),
                (True, True)
            ]
    )
    def test_log_multi_calls_single_timestep(self, scheduler_conditions, multi_run):
        lca = pnl.LCAMechanism(
                size=2,
                leak=0.5,
                threshold=0.515,
                reset_stateful_function_when=pnl.AtTrialStart()
        )
        lca.set_log_conditions(pnl.VALUE)
        m0 = pnl.ProcessingMechanism(
                size=2
        )
        comp = pnl.Composition()
        comp.add_linear_processing_pathway([m0, lca])
        if scheduler_conditions:
            comp.scheduler.add_condition(lca, pnl.AfterNCalls(m0, 2))
        comp.run(inputs={m0: [[1, 0], [1, 0], [1, 0]]})
        log_dict = lca.log.nparray_dictionary()['Composition-0']
        assert log_dict['Run'] == [[0], [0], [0]]
        assert log_dict['Trial'] == [[0], [1], [2]]
        assert log_dict['Pass'] == [[1], [1], [1]] if scheduler_conditions else [[0], [0], [0]]
        assert log_dict['Time_step'] == [[1], [1], [1]]
        # floats in value, so use np.allclose
        assert np.allclose(log_dict['value'], [[[0.52466739, 0.47533261]] * 3])
        if multi_run:
            comp.run(inputs={m0: [[1, 0], [1, 0], [1, 0]]})
            log_dict = lca.log.nparray_dictionary()['Composition-0']
            assert log_dict['Run'] == [[0], [0], [0], [1], [1], [1]]
            assert np.allclose(log_dict['value'], [[[0.52466739, 0.47533261]] * 6])


class TestClearLog:

    def test_clear_log(self):

        # Create System
        T_1 = pnl.TransferMechanism(name='log_test_T_1', size=2)
        T_2 = pnl.TransferMechanism(name='log_test_T_2', size=2)
        COMP = pnl.Composition(name="log_test_COMP", pathways=[T_1, T_2])
        PJ = T_2.path_afferents[0]

        # Set log conditions on each component
        T_1.set_log_conditions('mod_noise')
        T_1.set_log_conditions(pnl.RESULT)
        T_2.set_log_conditions('mod_slope')
        T_2.set_log_conditions(pnl.RESULT)
        PJ.set_log_conditions('mod_matrix')

        # Run system
        COMP.run(inputs={T_1: [1.0, 1.0]})

        # Create log dict for each component
        log_dict_T_1 = T_1.log.nparray_dictionary()
        log_dict_T_2 = T_2.log.nparray_dictionary()
        log_dict_PJ = PJ.log.nparray_dictionary()

        assert list(log_dict_T_1.keys()) == [COMP.default_execution_id]
        assert list(log_dict_T_2.keys()) == [COMP.default_execution_id]
        assert list(log_dict_PJ.keys()) == [COMP.default_execution_id]

        # Confirm that values were logged correctly
        sys_log_dict = log_dict_T_1[COMP.default_execution_id]
        assert np.allclose(sys_log_dict['RESULT'], np.array([[1.0, 1.0]]))
        assert np.allclose(sys_log_dict['mod_noise'], np.array([[0.0]]))

        sys_log_dict = log_dict_T_2[COMP.default_execution_id]
        assert np.allclose(sys_log_dict['RESULT'], np.array([[1.0, 1.0]]))
        assert np.allclose(sys_log_dict['mod_slope'], np.array([[1.0]]))

        sys_log_dict = log_dict_PJ[COMP.default_execution_id]
        assert np.allclose(sys_log_dict['mod_matrix'], np.array([[1.0, 0.0], [0.0, 1.0]]))

        # KDM 10/3/18: below was changed to delete_entry=True because it's not implemented in Parameter logs,
        # and it's not clear this option results in much difference than just deleting the entries and
        # is stated to be included only for future use
        # Clear T_1s log and DO NOT delete entries
        T_1.log.clear_entries(delete_entry=True)

        # Clear T_2s log and delete entries
        T_2.log.clear_entries(delete_entry=True)

        # Create new log dict for each component
        log_dict_T_1 = T_1.log.nparray_dictionary()
        log_dict_T_2 = T_2.log.nparray_dictionary()
        log_dict_PJ = PJ.log.nparray_dictionary()

        # Confirm that T_1 log values were removed
        assert log_dict_T_1 == OrderedDict()

        # Confirm that T_2 log values were removed and dictionary entries were destroyed
        assert log_dict_T_2 == OrderedDict()

        # Confirm that PJ log values were not affected by changes to T_1 and T_2's logs
        assert np.allclose(log_dict_PJ[COMP.default_execution_id]['mod_matrix'], np.array([[1.0, 0.0], [0.0, 1.0]]))

        # Run system again
        COMP.run(inputs={T_1: [2.0, 2.0]})

        # Create new log dict for each component
        log_dict_T_1 = T_1.log.nparray_dictionary()
        log_dict_T_2 = T_2.log.nparray_dictionary()
        log_dict_PJ = PJ.log.nparray_dictionary()

        # Confirm that T_1 log values only include most recent run
        sys_log_dict = log_dict_T_1[COMP.default_execution_id]
        assert np.allclose(sys_log_dict['RESULT'], np.array([[2.0, 2.0]]))
        assert np.allclose(sys_log_dict['mod_noise'], np.array([[0.0]]))
        # NOTE: "Run" value still incremented, but only the most recent one is returned (# runs does not reset to zero)
        assert np.allclose(sys_log_dict['Run'], np.array([[1]]))

        # Confirm that T_2 log values only include most recent run
        sys_log_dict = log_dict_T_2[COMP.default_execution_id]
        assert np.allclose(sys_log_dict['RESULT'], np.array([[2.0, 2.0]]))
        assert np.allclose(sys_log_dict['mod_slope'], np.array([[1.0]]))
        assert np.allclose(sys_log_dict['Run'], np.array([[1]]))

        # Confirm that PJ log values include all runs
        sys_log_dict = log_dict_PJ[COMP.default_execution_id]
        assert np.allclose(sys_log_dict['mod_matrix'], np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]))
        assert np.allclose(sys_log_dict['Run'], np.array([[0], [1]]))

    @pytest.mark.parametrize(
        'insertion_eids, deletion_eids, log_is_empty',
        [
            (['context'], 'context', True),     # fails if string handling not correct due to str being Iterable
            (['context'], ['context'], True),
        ]
    )
    def test_clear_log_arguments(self, insertion_eids, deletion_eids, log_is_empty):
        t = pnl.TransferMechanism()
        c = pnl.Composition()
        c.add_node(t)

        t.parameters.value.log_condition = True

        for eid in insertion_eids:
            c.run({t: 0}, context=eid)

        t.parameters.value.clear_log(deletion_eids)

        if log_is_empty:
            assert len(t.parameters.value.log) == 0
        else:
            assert len(t.parameters.value.log) != 0


class TestFiltering:

    @pytest.fixture(scope='module')
    def node_logged_in_simulation(self):
        Input = pnl.TransferMechanism(name='Input')
        reward = pnl.TransferMechanism(
            output_ports=[pnl.RESULT, pnl.MEAN, pnl.VARIANCE], name='reward')
        Decision = pnl.DDM(
            function=pnl.DriftDiffusionAnalytical(
                drift_rate=(1.0, pnl.ControlProjection(
                    function=pnl.Linear,
                    control_signal_params={pnl.ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)})
                ),
                threshold=(1.0, pnl.ControlProjection(
                    function=pnl.Linear,
                    control_signal_params={pnl.ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)})
                ),
                noise=0.5,
                starting_point=0,
                t0=0.45
            ),
            output_ports=[
                pnl.DECISION_VARIABLE,
                pnl.RESPONSE_TIME,
                pnl.PROBABILITY_UPPER_THRESHOLD],
            name='Decision'
        )

        comp = pnl.Composition(name="evc", retain_old_simulation_data=True)
        comp.add_node(reward, required_roles=[pnl.NodeRole.OUTPUT])
        comp.add_node(Decision, required_roles=[pnl.NodeRole.OUTPUT])
        task_execution_pathway = [Input, pnl.IDENTITY_MATRIX, Decision]
        comp.add_linear_processing_pathway(task_execution_pathway)

        comp.add_controller(
            controller=pnl.OptimizationControlMechanism(
                agent_rep=comp,
                features=[Input.input_port, reward.input_port],
                feature_function=pnl.AdaptiveIntegrator(rate=0.5),
                objective_mechanism=pnl.ObjectiveMechanism(
                    function=pnl.LinearCombination(operation=pnl.PRODUCT),
                    monitor=[
                        reward,
                        Decision.output_ports[pnl.PROBABILITY_UPPER_THRESHOLD],
                        (Decision.output_ports[pnl.RESPONSE_TIME], -1, 1)
                    ]
                ),
                function=pnl.GridSearch(),
                control_signals=[
                    {PROJECTIONS: ("drift_rate", Decision),
                     ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)},
                    {PROJECTIONS: ("threshold", Decision),
                     ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)}
                ]
            )
        )

        comp.enable_controller = True

        stim_list_dict = {
            Input: [0.5, 0.123],
            reward: [20, 20]
        }

        Input.parameters.value.log_condition = True

        comp.run(inputs=stim_list_dict)

        return Input

    def test_node_has_logged_sims(self, node_logged_in_simulation):
        for logged_value, eid_dict in node_logged_in_simulation.log.logged_entries.items():
            for eid in eid_dict:
                if pnl.EID_SIMULATION in str(eid):
                    return
        else:
            assert False, 'No simulation execution_id found in log'

    def test_nparray(self, node_logged_in_simulation):
        for eid in node_logged_in_simulation.log.nparray(exclude_sims=True)[0]:
            assert pnl.EID_SIMULATION not in str(eid)

    def test_nparray_dictionary(self, node_logged_in_simulation):
        for eid in node_logged_in_simulation.log.nparray_dictionary(exclude_sims=True):
            assert pnl.EID_SIMULATION not in str(eid)

    def test_csv(self, node_logged_in_simulation):
        full_csv = node_logged_in_simulation.log.csv(exclude_sims=True)

        # get each row, excluding header
        for row in full_csv.split('\n')[1:]:
            # if present in a row, context will be in the first cell
            assert pnl.EID_SIMULATION not in row.replace("'", '').split(',')[0]


class TestFullModels:
    def test_multilayer(self):

        input_layer = pnl.TransferMechanism(name='input_layer',
                                            function=pnl.Logistic,
                                            size=2)

        hidden_layer_1 = pnl.TransferMechanism(name='hidden_layer_1',
                                               function=pnl.Logistic,
                                               size=5)

        hidden_layer_2 = pnl.TransferMechanism(name='hidden_layer_2',
                                               function=pnl.Logistic,
                                               size=4)

        output_layer = pnl.TransferMechanism(name='output_layer',
                                             function=pnl.Logistic,
                                             size=3)

        input_weights_matrix = (np.arange(2 * 5).reshape((2, 5)) + 1) / (2 * 5)
        middle_weights_matrix = (np.arange(5 * 4).reshape((5, 4)) + 1) / (5 * 4)
        output_weights_matrix = (np.arange(4 * 3).reshape((4, 3)) + 1) / (4 * 3)

        # This projection will be used by the process below by referencing it in the process' pathway;
        #    note: sender and receiver args don't need to be specified
        input_weights = pnl.MappingProjection(
            name='Input Weights',
            matrix=input_weights_matrix,
        )

        # This projection will be used by the process below by assigning its sender and receiver args
        #    to mechanismss in the pathway
        middle_weights = pnl.MappingProjection(
            name='Middle Weights',
            sender=hidden_layer_1,
            receiver=hidden_layer_2,
            matrix=middle_weights_matrix,
        )

        # Commented lines in this projection illustrate variety of ways in which matrix and learning signals can be specified
        output_weights = pnl.MappingProjection(
            name='Output Weights',
            sender=hidden_layer_2,
            receiver=output_layer,
            matrix=output_weights_matrix,
        )

        comp = pnl.Composition(name='multilayer')

        p = [input_layer, input_weights, hidden_layer_1, middle_weights, hidden_layer_2, output_weights, output_layer]
        backprop_pathway = comp.add_backpropagation_learning_pathway(
            pathway=p,
            loss_function='sse',
            learning_rate=1.
        )

        input_dictionary = {backprop_pathway.target: [[0., 0., 1.]],
                            input_layer: [[-1., 30.]]}

        middle_weights.set_log_conditions(('mod_matrix', pnl.PROCESSING))

        comp.learn(inputs=input_dictionary,
                 num_trials=10)

        expected_log_val = np.array(
            [
                ['multilayer'],
                [[
                    [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
                    [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    [[2], [2], [2], [2], [2], [2], [2], [2], [2], [2]],
                    [ [[ 0.05,  0.1 ,  0.15,  0.2 ],
                       [ 0.25,  0.3 ,  0.35,  0.4 ],
                       [ 0.45,  0.5 ,  0.55,  0.6 ],
                       [ 0.65,  0.7 ,  0.75,  0.8 ],
                       [ 0.85,  0.9 ,  0.95,  1.  ]],
                      [[ 0.04789907,  0.09413833,  0.14134241,  0.18938924],
                       [ 0.24780811,  0.29388455,  0.34096758,  0.38892985],
                       [ 0.44772121,  0.49364209,  0.54060947,  0.58849095],
                       [ 0.64763875,  0.69341202,  0.74026967,  0.78807449],
                       [ 0.84756101,  0.89319513,  0.93994932,  0.98768187]],
                      [[ 0.04738148,  0.08891106,  0.13248753,  0.177898  ],
                       [ 0.24726841,  0.28843403,  0.33173452,  0.37694783],
                       [ 0.44716034,  0.48797777,  0.53101423,  0.57603893],
                       [ 0.64705774,  0.6875443 ,  0.73032986,  0.77517531],
                       [ 0.84696096,  0.88713512,  0.92968378,  0.97435998]],
                      [[ 0.04937771,  0.08530344,  0.12439361,  0.16640433],
                       [ 0.24934878,  0.28467436,  0.32329947,  0.36496974],
                       [ 0.44932147,  0.48407216,  0.52225175,  0.56359587],
                       [ 0.64929589,  0.68349948,  0.72125508,  0.76228876],
                       [ 0.84927212,  0.88295836,  0.92031297,  0.96105307]],
                      [[ 0.05440291,  0.08430585,  0.1183739 ,  0.15641064],
                       [ 0.25458348,  0.28363519,  0.3170288 ,  0.35455942],
                       [ 0.45475764,  0.48299299,  0.51573974,  0.55278488],
                       [ 0.65492462,  0.68238209,  0.7145124 ,  0.75109483],
                       [ 0.85508376,  0.88180465,  0.91335119,  0.94949538]],
                      [[ 0.06177218,  0.0860581 ,  0.11525064,  0.14926369],
                       [ 0.26225812,  0.28546004,  0.31377611,  0.34711631],
                       [ 0.46272625,  0.48488774,  0.51236246,  0.54505667],
                       [ 0.66317453,  0.68434373,  0.7110159 ,  0.74309381],
                       [ 0.86360121,  0.88382991,  0.9097413 ,  0.94123489]],
                      [[ 0.06989398,  0.08959148,  0.11465594,  0.14513241],
                       [ 0.27071639,  0.2891398 ,  0.31315677,  0.34281389],
                       [ 0.47150846,  0.48870843,  0.5117194 ,  0.54058946],
                       [ 0.67226675,  0.68829929,  0.71035014,  0.73846891],
                       [ 0.87298831,  0.88791376,  0.90905395,  0.93646   ]],
                      [[ 0.07750784,  0.09371987,  0.11555569,  0.143181  ],
                       [ 0.27864693,  0.29343991,  0.31409396,  0.3407813 ],
                       [ 0.47974374,  0.49317377,  0.5126926 ,  0.53847878],
                       [ 0.68079346,  0.69292265,  0.71135777,  0.73628353],
                       [ 0.88179203,  0.89268732,  0.91009431,  0.93420362]],
                      [[ 0.0841765 ,  0.09776672,  0.11711835,  0.14249779],
                       [ 0.28559463,  0.29765609,  0.31572199,  0.34006951],
                       [ 0.48695967,  0.49755273,  0.51438349,  0.5377395 ],
                       [ 0.68826567,  0.69745713,  0.71310872,  0.735518  ],
                       [ 0.88950757,  0.89736946,  0.91190228,  0.93341316]],
                      [[ 0.08992499,  0.10150104,  0.11891032,  0.14250149],
                       [ 0.29158517,  0.30154765,  0.31758943,  0.34007336],
                       [ 0.49318268,  0.50159531,  0.51632339,  0.5377435 ],
                       [ 0.69471052,  0.70164382,  0.71511777,  0.73552215],
                       [ 0.8961628 ,  0.90169281,  0.91397691,  0.93341744]]]
                ]]
            ],
            dtype=object
        )
        log_val = middle_weights.log.nparray(entries='mod_matrix', header=False)

        assert log_val[0] == expected_log_val[0]

        for i in range(1, len(log_val)):
            try:
                np.testing.assert_allclose(log_val[i], expected_log_val[i])
            except TypeError:
                for j in range(len(log_val[i])):
                    np.testing.assert_allclose(
                        np.array(log_val[i][j][0]),
                        np.array(expected_log_val[i][j][0]),
                        atol=1e-08,
                        err_msg='Failed on test item {0} of logged values'.format(i)
                    )

        middle_weights.log.print_entries()

        # Test Programatic logging
        hidden_layer_2.log.log_values(pnl.VALUE, comp)
        log_val = hidden_layer_2.log.nparray(header=False)
        expected_log_val = np.array(
            [
                ['multilayer'],
                [[
                    [[1]],
                    [[0]],
                    [[0]],
                    [[0]],
                    [[[0.8565238418942037, 0.8601053239957609, 0.8662098921116546, 0.8746933736954071]]]
                ]]
            ],
            dtype=object
        )
        assert log_val[0] == expected_log_val[0]

        for i in range(1, len(log_val)):
            try:
                np.testing.assert_allclose(log_val[i], expected_log_val[i])
            except TypeError:
                for j in range(len(log_val[i])):
                    np.testing.assert_allclose(
                        np.array(log_val[i][j][0]),
                        np.array(expected_log_val[i][j][0]),
                        atol=1e-08,
                        err_msg='Failed on test item {0} of logged values'.format(i)
                    )
        hidden_layer_2.log.print_entries()

        # Clear log and test with logging of weights set to LEARNING for another 5 trials of learning
        middle_weights.log.clear_entries(entries=None, confirm=False)
        middle_weights.set_log_conditions(('mod_matrix', pnl.LEARNING))
        comp.learn(
            num_trials=5,
            inputs=input_dictionary,
        )
        log_val = middle_weights.log.nparray(entries='mod_matrix', header=False)
        expected_log_val = np.array(
            [
                ['multilayer'],
                [[
                        [[1], [1], [1], [1], [1]],  # RUN
                        # [[0], [1], [2], [3], [4]],  # TRIAL
                        [[0], [1], [2], [3], [4]],  # TRIAL
                        [[1], [1], [1], [1], [1]],  # PASS
                        # [[0], [0], [0], [0], [0]],  # PASS
                        # [[1], [1], [1], [1], [1]],  # TIME_STEP
                        [[0], [0], [0], [0], [0]],  # TIME_STEP
                        [  [[0.09925812411381937, 0.1079522130303428, 0.12252820028789306, 0.14345816973727732],
                            [0.30131473371328343, 0.30827285172236585, 0.3213609999139731, 0.3410707131678078],
                            [0.5032924245149345, 0.5085833053183328, 0.5202423523987703, 0.5387798509126243],
                            [0.70518251216691, 0.7088822116145151, 0.7191771716324874, 0.7365956448426355],
                            [0.9069777724600303, 0.9091682860319945, 0.9181692763668221, 0.93452610920817]],
                           [[0.103113468050986, 0.11073719161508278, 0.12424368674464399, 0.14415219181047598],
                            [0.3053351724284921, 0.3111770895557729, 0.3231499474835138, 0.341794454877438],
                            [0.5074709829757806, 0.5116017638574931, 0.5221016574478528, 0.5395320566440044],
                            [0.7095115080472698, 0.7120093413898914, 0.7211034158081356, 0.7373749316571768],
                            [0.9114489813353512, 0.9123981459792809, 0.9201588001021687, 0.935330996581107]],
                          [[0.10656261740658036, 0.11328192907953168, 0.12587702586370172, 0.14490737831188183],
                           [0.30893272045369513, 0.31383131362555394, 0.32485356055342113, 0.3425821330631872],
                           [0.5112105492674988, 0.5143607671543178, 0.5238725230390068, 0.5403508295336265],
                           [0.7133860755337162, 0.7148679468096026, 0.7229382109974996, 0.7382232628724675],
                           [0.9154510531345043, 0.9153508224199809, 0.9220539747533424, 0.936207244690072]],
                          [[0.10967776822419642, 0.11562091141141007, 0.12742795007904037, 0.14569308665620523],
                           [0.3121824816018084, 0.316271366885665, 0.3264715025259811, 0.34340179304134666],
                           [0.5145890402653069, 0.5168974760377518, 0.5255545550838675, 0.5412029579613059],
                           [0.7168868378231593, 0.7174964619674593, 0.7246811176253708, 0.7391062307617761],
                           [0.9190671994078436, 0.9180659725806082, 0.923854327015523, 0.9371193149131859]],
                          [[0.11251466428344682, 0.11778293740676549, 0.12890014813698167, 0.14649079441816393],
                           [0.31514245505635713, 0.3185271913574249, 0.328007571201157, 0.3442341089776976],
                           [0.5176666356203712, 0.5192429413004418, 0.5271516632648602, 0.5420683480396268],
                           [0.7200760707077265, 0.7199270072739019, 0.7263361597421493, 0.7400030122347587],
                           [0.922361699102421, 0.9205767427437028, 0.9255639970037588, 0.9380456963960624]]]
                ]]
            ],
            dtype=object
        )

        assert log_val.shape == expected_log_val.shape
        assert log_val[0] == expected_log_val[0]
        assert len(log_val[1]) == len(expected_log_val[1]) == 1

        for i in range(len(log_val[1][0])):
            try:
                np.testing.assert_allclose(
                    log_val[1][0][i],
                    expected_log_val[1][0][i],
                    err_msg='Failed on test item {0} of logged values'.format(i)
                )
            except TypeError:
                for j in range(len(log_val[1][0][i])):
                    np.testing.assert_allclose(
                        np.array(log_val[1][0][i][j]),
                        np.array(expected_log_val[1][0][i][j]),
                        atol=1e-08,
                        err_msg='Failed on test item {0} of logged values'.format(i)
                    )
