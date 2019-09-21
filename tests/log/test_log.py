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
        PS = pnl.Process(name='log_test_PS', pathway=[T_1, T_2])
        PJ = T_2.path_afferents[0]

        assert T_1.loggable_items == {
            'InputState-0': 'OFF',
            'RESULTS': 'OFF',
            'clip': 'OFF',
            'convergence_criterion': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'function': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_passes': 'OFF',
            'mod_convergence_criterion': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'OFF',
            'noise': 'OFF',
            'previous_value': 'OFF',
            'value': 'OFF',
            'variable': 'OFF'
        }
        assert T_2.loggable_items == {
            'InputState-0': 'OFF',
            'RESULTS': 'OFF',
            'clip': 'OFF',
            'convergence_criterion': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'function': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_passes': 'OFF',
            'mod_convergence_criterion': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'OFF',
            'noise': 'OFF',
            'previous_value': 'OFF',
            'value': 'OFF',
            'variable': 'OFF'
        }
        assert PJ.loggable_items == {
            'exponent': 'OFF',
            'func_bounds': 'OFF',
            'func_has_initializers': 'OFF',
            'func_matrix': 'OFF',
            'func_value': 'OFF',
            'has_initializers': 'OFF',
            'func_variable': 'OFF',
            'matrix': 'OFF',
            'mod_matrix': 'OFF',
            'value': 'OFF',
            'variable': 'OFF',
            'weight': 'OFF'
        }

        T_1.set_log_conditions('mod_noise')
        T_1.set_log_conditions(pnl.RESULTS)
        PJ.set_log_conditions('mod_matrix')

        assert T_1.loggable_items == {
            'InputState-0': 'OFF',
            'RESULTS': 'EXECUTION',
            'clip': 'OFF',
            'convergence_criterion': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'function': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_passes': 'OFF',
            'mod_convergence_criterion': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'EXECUTION',
            'mod_slope': 'OFF',
            'noise': 'OFF',
            'previous_value': 'OFF',
            'value': 'OFF',
            'variable': 'OFF'
        }
        assert T_2.loggable_items == {
            'InputState-0': 'OFF',
            'RESULTS': 'OFF',
            'clip': 'OFF',
            'convergence_criterion': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'function': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_passes': 'OFF',
            'mod_convergence_criterion': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'OFF',
            'noise': 'OFF',
            'previous_value': 'OFF',
            'value': 'OFF',
            'variable': 'OFF'
        }
        assert PJ.loggable_items == {
            'exponent': 'OFF',
            'func_bounds': 'OFF',
            'func_has_initializers': 'OFF',
            'func_matrix': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'has_initializers': 'OFF',
            'matrix': 'OFF',
            'mod_matrix': 'EXECUTION',
            'value': 'OFF',
            'variable': 'OFF',
            'weight': 'OFF'
        }

        PS.execute()
        PS.execute()
        PS.execute()

        assert T_1.logged_items == {'RESULTS': 'EXECUTION', 'mod_noise': 'EXECUTION'}
        assert PJ.logged_items == {'mod_matrix': 'EXECUTION'}

        T_1.log.print_entries(contexts=PS)

        # assert T_1.log.print_entries() ==
        # # Log for mech_A:
        # #
        # # Index     Variable:                                          Context                                                                  Value
        # # 0         'RESULTS'.........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
        # # 1         'RESULTS'.........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
        # #
        # #
        # # 0         'noise'...........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
        # # 1         'noise'...........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
        #
        # assert T_2.log.print_entries() ==
        # # Log for mech_A:
        # #
        # # Index     Variable:                                          Context                                                                  Value
        # # 0         'RESULTS'.........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
        # # 1         'RESULTS'.........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
        # #
        # #
        # # 0         'noise'...........................................' EXECUTING  PROCESS Process-0'.......................................    0.0
        # # 1         'noise'...........................................' EXECUTING  PROCESS Process-0'.......................................    0.0

        T_1_csv = T_1.log.csv(entries=['mod_noise', 'RESULTS'], owner_name=False, quotes=None)
        print(T_1_csv)
        assert T_1_csv == \
            "'Execution Context', 'Data'\n" \
            + "'{0}', \'Index\', \'mod_noise\', \'RESULTS\'\n".format(PS.default_execution_id) \
            + ", 0, 0.0, 0.0 0.0\n" \
            + ", 1, 0.0, 0.0 0.0\n" \
            + ", 2, 0.0, 0.0 0.0\n"

        assert PJ.log.csv(entries='mod_matrix', owner_name=True, quotes=True) == \
            "'Execution Context', 'Data'\n" \
            + "'{0}', \'Index\', \'MappingProjection from log_test_T_1 to log_test_T_2[mod_matrix]\'\n".format(PS.default_execution_id) \
            + ", \'0\', \'1.0 0.0\' \'0.0 1.0\'\n" \
            + ", \'1\', \'1.0 0.0\' \'0.0 1.0\'\n" \
            + ", \'2\', \'1.0 0.0\' \'0.0 1.0\'\n"

        result = T_1.log.nparray(entries=['mod_noise', 'RESULTS'], header=False, owner_name=True)
        assert result[0] == PS.default_execution_id
        np.testing.assert_array_equal(result[1][0],
                                      np.array([[[0], [1], [2]],
                                                [[ 0.], [ 0.], [ 0.]],
                                                [[ 0.,  0.], [ 0.,  0.],[ 0., 0.]]]))

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
        PS = pnl.Process(name='log_test_PS', pathway=[T1, T2])
        PJ = T2.path_afferents[0]

        assert T1.loggable_items == {
            'InputState-0': 'OFF',
            'RESULTS': 'OFF',
            'clip': 'OFF',
            'convergence_criterion': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'function': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_passes': 'OFF',
            'mod_convergence_criterion': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'OFF',
            'noise': 'OFF',
            'previous_value': 'OFF',
            'value': 'OFF',
            'variable': 'OFF'
        }
        assert T2.loggable_items == {
            'InputState-0': 'OFF',
            'RESULTS': 'OFF',
            'clip': 'OFF',
            'convergence_criterion': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'function': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_passes': 'OFF',
            'mod_convergence_criterion': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'OFF',
            'noise': 'OFF',
            'previous_value': 'OFF',
            'value': 'OFF',
            'variable': 'OFF'
        }
        assert PJ.loggable_items == {
            'exponent': 'OFF',
            'func_bounds': 'OFF',
            'func_has_initializers': 'OFF',
            'func_matrix': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'has_initializers': 'OFF',
            'matrix': 'OFF',
            'mod_matrix': 'OFF',
            'value': 'OFF',
            'variable': 'OFF',
            'weight': 'OFF'
        }

        T1.set_log_conditions('mod_slope')
        T1.set_log_conditions(pnl.RESULTS)
        T1.set_log_conditions(pnl.VALUE)
        PJ.set_log_conditions('mod_matrix')
        T2.set_log_conditions('mod_slope')
        T2.set_log_conditions(pnl.RESULTS)
        T2.set_log_conditions(pnl.VALUE)

        assert T1.loggable_items == {
            'InputState-0': 'OFF',
            'RESULTS': 'EXECUTION',
            'clip': 'OFF',
            'convergence_criterion': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'function': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_passes': 'OFF',
            'mod_convergence_criterion': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'EXECUTION',
            'noise': 'OFF',
            'previous_value': 'OFF',
            'value': 'EXECUTION',
            'variable': 'OFF'
        }
        assert T2.loggable_items == {
            'InputState-0': 'OFF',
            'RESULTS': 'EXECUTION',
            'clip': 'OFF',
            'convergence_criterion': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'function': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_passes': 'OFF',
            'mod_convergence_criterion': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'EXECUTION',
            'noise': 'OFF',
            'previous_value': 'OFF',
            'value': 'EXECUTION',
            'variable': 'OFF'
        }
        assert PJ.loggable_items == {
            'exponent': 'OFF',
            'func_bounds': 'OFF',
            'func_has_initializers': 'OFF',
            'func_matrix': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'has_initializers': 'OFF',
            'matrix': 'OFF',
            'mod_matrix': 'EXECUTION',
            'value': 'OFF',
            'variable': 'OFF',
            'weight': 'OFF'
        }

        PS.execute([1.0, 2.0])
        PS.execute([3.0, 4.0])
        PS.execute([5.0, 6.0])

        assert T1.logged_items == {'RESULTS': 'EXECUTION',
                                   'mod_slope': 'EXECUTION',
                                   'value': 'EXECUTION'}
        assert T2.logged_items == {'RESULTS': 'EXECUTION',
                                   'mod_slope': 'EXECUTION',
                                   'value': 'EXECUTION'}
        assert PJ.logged_items == {'mod_matrix': 'EXECUTION'}

        log_dict_T1 = T1.log.nparray_dictionary(entries=['value', 'mod_slope', 'RESULTS'])

        expected_values_T1 = [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]
        expected_slopes_T1 = [[1.0], [1.0], [1.0]]
        expected_results_T1 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

        assert list(log_dict_T1.keys()) == [PS.default_execution_id]

        assert np.allclose(expected_values_T1, log_dict_T1[PS.default_execution_id]['value'])
        assert np.allclose(expected_slopes_T1, log_dict_T1[PS.default_execution_id]['mod_slope'])
        assert np.allclose(expected_results_T1, log_dict_T1[PS.default_execution_id]['RESULTS'])

        assert list(log_dict_T1[PS.default_execution_id].keys()) == ['Index', 'value', 'mod_slope', 'RESULTS']

        log_dict_T1_reorder = T1.log.nparray_dictionary(entries=['mod_slope', 'value', 'RESULTS'])

        assert list(log_dict_T1_reorder[PS.default_execution_id].keys()) == ['Index', 'mod_slope', 'value', 'RESULTS']

    def test_run_resets(self):
        import psyneulink as pnl
        T1 = pnl.TransferMechanism(name='log_test_T1',
                                   size=2)
        T2 = pnl.TransferMechanism(name='log_test_T2',
                                   size=2)
        PS = pnl.Process(name='log_test_PS', pathway=[T1, T2])
        SYS = pnl.System(name='log_test_SYS', processes=[PS])
        T1.set_log_conditions('mod_slope')
        T2.set_log_conditions('mod_slope')
        SYS.run(inputs={T1: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]})

        log_array_T1 = T1.log.nparray()
        log_array_T2 = T2.log.nparray()

        SYS.run(inputs={T1: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]})

        log_array_T1_second_run = T1.log.nparray()
        log_array_T2_second_run = T2.log.nparray()

    def test_log_dictionary_with_time(self):

        T1 = pnl.TransferMechanism(name='log_test_T1',
                                   size=2)
        T2 = pnl.TransferMechanism(name='log_test_T2',
                                   function=psyneulink.core.components.functions.transferfunctions.Linear(slope=2.0),
                                   size=2)
        PS = pnl.Process(name='log_test_PS', pathway=[T1, T2])
        SYS = pnl.System(name='log_test_SYS', processes=[PS])

        assert T1.loggable_items == {
            'InputState-0': 'OFF',
            'RESULTS': 'OFF',
            'clip': 'OFF',
            'convergence_criterion': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'function': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_passes': 'OFF',
            'mod_convergence_criterion': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'OFF',
            'noise': 'OFF',
            'previous_value': 'OFF',
            'value': 'OFF',
            'variable': 'OFF'
        }
        assert T2.loggable_items == {
            'InputState-0': 'OFF',
            'RESULTS': 'OFF',
            'clip': 'OFF',
            'convergence_criterion': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'function': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_passes': 'OFF',
            'mod_convergence_criterion': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'OFF',
            'noise': 'OFF',
            'previous_value': 'OFF',
            'value': 'OFF',
            'variable': 'OFF'
        }

        T1.set_log_conditions('mod_slope')
        T1.set_log_conditions(pnl.RESULTS)
        T1.set_log_conditions(pnl.VALUE)

        assert T1.loggable_items == {
            'InputState-0': 'OFF',
            'RESULTS': 'EXECUTION',
            'clip': 'OFF',
            'convergence_criterion': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'function': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_passes': 'OFF',
            'mod_convergence_criterion': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'EXECUTION',
            'noise': 'OFF',
            'previous_value': 'OFF',
            'value': 'EXECUTION',
            'variable': 'OFF'
        }

        T2.set_log_conditions('mod_slope')
        T2.set_log_conditions(pnl.RESULTS)
        T2.set_log_conditions(pnl.VALUE)

        assert T2.loggable_items == {
            'InputState-0': 'OFF',
            'RESULTS': 'EXECUTION',
            'clip': 'OFF',
            'convergence_criterion': 'OFF',
            'func_additive_param': 'OFF',
            'func_bounds': 'OFF',
            'func_has_initializers': 'OFF',
            'func_intercept': 'OFF',
            'func_multiplicative_param': 'OFF',
            'func_slope': 'OFF',
            'func_value': 'OFF',
            'func_variable': 'OFF',
            'function': 'OFF',
            'has_initializers': 'OFF',
            'initial_value': 'OFF',
            'integration_rate': 'OFF',
            'integrator_function_value': 'OFF',
            'integrator_mode': 'OFF',
            'max_passes': 'OFF',
            'mod_convergence_criterion': 'OFF',
            'mod_integration_rate': 'OFF',
            'mod_intercept': 'OFF',
            'mod_noise': 'OFF',
            'mod_slope': 'EXECUTION',
            'noise': 'OFF',
            'previous_value': 'OFF',
            'value': 'EXECUTION',
            'variable': 'OFF'
        }

        # RUN ZERO  |  TRIALS ZERO, ONE, TWO ----------------------------------

        SYS.run(inputs={T1: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]})

        assert T1.logged_items == {'RESULTS': 'EXECUTION',
                                   'mod_slope': 'EXECUTION',
                                   'value': 'EXECUTION'}
        assert T2.logged_items == {'RESULTS': 'EXECUTION',
                                   'mod_slope': 'EXECUTION',
                                   'value': 'EXECUTION'}

        # T1 log after zero-th run -------------------------------------------

        log_dict_T1 = T1.log.nparray_dictionary(entries=['value', 'mod_slope', 'RESULTS'])

        expected_run_T1 = [[0], [0], [0]]
        expected_trial_T1 = [[0], [1], [2]]
        expected_time_step_T1 = [[0], [0], [0]]
        expected_values_T1 = [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]
        expected_slopes_T1 = [[1.0], [1.0], [1.0]]
        expected_results_T1 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

        assert list(log_dict_T1.keys()) == [SYS.default_execution_id]
        sys_log_dict = log_dict_T1[SYS.default_execution_id]

        assert np.allclose(expected_run_T1, sys_log_dict['Run'])
        assert np.allclose(expected_trial_T1, sys_log_dict['Trial'])
        assert np.allclose(expected_time_step_T1, sys_log_dict['Time_step'])
        assert np.allclose(expected_values_T1, sys_log_dict['value'])
        assert np.allclose(expected_slopes_T1, sys_log_dict['mod_slope'])
        assert np.allclose(expected_results_T1, sys_log_dict['RESULTS'])

        # T2 log after zero-th run --------------------------------------------

        log_dict_T2 = T2.log.nparray_dictionary(entries=['value', 'mod_slope', 'RESULTS'])

        expected_run_T2 = [[0], [0], [0]]
        expected_trial_T2 = [[0], [1], [2]]
        expected_time_step_T2 = [[1], [1], [1]]
        expected_values_T2 = [[[2.0, 4.0]], [[6.0, 8.0]], [[10.0, 12.0]]]
        expected_slopes_T2 = [[2.0], [2.0], [2.0]]
        expected_results_T2 = [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]

        assert list(log_dict_T2.keys()) == [SYS.default_execution_id]
        sys_log_dict = log_dict_T2[SYS.default_execution_id]

        assert np.allclose(expected_run_T2, sys_log_dict['Run'])
        assert np.allclose(expected_trial_T2, sys_log_dict['Trial'])
        assert np.allclose(expected_time_step_T2, sys_log_dict['Time_step'])
        assert np.allclose(expected_values_T2, sys_log_dict['value'])
        assert np.allclose(expected_slopes_T2, sys_log_dict['mod_slope'])
        assert np.allclose(expected_results_T2, sys_log_dict['RESULTS'])

        # RUN ONE  |  TRIALS ZERO, ONE, TWO -------------------------------------

        SYS.run(inputs={T1: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]})

        # T1 log after first run -------------------------------------------

        log_dict_T1 = T1.log.nparray_dictionary(entries=['value', 'mod_slope', 'RESULTS'])

        assert list(log_dict_T1.keys()) == [SYS.default_execution_id]
        sys_log_dict = log_dict_T1[SYS.default_execution_id]

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
        assert np.allclose(expected_results_T1_2, sys_log_dict['RESULTS'])

        # T2 log after first run -------------------------------------------

        log_dict_T2_2 = T2.log.nparray_dictionary(entries=['value', 'mod_slope', 'RESULTS'])

        assert list(log_dict_T2_2.keys()) == [SYS.default_execution_id]
        sys_log_dict = log_dict_T2_2[SYS.default_execution_id]

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
        assert np.allclose(expected_results_T2_2, sys_log_dict['RESULTS'])

    def test_log_dictionary_with_scheduler(self):
        T1 = pnl.TransferMechanism(name='log_test_T1',
                                   integrator_mode=True,
                                   integration_rate=0.5)
        T2 = pnl.TransferMechanism(name='log_test_T2',
                                   function=psyneulink.core.components.functions.transferfunctions.Linear(slope=6.0))
        PS = pnl.Process(name='log_test_PS', pathway=[T1, T2])
        SYS = pnl.System(name='log_test_SYS', processes=[PS])

        def pass_threshold(mech, thresh):
            results = mech.output_states[0].parameters.value.get(SYS)
            for val in results:
                if abs(val) >= thresh:
                    return True
            return False

        terminate_trial = {
            pnl.TimeScale.TRIAL: pnl.While(pass_threshold, T2, 5.0)
        }

        T1.set_log_conditions(pnl.VALUE)
        T1.set_log_conditions('mod_slope')
        T1.set_log_conditions(pnl.RESULTS)
        T2.set_log_conditions(pnl.VALUE)
        T2.set_log_conditions('mod_slope')

        SYS.run(inputs={T1: [[1.0]]}, termination_processing=terminate_trial)

        log_dict_T1 = T1.log.nparray_dictionary(entries=['RESULTS', 'mod_slope', 'value'])
        log_dict_T2 = T2.log.nparray_dictionary(entries=['value', 'mod_slope'])

        assert list(log_dict_T1.keys()) == [SYS.default_execution_id]
        sys_log_dict = log_dict_T1[SYS.default_execution_id]

        # Check order of keys (must match order of specification)
        assert list(sys_log_dict.keys()) == ['Run', 'Trial', 'Pass', 'Time_step', 'RESULTS', 'mod_slope', 'value']
        assert list(log_dict_T2[SYS.default_execution_id].keys()) == ['Run', 'Trial', 'Pass', 'Time_step', 'value', 'mod_slope']

        # Check values T1
        assert np.allclose(sys_log_dict["Run"], [[0], [0], [0]])
        assert np.allclose(sys_log_dict["Trial"], [[0], [0], [0]])
        assert np.allclose(sys_log_dict["Time_step"], [[0], [0], [0]])
        assert np.allclose(sys_log_dict["RESULTS"], [[0.5], [0.75], [0.875]])
        assert np.allclose(sys_log_dict["value"], [[[0.5]], [[0.75]], [[0.875]]])
        assert np.allclose(sys_log_dict["mod_slope"], [[1], [1], [1]])

        # Check values T2
        assert np.allclose(log_dict_T2[SYS.default_execution_id]["Run"], [[0], [0], [0]])
        assert np.allclose(log_dict_T2[SYS.default_execution_id]["Trial"], [[0], [0], [0]])
        assert np.allclose(log_dict_T2[SYS.default_execution_id]["Time_step"], [[1], [1], [1]])
        assert np.allclose(log_dict_T2[SYS.default_execution_id]["value"], [[[3]], [[4.5]], [[5.25]]])
        assert np.allclose(log_dict_T2[SYS.default_execution_id]["mod_slope"], [[6], [6], [6]])

    def test_log_array_with_scheduler(self):
        T1 = pnl.TransferMechanism(name='log_test_T1',
                                   integrator_mode=True,
                                   integration_rate=0.5)
        T2 = pnl.TransferMechanism(name='log_test_T2',
                                   function=psyneulink.core.components.functions.transferfunctions.Linear(slope=6.0))
        PS = pnl.Process(name='log_test_PS', pathway=[T1, T2])
        SYS = pnl.System(name='log_test_SYS', processes=[PS])

        def pass_threshold(mech, thresh):
            results = mech.output_states[0].parameters.value.get(SYS)
            for val in results:
                if abs(val) >= thresh:
                    return True
            return False

        terminate_trial = {
            pnl.TimeScale.TRIAL: pnl.While(pass_threshold, T2, 5.0)
        }

        T1.set_log_conditions(pnl.VALUE)
        T1.set_log_conditions('mod_slope')
        T1.set_log_conditions(pnl.RESULTS)
        T2.set_log_conditions(pnl.VALUE)
        T2.set_log_conditions('mod_slope')

        SYS.run(inputs={T1: [[1.0]]}, termination_processing=terminate_trial)

        log_array_T1 = T1.log.nparray(entries=['RESULTS', 'mod_slope', 'value'])
        log_array_T2 = T2.log.nparray(entries=['value', 'mod_slope'])

        context_results = [pnl.Log.context_header, SYS.default_execution_id]
        # Check values
        run_results = [["Run"], [0], [0], [0]]
        trial_results = [["Trial"], [0], [0], [0]]
        pass_results = [["Pass"], [0], [1], [2]]
        time_step_results = [["Time_step"], [0], [0], [0]]
        results_results = ["RESULTS", [0.5], [0.75], [0.875]]
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
        PS = pnl.Process(name='log_test_PS', pathway=[T1])
        SYS = pnl.System(name='log_test_SYS', processes=[PS])

        def pass_threshold(mech, thresh):
            results = mech.output_states[0].parameters.value.get(SYS)
            for val in results:
                if abs(val) >= thresh:
                    return True
            return False

        terminate_trial = {
            pnl.TimeScale.TRIAL: pnl.While(pass_threshold, T1, 0.95)
        }

        T1.set_log_conditions(pnl.VALUE)

        SYS.run(inputs={T1: [[1.0]]}, termination_processing=terminate_trial)

        log_dict_T1 = T1.log.nparray_dictionary(entries=['value'])

        assert list(log_dict_T1.keys()) == [SYS.default_execution_id]
        sys_log_dict = log_dict_T1[SYS.default_execution_id]

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


class TestClearLog:

    def test_clear_log(self):

        # Create System
        T_1 = pnl.TransferMechanism(name='log_test_T_1', size=2)
        T_2 = pnl.TransferMechanism(name='log_test_T_2', size=2)
        PS = pnl.Process(name='log_test_PS', pathway=[T_1, T_2])
        PJ = T_2.path_afferents[0]
        SYS = pnl.System(name="log_test_SYS", processes=[PS])

        # Set log conditions on each component
        T_1.set_log_conditions('mod_noise')
        T_1.set_log_conditions(pnl.RESULTS)
        T_2.set_log_conditions('mod_slope')
        T_2.set_log_conditions(pnl.RESULTS)
        PJ.set_log_conditions('mod_matrix')

        # Run system
        SYS.run(inputs={T_1: [1.0, 1.0]})

        # Create log dict for each component
        log_dict_T_1 = T_1.log.nparray_dictionary()
        log_dict_T_2 = T_2.log.nparray_dictionary()
        log_dict_PJ = PJ.log.nparray_dictionary()

        assert list(log_dict_T_1.keys()) == [SYS.default_execution_id]
        assert list(log_dict_T_2.keys()) == [SYS.default_execution_id]
        assert list(log_dict_PJ.keys()) == [SYS.default_execution_id]

        # Confirm that values were logged correctly
        sys_log_dict = log_dict_T_1[SYS.default_execution_id]
        assert np.allclose(sys_log_dict['RESULTS'], np.array([[1.0, 1.0]]))
        assert np.allclose(sys_log_dict['mod_noise'], np.array([[0.0]]))

        sys_log_dict = log_dict_T_2[SYS.default_execution_id]
        assert np.allclose(sys_log_dict['RESULTS'], np.array([[1.0, 1.0]]))
        assert np.allclose(sys_log_dict['mod_slope'], np.array([[1.0]]))

        sys_log_dict = log_dict_PJ[SYS.default_execution_id]
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
        assert np.allclose(log_dict_PJ[SYS.default_execution_id]['mod_matrix'], np.array([[1.0, 0.0], [0.0, 1.0]]))

        # Run system again
        SYS.run(inputs={T_1: [2.0, 2.0]})

        # Create new log dict for each component
        log_dict_T_1 = T_1.log.nparray_dictionary()
        log_dict_T_2 = T_2.log.nparray_dictionary()
        log_dict_PJ = PJ.log.nparray_dictionary()

        # Confirm that T_1 log values only include most recent run
        sys_log_dict = log_dict_T_1[SYS.default_execution_id]
        assert np.allclose(sys_log_dict['RESULTS'], np.array([[2.0, 2.0]]))
        assert np.allclose(sys_log_dict['mod_noise'], np.array([[0.0]]))
        # NOTE: "Run" value still incremented, but only the most recent one is returned (# runs does not reset to zero)
        assert np.allclose(sys_log_dict['Run'], np.array([[1]]))

        # Confirm that T_2 log values only include most recent run
        sys_log_dict = log_dict_T_2[SYS.default_execution_id]
        assert np.allclose(sys_log_dict['RESULTS'], np.array([[2.0, 2.0]]))
        assert np.allclose(sys_log_dict['mod_slope'], np.array([[1.0]]))
        assert np.allclose(sys_log_dict['Run'], np.array([[1]]))

        # Confirm that PJ log values include all runs
        sys_log_dict = log_dict_PJ[SYS.default_execution_id]
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
            output_states=[pnl.RESULT, pnl.OUTPUT_MEAN, pnl.OUTPUT_VARIANCE], name='reward')
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
            output_states=[
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
                features=[Input.input_state, reward.input_state],
                feature_function=pnl.AdaptiveIntegrator(rate=0.5),
                objective_mechanism=pnl.ObjectiveMechanism(
                    function=pnl.LinearCombination(operation=pnl.PRODUCT),
                    monitor=[
                        reward,
                        Decision.output_states[pnl.PROBABILITY_UPPER_THRESHOLD],
                        (Decision.output_states[pnl.RESPONSE_TIME], -1, 1)
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
