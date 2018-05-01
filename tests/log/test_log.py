import pytest

import psyneulink as pnl
import numpy as np
from collections import OrderedDict

class TestLog:

    def test_log(self):

        T_1 = pnl.TransferMechanism(name='log_test_T_1', size=2)
        T_2 = pnl.TransferMechanism(name='log_test_T_2', size=2)
        PS = pnl.Process(name='log_test_PS', pathway=[T_1, T_2])
        PJ = T_2.path_afferents[0]

        assert T_1.loggable_items == {'InputState-0': 'OFF',
                                     'slope': 'OFF',
                                     'RESULTS': 'OFF',
                                     'intercept': 'OFF',
                                     'noise': 'OFF',
                                     'smoothing_factor': 'OFF',
                                     'value': 'OFF'}
        assert T_2.loggable_items == {'InputState-0': 'OFF',
                                     'slope': 'OFF',
                                     'RESULTS': 'OFF',
                                     'intercept': 'OFF',
                                     'noise': 'OFF',
                                     'smoothing_factor': 'OFF',
                                     'value': 'OFF'}
        assert PJ.loggable_items == {'matrix': 'OFF',
                                     'value': 'OFF'}

        T_1.set_log_conditions(pnl.NOISE)
        T_1.set_log_conditions(pnl.RESULTS)
        PJ.set_log_conditions(pnl.MATRIX)

        assert T_1.loggable_items == {'InputState-0': 'OFF',
                                     'slope': 'OFF',
                                     'RESULTS': 'EXECUTION',
                                     'intercept': 'OFF',
                                     'noise': 'EXECUTION',
                                     'smoothing_factor': 'OFF',
                                     'value': 'OFF'}
        assert T_2.loggable_items == {'InputState-0': 'OFF',
                                     'slope': 'OFF',
                                     'RESULTS': 'OFF',
                                     'intercept': 'OFF',
                                     'noise': 'OFF',
                                     'smoothing_factor': 'OFF',
                                     'value': 'OFF'}
        assert PJ.loggable_items == {'matrix': 'EXECUTION',
                                     'value': 'OFF'}

        PS.execute()
        PS.execute()
        PS.execute()

        assert T_1.logged_items == {'RESULTS': 'EXECUTION', 'noise': 'EXECUTION'}
        assert PJ.logged_items == {'matrix': 'EXECUTION'}

        T_1.log.print_entries()

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

        print(T_1.log.csv(entries=['noise', 'RESULTS'], owner_name=False, quotes=None))
        assert T_1.log.csv(entries=['noise', 'RESULTS'], owner_name=False, quotes=None) == \
                        "\'Index\', \'noise\', \'RESULTS\'\n0, 0.0, 0.0 0.0\n1, 0.0, 0.0 0.0\n2, 0.0, 0.0 0.0\n"

        assert PJ.log.csv(entries='matrix', owner_name=True, quotes=True) == \
               "\'Index\', \'MappingProjection from log_test_T_1 to log_test_T_2[matrix]\'\n" \
               "\'0\', \'1.0 0.0\' \'0.0 1.0\'\n" \
               "\'1\', \'1.0 0.0\' \'0.0 1.0\'\n" \
               "\'2\', \'1.0 0.0\' \'0.0 1.0\'\n"

        result = T_1.log.nparray(entries=['noise', 'RESULTS'], header=False, owner_name=True)
        np.testing.assert_array_equal(result,
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

        assert T1.loggable_items == {'InputState-0': 'OFF',
                                     'slope': 'OFF',
                                     'RESULTS': 'OFF',
                                     'intercept': 'OFF',
                                     'noise': 'OFF',
                                     'smoothing_factor': 'OFF',
                                     'value': 'OFF'}
        assert T2.loggable_items == {'InputState-0': 'OFF',
                                     'slope': 'OFF',
                                     'RESULTS': 'OFF',
                                     'intercept': 'OFF',
                                     'noise': 'OFF',
                                     'smoothing_factor': 'OFF',
                                     'value': 'OFF'}
        assert PJ.loggable_items == {'matrix': 'OFF',
                                     'value': 'OFF'}

        T1.set_log_conditions(pnl.SLOPE)
        T1.set_log_conditions(pnl.RESULTS)
        T1.set_log_conditions(pnl.VALUE)
        PJ.set_log_conditions(pnl.MATRIX)
        T2.set_log_conditions(pnl.SLOPE)
        T2.set_log_conditions(pnl.RESULTS)
        T2.set_log_conditions(pnl.VALUE)

        assert T1.loggable_items == {'InputState-0': 'OFF',
                                     'slope': 'EXECUTION',
                                     'RESULTS': 'EXECUTION',
                                     'intercept': 'OFF',
                                     'noise': 'OFF',
                                     'smoothing_factor': 'OFF',
                                     'value': 'EXECUTION'}
        assert T2.loggable_items == {'InputState-0': 'OFF',
                                     'slope': 'EXECUTION',
                                     'RESULTS': 'EXECUTION',
                                     'intercept': 'OFF',
                                     'noise': 'OFF',
                                     'smoothing_factor': 'OFF',
                                     'value': 'EXECUTION'}
        assert PJ.loggable_items == {'matrix': 'EXECUTION',
                                     'value': 'OFF'}

        PS.execute([1.0, 2.0])
        PS.execute([3.0, 4.0])
        PS.execute([5.0, 6.0])

        assert T1.logged_items == {'RESULTS': 'EXECUTION',
                                   'slope': 'EXECUTION',
                                   'value': 'EXECUTION'}
        assert T2.logged_items == {'RESULTS': 'EXECUTION',
                                   'slope': 'EXECUTION',
                                   'value': 'EXECUTION'}
        assert PJ.logged_items == {'matrix': 'EXECUTION'}

        log_dict_T1 = T1.log.nparray_dictionary(entries=['value', 'slope', 'RESULTS'])

        expected_values_T1 = [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]
        expected_slopes_T1 = [[1.0], [1.0], [1.0]]
        expected_results_T1 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

        assert np.allclose(expected_values_T1, log_dict_T1['value'])
        assert np.allclose(expected_slopes_T1, log_dict_T1['slope'])
        assert np.allclose(expected_results_T1, log_dict_T1['RESULTS'])

        assert list(log_dict_T1.keys()) == ['Index', 'value', 'slope', 'RESULTS']

        log_dict_T1_reorder = T1.log.nparray_dictionary(entries=['slope', 'value', 'RESULTS'])

        assert list(log_dict_T1_reorder.keys()) == ['Index', 'slope', 'value', 'RESULTS']

    def test_run_resets(self):
        import psyneulink as pnl
        T1 = pnl.TransferMechanism(name='log_test_T1',
                                   size=2)
        T2 = pnl.TransferMechanism(name='log_test_T2',
                                   size=2)
        PS = pnl.Process(name='log_test_PS', pathway=[T1, T2])
        SYS = pnl.System(name='log_test_SYS', processes=[PS])
        T1.set_log_conditions(pnl.SLOPE)
        T2.set_log_conditions(pnl.SLOPE)
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
                                   function=pnl.Linear(slope=2.0),
                                   size=2)
        PS = pnl.Process(name='log_test_PS', pathway=[T1, T2])
        SYS = pnl.System(name='log_test_SYS', processes=[PS])

        assert T1.loggable_items == {'InputState-0': 'OFF',
                                     'slope': 'OFF',
                                     'RESULTS': 'OFF',
                                     'intercept': 'OFF',
                                     'noise': 'OFF',
                                     'smoothing_factor': 'OFF',
                                     'value': 'OFF'}
        assert T2.loggable_items == {'InputState-0': 'OFF',
                                     'slope': 'OFF',
                                     'RESULTS': 'OFF',
                                     'intercept': 'OFF',
                                     'noise': 'OFF',
                                     'smoothing_factor': 'OFF',
                                     'value': 'OFF'}

        T1.set_log_conditions(pnl.SLOPE)
        T1.set_log_conditions(pnl.RESULTS)
        T1.set_log_conditions(pnl.VALUE)

        assert T1.loggable_items == {'InputState-0': 'OFF',
                                     'slope': 'EXECUTION',
                                     'RESULTS': 'EXECUTION',
                                     'intercept': 'OFF',
                                     'noise': 'OFF',
                                     'smoothing_factor': 'OFF',
                                     'value': 'EXECUTION'}

        T2.set_log_conditions(pnl.SLOPE)
        T2.set_log_conditions(pnl.RESULTS)
        T2.set_log_conditions(pnl.VALUE)

        assert T2.loggable_items == {'InputState-0': 'OFF',
                                     'slope': 'EXECUTION',
                                     'RESULTS': 'EXECUTION',
                                     'intercept': 'OFF',
                                     'noise': 'OFF',
                                     'smoothing_factor': 'OFF',
                                     'value': 'EXECUTION'}

        # RUN ZERO  |  TRIALS ZERO, ONE, TWO ----------------------------------

        SYS.run(inputs={T1: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]})

        assert T1.logged_items == {'RESULTS': 'EXECUTION',
                                   'slope': 'EXECUTION',
                                   'value': 'EXECUTION'}
        assert T2.logged_items == {'RESULTS': 'EXECUTION',
                                   'slope': 'EXECUTION',
                                   'value': 'EXECUTION'}

        # T1 log after zero-th run -------------------------------------------

        log_dict_T1 = T1.log.nparray_dictionary(entries=['value', 'slope', 'RESULTS'])

        expected_run_T1 = [[0], [0], [0]]
        expected_trial_T1 = [[0], [1], [2]]
        expected_time_step_T1 = [[0], [0], [0]]
        expected_values_T1 = [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]
        expected_slopes_T1 = [[1.0], [1.0], [1.0]]
        expected_results_T1 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

        assert np.allclose(expected_run_T1, log_dict_T1['Run'])
        assert np.allclose(expected_trial_T1, log_dict_T1['Trial'])
        assert np.allclose(expected_time_step_T1, log_dict_T1['Time_step'])
        assert np.allclose(expected_values_T1, log_dict_T1['value'])
        assert np.allclose(expected_slopes_T1, log_dict_T1['slope'])
        assert np.allclose(expected_results_T1, log_dict_T1['RESULTS'])

        # T2 log after zero-th run --------------------------------------------

        log_dict_T2 = T2.log.nparray_dictionary(entries=['value', 'slope', 'RESULTS'])

        expected_run_T2 = [[0], [0], [0]]
        expected_trial_T2 = [[0], [1], [2]]
        expected_time_step_T2 = [[1], [1], [1]]
        expected_values_T2 = [[[2.0, 4.0]], [[6.0, 8.0]], [[10.0, 12.0]]]
        expected_slopes_T2 = [[2.0], [2.0], [2.0]]
        expected_results_T2 = [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]

        assert np.allclose(expected_run_T2, log_dict_T2['Run'])
        assert np.allclose(expected_trial_T2, log_dict_T2['Trial'])
        assert np.allclose(expected_time_step_T2, log_dict_T2['Time_step'])
        assert np.allclose(expected_values_T2, log_dict_T2['value'])
        assert np.allclose(expected_slopes_T2, log_dict_T2['slope'])
        assert np.allclose(expected_results_T2, log_dict_T2['RESULTS'])

        # RUN ONE  |  TRIALS ZERO, ONE, TWO -------------------------------------

        SYS.run(inputs={T1: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]})

        # T1 log after first run -------------------------------------------

        log_dict_T1 = T1.log.nparray_dictionary(entries=['value', 'slope', 'RESULTS'])

        # expected_run_T1_2 = [[1], [1], [1]]
        expected_run_T1_2 = [[0], [0], [0]] + expected_run_T1
        expected_trial_T1_2 = [[0], [1], [2]] + expected_trial_T1
        expected_time_step_T1_2 = [[0], [0], [0]] + expected_time_step_T1
        expected_values_T1_2 = expected_values_T1 + expected_values_T1
        expected_slopes_T1_2 = expected_slopes_T1 + expected_slopes_T1
        expected_results_T1_2 = expected_results_T1 + expected_results_T1

        # assert np.allclose(expected_run_T1_2, log_dict_T1['Run'])
        # assert np.allclose(expected_trial_T1_2, log_dict_T1['Trial'])
        # assert np.allclose(expected_time_step_T1_2, log_dict_T1['Time_step'])
        assert np.allclose(expected_values_T1_2, log_dict_T1['value'])
        assert np.allclose(expected_slopes_T1_2, log_dict_T1['slope'])
        assert np.allclose(expected_results_T1_2, log_dict_T1['RESULTS'])

        # T2 log after first run -------------------------------------------

        log_dict_T2_2 = T2.log.nparray_dictionary(entries=['value', 'slope', 'RESULTS'])

        expected_run_T2_2 = [[0], [0], [0]] + expected_run_T2
        expected_trial_T2_2 = [[0], [1], [2]] + expected_trial_T2
        expected_time_step_T2_2 = [[1], [1], [1]] + expected_time_step_T2
        expected_values_T2_2 = [[[2.0, 4.0]], [[6.0, 8.0]], [[10.0, 12.0]]] + expected_values_T2
        expected_slopes_T2_2 = [[2.0], [2.0], [2.0]] + expected_slopes_T2
        expected_results_T2_2 = [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]] + expected_results_T2

        # assert np.allclose(expected_run_T2_2, log_dict_T2_2['Run'])
        # assert np.allclose(expected_trial_T2_2, log_dict_T2_2['Trial'])
        # assert np.allclose(expected_time_step_T2_2, log_dict_T2_2['Time_step'])
        assert np.allclose(expected_values_T2_2, log_dict_T2_2['value'])
        assert np.allclose(expected_slopes_T2_2, log_dict_T2_2['slope'])
        assert np.allclose(expected_results_T2_2, log_dict_T2_2['RESULTS'])

    def test_log_dictionary_with_scheduler(self):
        T1 = pnl.TransferMechanism(name='log_test_T1',
                                   integrator_mode=True,
                                   smoothing_factor=0.5)
        T2 = pnl.TransferMechanism(name='log_test_T2',
                                   function=pnl.Linear(slope=6.0))
        PS = pnl.Process(name='log_test_PS', pathway=[T1, T2])
        SYS = pnl.System(name='log_test_SYS', processes=[PS])

        def pass_threshold(mech, thresh):
            results = mech.output_states[0].value
            for val in results:
                if abs(val) >= thresh:
                    return True
            return False

        terminate_trial = {
            pnl.TimeScale.TRIAL: pnl.While(pass_threshold, T2, 5.0)
        }

        T1.set_log_conditions(pnl.VALUE)
        T1.set_log_conditions(pnl.SLOPE)
        T1.set_log_conditions(pnl.RESULTS)
        T2.set_log_conditions(pnl.VALUE)
        T2.set_log_conditions(pnl.SLOPE)

        SYS.run(inputs={T1: [[1.0]]}, termination_processing=terminate_trial)

        log_dict_T1 = T1.log.nparray_dictionary(entries=['RESULTS', 'slope', 'value'])
        log_dict_T2 = T2.log.nparray_dictionary(entries=['value', 'slope'])

        # Check order of keys (must match order of specification)
        assert list(log_dict_T1.keys()) == ['Run', 'Trial', 'Pass', 'Time_step', 'RESULTS', 'slope', 'value']
        assert list(log_dict_T2.keys()) == ['Run', 'Trial', 'Pass', 'Time_step', 'value', 'slope']

        # Check values T1
        assert np.allclose(log_dict_T1["Run"], [[0], [0], [0]])
        assert np.allclose(log_dict_T1["Trial"], [[0], [0], [0]])
        assert np.allclose(log_dict_T1["Time_step"], [[0], [0], [0]])
        assert np.allclose(log_dict_T1["RESULTS"], [[0.5], [0.75], [0.875]])
        assert np.allclose(log_dict_T1["value"], [[[0.5]], [[0.75]], [[0.875]]])
        assert np.allclose(log_dict_T1["slope"], [[1], [1], [1]])

        # Check values T2
        assert np.allclose(log_dict_T2["Run"], [[0], [0], [0]])
        assert np.allclose(log_dict_T2["Trial"], [[0], [0], [0]])
        assert np.allclose(log_dict_T2["Time_step"], [[1], [1], [1]])
        assert np.allclose(log_dict_T2["value"], [[[3]], [[4.5]], [[5.25]]])
        assert np.allclose(log_dict_T2["slope"], [[6], [6], [6]])

    def test_log_array_with_scheduler(self):
        T1 = pnl.TransferMechanism(name='log_test_T1',
                                   integrator_mode=True,
                                   smoothing_factor=0.5)
        T2 = pnl.TransferMechanism(name='log_test_T2',
                                   function=pnl.Linear(slope=6.0))
        PS = pnl.Process(name='log_test_PS', pathway=[T1, T2])
        SYS = pnl.System(name='log_test_SYS', processes=[PS])

        def pass_threshold(mech, thresh):
            results = mech.output_states[0].value
            for val in results:
                if abs(val) >= thresh:
                    return True
            return False

        terminate_trial = {
            pnl.TimeScale.TRIAL: pnl.While(pass_threshold, T2, 5.0)
        }

        T1.set_log_conditions(pnl.VALUE)
        T1.set_log_conditions(pnl.SLOPE)
        T1.set_log_conditions(pnl.RESULTS)
        T2.set_log_conditions(pnl.VALUE)
        T2.set_log_conditions(pnl.SLOPE)

        SYS.run(inputs={T1: [[1.0]]}, termination_processing=terminate_trial)

        log_array_T1 = T1.log.nparray(entries=['RESULTS', 'slope', 'value'])
        log_array_T2 = T2.log.nparray(entries=['value', 'slope'])

        # Check values
        run_results = [["Run"], [0], [0], [0]]
        trial_results = [["Trial"], [0], [0], [0]]
        pass_results = [["Pass"], [0], [1], [2]]
        time_step_results = [["Time_step"], [0], [0], [0]]
        results_results = ["RESULTS", [0.5], [0.75], [0.875]]
        slope_results = ["slope", [1], [1], [1]]
        value_results = ["value", [[0.5]], [[0.75]], [[0.875]]]
        for i in range(4):
            assert log_array_T1[0][i] == run_results[i]
            assert log_array_T1[1][i] == trial_results[i]
            assert log_array_T1[2][i] == pass_results[i]
            assert log_array_T1[3][i] == time_step_results[i]
            assert log_array_T1[4][i] == results_results[i]
            assert log_array_T1[5][i] == slope_results[i]
            assert log_array_T1[6][i] == value_results[i]

        # Check values
        run_results = [["Run"], [0], [0], [0]]
        trial_results = [["Trial"], [0], [0], [0]]
        pass_results = [["Pass"], [0], [1], [2]]
        time_step_results = [["Time_step"], [1], [1], [1]]
        value_results = ["value", [[3]], [[4.5]], [[5.25]]]
        slope_results = ["slope", [6], [6], [6]]
        for i in range(4):
            assert log_array_T2[0][i] == run_results[i]
            assert log_array_T2[1][i] == trial_results[i]
            assert log_array_T2[2][i] == pass_results[i]
            assert log_array_T2[3][i] == time_step_results[i]
            assert log_array_T2[4][i] == value_results[i]
            assert log_array_T2[5][i] == slope_results[i]

    def test_log_dictionary_with_scheduler_many_time_step_increments(self):
        T1 = pnl.TransferMechanism(name='log_test_T1',
                                   integrator_mode=True,
                                   smoothing_factor=0.05)
        PS = pnl.Process(name='log_test_PS', pathway=[T1])
        SYS = pnl.System(name='log_test_SYS', processes=[PS])

        def pass_threshold(mech, thresh):
            results = mech.output_states[0].value
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

        # Check order of keys (must match order of specification)
        assert list(log_dict_T1.keys()) == ['Run', 'Trial', 'Pass', 'Time_step', 'value']

        # # Check values T1
        assert len(log_dict_T1["Run"]) == 59
        assert np.allclose(log_dict_T1["Pass"][30], 30)
        assert np.allclose(log_dict_T1["Time_step"][30], 0)
        assert abs(log_dict_T1["value"][58]) >= 0.95
        assert abs(log_dict_T1["value"][57]) < 0.95

class TestClearLog:

    def test_clear_log(self):

        # Create System
        T_1 = pnl.TransferMechanism(name='log_test_T_1', size=2)
        T_2 = pnl.TransferMechanism(name='log_test_T_2', size=2)
        PS = pnl.Process(name='log_test_PS', pathway=[T_1, T_2])
        PJ = T_2.path_afferents[0]
        SYS = pnl.System(name="log_test_SYS", processes=[PS])

        # Set log conditions on each component
        T_1.set_log_conditions(pnl.NOISE)
        T_1.set_log_conditions(pnl.RESULTS)
        T_2.set_log_conditions(pnl.SLOPE)
        T_2.set_log_conditions(pnl.RESULTS)
        PJ.set_log_conditions(pnl.MATRIX)

        # Run system
        SYS.run(inputs={T_1: [1.0, 1.0]})

        # Create log dict for each component
        log_dict_T_1 = T_1.log.nparray_dictionary()
        log_dict_T_2 = T_2.log.nparray_dictionary()
        log_dict_PJ = PJ.log.nparray_dictionary()

        # Confirm that values were logged correctly
        assert np.allclose(log_dict_T_1['RESULTS'], np.array([[1.0, 1.0]])) and \
               np.allclose(log_dict_T_1['noise'], np.array([[0.0]]))

        assert np.allclose(log_dict_T_2['RESULTS'], np.array([[1.0, 1.0]])) and \
               np.allclose(log_dict_T_2['slope'], np.array([[1.0]]))

        assert np.allclose(log_dict_PJ['matrix'], np.array([[1.0, 0.0], [0.0, 1.0]]))

        # Clear T_1s log and delete entries
        T_1.log.clear_entries(delete_entry=False)

        # Clear T_2s log and DO NOT delete entries
        T_2.log.clear_entries(delete_entry=True)

        # Create new log dict for each component
        log_dict_T_1 = T_1.log.nparray_dictionary()
        log_dict_T_2 = T_2.log.nparray_dictionary()
        log_dict_PJ = PJ.log.nparray_dictionary()

        # Confirm that T_1 log values were removed
        assert np.allclose(log_dict_T_1['RESULTS'], np.array([])) and \
               np.allclose(log_dict_T_1['noise'], np.array([]))

        # Confirm that T_2 log values were removed and dictionary entries were destroyed
        assert log_dict_T_2 == OrderedDict()

        # Confirm that PJ log values were not affected by changes to T_1 and T_2's logs
        assert np.allclose(log_dict_PJ['matrix'], np.array([[1.0, 0.0], [0.0, 1.0]]))

        # Run system again
        SYS.run(inputs={T_1: [2.0, 2.0]})

        # Create new log dict for each component
        log_dict_T_1 = T_1.log.nparray_dictionary()
        log_dict_T_2 = T_2.log.nparray_dictionary()
        log_dict_PJ = PJ.log.nparray_dictionary()

        # Confirm that T_1 log values only include most recent run
        assert np.allclose(log_dict_T_1['RESULTS'], np.array([[2.0, 2.0]])) and \
               np.allclose(log_dict_T_1['noise'], np.array([[0.0]]))
        # NOTE: "Run" value still incremented, but only the most recent one is returned (# runs does not reset to zero)
        assert np.allclose(log_dict_T_1['Run'], np.array([[1]]))

        # Confirm that T_2 log values only include most recent run
        assert np.allclose(log_dict_T_2['RESULTS'], np.array([[2.0, 2.0]])) and \
               np.allclose(log_dict_T_2['slope'], np.array([[1.0]]))
        assert np.allclose(log_dict_T_2['Run'], np.array([[1]]))

        # Confirm that PJ log values include all runs
        assert np.allclose(log_dict_PJ['matrix'], np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])) and \
               np.allclose(log_dict_PJ['Run'], np.array([[0], [1]]))