import numpy as np
import psyneulink as pnl
import pytest

from queue import Queue
from collections import OrderedDict

import psyneulink.core.components.functions.transferfunctions
from psyneulink.core.globals.keywords import ALLOCATION_SAMPLES, PROJECTIONS

class TestRPC:

    def test_transfer_mech(self):

        T_1 = pnl.TransferMechanism(name='log_test_T_1', size=2)
        T_2 = pnl.TransferMechanism(name='log_test_T_2', size=2)
        PS = pnl.Composition(name='log_test_PS', pathways=[T_1, T_2])
        con_with_rpc_pipeline = pnl.Context(rpc_pipeline=Queue(), execution_id=PS)

        T_1.set_log_conditions('mod_noise')
        T_1.set_log_conditions(pnl.RESULT)

        T_1.set_delivery_conditions('mod_noise')
        T_1.set_delivery_conditions(pnl.RESULT)

        PS.run(inputs={T_1: [0, 0]}, context=con_with_rpc_pipeline)
        PS.run(inputs={T_1: [1, 2]}, context=con_with_rpc_pipeline)
        PS.run(inputs={T_1: [3, 4]}, context=con_with_rpc_pipeline)

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

        expected = [
            [[0], [1], [2]],
            [[0.], [0.], [0.]],
            [[0.], [0.], [0.]],
            [[0.], [0.], [0.]],
            [[0.0], [0.0], [0.0]],
            [[0., 0.], [1., 2.], [3., 4.]],
        ]

        actual = []
        pipeline = con_with_rpc_pipeline.rpc_pipeline
        while not pipeline.empty():
            actual.append(pipeline.get())
        t_1_entries = [i for i in actual if i.componentName == 'log_test_T_1']
        noise = [i for i in t_1_entries if i.parameterName == 'noise']
        results = [i for i in t_1_entries if i.parameterName == 'RESULT']
        assert all([
            noise[0].time == '0:0:0:0', noise[0].value.data == [0],results[0].value.data == [0.0, 0.0],
            noise[1].time == '1:0:0:0', noise[1].value.data == [0],results[1].value.data == [1.0, 2.0],
            noise[2].time == '2:0:0:0', noise[2].value.data == [0],results[2].value.data == [3.0, 4.0],
        ])

    def test_delivery_initialization(self):
        T = pnl.TransferMechanism(
                prefs={pnl.DELIVERY_PREF: pnl.PreferenceEntry(pnl.LogCondition.EXECUTION, pnl.PreferenceLevel.INSTANCE)}
        )
        comp = pnl.Composition(name='comp', nodes=[T])
        con_with_rpc_pipeline = pnl.Context(rpc_pipeline=Queue(), execution_id=comp)
        pipeline = con_with_rpc_pipeline.rpc_pipeline
        comp.run([1], context=con_with_rpc_pipeline)
        actual = []
        while not pipeline.empty():
            actual.append(pipeline.get())
        assert all([
            len(actual) == 1,
            actual[0].time == '0:0:0:0',
            actual[0].value.shape == [1, 1],
            actual[0].value.data == [1.0]
        ])

    def test_run_resets(self):
        T1 = pnl.TransferMechanism(name='log_test_T1',
                                   size=2)
        T2 = pnl.TransferMechanism(name='log_test_T2',
                                   size=2)
        COMP = pnl.Composition(name='COMP', pathways=[T1, T2])
        con_with_rpc_pipeline = pnl.Context(rpc_pipeline=Queue(), execution_id=COMP)
        pipeline = con_with_rpc_pipeline.rpc_pipeline
        T1.set_delivery_conditions('mod_slope')
        T2.set_delivery_conditions('value')
        COMP.run(inputs={T1: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]},
                 context=con_with_rpc_pipeline)
        pipeline = con_with_rpc_pipeline.rpc_pipeline
        actual = []
        while not pipeline.empty():
            actual.append(pipeline.get())
        assert all([i.context == 'COMP' for i in actual])
        assert np.allclose([
            np.ndarray(shape=np.array(actual[1].value.shape), buffer=np.array(actual[1].value.data)),
            np.ndarray(shape=np.array(actual[3].value.shape), buffer=np.array(actual[3].value.data)),
            np.ndarray(shape=np.array(actual[5].value.shape), buffer=np.array(actual[5].value.data)),
        ], [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]])

        COMP.run(inputs={T1: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]},
                 context=con_with_rpc_pipeline)
        actual = []
        while not pipeline.empty():
            actual.append(pipeline.get())
        assert np.allclose([
            np.ndarray(shape=np.array(actual[1].value.shape), buffer=np.array(actual[1].value.data)),
            np.ndarray(shape=np.array(actual[3].value.shape), buffer=np.array(actual[3].value.data)),
            np.ndarray(shape=np.array(actual[5].value.shape), buffer=np.array(actual[5].value.data)),
        ], [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]])

    def test_log_dictionary_with_time(self):
        T1 = pnl.TransferMechanism(name='log_test_T1',
                                   size=2)
        T2 = pnl.TransferMechanism(name='log_test_T2',
                                   function=pnl.Linear(slope=2.0),
                                   size=2)
        COMP = pnl.Composition(name='log_test_COMP', pathways=[T1, T2])
        con_with_rpc_pipeline = pnl.Context(rpc_pipeline=Queue(), execution_id=COMP)
        pipeline = con_with_rpc_pipeline.rpc_pipeline

        T1.set_delivery_conditions('mod_slope')
        T1.set_delivery_conditions(pnl.RESULT)
        T1.set_delivery_conditions(pnl.VALUE)

        T2.set_delivery_conditions('mod_slope')
        T2.set_delivery_conditions(pnl.RESULT)
        T2.set_delivery_conditions(pnl.VALUE)

        # RUN ZERO  |  TRIALS ZERO, ONE, TWO ----------------------------------

        COMP.run(inputs={T1: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]},
                 context=con_with_rpc_pipeline)

        actual = []
        while not pipeline.empty():
            actual.append(pipeline.get())

        t1_slope_entries = [i for i in actual if i.parameterName == pnl.SLOPE and i.componentName == 'log_test_T1']
        t1_slope_values = [np.ndarray(shape=np.array(i.value.shape), buffer=np.array(i.value.data)) for i in t1_slope_entries]
        t1_slope_times = [i.time for i in t1_slope_entries]

        t2_slope_entries = [i for i in actual if i.parameterName == pnl.SLOPE and i.componentName == 'log_test_T2']
        t2_slope_values = [np.ndarray(shape=np.array(i.value.shape), buffer=np.array(i.value.data)) for i in t2_slope_entries]
        t2_slope_times = [i.time for i in t2_slope_entries]

        t1_result_entries = [i for i in actual if i.parameterName == pnl.RESULT and i.componentName == 'log_test_T1']
        t1_result_values = [np.ndarray(shape=np.array(i.value.shape), buffer=np.array(i.value.data)) for i in t1_result_entries]
        t1_result_times = [i.time for i in t1_result_entries]

        t2_result_entries = [i for i in actual if i.parameterName == pnl.RESULT and i.componentName == 'log_test_T2']
        t2_result_values = [np.ndarray(shape=np.array(i.value.shape), buffer=np.array(i.value.data)) for i in t2_result_entries]
        t2_result_times = [i.time for i in t2_result_entries]

        t1_value_entries = [i for i in actual if i.parameterName == pnl.VALUE and i.componentName == 'log_test_T1']
        t1_value_values = [np.ndarray(shape=np.array(i.value.shape), buffer=np.array(i.value.data)) for i in t1_value_entries]
        t1_value_times = [i.time for i in t1_value_entries]

        t2_value_entries = [i for i in actual if i.parameterName == pnl.VALUE and i.componentName == 'log_test_T2']
        t2_value_values = [np.ndarray(shape=np.array(i.value.shape), buffer=np.array(i.value.data)) for i in t2_value_entries]
        t2_value_times = [i.time for i in t2_value_entries]

        # Test execution contexts for all entries

        assert all([True if i.context == COMP.default_execution_id else False for i in actual])

        # T1 log after zero-th run -------------------------------------------

        expected_times_T1 = ['0:0:0:0', '0:1:0:0', '0:2:0:0']
        expected_values_T1 = [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]
        expected_slopes_T1 = [[1.0], [1.0], [1.0]]
        expected_results_T1 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

        assert expected_times_T1 == t1_result_times == t1_slope_times == t1_value_times
        assert np.allclose(expected_values_T1, t1_value_values)
        assert np.allclose(expected_results_T1, t1_result_values)
        assert np.allclose(expected_slopes_T1, t1_slope_values)

        # T2 log after zero-th run --------------------------------------------

        expected_times_T2 = ['0:0:0:1', '0:1:0:1', '0:2:0:1']
        expected_values_T2 = [[[2.0, 4.0]], [[6.0, 8.0]], [[10.0, 12.0]]]
        expected_slopes_T2 = [[2.0], [2.0], [2.0]]
        expected_results_T2 = [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]

        assert expected_times_T2 == t2_result_times == t2_slope_times == t2_value_times
        assert np.allclose(expected_values_T2, t2_value_values)
        assert np.allclose(expected_results_T2, t2_result_values)
        assert np.allclose(expected_slopes_T2, t2_slope_values)

        # RUN ONE  |  TRIALS ZERO, ONE, TWO -------------------------------------

        COMP.run(inputs={T1: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]},
                 context=con_with_rpc_pipeline)

        actual = []
        while not pipeline.empty():
            actual.append(pipeline.get())

        t1_slope_entries = [i for i in actual if i.parameterName == pnl.SLOPE and i.componentName == 'log_test_T1']
        t1_slope_values = [np.ndarray(shape=np.array(i.value.shape), buffer=np.array(i.value.data)) for i in t1_slope_entries]
        t1_slope_times = [i.time for i in t1_slope_entries]

        t2_slope_entries = [i for i in actual if i.parameterName == pnl.SLOPE and i.componentName == 'log_test_T2']
        t2_slope_values = [np.ndarray(shape=np.array(i.value.shape), buffer=np.array(i.value.data)) for i in t2_slope_entries]
        t2_slope_times = [i.time for i in t2_slope_entries]

        t1_result_entries = [i for i in actual if i.parameterName == pnl.RESULT and i.componentName == 'log_test_T1']
        t1_result_values = [np.ndarray(shape=np.array(i.value.shape), buffer=np.array(i.value.data)) for i in t1_result_entries]
        t1_result_times = [i.time for i in t1_result_entries]

        t2_result_entries = [i for i in actual if i.parameterName == pnl.RESULT and i.componentName == 'log_test_T2']
        t2_result_values = [np.ndarray(shape=np.array(i.value.shape), buffer=np.array(i.value.data)) for i in t2_result_entries]
        t2_result_times = [i.time for i in t2_result_entries]

        t1_value_entries = [i for i in actual if i.parameterName == pnl.VALUE and i.componentName == 'log_test_T1']
        t1_value_values = [np.ndarray(shape=np.array(i.value.shape), buffer=np.array(i.value.data)) for i in t1_value_entries]
        t1_value_times = [i.time for i in t1_value_entries]

        t2_value_entries = [i for i in actual if i.parameterName == pnl.VALUE and i.componentName == 'log_test_T2']
        t2_value_values = [np.ndarray(shape=np.array(i.value.shape), buffer=np.array(i.value.data)) for i in t2_value_entries]
        t2_value_times = [i.time for i in t2_value_entries]

        # T1 log after first run -------------------------------------------

        # Test execution contexts for all entries

        assert all([True if i.context == COMP.default_execution_id else False for i in actual])

        expected_times_T1 = ['1:0:0:0', '1:1:0:0', '1:2:0:0']
        expected_values_T1 = [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]
        expected_slopes_T1 = [[1.0], [1.0], [1.0]]
        expected_results_T1 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

        assert expected_times_T1 == t1_result_times == t1_slope_times == t1_value_times
        assert np.allclose(expected_values_T1, t1_value_values)
        assert np.allclose(expected_results_T1, t1_result_values)
        assert np.allclose(expected_slopes_T1, t1_slope_values)

        # T2 log after first run -------------------------------------------

        expected_times_T2 = ['1:0:0:1', '1:1:0:1', '1:2:0:1']
        expected_values_T2 = [[[2.0, 4.0]], [[6.0, 8.0]], [[10.0, 12.0]]]
        expected_slopes_T2 = [[2.0], [2.0], [2.0]]
        expected_results_T2 = [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]

        assert expected_times_T2 == t2_result_times == t2_slope_times == t2_value_times
        assert np.allclose(expected_values_T2, t2_value_values)
        assert np.allclose(expected_results_T2, t2_result_values)
        assert np.allclose(expected_slopes_T2, t2_slope_values)

    def test_log_dictionary_with_scheduler(self):
        T1 = pnl.TransferMechanism(name='log_test_T1',
                                   integrator_mode=True,
                                   integration_rate=0.5)
        T2 = pnl.TransferMechanism(name='log_test_T2',
                                   function=pnl.Linear(slope=6.0))
        COMP = pnl.Composition(name='log_test_COMP', pathways=[T1, T2])
        con_with_rpc_pipeline = pnl.Context(rpc_pipeline=Queue(), execution_id=COMP)
        pipeline = con_with_rpc_pipeline.rpc_pipeline

        def pass_threshold(mech, thresh):
            results = mech.output_ports[0].parameters.value.get(COMP)
            for val in results:
                if abs(val) >= thresh:
                    return True
            return False

        terminate_trial = {
            pnl.TimeScale.TRIAL: pnl.While(pass_threshold, T2, 5.0)
        }

        T1.set_delivery_conditions(pnl.VALUE)
        T1.set_delivery_conditions('mod_slope')
        T1.set_delivery_conditions(pnl.RESULT)
        T2.set_delivery_conditions(pnl.VALUE)
        T2.set_delivery_conditions('mod_slope')

        COMP.run(inputs={T1: [[1.0]]}, termination_processing=terminate_trial,
                 context=con_with_rpc_pipeline)

        actual = []
        while not pipeline.empty():
            actual.append(pipeline.get())
        assert all([True if i.context == COMP.default_execution_id else False for i in actual])

        t1_slope_entries = [i for i in actual if i.parameterName == pnl.SLOPE and i.componentName == 'log_test_T1']
        t1_slope_values = [np.ndarray(shape=np.array(i.value.shape), buffer=np.array(i.value.data)) for i in t1_slope_entries]
        t1_slope_times = [i.time for i in t1_slope_entries]

        t2_slope_entries = [i for i in actual if i.parameterName == pnl.SLOPE and i.componentName == 'log_test_T2']
        t2_slope_values = [np.ndarray(shape=np.array(i.value.shape), buffer=np.array(i.value.data)) for i in t2_slope_entries]
        t2_slope_times = [i.time for i in t2_slope_entries]

        t1_result_entries = [i for i in actual if i.parameterName == pnl.RESULT and i.componentName == 'log_test_T1']
        t1_result_values = [np.ndarray(shape=np.array(i.value.shape), buffer=np.array(i.value.data)) for i in t1_result_entries]
        t1_result_times = [i.time for i in t1_result_entries]

        t1_value_entries = [i for i in actual if i.parameterName == pnl.VALUE and i.componentName == 'log_test_T1']
        t1_value_values = [np.ndarray(shape=np.array(i.value.shape), buffer=np.array(i.value.data)) for i in t1_value_entries]
        t1_value_times = [i.time for i in t1_value_entries]

        t2_value_entries = [i for i in actual if i.parameterName == pnl.VALUE and i.componentName == 'log_test_T2']
        t2_value_values = [np.ndarray(shape=np.array(i.value.shape), buffer=np.array(i.value.data)) for i in t2_value_entries]
        t2_value_times = [i.time for i in t2_value_entries]

        # Check values T1

        expected_times_T1 = ['0:0:0:0', '0:0:1:0', '0:0:2:0']
        expected_results_T1 = [[0.5], [0.75], [0.875]]
        expected_values_T1 = [[[0.5]], [[0.75]], [[0.875]]]
        expected_slopes_T1 = [[1], [1], [1]]
        assert expected_times_T1 == t1_result_times == t1_slope_times == t1_value_times
        assert np.allclose(expected_values_T1, t1_value_values)
        assert np.allclose(expected_results_T1, t1_result_values)
        assert np.allclose(expected_slopes_T1, t1_slope_values)

        # Check values T2

        expected_times_T2 = ['0:0:0:1', '0:0:1:1', '0:0:2:1']
        expected_values_T2 = [[[3]], [[4.5]], [[5.25]]]
        expected_slopes_T2 = [[6], [6], [6]]
        assert expected_times_T2 == t2_slope_times == t2_value_times
        assert np.allclose(expected_values_T2, t2_value_values)
        assert np.allclose(expected_slopes_T2, t2_slope_values)

    def test_log_dictionary_with_scheduler_many_time_step_increments(self):
        T1 = pnl.TransferMechanism(name='log_test_T1',
                                   integrator_mode=True,
                                   integration_rate=0.05)
        COMP = pnl.Composition(name='log_test_COMP', pathways=[T1])
        con_with_rpc_pipeline = pnl.Context(rpc_pipeline=Queue(), execution_id=COMP)
        pipeline = con_with_rpc_pipeline.rpc_pipeline

        def pass_threshold(mech, thresh):
            results = mech.output_ports[0].parameters.value.get(COMP)
            for val in results:
                if abs(val) >= thresh:
                    return True
            return False

        terminate_trial = {
            pnl.TimeScale.TRIAL: pnl.While(pass_threshold, T1, 0.95)
        }

        T1.set_delivery_conditions(pnl.VALUE)

        COMP.run(inputs={T1: [[1.0]]}, termination_processing=terminate_trial, context=con_with_rpc_pipeline)

        actual = []
        while not pipeline.empty():
            actual.append(pipeline.get())
        assert all([True if i.context == COMP.default_execution_id else False for i in actual])

        t1_value_entries = [i for i in actual if i.parameterName == pnl.VALUE and i.componentName == 'log_test_T1']
        t1_value_values = [np.ndarray(shape=np.array(i.value.shape), buffer=np.array(i.value.data)) for i in
                           t1_value_entries]

        # Check values T1
        assert len(actual) == 59
        assert actual[30].time == '0:0:30:0'
        assert t1_value_values[58] >= 0.95
        assert t1_value_values[57] < 0.95

    def test_log_csv_multiple_contexts(self):
        pipeline = Queue()
        con_X = pnl.Context(execution_id='comp X', rpc_pipeline=pipeline)
        con_Y = pnl.Context(execution_id='comp Y', rpc_pipeline=pipeline)

        A = pnl.TransferMechanism(name='A')
        B = pnl.TransferMechanism(name='B')
        C = pnl.TransferMechanism(name='C')

        C.set_delivery_conditions(pnl.VALUE)

        X = pnl.Composition(name='comp X')
        Y = pnl.Composition(name='comp Y')

        X.add_linear_processing_pathway([A, C])
        Y.add_linear_processing_pathway([B, C])

        # running with manual contexts for consistent output
        # because output is sorted by context
        X.run(inputs={A: 1}, context=con_X)
        Y.run(inputs={B: 2}, context=con_Y)

        actual = []
        while not pipeline.empty():
            actual.append(pipeline.get())

        assert actual[0].context == 'comp X'
        assert actual[0].time == '0:0:0:1'
        assert actual[0].value.data == [1]
        assert actual[1].context == 'comp Y'
        assert actual[1].time == '0:0:0:1'
        assert actual[1].value.data == [2]

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
        lca.set_delivery_conditions(pnl.VALUE)
        m0 = pnl.ProcessingMechanism(
            size=2
        )
        comp = pnl.Composition()
        comp.add_linear_processing_pathway([m0, lca])
        if scheduler_conditions:
            comp.scheduler.add_condition(lca, pnl.AfterNCalls(m0, 2))
        con_with_rpc_pipeline = pnl.Context(rpc_pipeline=Queue(), execution_id=comp)
        pipeline = con_with_rpc_pipeline.rpc_pipeline
        comp.run(inputs={m0: [[1, 0], [1, 0], [1, 0]]}, context=con_with_rpc_pipeline)

        actual = []
        while not pipeline.empty():
            actual.append(pipeline.get())
        integration_end_dict = {i.time: i for i in actual}
        if scheduler_conditions:
            expected_times = ['0:0:1:1', '0:1:1:1', '0:2:1:1']
        else:
            expected_times = ['0:0:0:1', '0:1:0:1', '0:2:0:1']
        assert list(integration_end_dict.keys()) == expected_times
        vals = [i.value.data for i in integration_end_dict.values()]
        # floats in value, so use np.allclose
        assert np.allclose(vals, [[[0.52466739, 0.47533261]] * 3])
        if multi_run:
            comp.run(inputs={m0: [[1, 0], [1, 0], [1, 0]]}, context=con_with_rpc_pipeline)
            actual = []
            while not pipeline.empty():
                actual.append(pipeline.get())
            integration_end_dict.update({i.time: i for i in actual})
            if scheduler_conditions:
                expected_times = ['0:0:1:1', '0:1:1:1', '0:2:1:1', '1:0:1:1', '1:1:1:1', '1:2:1:1']
            else:
                expected_times = ['0:0:0:1', '0:1:0:1', '0:2:0:1', '1:0:0:1', '1:1:0:1', '1:2:0:1']
            assert list(integration_end_dict.keys()) == expected_times
            vals = [i.value.data for i in integration_end_dict.values()]
            # floats in value, so use np.allclose
            assert np.allclose(vals, [[[0.52466739, 0.47533261]] * 6])

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

        middle_weights.set_delivery_conditions(('mod_matrix', pnl.PROCESSING))
        con_with_rpc_pipeline = pnl.Context(rpc_pipeline=Queue(), execution_id=comp)
        pipeline = con_with_rpc_pipeline.rpc_pipeline

        comp.learn(inputs=input_dictionary,
                   num_trials=10,
                   context=con_with_rpc_pipeline)

        expected_log_val = np.array(
            [
                ['multilayer'],
                [[
                    [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
                    [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
                    [[2], [2], [2], [2], [2], [2], [2], [2], [2], [2]],
                    [[[0.05, 0.1, 0.15, 0.2],
                      [0.25, 0.3, 0.35, 0.4],
                      [0.45, 0.5, 0.55, 0.6],
                      [0.65, 0.7, 0.75, 0.8],
                      [0.85, 0.9, 0.95, 1.]],
                     [[0.04789907, 0.09413833, 0.14134241, 0.18938924],
                      [0.24780811, 0.29388455, 0.34096758, 0.38892985],
                      [0.44772121, 0.49364209, 0.54060947, 0.58849095],
                      [0.64763875, 0.69341202, 0.74026967, 0.78807449],
                      [0.84756101, 0.89319513, 0.93994932, 0.98768187]],
                     [[0.04738148, 0.08891106, 0.13248753, 0.177898],
                      [0.24726841, 0.28843403, 0.33173452, 0.37694783],
                      [0.44716034, 0.48797777, 0.53101423, 0.57603893],
                      [0.64705774, 0.6875443, 0.73032986, 0.77517531],
                      [0.84696096, 0.88713512, 0.92968378, 0.97435998]],
                     [[0.04937771, 0.08530344, 0.12439361, 0.16640433],
                      [0.24934878, 0.28467436, 0.32329947, 0.36496974],
                      [0.44932147, 0.48407216, 0.52225175, 0.56359587],
                      [0.64929589, 0.68349948, 0.72125508, 0.76228876],
                      [0.84927212, 0.88295836, 0.92031297, 0.96105307]],
                     [[0.05440291, 0.08430585, 0.1183739, 0.15641064],
                      [0.25458348, 0.28363519, 0.3170288, 0.35455942],
                      [0.45475764, 0.48299299, 0.51573974, 0.55278488],
                      [0.65492462, 0.68238209, 0.7145124, 0.75109483],
                      [0.85508376, 0.88180465, 0.91335119, 0.94949538]],
                     [[0.06177218, 0.0860581, 0.11525064, 0.14926369],
                      [0.26225812, 0.28546004, 0.31377611, 0.34711631],
                      [0.46272625, 0.48488774, 0.51236246, 0.54505667],
                      [0.66317453, 0.68434373, 0.7110159, 0.74309381],
                      [0.86360121, 0.88382991, 0.9097413, 0.94123489]],
                     [[0.06989398, 0.08959148, 0.11465594, 0.14513241],
                      [0.27071639, 0.2891398, 0.31315677, 0.34281389],
                      [0.47150846, 0.48870843, 0.5117194, 0.54058946],
                      [0.67226675, 0.68829929, 0.71035014, 0.73846891],
                      [0.87298831, 0.88791376, 0.90905395, 0.93646]],
                     [[0.07750784, 0.09371987, 0.11555569, 0.143181],
                      [0.27864693, 0.29343991, 0.31409396, 0.3407813],
                      [0.47974374, 0.49317377, 0.5126926, 0.53847878],
                      [0.68079346, 0.69292265, 0.71135777, 0.73628353],
                      [0.88179203, 0.89268732, 0.91009431, 0.93420362]],
                     [[0.0841765, 0.09776672, 0.11711835, 0.14249779],
                      [0.28559463, 0.29765609, 0.31572199, 0.34006951],
                      [0.48695967, 0.49755273, 0.51438349, 0.5377395],
                      [0.68826567, 0.69745713, 0.71310872, 0.735518],
                      [0.88950757, 0.89736946, 0.91190228, 0.93341316]],
                     [[0.08992499, 0.10150104, 0.11891032, 0.14250149],
                      [0.29158517, 0.30154765, 0.31758943, 0.34007336],
                      [0.49318268, 0.50159531, 0.51632339, 0.5377435],
                      [0.69471052, 0.70164382, 0.71511777, 0.73552215],
                      [0.8961628, 0.90169281, 0.91397691, 0.93341744]]]
                ]]
            ],
            dtype=object
        )
        actual = []
        while not pipeline.empty():
            actual.append(pipeline.get())
        log_val = [np.ndarray(shape=np.array(i.value.shape), buffer=np.array(i.value.data)) for i in actual]
        assert all([True if i.context == 'multilayer' else False for i in actual])
        assert np.allclose(log_val, expected_log_val[1][0][4])

        # Test Programatic logging
        hidden_layer_2.log._deliver_values(pnl.VALUE, con_with_rpc_pipeline)
        actual = []
        while not pipeline.empty():
            actual.append(pipeline.get())
        log_val = np.ndarray(shape=np.array(actual[0].value.shape), buffer=np.array(actual[0].value.data))
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
        assert actual[0].context == 'multilayer'
        assert actual[0].time == '1:0:0:0'
        assert np.allclose(
            expected_log_val[1][0][4],
            log_val
        )

        # Clear log and test with logging of weights set to LEARNING for another 5 trials of learning
        middle_weights.set_delivery_conditions(('mod_matrix', pnl.LEARNING))
        comp.learn(
            num_trials=5,
            inputs=input_dictionary,
            context=con_with_rpc_pipeline
        )
        actual = []
        while not pipeline.empty():
            actual.append(pipeline.get())
        assert all([True if i.context == 'multilayer' else False for i in actual])
        matrices = [i for i in actual if i.parameterName == 'matrix']
        log_val = [np.ndarray(shape=np.array(i.value.shape), buffer=np.array(i.value.data)) for i in matrices]
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
                    [[[0.09925812411381937, 0.1079522130303428, 0.12252820028789306, 0.14345816973727732],
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

        assert [i.time for i in matrices] == ['1:0:1:0', '1:1:1:0', '1:2:1:0', '1:3:1:0', '1:4:1:0']
        assert np.allclose(
            expected_log_val[1][0][4],
            log_val
        )
