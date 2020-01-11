import numpy as np

from psyneulink.core.components.functions.distributionfunctions import DriftDiffusionAnalytical
from psyneulink.core.components.functions.transferfunctions import Linear, Logistic
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.process import Process
from psyneulink.core.components.projections.modulatory.controlprojection import ControlProjection
from psyneulink.core.components.system import System
from psyneulink.core.globals.keywords import ALLOCATION_SAMPLES
from psyneulink.core.globals.keywords import CYCLE, INITIALIZE_CYCLE, INTERNAL, ORIGIN, TERMINAL
from psyneulink.core.scheduling.condition import AfterTrial, Any, AtTrial
from psyneulink.library.components.mechanisms.modulatory.control.evc.evccontrolmechanism import EVCControlMechanism
from psyneulink.library.components.mechanisms.processing.integrator.ddm import DDM
from psyneulink.library.components.mechanisms.processing.transfer.recurrenttransfermechanism import RecurrentTransferMechanism


class TestInputSpecsExternalInputPortsOnly:

    def test_recurrent_transfer_origin(self):
        R = RecurrentTransferMechanism(has_recurrent_input_port=True)
        P = Process(pathway=[R])
        S = System(processes=[P])

        S.run(inputs={R: [[1.0], [2.0], [3.0]]})
        print(S.results)

class TestInputSpecsHeterogeneousVariables:

    def test_heterogeneous_variables_drop_outer_list(self):
        # from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
        a = TransferMechanism(name='a', default_variable=[[0.0], [0.0,0.0]])

        p1 = Process(pathway=[a])

        s = System(
            processes=[p1]
        )

        inputs = {a: [[1.0], [2.0, 2.0]]}

        s.run(inputs)

    def test_heterogeneous_variables(self):
        # from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
        a = TransferMechanism(name='a', default_variable=[[0.0], [0.0,0.0]])

        p1 = Process(pathway=[a])

        s = System(
            processes=[p1]
        )

        inputs = {a: [[[1.1], [2.1, 2.1]], [[1.2], [2.2, 2.2]]]}

        s.run(inputs)

class TestGraphAndInput:

    def test_input_not_provided_to_run(self):
        T = TransferMechanism(name='T',
                              default_variable=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        T2 = TransferMechanism(name='T2',
                               function=Linear(slope=2.0),
                               default_variable=[[0.0, 0.0]])
        P = Process(pathway=[T, T2])
        S = System(processes=[P])
        run_result = S.run()

        assert np.allclose(T.parameters.value.get(S), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert np.allclose(run_result, [[np.array([2.0, 4.0])]])

    def test_some_inputs_not_provided_to_run(self):
        Origin1 = TransferMechanism(name='Origin1',
                                    default_variable=[[1.0, 2.0]])
        Origin2 = TransferMechanism(name='Origin2',
                                    default_variable=[[3.0, 4.0]])
        Terminal = TransferMechanism(name='Terminal')

        P1 = Process(pathway=[Origin1, Terminal])
        P2 = Process(pathway=[Origin2, Terminal])
        S = System(processes=[P1, P2])
        run_result = S.run(inputs={Origin1: [[5.0, 6.0]]})
        # inputs={Origin1: [[5.0, 6.0], [7.0, 8.0]]}) # NOT currently allowed because inputs would be different lengths

        assert np.allclose(Origin1.parameters.value.get(S), [[5.0, 6.0]])
        assert np.allclose(Origin2.parameters.value.get(S), [[3.0, 4.0]])
        assert np.allclose(run_result, [[np.array([18.0])]])

    def test_branch(self):
        a = TransferMechanism(name='a', default_variable=[0, 0])
        b = TransferMechanism(name='b')
        c = TransferMechanism(name='c')
        d = TransferMechanism(name='d')

        p1 = Process(pathway=[a, b, c], name='p1')
        p2 = Process(pathway=[a, b, d], name='p2')

        s = System(
            processes=[p1, p2],
            name='Branch System',
            initial_values={a: [1, 1]},
        )

        inputs = {a: [[2, 2]]}
        s.run(inputs)

        assert [a] == s.origin_mechanisms.mechanisms
        assert set([c, d]) == set(s.terminal_mechanisms.mechanisms)

        assert a.systems[s] == ORIGIN
        assert b.systems[s] == INTERNAL
        assert c.systems[s] == TERMINAL
        assert d.systems[s] == TERMINAL

    def test_bypass(self):
        a = TransferMechanism(name='a', default_variable=[0, 0])
        b = TransferMechanism(name='b', default_variable=[0, 0])
        c = TransferMechanism(name='c')
        d = TransferMechanism(name='d')

        p1 = Process(pathway=[a, b, c, d], name='p1')
        p2 = Process(pathway=[a, b, d], name='p2')

        s = System(
            processes=[p1, p2],
            name='Bypass System',
            initial_values={a: [1, 1]},
        )

        inputs = {a: [[2, 2], [0, 0]]}
        s.run(inputs=inputs)

        assert [a] == s.origin_mechanisms.mechanisms
        assert [d] == s.terminal_mechanisms.mechanisms

        assert a.systems[s] == ORIGIN
        assert b.systems[s] == INTERNAL
        assert c.systems[s] == INTERNAL
        assert d.systems[s] == TERMINAL

    def test_chain(self):
        a = TransferMechanism(name='a', default_variable=[0, 0, 0])
        b = TransferMechanism(name='b')
        c = TransferMechanism(name='c')
        d = TransferMechanism(name='d')
        e = TransferMechanism(name='e')

        p1 = Process(pathway=[a, b, c], name='p1')
        p2 = Process(pathway=[c, d, e], name='p2')

        s = System(
            processes=[p1, p2],
            name='Chain System',
            initial_values={a: [1, 1, 1]},
        )

        inputs = {a: [[2, 2, 2], [0, 0, 0]]}
        s.run(inputs=inputs)

        assert [a] == s.origin_mechanisms.mechanisms
        assert [e] == s.terminal_mechanisms.mechanisms

        assert a.systems[s] == ORIGIN
        assert b.systems[s] == INTERNAL
        assert c.systems[s] == INTERNAL
        assert d.systems[s] == INTERNAL
        assert e.systems[s] == TERMINAL

    def test_convergent(self):
        a = TransferMechanism(name='a', default_variable=[0, 0])
        b = TransferMechanism(name='b')
        c = TransferMechanism(name='c')
        c = TransferMechanism(name='c', default_variable=[0])
        d = TransferMechanism(name='d')
        e = TransferMechanism(name='e')

        p1 = Process(pathway=[a, b, e], name='p1')
        p2 = Process(pathway=[c, d, e], name='p2')

        s = System(
            processes=[p1, p2],
            name='Convergent System',
            initial_values={a: [1, 1]},
        )

        inputs = {a: [[2, 2]], c: [[0]]}
        s.run(inputs=inputs)

        assert set([a, c]) == set(s.origin_mechanisms.mechanisms)
        assert [e] == s.terminal_mechanisms.mechanisms

        assert a.systems[s] == ORIGIN
        assert b.systems[s] == INTERNAL
        assert c.systems[s] == ORIGIN
        assert d.systems[s] == INTERNAL
        assert e.systems[s] == TERMINAL

    def cyclic_one_process(self):
        a = TransferMechanism(name='a', default_variable=[0, 0])
        b = TransferMechanism(name='b', default_variable=[0, 0])

        p1 = Process(pathway=[a, b, a], name='p1')

        s = System(
            processes=[p1],
            name='Cyclic System with one Process',
            initial_values={a: [1, 1]},
        )

        inputs = {a: [1, 1]}
        s.run(inputs=inputs)

        assert [a] == s.origin_mechanisms.mechanisms
        assert [] == s.terminal_mechanisms.mechanisms

        assert a.systems[s] == ORIGIN
        assert b.systems[s] == INITIALIZE_CYCLE

    def cyclic_two_processes(self):
        a = TransferMechanism(name='a', default_variable=[0, 0])
        b = TransferMechanism(name='b', default_variable=[0, 0])
        c = TransferMechanism(name='c', default_variable=[0, 0])

        p1 = Process(pathway=[a, b, a], name='p1')
        p2 = Process(pathway=[a, c, a], name='p2')

        s = System(
            processes=[p1, p2],
            name='Cyclic System with two Processes',
            initial_values={a: [1, 1]},
        )

        inputs = {a: [1, 1]}
        s.run(inputs=inputs)

        assert [a] == s.origin_mechanisms.mechanisms
        assert [] == s.terminal_mechanisms.mechanisms

        assert a.systems[s] == ORIGIN
        assert b.systems[s] == INITIALIZE_CYCLE
        assert c.systems[s] == INITIALIZE_CYCLE

    def cyclic_extended_loop(self):
        a = TransferMechanism(name='a', default_variable=[0, 0])
        b = TransferMechanism(name='b')
        c = TransferMechanism(name='c')
        d = TransferMechanism(name='d')
        e = TransferMechanism(name='e', default_variable=[0])
        f = TransferMechanism(name='f')

        p1 = Process(pathway=[a, b, c, d], name='p1')
        p2 = Process(pathway=[e, c, f, b, d], name='p2')

        s = System(
            processes=[p1, p2],
            name='Cyclic System with Extended Loop',
            initial_values={a: [1, 1]},
        )

        inputs = {a: [2, 2], e: [0]}
        s.run(inputs=inputs)

        assert set([a, c]) == set(s.origin_mechanisms.mechanisms)
        assert [d] == s.terminal_mechanisms.mechanisms

        assert a.systems[s] == ORIGIN
        assert b.systems[s] == CYCLE
        assert c.systems[s] == INTERNAL
        assert d.systems[s] == TERMINAL
        assert e.systems[s] == ORIGIN
        assert f.systems[s] == INITIALIZE_CYCLE

class TestConvergentLearning:

    def test_branch(self):
        from psyneulink.core.globals.keywords import ENABLED
        mech_1 = TransferMechanism(name='M1', size=1)
        mech_2 = TransferMechanism(name='M2', size=2)
        mech_3 = TransferMechanism(name='M3', size=3)
        mech_4 = TransferMechanism(name='M4', size=4)
        mech_5 = TransferMechanism(name='M5', size=5)
        mech_6 = TransferMechanism(name='M6', size=6)
        process_A = Process(pathway=[mech_1, mech_2, mech_3, mech_4], learning=ENABLED, name='Process A')
        process_B = Process(pathway=[mech_5, mech_6, mech_4], learning=ENABLED, name='Process B')
        S = System(processes=[process_A, process_B])

        lm = mech_1.efferents[0].learning_mechanism
        assert 'LearningMechanism for MappingProjection from M2 to M3' in [m.name for m in S.learningGraph[lm]]
        lm = mech_2.efferents[0].learning_mechanism
        assert 'LearningMechanism for MappingProjection from M3 to M4' in [m.name for m in S.learningGraph[lm]]
        lm = mech_3.efferents[0].learning_mechanism
        assert 'M4 ComparatorMechanism' in [m.name for m in S.learningGraph[lm]]
        cm = mech_4.efferents[0].receiver.owner
        assert cm in S.learningGraph.keys()
        lm = mech_5.efferents[0].learning_mechanism
        assert 'LearningMechanism for MappingProjection from M6 to M4' in [m.name for m in S.learningGraph[lm]]
        lm = mech_6.efferents[0].learning_mechanism
        assert 'M4 ComparatorMechanism' in [m.name for m in S.learningGraph[lm]]


class TestInitialize:

    def test_initialize_mechanisms(self):
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B')
        C = RecurrentTransferMechanism(name='C',
                                       auto=1.0)

        abc_process = Process(pathway=[A, B, C])

        abc_system = System(processes=[abc_process])

        C.log.set_log_conditions('value')

        abc_system.run(inputs={A: [1.0, 2.0, 3.0]},
                       initial_values={A: 1.0,
                                       B: 1.5,
                                       C: 2.0},
                       initialize=True)

        abc_system.run(inputs={A: [1.0, 2.0, 3.0]},
                       initial_values={A: 1.0,
                                       B: 1.5,
                                       C: 2.0},
                       initialize=False)

        # Run 1 --> Execution 1: 1 + 2 = 3    |    Execution 2: 3 + 2 = 5    |    Execution 3: 5 + 3 = 8
        # Run 2 --> Execution 1: 8 + 1 = 9    |    Execution 2: 9 + 2 = 11    |    Execution 3: 11 + 3 = 14
        assert np.allclose(
            C.log.nparray_dictionary('value')[abc_system.default_execution_id]['value'],
            [[[3]], [[5]], [[8]], [[9]], [[11]], [[14]]]
        )


class TestRuntimeParams:

    def test_mechanism_execute_function_param(self):

        # Construction
        T = TransferMechanism()
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0

        # Runtime param used for slope
        T.execute(runtime_params={"slope": 10.0}, input=2.0)
        assert T.function.slope == 10.0
        assert T.parameter_ports['slope'].value == 10.0
        assert T.value == 20.0

        # Runtime param NOT used for slope
        T.execute(input=2.0)
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0
        assert T.value == 2.0

    def test_mechanism_execute_mechanism_param(self):

        # Construction
        T = TransferMechanism()
        assert T.noise == 0.0
        assert T.parameter_ports['noise'].value == 0.0

        # Runtime param used for noise
        T.execute(runtime_params={"noise": 10.0}, input=2.0)
        assert T.noise == 10.0
        assert T.parameter_ports['noise'].value == 10.0
        assert T.value == 12.0

        # Runtime param NOT used for noise
        T.execute(input=2.0)
        assert T.noise == 0.0
        assert T.parameter_ports['noise'].value == 0.0
        assert T.value == 2.0

    def test_runtime_params_reset_isolated(self):

        T = TransferMechanism()

        # Intercept attr updated
        T.function.intercept = 2.0
        assert T.function.intercept == 2.0

        # Runtime param used for slope
        T.execute(runtime_params={"slope": 10.0}, input=2.0)
        assert T.function.slope == 10.0
        assert T.parameter_ports['slope'].value == 10.0

        # Intercept attr NOT affected by runtime params
        assert T.function.intercept == 2.0
        assert T.value == 22.0

        # Runtime param NOT used for slope
        T.execute(input=2.0)
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0

        # Intercept attr NOT affected by runtime params reset
        assert T.function.intercept == 2.0
        assert T.value == 4.0

    def test_runtime_params_reset_to_most_recent_val(self):
        # NOT instance defaults

        # Construction
        T = TransferMechanism()
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0

        # Set slope attribute value directly
        T.function.slope = 2.0
        assert T.function.slope == 2.0

        # Runtime param used for slope
        T.execute(runtime_params={"slope": 10.0}, input=2.0)
        assert T.function.slope == 10.0
        assert T.parameter_ports['slope'].value == 10.0
        assert T.value == 20.0

        # Runtime param NOT used for slope - reset to most recent slope value (2.0)
        T.execute(input=2.0)
        assert T.function.slope == 2.0
        assert T.value == 4.0

    def test_system_run_function_param_no_condition(self):

        # Construction
        T = TransferMechanism()
        P = Process(pathway=[T])
        S = System(processes=[P])
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0

        # Runtime param used for slope
        # ONLY mechanism value should reflect runtime param -- attr should be changed back by the time we inspect it
        S.run(inputs={T: 2.0}, runtime_params={T: {"slope": 10.0}})
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0
        assert T.parameters.value.get(S) == 20.0

        # Runtime param NOT used for slope
        S.run(inputs={T: 2.0})
        assert T.function.slope == 1.0
        assert T.parameter_ports['slope'].value == 1.0
        assert T.parameters.value.get(S) == 2.0

    def test_system_run_mechanism_param_no_condition(self):

        # Construction
        T = TransferMechanism()
        P = Process(pathway=[T])
        S = System(processes=[P])
        assert T.noise == 0.0
        assert T.parameter_ports['noise'].value == 0.0

        # Runtime param used for noise
        # ONLY mechanism value should reflect runtime param -- attr should be changed back by the time we inspect it
        S.run(inputs={T: 2.0}, runtime_params={T: {"noise": 10.0}})
        assert T.noise == 0.0
        assert T.parameter_ports['noise'].value == 0.0
        assert T.parameters.value.get(S) == 12.0

        # Runtime param NOT used for noise
        S.run(inputs={T: 2.0}, )
        assert T.noise == 0.0
        assert T.parameter_ports['noise'].value == 0.0
        assert T.parameters.value.get(S) == 2.0

    def test_system_run_with_condition(self):

        # Construction
        T = TransferMechanism()
        P = Process(pathway=[T])
        S = System(processes=[P])

        # Runtime param used for noise
        # ONLY mechanism value should reflect runtime param -- attr should be changed back by the time we inspect it
        S.run(inputs={T: 2.0},
              runtime_params={T: {"noise": (10.0, AtTrial(1))}},
              num_trials=4)

        # Runtime param NOT used for noise
        S.run(inputs={T: 2.0})

        assert np.allclose(S.results, [[np.array([2.])],     # Trial 0 - condition not satisfied yet
                                       [np.array([12.])],    # Trial 1 - condition satisfied
                                       [np.array([2.])],     # Trial 2 - condition no longer satisfied (not sticky)
                                       [np.array([2.])],     # Trial 3 - condition no longer satisfied (not sticky)
                                       [np.array([2.])]])    # New run (runtime param no longer applies)

    def test_system_run_with_sticky_condition(self):

        # Construction
        T = TransferMechanism()
        P = Process(pathway=[T])
        S = System(processes=[P])
        assert T.noise == 0.0
        assert T.parameter_ports['noise'].value == 0.0

        # Runtime param used for noise
        # ONLY mechanism value should reflect runtime param -- attr should be changed back by the time we inspect it
        S.run(inputs={T: 2.0},
              runtime_params={T: {"noise": (10.0, AfterTrial(1))}},
              num_trials=4)

        # Runtime param NOT used for noise
        S.run(inputs={T: 2.0})

        assert np.allclose(S.results, [[np.array([2.])],      # Trial 0 - condition not satisfied yet
                                       [np.array([2.])],      # Trial 1 - condition not satisfied yet
                                       [np.array([12.])],     # Trial 2 - condition satisfied
                                       [np.array([12.])],     # Trial 3 - condition satisfied (sticky)
                                       [np.array([2.])]])     # New run (runtime param no longer applies)

    def test_system_run_with_combined_condition(self):

        # Construction
        T = TransferMechanism()
        P = Process(pathway=[T])
        S = System(processes=[P])

        # Runtime param used for noise
        # ONLY mechanism value should reflect runtime param -- attr should be changed back by the time we inspect it
        S.run(inputs={T: 2.0},
              runtime_params={T: {"noise": (10.0, Any(AtTrial(1), AfterTrial(2)))}},
              num_trials=5)

        # Runtime param NOT used for noise
        S.run(inputs={T: 2.0})

        assert np.allclose(S.results, [[np.array([2.])],      # Trial 0 - NOT condition 0, NOT condition 1
                                       [np.array([12.])],     # Trial 1 - condition 0, NOT condition 1
                                       [np.array([2.])],      # Trial 2 - NOT condition 0, NOT condition 1
                                       [np.array([12.])],     # Trial 3 - NOT condition 0, condition 1
                                       [np.array([12.])],     # Trial 4 - NOT condition 0, condition 1
                                       [np.array([2.])]])     # New run (runtime param no longer applies)

from psyneulink.core.components.process import proc
from psyneulink.core.components.system import sys
class TestFactoryMethods:

    def test_process_factory_method(self):

        T1 = TransferMechanism()
        T2 = TransferMechanism()
        T3 = TransferMechanism()
        p = proc(T1, T2, T3, learning_rate = 0.55)
        assert p.learning_rate == 0.55
        assert T1 in p.origin_mechanisms
        assert not T2 in p.origin_mechanisms
        assert T3 in p.terminal_mechanisms

    def test_system_factory_method_solo_mechs(self):

        T1 = TransferMechanism()
        T2 = TransferMechanism()
        T3 = TransferMechanism()
        s = sys(T1, T2, T3, learning_rate = 0.65)
        assert s.learning_rate == 0.65
        assert T1 in s.origin_mechanisms
        assert not T2 in s.origin_mechanisms
        assert T3 in s.terminal_mechanisms


    def test_system_factory_method_solo_mech_and_list(self):

        T1 = TransferMechanism()
        T2 = TransferMechanism()
        T3 = TransferMechanism()
        s = sys(T1, [T2, T3], learning_rate = 0.75)
        assert s.learning_rate == 0.75
        assert T1 in s.origin_mechanisms
        assert T2 in s.origin_mechanisms
        assert T3 in s.terminal_mechanisms

    def test_system_factory_method_mech_list(self):

        T1 = TransferMechanism()
        T2 = TransferMechanism()
        T3 = TransferMechanism()
        s = sys([T1, T2, T3], learning_rate = 0.85)
        assert s.learning_rate == 0.85
        assert T1 in s.origin_mechanisms
        assert not T2 in s.origin_mechanisms
        assert T3 in s.terminal_mechanisms
