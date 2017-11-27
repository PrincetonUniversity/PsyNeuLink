import numpy as np

from psyneulink.components.functions.function import BogaczEtAl, Linear, Logistic
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.process import Process
from psyneulink.components.projections.modulatory.controlprojection import ControlProjection
from psyneulink.components.system import System
from psyneulink.globals.keywords import ALLOCATION_SAMPLES
from psyneulink.globals.keywords import CYCLE, INITIALIZE_CYCLE, INTERNAL, ORIGIN, TERMINAL
from psyneulink.library.mechanisms.processing.integrator.ddm import DDM
from psyneulink.library.subsystems.evc.evccontrolmechanism import EVCControlMechanism


def test_danglingControlledMech():
    #
    #   first section is from Stroop Demo
    #
    Color_Input = TransferMechanism(name='Color Input', function=Linear(slope=0.2995))
    Word_Input = TransferMechanism(name='Word Input', function=Linear(slope=0.2995))

    # Processing Mechanisms (Control)
    Color_Hidden = TransferMechanism(
        name='Colors Hidden',
        function=Logistic(gain=(1.0, ControlProjection)),
    )
    Word_Hidden = TransferMechanism(
        name='Words Hidden',
        function=Logistic(gain=(1.0, ControlProjection)),
    )
    Output = TransferMechanism(
        name='Output',
        function=Logistic(gain=(1.0, ControlProjection)),
    )

    # Decision Mechanisms
    Decision = DDM(
        function=BogaczEtAl(
            drift_rate=(1.0),
            threshold=(0.1654),
            noise=(0.5),
            starting_point=(0),
            t0=0.25,
        ),
        name='Decision',
    )
    # Outcome Mechanisms:
    Reward = TransferMechanism(name='Reward')

    # Processes:
    ColorNamingProcess = Process(
        default_variable=[0],
        pathway=[Color_Input, Color_Hidden, Output, Decision],
        name='Color Naming Process',
    )

    WordReadingProcess = Process(
        default_variable=[0],
        pathway=[Word_Input, Word_Hidden, Output, Decision],
        name='Word Reading Process',
    )

    RewardProcess = Process(
        default_variable=[0],
        pathway=[Reward],
        name='RewardProcess',
    )

    # add another DDM but do not add to system
    second_DDM = DDM(
        function=BogaczEtAl(
            drift_rate=(
                1.0,
                ControlProjection(
                    function=Linear,
                    control_signal_params={
                        ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                    },
                ),
            ),
            threshold=(
                1.0,
                ControlProjection(
                    function=Linear,
                    control_signal_params={
                        ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                    },
                ),
            ),
            noise=(0.5),
            starting_point=(0),
            t0=0.45
        ),
        name='second_DDM',
    )

    # System:
    mySystem = System(
        processes=[ColorNamingProcess, WordReadingProcess, RewardProcess],
        controller=EVCControlMechanism,
        enable_controller=True,
        # monitor_for_control=[Reward, (DDM_PROBABILITY_UPPER_THRESHOLD, 1, -1)],
        name='EVC Gratton System',
    )

    # no assert, should only complete without error

class TestInputSpecsDocumentationExamples:

    def test_example_1(self):
        # "If num_trials is not in use, the number of inputs provided determines the number of trials in the run. For
        # example, if five inputs are provided for each origin mechanism, and num_trials is not specified, the system
        # will execute five times."

        import psyneulink as pnl

        a = pnl.TransferMechanism(name='a',
                                  default_variable=[[0.0, 0.0]])
        b = pnl.TransferMechanism(name='b',
                                  default_variable=[[0.0], [0.0]])
        c = pnl.TransferMechanism(name='c')

        p1 = pnl.Process(pathway=[a, c],
                         name='p1')
        p2 = pnl.Process(pathway=[b, c],
                         name='p2')

        s = pnl.System(processes=[p1, p2])

        input_dictionary = {a: [[[1.0, 1.0]], [[1.0, 1.0]]],
                            b: [[[2.0], [3.0]], [[2.0], [3.0]]]}

        check_inputs_dictionary = {a: [],
                                   b: []}
        def store_inputs():
            check_inputs_dictionary[a].append(a.input_values)
            check_inputs_dictionary[b].append(b.input_values)

        s.run(inputs=input_dictionary, call_after_trial=store_inputs)

        for mech in input_dictionary:
            assert np.allclose(check_inputs_dictionary[mech], input_dictionary[mech])


    def test_example_2(self):
        # "If num_trials is in use, run will iterate over the inputs until num_trials is reached. For example, if five
        # inputs are provided for each ORIGIN mechanism, and num_trials = 7, the system will execute seven times. The
        # first two items in the list of inputs will be used on the 6th and 7th trials, respectively."

        import psyneulink as pnl

        a = pnl.TransferMechanism(name='a')
        b = pnl.TransferMechanism(name='b')

        p1 = pnl.Process(pathway=[a, b])

        s = pnl.System(processes=[p1])

        input_dictionary = {a: [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]]}

        expected_inputs = [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]], [[1.0]], [[2.0]]]

        check_inputs = []

        def store_inputs():
            check_inputs.append(a.input_values)

        s.run(inputs=input_dictionary,
              num_trials=7,
              call_after_trial=store_inputs)

        assert np.allclose(check_inputs, expected_inputs)

    def test_example_3(self):
        # Origin mechanism has only one input state
        # COMPLETE specification

        import psyneulink as pnl

        a = pnl.TransferMechanism(name='a')
        b = pnl.TransferMechanism(name='b')

        p1 = pnl.Process(pathway=[a, b])

        s = pnl.System(processes=[p1])

        input_dictionary = {a: [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.1]]]}

        check_inputs = []

        def store_inputs():
            check_inputs.append(a.input_values)

        s.run(inputs=input_dictionary,
              call_after_trial=store_inputs)

        assert np.allclose(check_inputs, input_dictionary[a])


    def test_example_4(self):
        #  Origin mechanism has only one input state
        # SHORTCUT: drop the outer list on each input because 'a' only has one input state

        import psyneulink as pnl

        a = pnl.TransferMechanism(name='a')
        b = pnl.TransferMechanism(name='b')

        p1 = pnl.Process(pathway=[a, b])

        s = pnl.System(processes=[p1])

        input_dictionary = {a: [[1.0], [2.0], [3.0], [4.0], [5.2]]}

        check_inputs = []

        def store_inputs():
            check_inputs.append(a.input_values)

        s.run(inputs=input_dictionary,
              call_after_trial=store_inputs)

        assert np.allclose(check_inputs, [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.2]]])

    def test_example_5(self):
        #  Origin mechanism has only one input state
        # SHORTCUT: drop the remaining list on each input because 'a' only has one element

        import psyneulink as pnl

        a = pnl.TransferMechanism(name='a')
        b = pnl.TransferMechanism(name='b')

        p1 = pnl.Process(pathway=[a, b])

        s = pnl.System(processes=[p1])

        input_dictionary = {a: [1.0, 2.0, 3.0, 4.0, 5.3]}

        check_inputs = []

        def store_inputs():
            check_inputs.append(a.input_values)

        s.run(inputs=input_dictionary,
              call_after_trial=store_inputs)

        assert np.allclose(check_inputs, [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.3]]])

    def test_example_6(self):
        # Only one input is provided for the mechanism [single trial]
        # COMPLETE input specification

        import psyneulink as pnl

        a = pnl.TransferMechanism(name='a',
                                  default_variable=[[0.0], [0.0]])
        b = pnl.TransferMechanism(name='b')

        p1 = pnl.Process(pathway=[a, b])

        s = pnl.System(processes=[p1])

        check_inputs = []

        def store_inputs():
            check_inputs.append(a.input_values)

        input_dictionary = {a: [[[1.0], [2.0]]]}

        s.run(inputs=input_dictionary,
              call_after_trial=store_inputs)

        assert np.allclose(check_inputs, [[[1.0], [2.0]]])

    def test_example_7(self):
        # Only one input is provided for the mechanism [single trial]
        # SHORTCUT: Remove outer list because we only have one trial

        import psyneulink as pnl

        a = pnl.TransferMechanism(name='a',
                                  default_variable=[[0.0], [0.0]])
        b = pnl.TransferMechanism(name='b')

        p1 = pnl.Process(pathway=[a, b])

        s = pnl.System(processes=[p1])

        check_inputs = []

        def store_inputs():
            check_inputs.append(a.input_values)

        input_dictionary = {a: [[1.0], [2.0]]}

        s.run(inputs=input_dictionary,
              call_after_trial=store_inputs)

        assert np.allclose(check_inputs, [[[1.0], [2.0]]])

    def test_example_8(self):
        # Only one input is provided for the mechanism [repeat]
        # COMPLETE SPECIFICATION

        import psyneulink as pnl

        a = pnl.TransferMechanism(name='a',
                                  default_variable=[[0.0], [0.0]])
        b = pnl.TransferMechanism(name='b')

        p1 = pnl.Process(pathway=[a, b])

        s = pnl.System(processes=[p1])

        check_inputs = []

        def store_inputs():
            check_inputs.append(a.input_values)

        input_dictionary = {a: [[[1.0], [2.0]], [[1.0], [2.0]], [[1.0], [2.0]], [[1.0], [2.0]], [[1.0], [2.0]]]}

        s.run(inputs=input_dictionary,
              call_after_trial=store_inputs)

        assert np.allclose(check_inputs, input_dictionary[a])

    def test_example_9(self):
        # Only one input is provided for the mechanism [REPEAT]
        # SHORTCUT: Remove outer list because we want to use the same input on every trial

        import psyneulink as pnl

        a = pnl.TransferMechanism(name='a',
                                  default_variable=[[0.0], [0.0]])
        b = pnl.TransferMechanism(name='b')

        p1 = pnl.Process(pathway=[a, b])

        s = pnl.System(processes=[p1])

        check_inputs = []

        def store_inputs():
            check_inputs.append(a.input_values)

        input_dictionary = {a: [[1.0], [2.0]]}

        s.run(inputs=input_dictionary,
              num_trials=5,
              call_after_trial=store_inputs)

        assert np.allclose(check_inputs, [[[1.0], [2.0]], [[1.0], [2.0]], [[1.0], [2.0]], [[1.0], [2.0]], [[1.0], [2.0]]])

    def test_example_10(self):
        # There is only one origin mechanism in the system
        # COMPLETE SPECIFICATION

        import psyneulink as pnl

        a = pnl.TransferMechanism(name='a',
                                  default_variable=[[1.0, 2.0, 3.0]])
        b = pnl.TransferMechanism(name='b')

        p1 = pnl.Process(pathway=[a, b])

        s = pnl.System(processes=[p1])

        check_inputs = []

        def store_inputs():
            check_inputs.append(a.input_values)

        input_dictionary = {a: [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]}

        s.run(inputs=input_dictionary,
              call_after_trial=store_inputs)

        assert np.allclose(check_inputs, [[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]])

    def test_example_11(self):
        # There is only one origin mechanism in the system
        # SHORT CUT - specify inputs as a list instead of a dictionary

        import psyneulink as pnl

        a = pnl.TransferMechanism(name='a',
                                  default_variable=[[1.0, 2.0, 3.0]])
        b = pnl.TransferMechanism(name='b')

        p1 = pnl.Process(pathway=[a, b])

        s = pnl.System(processes=[p1])

        check_inputs = []

        def store_inputs():
            check_inputs.append(a.input_values)

        input_list = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]

        s.run(inputs=input_list,
              call_after_trial=store_inputs)

        assert np.allclose(check_inputs, [[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]])

    def test_example_12(self):
        # run process

        import psyneulink as pnl

        a = pnl.TransferMechanism(name='a',
                                  default_variable=[[1.0, 2.0, 3.0]])
        b = pnl.TransferMechanism(name='b')

        p1 = pnl.Process(pathway=[a, b])

        check_inputs = []

        def store_inputs():
            check_inputs.append(a.input_values)

        input_dictionary = [1.0, 2.0, 3.0]

        p1.run(inputs=input_dictionary,
              call_after_trial=store_inputs)

        assert np.allclose(check_inputs, [[[1.0, 2.0, 3.0]]])

        p1.execute(input_dictionary)

class TestInputSpecsHeterogeneousVariables:

    def test_heterogeneous_variables_drop_outer_list(self):
        # from psyneulink.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
        a = TransferMechanism(name='a', default_variable=[[0.0], [0.0,0.0]])

        p1 = Process(pathway=[a])

        s = System(
            processes=[p1]
        )

        inputs = {a: [[1.0], [2.0, 2.0]]}

        s.run(inputs)

    def test_heterogeneous_variables(self):
        # from psyneulink.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
        a = TransferMechanism(name='a', default_variable=[[0.0], [0.0,0.0]])

        p1 = Process(pathway=[a])

        s = System(
            processes=[p1]
        )

        inputs = {a: [[[1.1], [2.1, 2.1]], [[1.2], [2.2, 2.2]]]}

        s.run(inputs)
class TestGraphAndInput:

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
