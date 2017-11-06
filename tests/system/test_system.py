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

class TestInputAndInitialValueSpecs:
    def test_all_mechanisms_one_input_state_one_trial(self):
        a = TransferMechanism(name='a',
                              default_variable=[0.0, 0.0])
        b = TransferMechanism(name='b',
                              default_variable=[0.0, 0.0, 0.0])
        c = TransferMechanism(name='c')


        p1 = Process(pathway=[a, c],
                     name='p1')
        p2 = Process(pathway=[b, c],
                     name='p2')

        s = System(
            processes=[p1, p2]
        )

        # 1d list --> One input state, One trial
        inputs = {a: [1.0, 1.0],
                  b: [2.0, 2.0, 2.0]}

        s.run(inputs)

        # -----------------------------------------------------

        # 1d array --> One input state, One trial
        inputs = {a: np.array([1.0, 1.0]),
                  b: np.array([2.0, 2.0, 2.0])}

        s.run(inputs)

        # -----------------------------------------------------

        # 2d list, one element --> One input state, One trial
        inputs = {a: [[1.0, 1.0]],
                  b: [[2.0, 2.0, 2.0]]}

        s.run(inputs)

        # -----------------------------------------------------

        # 2d array --> One input state, One trial
        inputs = {a: np.array([[1.0, 1.0]]),
                  b: np.array([[2.0, 2.0, 2.0]])}

        s.run(inputs)

        # -----------------------------------------------------

        # 3d list, one element --> One input state, One trial
        inputs = {a: [[[1.0, 1.0]]],
                  b: [[[2.0, 2.0, 2.0]]]}

        s.run(inputs)

        # -----------------------------------------------------

        # 3d array --> One input state, One trial
        inputs = {a: np.array([[[1.0, 1.0]]]),
                  b: np.array([[[2.0, 2.0, 2.0]]])}

        s.run(inputs)

        # -----------------------------------------------------

    def test_documentation_example(self):
        a = TransferMechanism(name='a',
                              default_variable=[0.0, 0.0])
        b = TransferMechanism(name='b',
                              # default_variable=[[0.0, 0.0], [0.0, 0.0, 0.0]]
                              default_variable=[[0.0, 0.0], [0.0, 0.0]]
                              )
        # ignore Mechanism c for now

        p1 = Process(pathway=[a],
                     name='p1')
        p2 = Process(pathway=[b],
                     name='p2')

        s = System(
            processes=[p1, p2]
        )

        inputs = {a: [[1.0, 1.0], [1.0, 1.0]],
                  #  b: [[[2.0, 2.0], [3.0, 3.0, 3.0]], [[2.0, 2.0], [3.0, 3.0, 3.0]]]
                  b: [[[2.0, 2.0], [3.0, 3.0]], [[2.0, 2.0], [3.0, 3.0]]]
                  }

        s.run(inputs)


        # -----------------------------------------------------

        # # simplified documentation example -- ignore mechanism a
        # simple_system = System(processes=[p2])
        #
        # simple_inputs = {b: [[[2.0, 2.0], [3.0, 3.0]], [[2.0, 2.0], [3.0, 3.0]]]}
        # simple_system.run(simple_inputs)

    def test_all_mechanisms_multiple_input_states_one_trial(self):
        a = TransferMechanism(name='a',
                              default_variable=[[0.0], [0.0]])
        b = TransferMechanism(name='b',
                              default_variable=[0.0, 0.0, 0.0])
        c = TransferMechanism(name='c')


        p1 = Process(pathway=[a, c],
                     name='p1')
        p2 = Process(pathway=[b, c],
                     name='p2')

        s = System(
            processes=[p1, p2]
        )

        # -----------------------------------------------------
        inputs = {a: [ [[1.0], [1.0]] ],
                  b: [ [[2.0, 2.0, 2.0]] ]}

        s.run(inputs)

        # -----------------------------------------------------
        # inputs = {a: [[1.0], [1.0]],
        #           b: [[2.0, 2.0, 2.0]]}
        #
        # s.run(inputs)

        # -----------------------------------------------------
        #
        # # 2d array --> One input state, One trial
        # inputs = {a: np.array([[1.0], [1.0]]),
        #           b: np.array([[2.0, 2.0, 2.0]])}
        #
        # s.run(inputs)
        #
        # # -----------------------------------------------------
        #
        # # 3d list, one element --> One input state, One trial
        # inputs = {a: [[[1.0], [1.0]]],
        #           b: [[[2.0, 2.0, 2.0]]]}
        #
        # s.run(inputs)
        #
        # # -----------------------------------------------------
        #
        # # 3d array --> One input state, One trial
        # inputs = {a: np.array([[[1.0], [1.0]]]),
        #           b: np.array([[[2.0, 2.0, 2.0]]])}
        #
        # s.run(inputs)

        # -----------------------------------------------------


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

        inputs = {a: [2, 2]}
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
