import numpy

from psyneulink.core.compositions.composition import Composition
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import SimpleIntegrator
from psyneulink.core.components.functions.distributionfunctions import DriftDiffusionAnalytical
from psyneulink.core.components.functions.transferfunctions import Linear, Logistic
from psyneulink.core.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.projections.modulatory.controlprojection import ControlProjection
from psyneulink.core.scheduling.condition import AfterNCalls, All, Any, AtNCalls, AtPass, EveryNCalls, JustRan, Never
from psyneulink.core.scheduling.scheduler import Scheduler
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.library.components.mechanisms.processing.integrator.ddm import DDM


class TestInit:

    def test_create_scheduler_from_system_StroopDemo(self):
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
            function=DriftDiffusionAnalytical(
                drift_rate=(1.0),
                threshold=(0.1654),
                noise=(0.5),
                starting_point=(0),
                t0=0.25,
            ),
            name='Decision',
        )
        # Outcome Mechanism:
        Reward = TransferMechanism(name='Reward')

        myComposition = Composition(pathways=[[Color_Input, Color_Hidden, Output, Decision],
                                              [Word_Input, Word_Hidden, Output, Decision],
                                              [Reward]])

        sched = Scheduler(composition=myComposition)

        expected_consideration_queue = [
            {Color_Input, Word_Input, Reward},
            {Color_Hidden, Word_Hidden},
            {Output},
            {Decision}
        ]

        assert sched.consideration_queue == expected_consideration_queue


class TestLinear:

    def test_one_run_twice(self):
        A = IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=.5,
            )
        )

        c = Composition(pathways=[A])

        term_conds = {TimeScale.TRIAL: AfterNCalls(A, 2)}
        stim_list = {A: [[1]]}

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mech = A
        expected_output = [
            numpy.array([1.]),
        ]

        for i in range(len(expected_output)):
            numpy.testing.assert_allclose(expected_output[i], terminal_mech.get_output_values(c)[i])

    def test_two_AAB(self):
        A = IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=.5
            )
        )

        B = TransferMechanism(
            name='B',
            default_variable=[0],
            function=Linear(slope=2.0),
        )

        c = Composition(pathways=[A, B])

        term_conds = {TimeScale.TRIAL: AfterNCalls(B, 1)}
        stim_list = {A: [[1]]}

        sched = Scheduler(composition=c)
        sched.add_condition(B, EveryNCalls(A, 2))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mech = B
        expected_output = [
            numpy.array([2.]),
        ]

        for i in range(len(expected_output)):
            numpy.testing.assert_allclose(expected_output[i], terminal_mech.get_output_values(c)[i])

    def test_two_ABB(self):
        A = TransferMechanism(
            name='A',
            default_variable=[0],
            function=Linear(slope=2.0),
        )

        B = IntegratorMechanism(
            name='B',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=.5
            )
        )

        c = Composition(pathways=[A, B])

        term_conds = {TimeScale.TRIAL: AfterNCalls(B, 2)}
        stim_list = {A: [[1]]}

        sched = Scheduler(composition=c)
        sched.add_condition(A, Any(AtPass(0), AfterNCalls(B, 2)))
        sched.add_condition(B, Any(JustRan(A), JustRan(B)))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mech = B
        expected_output = [
            numpy.array([2.]),
        ]

        for i in range(len(expected_output)):
            numpy.testing.assert_allclose(expected_output[i], terminal_mech.get_output_values(c)[i])


class TestBranching:

    def test_three_ABAC(self):
        A = IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=.5
            )
        )

        B = TransferMechanism(
            name='B',
            default_variable=[0],
            function=Linear(slope=2.0),
        )
        C = TransferMechanism(
            name='C',
            default_variable=[0],
            function=Linear(slope=2.0),
        )

        c = Composition(pathways=[[A,B],[A,C]])

        term_conds = {TimeScale.TRIAL: AfterNCalls(C, 1)}
        stim_list = {A: [[1]]}

        sched = Scheduler(composition=c)
        sched.add_condition(B, Any(AtNCalls(A, 1), EveryNCalls(A, 2)))
        sched.add_condition(C, EveryNCalls(A, 2))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mechs = [B, C]
        expected_output = [
            [
                numpy.array([1.]),
            ],
            [
                numpy.array([2.]),
            ],
        ]

        for m in range(len(terminal_mechs)):
            for i in range(len(expected_output[m])):
                numpy.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].get_output_values(c)[i])

    def test_three_ABAC_convenience(self):
        A = IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=.5
            )
        )

        B = TransferMechanism(
            name='B',
            default_variable=[0],
            function=Linear(slope=2.0),
        )
        C = TransferMechanism(
            name='C',
            default_variable=[0],
            function=Linear(slope=2.0),
        )

        c = Composition(pathways=[[A,B],[A,C]])

        term_conds = {TimeScale.TRIAL: AfterNCalls(C, 1)}
        stim_list = {A: [[1]]}

        c.scheduler.add_condition(B, Any(AtNCalls(A, 1), EveryNCalls(A, 2)))
        c.scheduler.add_condition(C, EveryNCalls(A, 2))

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mechs = [B, C]
        expected_output = [
            [
                numpy.array([1.]),
            ],
            [
                numpy.array([2.]),
            ],
        ]

        for m in range(len(terminal_mechs)):
            for i in range(len(expected_output[m])):
                numpy.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].get_output_values(c)[i])

    def test_three_ABACx2(self):
        A = IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=.5
            )
        )

        B = TransferMechanism(
            name='B',
            default_variable=[0],
            function=Linear(slope=2.0),
        )
        C = TransferMechanism(
            name='C',
            default_variable=[0],
            function=Linear(slope=2.0),
        )

        c = Composition(pathways=[[A,B],[A,C]])

        term_conds = {TimeScale.TRIAL: AfterNCalls(C, 2)}
        stim_list = {A: [[1]]}

        sched = Scheduler(composition=c)
        sched.add_condition(B, Any(AtNCalls(A, 1), EveryNCalls(A, 2)))
        sched.add_condition(C, EveryNCalls(A, 2))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mechs = [B, C]
        expected_output = [
            [
                numpy.array([3.]),
            ],
            [
                numpy.array([4.]),
            ],
        ]

        for m in range(len(terminal_mechs)):
            for i in range(len(expected_output[m])):
                numpy.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].get_output_values(c)[i])

    def test_three_2_ABC(self):
        A = IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=.5
            )
        )

        B = IntegratorMechanism(
            name='B',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=1
            )
        )

        C = TransferMechanism(
            name='C',
            default_variable=[0],
            function=Linear(slope=2.0),
        )

        c = Composition(pathways=[[A,C],[B,C]])

        term_conds = {TimeScale.TRIAL: AfterNCalls(C, 1)}
        stim_list = {A: [[1]], B: [[2]]}

        sched = Scheduler(composition=c)
        sched.add_condition(C, All(EveryNCalls(A, 1), EveryNCalls(B, 1)))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mechs = [C]
        expected_output = [
            [
                numpy.array([5.]),
            ],
        ]

        for m in range(len(terminal_mechs)):
            for i in range(len(expected_output[m])):
                numpy.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].get_output_values(c)[i])

    def test_three_2_ABCx2(self):
        A = IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=.5
            )
        )

        B = IntegratorMechanism(
            name='B',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=1
            )
        )

        C = TransferMechanism(
            name='C',
            default_variable=[0],
            function=Linear(slope=2.0),
        )

        c = Composition(pathways=[[A,C],[B,C]])

        term_conds = {TimeScale.TRIAL: AfterNCalls(C, 2)}
        stim_list = {A: [[1]], B: [[2]]}

        sched = Scheduler(composition=c)
        sched.add_condition(C, All(EveryNCalls(A, 1), EveryNCalls(B, 1)))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mechs = [C]
        expected_output = [
            [
                numpy.array([10.]),
            ],
        ]

        for m in range(len(terminal_mechs)):
            for i in range(len(expected_output[m])):
                numpy.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].get_output_values(c)[i])

    def test_three_integrators(self):
        A = IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=1
            )
        )

        B = IntegratorMechanism(
            name='B',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=1
            )
        )

        C = IntegratorMechanism(
            name='C',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=1
            )
        )

        c = Composition(pathways=[[A,C],[B,C]])

        term_conds = {TimeScale.TRIAL: AfterNCalls(C, 2)}
        stim_list = {A: [[1]], B: [[1]]}

        sched = Scheduler(composition=c)
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, Any(EveryNCalls(A, 1), EveryNCalls(B, 1)))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        mechs = [A, B, C]
        expected_output = [
            [
                numpy.array([2.]),
            ],
            [
                numpy.array([1.]),
            ],
            [
                numpy.array([4.]),
            ],
        ]

        for m in range(len(mechs)):
            for i in range(len(expected_output[m])):
                numpy.testing.assert_allclose(expected_output[m][i], mechs[m].get_output_values(c)[i])

    def test_four_ABBCD(self):
        A = TransferMechanism(
            name='A',
            default_variable=[0],
            function=Linear(slope=2.0),
        )

        B = IntegratorMechanism(
            name='B',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=.5
            )
        )

        C = IntegratorMechanism(
            name='C',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=.5
            )
        )

        D = TransferMechanism(
            name='D',
            default_variable=[0],
            function=Linear(slope=1.0),
        )

        c = Composition(pathways=[[A,B,D],[A,C,D]])

        term_conds = {TimeScale.TRIAL: AfterNCalls(D, 1)}
        stim_list = {A: [[1]]}

        sched = Scheduler(composition=c)
        sched.add_condition(B, EveryNCalls(A, 1))
        sched.add_condition(C, EveryNCalls(A, 2))
        sched.add_condition(D, Any(EveryNCalls(B, 3), EveryNCalls(C, 3)))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mechs = [D]
        expected_output = [
            [
                numpy.array([4.]),
            ],
        ]

        for m in range(len(terminal_mechs)):
            for i in range(len(expected_output[m])):
                numpy.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].get_output_values(c)[i])

    def test_four_integrators_mixed(self):
        A = IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=1
            )
        )

        B = IntegratorMechanism(
            name='B',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=1
            )
        )

        C = IntegratorMechanism(
            name='C',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=1
            )
        )

        D = IntegratorMechanism(
            name='D',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=1
            )
        )

        c = Composition(pathways=[[A,C],[A,D],[B,C],[B,D]])

        term_conds = {TimeScale.TRIAL: All(AfterNCalls(C, 1), AfterNCalls(D, 1))}
        stim_list = {A: [[1]], B: [[1]]}

        sched = Scheduler(composition=c)
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, EveryNCalls(A, 1))
        sched.add_condition(D, EveryNCalls(B, 1))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        mechs = [A, B, C, D]
        expected_output = [
            [
                numpy.array([2.]),
            ],
            [
                numpy.array([1.]),
            ],
            [
                numpy.array([4.]),
            ],
            [
                numpy.array([3.]),
            ],
        ]

        for m in range(len(mechs)):
            for i in range(len(expected_output[m])):
                numpy.testing.assert_allclose(expected_output[m][i], mechs[m].get_output_values(c)[i])

    def test_five_ABABCDE(self):
        A = TransferMechanism(
            name='A',
            default_variable=[0],
            function=Linear(slope=2.0),
        )

        B = TransferMechanism(
            name='B',
            default_variable=[0],
            function=Linear(slope=2.0),
        )

        C = IntegratorMechanism(
            name='C',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=.5
            )
        )

        D = TransferMechanism(
            name='D',
            default_variable=[0],
            function=Linear(slope=1.0),
        )

        E = TransferMechanism(
            name='E',
            default_variable=[0],
            function=Linear(slope=2.0),
        )

        c = Composition(pathways=[[A,C,D],[B,C,E]])

        term_conds = {TimeScale.TRIAL: AfterNCalls(E, 1)}
        stim_list = {A: [[1]], B: [[2]]}

        sched = Scheduler(composition=c)
        sched.add_condition(C, Any(EveryNCalls(A, 1), EveryNCalls(B, 1)))
        sched.add_condition(D, EveryNCalls(C, 1))
        sched.add_condition(E, EveryNCalls(C, 1))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mechs = [D, E]
        expected_output = [
            [
                numpy.array([3.]),
            ],
            [
                numpy.array([6.]),
            ],
        ]

        for m in range(len(terminal_mechs)):
            for i in range(len(expected_output[m])):
                numpy.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].get_output_values(c)[i])

    #
    #   A  B
    #   |\/|
    #   C  D
    #   |\/|
    #   E  F
    #
    def test_six_integrators_threelayer_mixed(self):
        A = IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=1
            )
        )

        B = IntegratorMechanism(
            name='B',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=1
            )
        )

        C = IntegratorMechanism(
            name='C',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=1
            )
        )

        D = IntegratorMechanism(
            name='D',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=1
            )
        )

        E = IntegratorMechanism(
            name='E',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=1
            )
        )

        F = IntegratorMechanism(
            name='F',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=1
            )
        )

        c = Composition(pathways=[[A,C,E],[A,C,F],[A,D,E],[A,D,F],[B,C,E],[B,C,F],[B,D,E],[B,D,F]])

        term_conds = {TimeScale.TRIAL: All(AfterNCalls(E, 1), AfterNCalls(F, 1))}
        stim_list = {A: [[1]], B: [[1]]}

        sched = Scheduler(composition=c)
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, EveryNCalls(A, 1))
        sched.add_condition(D, EveryNCalls(B, 1))
        sched.add_condition(E, EveryNCalls(C, 1))
        sched.add_condition(F, EveryNCalls(D, 2))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        # Intermediate time steps
        #
        #     0   1   2   3
        #
        # A   1   2   3   4
        # B       1       2
        # C   1   4   8   14
        # D       3       9
        # E   1   8   19  42
        # F               23
        #
        expected_output = {
            A: [
                numpy.array([4.]),
            ],
            B: [
                numpy.array([2.]),
            ],
            C: [
                numpy.array([14.]),
            ],
            D: [
                numpy.array([9.]),
            ],
            E: [
                numpy.array([42.]),
            ],
            F: [
                numpy.array([23.]),
            ],
        }

        for m in expected_output:
            for i in range(len(expected_output[m])):
                numpy.testing.assert_allclose(expected_output[m][i], m.get_output_values(c)[i])


class TestTermination:

    def test_termination_conditions_reset(self):
        A = IntegratorMechanism(
            name='A',
            default_variable=[0],
            function=SimpleIntegrator(
                rate=.5
            )
        )

        B = TransferMechanism(
            name='B',
            default_variable=[0],
            function=Linear(slope=2.0),
        )

        c = Composition(pathways=[[A,B]])

        term_conds = {TimeScale.TRIAL: AfterNCalls(B, 2)}
        stim_list = {A: [[1]]}

        sched = Scheduler(composition=c)
        sched.add_condition(B, EveryNCalls(A, 2))
        c.scheduler = sched

        c.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        # A should run four times
        terminal_mech = B
        expected_output = [
            numpy.array([4.]),
        ]

        for i in range(len(expected_output)):
            numpy.testing.assert_allclose(expected_output[i], terminal_mech.get_output_values(c)[i])

        c.run(
            inputs=stim_list,
        )

        # A should run an additional two times
        terminal_mech = B
        expected_output = [
            numpy.array([6.]),
        ]

        for i in range(len(expected_output)):
            numpy.testing.assert_allclose(expected_output[i], terminal_mech.get_output_values(c)[i])
