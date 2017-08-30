import numpy
from PsyNeuLink.Components.Functions.Function import BogaczEtAl, Linear, Logistic, SimpleIntegrator
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
from PsyNeuLink.Components.System import system
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.IntegratorMechanisms import IntegratorMechanism
from PsyNeuLink.Library.Mechanisms.AdaptiveMechanisms.EVC import EVCMechanism
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.IntegratorMechanisms import DDM
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.TransferMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Scheduling.Condition import AfterNCalls, All, Any, AtNCalls, AtPass, EveryNCalls, JustRan
from PsyNeuLink.Scheduling.Scheduler import Scheduler
from PsyNeuLink.Scheduling.TimeScale import TimeScale


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
        ColorNamingProcess = process(
            default_variable=[0],
            pathway=[Color_Input, Color_Hidden, Output, Decision],
            name='Color Naming Process',
        )

        WordReadingProcess = process(
            default_variable=[0],
            pathway=[Word_Input, Word_Hidden, Output, Decision],
            name='Word Reading Process',
        )

        RewardProcess = process(
            default_variable=[0],
            pathway=[Reward],
            name='RewardProcess',
        )

        # System:
        mySystem = system(
            processes=[ColorNamingProcess, WordReadingProcess, RewardProcess],
            controller=EVCMechanism,
            enable_controller=True,
            # monitor_for_control=[Reward, (PROBABILITY_UPPER_THRESHOLD, 1, -1)],
            name='EVC Gratton System',
        )

        sched = Scheduler(system=mySystem)

        integrator_ColorInputPrediction = mySystem.execution_list[7]
        integrator_RewardPrediction = mySystem.execution_list[8]
        integrator_WordInputPrediction = mySystem.execution_list[9]
        objective_EVC_mech = mySystem.execution_list[10]

        expected_consideration_queue = [
            {Color_Input, Word_Input, Reward, integrator_ColorInputPrediction, integrator_WordInputPrediction, integrator_RewardPrediction},
            {Color_Hidden, Word_Hidden},
            {Output},
            {Decision},
            {objective_EVC_mech},
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

        p = process(
            default_variable=[0],
            pathway=[A],
            name='p'
        )

        s = system(
            processes=[p],
            name='s'
        )

        term_conds = {TimeScale.TRIAL: AfterNCalls(A, 2)}
        stim_list = {A: [[1]]}

        s.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mech = A
        expected_output = [
            numpy.array([1.]),
        ]

        for i in range(len(expected_output)):
            numpy.testing.assert_allclose(expected_output[i], terminal_mech.output_values[i])

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

        p = process(
            default_variable=[0],
            pathway=[A, B],
            name='p'
        )

        s = system(
            processes=[p],
            name='s'
        )

        term_conds = {TimeScale.TRIAL: AfterNCalls(B, 1)}
        stim_list = {A: [[1]]}

        sched = Scheduler(system=s)
        sched.add_condition(B, EveryNCalls(A, 2))
        s.scheduler_processing = sched

        s.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mech = B
        expected_output = [
            numpy.array([2.]),
        ]

        for i in range(len(expected_output)):
            numpy.testing.assert_allclose(expected_output[i], terminal_mech.output_values[i])

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

        p = process(
            default_variable=[0],
            pathway=[A, B],
            name='p'
        )

        s = system(
            processes=[p],
            name='s'
        )

        term_conds = {TimeScale.TRIAL: AfterNCalls(B, 2)}
        stim_list = {A: [[1]]}

        sched = Scheduler(system=s)
        sched.add_condition(A, Any(AtPass(0), AfterNCalls(B, 2)))
        sched.add_condition(B, Any(JustRan(A), JustRan(B)))
        s.scheduler_processing = sched

        s.run(
            inputs=stim_list,
            termination_processing=term_conds
        )

        terminal_mech = B
        expected_output = [
            numpy.array([2.]),
        ]

        for i in range(len(expected_output)):
            numpy.testing.assert_allclose(expected_output[i], terminal_mech.output_values[i])


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

        p = process(
            default_variable=[0],
            pathway=[A, B],
            name='p'
        )

        q = process(
            default_variable=[0],
            pathway=[A, C],
            name='q'
        )

        s = system(
            processes=[p, q],
            name='s'
        )

        term_conds = {TimeScale.TRIAL: AfterNCalls(C, 1)}
        stim_list = {A: [[1]]}

        sched = Scheduler(system=s)
        sched.add_condition(B, Any(AtNCalls(A, 1), EveryNCalls(A, 2)))
        sched.add_condition(C, EveryNCalls(A, 2))
        s.scheduler_processing = sched

        s.run(
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
                numpy.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].output_values[i])

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

        p = process(
            default_variable=[0],
            pathway=[A, B],
            name='p'
        )

        q = process(
            default_variable=[0],
            pathway=[A, C],
            name='q'
        )

        s = system(
            processes=[p, q],
            name='s'
        )

        term_conds = {TimeScale.TRIAL: AfterNCalls(C, 2)}
        stim_list = {A: [[1]]}

        sched = Scheduler(system=s)
        sched.add_condition(B, Any(AtNCalls(A, 1), EveryNCalls(A, 2)))
        sched.add_condition(C, EveryNCalls(A, 2))
        s.scheduler_processing = sched

        s.run(
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
                numpy.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].output_values[i])

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

        p = process(
            default_variable=[0],
            pathway=[A, C],
            name='p'
        )

        q = process(
            default_variable=[0],
            pathway=[B, C],
            name='q'
        )

        s = system(
            processes=[p, q],
            name='s'
        )

        term_conds = {TimeScale.TRIAL: AfterNCalls(C, 1)}
        stim_list = {A: [[1]], B: [[2]]}

        sched = Scheduler(system=s)
        sched.add_condition(C, All(EveryNCalls(A, 1), EveryNCalls(B, 1)))
        s.scheduler_processing = sched

        s.run(
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
                numpy.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].output_values[i])

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

        p = process(
            default_variable=[0],
            pathway=[A, C],
            name='p'
        )

        q = process(
            default_variable=[0],
            pathway=[B, C],
            name='q'
        )

        s = system(
            processes=[p, q],
            name='s'
        )

        term_conds = {TimeScale.TRIAL: AfterNCalls(C, 2)}
        stim_list = {A: [[1]], B: [[2]]}

        sched = Scheduler(system=s)
        sched.add_condition(C, All(EveryNCalls(A, 1), EveryNCalls(B, 1)))
        s.scheduler_processing = sched

        s.run(
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
                numpy.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].output_values[i])

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

        p = process(
            default_variable=[0],
            pathway=[A, C],
            name='p'
        )

        q = process(
            default_variable=[0],
            pathway=[B, C],
            name='q'
        )

        s = system(
            processes=[p, q],
            name='s'
        )

        term_conds = {TimeScale.TRIAL: AfterNCalls(C, 2)}
        stim_list = {A: [[1]], B: [[1]]}

        sched = Scheduler(system=s)
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, Any(EveryNCalls(A, 1), EveryNCalls(B, 1)))
        s.scheduler_processing = sched

        s.run(
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
                numpy.testing.assert_allclose(expected_output[m][i], mechs[m].output_values[i])

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

        p = process(
            default_variable=[0],
            pathway=[A, B, D],
            name='p'
        )

        q = process(
            default_variable=[0],
            pathway=[A, C, D],
            name='q'
        )

        s = system(
            processes=[p, q],
            name='s'
        )

        term_conds = {TimeScale.TRIAL: AfterNCalls(D, 1)}
        stim_list = {A: [[1]]}

        sched = Scheduler(system=s)
        sched.add_condition(B, EveryNCalls(A, 1))
        sched.add_condition(C, EveryNCalls(A, 2))
        sched.add_condition(D, Any(EveryNCalls(B, 3), EveryNCalls(C, 3)))
        s.scheduler_processing = sched

        s.run(
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
                numpy.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].output_values[i])

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

        p = process(
            default_variable=[0],
            pathway=[A, C],
            name='p'
        )

        p1 = process(
            default_variable=[0],
            pathway=[A, D],
            name='p1'
        )

        q = process(
            default_variable=[0],
            pathway=[B, C],
            name='q'
        )

        q1 = process(
            default_variable=[0],
            pathway=[B, D],
            name='q1'
        )

        s = system(
            processes=[p, p1, q, q1],
            name='s'
        )

        term_conds = {TimeScale.TRIAL: All(AfterNCalls(C, 1), AfterNCalls(D, 1))}
        stim_list = {A: [[1]], B: [[1]]}

        sched = Scheduler(system=s)
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, EveryNCalls(A, 1))
        sched.add_condition(D, EveryNCalls(B, 1))
        s.scheduler_processing = sched

        s.run(
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
                numpy.testing.assert_allclose(expected_output[m][i], mechs[m].output_values[i])

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

        p = process(
            default_variable=[0],
            pathway=[A, C, D],
            name='p'
        )

        q = process(
            default_variable=[0],
            pathway=[B, C, E],
            name='q'
        )

        s = system(
            processes=[p, q],
            name='s'
        )

        term_conds = {TimeScale.TRIAL: AfterNCalls(E, 1)}
        stim_list = {A: [[1]], B: [[2]]}

        sched = Scheduler(system=s)
        sched.add_condition(C, Any(EveryNCalls(A, 1), EveryNCalls(B, 1)))
        sched.add_condition(D, EveryNCalls(C, 1))
        sched.add_condition(E, EveryNCalls(C, 1))
        s.scheduler_processing = sched

        s.run(
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
                numpy.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].output_values[i])

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

        p = [
            process(
                default_variable=[0],
                pathway=[A, C, E],
                name='p'
            ),
            process(
                default_variable=[0],
                pathway=[A, C, F],
                name='p1'
            ),
            process(
                default_variable=[0],
                pathway=[A, D, E],
                name='p2'
            ),
            process(
                default_variable=[0],
                pathway=[A, D, F],
                name='p3'
            ),
            process(
                default_variable=[0],
                pathway=[B, C, E],
                name='q'
            ),
            process(
                default_variable=[0],
                pathway=[B, C, F],
                name='q1'
            ),
            process(
                default_variable=[0],
                pathway=[B, D, E],
                name='q2'
            ),
            process(
                default_variable=[0],
                pathway=[B, D, F],
                name='q3'
            )
        ]

        s = system(
            processes=p,
            name='s'
        )

        term_conds = {TimeScale.TRIAL: All(AfterNCalls(E, 1), AfterNCalls(F, 1))}
        stim_list = {A: [[1]], B: [[1]]}

        sched = Scheduler(system=s)
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, EveryNCalls(A, 1))
        sched.add_condition(D, EveryNCalls(B, 1))
        sched.add_condition(E, EveryNCalls(C, 1))
        sched.add_condition(F, EveryNCalls(D, 2))
        s.scheduler_processing = sched

        s.run(
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
                numpy.testing.assert_allclose(expected_output[m][i], m.output_values[i])
