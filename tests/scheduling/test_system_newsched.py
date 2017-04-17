import logging
import numpy

from PsyNeuLink.Components.System import system
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.IntegratorMechanism import IntegratorMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import DDM, DDM_PROBABILITY_UPPER_THRESHOLD
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCMechanism import EVCMechanism
from PsyNeuLink.Components.Projections.ControlProjection import ControlProjection
from PsyNeuLink.Components.Functions.Function import Linear, Logistic, BogaczEtAl, Integrator
from PsyNeuLink.scheduling.Scheduler import Scheduler
from PsyNeuLink.scheduling.condition import *
from PsyNeuLink.Globals.Keywords import SIMPLE

logger = logging.getLogger(__name__)

class TestInit:
    def test_create_scheduler_from_system_StroopDemo(self):
        Color_Input = TransferMechanism(name='Color Input', function=Linear(slope = 0.2995))
        Word_Input = TransferMechanism(name='Word Input', function=Linear(slope = 0.2995))

        # Processing Mechanisms (Control)
        Color_Hidden = TransferMechanism(name='Colors Hidden',
                                       function=Logistic(gain=(1.0, ControlProjection)))
        Word_Hidden = TransferMechanism(name='Words Hidden',
                                       function=Logistic(gain=(1.0, ControlProjection)))
        Output = TransferMechanism(name='Output',
                                       function=Logistic(gain=(1.0, ControlProjection)))

        # Decision Mechanisms
        Decision = DDM(function=BogaczEtAl(drift_rate=(1.0),
                                           threshold=(0.1654),
                                           noise=(0.5),
                                           starting_point=(0),
                                           t0=0.25),
                       name='Decision')
        # Outcome Mechanisms:
        Reward = TransferMechanism(name='Reward')

        # Processes:
        ColorNamingProcess = process(
            default_input_value = [0],
            pathway = [Color_Input, Color_Hidden, Output, Decision],
            name = 'Color Naming Process')

        WordReadingProcess = process(
            default_input_value = [0],
            pathway = [Word_Input, Word_Hidden, Output, Decision],
            name = 'Word Reading Process')

        RewardProcess = process(
            default_input_value = [0],
            pathway = [(Reward, 1)],
            name = 'RewardProcess')

        # System:
        mySystem = system(processes=[ColorNamingProcess, WordReadingProcess, RewardProcess],
                          controller=EVCMechanism,
                          enable_controller=True,
                          monitor_for_control=[Reward, (DDM_PROBABILITY_UPPER_THRESHOLD, 1, -1)],
                          name='EVC Gratton System')

        sched = Scheduler(system=mySystem)

        integrator_ColorInputPrediction = mySystem.mechanisms[7]
        integrator_RewardPrediction = mySystem.mechanisms[8]
        integrator_WordInputPrediction = mySystem.mechanisms[9]

        expected_consideration_queue = [
            {Color_Input, Word_Input, Reward},
            {Color_Hidden, Word_Hidden, integrator_ColorInputPrediction, integrator_WordInputPrediction, integrator_RewardPrediction},
            {Output},
            {Decision}
        ]

        assert sched.consideration_queue == expected_consideration_queue


class TestLinear:
    def test_one_run_twice(self):
        A = IntegratorMechanism(
            name='A',
            default_input_value = [0],
            function=Integrator(
                rate=.5,
                integration_type=SIMPLE
            )
        )

        p = process(
            default_input_value = [0],
            pathway = [A],
            name = 'p'
        )

        s = system(
            processes=[p],
            name = 's'
        )

        term_conds = {TimeScale.TRIAL: AfterNCalls(A, 2)}
        stim_list = {A: [[1]]}

        results = s.run(
            inputs=stim_list,
            termination_conditions=term_conds
        )

        terminal_mech = A
        expected_output = [
            numpy.array([1.]),
        ]

        for i in range(len(expected_output)):
            numpy.testing.assert_allclose(expected_output[i], terminal_mech.outputValue[i])

    def test_two_AAB(self):
        A = IntegratorMechanism(
            name='A',
            default_input_value = [0],
            function=Integrator(
                rate=.5,
                integration_type=SIMPLE
            )
        )

        B = TransferMechanism(
            name='B',
            default_input_value = [0],
            function=Linear(slope=2.0),
        )

        p = process(
            default_input_value = [0],
            pathway = [A, B],
            name = 'p'
        )

        s = system(
            processes=[p],
            name = 's'
        )

        term_conds = {TimeScale.TRIAL: AfterNCalls(B, 1)}
        stim_list = {A: [[1]]}

        sched = Scheduler(system=s)
        sched.add_condition(B, EveryNCalls(A, 2))
        s.scheduler = sched

        results = s.run(
            inputs=stim_list,
            termination_conditions=term_conds
        )

        terminal_mech = B
        expected_output = [
            numpy.array([2.]),
        ]

        for i in range(len(expected_output)):
            numpy.testing.assert_allclose(expected_output[i], terminal_mech.outputValue[i])

    def test_two_ABB(self):
        A = TransferMechanism(
            name='A',
            default_input_value = [0],
            function=Linear(slope=2.0),
        )

        B = IntegratorMechanism(
            name='B',
            default_input_value = [0],
            function=Integrator(
                rate=.5,
                integration_type=SIMPLE
            )
        )

        p = process(
            default_input_value = [0],
            pathway = [A, B],
            name = 'p'
        )

        s = system(
            processes=[p],
            name = 's'
        )

        term_conds = {TimeScale.TRIAL: AfterNCalls(B, 2)}
        stim_list = {A: [[1]]}

        sched = Scheduler(system=s)
        sched.add_condition(A, Any(AtPass(0), AfterNCalls(B, 2)))
        sched.add_condition(B, Any(JustRan(A), JustRan(B)))
        s.scheduler = sched

        results = s.run(
            inputs=stim_list,
            termination_conditions=term_conds
        )

        terminal_mech = B
        expected_output = [
            numpy.array([2.]),
        ]

        for i in range(len(expected_output)):
            numpy.testing.assert_allclose(expected_output[i], terminal_mech.outputValue[i])

class TestBranching:
    def test_three_ABAC(self):
        A = IntegratorMechanism(
            name='A',
            default_input_value = [0],
            function=Integrator(
                rate=.5,
                integration_type=SIMPLE
            )
        )

        B = TransferMechanism(
            name='B',
            default_input_value = [0],
            function=Linear(slope=2.0),
        )
        C = TransferMechanism(
            name='C',
            default_input_value = [0],
            function=Linear(slope=2.0),
        )

        p = process(
            default_input_value = [0],
            pathway = [A, B],
            name = 'p'
        )

        q = process(
            default_input_value = [0],
            pathway = [A, C],
            name = 'q'
        )

        s = system(
            processes=[p, q],
            name = 's'
        )

        term_conds = {TimeScale.TRIAL: AfterNCalls(C, 1)}
        stim_list = {A: [[1]]}

        sched = Scheduler(system=s)
        sched.add_condition(B, Any(AtNCalls(A, 1), EveryNCalls(A, 2)))
        sched.add_condition(C, EveryNCalls(A, 2))
        s.scheduler = sched

        results = s.run(
            inputs=stim_list,
            termination_conditions=term_conds
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
                numpy.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].outputValue[i])

    def test_three_ABACx2(self):
        A = IntegratorMechanism(
            name='A',
            default_input_value = [0],
            function=Integrator(
                rate=.5,
                integration_type=SIMPLE
            )
        )

        B = TransferMechanism(
            name='B',
            default_input_value = [0],
            function=Linear(slope=2.0),
        )
        C = TransferMechanism(
            name='C',
            default_input_value = [0],
            function=Linear(slope=2.0),
        )

        p = process(
            default_input_value = [0],
            pathway = [A, B],
            name = 'p'
        )

        q = process(
            default_input_value = [0],
            pathway = [A, C],
            name = 'q'
        )

        s = system(
            processes=[p, q],
            name = 's'
        )

        term_conds = {TimeScale.TRIAL: AfterNCalls(C, 2)}
        stim_list = {A: [[1]]}

        sched = Scheduler(system=s)
        sched.add_condition(B, Any(AtNCalls(A, 1), EveryNCalls(A, 2)))
        sched.add_condition(C, EveryNCalls(A, 2))
        s.scheduler = sched

        results = s.run(
            inputs=stim_list,
            termination_conditions=term_conds
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
                numpy.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].outputValue[i])

    def test_three_2_ABC(self):
        A = IntegratorMechanism(
            name='A',
            default_input_value = [0],
            function=Integrator(
                rate=.5,
                integration_type=SIMPLE
            )
        )

        B = IntegratorMechanism(
            name='B',
            default_input_value = [0],
            function=Integrator(
                rate=1,
                integration_type=SIMPLE
            )
        )

        C = TransferMechanism(
            name='C',
            default_input_value = [0],
            function=Linear(slope=2.0),
        )

        p = process(
            default_input_value = [0],
            pathway = [A, C],
            name = 'p'
        )

        q = process(
            default_input_value = [0],
            pathway = [B, C],
            name = 'q'
        )

        s = system(
            processes=[p, q],
            name = 's'
        )

        term_conds = {TimeScale.TRIAL: AfterNCalls(C, 1)}
        stim_list = {A: [[1]], B: [[2]]}

        sched = Scheduler(system=s)
        sched.add_condition(C, All(EveryNCalls(A, 1), EveryNCalls(B, 1)))
        s.scheduler = sched

        results = s.run(
            inputs=stim_list,
            termination_conditions=term_conds
        )

        terminal_mechs = [C]
        expected_output = [
            [
                numpy.array([5.]),
            ],
        ]

        for m in range(len(terminal_mechs)):
            for i in range(len(expected_output[m])):
                numpy.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].outputValue[i])

    def test_three_2_ABCx2(self):
        A = IntegratorMechanism(
            name='A',
            default_input_value = [0],
            function=Integrator(
                rate=.5,
                integration_type=SIMPLE
            )
        )

        B = IntegratorMechanism(
            name='B',
            default_input_value = [0],
            function=Integrator(
                rate=1,
                integration_type=SIMPLE
            )
        )

        C = TransferMechanism(
            name='C',
            default_input_value = [0],
            function=Linear(slope=2.0),
        )

        p = process(
            default_input_value = [0],
            pathway = [A, C],
            name = 'p'
        )

        q = process(
            default_input_value = [0],
            pathway = [B, C],
            name = 'q'
        )

        s = system(
            processes=[p, q],
            name = 's'
        )

        term_conds = {TimeScale.TRIAL: AfterNCalls(C, 2)}
        stim_list = {A: [[1]], B: [[2]]}

        sched = Scheduler(system=s)
        sched.add_condition(C, All(EveryNCalls(A, 1), EveryNCalls(B, 1)))
        s.scheduler = sched

        results = s.run(
            inputs=stim_list,
            termination_conditions=term_conds
        )

        terminal_mechs = [C]
        expected_output = [
            [
                numpy.array([10.]),
            ],
        ]

        # import code
        # code.interact(local=locals())

        for m in range(len(terminal_mechs)):
            for i in range(len(expected_output[m])):
                numpy.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].outputValue[i])

    def test_three_integrators(self):
        A = IntegratorMechanism(
            name='A',
            default_input_value = [0],
            function=Integrator(
                rate=1,
                integration_type=SIMPLE
            )
        )

        B = IntegratorMechanism(
            name='B',
            default_input_value = [0],
            function=Integrator(
                rate=1,
                integration_type=SIMPLE
            )
        )

        C = IntegratorMechanism(
            name='C',
            default_input_value = [0],
            function=Integrator(
                rate=1,
                integration_type=SIMPLE
            )
        )

        p = process(
            default_input_value = [0],
            pathway = [A, C],
            name = 'p'
        )

        q = process(
            default_input_value = [0],
            pathway = [B, C],
            name = 'q'
        )

        s = system(
            processes=[p, q],
            name = 's'
        )

        term_conds = {TimeScale.TRIAL: AfterNCalls(C, 2)}
        stim_list = {A: [[1]], B: [[1]]}

        sched = Scheduler(system=s)
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, Any(EveryNCalls(A, 1), EveryNCalls(B, 1)))
        s.scheduler = sched

        results = s.run(
            inputs=stim_list,
            termination_conditions=term_conds
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

        # import code
        # code.interact(local=locals())

        for m in range(len(mechs)):
            for i in range(len(expected_output[m])):
                numpy.testing.assert_allclose(expected_output[m][i], mechs[m].outputValue[i])

    def test_four_ABBCD(self):
        A = TransferMechanism(
            name='A',
            default_input_value = [0],
            function=Linear(slope=2.0),
        )

        B = IntegratorMechanism(
            name='B',
            default_input_value = [0],
            function=Integrator(
                rate=.5,
                integration_type=SIMPLE
            )
        )

        C = IntegratorMechanism(
            name='C',
            default_input_value = [0],
            function=Integrator(
                rate=.5,
                integration_type=SIMPLE
            )
        )

        D = TransferMechanism(
            name='D',
            default_input_value = [0],
            function=Linear(slope=1.0),
        )

        p = process(
            default_input_value = [0],
            pathway = [A, B, D],
            name = 'p'
        )

        q = process(
            default_input_value = [0],
            pathway = [A, C, D],
            name = 'q'
        )

        s = system(
            processes=[p, q],
            name = 's'
        )

        term_conds = {TimeScale.TRIAL: AfterNCalls(D, 1)}
        stim_list = {A: [[1]]}

        sched = Scheduler(system=s)
        sched.add_condition(B, EveryNCalls(A, 1))
        sched.add_condition(C, EveryNCalls(A, 2))
        sched.add_condition(D, Any(EveryNCalls(B, 3), EveryNCalls(C, 3)))
        s.scheduler = sched

        results = s.run(
            inputs=stim_list,
            termination_conditions=term_conds
        )

        terminal_mechs = [D]
        expected_output = [
            [
                numpy.array([4.]),
            ],
        ]

        # import code
        # code.interact(local=locals())

        for m in range(len(terminal_mechs)):
            for i in range(len(expected_output[m])):
                numpy.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].outputValue[i])

    def test_five_ABABCDE(self):
        A = TransferMechanism(
            name='A',
            default_input_value = [0],
            function=Linear(slope=2.0),
        )

        B = TransferMechanism(
            name='B',
            default_input_value = [0],
            function=Linear(slope=2.0),
        )

        C = IntegratorMechanism(
            name='C',
            default_input_value = [0],
            function=Integrator(
                rate=.5,
                integration_type=SIMPLE
            )
        )

        D = TransferMechanism(
            name='D',
            default_input_value = [0],
            function=Linear(slope=1.0),
        )

        E = TransferMechanism(
            name='E',
            default_input_value = [0],
            function=Linear(slope=2.0),
        )

        p = process(
            default_input_value = [0],
            pathway = [A, C, D],
            name = 'p'
        )

        q = process(
            default_input_value = [0],
            pathway = [B, C, E],
            name = 'q'
        )

        s = system(
            processes=[p, q],
            name = 's'
        )

        term_conds = {TimeScale.TRIAL: AfterNCalls(E, 1)}
        stim_list = {A: [[1]], B:[[2]]}

        sched = Scheduler(system=s)
        sched.add_condition(C, Any(EveryNCalls(A, 1), EveryNCalls(B, 1)))
        sched.add_condition(D, EveryNCalls(C, 1))
        sched.add_condition(E, EveryNCalls(C, 1))
        s.scheduler = sched

        results = s.run(
            inputs=stim_list,
            termination_conditions=term_conds
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

        # import code
        # code.interact(local=locals())

        for m in range(len(terminal_mechs)):
            for i in range(len(expected_output[m])):
                numpy.testing.assert_allclose(expected_output[m][i], terminal_mechs[m].outputValue[i])
