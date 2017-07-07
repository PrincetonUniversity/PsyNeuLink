import logging
import pytest

from PsyNeuLink.Components.Functions.Function import Linear
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Scheduling.Condition import AfterNCalls, AfterNTrials, AfterPass, All, Always, Any, AtPass, BeforePass, EveryNCalls, EveryNPasses, JustRan, WhenFinished
from PsyNeuLink.Scheduling.Scheduler import Scheduler
from PsyNeuLink.Scheduling.TimeScale import TimeScale
from PsyNeuLink.composition import Composition

logger = logging.getLogger(__name__)


class TestScheduler:

    def test_copy(self):
        pass

    def test_deepcopy(self):
        pass


class TestLinear:

    def test_no_termination_conds(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(B, MappingProjection(), C)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, EveryNCalls(B, 3))

        output = list(sched.run())

        expected_output = [
            A, A, B, A, A, B, A, A, B, C,
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    # tests below are copied from old scheduler, need renaming
    def test_1(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(B, MappingProjection(), C)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, EveryNCalls(B, 3))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 4, time_scale=TimeScale.TRIAL)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, B, A, A, B, A, A, B, C,
            A, A, B, A, A, B, A, A, B, C,
            A, A, B, A, A, B, A, A, B, C,
            A, A, B, A, A, B, A, A, B, C,
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_1b(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(B, MappingProjection(), C)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, Any(EveryNCalls(A, 2), AfterPass(1)))
        sched.add_condition(C, EveryNCalls(B, 3))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 4, time_scale=TimeScale.TRIAL)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, B, A, B, A, B, C,
            A, B, A, B, A, B, C,
            A, B, A, B, A, B, C,
            A, B, A, B, A, B, C,
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_2(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(B, MappingProjection(), C)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, EveryNCalls(B, 2))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 1, time_scale=TimeScale.TRIAL)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, B, A, A, B, C]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_3(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(B, MappingProjection(), C)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, All(AfterNCalls(B, 2), EveryNCalls(B, 1)))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 4, time_scale=TimeScale.TRIAL)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, B, A, A, B, C, A, A, B, C, A, A, B, C, A, A, B, C
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_6(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(B, MappingProjection(), C)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, BeforePass(5))
        sched.add_condition(B, AfterNCalls(A, 5))
        sched.add_condition(C, AfterNCalls(B, 1))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 3)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, A, A, A, B, C, B, C, B, C
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_6_two_trials(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(B, MappingProjection(), C)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, BeforePass(5))
        sched.add_condition(B, AfterNCalls(A, 5))
        sched.add_condition(C, AfterNCalls(B, 1))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(2)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 3)
        comp.run(
                inputs={A: range(6)},
                scheduler_processing=sched,
                termination_processing=termination_conds
        )
        output = sched.execution_list

        expected_output = [
            A, A, A, A, A, B, C, B, C, B, C,
            A, A, A, A, A, B, C, B, C, B, C
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_7(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        for m in [A, B]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 2))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = Any(AfterNCalls(A, 1), AfterNCalls(B, 1))
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_8(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        for m in [A, B]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 2))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = All(AfterNCalls(A, 1), AfterNCalls(B, 1))
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_9(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        for m in [A, B]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, WhenFinished(A))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(B, 2)

        output = []
        i = 0
        for step in sched.run(termination_conds=termination_conds):
            if i == 3:
                A.is_finished = True
            output.append(step)
            i += 1

        expected_output = [A, A, A, A, B, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_9b(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        A.is_finished = False
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        for m in [A, B]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, WhenFinished(A))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AtPass(5)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, A, A, A]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_10(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        A.is_finished = True
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')

        for m in [A, B]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, Any(WhenFinished(A), AfterNCalls(A, 3)))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(B, 5)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, B, A, B, A, B, A, B, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_10b(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        A.is_finished = False
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')

        for m in [A, B]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, Any(WhenFinished(A), AfterNCalls(A, 3)))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(B, 4)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, A, B, A, B, A, B, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_10c(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        A.is_finished = True
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')

        for m in [A, B]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, All(WhenFinished(A), AfterNCalls(A, 3)))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(B, 4)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, A, B, A, B, A, B, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_10d(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        A.is_finished = False
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')

        for m in [A, B]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, All(WhenFinished(A), AfterNCalls(A, 3)))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AtPass(10)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, A, A, A, A, A, A, A, A]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    ########################################
    # tests with linear compositions
    ########################################
    def test_linear_AAB(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        for m in [A, B]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 2))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNCalls(B, 2, time_scale=TimeScale.RUN)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(B, 2, time_scale=TimeScale.TRIAL)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, B, A, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_linear_ABB(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        for m in [A, B]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, Any(AtPass(0), EveryNCalls(B, 2)))
        sched.add_condition(B, Any(EveryNCalls(A, 1), EveryNCalls(B, 1)))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(B, 8, time_scale=TimeScale.TRIAL)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, B, B, A, B, B, A, B, B, A, B, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_linear_ABBCC(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(B, MappingProjection(), C)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, Any(AtPass(0), EveryNCalls(C, 2)))
        sched.add_condition(B, Any(JustRan(A), JustRan(B)))
        sched.add_condition(C, Any(EveryNCalls(B, 2), JustRan(C)))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 4, time_scale=TimeScale.TRIAL)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, B, B, C, C, A, B, B, C, C]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_linear_ABCBC(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(B, MappingProjection(), C)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, Any(AtPass(0), EveryNCalls(C, 2)))
        sched.add_condition(B, Any(EveryNCalls(A, 1), EveryNCalls(C, 1)))
        sched.add_condition(C, EveryNCalls(B, 1))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 4, time_scale=TimeScale.TRIAL)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, B, C, B, C, A, B, C, B, C]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    ########################################
    # tests with small branching compositions
    ########################################


class TestBranching:
    #   triangle:         A
    #                    / \
    #                   B   C

    def test_triangle_1(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(A, MappingProjection(), C)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 1))
        sched.add_condition(C, EveryNCalls(A, 1))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 3, time_scale=TimeScale.TRIAL)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, set([B, C]),
            A, set([B, C]),
            A, set([B, C]),
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_triangle_2(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(A, MappingProjection(), C)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 1))
        sched.add_condition(C, EveryNCalls(A, 2))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 3, time_scale=TimeScale.TRIAL)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, B,
            A, set([B, C]),
            A, B,
            A, set([B, C]),
            A, B,
            A, set([B, C]),
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_triangle_3(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(A, MappingProjection(), C)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, EveryNCalls(A, 3))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 2, time_scale=TimeScale.TRIAL)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, A, B, A, C, A, B, A, A, set([B, C])
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    # this is test 11 of original constraint_scheduler.py
    def test_triangle_4(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')

        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(A, MappingProjection(), C)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, All(WhenFinished(A), AfterNCalls(B, 3)))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 1)
        output = []
        i = 0
        for step in sched.run(termination_conds=termination_conds):
            if i == 3:
                A.is_finished = True
            output.append(step)
            i += 1

        expected_output = [A, A, B, A, A, B, A, A, set([B, C])]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_triangle_4b(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')

        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(A, MappingProjection(), C)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, All(WhenFinished(A), AfterNCalls(B, 3)))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 1)
        output = []
        i = 0
        for step in sched.run(termination_conds=termination_conds):
            if i == 10:
                A.is_finished = True
            output.append(step)
            i += 1

        expected_output = [A, A, B, A, A, B, A, A, B, A, A, set([B, C])]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    #   inverted triangle:           A   B
    #                                 \ /
    #                                  C

    # this is test 4 of original constraint_scheduler.py
    # this test has an implicit priority set of A<B !
    def test_invtriangle_1(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), C)
        comp.add_projection(B, MappingProjection(), C)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, Any(AfterNCalls(A, 3), AfterNCalls(B, 3)))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 4, time_scale=TimeScale.TRIAL)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, set([A, B]), A, C, set([A, B]), C, A, C, set([A, B]), C
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    # this is test 5 of original constraint_scheduler.py
    # this test has an implicit priority set of A<B !
    def test_invtriangle_2(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        for m in [A, B, C]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), C)
        comp.add_projection(B, MappingProjection(), C)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, All(AfterNCalls(A, 3), AfterNCalls(B, 3)))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 2, time_scale=TimeScale.TRIAL)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, set([A, B]), A, set([A, B]), A, set([A, B]), C, A, C
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    #   checkmark:                   A
    #                                 \
    #                                  B   C
    #                                   \ /
    #                                    D

    # testing toposort
    def test_checkmark_1(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        D = TransferMechanism(function=Linear(intercept=.5), name='D')
        for m in [A, B, C, D]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(B, MappingProjection(), D)
        comp.add_projection(C, MappingProjection(), D)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, Always())
        sched.add_condition(B, Always())
        sched.add_condition(C, Always())
        sched.add_condition(D, Always())

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(D, 1, time_scale=TimeScale.TRIAL)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            set([A, C]), B, D
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_checkmark_2(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        D = TransferMechanism(function=Linear(intercept=.5), name='D')
        for m in [A, B, C, D]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(B, MappingProjection(), D)
        comp.add_projection(C, MappingProjection(), D)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, EveryNCalls(A, 2))
        sched.add_condition(D, All(EveryNCalls(B, 2), EveryNCalls(C, 2)))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(D, 1, time_scale=TimeScale.TRIAL)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, set([A, C]), B, A, set([A, C]), B, D
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_checkmark2_1(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='C')
        D = TransferMechanism(function=Linear(intercept=.5), name='D')
        for m in [A, B, C, D]:
            comp.add_mechanism(m)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(A, MappingProjection(), D)
        comp.add_projection(B, MappingProjection(), D)
        comp.add_projection(C, MappingProjection(), D)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, EveryNCalls(A, 2))
        sched.add_condition(D, All(EveryNCalls(B, 2), EveryNCalls(C, 2)))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(D, 1, time_scale=TimeScale.TRIAL)
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            A, set([A, C]), B, A, set([A, C]), B, D
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    #   multi source:                   A1    A2
    #                                   / \  / \
    #                                  B1  B2  B3
    #                                   \ /  \ /
    #                                    C1   C2
    def test_multisource_1(self):
        comp = Composition()
        A1 = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A1')
        A2 = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A2')
        B1 = TransferMechanism(function=Linear(intercept=4.0), name='B1')
        B2 = TransferMechanism(function=Linear(intercept=4.0), name='B2')
        B3 = TransferMechanism(function=Linear(intercept=4.0), name='B3')
        C1 = TransferMechanism(function=Linear(intercept=1.5), name='C1')
        C2 = TransferMechanism(function=Linear(intercept=.5), name='C2')
        for m in [A1, A2, B1, B2, B3, C1, C2]:
            comp.add_mechanism(m)
        comp.add_projection(A1, MappingProjection(), B1)
        comp.add_projection(A1, MappingProjection(), B2)
        comp.add_projection(A2, MappingProjection(), B1)
        comp.add_projection(A2, MappingProjection(), B2)
        comp.add_projection(A2, MappingProjection(), B3)
        comp.add_projection(B1, MappingProjection(), C1)
        comp.add_projection(B2, MappingProjection(), C1)
        comp.add_projection(B1, MappingProjection(), C2)
        comp.add_projection(B3, MappingProjection(), C2)

        sched = Scheduler(composition=comp)

        for m in comp.mechanisms:
            sched.add_condition(m, Always())

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = All(AfterNCalls(C1, 1), AfterNCalls(C2, 1))
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            set([A1, A2]), set([B1, B2, B3]), set([C1, C2])
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_multisource_2(self):
        comp = Composition()
        A1 = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A1')
        A2 = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A2')
        B1 = TransferMechanism(function=Linear(intercept=4.0), name='B1')
        B2 = TransferMechanism(function=Linear(intercept=4.0), name='B2')
        B3 = TransferMechanism(function=Linear(intercept=4.0), name='B3')
        C1 = TransferMechanism(function=Linear(intercept=1.5), name='C1')
        C2 = TransferMechanism(function=Linear(intercept=.5), name='C2')
        for m in [A1, A2, B1, B2, B3, C1, C2]:
            comp.add_mechanism(m)
        comp.add_projection(A1, MappingProjection(), B1)
        comp.add_projection(A1, MappingProjection(), B2)
        comp.add_projection(A2, MappingProjection(), B1)
        comp.add_projection(A2, MappingProjection(), B2)
        comp.add_projection(A2, MappingProjection(), B3)
        comp.add_projection(B1, MappingProjection(), C1)
        comp.add_projection(B2, MappingProjection(), C1)
        comp.add_projection(B1, MappingProjection(), C2)
        comp.add_projection(B3, MappingProjection(), C2)

        sched = Scheduler(composition=comp)

        sched.add_condition_set({
            A1: Always(),
            A2: Always(),
            B1: EveryNCalls(A1, 2),
            B3: EveryNCalls(A2, 2),
            B2: All(EveryNCalls(A1, 4), EveryNCalls(A2, 4)),
            C1: Any(AfterNCalls(B1, 2), AfterNCalls(B2, 2)),
            C2: Any(AfterNCalls(B2, 2), AfterNCalls(B3, 2)),
        })

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = All(AfterNCalls(C1, 1), AfterNCalls(C2, 1))
        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [
            set([A1, A2]), set([A1, A2]), set([B1, B3]), set([A1, A2]), set([A1, A2]), set([B1, B2, B3]), set([C1, C2])
        ]
        assert output == pytest.helpers.setify_expected_output(expected_output)
