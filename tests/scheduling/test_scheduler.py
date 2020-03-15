import logging
import numpy as np
import pytest

from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import DriftDiffusionIntegrator
from psyneulink.core.components.functions.transferfunctions import Linear
from psyneulink.core.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.process import Process
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.system import System
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.globals.context import Context
from psyneulink.core.globals.keywords import VALUE
from psyneulink.core.scheduling.condition import AfterNCalls, AfterNPasses, AfterNTrials, AfterPass, All, AllHaveRun, Always, Any, AtPass, BeforeNCalls, BeforePass, \
    EveryNCalls, EveryNPasses, JustRan, WhenFinished
from psyneulink.core.scheduling.scheduler import Scheduler
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.library.components.mechanisms.processing.integrator.ddm import DDM

logger = logging.getLogger(__name__)


class TestScheduler:
    @classmethod
    def setup_class(self):
        self.orig_is_finished_flag = TransferMechanism.is_finished_flag
        self.orig_is_finished = TransferMechanism.is_finished
        TransferMechanism.is_finished_flag = True
        TransferMechanism.is_finished = lambda self, context: self.is_finished_flag

    @classmethod
    def teardown_class(self):
        del TransferMechanism.is_finished_flag
        del TransferMechanism.is_finished
        TransferMechanism.is_finished_flag = self.orig_is_finished_flag
        TransferMechanism.is_finished = self.orig_is_finished

    def test_copy(self):
        pass

    def test_deepcopy(self):
        pass

    def test_create_multiple_contexts(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        comp.add_node(A)

        comp.scheduler.clock._increment_time(TimeScale.TRIAL)

        eid = 'eid'
        eid1 = 'eid1'
        comp.scheduler._init_counts(execution_id=eid)

        assert comp.scheduler.clocks[eid].time.trial == 0

        comp.scheduler.clock._increment_time(TimeScale.TRIAL)

        assert comp.scheduler.clocks[eid].time.trial == 0

        comp.scheduler._init_counts(execution_id=eid1, base_execution_id=comp.scheduler.default_execution_id)

        assert comp.scheduler.clocks[eid1].time.trial == 2

    def test_two_compositions_one_scheduler(self):
        comp1 = Composition()
        comp2 = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        comp1.add_node(A)
        comp2.add_node(A)

        sched = Scheduler(composition=comp1)

        sched.add_condition(A, BeforeNCalls(A, 5, time_scale=TimeScale.LIFE))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(6)
        termination_conds[TimeScale.TRIAL] = AfterNPasses(1)
        comp1.run(
            inputs={A: [[0], [1], [2], [3], [4], [5]]},
            scheduler=sched,
            termination_processing=termination_conds
        )
        output = sched.execution_list[comp1.default_execution_id]

        expected_output = [
            A, A, A, A, A, set()
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

        comp2.run(
            inputs={A: [[0], [1], [2], [3], [4], [5]]},
            scheduler=sched,
            termination_processing=termination_conds
        )
        output = sched.execution_list[comp2.default_execution_id]

        expected_output = [
            A, A, A, A, A, set()
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_one_composition_two_contexts(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        comp.add_node(A)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, BeforeNCalls(A, 5, time_scale=TimeScale.LIFE))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(6)
        termination_conds[TimeScale.TRIAL] = AfterNPasses(1)
        eid = 'eid'
        comp.run(
            inputs={A: [[0], [1], [2], [3], [4], [5]]},
            scheduler=sched,
            termination_processing=termination_conds,
            context=eid,
        )
        output = sched.execution_list[eid]

        expected_output = [
            A, A, A, A, A, set()
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

        comp.run(
            inputs={A: [[0], [1], [2], [3], [4], [5]]},
            scheduler=sched,
            termination_processing=termination_conds,
            context=eid,
        )
        output = sched.execution_list[eid]

        expected_output = [
            A, A, A, A, A, set(), set(), set(), set(), set(), set(), set()
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

        eid = 'eid1'
        comp.run(
            inputs={A: [[0], [1], [2], [3], [4], [5]]},
            scheduler=sched,
            termination_processing=termination_conds,
            context=eid,
        )
        output = sched.execution_list[eid]

        expected_output = [
            A, A, A, A, A, set()
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_change_termination_condition(self):
        D = DDM(function=DriftDiffusionIntegrator(threshold=10))
        P = Process(pathway=[D])
        S = System(processes=[P])

        D.set_log_conditions(VALUE)

        def change_termination_processing():
            if S.termination_processing is None:
                S.scheduler.termination_conds = {TimeScale.TRIAL: WhenFinished(D)}
                S.termination_processing = {TimeScale.TRIAL: WhenFinished(D)}
            elif isinstance(S.termination_processing[TimeScale.TRIAL], AllHaveRun):
                S.scheduler.termination_conds = {TimeScale.TRIAL: WhenFinished(D)}
                S.termination_processing = {TimeScale.TRIAL: WhenFinished(D)}
            else:
                S.scheduler.termination_conds = {TimeScale.TRIAL: AllHaveRun()}
                S.termination_processing = {TimeScale.TRIAL: AllHaveRun()}

        change_termination_processing()
        S.run(inputs={D: [[1.0], [2.0]]},
              # termination_processing={TimeScale.TRIAL: WhenFinished(D)},
              call_after_trial=change_termination_processing,
              num_trials=4)
        # Trial 0:
        # input = 1.0, termination condition = WhenFinished
        # 10 passes (value = 1.0, 2.0 ... 9.0, 10.0)
        # Trial 1:
        # input = 2.0, termination condition = AllHaveRun
        # 1 pass (value = 2.0)
        expected_results = [[np.array([[10.]]), np.array([[10.]])],
                            [np.array([[2.]]), np.array([[1.]])],
                            [np.array([[10.]]), np.array([[10.]])],
                            [np.array([[2.]]), np.array([[1.]])]]
        assert np.allclose(expected_results, np.asfarray(S.results))


class TestLinear:

    @classmethod
    def setup_class(self):
        self.orig_is_finished_flag = TransferMechanism.is_finished_flag
        self.orig_is_finished = TransferMechanism.is_finished
        TransferMechanism.is_finished_flag = True
        TransferMechanism.is_finished = lambda self, context: self.is_finished_flag

    @classmethod
    def teardown_class(self):
        del TransferMechanism.is_finished_flag
        del TransferMechanism.is_finished
        TransferMechanism.is_finished_flag = self.orig_is_finished_flag
        TransferMechanism.is_finished = self.orig_is_finished

    def test_no_termination_conds(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)
        comp.add_projection(MappingProjection(), B, C)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)
        comp.add_projection(MappingProjection(), B, C)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)
        comp.add_projection(MappingProjection(), B, C)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)
        comp.add_projection(MappingProjection(), B, C)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)
        comp.add_projection(MappingProjection(), B, C)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)
        comp.add_projection(MappingProjection(), B, C)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)
        comp.add_projection(MappingProjection(), B, C)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, BeforePass(5))
        sched.add_condition(B, AfterNCalls(A, 5))
        sched.add_condition(C, AfterNCalls(B, 1))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(2)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 3)
        comp.run(
                inputs={A: [[0], [1], [2], [3], [4], [5]]},
                scheduler=sched,
                termination_processing=termination_conds
        )
        output = sched.execution_list[comp.default_execution_id]

        expected_output = [
            A, A, A, A, A, B, C, B, C, B, C,
            A, A, A, A, A, B, C, B, C, B, C
        ]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_7(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, WhenFinished(A))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(B, 2)

        output = []
        i = 0
        A.is_finished_flag = False
        for step in sched.run(termination_conds=termination_conds):
            if i == 3:
                A.is_finished_flag = True
            output.append(step)
            i += 1

        expected_output = [A, A, A, A, B, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_9b(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        A.is_finished_flag = False
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        A.is_finished_flag = True
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')

        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        A.is_finished_flag = False
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')

        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        A.is_finished_flag = True
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')

        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        A.is_finished_flag = False
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')

        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)
        comp.add_projection(MappingProjection(), B, C)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)
        comp.add_projection(MappingProjection(), B, C)

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
    @classmethod
    def setup_class(self):
        self.orig_is_finished_flag = TransferMechanism.is_finished_flag
        self.orig_is_finished = TransferMechanism.is_finished
        TransferMechanism.is_finished_flag = True
        TransferMechanism.is_finished = lambda self, context: self.is_finished_flag

    @classmethod
    def teardown_class(self):
        del TransferMechanism.is_finished_flag
        del TransferMechanism.is_finished
        TransferMechanism.is_finished_flag = self.orig_is_finished_flag
        TransferMechanism.is_finished = self.orig_is_finished

    #   triangle:         A
    #                    / \
    #                   B   C

    def test_triangle_1(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)
        comp.add_projection(MappingProjection(), A, C)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)
        comp.add_projection(MappingProjection(), A, C)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)
        comp.add_projection(MappingProjection(), A, C)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='scheduler-pytests-C')

        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)
        comp.add_projection(MappingProjection(), A, C)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, All(WhenFinished(A), AfterNCalls(B, 3)))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 1)
        output = []
        i = 0
        A.is_finished_flag = False
        for step in sched.run(termination_conds=termination_conds):
            if i == 3:
                A.is_finished_flag = True
            output.append(step)
            i += 1

        expected_output = [A, A, B, A, A, B, A, A, set([B, C])]
        # pprint.pprint(output)
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_triangle_4b(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='scheduler-pytests-C')

        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)
        comp.add_projection(MappingProjection(), A, C)

        sched = Scheduler(composition=comp)

        sched.add_condition(A, EveryNPasses(1))
        sched.add_condition(B, EveryNCalls(A, 2))
        sched.add_condition(C, All(WhenFinished(A), AfterNCalls(B, 3)))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(C, 1)
        output = []
        i = 0
        A.is_finished_flag = False
        for step in sched.run(termination_conds=termination_conds):
            if i == 10:
                A.is_finished_flag = True
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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, C)
        comp.add_projection(MappingProjection(), B, C)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='scheduler-pytests-C')
        for m in [A, B, C]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, C)
        comp.add_projection(MappingProjection(), B, C)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='scheduler-pytests-C')
        D = TransferMechanism(function=Linear(intercept=.5), name='scheduler-pytests-D')
        for m in [A, B, C, D]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)
        comp.add_projection(MappingProjection(), B, D)
        comp.add_projection(MappingProjection(), C, D)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='scheduler-pytests-C')
        D = TransferMechanism(function=Linear(intercept=.5), name='scheduler-pytests-D')
        for m in [A, B, C, D]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)
        comp.add_projection(MappingProjection(), B, D)
        comp.add_projection(MappingProjection(), C, D)

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
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        C = TransferMechanism(function=Linear(intercept=1.5), name='scheduler-pytests-C')
        D = TransferMechanism(function=Linear(intercept=.5), name='scheduler-pytests-D')
        for m in [A, B, C, D]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)
        comp.add_projection(MappingProjection(), A, D)
        comp.add_projection(MappingProjection(), B, D)
        comp.add_projection(MappingProjection(), C, D)

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
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A1, B1)
        comp.add_projection(MappingProjection(), A1, B2)
        comp.add_projection(MappingProjection(), A2, B1)
        comp.add_projection(MappingProjection(), A2, B2)
        comp.add_projection(MappingProjection(), A2, B3)
        comp.add_projection(MappingProjection(), B1, C1)
        comp.add_projection(MappingProjection(), B2, C1)
        comp.add_projection(MappingProjection(), B1, C2)
        comp.add_projection(MappingProjection(), B3, C2)

        sched = Scheduler(composition=comp)

        for m in comp.nodes:
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
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A1, B1)
        comp.add_projection(MappingProjection(), A1, B2)
        comp.add_projection(MappingProjection(), A2, B1)
        comp.add_projection(MappingProjection(), A2, B2)
        comp.add_projection(MappingProjection(), A2, B3)
        comp.add_projection(MappingProjection(), B1, C1)
        comp.add_projection(MappingProjection(), B2, C1)
        comp.add_projection(MappingProjection(), B1, C2)
        comp.add_projection(MappingProjection(), B3, C2)

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


class TestTermination:

    @classmethod
    def setup_class(self):
        self.orig_is_finished_flag = TransferMechanism.is_finished_flag
        self.orig_is_finished = TransferMechanism.is_finished
        TransferMechanism.is_finished_flag = True
        TransferMechanism.is_finished = lambda self, context: self.is_finished_flag

    @classmethod
    def teardown_class(self):
        del TransferMechanism.is_finished_flag
        del TransferMechanism.is_finished
        TransferMechanism.is_finished_flag = self.orig_is_finished_flag
        TransferMechanism.is_finished = self.orig_is_finished

    def test_termination_conditions_reset(self):
        comp = Composition()
        A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='scheduler-pytests-A')
        B = TransferMechanism(function=Linear(intercept=4.0), name='scheduler-pytests-B')
        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)

        sched = Scheduler(composition=comp)

        sched.add_condition(B, EveryNCalls(A, 2))

        termination_conds = {}
        termination_conds[TimeScale.RUN] = AfterNTrials(1)
        termination_conds[TimeScale.TRIAL] = AfterNCalls(B, 2)

        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, B, A, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

        # reset the RUN because schedulers run TRIALs
        sched.clock._increment_time(TimeScale.RUN)
        sched._reset_counts_total(TimeScale.RUN, execution_id=sched.default_execution_id)

        output = list(sched.run())

        expected_output = [A, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_partial_override_scheduler(self):
        comp = Composition()
        A = TransferMechanism(name='scheduler-pytests-A')
        B = TransferMechanism(name='scheduler-pytests-B')
        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)

        sched = Scheduler(composition=comp)
        sched.add_condition(B, EveryNCalls(A, 2))
        termination_conds = {TimeScale.TRIAL: AfterNCalls(B, 2)}

        output = list(sched.run(termination_conds=termination_conds))

        expected_output = [A, A, B, A, A, B]
        assert output == pytest.helpers.setify_expected_output(expected_output)

    def test_partial_override_composition(self):
        comp = Composition()
        A = TransferMechanism(name='scheduler-pytests-A')
        B = IntegratorMechanism(name='scheduler-pytests-B')
        for m in [A, B]:
            comp.add_node(m)
        comp.add_projection(MappingProjection(), A, B)

        termination_conds = {TimeScale.TRIAL: AfterNCalls(B, 2)}

        output = comp.run(inputs={A: 1}, termination_processing=termination_conds)
        # two executions of B
        assert output == [.75]
