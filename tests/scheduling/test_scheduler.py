import logging
import numpy as np
import psyneulink as pnl
import pytest

from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import DriftDiffusionIntegrator
from psyneulink.core.components.functions.transferfunctions import Linear
from psyneulink.core.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.compositions.composition import Composition, EdgeType
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
        C = Composition(pathways=[D])

        D.set_log_conditions(VALUE)

        def change_termination_processing():
            if C.termination_processing is None:
                C.scheduler.termination_conds = {TimeScale.TRIAL: WhenFinished(D)}
                C.termination_processing = {TimeScale.TRIAL: WhenFinished(D)}
            elif isinstance(C.termination_processing[TimeScale.TRIAL], AllHaveRun):
                C.scheduler.termination_conds = {TimeScale.TRIAL: WhenFinished(D)}
                C.termination_processing = {TimeScale.TRIAL: WhenFinished(D)}
            else:
                C.scheduler.termination_conds = {TimeScale.TRIAL: AllHaveRun()}
                C.termination_processing = {TimeScale.TRIAL: AllHaveRun()}

        change_termination_processing()
        C.run(inputs={D: [[1.0], [2.0]]},
              # termination_processing={TimeScale.TRIAL: WhenFinished(D)},
              call_after_trial=change_termination_processing,
              reset_stateful_functions_when=pnl.AtTimeStep(0),
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
        assert np.allclose(expected_results, np.asfarray(C.results))


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


def _get_vertex_feedback_type(graph, sender_port, receiver_mech):
    # there is only one projection per pair
    projection = [
        p for p in sender_port.efferents
        if p.receiver.owner is receiver_mech
    ][0]
    return graph.comp_to_vertex[projection].feedback


def _get_feedback_source_type(graph, sender, receiver):
    try:
        return graph.comp_to_vertex[receiver].source_types[graph.comp_to_vertex[sender]]
    except KeyError:
        return EdgeType.NON_FEEDBACK


class TestFeedback:

    def test_unspecified_feedback(self):
        A = pnl.TransferMechanism(name='A')
        B = pnl.TransferMechanism(name='B')
        C = pnl.ControlMechanism(
            name='C',
            monitor_for_control=B,
            control_signals=[('slope', A)]
        )
        comp = pnl.Composition()
        comp.add_linear_processing_pathway(pathway=[A, B])
        comp.add_node(C)
        comp._analyze_graph()

        assert _get_vertex_feedback_type(comp.graph, A.output_port, B) is EdgeType.NON_FEEDBACK
        assert _get_vertex_feedback_type(comp.graph, B.output_port, C) is EdgeType.NON_FEEDBACK

        assert _get_vertex_feedback_type(comp.graph, C.control_signals[0], A) is EdgeType.FLEXIBLE
        assert _get_feedback_source_type(comp.graph_processing, C, A) is EdgeType.FEEDBACK

    @pytest.mark.parametrize(
        'terminal_mech',
        [
            pnl.TransferMechanism,
            pnl.RecurrentTransferMechanism
        ]
    )
    def test_inline_control_acyclic(self, terminal_mech):
        terminal_mech = terminal_mech(name='terminal_mech')
        A = pnl.TransferMechanism(name='A')
        C = pnl.ControlMechanism(
            name='C',
            monitor_for_control=A,
            control_signals=[('slope', terminal_mech)]
        )
        comp = pnl.Composition()
        comp.add_linear_processing_pathway(pathway=[A, terminal_mech])
        comp.add_nodes([C, terminal_mech])
        comp._analyze_graph()

        # "is" comparisons because MAYBE can be assigned to feedback
        assert _get_vertex_feedback_type(comp.graph, A.output_port, terminal_mech) is EdgeType.NON_FEEDBACK

        assert _get_vertex_feedback_type(comp.graph, C.control_signals[0], terminal_mech) is EdgeType.FLEXIBLE
        assert _get_feedback_source_type(comp.graph_processing, C, A) is EdgeType.NON_FEEDBACK

    # any of the projections in the B, D, E, F cycle may be deleted
    # based on feedback specification. There are individual parametrized
    # tests for each scenario
    #    A -> B -> C
    #        ^  \
    #       /    v
    #      F      D
    #      ^     /
    #       \  v
    #         E
    @pytest.fixture
    def seven_node_cycle_composition(self):
        A = pnl.TransferMechanism(name='A')
        B = pnl.TransferMechanism(name='B')
        C = pnl.TransferMechanism(name='C')
        D = pnl.TransferMechanism(name='D')
        E = pnl.TransferMechanism(name='E')
        F = pnl.TransferMechanism(name='F')

        comp = Composition()
        comp.add_linear_processing_pathway([A, B, C])
        comp.add_nodes([D, E, F])

        return comp.nodes, comp

    @pytest.mark.parametrize(
        'cycle_feedback_proj_pair',
        [
            '(B, D)',
            '(D, E)',
            '(E, F)',
            '(F, B)',
        ]
    )
    def test_cycle_manual_feedback_projections(
        self,
        seven_node_cycle_composition,
        cycle_feedback_proj_pair
    ):
        [A, B, C, D, E, F], comp = seven_node_cycle_composition
        fb_sender, fb_receiver = eval(cycle_feedback_proj_pair)

        cycle_nodes = [B, D, E, F]
        for s_i in range(len(cycle_nodes)):
            r_i = (s_i + 1) % len(cycle_nodes)

            if (
                cycle_nodes[s_i] is not fb_sender
                or cycle_nodes[r_i] is not fb_receiver
            ):
                comp.add_projection(
                    sender=cycle_nodes[s_i],
                    receiver=cycle_nodes[r_i]
                )

        comp.add_projection(
            sender=fb_sender, receiver=fb_receiver,
            feedback=EdgeType.FLEXIBLE
        )
        comp._analyze_graph()

        for s_i in range(len(cycle_nodes)):
            r_i = (s_i + 1) % len(cycle_nodes)

            if (
                cycle_nodes[s_i] is not fb_sender
                or cycle_nodes[r_i] is not fb_receiver
            ):
                assert (
                    _get_feedback_source_type(
                        comp.graph_processing,
                        cycle_nodes[s_i],
                        cycle_nodes[r_i]
                    )
                    is EdgeType.NON_FEEDBACK
                )

        assert (
            _get_feedback_source_type(
                comp.graph_processing,
                fb_sender,
                fb_receiver
            )
            is EdgeType.FEEDBACK
        )

    @pytest.mark.parametrize(
        'cycle_feedback_proj_pair, expected_dependencies',
        [
            ('(B, D)', '{A: set(), B: {A, F}, C: {B}, D: set(), E: {D}, F: {E}}'),
            ('(D, E)', '{A: set(), B: {A, F}, C: {B}, D: {B}, E: set(), F: {E}}'),
            ('(E, F)', '{A: set(), B: {A, F}, C: {B}, D: {B}, E: {D}, F: set()}'),
            ('(F, B)', '{A: set(), B: {A}, C: {B}, D: {B}, E: {D}, F: {E}}'),
        ]
    )
    def test_cycle_manual_feedback_dependencies(
        self,
        seven_node_cycle_composition,
        cycle_feedback_proj_pair,
        expected_dependencies
    ):
        [A, B, C, D, E, F], comp = seven_node_cycle_composition
        fb_sender, fb_receiver = eval(cycle_feedback_proj_pair)
        expected_dependencies = eval(expected_dependencies)

        cycle_nodes = [B, D, E, F]
        for s_i in range(len(cycle_nodes)):
            r_i = (s_i + 1) % len(cycle_nodes)

            if (
                cycle_nodes[s_i] is not fb_sender
                or cycle_nodes[r_i] is not fb_receiver
            ):
                comp.add_projection(
                    sender=cycle_nodes[s_i],
                    receiver=cycle_nodes[r_i]
                )

        comp.add_projection(
            sender=fb_sender, receiver=fb_receiver,
            feedback=EdgeType.FLEXIBLE
        )
        comp._analyze_graph()

        assert comp.scheduler.dependency_dict == expected_dependencies

    def test_cycle_multiple_acyclic_parents(self):
        A = pnl.TransferMechanism(name='A')
        B = pnl.TransferMechanism(name='B')
        C = pnl.TransferMechanism(name='C')
        D = pnl.TransferMechanism(name='D')
        E = pnl.TransferMechanism(name='E')

        comp = Composition()
        comp.add_linear_processing_pathway([C, D, E, C])
        comp.add_linear_processing_pathway([A, C])
        comp.add_linear_processing_pathway([B, C])

        expected_dependencies = {
            A: set(),
            B: set(),
            C: {A, B},
            D: {A, B},
            E: {A, B},
        }
        assert comp.scheduler.dependency_dict == expected_dependencies


    def test_objective_and_control(self):
        # taken from test_3_mechanisms_2_origins_1_additive_control_1_terminal
        comp = pnl.Composition()
        B = pnl.TransferMechanism(name="B", function=pnl.Linear(slope=5.0))
        C = pnl.TransferMechanism(name="C", function=pnl.Linear(slope=5.0))
        A = pnl.ObjectiveMechanism(
            function=Linear,
            monitor=[B],
            name="A"
        )
        LC = pnl.LCControlMechanism(
            name="LC",
            modulation=pnl.ADDITIVE,
            modulated_mechanisms=C,
            objective_mechanism=A)

        D = pnl.TransferMechanism(name="D", function=pnl.Linear(slope=5.0))
        comp.add_linear_processing_pathway([B, C, D])
        comp.add_linear_processing_pathway([B, D])
        comp.add_node(A)
        comp.add_node(LC)

        expected_dependencies = {
            B: set(),
            A: {B},
            LC: {A},
            C: set([LC, B]),
            D: set([C, B])
        }
        assert comp.scheduler.dependency_dict == expected_dependencies

    def test_inline_control_mechanism_example(self):
        cueInterval = pnl.TransferMechanism(
            default_variable=[[0.0]],
            size=1,
            function=pnl.Linear(slope=1, intercept=0),
            output_ports=[pnl.RESULT],
            name='Cue-Stimulus Interval'
        )
        taskLayer = pnl.TransferMechanism(
            default_variable=[[0.0, 0.0]],
            size=2,
            function=pnl.Linear(slope=1, intercept=0),
            output_ports=[pnl.RESULT],
            name='Task Input [I1, I2]'
        )
        activation = pnl.LCAMechanism(
            default_variable=[[0.0, 0.0]],
            size=2,
            function=pnl.Logistic(gain=1),
            leak=.5,
            competition=2,
            noise=0,
            time_step_size=.1,
            termination_measure=pnl.TimeScale.TRIAL,
            termination_threshold=3,
            name='Task Activations [Act 1, Act 2]'
        )
        csiController = pnl.ControlMechanism(
            name='Control Mechanism',
            monitor_for_control=cueInterval,
            control_signals=[(pnl.TERMINATION_THRESHOLD, activation)],
            modulation=pnl.OVERRIDE
        )
        comp = pnl.Composition()
        comp.add_linear_processing_pathway(pathway=[taskLayer, activation])
        comp.add_node(cueInterval)
        comp.add_node(csiController)

        expected_dependencies = {
            cueInterval: set(),
            taskLayer: set(),
            activation: set([csiController, taskLayer]),
            csiController: set([cueInterval])
        }
        assert comp.scheduler.dependency_dict == expected_dependencies

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.parametrize('mode', ['Python',
                                      # 'LLVM' mode is not supported
                                      # the comparison values and checks
                                      # are not synced between binary
                                      # and Python structures
                                      pytest.param('LLVMExec', marks=pytest.mark.llvm),
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                      pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                      pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                     ])
    @pytest.mark.parametrize('timescale, expected',
                             [(TimeScale.TIME_STEP, [[0.5], [0.4375]]),
                              (TimeScale.PASS, [[0.5], [0.4375]]),
                              (TimeScale.TRIAL, [[1.5], [0.4375]]),
                              (TimeScale.RUN, [[1.5], [0.4375]])],
                              ids=lambda x: x if isinstance(x, TimeScale) else "")
    def test_time_termination_measures(self, mode, timescale, expected):
        in_one_pass = timescale in {TimeScale.TIME_STEP, TimeScale.PASS}
        attention = pnl.TransferMechanism(name='Attention',
                                 integrator_mode=True,
                                 termination_threshold=3,
                                 termination_measure=timescale,
                                 execute_until_finished=in_one_pass)
        counter = pnl.IntegratorMechanism(
            function=pnl.AdaptiveIntegrator(rate=0.0, offset=1.0))

        response = pnl.IntegratorMechanism(
            function=pnl.AdaptiveIntegrator(rate=0.5))

        comp = Composition()
        comp.add_linear_processing_pathway([counter, response])
        comp.add_node(attention)
        comp.scheduler.add_condition(response, pnl.WhenFinished(attention))
        comp.scheduler.add_condition(counter, pnl.Always())
        inputs = {attention: [[0.5]], counter: [[2.0]]}
        result = comp.run(inputs=inputs, bin_execute=mode)
        if mode == 'Python':
            assert attention.execution_count == 3
            assert counter.execution_count == 1 if in_one_pass else 3
            assert response.execution_count == 1
        assert np.allclose(result, expected)
