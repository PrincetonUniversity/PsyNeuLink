import functools
import logging
from timeit import timeit

import numpy as np
import pytest

from psyneulink.components.functions.function import Linear, SimpleIntegrator
from psyneulink.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism, TRANSFER_OUTPUT
from psyneulink.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.library.mechanisms.processing.transfer.recurrenttransfermechanism import RecurrentTransferMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.states.inputstate import InputState
from psyneulink.compositions.composition import Composition, CompositionError, MechanismRole
from psyneulink.compositions.pathwaycomposition import PathwayComposition
from psyneulink.compositions.systemcomposition import SystemComposition
from psyneulink.scheduling.condition import EveryNCalls
from psyneulink.scheduling.scheduler import Scheduler
from psyneulink.scheduling.condition import EveryNPasses, AfterNCalls
from psyneulink.scheduling.time import TimeScale
from psyneulink.globals.keywords import NAME, INPUT_STATE, HARD_CLAMP, SOFT_CLAMP, NO_CLAMP, PULSE_CLAMP

logger = logging.getLogger(__name__)

# All tests are set to run. If you need to skip certain tests,
# see http://doc.pytest.org/en/latest/skipping.html


# Unit tests for each function of the Composition class #######################
# Unit tests for Composition.Composition(

class TestConstructor:

    def test_no_args(self):
        comp = Composition()
        assert isinstance(comp, Composition)

    def test_two_calls_no_args(self):
        comp = Composition()
        assert isinstance(comp, Composition)

        comp_2 = Composition()
        assert isinstance(comp, Composition)

    @pytest.mark.stress
    @pytest.mark.parametrize(
        'count', [
            10000,
        ]
    )
    def test_timing_no_args(self, count):
        t = timeit('comp = Composition()', setup='from psyneulink.compositions.composition import Composition', number=count)
        print()
        logger.info('completed {0} creation{2} of Composition() in {1:.8f}s'.format(count, t, 's' if count != 1 else ''))


class TestAddMechanism:

    def test_add_once(self):
        comp = Composition()
        comp.add_mechanism(TransferMechanism())

    def test_add_twice(self):
        comp = Composition()
        comp.add_mechanism(TransferMechanism())
        comp.add_mechanism(TransferMechanism())

    def test_add_same_twice(self):
        comp = Composition()
        mech = TransferMechanism()
        comp.add_mechanism(mech)
        comp.add_mechanism(mech)

    @pytest.mark.stress
    @pytest.mark.parametrize(
        'count', [
            100,
        ]
    )
    def test_timing_stress(self, count):
        t = timeit(
            'comp.add_mechanism(TransferMechanism())',
            setup='''

from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.compositions.composition import Composition
comp = Composition()
''',
            number=count
        )
        print()
        logger.info('completed {0} addition{2} of a Mechanism to a Composition in {1:.8f}s'.
                    format(count, t, 's' if count != 1 else ''))


class TestAddProjection:

    def test_add_once(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)

    def test_add_twice(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(A, MappingProjection(), B)

    def test_add_same_twice(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        proj = MappingProjection()
        comp.add_projection(A, proj, B)
        comp.add_projection(A, proj, B)

    @pytest.mark.stress
    @pytest.mark.parametrize(
        'count', [
            1000,
        ]
    )
    def test_timing_stress(self, count):
        t = timeit('comp.add_projection(A, MappingProjection(), B)',
                   setup='''

from psyneulink.components.mechanisms.processingmechanisms.transfermechanism import TransferMechanism
from psyneulink.components.projections.pathwayprojections.mappingprojection import MappingProjection
from psyneulink.compositions.composition import Composition

comp = Composition()
A = TransferMechanism(name='composition-pytests-A')
B = TransferMechanism(name='composition-pytests-B')
comp.add_mechanism(A)
comp.add_mechanism(B)
''',
                   number=count
                   )
        print()
        logger.info('completed {0} addition{2} of a projection to a composition in {1:.8f}s'.format(count, t, 's' if count != 1 else ''))

    @pytest.mark.stress
    @pytest.mark.parametrize(
        'count', [
            1000,
        ]
    )
    def test_timing_stress(self, count):
        t = timeit('comp.add_projection(A, MappingProjection(), B)',
                   setup='''
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.compositions.composition import Composition
comp = Composition()
A = TransferMechanism(name='composition-pytests-A')
B = TransferMechanism(name='composition-pytests-B')
comp.add_mechanism(A)
comp.add_mechanism(B)
''',
                   number=count
                   )
        print()
        logger.info('completed {0} addition{2} of a projection to a composition in {1:.8f}s'.format(count, t, 's' if count != 1 else ''))


class TestAnalyzeGraph:

    def test_empty_call(self):
        comp = Composition()
        comp._analyze_graph()

    def test_singleton(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        comp.add_mechanism(A)
        comp._analyze_graph()
        assert A in comp.get_mechanisms_by_role(MechanismRole.ORIGIN)
        assert A in comp.get_mechanisms_by_role(MechanismRole.TERMINAL)

    def test_two_independent(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp._analyze_graph()
        assert A in comp.get_mechanisms_by_role(MechanismRole.ORIGIN)
        assert B in comp.get_mechanisms_by_role(MechanismRole.ORIGIN)
        assert A in comp.get_mechanisms_by_role(MechanismRole.TERMINAL)
        assert B in comp.get_mechanisms_by_role(MechanismRole.TERMINAL)

    def test_two_in_a_row(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp._analyze_graph()
        assert A in comp.get_mechanisms_by_role(MechanismRole.ORIGIN)
        assert B not in comp.get_mechanisms_by_role(MechanismRole.ORIGIN)
        assert A not in comp.get_mechanisms_by_role(MechanismRole.TERMINAL)
        assert B in comp.get_mechanisms_by_role(MechanismRole.TERMINAL)

    # (A)<->(B)
    def test_two_recursive(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(B, MappingProjection(), A)
        comp._analyze_graph()
        assert A not in comp.get_mechanisms_by_role(MechanismRole.ORIGIN)
        assert B not in comp.get_mechanisms_by_role(MechanismRole.ORIGIN)
        assert A not in comp.get_mechanisms_by_role(MechanismRole.TERMINAL)
        assert B not in comp.get_mechanisms_by_role(MechanismRole.TERMINAL)
        assert A in comp.get_mechanisms_by_role(MechanismRole.CYCLE)
        assert B in comp.get_mechanisms_by_role(MechanismRole.RECURRENT_INIT)

    # (A)->(B)<->(C)<-(D)
    @pytest.mark.skip
    def test_two_origins_pointing_to_recursive_pair(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B')
        C = TransferMechanism(name='composition-pytests-C')
        D = TransferMechanism(name='composition-pytests-D')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_mechanism(C)
        comp.add_mechanism(D)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(C, MappingProjection(), B)
        comp.add_projection(B, MappingProjection(), C)
        comp.add_projection(D, MappingProjection(), C)
        comp._analyze_graph()
        assert A in comp.get_mechanisms_by_role(MechanismRole.ORIGIN)
        assert D in comp.get_mechanisms_by_role(MechanismRole.ORIGIN)
        assert B in comp.get_mechanisms_by_role(MechanismRole.CYCLE)
        assert C in comp.get_mechanisms_by_role(MechanismRole.RECURRENT_INIT)


class TestValidateFeedDict:

    def test_empty_feed_dicts(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp._analyze_graph()
        feed_dict_origin = {}
        feed_dict_terminal = {}
        comp._validate_feed_dict(feed_dict_origin, comp.get_mechanisms_by_role(MechanismRole.ORIGIN), "origin")
        comp._validate_feed_dict(feed_dict_terminal, comp.get_mechanisms_by_role(MechanismRole.TERMINAL), "terminal")

    def test_origin_and_terminal_with_mapping(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp._analyze_graph()
        feed_dict_origin = {A: [[0]]}
        feed_dict_terminal = {B: [[0]]}
        comp._validate_feed_dict(feed_dict_origin, comp.get_mechanisms_by_role(MechanismRole.ORIGIN), "origin")
        comp._validate_feed_dict(feed_dict_terminal, comp.get_mechanisms_by_role(MechanismRole.TERMINAL), "terminal")

    def test_origin_and_terminal_with_swapped_feed_dicts_1(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp._analyze_graph()
        feed_dict_origin = {B: [[0]]}
        feed_dict_terminal = {A: [[0]]}
        with pytest.raises(ValueError):
            comp._validate_feed_dict(feed_dict_origin, comp.get_mechanisms_by_role(MechanismRole.ORIGIN), "origin")

    def test_origin_and_terminal_with_swapped_feed_dicts_2(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp._analyze_graph()
        feed_dict_origin = {B: [[0]]}
        feed_dict_terminal = {A: [[0]]}
        with pytest.raises(ValueError):
            comp._validate_feed_dict(feed_dict_terminal, comp.get_mechanisms_by_role(MechanismRole.TERMINAL), "terminal")

    def test_multiple_origin_mechs(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B')
        C = TransferMechanism(name='composition-pytests-C')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_mechanism(C)
        comp.add_projection(A, MappingProjection(), C)
        comp.add_projection(B, MappingProjection(), C)
        comp._analyze_graph()
        feed_dict_origin = {A: [[0]], B: [[0]]}
        feed_dict_terminal = {C: [[0]]}
        comp._validate_feed_dict(feed_dict_origin, comp.get_mechanisms_by_role(MechanismRole.ORIGIN), "origin")
        comp._validate_feed_dict(feed_dict_terminal, comp.get_mechanisms_by_role(MechanismRole.TERMINAL), "terminal")

    def test_multiple_origin_mechs_only_one_in_feed_dict(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B')
        C = TransferMechanism(name='composition-pytests-C')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_mechanism(C)
        comp.add_projection(A, MappingProjection(), C)
        comp.add_projection(B, MappingProjection(), C)
        comp._analyze_graph()
        feed_dict_origin = {B: [[0]]}
        feed_dict_terminal = {C: [[0]]}
        comp._validate_feed_dict(feed_dict_origin, comp.get_mechanisms_by_role(MechanismRole.ORIGIN), "origin")
        comp._validate_feed_dict(feed_dict_terminal, comp.get_mechanisms_by_role(MechanismRole.TERMINAL), "terminal")

    def test_input_state_len_3(self):
        comp = Composition()
        A = TransferMechanism(default_variable=[0, 1, 2], name='composition-pytests-A')
        B = TransferMechanism(default_variable=[0, 1, 2], name='composition-pytests-B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp._analyze_graph()
        feed_dict_origin = {A: [[0, 1, 2]]}
        feed_dict_terminal = {B: [[0, 1, 2]]}
        comp._validate_feed_dict(feed_dict_origin, comp.get_mechanisms_by_role(MechanismRole.ORIGIN), "origin")
        comp._validate_feed_dict(feed_dict_terminal, comp.get_mechanisms_by_role(MechanismRole.TERMINAL), "terminal")

    def test_input_state_len_3_feed_dict_len_2(self):
        comp = Composition()
        A = TransferMechanism(default_variable=[0, 1, 2], name='composition-pytests-A')
        B = TransferMechanism(default_variable=[0, 1, 2], name='composition-pytests-B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp._analyze_graph()
        feed_dict_origin = {A: [[0, 1]]}
        feed_dict_terminal = {B: [[0]]}
        with pytest.raises(ValueError):
            comp._validate_feed_dict(feed_dict_origin, comp.get_mechanisms_by_role(MechanismRole.ORIGIN), "origin")

    def test_input_state_len_2_feed_dict_len_3(self):
        comp = Composition()
        A = TransferMechanism(default_variable=[0, 1], name='composition-pytests-A')
        B = TransferMechanism(default_variable=[0, 1], name='composition-pytests-B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp._analyze_graph()
        feed_dict_origin = {A: [[0, 1, 2]]}
        feed_dict_terminal = {B: [[0]]}
        with pytest.raises(ValueError):
            comp._validate_feed_dict(feed_dict_origin, comp.get_mechanisms_by_role(MechanismRole.ORIGIN), "origin")

    def test_feed_dict_includes_mechs_of_correct_and_incorrect_types(self):
        comp = Composition()
        A = TransferMechanism(default_variable=[0], name='composition-pytests-A')
        B = TransferMechanism(default_variable=[0], name='composition-pytests-B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp._analyze_graph()
        feed_dict_origin = {A: [[0]], B: [[0]]}
        with pytest.raises(ValueError):
            comp._validate_feed_dict(feed_dict_origin, comp.get_mechanisms_by_role(MechanismRole.ORIGIN), "origin")

    def test_input_state_len_3_brackets_extra_1(self):
        comp = Composition()
        A = TransferMechanism(default_variable=[0, 1, 2], name='composition-pytests-A')
        B = TransferMechanism(default_variable=[0, 1, 2], name='composition-pytests-B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp._analyze_graph()
        feed_dict_origin = {A: [[[0, 1, 2]]]}
        feed_dict_terminal = {B: [[[0, 1, 2]]]}
        comp._validate_feed_dict(feed_dict_origin, comp.get_mechanisms_by_role(MechanismRole.ORIGIN), "origin")
        comp._validate_feed_dict(feed_dict_terminal, comp.get_mechanisms_by_role(MechanismRole.TERMINAL), "terminal")

    def test_input_state_len_3_brackets_missing_1(self):
        comp = Composition()
        A = TransferMechanism(default_variable=[0, 1, 2], name='composition-pytests-A')
        B = TransferMechanism(default_variable=[0, 1, 2], name='composition-pytests-B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp._analyze_graph()
        feed_dict_origin = {A:  [0, 1, 2]}
        feed_dict_terminal = {B: [[0]]}
        with pytest.raises(TypeError):
            comp._validate_feed_dict(feed_dict_origin, comp.get_mechanisms_by_role(MechanismRole.ORIGIN), "origin")

    def test_empty_feed_dict_for_empty_type(self):
        comp = Composition()
        A = TransferMechanism(default_variable=[0], name='composition-pytests-A')
        B = TransferMechanism(default_variable=[0], name='composition-pytests-B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp._analyze_graph()
        feed_dict_origin = {A: [[0]]}
        feed_dict_monitored = {}
        comp._validate_feed_dict(feed_dict_monitored, comp.get_mechanisms_by_role(MechanismRole.MONITORED), "monitored")

    def test_mech_in_feed_dict_for_empty_type(self):
        comp = Composition()
        A = TransferMechanism(default_variable=[0])
        B = TransferMechanism(name='composition-pytests-B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp._analyze_graph()
        feed_dict_origin = {A: [[0]]}
        feed_dict_monitored = {B: [[0]]}
        with pytest.raises(ValueError):
            comp._validate_feed_dict(feed_dict_monitored, comp.get_mechanisms_by_role(MechanismRole.MONITORED), "monitored")

    def test_one_mech_1(self):
        comp = Composition()
        A = TransferMechanism(default_variable=[0])
        comp.add_mechanism(A)
        comp._analyze_graph()
        feed_dict_origin = {A: [[0]]}
        feed_dict_terminal = {A: [[0]]}
        comp._validate_feed_dict(feed_dict_origin, comp.get_mechanisms_by_role(MechanismRole.ORIGIN), "origin")

    def test_one_mech_2(self):
        comp = Composition()
        A = TransferMechanism(default_variable=[0])
        comp.add_mechanism(A)
        comp._analyze_graph()
        feed_dict_origin = {A: [[0]]}
        feed_dict_terminal = {A: [[0]]}
        comp._validate_feed_dict(feed_dict_terminal, comp.get_mechanisms_by_role(MechanismRole.TERMINAL), "terminal")

    def test_multiple_time_steps_1(self):
        comp = Composition()
        A = TransferMechanism(default_variable=[[0, 1, 2]], name='composition-pytests-A')
        B = TransferMechanism(default_variable=[[0, 1, 2]], name='composition-pytests-B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp._analyze_graph()
        feed_dict_origin = {A: [[0, 1, 2], [0, 1, 2]]}
        feed_dict_terminal = {B: [[0, 1, 2]]}
        comp._validate_feed_dict(feed_dict_origin, comp.get_mechanisms_by_role(MechanismRole.ORIGIN), "origin")
        comp._validate_feed_dict(feed_dict_terminal, comp.get_mechanisms_by_role(MechanismRole.TERMINAL), "terminal")

    def test_multiple_time_steps_2(self):
        comp = Composition()
        A = TransferMechanism(default_variable=[[0, 1, 2]], name='composition-pytests-A')
        B = TransferMechanism(default_variable=[[0, 1, 2]], name='composition-pytests-B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp._analyze_graph()
        feed_dict_origin = {A: [[[0, 1, 2]], [[0, 1, 2]]]}
        feed_dict_terminal = {B: [[0, 1, 2]]}
        comp._validate_feed_dict(feed_dict_origin, comp.get_mechanisms_by_role(MechanismRole.ORIGIN), "origin")
        comp._validate_feed_dict(feed_dict_terminal, comp.get_mechanisms_by_role(MechanismRole.TERMINAL), "terminal")


class TestGetMechanismsByRole:

    def test_multiple_roles(self):

        comp = Composition()
        mechs = [TransferMechanism() for x in range(4)]

        for mech in mechs:
            comp.add_mechanism(mech)

        comp._add_mechanism_role(mechs[0], MechanismRole.ORIGIN)
        comp._add_mechanism_role(mechs[1], MechanismRole.INTERNAL)
        comp._add_mechanism_role(mechs[2], MechanismRole.INTERNAL)
        comp._add_mechanism_role(mechs[3], MechanismRole.CYCLE)

        for role in list(MechanismRole):
            if role is MechanismRole.ORIGIN:
                assert comp.get_mechanisms_by_role(role) == [mechs[0]]
            elif role is MechanismRole.INTERNAL:
                assert comp.get_mechanisms_by_role(role) == [mechs[1], mechs[2]]
            elif role is MechanismRole.CYCLE:
                assert comp.get_mechanisms_by_role(role) == [mechs[3]]
            else:
                assert comp.get_mechanisms_by_role(role) == []

    def test_nonexistent_role(self):

        comp = Composition()

        with pytest.raises(CompositionError):
            comp.get_mechanisms_by_role(None)


class TestGraph:

    class TestProcessingGraph:

        def test_all_mechanisms(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='composition-pytests-A')
            B = TransferMechanism(function=Linear(intercept=4.0), name='composition-pytests-B')
            C = TransferMechanism(function=Linear(intercept=1.5), name='composition-pytests-C')
            mechs = [A, B, C]
            for m in mechs:
                comp.add_mechanism(m)

            assert len(comp.graph_processing.vertices) == 3
            assert len(comp.graph_processing.comp_to_vertex) == 3
            for m in mechs:
                assert m in comp.graph_processing.comp_to_vertex

            assert comp.graph_processing.get_parents_from_component(A) == []
            assert comp.graph_processing.get_parents_from_component(B) == []
            assert comp.graph_processing.get_parents_from_component(C) == []

            assert comp.graph_processing.get_children_from_component(A) == []
            assert comp.graph_processing.get_children_from_component(B) == []
            assert comp.graph_processing.get_children_from_component(C) == []

        def test_triangle(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='composition-pytests-A')
            B = TransferMechanism(function=Linear(intercept=4.0), name='composition-pytests-B')
            C = TransferMechanism(function=Linear(intercept=1.5), name='composition-pytests-C')
            mechs = [A, B, C]
            for m in mechs:
                comp.add_mechanism(m)
            comp.add_projection(A, MappingProjection(), B)
            comp.add_projection(B, MappingProjection(), C)

            assert len(comp.graph_processing.vertices) == 3
            assert len(comp.graph_processing.comp_to_vertex) == 3
            for m in mechs:
                assert m in comp.graph_processing.comp_to_vertex

            assert comp.graph_processing.get_parents_from_component(A) == []
            assert comp.graph_processing.get_parents_from_component(B) == [comp.graph_processing.comp_to_vertex[A]]
            assert comp.graph_processing.get_parents_from_component(C) == [comp.graph_processing.comp_to_vertex[B]]

            assert comp.graph_processing.get_children_from_component(A) == [comp.graph_processing.comp_to_vertex[B]]
            assert comp.graph_processing.get_children_from_component(B) == [comp.graph_processing.comp_to_vertex[C]]
            assert comp.graph_processing.get_children_from_component(C) == []

        def test_x(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='composition-pytests-A')
            B = TransferMechanism(function=Linear(intercept=4.0), name='composition-pytests-B')
            C = TransferMechanism(function=Linear(intercept=1.5), name='composition-pytests-C')
            D = TransferMechanism(function=Linear(intercept=1.5), name='composition-pytests-D')
            E = TransferMechanism(function=Linear(intercept=1.5), name='composition-pytests-E')
            mechs = [A, B, C, D, E]
            for m in mechs:
                comp.add_mechanism(m)
            comp.add_projection(A, MappingProjection(), C)
            comp.add_projection(B, MappingProjection(), C)
            comp.add_projection(C, MappingProjection(), D)
            comp.add_projection(C, MappingProjection(), E)

            assert len(comp.graph_processing.vertices) == 5
            assert len(comp.graph_processing.comp_to_vertex) == 5
            for m in mechs:
                assert m in comp.graph_processing.comp_to_vertex

            assert comp.graph_processing.get_parents_from_component(A) == []
            assert comp.graph_processing.get_parents_from_component(B) == []
            assert set(comp.graph_processing.get_parents_from_component(C)) == set([
                comp.graph_processing.comp_to_vertex[A],
                comp.graph_processing.comp_to_vertex[B],
            ])
            assert comp.graph_processing.get_parents_from_component(D) == [comp.graph_processing.comp_to_vertex[C]]
            assert comp.graph_processing.get_parents_from_component(E) == [comp.graph_processing.comp_to_vertex[C]]

            assert comp.graph_processing.get_children_from_component(A) == [comp.graph_processing.comp_to_vertex[C]]
            assert comp.graph_processing.get_children_from_component(B) == [comp.graph_processing.comp_to_vertex[C]]
            assert set(comp.graph_processing.get_children_from_component(C)) == set([
                comp.graph_processing.comp_to_vertex[D],
                comp.graph_processing.comp_to_vertex[E],
            ])
            assert comp.graph_processing.get_children_from_component(D) == []
            assert comp.graph_processing.get_children_from_component(E) == []

        def test_cycle_linear(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='composition-pytests-A')
            B = TransferMechanism(function=Linear(intercept=4.0), name='composition-pytests-B')
            C = TransferMechanism(function=Linear(intercept=1.5), name='composition-pytests-C')
            mechs = [A, B, C]
            for m in mechs:
                comp.add_mechanism(m)
            comp.add_projection(A, MappingProjection(), B)
            comp.add_projection(B, MappingProjection(), C)
            comp.add_projection(C, MappingProjection(), A)

            assert len(comp.graph_processing.vertices) == 3
            assert len(comp.graph_processing.comp_to_vertex) == 3
            for m in mechs:
                assert m in comp.graph_processing.comp_to_vertex

            assert comp.graph_processing.get_parents_from_component(A) == [comp.graph_processing.comp_to_vertex[C]]
            assert comp.graph_processing.get_parents_from_component(B) == [comp.graph_processing.comp_to_vertex[A]]
            assert comp.graph_processing.get_parents_from_component(C) == [comp.graph_processing.comp_to_vertex[B]]

            assert comp.graph_processing.get_children_from_component(A) == [comp.graph_processing.comp_to_vertex[B]]
            assert comp.graph_processing.get_children_from_component(B) == [comp.graph_processing.comp_to_vertex[C]]
            assert comp.graph_processing.get_children_from_component(C) == [comp.graph_processing.comp_to_vertex[A]]

        def test_cycle_x(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='composition-pytests-A')
            B = TransferMechanism(function=Linear(intercept=4.0), name='composition-pytests-B')
            C = TransferMechanism(function=Linear(intercept=1.5), name='composition-pytests-C')
            D = TransferMechanism(function=Linear(intercept=1.5), name='composition-pytests-D')
            E = TransferMechanism(function=Linear(intercept=1.5), name='composition-pytests-E')
            mechs = [A, B, C, D, E]
            for m in mechs:
                comp.add_mechanism(m)
            comp.add_projection(A, MappingProjection(), C)
            comp.add_projection(B, MappingProjection(), C)
            comp.add_projection(C, MappingProjection(), D)
            comp.add_projection(C, MappingProjection(), E)
            comp.add_projection(D, MappingProjection(), A)
            comp.add_projection(E, MappingProjection(), B)

            assert len(comp.graph_processing.vertices) == 5
            assert len(comp.graph_processing.comp_to_vertex) == 5
            for m in mechs:
                assert m in comp.graph_processing.comp_to_vertex

            assert comp.graph_processing.get_parents_from_component(A) == [comp.graph_processing.comp_to_vertex[D]]
            assert comp.graph_processing.get_parents_from_component(B) == [comp.graph_processing.comp_to_vertex[E]]
            assert set(comp.graph_processing.get_parents_from_component(C)) == set([
                comp.graph_processing.comp_to_vertex[A],
                comp.graph_processing.comp_to_vertex[B],
            ])
            assert comp.graph_processing.get_parents_from_component(D) == [comp.graph_processing.comp_to_vertex[C]]
            assert comp.graph_processing.get_parents_from_component(E) == [comp.graph_processing.comp_to_vertex[C]]

            assert comp.graph_processing.get_children_from_component(A) == [comp.graph_processing.comp_to_vertex[C]]
            assert comp.graph_processing.get_children_from_component(B) == [comp.graph_processing.comp_to_vertex[C]]
            assert set(comp.graph_processing.get_children_from_component(C)) == set([
                comp.graph_processing.comp_to_vertex[D],
                comp.graph_processing.comp_to_vertex[E],
            ])
            assert comp.graph_processing.get_children_from_component(D) == [comp.graph_processing.comp_to_vertex[A]]
            assert comp.graph_processing.get_children_from_component(E) == [comp.graph_processing.comp_to_vertex[B]]

        def test_cycle_x_multiple_incoming(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='composition-pytests-A')
            B = TransferMechanism(function=Linear(intercept=4.0), name='composition-pytests-B')
            C = TransferMechanism(function=Linear(intercept=1.5), name='composition-pytests-C')
            D = TransferMechanism(function=Linear(intercept=1.5), name='composition-pytests-D')
            E = TransferMechanism(function=Linear(intercept=1.5), name='composition-pytests-E')
            mechs = [A, B, C, D, E]
            for m in mechs:
                comp.add_mechanism(m)
            comp.add_projection(A, MappingProjection(), C)
            comp.add_projection(B, MappingProjection(), C)
            comp.add_projection(C, MappingProjection(), D)
            comp.add_projection(C, MappingProjection(), E)
            comp.add_projection(D, MappingProjection(), A)
            comp.add_projection(D, MappingProjection(), B)
            comp.add_projection(E, MappingProjection(), A)
            comp.add_projection(E, MappingProjection(), B)

            assert len(comp.graph_processing.vertices) == 5
            assert len(comp.graph_processing.comp_to_vertex) == 5
            for m in mechs:
                assert m in comp.graph_processing.comp_to_vertex

            assert set(comp.graph_processing.get_parents_from_component(A)) == set([
                comp.graph_processing.comp_to_vertex[D],
                comp.graph_processing.comp_to_vertex[E],
            ])
            assert set(comp.graph_processing.get_parents_from_component(B)) == set([
                comp.graph_processing.comp_to_vertex[D],
                comp.graph_processing.comp_to_vertex[E],
            ])
            assert set(comp.graph_processing.get_parents_from_component(C)) == set([
                comp.graph_processing.comp_to_vertex[A],
                comp.graph_processing.comp_to_vertex[B],
            ])
            assert comp.graph_processing.get_parents_from_component(D) == [comp.graph_processing.comp_to_vertex[C]]
            assert comp.graph_processing.get_parents_from_component(E) == [comp.graph_processing.comp_to_vertex[C]]

            assert comp.graph_processing.get_children_from_component(A) == [comp.graph_processing.comp_to_vertex[C]]
            assert comp.graph_processing.get_children_from_component(B) == [comp.graph_processing.comp_to_vertex[C]]
            assert set(comp.graph_processing.get_children_from_component(C)) == set([
                comp.graph_processing.comp_to_vertex[D],
                comp.graph_processing.comp_to_vertex[E],
            ])
            assert set(comp.graph_processing.get_children_from_component(D)) == set([
                comp.graph_processing.comp_to_vertex[A],
                comp.graph_processing.comp_to_vertex[B],
            ])
            assert set(comp.graph_processing.get_children_from_component(E)) == set([
                comp.graph_processing.comp_to_vertex[A],
                comp.graph_processing.comp_to_vertex[B],
            ])


class TestRun:

    # def test_run_2_mechanisms_default_input_1(self):
    #     comp = Composition()
    #     A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
    #     B = TransferMechanism(function=Linear(slope=5.0))
    #     comp.add_mechanism(A)
    #     comp.add_mechanism(B)
    #     comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
    #     comp._analyze_graph()
    #     sched = Scheduler(composition=comp)
    #     output = comp.run(
    #         scheduler_processing=sched
    #     )
    #     assert 25 == output[0][0]

    def test_run_2_mechanisms_input_5(self):
        comp = Composition()
        A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
        B = TransferMechanism(function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp._analyze_graph()
        inputs_dict = {A: [5]}
        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )
        assert 125 == output[0][0]

    def test_projection_assignment_mistake_swap(self):

        comp = Composition()
        A = TransferMechanism(name="composition-pytests-A", function=Linear(slope=1.0))
        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=1.0))
        C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=5.0))
        D = TransferMechanism(name="composition-pytests-D", function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_mechanism(C)
        comp.add_mechanism(D)
        comp.add_projection(A, MappingProjection(sender=A, receiver=C), C)
        with pytest.raises(CompositionError) as error_text:
            comp.add_projection(B, MappingProjection(sender=B, receiver=D), C)
        assert "is incompatible with the positions of these Components in their Composition" in str(error_text.value)

    def test_projection_assignment_mistake_swap2(self):
        # A ----> C --
        #              ==> E
        # B ----> D --

        comp = Composition()
        A = TransferMechanism(name="composition-pytests-A", function=Linear(slope=1.0))
        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=1.0))
        C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=5.0))
        D = TransferMechanism(name="composition-pytests-D", function=Linear(slope=5.0))
        E = TransferMechanism(name="composition-pytests-E", function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_mechanism(C)
        comp.add_mechanism(D)
        comp.add_projection(A, MappingProjection(sender=A, receiver=C), C)
        with pytest.raises(CompositionError) as error_text:
            comp.add_projection(B, MappingProjection(sender=B, receiver=C), D)

        assert "is incompatible with the positions of these Components in their Composition" in str(error_text.value)

    def test_run_5_mechanisms_2_origins_1_terminal(self):
        # A ----> C --
        #              ==> E
        # B ----> D --

        # 5 x 1 = 5 ----> 5 x 5 = 25 --
        #                                25 + 25 = 50  ==> 50 * 5 = 250
        # 5 * 1 = 5 ----> 5 x 5 = 25 --

        comp = Composition()
        A = TransferMechanism(name="composition-pytests-A", function=Linear(slope=1.0))
        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=1.0))
        C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=5.0))
        D = TransferMechanism(name="composition-pytests-D", function=Linear(slope=5.0))
        E = TransferMechanism(name="composition-pytests-E", function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_mechanism(C)
        comp.add_mechanism(D)
        comp.add_projection(A, MappingProjection(sender=A, receiver=C), C)
        comp.add_projection(B, MappingProjection(sender=B, receiver=D), D)
        comp.add_mechanism(E)
        comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
        comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
        comp._analyze_graph()
        inputs_dict = {A: [5],
                       B: [5]}
        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )
        assert 250 == output[0][0]

    def test_run_2_mechanisms_with_scheduling_AAB_integrator(self):
        comp = Composition()

        A = IntegratorMechanism(name="A [integrator]", default_variable=2.0, function=SimpleIntegrator(rate=1.0))
        # (1) value = 0 + (5.0 * 1.0) + 0  --> return 5.0
        # (2) value = 5.0 + (5.0 * 1.0) + 0  --> return 10.0
        B = TransferMechanism(name="B [transfer]", function=Linear(slope=5.0))
        # value = 10.0 * 5.0 --> return 50.0
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp._analyze_graph()
        inputs_dict = {A: [5]}
        sched = Scheduler(composition=comp)
        sched.add_condition(B, EveryNCalls(A, 2))
        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )
        assert 50.0 == output[0][0]

    def test_run_2_mechanisms_with_scheduling_AAB_transfer(self):
        comp = Composition()

        A = TransferMechanism(name="A [transfer]", function=Linear(slope=2.0))
        # (1) value = 5.0 * 2.0  --> return 10.0
        # (2) value = 5.0 * 2.0  --> return 10.0
        # ** TransferMechanism runs with the SAME input **
        B = TransferMechanism(name="B [transfer]", function=Linear(slope=5.0))
        # value = 10.0 * 5.0 --> return 50.0
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp._analyze_graph()
        inputs_dict = {A: [5]}
        sched = Scheduler(composition=comp)
        sched.add_condition(B, EveryNCalls(A, 2))
        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )
        assert 50.0 == output[0][0]

    def test_run_2_mechanisms_with_multiple_trials_of_input_values(self):
        comp = Composition()

        A = TransferMechanism(name="A [transfer]", function=Linear(slope=2.0))
        B = TransferMechanism(name="B [transfer]", function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp._analyze_graph()
        inputs_dict = {A: [1, 2, 3, 4]}
        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )

        assert 40.0 == output[0][0]

    def test_sender_receiver_not_specified(self):
        comp = Composition()

        A = TransferMechanism(name="A [transfer]", function=Linear(slope=2.0))
        B = TransferMechanism(name="B [transfer]", function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp._analyze_graph()
        inputs_dict = {A: [1, 2, 3, 4]}
        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )

        assert 40.0 == output[0][0]

    def test_run_2_mechanisms_reuse_input(self):
        comp = Composition()
        A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
        B = TransferMechanism(function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp._analyze_graph()
        inputs_dict = {A: [5]}
        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched,
            num_trials=5
        )
        assert 125 == output[0][0]

    def test_run_2_mechanisms_double_trial_specs(self):
        comp = Composition()
        A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
        B = TransferMechanism(function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp._analyze_graph()
        inputs_dict = {A: [[5], [4], [3]]}
        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched,
            num_trials=3
        )
        assert 75 == output[0][0]

    def test_execute_composition(self):
        comp = Composition()
        A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
        B = TransferMechanism(function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp._analyze_graph()
        inputs_dict = {A: 3}
        sched = Scheduler(composition=comp)
        output = comp.execute(
            inputs=inputs_dict,
            scheduler_processing=sched
        )
        assert 75 == output[0][0]

    def test_LPP(self):

        comp = Composition()
        A = TransferMechanism(name="composition-pytests-A", function=Linear(slope=2.0))   # 1 x 2 = 2
        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=2.0))   # 2 x 2 = 4
        C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=2.0))   # 4 x 2 = 8
        D = TransferMechanism(name="composition-pytests-D", function=Linear(slope=2.0))   # 8 x 2 = 16
        E = TransferMechanism(name="composition-pytests-E", function=Linear(slope=2.0))  # 16 x 2 = 32
        comp.add_linear_processing_pathway([A, B, C, D, E])
        comp._analyze_graph()
        inputs_dict = {A: [[1]]}
        sched = Scheduler(composition=comp)
        output = comp.execute(
            inputs=inputs_dict,
            scheduler_processing=sched
        )
        assert 32 == output[0][0]

    def test_LPP_with_projections(self):
        comp = Composition()
        A = TransferMechanism(name="composition-pytests-A", function=Linear(slope=2.0))  # 1 x 2 = 2
        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=2.0))  # 2 x 2 = 4
        C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=2.0))  # 4 x 2 = 8
        D = TransferMechanism(name="composition-pytests-D", function=Linear(slope=2.0))  # 8 x 2 = 16
        E = TransferMechanism(name="composition-pytests-E", function=Linear(slope=2.0))  # 16 x 2 = 32
        A_to_B = MappingProjection(sender=A, receiver=B)
        D_to_E = MappingProjection(sender=D, receiver=E)
        comp.add_linear_processing_pathway([A, A_to_B, B, C, D, D_to_E, E])
        comp._analyze_graph()
        inputs_dict = {A: [[1]]}
        sched = Scheduler(composition=comp)
        output = comp.execute(
            inputs=inputs_dict,
            scheduler_processing=sched
        )
        assert 32 == output[0][0]

    def test_LPP_end_with_projection(self):
        comp = Composition()
        A = TransferMechanism(name="composition-pytests-A", function=Linear(slope=2.0))
        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=2.0))
        C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=2.0))
        D = TransferMechanism(name="composition-pytests-D", function=Linear(slope=2.0))
        E = TransferMechanism(name="composition-pytests-E", function=Linear(slope=2.0))
        A_to_B = MappingProjection(sender=A, receiver=B)
        D_to_E = MappingProjection(sender=D, receiver=E)
        with pytest.raises(CompositionError) as error_text:
            comp.add_linear_processing_pathway([A, A_to_B, B, C, D, E, D_to_E])

        assert "A projection cannot be the last item in a linear processing pathway." in str(error_text.value)

    def test_LPP_two_projections_in_a_row(self):
        comp = Composition()
        A = TransferMechanism(name="composition-pytests-A", function=Linear(slope=2.0))
        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=2.0))
        C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=2.0))
        A_to_B = MappingProjection(sender=A, receiver=B)
        B_to_C = MappingProjection(sender=B, receiver=C)
        with pytest.raises(CompositionError) as error_text:
            comp.add_linear_processing_pathway([A, B_to_C, A_to_B, B, C])

        assert "A Projection in a linear processing pathway must be preceded by a Mechanism and followed by a " \
               "Mechanism" \
               in str(error_text.value)

    def test_LPP_start_with_projection(self):
        comp = Composition()
        Nonsense_Projection = MappingProjection()
        A = TransferMechanism(name="composition-pytests-A", function=Linear(slope=2.0))
        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=2.0))
        with pytest.raises(CompositionError) as error_text:
            comp.add_linear_processing_pathway([Nonsense_Projection, A, B])

        assert "The first item in a linear processing pathway must be a Mechanism." in str(
            error_text.value)

    def test_LPP_wrong_component(self):
        comp = Composition()
        Nonsense = "string"
        A = TransferMechanism(name="composition-pytests-A", function=Linear(slope=2.0))
        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=2.0))
        with pytest.raises(CompositionError) as error_text:
            comp.add_linear_processing_pathway([A, Nonsense, B])

        assert "A linear processing pathway must be made up of Projections and Mechanisms." in str(
            error_text.value)

    def test_LPP_two_origins_one_terminal(self):
        # A ----> C --
        #              ==> E
        # B ----> D --

        # 5 x 1 = 5 ----> 5 x 5 = 25 --
        #                                25 + 25 = 50  ==> 50 * 5 = 250
        # 5 * 1 = 5 ----> 5 x 5 = 25 --

        comp = Composition()
        A = TransferMechanism(name="composition-pytests-A", function=Linear(slope=1.0))
        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=1.0))
        C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=5.0))
        D = TransferMechanism(name="composition-pytests-D", function=Linear(slope=5.0))
        E = TransferMechanism(name="composition-pytests-E", function=Linear(slope=5.0))
        comp.add_linear_processing_pathway([A, C, E])
        comp.add_linear_processing_pathway([B, D, E])
        comp._analyze_graph()
        inputs_dict = {A: [5],
                       B: [5]}
        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )
        assert 250 == output[0][0]


class TestCallBeforeAfterTimescale:

    def test_call_before_record_timescale(self):

        comp = Composition()

        A = TransferMechanism(name="A [transfer]", function=Linear(slope=2.0))
        B = TransferMechanism(name="B [transfer]", function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp._analyze_graph()
        inputs_dict = {A: [1, 2, 3, 4]}
        sched = Scheduler(composition=comp)

        time_step_array = []
        trial_array = []
        pass_array = []

        def cb_timestep(scheduler, arr):

            def record_timestep():

                arr.append(scheduler.clocks[comp._execution_id].get_total_times_relative(TimeScale.TIME_STEP, TimeScale.TRIAL))

            return record_timestep

        def cb_pass(scheduler, arr):

            def record_pass():

                arr.append(scheduler.clocks[comp._execution_id].get_total_times_relative(TimeScale.PASS, TimeScale.RUN))

            return record_pass

        def cb_trial(scheduler, arr):

            def record_trial():

                arr.append(scheduler.clocks[comp._execution_id].get_total_times_relative(TimeScale.TRIAL, TimeScale.LIFE))

            return record_trial

        comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched,
            call_after_time_step=cb_timestep(sched, time_step_array),
            call_before_trial=cb_trial(sched, trial_array),
            call_before_pass=cb_pass(sched, pass_array)
        )
        assert time_step_array == [0, 1, 0, 1, 0, 1, 0, 1]
        assert trial_array == [0, 1, 2, 3]
        assert pass_array == [0, 1, 2, 3]

    def test_call_beforeafter_values_onepass(self):

        def record_values(d, time_scale, *mechs):
            if time_scale not in d:
                d[time_scale] = {}
            for mech in mechs:
                if mech not in d[time_scale]:
                    d[time_scale][mech] = []
                if mech.value is None:
                    d[time_scale][mech].append(np.nan)
                else:
                    d[time_scale][mech].append(mech.value[0])

        comp = Composition()

        A = TransferMechanism(name="A [transfer]", function=Linear(slope=2.0))
        B = TransferMechanism(name="B [transfer]", function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp._analyze_graph()
        inputs_dict = {A: [1, 2, 3, 4]}
        sched = Scheduler(composition=comp)

        before = {}
        after = {}

        before_expected = {
            TimeScale.TIME_STEP: {
                A: [np.nan, 2, 2, 4, 4, 6, 6, 8],
                B: [np.nan, np.nan, 10, 10, 20, 20, 30, 30]
            },
            TimeScale.PASS: {
                A: [np.nan, 2, 4, 6],
                B: [np.nan, 10, 20, 30]
            },
            TimeScale.TRIAL: {
                A: [np.nan, 2, 4, 6],
                B: [np.nan, 10, 20, 30]
            },
        }

        after_expected = {
            TimeScale.TIME_STEP: {
                A: [2, 2, 4, 4, 6, 6, 8, 8],
                B: [np.nan, 10, 10, 20, 20, 30, 30, 40]
            },
            TimeScale.PASS: {
                A: [2, 4, 6, 8],
                B: [10, 20, 30, 40]
            },
            TimeScale.TRIAL: {
                A: [2, 4, 6, 8],
                B: [10, 20, 30, 40]
            },
        }

        comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched,
            call_before_time_step=functools.partial(record_values, before, TimeScale.TIME_STEP, A, B),
            call_before_pass=functools.partial(record_values, before, TimeScale.PASS, A, B),
            call_before_trial=functools.partial(record_values, before, TimeScale.TRIAL, A, B),
            call_after_time_step=functools.partial(record_values, after, TimeScale.TIME_STEP, A, B),
            call_after_pass=functools.partial(record_values, after, TimeScale.PASS, A, B),
            call_after_trial=functools.partial(record_values, after, TimeScale.TRIAL, A, B),
        )

        for ts in before_expected:
            for mech in before_expected[ts]:
                np.testing.assert_allclose(before[ts][mech], before_expected[ts][mech], err_msg='Failed on before[{0}][{1}]'.format(ts, mech))

        for ts in after_expected:
            for mech in after_expected[ts]:
                comp = []
                for x in after[ts][mech]:
                    try:
                        comp.append(x[0])
                    except TypeError:
                        comp.append(x)
                np.testing.assert_allclose(comp, after_expected[ts][mech], err_msg='Failed on after[{0}][{1}]'.format(ts, mech))

    def test_call_beforeafter_values_twopass(self):

        def record_values(d, time_scale, *mechs):
            if time_scale not in d:
                d[time_scale] = {}
            for mech in mechs:
                if mech not in d[time_scale]:
                    d[time_scale][mech] = []
                if mech.value is None:
                    d[time_scale][mech].append(np.nan)
                else:
                    d[time_scale][mech].append(mech.value[0])

        comp = Composition()

        A = IntegratorMechanism(name="A [transfer]", function=SimpleIntegrator(rate=1))
        B = IntegratorMechanism(name="B [transfer]", function=SimpleIntegrator(rate=2))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp._analyze_graph()
        inputs_dict = {A: [1, 2]}
        sched = Scheduler(composition=comp)
        sched.add_condition(B, EveryNCalls(A, 2))

        before = {}
        after = {}

        before_expected = {
            TimeScale.TIME_STEP: {
                A: [
                    np.nan, 1, 2,
                    2, 4, 6,
                ],
                B: [
                    np.nan, np.nan, np.nan,
                    4, 4, 4,
                ]
            },
            TimeScale.PASS: {
                A: [
                    np.nan, 1,
                    2, 4,
                ],
                B: [
                    np.nan, np.nan,
                    4, 4,
                ]
            },
            TimeScale.TRIAL: {
                A: [np.nan, 2],
                B: [np.nan, 4]
            },
        }

        after_expected = {
            TimeScale.TIME_STEP: {
                A: [
                    1, 2, 2,
                    4, 6, 6,
                ],
                B: [
                    np.nan, np.nan, 4,
                    4, 4, 16,
                ]
            },
            TimeScale.PASS: {
                A: [
                    1, 2,
                    4, 6,
                ],
                B: [
                    np.nan, 4,
                    4, 16,
                ]
            },
            TimeScale.TRIAL: {
                A: [2, 6],
                B: [4, 16]
            },
        }

        comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched,
            call_before_time_step=functools.partial(record_values, before, TimeScale.TIME_STEP, A, B),
            call_before_pass=functools.partial(record_values, before, TimeScale.PASS, A, B),
            call_before_trial=functools.partial(record_values, before, TimeScale.TRIAL, A, B),
            call_after_time_step=functools.partial(record_values, after, TimeScale.TIME_STEP, A, B),
            call_after_pass=functools.partial(record_values, after, TimeScale.PASS, A, B),
            call_after_trial=functools.partial(record_values, after, TimeScale.TRIAL, A, B),
        )

        for ts in before_expected:
            for mech in before_expected[ts]:
                np.testing.assert_allclose(before[ts][mech], before_expected[ts][mech], err_msg='Failed on before[{0}][{1}]'.format(ts, mech))

        for ts in after_expected:
            for mech in after_expected[ts]:
                comp = []
                for x in after[ts][mech]:
                    try:
                        comp.append(x[0])
                    except TypeError:
                        comp.append(x)
                np.testing.assert_allclose(comp, after_expected[ts][mech], err_msg='Failed on after[{0}][{1}]'.format(ts, mech))

    # when self.sched is ready:
    # def test_run_default_scheduler(self):
    #     comp = Composition()
    #     A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
    #     B = TransferMechanism(function=Linear(slope=5.0))
    #     comp.add_mechanism(A)
    #     comp.add_mechanism(B)
    #     comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
    #     comp._analyze_graph()
    #     inputs_dict = {A: [[5], [4], [3]]}
    #     output = comp.run(
    #         inputs=inputs_dict,
    #         num_trials=3
    #     )
    #     assert 75 == output[0][0]

    # def test_multilayer_no_learning(self):
    #     Input_Layer = TransferMechanism(
    #         name='Input Layer',
    #         function=Logistic,
    #         default_variable=np.zeros((2,)),
    #     )
    #
    #     Hidden_Layer_1 = TransferMechanism(
    #         name='Hidden Layer_1',
    #         function=Logistic(),
    #         default_variable=np.zeros((5,)),
    #     )
    #
    #     Hidden_Layer_2 = TransferMechanism(
    #         name='Hidden Layer_2',
    #         function=Logistic(),
    #         default_variable=[0, 0, 0, 0],
    #     )
    #
    #     Output_Layer = TransferMechanism(
    #         name='Output Layer',
    #         function=Logistic,
    #         default_variable=[0, 0, 0],
    #     )
    #
    #     Input_Weights_matrix = (np.arange(2 * 5).reshape((2, 5)) + 1) / (2 * 5)
    #
    #     Input_Weights = MappingProjection(
    #         name='Input Weights',
    #         matrix=Input_Weights_matrix,
    #     )
    #
    #     comp = Composition()
    #     comp.add_mechanism(Input_Layer)
    #     comp.add_mechanism(Hidden_Layer_1)
    #     comp.add_mechanism(Hidden_Layer_2)
    #     comp.add_mechanism(Output_Layer)
    #
    #     comp.add_projection(Input_Layer, Input_Weights, Hidden_Layer_1)
    #     comp.add_projection(Hidden_Layer_1, MappingProjection(), Hidden_Layer_2)
    #     comp.add_projection(Hidden_Layer_2, MappingProjection(), Output_Layer)
    #
    #     comp._analyze_graph()
    #     stim_list = {Input_Layer: [[-1, 30]]}
    #     sched = Scheduler(composition=comp)
    #     output = comp.run(
    #         inputs=stim_list,
    #         scheduler_processing=sched,
    #         num_trials=10
    #     )
    #
    #     # p = Process(
    #     #     default_variable=[0, 0],
    #     #     pathway=[
    #     #         Input_Layer,
    #     #         # The following reference to Input_Weights is needed to use it in the pathway
    #     #         #    since it's sender and receiver args are not specified in its declaration above
    #     #         Input_Weights,
    #     #         Hidden_Layer_1,
    #     #         # No projection specification is needed here since the sender arg for Middle_Weights
    #     #         #    is Hidden_Layer_1 and its receiver arg is Hidden_Layer_2
    #     #         # Middle_Weights,
    #     #         Hidden_Layer_2,
    #     #         # Output_Weights does not need to be listed for the same reason as Middle_Weights
    #     #         # If Middle_Weights and/or Output_Weights is not declared above, then the process
    #     #         #    will assign a default for missing projection
    #     #         # Output_Weights,
    #     #         Output_Layer
    #     #     ],
    #     #     clamp_input=SOFT_CLAMP,
    #     #     target=[0, 0, 1]
    #     #
    #     #
    #     # )
    #     #
    #     # s.run(
    #     #     num_executions=10,
    #     #     inputs=stim_list,
    #     # )
    #
    #     expected_Output_Layer_output = [np.array([0.97988347, 0.97988347, 0.97988347])]
    #
    #     np.testing.assert_allclose(expected_Output_Layer_output, Output_Layer.output_values)

# Waiting to reintroduce ClampInput tests until we decide how this feature interacts with input specification

# class TestClampInput:
#
#     def test_run_5_mechanisms_2_origins_1_terminal_hard_clamp(self):
#
#         comp = Composition()
#         A = RecurrentTransferMechanism(name="composition-pytests-A", function=Linear(slope=1.0))
#         B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=1.0))
#         C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=5.0))
#         D = TransferMechanism(name="composition-pytests-D", function=Linear(slope=5.0))
#         E = TransferMechanism(name="composition-pytests-E", function=Linear(slope=5.0))
#         comp.add_mechanism(A)
#         comp.add_mechanism(B)
#         comp.add_mechanism(C)
#         comp.add_mechanism(D)
#         comp.add_projection(A, MappingProjection(sender=A, receiver=C), C)
#         comp.add_projection(B, MappingProjection(sender=B, receiver=D), D)
#         comp.add_mechanism(E)
#         comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
#         comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
#         comp._analyze_graph()
#         inputs_dict = {
#             A: [[5]],
#             B: [[5]]
#         }
#         sched = Scheduler(composition=comp)
#         sched.add_condition(A, EveryNPasses(1))
#         sched.add_condition(B, EveryNCalls(A, 2))
#         sched.add_condition(C, AfterNCalls(A, 2))
#         sched.add_condition(D, AfterNCalls(A, 2))
#         sched.add_condition(E, AfterNCalls(C, 1))
#         sched.add_condition(E, AfterNCalls(D, 1))
#         output = comp.run(
#             inputs=inputs_dict,
#             scheduler_processing=sched,
#             # clamp_input=HARD_CLAMP
#         )
#         assert 250 == output[0][0]
#
#     def test_run_5_mechanisms_2_origins_1_terminal_soft_clamp(self):
#
#         comp = Composition()
#         A = RecurrentTransferMechanism(name="composition-pytests-A", function=Linear(slope=1.0))
#         B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=1.0))
#         C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=5.0))
#         D = TransferMechanism(name="composition-pytests-D", function=Linear(slope=5.0))
#         E = TransferMechanism(name="composition-pytests-E", function=Linear(slope=5.0))
#         comp.add_mechanism(A)
#         comp.add_mechanism(B)
#         comp.add_mechanism(C)
#         comp.add_mechanism(D)
#         comp.add_projection(A, MappingProjection(sender=A, receiver=C), C)
#         comp.add_projection(B, MappingProjection(sender=B, receiver=D), D)
#         comp.add_mechanism(E)
#         comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
#         comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
#         comp._analyze_graph()
#         inputs_dict = {
#             A: [[5.]],
#             B: [[5.]]
#         }
#         sched = Scheduler(composition=comp)
#         sched.add_condition(A, EveryNPasses(1))
#         sched.add_condition(B, EveryNCalls(A, 2))
#         sched.add_condition(C, AfterNCalls(A, 2))
#         sched.add_condition(D, AfterNCalls(A, 2))
#         sched.add_condition(E, AfterNCalls(C, 1))
#         sched.add_condition(E, AfterNCalls(D, 1))
#         output = comp.run(
#             inputs=inputs_dict,
#             scheduler_processing=sched,
#             clamp_input=SOFT_CLAMP
#         )
#         assert 375 == output[0][0]
#
#     def test_run_5_mechanisms_2_origins_1_terminal_pulse_clamp(self):
#
#         comp = Composition()
#         A = RecurrentTransferMechanism(name="composition-pytests-A", function=Linear(slope=2.0))
#         B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=1.0))
#         C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=5.0))
#         D = TransferMechanism(name="composition-pytests-D", function=Linear(slope=5.0))
#         E = TransferMechanism(name="composition-pytests-E", function=Linear(slope=5.0))
#         comp.add_mechanism(A)
#         comp.add_mechanism(B)
#         comp.add_mechanism(C)
#         comp.add_mechanism(D)
#         comp.add_projection(A, MappingProjection(sender=A, receiver=C), C)
#         comp.add_projection(B, MappingProjection(sender=B, receiver=D), D)
#         comp.add_mechanism(E)
#         comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
#         comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
#         comp._analyze_graph()
#         inputs_dict = {
#             A: [[5]],
#             B: [[5]]
#         }
#         sched = Scheduler(composition=comp)
#         sched.add_condition(A, EveryNPasses(1))
#         sched.add_condition(B, EveryNCalls(A, 2))
#         sched.add_condition(C, AfterNCalls(A, 2))
#         sched.add_condition(D, AfterNCalls(A, 2))
#         sched.add_condition(E, AfterNCalls(C, 1))
#         sched.add_condition(E, AfterNCalls(D, 1))
#         output = comp.run(
#             inputs=inputs_dict,
#             scheduler_processing=sched,
#             clamp_input=PULSE_CLAMP
#         )
#         assert 625 == output[0][0]
#
#     def test_run_5_mechanisms_2_origins_1_hard_clamp_1_soft_clamp(self):
#
#         #          __
#         #         |  |
#         #         V  |
#         # 5 -#1-> A -^--> C --
#         #                       ==> E
#         # 5 ----> B ----> D --
#
#         #         v Recurrent
#         # 5 * 1 = (5 + 5) x 1 = 10
#         # 5 x 1 = 5 ---->      10 x 5 = 50 --
#         #                                       50 + 25 = 75  ==> 75 * 5 = 375
#         # 5 * 1 = 5 ---->       5 x 5 = 25 --
#
#         comp = Composition()
#         A = RecurrentTransferMechanism(name="composition-pytests-A", function=Linear(slope=1.0))
#         B = RecurrentTransferMechanism(name="composition-pytests-B", function=Linear(slope=1.0))
#         C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=5.0))
#         D = TransferMechanism(name="composition-pytests-D", function=Linear(slope=5.0))
#         E = TransferMechanism(name="composition-pytests-E", function=Linear(slope=5.0))
#         comp.add_mechanism(A)
#         comp.add_mechanism(B)
#         comp.add_mechanism(C)
#         comp.add_mechanism(D)
#         comp.add_projection(A, MappingProjection(sender=A, receiver=C), C)
#         comp.add_projection(B, MappingProjection(sender=B, receiver=D), D)
#         comp.add_mechanism(E)
#         comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
#         comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
#         comp._analyze_graph()
#         inputs_dict = {
#             A: [[5]],
#             B: [[5]]
#         }
#         sched = Scheduler(composition=comp)
#         sched.add_condition(A, EveryNPasses(1))
#         sched.add_condition(B, EveryNPasses(1))
#         sched.add_condition(B, EveryNCalls(A, 1))
#         sched.add_condition(C, AfterNCalls(A, 2))
#         sched.add_condition(D, AfterNCalls(A, 2))
#         sched.add_condition(E, AfterNCalls(C, 1))
#         sched.add_condition(E, AfterNCalls(D, 1))
#         output = comp.run(
#             inputs=inputs_dict,
#             scheduler_processing=sched,
#             clamp_input={A: SOFT_CLAMP,
#                          B: HARD_CLAMP}
#         )
#         assert 375 == output[0][0]
#
#     def test_run_5_mechanisms_2_origins_1_terminal_no_clamp(self):
#         # input ignored on all executions
#         #          _r_
#         #         |   |
#         # 0 -#2-> V   |
#         # 0 -#1-> A -^--> C --
#         #                       ==> E
#         # 0 ----> B ----> D --
#
#         # 1 * 2 + 1 = 3
#         # 0 x 2 + 1 = 1 ----> 4 x 5 = 20 --
#         #                                   20 + 5 = 25  ==> 25 * 5 = 125
#         # 0 x 1 + 1 = 1 ----> 1 x 5 = 5 --
#
#         comp = Composition()
#
#         A = RecurrentTransferMechanism(name="composition-pytests-A", function=Linear(slope=2.0, intercept=5.0))
#         B = RecurrentTransferMechanism(name="composition-pytests-B", function=Linear(slope=1.0, intercept=1.0))
#         C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=5.0))
#         D = TransferMechanism(name="composition-pytests-D", function=Linear(slope=5.0))
#         E = TransferMechanism(name="composition-pytests-E", function=Linear(slope=5.0))
#         comp.add_mechanism(A)
#         comp.add_mechanism(B)
#         comp.add_mechanism(C)
#         comp.add_mechanism(D)
#         comp.add_projection(A, MappingProjection(sender=A, receiver=C), C)
#         comp.add_projection(B, MappingProjection(sender=B, receiver=D), D)
#         comp.add_mechanism(E)
#         comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
#         comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
#         comp._analyze_graph()
#         inputs_dict = {
#             A: [[100.0]],
#             B: [[500.0]]
#         }
#         sched = Scheduler(composition=comp)
#         sched.add_condition(A, EveryNPasses(1))
#         sched.add_condition(B, EveryNCalls(A, 2))
#         sched.add_condition(C, AfterNCalls(A, 2))
#         sched.add_condition(D, AfterNCalls(A, 2))
#         sched.add_condition(E, AfterNCalls(C, 1))
#         sched.add_condition(E, AfterNCalls(D, 1))
#         output = comp.run(
#             inputs=inputs_dict,
#             scheduler_processing=sched,
#             clamp_input=NO_CLAMP
#         )
#         # FIX: This value is correct given that there is a BUG in Recurrent Transfer Mech --
#         # Recurrent projection BEGINS with a value leftover from initialization
#         # (only shows up if the function has an additive component or default variable is not zero)
#         assert 925 == output[0][0]


class TestSystemComposition:

    # def test_run_2_mechanisms_default_input_1(self):
    #     sys = SystemComposition()
    #     A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
    #     B = TransferMechanism(function=Linear(slope=5.0))
    #     sys.add_mechanism(A)
    #     sys.add_mechanism(B)
    #     sys.add_projection(A, MappingProjection(sender=A, receiver=B), B)
    #     sys._analyze_graph()
    #     sched = Scheduler(composition=sys)
    #     output = sys.run(
    #         scheduler_processing=sched
    #     )
    #     assert 25 == output[0][0]

    def test_run_2_mechanisms_input_5(self):
        sys = SystemComposition()
        A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
        B = TransferMechanism(function=Linear(slope=5.0))
        sys.add_mechanism(A)
        sys.add_mechanism(B)
        sys.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        sys._analyze_graph()
        inputs_dict = {A: [[5]]}
        sched = Scheduler(composition=sys)
        output = sys.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )
        assert 125 == output[0][0]

    def test_call_beforeafter_values_onepass(self):

        def record_values(d, time_scale, *mechs):
            if time_scale not in d:
                d[time_scale] = {}
            for mech in mechs:
                if mech not in d[time_scale]:
                    d[time_scale][mech] = []
                if mech.value is None:
                    d[time_scale][mech].append(np.nan)
                else:
                    d[time_scale][mech].append(mech.value)

        comp = Composition()

        A = TransferMechanism(name="A [transfer]", function=Linear(slope=2.0))
        B = TransferMechanism(name="B [transfer]", function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp._analyze_graph()
        inputs_dict = {A: [[1],[ 2], [3], [4]]}
        sched = Scheduler(composition=comp)

        before = {}
        after = {}

        before_expected = {
            TimeScale.TIME_STEP: {
                A: [np.nan, 2, 2, 4, 4, 6, 6, 8],
                B: [np.nan, np.nan, 10, 10, 20, 20, 30, 30]
            },
            TimeScale.PASS: {
                A: [np.nan, 2, 4, 6],
                B: [np.nan, 10, 20, 30]
            },
            TimeScale.TRIAL: {
                A: [np.nan, 2, 4, 6],
                B: [np.nan, 10, 20, 30]
            },
        }

        after_expected = {
            TimeScale.TIME_STEP: {
                A: [2, 2, 4, 4, 6, 6, 8, 8],
                B: [np.nan, 10, 10, 20, 20, 30, 30, 40]
            },
            TimeScale.PASS: {
                A: [2, 4, 6, 8],
                B: [10, 20, 30, 40]
            },
            TimeScale.TRIAL: {
                A: [2, 4, 6, 8],
                B: [10, 20, 30, 40]
            },
        }

        comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched,
            call_before_time_step=functools.partial(record_values, before, TimeScale.TIME_STEP, A, B),
            call_before_pass=functools.partial(record_values, before, TimeScale.PASS, A, B),
            call_before_trial=functools.partial(record_values, before, TimeScale.TRIAL, A, B),
            call_after_time_step=functools.partial(record_values, after, TimeScale.TIME_STEP, A, B),
            call_after_pass=functools.partial(record_values, after, TimeScale.PASS, A, B),
            call_after_trial=functools.partial(record_values, after, TimeScale.TRIAL, A, B),
        )

        for ts in before_expected:
            for mech in before_expected[ts]:
                # extra brackets around 'before_expected[ts][mech]' were needed for np assert to work
                np.testing.assert_allclose([before[ts][mech]], [before_expected[ts][mech]], err_msg='Failed on before[{0}][{1}]'.format(ts, mech))

        for ts in after_expected:
            for mech in after_expected[ts]:
                comp = []
                for x in after[ts][mech]:
                    try:
                        comp.append(x[0][0])
                    except TypeError:
                        comp.append(x)
                np.testing.assert_allclose(comp, after_expected[ts][mech], err_msg='Failed on after[{0}][{1}]'.format(ts, mech))

    def test_call_beforeafter_values_twopass(self):

        def record_values(d, time_scale, *mechs):
            if time_scale not in d:
                d[time_scale] = {}
            for mech in mechs:
                if mech not in d[time_scale]:
                    d[time_scale][mech] = []
                if mech.value is None:
                    d[time_scale][mech].append(np.nan)
                else:
                    d[time_scale][mech].append(mech.value)

        comp = Composition()

        A = IntegratorMechanism(name="A [transfer]", function=SimpleIntegrator(rate=1))
        B = IntegratorMechanism(name="B [transfer]", function=SimpleIntegrator(rate=2))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp._analyze_graph()
        inputs_dict = {A: [[1], [2]]}
        sched = Scheduler(composition=comp)
        sched.add_condition(B, EveryNCalls(A, 2))

        before = {}
        after = {}

        before_expected = {
            TimeScale.TIME_STEP: {
                A: [
                    np.nan, 1, 2,
                    2, 4, 6,
                ],
                B: [
                    np.nan, np.nan, np.nan,
                    4, 4, 4,
                ]
            },
            TimeScale.PASS: {
                A: [
                    np.nan, 1,
                    2, 4,
                ],
                B: [
                    np.nan, np.nan,
                    4, 4,
                ]
            },
            TimeScale.TRIAL: {
                A: [np.nan, 2],
                B: [np.nan, 4]
            },
        }

        after_expected = {
            TimeScale.TIME_STEP: {
                A: [
                    1, 2, 2,
                    4, 6, 6,
                ],
                B: [
                    np.nan, np.nan, 4,
                    4, 4, 16,
                ]
            },
            TimeScale.PASS: {
                A: [
                    1, 2,
                    4, 6,
                ],
                B: [
                    np.nan, 4,
                    4, 16,
                ]
            },
            TimeScale.TRIAL: {
                A: [2, 6],
                B: [4, 16]
            },
        }

        comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched,
            call_before_time_step=functools.partial(record_values, before, TimeScale.TIME_STEP, A, B),
            call_before_pass=functools.partial(record_values, before, TimeScale.PASS, A, B),
            call_before_trial=functools.partial(record_values, before, TimeScale.TRIAL, A, B),
            call_after_time_step=functools.partial(record_values, after, TimeScale.TIME_STEP, A, B),
            call_after_pass=functools.partial(record_values, after, TimeScale.PASS, A, B),
            call_after_trial=functools.partial(record_values, after, TimeScale.TRIAL, A, B),
        )

        for ts in before_expected:
            for mech in before_expected[ts]:
                np.testing.assert_allclose(before[ts][mech], before_expected[ts][mech], err_msg='Failed on before[{0}][{1}]'.format(ts, mech))

        for ts in after_expected:
            for mech in after_expected[ts]:
                comp = []
                for x in after[ts][mech]:
                    try:
                        comp.append(x[0][0])
                    except TypeError:
                        comp.append(x)
                np.testing.assert_allclose(comp, after_expected[ts][mech], err_msg='Failed on after[{0}][{1}]'.format(ts, mech))

    # when self.sched is ready:
    # def test_run_default_scheduler(self):
    #     comp = Composition()
    #     A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
    #     B = TransferMechanism(function=Linear(slope=5.0))
    #     comp.add_mechanism(A)
    #     comp.add_mechanism(B)
    #     comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
    #     comp._analyze_graph()
    #     inputs_dict = {A: [[5], [4], [3]]}
    #     output = comp.run(
    #         inputs=inputs_dict,
    #         num_trials=3
    #     )
    #     assert 75 == output[0][0]

    # def test_multilayer_no_learning(self):
    #     Input_Layer = TransferMechanism(
    #         name='Input Layer',
    #         function=Logistic,
    #         default_variable=np.zeros((2,)),
    #     )
    #
    #     Hidden_Layer_1 = TransferMechanism(
    #         name='Hidden Layer_1',
    #         function=Logistic(),
    #         default_variable=np.zeros((5,)),
    #     )
    #
    #     Hidden_Layer_2 = TransferMechanism(
    #         name='Hidden Layer_2',
    #         function=Logistic(),
    #         default_variable=[0, 0, 0, 0],
    #     )
    #
    #     Output_Layer = TransferMechanism(
    #         name='Output Layer',
    #         function=Logistic,
    #         default_variable=[0, 0, 0],
    #     )
    #
    #     Input_Weights_matrix = (np.arange(2 * 5).reshape((2, 5)) + 1) / (2 * 5)
    #
    #     Input_Weights = MappingProjection(
    #         name='Input Weights',
    #         matrix=Input_Weights_matrix,
    #     )
    #
    #     comp = Composition()
    #     comp.add_mechanism(Input_Layer)
    #     comp.add_mechanism(Hidden_Layer_1)
    #     comp.add_mechanism(Hidden_Layer_2)
    #     comp.add_mechanism(Output_Layer)
    #
    #     comp.add_projection(Input_Layer, Input_Weights, Hidden_Layer_1)
    #     comp.add_projection(Hidden_Layer_1, MappingProjection(), Hidden_Layer_2)
    #     comp.add_projection(Hidden_Layer_2, MappingProjection(), Output_Layer)
    #
    #     comp._analyze_graph()
    #     stim_list = {Input_Layer: [[-1, 30]]}
    #     sched = Scheduler(composition=comp)
    #     output = comp.run(
    #         inputs=stim_list,
    #         scheduler_processing=sched,
    #         num_trials=10
    #     )
    #
    #     # p = process(
    #     #     default_variable=[0, 0],
    #     #     pathway=[
    #     #         Input_Layer,
    #     #         # The following reference to Input_Weights is needed to use it in the pathway
    #     #         #    since it's sender and receiver args are not specified in its declaration above
    #     #         Input_Weights,
    #     #         Hidden_Layer_1,
    #     #         # No projection specification is needed here since the sender arg for Middle_Weights
    #     #         #    is Hidden_Layer_1 and its receiver arg is Hidden_Layer_2
    #     #         # Middle_Weights,
    #     #         Hidden_Layer_2,
    #     #         # Output_Weights does not need to be listed for the same reason as Middle_Weights
    #     #         # If Middle_Weights and/or Output_Weights is not declared above, then the process
    #     #         #    will assign a default for missing projection
    #     #         # Output_Weights,
    #     #         Output_Layer
    #     #     ],
    #     #     clamp_input=SOFT_CLAMP,
    #     #     target=[0, 0, 1]
    #     #
    #     #
    #     # )
    #     #
    #     # s.run(
    #     #     num_executions=10,
    #     #     inputs=stim_list,
    #     # )
    #
    #     expected_Output_Layer_output = [np.array([0.97988347, 0.97988347, 0.97988347])]
    #
    #     np.testing.assert_allclose(expected_Output_Layer_output, Output_Layer.output_values)


# Cannot test old syntax until we are ready for the current System and Process classes to create compositions
# class TestOldSyntax:
#
#     # new syntax pathway, old syntax system
#     def test_one_pathway_inside_one_system_old_syntax(self):
#         # create a PathwayComposition | blank slate for composition
#         myPath = PathwayComposition()
#
#         # create mechanisms to add to myPath
#         myMech1 = TransferMechanism(function=Linear(slope=2.0))  # 1 x 2 = 2
#         myMech2 = TransferMechanism(function=Linear(slope=2.0))  # 2 x 2 = 4
#         myMech3 = TransferMechanism(function=Linear(slope=2.0))  # 4 x 2 = 8
#
#         # add mechanisms to myPath with default MappingProjections between them
#         myPath.add_linear_processing_pathway([myMech1, myMech2, myMech3])
#
#         # analyze graph (assign roles)
#         myPath._analyze_graph()
#
#         # Create a system using the old factory method syntax
#         sys = system(processes=[myPath])
#
#         # assign input to origin mech
#         stimulus = {myMech1: [[1]]}
#
#         # schedule = Scheduler(composition=sys)
#         output = sys.run(
#             inputs=stimulus,
#             # scheduler_processing=schedule
#         )
#         assert 8 == output[0][0]
#
#     # old syntax pathway (process)
#     def test_one_process_old_syntax(self):
#
#         # create mechanisms to add to myPath
#         myMech1 = TransferMechanism(function=Linear(slope=2.0))  # 1 x 2 = 2
#         myMech2 = TransferMechanism(function=Linear(slope=2.0))  # 2 x 2 = 4
#         myMech3 = TransferMechanism(function=Linear(slope=2.0))  # 4 x 2 = 8
#
#         # create a PathwayComposition | blank slate for composition
#         myPath = process(pathway=[myMech1, myMech2, myMech3])
#
#         # assign input to origin mech
#         stimulus = {myMech1: [[1]]}
#
#         # schedule = Scheduler(composition=sys)
#         output = myPath.run(
#             inputs=stimulus,
#             # scheduler_processing=schedule
#         )
#         assert 8 == output[0][0]
#
#     # old syntax pathway (process), old syntax system
#     def test_one_process_inside_one_system_old_syntax(self):
#         # create mechanisms to add to myPath
#         myMech1 = TransferMechanism(function=Linear(slope=2.0))  # 1 x 2 = 2
#         myMech2 = TransferMechanism(function=Linear(slope=2.0))  # 2 x 2 = 4
#         myMech3 = TransferMechanism(function=Linear(slope=2.0))  # 4 x 2 = 8
#
#         # create a PathwayComposition | blank slate for composition
#         myPath = process(pathway=[myMech1, myMech2, myMech3])
#
#         # Create a system using the old factory method syntax
#         sys = system(processes=[myPath])
#
#         # assign input to origin mech
#         stimulus = {myMech1: [[1]]}
#
#         # schedule = Scheduler(composition=sys)
#         output = sys.run(
#             inputs=stimulus,
#             # scheduler_processing=schedule
#         )
#         assert 8 == output[0][0]
#
#     # old syntax pathway (process), old syntax system; 2 processes in series
#     def test_two_processes_in_series_in_system_old_syntax(self):
#
#         # create mechanisms to add to myPath
#         myMech1 = TransferMechanism(function=Linear(slope=2.0))  # 1 x 2 = 2
#         myMech2 = TransferMechanism(function=Linear(slope=2.0))  # 2 x 2 = 4
#         myMech3 = TransferMechanism(function=Linear(slope=2.0))  # 4 x 2 = 8
#         # create a PathwayComposition | blank slate for composition
#         myPath = process(pathway=[myMech1, myMech2, myMech3])
#
#         # create a PathwayComposition | blank slate for composition
#         myPath2 = PathwayComposition()
#
#         # create mechanisms to add to myPath2
#         myMech4 = TransferMechanism(function=Linear(slope=2.0))  # 8 x 2 = 16
#         myMech5 = TransferMechanism(function=Linear(slope=2.0))  # 16 x 2 = 32
#         myMech6 = TransferMechanism(function=Linear(slope=2.0))  # 32 x 2 = 64
#
#         # add mechanisms to myPath2 with default MappingProjections between them
#         myPath2.add_linear_processing_pathway([myMech4, myMech5, myMech6])
#
#         # analyze graph (assign roles)
#         myPath2._analyze_graph()
#
#         # Create a system using the old factory method syntax
#         sys = system(processes=[myPath, myPath2])
#
#         # connect the two pathways in series
#         sys.add_projection(sender=myMech3,
#                            projection=MappingProjection(sender=myMech3, receiver=myMech4),
#                            receiver=myMech4)
#         # assign input to origin mech
#         stimulus = {myMech1: [[1]]}
#
#         # schedule = Scheduler(composition=sys)
#         output = sys.run(
#             inputs=stimulus,
#             # scheduler_processing=schedule
#         )
#         assert 64 == output[0][0]
#
#     # old syntax pathway (process), old syntax system; 2 processes converge
#     def test_two_processes_converge_in_system_old_syntax(self):
#         # create a PathwayComposition | blank slate for composition
#         myPath = PathwayComposition()
#
#         # create mechanisms to add to myPath
#         myMech1 = TransferMechanism(function=Linear(slope=2.0))  # 1 x 2 = 2
#         myMech2 = TransferMechanism(function=Linear(slope=2.0))  # 2 x 2 = 4
#         myMech3 = TransferMechanism(function=Linear(slope=2.0))
#
#         # add mechanisms to myPath with default MappingProjections between them
#         myPath.add_linear_processing_pathway([myMech1, myMech2, myMech3])
#
#         # analyze graph (assign roles)
#         myPath._analyze_graph()
#
#         # create a PathwayComposition | blank slate for composition
#         myPath2 = PathwayComposition()
#
#         # create mechanisms to add to myPath2
#         myMech4 = TransferMechanism(function=Linear(slope=2.0))  # 1 x 2 = 2
#         myMech5 = TransferMechanism(function=Linear(slope=2.0))  # 2 x 2 = 4
#
#         # add mechanisms to myPath2 with default MappingProjections between them
#         myPath2.add_linear_processing_pathway([myMech4, myMech5, myMech3])
#
#         # analyze graph (assign roles)
#         myPath2._analyze_graph()
#
#         # Create a system using the old factory method syntax
#         sys = system(processes=[myPath, myPath2])
#
#         # assign input to origin mech
#         stimulus = {myMech1: [[1]],
#                     myMech4: [[1]]}
#
#         # schedule = Scheduler(composition=sys)
#         output = sys.run(
#             inputs=stimulus,
#             # scheduler_processing=schedule
#         )
#         assert 16 == output[0][0]
#

class TestNestedCompositions:
    def test_combine_two_disjunct_trees(self):
        # Goal:

        # Mech1 --
        #          --> Mech3 ----> Mech4 --
        # Mech2 --                          --> Mech6
        #                          Mech5 --

        # create first composition -----------------------------------------------

        # Mech1 --
        #           --> Mech3
        # Mech2 --

        tree1 = Composition()

        myMech1 = TransferMechanism(name="myMech1")
        myMech2 = TransferMechanism(name="myMech2")
        myMech3 = TransferMechanism(name="myMech3")
        myMech4 = TransferMechanism(name="myMech4")
        myMech5 = TransferMechanism(name="myMech5")
        myMech6 = TransferMechanism(name="myMech6")

        tree1.add_mechanism(myMech1)
        tree1.add_mechanism(myMech2)
        tree1.add_mechanism(myMech3)
        tree1.add_projection(myMech1, MappingProjection(sender=myMech1, receiver=myMech3), myMech3)
        tree1.add_projection(myMech2, MappingProjection(sender=myMech2, receiver=myMech3), myMech3)

        # validate first composition ---------------------------------------------

        tree1._analyze_graph()
        origins = tree1.get_mechanisms_by_role(MechanismRole.ORIGIN)
        assert len(origins) == 2
        assert myMech1 in origins
        assert myMech2 in origins
        terminals = tree1.get_mechanisms_by_role(MechanismRole.TERMINAL)
        assert len(terminals) == 1
        assert myMech3 in terminals

        # create second composition ----------------------------------------------

        # Mech4 --
        #           --> Mech6
        # Mech5 --

        tree2 = Composition()
        tree2.add_mechanism(myMech4)
        tree2.add_mechanism(myMech5)
        tree2.add_mechanism(myMech6)
        tree2.add_projection(myMech4, MappingProjection(sender=myMech4, receiver=myMech6), myMech6)
        tree2.add_projection(myMech5, MappingProjection(sender=myMech5, receiver=myMech6), myMech6)

        # validate second composition ----------------------------------------------

        tree2._analyze_graph()
        origins = tree2.get_mechanisms_by_role(MechanismRole.ORIGIN)
        assert len(origins) == 2
        assert myMech4 in origins
        assert myMech5 in origins
        terminals = tree2.get_mechanisms_by_role(MechanismRole.TERMINAL)
        assert len(terminals) == 1
        assert myMech6 in terminals

        # combine the compositions -------------------------------------------------

        tree1.add_pathway(tree2)
        tree1._analyze_graph()

        # BEFORE linking via 3 --> 4 projection ------------------------------------
        # Mech1 --
        #           --> Mech3
        # Mech2 --
        # Mech4 --
        #           --> Mech6
        # Mech5 --

        origins = tree1.get_mechanisms_by_role(MechanismRole.ORIGIN)
        assert len(origins) == 4
        assert myMech1 in origins
        assert myMech2 in origins
        assert myMech4 in origins
        assert myMech5 in origins
        terminals = tree1.get_mechanisms_by_role(MechanismRole.TERMINAL)
        assert len(terminals) == 2
        assert myMech3 in terminals
        assert myMech6 in terminals

        # AFTER linking via 3 --> 4 projection ------------------------------------
        # Mech1 --
        #          --> Mech3 ----> Mech4 --
        # Mech2 --                          --> Mech6
        #                          Mech5 --

        tree1.add_projection(myMech3, MappingProjection(sender=myMech3, receiver=myMech4), myMech4)
        tree1._analyze_graph()

        origins = tree1.get_mechanisms_by_role(MechanismRole.ORIGIN)
        assert len(origins) == 3
        assert myMech1 in origins
        assert myMech2 in origins
        assert myMech5 in origins
        terminals = tree1.get_mechanisms_by_role(MechanismRole.TERMINAL)
        assert len(terminals) == 1
        assert myMech6 in terminals

    def test_combine_two_overlapping_trees(self):
            # Goal:

            # Mech1 --
            #          --> Mech3 --
            # Mech2 --              --> Mech5
            #              Mech4 --

            # create first composition -----------------------------------------------

            # Mech1 --
            #           --> Mech3
            # Mech2 --

            tree1 = Composition()

            myMech1 = TransferMechanism(name="myMech1")
            myMech2 = TransferMechanism(name="myMech2")
            myMech3 = TransferMechanism(name="myMech3")
            myMech4 = TransferMechanism(name="myMech4")
            myMech5 = TransferMechanism(name="myMech5")

            tree1.add_mechanism(myMech1)
            tree1.add_mechanism(myMech2)
            tree1.add_mechanism(myMech3)
            tree1.add_projection(myMech1, MappingProjection(sender=myMech1, receiver=myMech3), myMech3)
            tree1.add_projection(myMech2, MappingProjection(sender=myMech2, receiver=myMech3), myMech3)

            # validate first composition ---------------------------------------------

            tree1._analyze_graph()
            origins = tree1.get_mechanisms_by_role(MechanismRole.ORIGIN)
            assert len(origins) == 2
            assert myMech1 in origins
            assert myMech2 in origins
            terminals = tree1.get_mechanisms_by_role(MechanismRole.TERMINAL)
            assert len(terminals) == 1
            assert myMech3 in terminals

            # create second composition ----------------------------------------------

            # Mech3 --
            #           --> Mech5
            # Mech4 --

            tree2 = Composition()
            tree2.add_mechanism(myMech3)
            tree2.add_mechanism(myMech4)
            tree2.add_mechanism(myMech5)
            tree2.add_projection(myMech3, MappingProjection(sender=myMech3, receiver=myMech5), myMech5)
            tree2.add_projection(myMech4, MappingProjection(sender=myMech4, receiver=myMech5), myMech5)

            # validate second composition ----------------------------------------------

            tree2._analyze_graph()
            origins = tree2.get_mechanisms_by_role(MechanismRole.ORIGIN)
            assert len(origins) == 2
            assert myMech3 in origins
            assert myMech4 in origins
            terminals = tree2.get_mechanisms_by_role(MechanismRole.TERMINAL)
            assert len(terminals) == 1
            assert myMech5 in terminals

            # combine the compositions -------------------------------------------------

            tree1.add_pathway(tree2)
            tree1._analyze_graph()
            # no need for a projection connecting the two compositions because they share myMech3

            origins = tree1.get_mechanisms_by_role(MechanismRole.ORIGIN)
            assert len(origins) == 3
            assert myMech1 in origins
            assert myMech2 in origins
            assert myMech4 in origins
            terminals = tree1.get_mechanisms_by_role(MechanismRole.TERMINAL)
            assert len(terminals) == 1
            assert myMech5 in terminals

    def test_one_pathway_inside_one_system(self):
        # create a PathwayComposition | blank slate for composition
        myPath = PathwayComposition()

        # create mechanisms to add to myPath
        myMech1 = TransferMechanism(function=Linear(slope=2.0))  # 1 x 2 = 2
        myMech2 = TransferMechanism(function=Linear(slope=2.0))  # 2 x 2 = 4
        myMech3 = TransferMechanism(function=Linear(slope=2.0))  # 4 x 2 = 8

        # add mechanisms to myPath with default MappingProjections between them
        myPath.add_linear_processing_pathway([myMech1, myMech2, myMech3])

        # analyze graph (assign roles)
        myPath._analyze_graph()

        # assign input to origin mech
        stimulus = {myMech1: [[1]]}

        # execute path (just for comparison)
        myPath.run(inputs=stimulus)

        # create a SystemComposition | blank slate for composition
        sys = SystemComposition()

        # add a PathwayComposition [myPath] to the SystemComposition [sys]
        sys.add_pathway(myPath)

        # execute the SystemComposition
        output = sys.run(
            inputs=stimulus,
        )
        assert 8 == output[0][0]

    def test_two_paths_converge_one_system(self):

        # mech1 ---> mech2 --
        #                   --> mech3
        # mech4 ---> mech5 --

        # 1x2=2 ---> 2x2=4 --
        #                   --> (4+4)x2=16
        # 1x2=2 ---> 2x2=4 --

        # create a PathwayComposition | blank slate for composition
        myPath = PathwayComposition()

        # create mechanisms to add to myPath
        myMech1 = TransferMechanism(function=Linear(slope=2.0))  # 1 x 2 = 2
        myMech2 = TransferMechanism(function=Linear(slope=2.0))  # 2 x 2 = 4
        myMech3 = TransferMechanism(function=Linear(slope=2.0))  # 4 x 2 = 8

        # add mechanisms to myPath with default MappingProjections between them
        myPath.add_linear_processing_pathway([myMech1, myMech2, myMech3])

        # analyze graph (assign roles)
        myPath._analyze_graph()

        myPath2 = PathwayComposition()
        myMech4 = TransferMechanism(function=Linear(slope=2.0))  # 1 x 2 = 2
        myMech5 = TransferMechanism(function=Linear(slope=2.0))  # 2 x 2 = 4
        myPath2.add_linear_processing_pathway([myMech4, myMech5, myMech3])
        myPath2._analyze_graph()

        sys = SystemComposition()
        sys.add_pathway(myPath)
        sys.add_pathway(myPath2)
        # assign input to origin mechs
        stimulus = {myMech1: [[1]], myMech4: [[1]]}

        # schedule = Scheduler(composition=sys)
        output = sys.run(
            inputs=stimulus,
            # scheduler_processing=schedule
        )
        assert 16 == output[0][0]

    def test_two_paths_in_series_one_system(self):

        # [ mech1 --> mech2 --> mech3 ] -->   [ mech4  -->  mech5  -->  mech6 ]
        #   1x2=2 --> 2x2=4 --> 4x2=8   --> (8+1)x2=18 --> 18x2=36 --> 36*2=64
        #                                X
        #                                |
        #                                1
        # (if mech4 were recognized as an origin mech, and used SOFT_CLAMP, we would expect the final result to be 72)
        # create a PathwayComposition | blank slate for composition
        myPath = PathwayComposition()

        # create mechanisms to add to myPath
        myMech1 = TransferMechanism(function=Linear(slope=2.0))  # 1 x 2 = 2
        myMech2 = TransferMechanism(function=Linear(slope=2.0))  # 2 x 2 = 4
        myMech3 = TransferMechanism(function=Linear(slope=2.0))  # 4 x 2 = 8

        # add mechanisms to myPath with default MappingProjections between them
        myPath.add_linear_processing_pathway([myMech1, myMech2, myMech3])

        # analyze graph (assign roles)
        myPath._analyze_graph()

        myPath2 = PathwayComposition()
        myMech4 = TransferMechanism(function=Linear(slope=2.0))
        myMech5 = TransferMechanism(function=Linear(slope=2.0))
        myMech6 = TransferMechanism(function=Linear(slope=2.0))
        myPath2.add_linear_processing_pathway([myMech4, myMech5, myMech6])
        myPath2._analyze_graph()

        sys = SystemComposition()
        sys.add_pathway(myPath)
        sys.add_pathway(myPath2)
        sys.add_projection(sender=myMech3, projection=MappingProjection(sender=myMech3,
                                                                        receiver=myMech4), receiver=myMech4)
        # assign input to origin mechs
        # myMech4 ignores its input from the outside world because it is no longer considered an origin!
        stimulus = {myMech1: [[1]]}
        sys._analyze_graph()
        # schedule = Scheduler(composition=sys)
        output = sys.run(
            inputs=stimulus,
            # scheduler_processing=schedule
        )
        assert 64 == output[0][0]

    def test_two_paths_converge_one_system_scheduling_matters(self):

        # mech1 ---> mech2 --
        #                   --> mech3
        # mech4 ---> mech5 --

        # 1x2=2 ---> 2x2=4 --
        #                   --> (4+4)x2=16
        # 1x2=2 ---> 2x2=4 --

        # create a PathwayComposition | blank slate for composition
        myPath = PathwayComposition()

        # create mechanisms to add to myPath
        myMech1 = IntegratorMechanism(function=Linear(slope=2.0))  # 1 x 2 = 2
        myMech2 = TransferMechanism(function=Linear(slope=2.0))  # 2 x 2 = 4
        myMech3 = TransferMechanism(function=Linear(slope=2.0))  # 4 x 2 = 8

        # add mechanisms to myPath with default MappingProjections between them
        myPath.add_linear_processing_pathway([myMech1, myMech2, myMech3])

        # analyze graph (assign roles)
        myPath._analyze_graph()
        myPathScheduler = Scheduler(composition=myPath)
        myPathScheduler.add_condition(myMech2, AfterNCalls(myMech1, 2))

        myPath.run(inputs={myMech1: [[1]]}, scheduler_processing=myPathScheduler)
        myPath.run(inputs={myMech1: [[1]]}, scheduler_processing=myPathScheduler)
        myPath2 = PathwayComposition()
        myMech4 = TransferMechanism(function=Linear(slope=2.0))  # 1 x 2 = 2
        myMech5 = TransferMechanism(function=Linear(slope=2.0))  # 2 x 2 = 4
        myPath2.add_linear_processing_pathway([myMech4, myMech5, myMech3])
        myPath2._analyze_graph()

        sys = SystemComposition()
        sys.add_pathway(myPath)
        sys.add_pathway(myPath2)
        # assign input to origin mechs
        stimulus = {myMech1: [[1]], myMech4: [[1]]}

        # schedule = Scheduler(composition=sys)
        output = sys.run(
            inputs=stimulus,
            # scheduler_processing=schedule
        )
        assert 16 == output[0][0]


class TestCompositionInterface:

    def test_one_input_state_per_origin_two_origins(self):

        # 5 -#1-> A --^ --> C --
        #                       ==> E
        # 5 ----> B ------> D --

        # 5 x 1 = 5 ----> 5 x 5 = 25 --
        #                                25 + 25 = 50  ==> 50 * 5 = 250
        # 5 * 1 = 5 ----> 5 x 5 = 25 --

        comp = Composition()
        A = TransferMechanism(name="composition-pytests-A",
                              function=Linear(slope=1.0)
                              )

        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=1.0))
        C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=5.0))
        D = TransferMechanism(name="composition-pytests-D", function=Linear(slope=5.0))
        E = TransferMechanism(name="composition-pytests-E", function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_mechanism(C)
        comp.add_mechanism(D)
        comp.add_projection(A, MappingProjection(sender=A, receiver=C), C)
        comp.add_projection(B, MappingProjection(sender=B, receiver=D), D)
        comp.add_mechanism(E)
        comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
        comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
        comp._analyze_graph()
        inputs_dict = {
            A: [[5.]],
            # two trials of one input state each
            #        TRIAL 1     TRIAL 2
            # A : [ [ [0,0] ] , [ [0, 0] ]  ]

            # two trials of multiple input states each
            #        TRIAL 1     TRIAL 2

            #       TRIAL1 IS1      IS2      IS3     TRIAL2    IS1      IS2
            # A : [ [     [0,0], [0,0,0], [0,0,0,0] ] ,     [ [0, 0],   [0] ]  ]
            B: [[5.]]
        }
        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )

        assert 250 == output[0][0]

    def test_updating_input_values_for_second_execution(self):
        # 5 -#1-> A --^ --> C --
        #                       ==> E
        # 5 ----> B ------> D --

        # 5 x 1 = 5 ----> 5 x 5 = 25 --
        #                                25 + 25 = 50  ==> 50 * 5 = 250
        # 5 * 1 = 5 ----> 5 x 5 = 25 --

        comp = Composition()
        A = TransferMechanism(name="composition-pytests-A",
                              function=Linear(slope=1.0)
                              )

        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=1.0))
        C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=5.0))
        D = TransferMechanism(name="composition-pytests-D", function=Linear(slope=5.0))
        E = TransferMechanism(name="composition-pytests-E", function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_mechanism(C)
        comp.add_mechanism(D)
        comp.add_projection(A, MappingProjection(sender=A, receiver=C), C)
        comp.add_projection(B, MappingProjection(sender=B, receiver=D), D)
        comp.add_mechanism(E)
        comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
        comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
        comp._analyze_graph()
        inputs_dict = {
            A: [[5.]],
            B: [[5.]]
        }
        sched = Scheduler(composition=comp)

        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )

        inputs_dict2 = {
            A: [[2.]],
            B: [[5.]],
            # two trials of one input state each
            #        TRIAL 1     TRIAL 2
            # A : [ [ [0,0] ] , [ [0, 0] ]  ]

            # two trials of multiple input states each
            #        TRIAL 1     TRIAL 2

            #       TRIAL1 IS1      IS2      IS3     TRIAL2    IS1      IS2
            # A : [ [     [0,0], [0,0,0], [0,0,0,0] ] ,     [ [0, 0],   [0] ]  ]
            B: [[5.]]
        }
        sched = Scheduler(composition=comp)

        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )

        # add a new branch to the composition
        F = TransferMechanism(name="composition-pytests-F", function=Linear(slope=2.0))
        G = TransferMechanism(name="composition-pytests-G", function=Linear(slope=2.0))
        comp.add_mechanism(F)
        comp.add_mechanism(G)
        comp.add_projection(sender=F, projection=MappingProjection(sender=F, receiver=G), receiver=G)
        comp.add_projection(sender=G, projection=MappingProjection(sender=G, receiver=E), receiver=E)

        # reassign roles
        comp._analyze_graph()

        # execute the updated composition
        inputs_dict2 = {
            A: [[1.]],
            B: [[2.]],
            F: [[3.]]
        }

        sched = Scheduler(composition=comp)
        output2 = comp.run(
            inputs=inputs_dict2,
            scheduler_processing=sched
        )

        assert 250 == output[0][0]
        assert 135 == output2[0][0]

    def test_changing_origin_for_second_execution(self):

        comp = Composition()
        A = TransferMechanism(name="composition-pytests-A",
                              function=Linear(slope=1.0)
                              )

        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=1.0))
        C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_mechanism(C)
        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp.add_projection(B, MappingProjection(sender=B, receiver=C), C)
        comp._analyze_graph()
        inputs_dict = {A: [[5.]]}
        sched = Scheduler(composition=comp)

        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )

        assert 25 == output[0][0]

        # add a new origin to the composition
        F = TransferMechanism(name="composition-pytests-F", function=Linear(slope=2.0))
        comp.add_mechanism(F)
        comp.add_projection(sender=F, projection=MappingProjection(sender=F, receiver=A), receiver=A)

        # reassign roles
        comp._analyze_graph()

        # execute the updated composition
        inputs_dict2 = {F: [[3.]]}

        sched = Scheduler(composition=comp)
        output2 = comp.run(
            inputs=inputs_dict2,
            scheduler_processing=sched
        )

        connections_to_A = []
        expected_connections_to_A = [(F.output_states[0], A.input_states[0])]
        for input_state in A.input_states:
            for p_a in input_state.path_afferents:
                connections_to_A.append((p_a.sender, p_a.receiver))

        assert connections_to_A == expected_connections_to_A
        assert 30 == output2[0][0]

    def test_two_input_states_new_inputs_second_trial(self):

        comp = Composition()
        my_fun = Linear(
            # default_variable=[[0], [0]],
            # ^ setting default_variable on the function actually does not matter -- does the mechanism update it?
            slope=1.0)
        A = TransferMechanism(name="composition-pytests-A",
                              default_variable=[[0], [0]],
                              input_states=[{NAME: "Input State 1",
                                             },
                                            {NAME: "Input State 2",
                                             }],
                              function=my_fun
                              )
        comp.add_mechanism(A)
        comp._analyze_graph()
        inputs_dict = {A: [[5.], [5.]]}

        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )

        inputs_dict2 = {A: [[2.], [4.]]}

        output2 = comp.run(
            inputs=inputs_dict2,
            scheduler_processing=sched
        )

        assert np.allclose(A.input_states[0].value, [2.])
        assert np.allclose(A.input_states[1].value, [4.])
        assert np.allclose(A.variable, [[2.], [4.]])
        assert np.allclose(output, [[5.], [5.]])
        assert np.allclose(output2, [[2.], [4.]])

    def test_two_input_states_new_origin_second_trial(self):

        # A --> B --> C

        comp = Composition()
        my_fun = Linear(
            # default_variable=[[0], [0]],
            # ^ setting default_variable on the function actually does not matter -- does the mechanism update it?
            slope=1.0)
        A = TransferMechanism(
            name="composition-pytests-A",
            default_variable=[[0], [0]],
            input_states=[
                {NAME: "Input State 1", },
                {NAME: "Input State 2", }
            ],
            function=my_fun
        )

        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=2.0))
        C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_mechanism(C)
        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp.add_projection(B, MappingProjection(sender=B, receiver=C), C)
        comp._analyze_graph()

        inputs_dict = {A: [[5.], [5.]]}

        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )
        assert np.allclose(A.input_states[0].value, [5.])
        assert np.allclose(A.input_states[1].value, [5.])
        assert np.allclose(A.variable, [[5.], [5.]])
        assert np.allclose(output, [[50.]])

        # A --> B --> C
        #     ^
        # D __|

        D = TransferMechanism(
            name="composition-pytests-D",
            default_variable=[[0], [0]],
            input_states=[
                {NAME: "Input State 1", },
                {NAME: "Input State 2", }
            ],
            function=my_fun
        )
        comp.add_mechanism(D)
        comp.add_projection(D, MappingProjection(sender=D, receiver=B), B)
        # Need to analyze graph again (identify D as an origin so that we can assign input) AND create the scheduler
        # again (sched, even though it is tied to comp, will not update according to changes in comp)
        comp._analyze_graph()
        sched = Scheduler(composition=comp)

        inputs_dict2 = {A: [[2.], [4.]],
                        D: [[2.], [4.]]}
        output2 = comp.run(
            inputs=inputs_dict2,
            scheduler_processing=sched
        )
        assert np.allclose(A.input_states[0].value, [2.])
        assert np.allclose(A.input_states[1].value, [4.])
        assert np.allclose(A.variable, [[2.], [4.]])

        assert np.allclose(D.input_states[0].value, [2.])
        assert np.allclose(D.input_states[1].value, [4.])
        assert np.allclose(D.variable, [[2.], [4.]])

        assert np.allclose(output2, [[40]])

    def test_output_cim_one_terminal_mechanism_multiple_output_states(self):

        comp = Composition()
        A = TransferMechanism(name="composition-pytests-A",
                              function=Linear(slope=1.0))
        B = TransferMechanism(name="composition-pytests-B",
                              function=Linear(slope=1.0))
        C = TransferMechanism(name="composition-pytests-C",
                              function=Linear(slope=2.0),
                              output_states=[TRANSFER_OUTPUT.RESULT,
                                             TRANSFER_OUTPUT.VARIANCE])
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_mechanism(C)

        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp.add_projection(B, MappingProjection(sender=B, receiver=C), C)

        comp._analyze_graph()
        comp.run(inputs={A: [1.0]})

        for CIM_output_state in comp.output_CIM_output_states:
            # all CIM output state keys in the CIM --> Terminal mapping dict are on the actual output CIM
            assert comp.output_CIM_output_states[CIM_output_state] in comp.output_CIM.output_states

        # all Terminal Output states are in the CIM --> Terminal mapping dict
        assert C.output_states[0] in comp.output_CIM_output_states.keys()
        assert C.output_states[1] in comp.output_CIM_output_states.keys()

        # May change to 2 in the future if we get rid of the original primary output state
        assert len(comp.output_CIM.output_states) == 3

    def test_output_cim_many_terminal_mechanisms(self):

        comp = Composition()
        A = TransferMechanism(name="composition-pytests-A",
                              function=Linear(slope=1.0))
        B = TransferMechanism(name="composition-pytests-B",
                              function=Linear(slope=1.0))
        C = TransferMechanism(name="composition-pytests-C",
                              function=Linear(slope=2.0))
        D = TransferMechanism(name="composition-pytests-D",
                              function=Linear(slope=3.0))
        E = TransferMechanism(name="composition-pytests-E",
                              function=Linear(slope=4.0),
                              output_states=[TRANSFER_OUTPUT.RESULT,
                                             TRANSFER_OUTPUT.VARIANCE])
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_mechanism(C)
        comp.add_mechanism(D)
        comp.add_mechanism(E)
        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp.add_projection(B, MappingProjection(sender=B, receiver=C), C)
        comp.add_projection(B, MappingProjection(sender=B, receiver=D), D)
        comp.add_projection(B, MappingProjection(sender=B, receiver=E), E)
        comp._analyze_graph()
        comp.run(inputs={A: [1.0]})

        for CIM_output_state in comp.output_CIM_output_states:
            # all CIM output state keys in the CIM --> Terminal mapping dict are on the actual output CIM
            assert comp.output_CIM_output_states[CIM_output_state] in comp.output_CIM.output_states

        # all Terminal Output states are in the CIM --> Terminal mapping dict
        assert C.output_state in comp.output_CIM_output_states.keys()
        assert D.output_state in comp.output_CIM_output_states.keys()
        assert E.output_states[0] in comp.output_CIM_output_states.keys()
        assert E.output_states[1] in comp.output_CIM_output_states.keys()

        # May change to 4 in the future if we get rid of the original primary output state
        assert len(comp.output_CIM.output_states) == 5


class TestInputStateSpecifications:

    def test_two_input_states_created_with_dictionaries(self):

        comp = Composition()
        A = ProcessingMechanism(
            name="composition-pytests-A",
            default_variable=[[0], [0]],
            # input_states=[
            #     {NAME: "Input State 1", },
            #     {NAME: "Input State 2", }
            # ],
            function=Linear(slope=1.0)
            # specifying default_variable on the function doesn't seem to matter?
        )

        comp.add_mechanism(A)

        comp._analyze_graph()

        inputs_dict = {A: [[2.], [4.]]}
        sched = Scheduler(composition=comp)
        comp.run(inputs=inputs_dict,
                 scheduler_processing=sched)

        assert np.allclose(A.input_states[0].value, [2.0])
        assert np.allclose(A.input_states[1].value, [4.0])
        assert np.allclose(A.variable, [[2.0], [4.0]])

    def test_two_input_states_created_first_with_deferred_init(self):
        comp = Composition()

        # create mechanism A
        I1 = InputState(
            name="Input State 1",
            reference_value=[0]
        )
        I2 = InputState(
            name="Input State 2",
            reference_value=[0]
        )
        A = TransferMechanism(
            name="composition-pytests-A",
            default_variable=[[0], [0]],
            input_states=[I1, I2],
            function=Linear(slope=1.0)
        )

        # add mech A to composition
        comp.add_mechanism(A)

        # get comp ready to run (identify roles, create sched, assign inputs)
        comp._analyze_graph()
        inputs_dict = { A: [[2.],[4.]]}

        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )

        assert np.allclose(A.input_states[0].value, [2.0])
        assert np.allclose(A.input_states[1].value, [4.0])
        assert np.allclose(A.variable, [[2.0], [4.0]])

    def test_two_input_states_created_with_keyword(self):
        comp = Composition()

        # create mechanism A

        A = TransferMechanism(
            name="composition-pytests-A",
            default_variable=[[0], [0]],
            input_states=[INPUT_STATE, INPUT_STATE],
            function=Linear(slope=1.0)
        )

        # add mech A to composition
        comp.add_mechanism(A)

        # get comp ready to run (identify roles, create sched, assign inputs)
        comp._analyze_graph()
        inputs_dict = {A: [[2.], [4.]]}

        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )

        assert np.allclose(A.input_states[0].value, [2.0])
        assert np.allclose(A.input_states[1].value, [4.0])
        assert np.allclose(A.variable, [[2.0], [4.0]])

        assert 2 == output[0][0]

    def test_two_input_states_created_with_strings(self):
        comp = Composition()

        # create mechanism A

        A = TransferMechanism(
            name="composition-pytests-A",
            default_variable=[[0], [0]],
            input_states=["Input State 1", "Input State 2"],
            function=Linear(slope=1.0)
        )

        # add mech A to composition
        comp.add_mechanism(A)

        # get comp ready to run (identify roles, create sched, assign inputs)
        comp._analyze_graph()

        inputs_dict = {A: [[2.], [4.]]}

        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )

        assert np.allclose(A.input_states[0].value, [2.0])
        assert np.allclose(A.input_states[1].value, [4.0])
        assert np.allclose(A.variable, [[2.0], [4.0]])

    def test_two_input_states_created_with_values(self):
        comp = Composition()

        # create mechanism A

        A = TransferMechanism(
            name="composition-pytests-A",
            default_variable=[[0], [0]],
            input_states=[[0.], [0.]],
            function=Linear(slope=1.0)
        )

        # add mech A to composition
        comp.add_mechanism(A)

        # get comp ready to run (identify roles, create sched, assign inputs)
        comp._analyze_graph()
        inputs_dict = {A: [[2.], [4.]]}

        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )

        assert np.allclose(A.input_states[0].value, [2.0])
        assert np.allclose(A.input_states[1].value, [4.0])
        assert np.allclose(A.variable, [[2.0], [4.0]])


class TestInputSpecifications:

    # def test_2_mechanisms_default_input_1(self):
    #     comp = Composition()
    #     A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
    #     B = TransferMechanism(function=Linear(slope=5.0))
    #     comp.add_mechanism(A)
    #     comp.add_mechanism(B)
    #     comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
    #     comp._analyze_graph()
    #     sched = Scheduler(composition=comp)
    #     output = comp.run(
    #         scheduler_processing=sched
    #     )
    #     assert 25 == output[0][0]

    def test_3_origins(self):
        comp = Composition()
        I1 = InputState(
                        name="Input State 1",
                        reference_value=[0]
        )
        I2 = InputState(
                        name="Input State 2",
                        reference_value=[0]
        )
        A = TransferMechanism(
                            name="composition-pytests-A",
                            default_variable=[[0], [0]],
                            input_states=[I1, I2],
                            function=Linear(slope=1.0)
        )
        B = TransferMechanism(
                            name="composition-pytests-B",
                            default_variable=[0,0],
                            function=Linear(slope=1.0))
        C = TransferMechanism(
                            name="composition-pytests-C",
                            default_variable=[0, 0, 0],
                            function=Linear(slope=1.0))
        D = TransferMechanism(
                            name="composition-pytests-D",
                            default_variable=[0],
                            function=Linear(slope=1.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_mechanism(C)
        comp.add_mechanism(D)
        comp.add_projection(A, MappingProjection(sender=A, receiver=D), D)
        comp.add_projection(B, MappingProjection(sender=B, receiver=D), D)
        comp.add_projection(C, MappingProjection(sender=C, receiver=D), D)
        comp._analyze_graph()
        inputs = {A: [[[0],[0]], [[1],[1]], [[2],[2]]],
                  B: [[0,0], [1,1], [2,2]],
                  C: [[0,0,0], [1,1,1], [2,2,2]]

        }
        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs,
            scheduler_processing=sched
        )
        assert 12 == output[0][0]

    def test_2_mechanisms_input_5(self):
        comp = Composition()
        A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
        B = TransferMechanism(function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp._analyze_graph()
        inputs_dict = {A: [[5]]}
        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )
        assert 125 == output[0][0]

    def test_run_2_mechanisms_reuse_input(self):
        comp = Composition()
        A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
        B = TransferMechanism(function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp._analyze_graph()
        inputs_dict = {A: [[5]]}
        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched,
            num_trials=5
        )
        assert 125 == output[0][0]

