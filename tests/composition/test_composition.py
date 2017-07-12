import logging

from timeit import timeit

import pytest

from PsyNeuLink.Components.Functions.Function import Linear, SimpleIntegrator
from PsyNeuLink.Components.Mechanisms.Mechanism import mechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.IntegratorMechanism import IntegratorMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Composition import Composition, CompositionError, MechanismRole
from PsyNeuLink.Scheduling.Condition import EveryNCalls
from PsyNeuLink.Scheduling.Scheduler import Scheduler

logger = logging.getLogger(__name__)

# All tests are set to run. If you need to skip certain tests,
# see http://doc.pytest.org/en/latest/skipping.html


# Unit tests for each function of the Composition class #######################
# Unit tests for Composition.Composition()
@pytest.mark.skip
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
        t = timeit('comp = Composition()', setup='from PsyNeuLink.Composition import Composition', number=count)
        print()
        logger.info('completed {0} creation{2} of Composition() in {1:.8f}s'.format(count, t, 's' if count != 1 else ''))


# Unit tests for Composition.add_mechanism
@pytest.mark.skip
class TestAddMechanism:

    def test_add_once(self):
        comp = Composition()
        comp.add_mechanism(mechanism())

    def test_add_twice(self):
        comp = Composition()
        comp.add_mechanism(mechanism())
        comp.add_mechanism(mechanism())

    def test_add_same_twice(self):
        comp = Composition()
        mech = mechanism()
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
            'comp.add_mechanism(mechanism())',
            setup='''
from PsyNeuLink.Components.Mechanisms.Mechanism import mechanism
from PsyNeuLink.Composition import Composition
comp = Composition()
''',
            number=count
        )
        print()
        logger.info('completed {0} addition{2} of a mechanism to a composition in {1:.8f}s'.format(count, t, 's' if count != 1 else ''))


# Unit tests for Composition.add_projection
@pytest.mark.skip
class TestAddProjection:

    def test_add_once(self):
        comp = Composition()
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)

    def test_add_twice(self):
        comp = Composition()
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(A, MappingProjection(), B)

    def test_add_same_twice(self):
        comp = Composition()
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B')
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
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Composition import Composition
comp = Composition()
A = TransferMechanism(name='A')
B = TransferMechanism(name='B')
comp.add_mechanism(A)
comp.add_mechanism(B)
''',
                   number=count
                   )
        print()
        logger.info('completed {0} addition{2} of a projection to a composition in {1:.8f}s'.format(count, t, 's' if count != 1 else ''))


@pytest.mark.skip
class TestAnalyzeGraph:

    def test_empty_call(self):
        comp = Composition()
        comp._analyze_graph()

    def test_singleton(self):
        comp = Composition()
        A = TransferMechanism(name='A')
        comp.add_mechanism(A)
        comp._analyze_graph()
        assert A in comp.get_mechanisms_by_role(MechanismRole.ORIGIN)
        assert A in comp.get_mechanisms_by_role(MechanismRole.TERMINAL)

    def test_two_independent(self):
        comp = Composition()
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp._analyze_graph()
        assert A in comp.get_mechanisms_by_role(MechanismRole.ORIGIN)
        assert B in comp.get_mechanisms_by_role(MechanismRole.ORIGIN)
        assert A in comp.get_mechanisms_by_role(MechanismRole.TERMINAL)
        assert B in comp.get_mechanisms_by_role(MechanismRole.TERMINAL)

    def test_two_in_a_row(self):
        comp = Composition()
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B')
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
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B')
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
    def test_two_origins_pointing_to_recursive_pair(self):
        comp = Composition()
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B')
        C = TransferMechanism(name='C')
        D = TransferMechanism(name='D')
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


@pytest.mark.skip
class TestValidateFeedDict:

    def test_empty_feed_dicts(self):
        comp = Composition()
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B')
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
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B')
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
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B')
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
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B')
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
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B')
        C = TransferMechanism(name='C')
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
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B')
        C = TransferMechanism(name='C')
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
        A = TransferMechanism(default_input_value=[0, 1, 2], name='A')
        B = TransferMechanism(default_input_value=[0, 1, 2], name='B')
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
        A = TransferMechanism(default_input_value=[0, 1, 2], name='A')
        B = TransferMechanism(default_input_value=[0, 1, 2], name='B')
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
        A = TransferMechanism(default_input_value=[0, 1], name='A')
        B = TransferMechanism(default_input_value=[0, 1], name='B')
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
        A = TransferMechanism(default_input_value=[0], name='A')
        B = TransferMechanism(default_input_value=[0], name='B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp._analyze_graph()
        feed_dict_origin = {A: [[0]], B: [[0]]}
        with pytest.raises(ValueError):
            comp._validate_feed_dict(feed_dict_origin, comp.get_mechanisms_by_role(MechanismRole.ORIGIN), "origin")

    def test_input_state_len_3_brackets_extra_1(self):
        comp = Composition()
        A = TransferMechanism(default_input_value=[0, 1, 2], name='A')
        B = TransferMechanism(default_input_value=[0, 1, 2], name='B')
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
        A = TransferMechanism(default_input_value=[0, 1, 2], name='A')
        B = TransferMechanism(default_input_value=[0, 1, 2], name='B')
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
        A = TransferMechanism(default_input_value=[0], name='A')
        B = TransferMechanism(default_input_value=[0], name='B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp._analyze_graph()
        feed_dict_origin = {A: [[0]]}
        feed_dict_monitored = {}
        comp._validate_feed_dict(feed_dict_monitored, comp.get_mechanisms_by_role(MechanismRole.MONITORED), "monitored")

    def test_mech_in_feed_dict_for_empty_type(self):
        comp = Composition()
        A = TransferMechanism(default_input_value=[0])
        B = TransferMechanism(name='B')
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
        A = TransferMechanism(default_input_value=[0])
        comp.add_mechanism(A)
        comp._analyze_graph()
        feed_dict_origin = {A: [[0]]}
        feed_dict_terminal = {A: [[0]]}
        comp._validate_feed_dict(feed_dict_origin, comp.get_mechanisms_by_role(MechanismRole.ORIGIN), "origin")

    def test_one_mech_2(self):
        comp = Composition()
        A = TransferMechanism(default_input_value=[0])
        comp.add_mechanism(A)
        comp._analyze_graph()
        feed_dict_origin = {A: [[0]]}
        feed_dict_terminal = {A: [[0]]}
        comp._validate_feed_dict(feed_dict_terminal, comp.get_mechanisms_by_role(MechanismRole.TERMINAL), "terminal")

    def test_multiple_time_steps_1(self):
        comp = Composition()
        A = TransferMechanism(default_input_value=[[0, 1, 2]], name='A')
        B = TransferMechanism(default_input_value=[[0, 1, 2]], name='B')
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
        A = TransferMechanism(default_input_value=[[0, 1, 2]], name='A')
        B = TransferMechanism(default_input_value=[[0, 1, 2]], name='B')
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp._analyze_graph()
        feed_dict_origin = {A: [[[0, 1, 2]], [[0, 1, 2]]]}
        feed_dict_terminal = {B: [[0, 1, 2]]}
        comp._validate_feed_dict(feed_dict_origin, comp.get_mechanisms_by_role(MechanismRole.ORIGIN), "origin")
        comp._validate_feed_dict(feed_dict_terminal, comp.get_mechanisms_by_role(MechanismRole.TERMINAL), "terminal")


@pytest.mark.skip
class TestGetMechanismsByRole:

    def test_multiple_roles(self):

        comp = Composition()
        mechs = [mechanism() for x in range(4)]

        for mech in mechs:
            comp.add_mechanism(mech)

        comp._add_mechanism_role(mechs[0], MechanismRole.ORIGIN)
        comp._add_mechanism_role(mechs[1], MechanismRole.INTERNAL)
        comp._add_mechanism_role(mechs[2], MechanismRole.INTERNAL)
        comp._add_mechanism_role(mechs[3], MechanismRole.CYCLE)

        for role in list(MechanismRole):
            if role is MechanismRole.ORIGIN:
                assert comp.get_mechanisms_by_role(role) == {mechs[0]}
            elif role is MechanismRole.INTERNAL:
                assert comp.get_mechanisms_by_role(role) == set([mechs[1], mechs[2]])
            elif role is MechanismRole.CYCLE:
                assert comp.get_mechanisms_by_role(role) == {mechs[3]}
            else:
                assert comp.get_mechanisms_by_role(role) == set()

    def test_nonexistent_role(self):

        comp = Composition()

        with pytest.raises(CompositionError):
            comp.get_mechanisms_by_role(None)


@pytest.mark.skip
class TestGraph:

    @pytest.mark.skip
    class TestProcessingGraph:

        def test_all_mechanisms(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            B = TransferMechanism(function=Linear(intercept=4.0), name='B')
            C = TransferMechanism(function=Linear(intercept=1.5), name='C')
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
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            B = TransferMechanism(function=Linear(intercept=4.0), name='B')
            C = TransferMechanism(function=Linear(intercept=1.5), name='C')
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
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            B = TransferMechanism(function=Linear(intercept=4.0), name='B')
            C = TransferMechanism(function=Linear(intercept=1.5), name='C')
            D = TransferMechanism(function=Linear(intercept=1.5), name='D')
            E = TransferMechanism(function=Linear(intercept=1.5), name='E')
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
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            B = TransferMechanism(function=Linear(intercept=4.0), name='B')
            C = TransferMechanism(function=Linear(intercept=1.5), name='C')
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
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            B = TransferMechanism(function=Linear(intercept=4.0), name='B')
            C = TransferMechanism(function=Linear(intercept=1.5), name='C')
            D = TransferMechanism(function=Linear(intercept=1.5), name='D')
            E = TransferMechanism(function=Linear(intercept=1.5), name='E')
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
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='A')
            B = TransferMechanism(function=Linear(intercept=4.0), name='B')
            C = TransferMechanism(function=Linear(intercept=1.5), name='C')
            D = TransferMechanism(function=Linear(intercept=1.5), name='D')
            E = TransferMechanism(function=Linear(intercept=1.5), name='E')
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


@pytest.mark.skip
class TestRun:

    def test_run_2_mechanisms_default_input_1(self):
        comp = Composition()
        A = IntegratorMechanism(default_input_value=1.0, function=Linear(slope=5.0))
        B = TransferMechanism(function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp._analyze_graph()
        sched = Scheduler(composition=comp)
        output = comp.run(
            scheduler_processing=sched
        )
        assert 25 == output[0][0]

    def test_run_2_mechanisms_input_5(self):
        comp = Composition()
        A = IntegratorMechanism(default_input_value=1.0, function=Linear(slope=5.0))
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
        A = TransferMechanism(name="A", function=Linear(slope=1.0))
        B = TransferMechanism(name="B", function=Linear(slope=1.0))
        C = TransferMechanism(name="C", function=Linear(slope=5.0))
        D = TransferMechanism(name="D", function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_mechanism(C)
        comp.add_mechanism(D)
        comp.add_projection(A, MappingProjection(sender=A, receiver=C), C)
        with pytest.raises(CompositionError) as error_text:
            comp.add_projection(B, MappingProjection(sender=B, receiver=D), C)

        assert "is incompatible with the positions of these components in their composition" in str(error_text.value)

    def test_projection_assignment_mistake_swap2(self):
        # A ----> C --
        #              ==> E
        # B ----> D --

        comp = Composition()
        A = TransferMechanism(name="A", function=Linear(slope=1.0))
        B = TransferMechanism(name="B", function=Linear(slope=1.0))
        C = TransferMechanism(name="C", function=Linear(slope=5.0))
        D = TransferMechanism(name="D", function=Linear(slope=5.0))
        E = TransferMechanism(name="E", function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_mechanism(C)
        comp.add_mechanism(D)
        comp.add_projection(A, MappingProjection(sender=A, receiver=C), C)
        with pytest.raises(CompositionError) as error_text:
            comp.add_projection(B, MappingProjection(sender=B, receiver=C), D)

        assert "is incompatible with the positions of these components in their composition" in str(error_text.value)

    def test_run_5_mechanisms_2_origins_1_terminal(self):
        # A ----> C --
        #              ==> E
        # B ----> D --

        # 5 x 1 = 5 ----> 5 x 5 = 25 --
        #                                25 + 25 = 50  ==> 50 * 5 = 250
        # 5 * 1 = 5 ----> 5 x 5 = 25 --

        comp = Composition()
        A = TransferMechanism(name="A", function=Linear(slope=1.0))
        B = TransferMechanism(name="B", function=Linear(slope=1.0))
        C = TransferMechanism(name="C", function=Linear(slope=5.0))
        D = TransferMechanism(name="D", function=Linear(slope=5.0))
        E = TransferMechanism(name="E", function=Linear(slope=5.0))
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

        A = IntegratorMechanism(name="A [integrator]", default_input_value=2.0, function=SimpleIntegrator(rate=1.0))
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
        A = IntegratorMechanism(default_input_value=1.0, function=Linear(slope=5.0))
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

    def test_run_2_mechanisms_incorrect_trial_spec(self):
        comp = Composition()
        A = IntegratorMechanism(default_input_value=1.0, function=Linear(slope=5.0))
        B = TransferMechanism(function=Linear(slope=5.0))
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp._analyze_graph()
        inputs_dict = {A: [[5], [4], [3]]}
        sched = Scheduler(composition=comp)
        with pytest.raises(CompositionError) as error_text:
            comp.run(
                inputs=inputs_dict,
                scheduler_processing=sched,
                num_trials=5
            )
        assert "number of trials" in str(error_text.value) and "does not match the length" in str(error_text.value)

    def test_run_2_mechanisms_double_trial_specs(self):
        comp = Composition()
        A = IntegratorMechanism(default_input_value=1.0, function=Linear(slope=5.0))
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
        A = IntegratorMechanism(default_input_value=1.0, function=Linear(slope=5.0))
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
        A = TransferMechanism(name="A", function=Linear(slope=2.0))   # 1 x 2 = 2
        B = TransferMechanism(name="B", function=Linear(slope=2.0))   # 2 x 2 = 4
        C = TransferMechanism(name="C", function=Linear(slope=2.0))   # 4 x 2 = 8
        D = TransferMechanism(name="D", function=Linear(slope=2.0))   # 8 x 2 = 16
        E = TransferMechanism(name="E", function=Linear(slope=2.0))  # 16 x 2 = 32
        comp.add_linear_processing_pathway([A,B,C,D,E])
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
        A = TransferMechanism(name="A", function=Linear(slope=2.0))  # 1 x 2 = 2
        B = TransferMechanism(name="B", function=Linear(slope=2.0))  # 2 x 2 = 4
        C = TransferMechanism(name="C", function=Linear(slope=2.0))  # 4 x 2 = 8
        D = TransferMechanism(name="D", function=Linear(slope=2.0))  # 8 x 2 = 16
        E = TransferMechanism(name="E", function=Linear(slope=2.0))  # 16 x 2 = 32
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
        A = TransferMechanism(name="A", function=Linear(slope=2.0))
        B = TransferMechanism(name="B", function=Linear(slope=2.0))
        C = TransferMechanism(name="C", function=Linear(slope=2.0))
        D = TransferMechanism(name="D", function=Linear(slope=2.0))
        E = TransferMechanism(name="E", function=Linear(slope=2.0))
        A_to_B = MappingProjection(sender=A, receiver=B)
        D_to_E = MappingProjection(sender=D, receiver=E)
        with pytest.raises(CompositionError) as error_text:
            comp.add_linear_processing_pathway([A, A_to_B, B, C, D, E, D_to_E])

        assert "A projection cannot be the last item in a linear processing pathway." in str(error_text.value)

    def test_LPP_two_projections_in_a_row(self):
        comp = Composition()
        A = TransferMechanism(name="A", function=Linear(slope=2.0))
        B = TransferMechanism(name="B", function=Linear(slope=2.0))
        C = TransferMechanism(name="C", function=Linear(slope=2.0))
        A_to_B = MappingProjection(sender=A, receiver=B)
        B_to_C = MappingProjection(sender=B, receiver=C)
        with pytest.raises(CompositionError) as error_text:
            comp.add_linear_processing_pathway([A, B_to_C, A_to_B, B, C])

        assert "A projection in a linear processing pathway must be preceded by a mechanism and followed by a mechanism" \
               in str(error_text.value)

    def test_LPP_start_with_projection(self):
        comp = Composition()
        Nonsense_Projection = MappingProjection()
        A = TransferMechanism(name="A", function=Linear(slope=2.0))
        B = TransferMechanism(name="B", function=Linear(slope=2.0))
        with pytest.raises(CompositionError) as error_text:
            comp.add_linear_processing_pathway([Nonsense_Projection, A, B])

        assert "The first item in a linear processing pathway must be a mechanism." in str(
            error_text.value)

    def test_LPP_wrong_component(self):
        comp = Composition()
        Nonsense = "string"
        A = TransferMechanism(name="A", function=Linear(slope=2.0))
        B = TransferMechanism(name="B", function=Linear(slope=2.0))
        with pytest.raises(CompositionError) as error_text:
            comp.add_linear_processing_pathway([ A, Nonsense, B])

        assert "A linear processing pathway must be made up of projections and mechanisms." in str(
            error_text.value)

    def test_LPP_two_origins_one_terminal(self):
        # A ----> C --
        #              ==> E
        # B ----> D --

        # 5 x 1 = 5 ----> 5 x 5 = 25 --
        #                                25 + 25 = 50  ==> 50 * 5 = 250
        # 5 * 1 = 5 ----> 5 x 5 = 25 --

        comp = Composition()
        A = TransferMechanism(name="A", function=Linear(slope=1.0))
        B = TransferMechanism(name="B", function=Linear(slope=1.0))
        C = TransferMechanism(name="C", function=Linear(slope=5.0))
        D = TransferMechanism(name="D", function=Linear(slope=5.0))
        E = TransferMechanism(name="E", function=Linear(slope=5.0))
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





                        # when self.sched is ready:
    # def test_run_default_scheduler(self):
    #     comp = Composition()
    #     A = IntegratorMechanism(default_input_value=1.0, function=Linear(slope=5.0))
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
    #         default_input_value=np.zeros((2,)),
    #     )
    #
    #     Hidden_Layer_1 = TransferMechanism(
    #         name='Hidden Layer_1',
    #         function=Logistic(),
    #         default_input_value=np.zeros((5,)),
    #     )
    #
    #     Hidden_Layer_2 = TransferMechanism(
    #         name='Hidden Layer_2',
    #         function=Logistic(),
    #         default_input_value=[0, 0, 0, 0],
    #     )
    #
    #     Output_Layer = TransferMechanism(
    #         name='Output Layer',
    #         function=Logistic,
    #         default_input_value=[0, 0, 0],
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
    #     #     default_input_value=[0, 0],
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
