import logging
import pytest

from timeit import timeit

from PsyNeuLink.composition import Composition
from PsyNeuLink.Components.Mechanisms.Mechanism import mechanism
from PsyNeuLink.Components.Projections.TransmissiveProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism

logger = logging.getLogger(__name__)

# All tests are set to run. If you need to skip certain tests,
# see http://doc.pytest.org/en/latest/skipping.html


# Unit tests for each function of the Composition class #######################
# Unit tests for Composition.Composition()
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
        t = timeit('comp = Composition()', setup='from PsyNeuLink.composition import Composition', number=count)
        print()
        logger.info('completed {0} creation{2} of Composition() in {1:.8f}s'.format(count, t, 's' if count != 1 else ''))


# Unit tests for Composition.add_mechanism
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
from PsyNeuLink.composition import Composition
from PsyNeuLink.Components.Mechanisms.Mechanism import mechanism
comp = Composition()
''',
            number=count
        )
        print()
        logger.info('completed {0} addition{2} of a mechanism to a composition in {1:.8f}s'.format(count, t, 's' if count != 1 else ''))


# Unit tests for Composition.add_projection
class TestAddProjection:

    def test_add_once(self):
        comp = Composition()
        A = mechanism()
        B = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)

    def test_add_twice(self):
        comp = Composition()
        A = mechanism()
        B = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(A, MappingProjection(), B)

    def test_add_same_twice(self):
        comp = Composition()
        A = mechanism()
        B = mechanism()
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
from PsyNeuLink.composition import Composition
from PsyNeuLink.Components.Mechanisms.Mechanism import mechanism
from PsyNeuLink.Components.Projections.TransmissiveProjections.MappingProjection import MappingProjection
comp = Composition()
A = mechanism()
B = mechanism()
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
        comp.analyze_graph()

    def test_singleton(self):
        comp = Composition()
        A = mechanism()
        comp.add_mechanism(A)
        comp.analyze_graph()
        assert A in comp.graph.mechanisms
        assert A in comp.origin_mechanisms
        assert A in comp.terminal_mechanisms

    def test_two_independent(self):
        comp = Composition()
        A = mechanism()
        B = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.analyze_graph()
        assert A in comp.origin_mechanisms
        assert B in comp.origin_mechanisms
        assert A in comp.terminal_mechanisms
        assert B in comp.terminal_mechanisms

    def test_two_in_a_row(self):
        comp = Composition()
        A = mechanism()
        B = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp.analyze_graph()
        assert A in comp.origin_mechanisms
        assert B not in comp.origin_mechanisms
        assert A not in comp.terminal_mechanisms
        assert B in comp.terminal_mechanisms

    # (A)<->(B)
    def test_two_recursive(self):
        comp = Composition()
        A = mechanism()
        B = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(B, MappingProjection(), A)
        comp.analyze_graph()
        assert A not in comp.origin_mechanisms
        assert B not in comp.origin_mechanisms
        assert A not in comp.terminal_mechanisms
        assert B not in comp.terminal_mechanisms
        assert A in comp.cycle_mechanisms
        assert B in comp.recurrent_init_mechanisms

    # (A)->(B)<->(C)<-(D)
    def test_two_origins_pointing_to_recursive_pair(self):
        comp = Composition()
        A = mechanism()
        B = mechanism()
        C = mechanism()
        D = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_mechanism(C)
        comp.add_mechanism(D)
        comp.add_projection(A, MappingProjection(), B)
        comp.add_projection(C, MappingProjection(), B)
        comp.add_projection(B, MappingProjection(), C)
        comp.add_projection(D, MappingProjection(), C)
        comp.analyze_graph()
        assert A in comp.origin_mechanisms
        assert D in comp.origin_mechanisms
        assert B in comp.cycle_mechanisms
        assert C in comp.recurrent_init_mechanisms


class TestValidateFeedDict:

    def test_empty_feed_dicts(self):
        comp = Composition()
        A = mechanism()
        B = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp.analyze_graph()
        feed_dict_origin = {}
        feed_dict_terminal = {}
        comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
        comp.validate_feed_dict(feed_dict_terminal, comp.terminal_mechanisms, "terminal")

    def test_origin_and_terminal_with_mapping(self):
        comp = Composition()
        A = mechanism()
        B = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp.analyze_graph()
        feed_dict_origin = {A: [[0]]}
        feed_dict_terminal = {B: [[0]]}
        comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
        comp.validate_feed_dict(feed_dict_terminal, comp.terminal_mechanisms, "terminal")

    def test_origin_and_terminal_with_swapped_feed_dicts_1(self):
        comp = Composition()
        A = mechanism()
        B = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp.analyze_graph()
        feed_dict_origin = {B: [[0]]}
        feed_dict_terminal = {A: [[0]]}
        with pytest.raises(ValueError):
            comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")

    def test_origin_and_terminal_with_swapped_feed_dicts_2(self):
        comp = Composition()
        A = mechanism()
        B = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp.analyze_graph()
        feed_dict_origin = {B: [[0]]}
        feed_dict_terminal = {A: [[0]]}
        with pytest.raises(ValueError):
            comp.validate_feed_dict(feed_dict_terminal, comp.terminal_mechanisms, "terminal")

    def test_multiple_origin_mechs(self):
        comp = Composition()
        A = mechanism()
        B = mechanism()
        C = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_mechanism(C)
        comp.add_projection(A, MappingProjection(), C)
        comp.add_projection(B, MappingProjection(), C)
        comp.analyze_graph()
        feed_dict_origin = {A: [[0]], B: [[0]]}
        feed_dict_terminal = {C: [[0]]}
        comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
        comp.validate_feed_dict(feed_dict_terminal, comp.terminal_mechanisms, "terminal")

    def test_multiple_origin_mechs_only_one_in_feed_dict(self):
        comp = Composition()
        A = mechanism()
        B = mechanism()
        C = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_mechanism(C)
        comp.add_projection(A, MappingProjection(), C)
        comp.add_projection(B, MappingProjection(), C)
        comp.analyze_graph()
        feed_dict_origin = {B: [[0]]}
        feed_dict_terminal = {C: [[0]]}
        comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
        comp.validate_feed_dict(feed_dict_terminal, comp.terminal_mechanisms, "terminal")

    def test_input_state_len_3(self):
        comp = Composition()
        A = TransferMechanism(default_input_value=[0, 1, 2])
        B = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp.analyze_graph()
        feed_dict_origin = {A: [[0, 1, 2]]}
        feed_dict_terminal = {B: [[0]]}
        comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
        comp.validate_feed_dict(feed_dict_terminal, comp.terminal_mechanisms, "terminal")

    def test_input_state_len_3_feed_dict_len_2(self):
        comp = Composition()
        A = TransferMechanism(default_input_value=[0, 1, 2])
        B = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp.analyze_graph()
        feed_dict_origin = {A: [[0, 1]]}
        feed_dict_terminal = {B: [[0]]}
        with pytest.raises(ValueError):
            comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")

    def test_input_state_len_2_feed_dict_len_3(self):
        comp = Composition()
        A = TransferMechanism(default_input_value=[0, 1])
        B = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp.analyze_graph()
        feed_dict_origin = {A: [[0, 1, 2]]}
        feed_dict_terminal = {B: [[0]]}
        with pytest.raises(ValueError):
            comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")

    def test_feed_dict_includes_mechs_of_correct_and_incorrect_types(self):
        comp = Composition()
        A = TransferMechanism(default_input_value=[0])
        B = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp.analyze_graph()
        feed_dict_origin = {A: [[0]], B: [[0]]}
        with pytest.raises(ValueError):
            comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")

    def test_input_state_len_3_brackets_extra_1(self):
        comp = Composition()
        A = TransferMechanism(default_input_value=[0, 1, 2])
        B = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp.analyze_graph()
        feed_dict_origin = {A: [[[0, 1, 2]]]}
        feed_dict_terminal = {B: [[0]]}
        comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
        comp.validate_feed_dict(feed_dict_terminal, comp.terminal_mechanisms, "terminal")

    def test_input_state_len_3_brackets_missing_1(self):
        comp = Composition()
        A = TransferMechanism(default_input_value=[0, 1, 2])
        B = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp.analyze_graph()
        feed_dict_origin = {A:  [0, 1, 2]}
        feed_dict_terminal = {B: [[0]]}
        with pytest.raises(TypeError):
            comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")

    def test_empty_feed_dict_for_empty_type(self):
        comp = Composition()
        A = TransferMechanism(default_input_value=[0])
        B = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp.analyze_graph()
        feed_dict_origin = {A: [[0]]}
        feed_dict_monitored = {}
        comp.validate_feed_dict(feed_dict_monitored, comp.monitored_mechanisms, "monitored")

    def test_mech_in_feed_dict_for_empty_type(self):
        comp = Composition()
        A = TransferMechanism(default_input_value=[0])
        B = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp.analyze_graph()
        feed_dict_origin = {A: [[0]]}
        feed_dict_monitored = {B: [[0]]}
        with pytest.raises(ValueError):
            comp.validate_feed_dict(feed_dict_monitored, comp.monitored_mechanisms, "monitored")

    def test_one_mech_1(self):
        comp = Composition()
        A = TransferMechanism(default_input_value=[0])
        comp.add_mechanism(A)
        comp.analyze_graph()
        feed_dict_origin = {A: [[0]]}
        feed_dict_terminal = {A: [[0]]}
        comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")

    def test_one_mech_2(self):
        comp = Composition()
        A = TransferMechanism(default_input_value=[0])
        comp.add_mechanism(A)
        comp.analyze_graph()
        feed_dict_origin = {A: [[0]]}
        feed_dict_terminal = {A: [[0]]}
        comp.validate_feed_dict(feed_dict_terminal, comp.terminal_mechanisms, "terminal")

    def test_multiple_time_steps_1(self):
        comp = Composition()
        A = TransferMechanism(default_input_value=[[0, 1, 2]])
        B = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp.analyze_graph()
        feed_dict_origin = {A: [[0, 1, 2], [0, 1, 2]]}
        feed_dict_terminal = {B: [[0]]}
        comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
        comp.validate_feed_dict(feed_dict_terminal, comp.terminal_mechanisms, "terminal")

    def test_multiple_time_steps_2(self):
        comp = Composition()
        A = TransferMechanism(default_input_value=[[0, 1, 2]])
        B = mechanism()
        comp.add_mechanism(A)
        comp.add_mechanism(B)
        comp.add_projection(A, MappingProjection(), B)
        comp.analyze_graph()
        feed_dict_origin = {A: [[[0, 1, 2]], [[0, 1, 2]]]}
        feed_dict_terminal = {B: [[0]]}
        comp.validate_feed_dict(feed_dict_origin, comp.origin_mechanisms, "origin")
        comp.validate_feed_dict(feed_dict_terminal, comp.terminal_mechanisms, "terminal")
