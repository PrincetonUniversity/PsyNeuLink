import collections
import functools
import logging
from timeit import timeit

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.components.functions.nonstateful.combinationfunctions import LinearCombination
from psyneulink.core.components.functions.nonstateful.learningfunctions import Reinforcement, BackPropagation
from psyneulink.core.components.functions.nonstateful.optimizationfunctions import GridSearch
from psyneulink.core.components.functions.nonstateful.transferfunctions import \
    Linear, Logistic, INTENSITY_COST_FCT_MULTIPLICATIVE_PARAM
from psyneulink.core.components.functions.stateful.integratorfunctions import \
    AdaptiveIntegrator, DriftDiffusionIntegrator, IntegratorFunction, SimpleIntegrator
from psyneulink.core.components.functions.userdefinedfunction import UserDefinedFunction
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.modulatory.control.optimizationcontrolmechanism import \
    OptimizationControlMechanism
from psyneulink.core.components.mechanisms.modulatory.learning.learningmechanism import LearningMechanism
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.core.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.modulatorysignals.controlsignal import ControlSignal, CostFunctions
from psyneulink.core.components.projections.modulatory.controlprojection import ControlProjection
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.compositions.composition import Composition, NodeRole, CompositionError, RunError
from psyneulink.core.compositions.pathway import Pathway, PathwayRole
from psyneulink.core.globals.context import Context
from psyneulink.core.globals.keywords import \
    ADDITIVE, ALLOCATION_SAMPLES, BEFORE, DEFAULT, DISABLE, INPUT_PORT, INTERCEPT, LEARNING_MECHANISMS, \
    LEARNED_PROJECTIONS, RANDOM_CONNECTIVITY_MATRIX, CONTROL, \
    NAME, PROJECTIONS, RESULT, OBJECTIVE_MECHANISM, OUTPUT_MECHANISM, OVERRIDE, SLOPE, TARGET_MECHANISM, VARIANCE
from psyneulink.core.scheduling.condition import AtTimeStep, AtTrial, Never, TimeInterval
from psyneulink.core.scheduling.condition import EveryNCalls
from psyneulink.core.scheduling.scheduler import Scheduler, SchedulingMode
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.library.components.mechanisms.modulatory.control.agt.lccontrolmechanism import LCControlMechanism
from psyneulink.library.components.mechanisms.processing.transfer.recurrenttransfermechanism import \
    RecurrentTransferMechanism

logger = logging.getLogger(__name__)

# All tests are set to run. If you need to skip certain tests,
# see http://doc.pytest.org/en/latest/skipping.html


def record_values(d, time_scale, *mechs, comp=None):
    if time_scale not in d:
        d[time_scale] = {}
    for mech in mechs:
        if mech not in d[time_scale]:
            d[time_scale][mech] = []
        mech_value = mech.parameters.value.get(comp)
        if mech_value is None:
            d[time_scale][mech].append(np.nan)
        else:
            d[time_scale][mech].append(mech_value[0][0])

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
        t = timeit('comp = Composition()', setup='from psyneulink.core.compositions.composition import Composition', number=count)
        print()
        logger.info('completed {0} creation{2} of Composition() in {1:.8f}s'.format(count, t, 's' if count != 1 else ''))

    def test_call_after_construction_with_no_arg_then_run_then_illegal_args_error(self):
        A = ProcessingMechanism()
        B = ProcessingMechanism(function=Linear(slope=2))
        C = ProcessingMechanism(function=Logistic)
        c = Composition(pathways=[[A],[B],[C]])
        assert c() is None
        result = c(inputs={A:[[1],[100]],B:[[2],[200]],C:[[3],[1]]})
        assert np.allclose(result, [[100],[400],[0.73105858]])
        assert np.allclose(c(), [[100],[400],[0.73105858]])
        with pytest.raises(CompositionError) as err:
            c(23, 'bad_arg', bad_kwarg=1)
        assert f" called with illegal argument(s): 23, bad_arg, bad_kwarg" in str(err.value)

    def test_call_after_construction_with_learning_pathway(self):
        A = ProcessingMechanism()
        B = ProcessingMechanism(function=Linear(slope=0.5))
        C = ProcessingMechanism(function=Logistic)
        c = Composition(pathways=[[A],{'LEARNING_PATHWAY':([B,C], BackPropagation)}])
        assert c() is None

        # Run without learning
        result = c(inputs={A:[[1],[100]],B:[[2],[1]]})
        print(result)
        assert np.allclose(result, [[100.],[0.62245933]])
        assert np.allclose(c(), [[100.],[0.62245933]])

        # Run with learning
        target = c.pathways['LEARNING_PATHWAY'].target
        result = c(inputs={A:[[1],[100]],B:[[2],[1]],target:[[3],[300]]})
        np.allclose(result, [[[1.], [0.73105858]], [[100.], [0.62507661]]])


class TestAddMechanism:

    def test_add_once(self):
        comp = Composition()
        comp.add_node(TransferMechanism())
    def test_add_twice(self):
        comp = Composition()
        comp.add_node(TransferMechanism())
        comp.add_node(TransferMechanism())

    def test_add_same_twice(self):
        comp = Composition()
        mech = TransferMechanism()
        comp.add_node(mech)
        comp.add_node(mech)

    def test_add_multiple_projections_at_once(self):
        comp = Composition(name='comp')
        a = TransferMechanism(name='a')
        b = TransferMechanism(name='b',
                              function=Linear(slope=2.0))
        c = TransferMechanism(name='a',
                              function=Linear(slope=4.0))
        nodes = [a, b, c]
        comp.add_nodes(nodes)

        ab = MappingProjection(sender=a, receiver=b)
        bc = MappingProjection(sender=b, receiver=c, matrix=[[3.0]])
        projections = [ab, bc]
        comp.add_projections(projections)

        comp.run(inputs={a: 1.0})

        assert np.allclose(a.value, [[1.0]])
        assert np.allclose(b.value, [[2.0]])
        assert np.allclose(c.value, [[24.0]])
        assert ab in comp.projections
        assert bc in comp.projections

    def test_add_multiple_projections_no_sender(self):
        comp = Composition(name='comp')
        a = TransferMechanism(name='a')
        b = TransferMechanism(name='b',
                              function=Linear(slope=2.0))
        c = TransferMechanism(name='a',
                              function=Linear(slope=4.0))
        nodes = [a, b, c]
        comp.add_nodes(nodes)

        ab = MappingProjection(sender=a, receiver=b)
        bc = MappingProjection(sender=b)
        projections = [ab, bc]
        with pytest.raises(CompositionError) as err:
            comp.add_projections(projections)
        assert "The add_projections method of Composition requires a list of Projections" in str(err.value)

    def test_add_multiple_projections_no_receiver(self):
        comp = Composition(name='comp')
        a = TransferMechanism(name='a')
        b = TransferMechanism(name='b',
                              function=Linear(slope=2.0))
        c = TransferMechanism(name='a',
                              function=Linear(slope=4.0))
        nodes = [a, b, c]
        comp.add_nodes(nodes)

        ab = MappingProjection(sender=a, receiver=b)
        bc = MappingProjection(receiver=c)
        projections = [ab, bc]
        with pytest.raises(CompositionError) as err:
            comp.add_projections(projections)
        assert "The add_projections method of Composition requires a list of Projections" in str(err.value)

    def test_add_multiple_projections_not_a_proj(self):
        comp = Composition(name='comp')
        a = TransferMechanism(name='a')
        b = TransferMechanism(name='b',
                              function=Linear(slope=2.0))
        c = TransferMechanism(name='a',
                              function=Linear(slope=4.0))
        nodes = [a, b, c]
        comp.add_nodes(nodes)

        ab = MappingProjection(sender=a, receiver=b)
        bc = [[3.0]]
        projections = [ab, bc]
        with pytest.raises(CompositionError) as err:
            comp.add_projections(projections)
        assert "The add_projections method of Composition requires a list of Projections" in str(err.value)

    def test_add_multiple_nodes_at_once(self):
        comp = Composition()
        a = TransferMechanism()
        b = TransferMechanism()
        c = TransferMechanism()
        nodes = [a, b, c]
        comp.add_nodes(nodes)
        output = comp.run(inputs={a: [1.0],
                                  b: [2.0],
                                  c: [3.0]})
        assert set(comp.get_nodes_by_role(NodeRole.INPUT)) == set(nodes)
        assert set(comp.get_nodes_by_role(NodeRole.OUTPUT)) == set(nodes)
        assert np.allclose(output, [[1.0], [2.0], [3.0]])
    @pytest.mark.stress
    @pytest.mark.parametrize(
        'count', [
            100,
        ]
    )
    def test_timing_stress(self, count):
        t = timeit(
            'comp.add_node(TransferMechanism())',
            setup="""

from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.compositions.composition import Composition
comp = Composition()
""",
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
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(), A, B)

    def test_add_twice(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B')
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(), A, B)
        comp.add_projection(MappingProjection(), A, B)
    #
    # def test_add_same_twice(self):
    #     comp = Composition()
    #     A = TransferMechanism(name='composition-pytests-A')
    #     B = TransferMechanism(name='composition-pytests-B')
    #     comp.add_node(A)
    #     comp.add_node(B)
    #     proj = MappingProjection()
    #     comp.add_projection(proj, A, B)
    #     with pytest.raises(CompositionError) as error_text:
    #         comp.add_projection(proj, A, B)
    #     assert "This Projection is already in the Composition" in str(error_text.value)

    def test_add_fully_specified_projection_object(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B')
        comp.add_node(A)
        comp.add_node(B)
        proj = MappingProjection(sender=A, receiver=B)
        comp.add_projection(proj)

    def test_add_proj_sender_and_receiver_only(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B',
                              function=Linear(slope=2.0))
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(sender=A, receiver=B)
        result = comp.run(inputs={A: [1.0]})
        assert np.allclose(result, [[np.array([2.])]])

    def test_add_proj_missing_sender(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B',
                              function=Linear(slope=2.0))
        comp.add_node(A)
        comp.add_node(B)
        with pytest.raises(CompositionError) as error_text:
            comp.add_projection(receiver=B)
        assert "a sender must be specified" in str(error_text.value)

    def test_add_proj_missing_receiver(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B',
                              function=Linear(slope=2.0))
        comp.add_node(A)
        comp.add_node(B)
        with pytest.raises(CompositionError) as error_text:
            comp.add_projection(sender=A)
        assert "a receiver must be specified" in str(error_text.value)

    def test_add_proj_invalid_projection_spec(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B',
                              function=Linear(slope=2.0))
        comp.add_node(A)
        comp.add_node(B)
        with pytest.raises(CompositionError) as error_text:
            comp.add_projection("projection")
        assert "Invalid projection" in str(error_text.value)

    # KAM commented out this test 7/24/18 because it does not work. Should it work?
    # Or should the add_projection method of Composition only consider composition nodes as senders and receivers

    # def test_add_proj_states_as_sender_and_receiver(self):
    #     comp = Composition()
    #     A = TransferMechanism(name='composition-pytests-A',
    #                           default_variable=[[0.], [0.]])
    #     B = TransferMechanism(name='composition-pytests-B',
    #                           function=Linear(slope=2.0),
    #                           default_variable=[[0.], [0.]])
    #     comp.add_node(A)
    #     comp.add_node(B)
    #
    #     comp.add_projection(sender=A.output_ports[0], receiver=B.input_ports[0])
    #     comp.add_projection(sender=A.output_ports[1], receiver=B.input_ports[1])
    #
    #     print(comp.run(inputs={A: [[1.0], [2.0]]}))

    def test_add_proj_weights_only(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A',
                              default_variable=[[0., 0., 0.]])
        B = TransferMechanism(name='composition-pytests-B',
                              default_variable=[[0., 0.]],
                              function=Linear(slope=2.0))
        weights = [[1., 2.], [3., 4.], [5., 6.]]
        comp.add_node(A)
        comp.add_node(B)
        proj = comp.add_projection(weights, A, B)
        comp.run(inputs={A: [[1.1, 1.2, 1.3]]})
        assert np.allclose(A.parameters.value.get(comp), [[1.1, 1.2, 1.3]])
        assert np.allclose(B.get_input_values(comp), [[11.2,  14.8]])
        assert np.allclose(B.parameters.value.get(comp), [[22.4,  29.6]])
        assert np.allclose(proj.matrix.base, weights)

    test_args = [(None, ([1],[1],[1],[1]), 3.0),
        ('list', ([[0.60276338]],[[0.64589411]],[[0.96366276]]), 2.02947612),
        ('set', ([[0.60276338]],[[0.64589411]],[[0.96366276]]), 2.02947612)]
    @pytest.mark.parametrize('projs, expected_matrices, expected_result', test_args, ids=[x[0] for x in test_args])
    def test_add_multiple_projections_for_nested_compositions(self, projs, expected_matrices, expected_result):
        """Test both automatic creation as well as explicit specification of Projections from outer Composition to
        multiple Nodes of a nested Composition, and between Nodes of nested Compositions.
        """

        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        E = ProcessingMechanism(name='E')
        F = ProcessingMechanism(name='F')
        X = ProcessingMechanism(name='INPUT NODE')
        M = ProcessingMechanism(name='MIDDLE NODE')
        Y = ProcessingMechanism(name='OUTPUT NODE')
        if projs == 'list':
            iprojs = [MappingProjection(sender=C, receiver=D, matrix=RANDOM_CONNECTIVITY_MATRIX)]
            oprojs = [MappingProjection(sender=X, receiver=A, matrix=RANDOM_CONNECTIVITY_MATRIX),
                     MappingProjection(sender=X, receiver=M, matrix=RANDOM_CONNECTIVITY_MATRIX)]
        elif projs == 'set':
            iprojs = {MappingProjection(sender=C, receiver=D, matrix=RANDOM_CONNECTIVITY_MATRIX)}
            oprojs = {MappingProjection(sender=X, receiver=A, matrix=RANDOM_CONNECTIVITY_MATRIX),
                     MappingProjection(sender=X, receiver=M, matrix=RANDOM_CONNECTIVITY_MATRIX)}

        comp1 = Composition(pathways=[A,B,C], name='COMP 1')
        comp2 = Composition(pathways=[D,E,F], name='COMP 2')

        if not projs:
            ipway = [comp1, comp2]
        else:
            ipway = [comp1, iprojs, comp2]
        mcomp = Composition(pathways=[ipway,M], name='MIDDLE COMPOSITION')

        if not projs:
            opway = [[X, mcomp, Y, C]]
        else:
            opway = [[X, oprojs, mcomp, Y, C]]
        ocomp = Composition(pathways=opway, name='OUTER COMPOSITION')

        assert np.allclose(ocomp.run(inputs=[[1.5]]), expected_result)

        if not projs:
            assert (comp1.output_CIM.output_ports[0].efferents[0].matrix.base ==
                    comp2.input_CIM.input_ports[0].path_afferents[0].matrix.base == expected_matrices[0])
            assert (X.output_ports[0].efferents[0].matrix.base ==
                    mcomp.input_CIM.input_ports[0].path_afferents[0].matrix.base == expected_matrices[1])
            assert (X.output_ports[0].efferents[1].matrix.base ==
                    mcomp.input_CIM.input_ports[1].path_afferents[0].matrix.base == expected_matrices[2])
        else:
            assert np.allclose(comp1.output_CIM.output_ports[0].efferents[0].matrix.base, expected_matrices[0])
            assert np.allclose(comp2.input_CIM.input_ports[0].path_afferents[0].matrix.base, expected_matrices[0])
            assert np.allclose(X.output_ports[0].efferents[0].matrix.base, expected_matrices[1])
            assert np.allclose(mcomp.input_CIM.input_ports[0].path_afferents[0].matrix.base, expected_matrices[1])
            assert np.allclose(X.output_ports[0].efferents[1].matrix.base, expected_matrices[2])
            assert np.allclose(mcomp.input_CIM.input_ports[1].path_afferents[0].matrix.base, expected_matrices[2])

    def test_add_linear_processing_pathway_with_noderole_specified_in_tuple(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B')
        C = TransferMechanism(name='composition-pytests-C')
        comp.add_linear_processing_pathway([
            (A,pnl.NodeRole.LEARNING),
            (B,pnl.NodeRole.LEARNING),
            C
        ])
        comp._analyze_graph()
        autoassociative_learning_nodes = comp.get_nodes_by_role(pnl.NodeRole.LEARNING)
        assert A in autoassociative_learning_nodes
        assert B in autoassociative_learning_nodes

    def test_add_linear_processing_pathway_containing_nodes_with_existing_projections(self):
        """ Test that add_linear_processing_pathway uses MappingProjections already specified for
                Hidden_layer_2 and Output_Layer in the pathway it creates within the Composition"""
        Input_Layer = TransferMechanism(name='Input Layer', size=2)
        Hidden_Layer_1 = TransferMechanism(name='Hidden Layer_1', size=5)
        Hidden_Layer_2 = TransferMechanism(name='Hidden Layer_2', size=4)
        Output_Layer = TransferMechanism(name='Output Layer', size=3)
        Input_Weights_matrix = (np.arange(2 * 5).reshape((2, 5)) + 1) / (2 * 5)
        Middle_Weights_matrix = (np.arange(5 * 4).reshape((5, 4)) + 1) / (5 * 4)
        Output_Weights_matrix = (np.arange(4 * 3).reshape((4, 3)) + 1) / (4 * 3)
        Input_Weights = MappingProjection(name='Input Weights', matrix=Input_Weights_matrix)
        Middle_Weights = MappingProjection(name='Middle Weights',sender=Hidden_Layer_1, receiver=Hidden_Layer_2,
                                           matrix=Middle_Weights_matrix),
        Output_Weights = MappingProjection(name='Output Weights',sender=Hidden_Layer_2,receiver=Output_Layer,
                                           matrix=Output_Weights_matrix)
        pathway = [Input_Layer, Input_Weights, Hidden_Layer_1, Hidden_Layer_2, Output_Layer]
        comp = Composition()
        comp.add_linear_processing_pathway(pathway=pathway)
        stim_list = {Input_Layer: [[-1, 30]]}
        results = comp.run(num_trials=2, inputs=stim_list)

    def test_add_backpropagation_learning_pathway_containing_nodes_with_existing_projections(self):
        """ Test that add_backpropagation_learning_pathway uses MappingProjections already specified for
                Hidden_layer_2 and Output_Layer in the pathway it creates within the Composition"""
        Input_Layer = TransferMechanism(name='Input Layer', size=2)
        Hidden_Layer_1 = TransferMechanism(name='Hidden Layer_1', size=5)
        Hidden_Layer_2 = TransferMechanism(name='Hidden Layer_2', size=4)
        Output_Layer = TransferMechanism(name='Output Layer', size=3)
        Input_Weights_matrix = (np.arange(2 * 5).reshape((2, 5)) + 1) / (2 * 5)
        Middle_Weights_matrix = (np.arange(5 * 4).reshape((5, 4)) + 1) / (5 * 4)
        Output_Weights_matrix = (np.arange(4 * 3).reshape((4, 3)) + 1) / (4 * 3)
        Input_Weights = MappingProjection(name='Input Weights', matrix=Input_Weights_matrix)
        Middle_Weights = MappingProjection(name='Middle Weights',sender=Hidden_Layer_1, receiver=Hidden_Layer_2,
                                           matrix=Middle_Weights_matrix),
        Output_Weights = MappingProjection(name='Output Weights',sender=Hidden_Layer_2,receiver=Output_Layer,
                                           matrix=Output_Weights_matrix)
        pathway = [Input_Layer, Input_Weights, Hidden_Layer_1, Hidden_Layer_2, Output_Layer]
        comp = Composition()
        backprop_pathway = comp.add_backpropagation_learning_pathway(pathway=pathway)
        stim_list = {
            Input_Layer: [[-1, 30]],
            backprop_pathway.target: [[0, 0, 1]]}
        results = comp.run(num_trials=2, inputs=stim_list)

    def test_linear_processing_pathway_weights_only(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A',
                              default_variable=[[0., 0., 0.]])
        B = TransferMechanism(name='composition-pytests-B',
                              default_variable=[[0., 0.]],
                              function=Linear(slope=2.0))
        weights = [[1., 2.], [3., 4.], [5., 6.]]
        comp.add_linear_processing_pathway([A, weights, B])
        comp.run(inputs={A: [[1.1, 1.2, 1.3]]})
        assert np.allclose(A.parameters.value.get(comp), [[1.1, 1.2, 1.3]])
        assert np.allclose(B.get_input_values(comp), [[11.2,  14.8]])
        assert np.allclose(B.parameters.value.get(comp), [[22.4,  29.6]])

    def test_add_conflicting_projection_object(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B')
        C = TransferMechanism(name='composition-pytests-C')
        comp.add_node(A)
        comp.add_node(B)
        comp.add_node(C)
        proj = MappingProjection(sender=A, receiver=B)
        with pytest.raises(CompositionError) as error:
            comp.add_projection(projection=proj, receiver=C)
            assert '"Receiver (\'composition-pytests-C\') assigned to ' \
                   '\'MappingProjection from composition-pytests-A[RESULT] to composition-pytests-B[InputPort-0] ' \
                   'is incompatible with the positions of these Components in \'Composition-0\'."' == str(error.value)

    @pytest.mark.stress
    @pytest.mark.parametrize(
        'count', [
            1000,
        ]
    )
    def test_timing_stress(self, count):
        t = timeit('comp.add_projection(A, MappingProjection(), B)',
                   setup="""

from psyneulink.core.components.mechanisms.processingmechanisms.transfermechanism import TransferMechanism
from psyneulink.core.components.projections.pathwayprojections.mappingprojection import MappingProjection
from psyneulink.core.compositions.composition import Composition

comp = Composition()
A = TransferMechanism(name='composition-pytests-A')
B = TransferMechanism(name='composition-pytests-B')
comp.add_node(A)
comp.add_node(B)
""",
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
                   setup="""
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.compositions.composition import Composition
comp = Composition()
A = TransferMechanism(name='composition-pytests-A')
B = TransferMechanism(name='composition-pytests-B')
comp.add_node(A)
comp.add_node(B)
""",
                   number=count
                   )
        print()
        logger.info('completed {0} addition{2} of a projection to a composition in {1:.8f}s'.format(count, t, 's' if count != 1 else ''))

    def test_unused_projections_warning(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C', input_ports=[A])
        D = ProcessingMechanism(name='D')
        icomp = Composition(name='iCOMP', pathways=[A, B])
        comp1 = Composition(name='COMP_1', pathways=[icomp])
        comp2 = Composition(name='COMP_2', pathways=[C, D])
        with pytest.warns(UserWarning) as warning:
            ocomp = Composition(name='OUTER COMPOSITION',pathways=[comp1, comp2])
            ocomp.run()
        assert repr(warning[0].message.args[0]) == '"\\nThe following Projections were specified but are not being used by Nodes in \'iCOMP\':\\n\\tMappingProjection from A[OutputPort-0] to C[InputPort-0] (from \'A\' to \'C\')."'
        assert repr(warning[1].message.args[0]) == '"\\nThe following Projections were specified but are not being used by Nodes in \'COMP_2\':\\n\\tMappingProjection from A[OutputPort-0] to C[InputPort-0] (to \'C\' from \'A\')."'


@pytest.mark.pathways
class TestPathway:

    def test_pathway_standalone_object(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        p = Pathway(pathway=[A,B,C], name='P')
        assert p.pathway == [A, B, C]
        assert p.composition is None
        assert p.name == 'P'
        assert p.input is None
        assert p.output is None
        assert p.target is None
        assert p.roles is None
        assert p.learning_components is None

    def test_pathway_assign_composition_arg_error(self):
        c = Composition()
        with pytest.raises(pnl.CompositionError) as error_text:
            p = Pathway(pathway=[], composition='c')
        assert "\'composition\' can not be specified as an arg in the constructor for a Pathway" in str(
                error_text.value)

    def test_pathway_assign_roles_error(self):
        A = ProcessingMechanism()
        c = Composition()
        p = Pathway(pathway=[A])
        with pytest.raises(AssertionError) as error_text:
            p._assign_roles(composition=c)
        assert (f"_assign_roles() cannot be called " in str(error_text.value) and
                f"because it has not been assigned to a Composition" in str(error_text.value))
        c.add_linear_processing_pathway(pathway=p)
        p_c = c.pathways[0]
        assert p_c._assign_roles(composition=c) is None

    def test_pathway_illegal_arg_error(self):
        with pytest.raises(pnl.CompositionError) as error_text:
            Pathway(pathway=[], foo='bar')
        assert "Illegal argument(s) used in constructor for Pathway: foo." in str(error_text.value)


@pytest.mark.pathways
class TestCompositionPathwayArgsAndAdditionMethods:

    def test_add_pathways_with_all_types(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        E = ProcessingMechanism(name='E')
        X = ProcessingMechanism(name='X')
        Y = ProcessingMechanism(name='Y')
        F = ProcessingMechanism(name='F')
        G = ProcessingMechanism(name='G')
        H = ProcessingMechanism(name='H')
        J = ProcessingMechanism(name='J')
        K = ProcessingMechanism(name='K')
        L = ProcessingMechanism(name='L')
        M = ProcessingMechanism(name='M')

        # FIX: 4/9/22 - ADD SET SPEC
        p = Pathway(pathway=[L,M], name='P')
        c = Composition()
        c.add_pathways(pathways=[A,
                                 [B,C],
                                 (D,E),
                                 {X,Y},
                                 {'DICT PATHWAY': F},
                                 ([G, H], BackPropagation),
                                 {'LEARNING PATHWAY': ([J,K], Reinforcement)},
                                 p])
        assert len(c.pathways) == 8
        assert isinstance(c.pathways[0].pathway, list) and len(c.pathways[0].pathway) == 1
        assert isinstance(c.pathways[1].pathway, list) and len(c.pathways[1].pathway) == 3
        assert isinstance(c.pathways[2].pathway, list) and len(c.pathways[2].pathway) == 3
        assert isinstance(c.pathways[3].pathway[0], set) and len(c.pathways[3].pathway) == 1
        assert c.pathways['P'].input == L
        assert c.pathways['DICT PATHWAY'].input == F
        assert c.pathways['DICT PATHWAY'].output == F
        assert c.pathways['LEARNING PATHWAY'].output == K
        assert [p for p in c.pathways if p.input == G][0].learning_function == BackPropagation
        assert c.pathways['LEARNING PATHWAY'].learning_function == Reinforcement

    def test_pathway_attributes(self):
        c = Composition()
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        E = ProcessingMechanism(name='E')
        F = ProcessingMechanism(name='F')
        G = ProcessingMechanism(name='G')
        p1 = c.add_linear_processing_pathway(pathway=[A,B,C], name='P')
        p2 = c.add_linear_processing_pathway(pathway=[D,B])
        p3 = c.add_linear_processing_pathway(pathway=[B,E])
        l = c.add_linear_learning_pathway(pathway=[F,G], learning_function=Reinforcement, name='L')
        assert p1.name == 'P'
        assert p1.input == A
        assert p1.output == C
        assert p1.target is None
        assert p2.input == D
        assert p2.output is None
        assert p2.target is None
        assert p3.input is None
        assert p3.output == E
        assert p3.target is None
        assert l.name == 'L'
        assert l.input == F
        assert l.output == G
        assert l.target == c.nodes['Target']
        assert l.learning_components[pnl.LEARNING_MECHANISMS] == \
               c.nodes['Learning Mechanism for MappingProjection from F[OutputPort-0] to G[InputPort-0]']
        assert l.learning_objective == c.nodes['Comparator']
        assert all(p in {p1, p2, p3, l} for p in c.pathways)

    def test_pathway_order_processing_then_learning_RL(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        c = Composition()
        c.add_linear_processing_pathway(pathway=[A,B])
        c.add_linear_learning_pathway(pathway=[C,D], learning_function=Reinforcement)
        assert all(n in {B, D} for n in c.get_nodes_by_role(NodeRole.OUTPUT))

    def test_pathway_order_processing_then_learning_BP(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        c = Composition()
        c.add_linear_processing_pathway(pathway=[A,B])
        c.add_linear_learning_pathway(pathway=[C,D], learning_function=BackPropagation)
        assert all(n in {B, D} for n in c.get_nodes_by_role(NodeRole.OUTPUT))

    def test_pathway_order_learning_RL_then_processing(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        c = Composition()
        c.add_linear_learning_pathway(pathway=[A,B], learning_function=Reinforcement)
        c.add_linear_processing_pathway(pathway=[C,D])
        assert all(n in {B, D} for n in c.get_nodes_by_role(NodeRole.OUTPUT))

    def test_pathway_order_learning_BP_then_processing(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        c = Composition()
        c.add_linear_learning_pathway(pathway=[A,B], learning_function=BackPropagation)
        c.add_linear_processing_pathway(pathway=[C,D])
        assert all(n in {B, D} for n in c.get_nodes_by_role(NodeRole.OUTPUT))

    def test_pathway_order_learning_RL_then_BP(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        c = Composition()
        c.add_linear_learning_pathway(pathway=[A,B], learning_function=Reinforcement)
        c.add_linear_learning_pathway(pathway=[C,D], learning_function=BackPropagation)
        assert all(n in {B, D} for n in c.get_nodes_by_role(NodeRole.OUTPUT))

    def test_pathway_order_learning_BP_then_RL(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        c = Composition()
        c.add_linear_learning_pathway(pathway=[A,B], learning_function=BackPropagation)
        c.add_linear_learning_pathway(pathway=[C,D], learning_function=Reinforcement)
        assert all(n in {B, D} for n in c.get_nodes_by_role(NodeRole.OUTPUT))

    def test_add_processing_pathway_arg_mech(self):
        A = ProcessingMechanism(name='A')
        c = Composition()
        c.add_linear_processing_pathway(pathway=A)
        assert set(c.get_roles_by_node(A)) == {NodeRole.INPUT,
                                               NodeRole.ORIGIN,
                                               NodeRole.SINGLETON,
                                               NodeRole.OUTPUT,
                                               NodeRole.TERMINAL}
        assert set(c.pathways[0].roles) == {PathwayRole.INPUT,
                                            PathwayRole.ORIGIN,
                                            PathwayRole.SINGLETON,
                                            PathwayRole.OUTPUT,
                                            PathwayRole.TERMINAL}

    def test_add_processing_pathway_arg_pathway(self):
        A = ProcessingMechanism(name='A')
        p = Pathway(pathway=A, name='P')
        c = Composition()
        c.add_linear_processing_pathway(pathway=p)
        assert set(c.get_roles_by_node(A)) == {NodeRole.INPUT,
                                               NodeRole.ORIGIN,
                                               NodeRole.SINGLETON,
                                               NodeRole.OUTPUT,
                                               NodeRole.TERMINAL}
        assert set(c.pathways['P'].roles) == {PathwayRole.INPUT,
                                              PathwayRole.ORIGIN,
                                              PathwayRole.SINGLETON,
                                              PathwayRole.OUTPUT,
                                              PathwayRole.TERMINAL}

    def test_add_processing_pathway_with_errant_learning_function_warning(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        p = Pathway(pathway=([A,B], Reinforcement), name='P')
        c = Composition()

        regexp = "LearningFunction found in specification of 'pathway' arg for "\
                 "add_linear_procesing_pathway method .*"\
                r"Reinforcement'>; it will be ignored"
        with pytest.warns(UserWarning, match=regexp):
            c.add_linear_processing_pathway(pathway=p)

        assert set(c.get_roles_by_node(A)) == {NodeRole.INPUT, NodeRole.ORIGIN}
        assert set(c.get_roles_by_node(B)) == {NodeRole.OUTPUT, NodeRole.TERMINAL}
        assert set(c.pathways['P'].roles) == {PathwayRole.INPUT,
                                              PathwayRole.ORIGIN,
                                              PathwayRole.OUTPUT,
                                              PathwayRole.TERMINAL}

    def test_add_learning_pathway_arg_pathway(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        p = Pathway(pathway=[A,B], name='P')
        c = Composition()
        c.add_linear_learning_pathway(pathway=p, learning_function=BackPropagation)
        assert set(c.get_roles_by_node(A)) == {NodeRole.INPUT, NodeRole.ORIGIN}
        assert {NodeRole.OUTPUT}.issubset(c.get_roles_by_node(B))
        assert set(c.pathways['P'].roles) == {PathwayRole.INPUT,
                                              PathwayRole.ORIGIN,
                                              PathwayRole.LEARNING,
                                              PathwayRole.OUTPUT}

    def test_add_learning_pathway_with_errant_learning_function_in_tuple_spec_error(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        p = Pathway(pathway=([A,B], Reinforcement), name='P')
        c = Composition()
        with pytest.raises(pnl.CompositionError) as error_text:
            c.add_linear_learning_pathway(pathway=p, learning_function=BackPropagation)
        assert ("Specification in 'pathway' arg for " in str(error_text.value) and
                "add_linear_procesing_pathway method" in str(error_text.value) and
                "contains a tuple that specifies a different LearningFunction (Reinforcement)" in str(error_text.value)
                and "than the one specified in its 'learning_function' arg (BackPropagation)" in str(error_text.value))

    def test_add_bp_learning_pathway_arg_pathway(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        p = Pathway(pathway=[A,B], name='P')
        c = Composition()
        c.add_backpropagation_learning_pathway(pathway=p)
        assert {NodeRole.INPUT, NodeRole.ORIGIN}.issubset(c.get_roles_by_node(A))
        assert {NodeRole.OUTPUT}.issubset(c.get_roles_by_node(B))
        assert set(c.pathways['P'].roles) == {PathwayRole.INPUT,
                                              PathwayRole.ORIGIN,
                                              PathwayRole.LEARNING,
                                              PathwayRole.OUTPUT}

    def test_add_bp_learning_pathway_arg_pathway_name_in_method(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        p = Pathway(pathway=[A,B], name='P')
        c = Composition()
        c.add_backpropagation_learning_pathway(pathway=p, name='BP')
        assert {NodeRole.INPUT, NodeRole.ORIGIN}.issubset(set(c.get_roles_by_node(A)))
        assert {NodeRole.OUTPUT}.issubset(set(c.get_roles_by_node(B)))
        assert set(c.pathways['BP'].roles) == {PathwayRole.INPUT,
                                               PathwayRole.ORIGIN,
                                               PathwayRole.LEARNING,
                                               PathwayRole.OUTPUT}

    def test_add_rl_learning_pathway_arg_pathway(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        p = Pathway(pathway=[A,B], name='P')
        c = Composition()
        c.add_reinforcement_learning_pathway(pathway=p)
        assert {NodeRole.INPUT, NodeRole.ORIGIN}.issubset(set(c.get_roles_by_node(A)))
        assert {NodeRole.OUTPUT}.issubset(set(c.get_roles_by_node(B)))
        assert set(c.pathways['P'].roles) == {PathwayRole.INPUT,
                                              PathwayRole.ORIGIN,
                                              PathwayRole.LEARNING,
                                              PathwayRole.OUTPUT}

    def test_add_td_learning_pathway_arg_pathway(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        p = Pathway(pathway=[A,B], name='P')
        c = Composition()
        c.add_td_learning_pathway(pathway=p)
        assert {NodeRole.INPUT, NodeRole.ORIGIN}.issubset(set(c.get_roles_by_node(A)))
        assert {NodeRole.OUTPUT}.issubset(set(c.get_roles_by_node(B)))
        assert set(c.pathways['P'].roles) == {PathwayRole.INPUT,
                                              PathwayRole.ORIGIN,
                                              PathwayRole.LEARNING,
                                              PathwayRole.OUTPUT}

    config = [
        ('[A,{B,C}]', 's1'),                      # SEQUENTIAL A->{B,C})
        ('[A,[B,C]]', 'p1'),                      # PARALLEL:  A, B->C
        ('[{A},{B,C}]', 's1'),                    # SEQUENTIAL: A->{B,C}
        ('[[A],{B,C}]', 'p2'),                    # PARALLEL: A, B, C
        ('[[A,B],{C,D}]', 'p3'),                  # PARALLEL: A->B, C, D
        ('[[A,B],C,D ]', 'p3'),                   # PARALLEL: A->B, C, D
        ('[[A,B],[C,D]]', 'p5'),                  # PARALLEL: A->B, C->D
        ('[{A,B}, MapProj(B,D), C, D]', 's2'),    # SEQUENTIAL: A, B->D, C->D
        ('[{A,B}, [MapProj(B,D)], C, D]', 's2'),  # SEQUENTIAL: A, B->D, C->D
        ('[{A,B}, {MapProj(B,D)}, C, D]', 's2'),  # SEQUENTIAL: A, B->D, C->D
        ('[{A,B}, [[C,D]]]', 'p4'),               # PARALLEL: A, B, C->D (FORGIVES EMBEDDED LIST OF [C,D])
        ('[[A,B], [[C,D]]]', 'p5'),               # PARALLEL: A->B, C->D (FORGIVES EMBEDDED LIST OF [C,D])
        ('[[[A,B]], [[C,D]]]','p5'),              # PARALLEL: A->B, C->D (FORGIVES EMBEDDED LISTS OF [A, B] and [C,D])
        ('[A, "B"]','e1'),                        # BAD ITEM ERROR
        ('[[A,B, [C,D]],[E,F]]','e2'),            # EMBEDDED LIST ERROR
        ('[{A,B}, [MapProj(B,D)], [C,D]]', 'e3')  # BAD ITEM ERROR, FIX: SHOULD ALLOW EMBEDDED PER ABOVE
    ]
    @pytest.mark.parametrize('config', config, ids=[x[0] for x in config])
    def test_various_pathway_configurations_in_constructor(self, config):
        """Test combinations of sets and lists in pathways specification of Composition constructor
        Principles:
          if outer list (pathways spec) contains:
          - single item or only sets, treat single (sequential) pathway
          - one or more lists within it, treat all items as a separate (parallel) pathways
          - one or more lists each with a single list within it ([[[A,B]],[[C,D]]]}), each is treated as a pathway
          - any list with more than a single list within it ([[[A,B],[C,D]}), an error is generated
          - any bad items (strings, misplaced items), an error is generated
        """

        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        # B_comparator = ComparatorMechanism(name='B COMPARATOR')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        E = ProcessingMechanism(name='E')
        F = ProcessingMechanism(name='F')

        # LEGAL:
        if config[0] == '[A,{B,C}]':                 # SEQUENTIAL A->{B,C}) (s1)
            comp = Composition([A,{B,C}])
        elif config[0] == '[A,[B,C]]':               # PARALLEL:  A, B->C (p1)
            comp = Composition([A,[B,C]])
        elif config[0] == '[{A},{B,C}]':             # SEQUENTIAL: A->{B,C} (s1)
            comp = Composition([{A},{B,C}])
        elif config[0] == '[[A],{B,C}]':             # PARALLEL: A, B, C (p2)
            comp = Composition([[A],{B,C}])
        elif config[0] == '[[A,B],{C,D}]':           # PARALLEL: A->B, C, D (p3)
            comp = Composition([[A,B],{C,D}])
        elif config[0] == '[[A,B],C,D ]':            # PARALLEL: A->B, C, D (p3)
            comp = Composition([[A,B],C,D ])
        elif config[0] == '[[A,B],[C,D]]':           # PARALLEL: A->B, C->D {p5)
            comp = Composition([[A,B],[C,D]])
        elif config[0] == '[{A,B}, MapProj(B,D), C, D]':                   # SEQUENTIAL: A, B->D, C->D (s2)
            comp = Composition([{A,B}, MappingProjection(B,D), C, D])
        elif config[0] == '[{A,B}, [MapProj(B,D)], C, D]':                 # SEQUENTIAL: A, B->D, C->D (s2)
            comp = Composition([{A,B}, [MappingProjection(B,D)], C, D])
        elif config[0] == '[{A,B}, {MapProj(B,D)}, C, D]':                 # SEQUENTIAL: A, B->D, C->D (s2)
            comp = Composition([{A,B}, {MappingProjection(B,D)}, C, D])
        elif config[0] == '[{A,B}, [[C,D]]]':        # PARALLEL: A, B, C->D (FORGIVES EMBEDDED LIST [C,D]) (p4)
            comp = Composition([{A,B}, [[C,D]]])
        elif config[0] == '[[A,B], [[C,D]]]':        # PARALLEL: A->B, C->D (SINGLE EMBEDDED LIST OK [C,D])  (p5)
            comp = Composition([[A,B], [[C,D]]])
        elif config[0] == '[[[A,B]], [[C,D]]]':      # PARALLEL: A->B, C->D (FORGIVES EMBEDDED LISTS [A,B] & [C,D]) (p5)
            comp = Composition([[[A,B]], [[C,D]]])

        # ERRORS:
        elif config[0] == '[A, "B"]':                                      # BAD ITEM ERROR (e1)
            with pytest.raises(CompositionError) as error_text:
                comp = Composition([A, "B"])
            assert f"Every item in the 'pathways' arg of the constructor for Composition-0 must be " \
                   f"a Node, list, set, tuple or dict; the following are not: 'B'" in str(error_text.value)
        elif config[0] == '[[A,B, [C,D]],[E,F]]':                          # EMBEDDED LIST ERROR (e2)
            with pytest.raises(CompositionError) as error_text:
                comp = Composition([[A,B, [C,D]],[E,F]])
            assert f"The following entries in a pathway specified for \'Composition-0\' are not " \
                   f"a Node (Mechanism or Composition) or a Projection nor a set of either: " \
                   f"[(ProcessingMechanism C), (ProcessingMechanism D)]" in str(error_text.value)
        elif config[0] == '[{A,B}, [MapProj(B,D)], [C,D]]':                # BAD ITEM ERROR (e3)
            with pytest.raises(CompositionError) as error_text:
                comp = Composition([{A,B}, [MappingProjection(B,D)], [C,D]])
            assert f"Every item in the 'pathways' arg of the constructor for Composition-0 must be " \
                   f"a Node, list, set, tuple or dict; the following are not: " \
                   f"(MappingProjection MappingProjection from B[OutputPort-0] to D[InputPort-0])" \
                   in str(error_text.value)

        else:
            assert False, f"BAD CONFIG ARG: {config}"

        # Tests:
        if config[1] == 's1':
            assert len(A.efferents) == 2
            assert all(len(receiver.path_afferents) == 1 for receiver in {B,C})
            assert all(receiver in [p.receiver.owner for p in A.efferents] for receiver in {B,C})
            assert [A] == comp.get_nodes_by_role(NodeRole.INPUT)
            assert all(node in comp.get_nodes_by_role(NodeRole.OUTPUT) for node in {B,C})
        if config[1] == 's2':
            assert all(len(sender.efferents) == 1 for sender in {B,C})
            assert len(D.path_afferents) == 2
            assert all(D in [p.receiver.owner for p in receiver.efferents] for receiver in {B,C})
            assert [A] == comp.get_nodes_by_role(NodeRole.SINGLETON)
            assert all(node in comp.get_nodes_by_role(NodeRole.INPUT) for node in {B,C})
            assert all(node in comp.get_nodes_by_role(NodeRole.OUTPUT) for node in {A,D})
        if config[1] == 'p1':
            assert len(B.efferents) == 1
            assert len(C.path_afferents) == 1
            assert B.efferents[0].receiver.owner == C
            assert [A] == comp.get_nodes_by_role(NodeRole.SINGLETON)
            assert all(node in comp.get_nodes_by_role(NodeRole.INPUT) for node in {A,B})
            assert all(node in comp.get_nodes_by_role(NodeRole.OUTPUT) for node in {A,C})
        if config[1] == 'p2':
            assert all(node in comp.get_nodes_by_role(NodeRole.SINGLETON) for node in {A,B,C})
        if config[1] == 'p3':
            assert len(A.efferents) == 1
            assert len(B.path_afferents) == 1
            assert A.efferents[0].receiver.owner == B
            assert all(node in comp.get_nodes_by_role(NodeRole.SINGLETON) for node in {C,D})
        if config[1] == 'p4':
            assert len(C.efferents) == 1
            assert len(D.path_afferents) == 1
            assert C.efferents[0].receiver.owner == D
            assert all(node in comp.get_nodes_by_role(NodeRole.SINGLETON) for node in {A,B})
            assert all(node in comp.get_nodes_by_role(NodeRole.INPUT) for node in {A,B,C})
            assert all(node in comp.get_nodes_by_role(NodeRole.OUTPUT) for node in {A,D})
        if config[1] == 'p5':
            assert len(A.efferents) == 1
            assert len(B.path_afferents) == 1
            assert A.efferents[0].receiver.owner == B
            assert len(C.efferents) == 1
            assert len(D.path_afferents) == 1
            assert C.efferents[0].receiver.owner == D
            assert all(node in comp.get_nodes_by_role(NodeRole.INPUT) for node in {A,C})
            assert all(node in comp.get_nodes_by_role(NodeRole.OUTPUT) for node in {B,D})

    def test_add_pathways_bad_arg_error(self):
        I = InputPort(name='I')
        c = Composition()
        with pytest.raises(pnl.CompositionError) as error_text:
            c.add_pathways(pathways=I)
        assert f"The 'pathways' arg for the add_pathways method of Composition-0 must be a " \
               f"Node, list, set, tuple, dict or Pathway object: (InputPort I [Deferred Init])." \
               in str(error_text.value)

    def test_add_pathways_arg_pathways_list_and_item_not_list_or_dict_or_node_error(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        c = Composition()
        with pytest.raises(pnl.CompositionError) as error_text:
            c.add_pathways(pathways=[[A,B], 'C'])
        assert f"Every item in the 'pathways' arg for the add_pathways method of Composition-0 must be a " \
              f"Node, list, set, tuple or dict; the following are not: 'C'" in str(error_text.value)

    def test_for_add_processing_pathway_recursion_error(self):
        A = TransferMechanism()
        C = Composition()
        with pytest.raises(pnl.CompositionError) as error_text:
            C.add_linear_processing_pathway(pathway=[A,C])
        assert f"Attempt to add Composition as a Node to itself in 'pathway' arg for " \
               f"add_linear_procesing_pathway method of '{C.name}'." in str(error_text.value)

    def test_for_add_learning_pathway_recursion_error(self):
        A = TransferMechanism()
        C = Composition()
        with pytest.raises(pnl.CompositionError) as error_text:
            C.add_backpropagation_learning_pathway(pathway=[A,C])
        assert f"Attempt to add Composition as a Node to itself in 'pathway' arg for " \
               f"add_backpropagation_learning_pathway method of {C.name}." in str(error_text.value)


@pytest.mark.pathways
class TestDuplicatePathwayWarnings:

    def test_add_processing_pathway_exact_duplicate_warning(self):
        A = TransferMechanism()
        B = TransferMechanism()
        P = MappingProjection(sender=A, receiver=B)
        comp = Composition()
        comp.add_linear_processing_pathway(pathway=[A,P,B])

        regexp = "Pathway specified in 'pathway' arg for add_linear_procesing_pathway method .*"\
                f"already exists in {comp.name}"
        with pytest.warns(UserWarning, match=regexp):
            comp.add_linear_processing_pathway(pathway=[A,P,B])

    def test_add_processing_pathway_inferred_duplicate_warning(self):
        A = TransferMechanism()
        B = TransferMechanism()
        C = TransferMechanism()
        comp = Composition()
        comp.add_linear_processing_pathway(pathway=[A,B,C])

        regexp = "Pathway specified in 'pathway' arg for add_linear_procesing_pathway method .*"\
                f"has same Nodes in same order as one already in {comp.name}"
        with pytest.warns(UserWarning, match=regexp):
            comp.add_linear_processing_pathway(pathway=[A,B,C])

    def test_add_processing_pathway_subset_duplicate_warning(self):
        A = TransferMechanism()
        B = TransferMechanism()
        C = TransferMechanism()
        comp = Composition()
        comp.add_linear_processing_pathway(pathway=[A,B,C])

        regexp = "Pathway specified in 'pathway' arg for add_linear_procesing_pathway method .*"\
                 f"has same Nodes in same order as one already in {comp.name}"
        with pytest.warns(UserWarning, match=regexp):
            comp.add_linear_processing_pathway(pathway=[A,B])
            assert True

    def test_add_backpropagation_pathway_exact_duplicate_warning(self):
        A = TransferMechanism()
        B = TransferMechanism()
        P = MappingProjection(sender=A, receiver=B)
        comp = Composition()
        comp.add_backpropagation_learning_pathway(pathway=[A,P,B])

        regexp = "Pathway specified in 'pathway' arg for add_backpropagation_learning_pathway method .*"\
                f"already exists in {comp.name}"
        with pytest.warns(UserWarning, match=regexp):
            comp.add_backpropagation_learning_pathway(pathway=[A,P,B])

    def test_add_backpropagation_pathway_inferred_duplicate_warning(self):
        A = TransferMechanism()
        B = TransferMechanism()
        C = TransferMechanism()
        comp = Composition()
        comp.add_backpropagation_learning_pathway(pathway=[A,B,C])

        regexp = "Pathway specified in 'pathway' arg for add_backpropagation_learning_pathway method .*"\
               f"has same Nodes in same order as one already in {comp.name}"
        with pytest.warns(UserWarning, match=regexp):
            comp.add_backpropagation_learning_pathway(pathway=[A,B,C])

    def test_add_backpropagation_pathway_contiguous_subset_duplicate_warning(self):
        A = TransferMechanism()
        B = TransferMechanism()
        C = TransferMechanism()
        comp = Composition()
        comp.add_backpropagation_learning_pathway(pathway=[A,B,C])

        regexp = "Pathway specified in 'pathway' arg for add_backpropagation_learning_pathway method .*"\
                 f"has same Nodes in same order as one already in {comp.name}"
        with pytest.warns(UserWarning, match=regexp):
            comp.add_backpropagation_learning_pathway(pathway=[A,B])

    def test_add_processing_pathway_non_contiguous_subset_is_OK(self):
        A = TransferMechanism()
        B = TransferMechanism()
        C = TransferMechanism()
        comp = Composition()
        comp.add_linear_processing_pathway(pathway=[A,B,C])
        comp.add_linear_processing_pathway(pathway=[A,C])
        {A,B,C} == set(comp.nodes)
        len(comp.pathways)==2

    def test_add_processing_pathway_same_nodes_but_reversed_order_is_OK(self):
        A = TransferMechanism()
        B = TransferMechanism()
        comp = Composition()
        comp.add_linear_processing_pathway(pathway=[A,B])
        comp.add_linear_processing_pathway(pathway=[B,A])
        {A,B} == set(comp.nodes)
        len(comp.pathways)==2


@pytest.mark.pathways
class TestCompositionPathwaysArg:

    def test_composition_pathways_arg_pathway_object(self):
        A = ProcessingMechanism(name='A')
        p = Pathway(pathway=A, name='P')
        c = Composition(pathways=p)
        assert set(c.get_roles_by_node(A)) == {NodeRole.INPUT,
                                               NodeRole.ORIGIN,
                                               NodeRole.SINGLETON,
                                               NodeRole.OUTPUT,
                                               NodeRole.TERMINAL}
        assert set(c.pathways['P'].roles) == {PathwayRole.INPUT,
                                              PathwayRole.ORIGIN,
                                              PathwayRole.SINGLETON,
                                              PathwayRole.OUTPUT,
                                              PathwayRole.TERMINAL}

    def test_composition_pathways_arg_pathway_object_in_dict_with_name(self):
        A = ProcessingMechanism(name='A')
        p = Pathway(pathway=[A], name='P')
        c = Composition(pathways={'DICT NAMED':p})
        assert set(c.get_roles_by_node(A)) == {NodeRole.INPUT,
                                               NodeRole.ORIGIN,
                                               NodeRole.SINGLETON,
                                               NodeRole.OUTPUT,
                                               NodeRole.TERMINAL}
        assert set(c.pathways['DICT NAMED'].roles) == {PathwayRole.INPUT,
                                                       PathwayRole.ORIGIN,
                                                       PathwayRole.SINGLETON,
                                                       PathwayRole.OUTPUT,
                                                       PathwayRole.TERMINAL}

    def test_composition_pathways_arg_mech(self):
        A = ProcessingMechanism(name='A')
        c = Composition(pathways=A)
        assert set(c.get_roles_by_node(A)) == {NodeRole.INPUT,
                                               NodeRole.ORIGIN,
                                               NodeRole.SINGLETON,
                                               NodeRole.OUTPUT,
                                               NodeRole.TERMINAL}
        assert set(c.pathways[0].roles) == {PathwayRole.INPUT,
                                            PathwayRole.ORIGIN,
                                            PathwayRole.SINGLETON,
                                            PathwayRole.OUTPUT,
                                            PathwayRole.TERMINAL}

    def test_composition_pathways_arg_set(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        c = Composition({A,B,C})
        # # MODIFIED 4/11/22 OLD:
        # # assert all(name in c.pathways.names for name in {'Pathway-0', 'Pathway-1', 'Pathway-2'})
        # MODIFIED 4/11/22 NEW:
        assert all(name in c.pathways.names for name in {'Pathway-0'})
        # MODIFIED 4/11/22 END
        assert all(set(c.get_roles_by_node(node)) == {NodeRole.INPUT,
                                                      NodeRole.ORIGIN,
                                                      NodeRole.SINGLETON,
                                                      NodeRole.OUTPUT,
                                                      NodeRole.TERMINAL}
                   for node in {A,B,C})
        # assert all(set(c.pathways[i].roles) == {PathwayRole.INPUT,
        #                                     PathwayRole.ORIGIN,
        #                                     PathwayRole.SINGLETON,
        #                                     PathwayRole.OUTPUT,
        #                                     PathwayRole.TERMINAL}
        #            for i in range(0,1))

        with pytest.raises(CompositionError) as err:
            d = Composition({A,B,C.input_port})
        assert f"Every item in the \'pathways\' arg of the constructor for Composition-1 must be " \
               f"a Node, list, set, tuple or dict; the following are not: (InputPort InputPort-0)" in str(err.value)

    @pytest.mark.parametrize("nodes_config", [
        "many_many",
        "many_one_many",
    ])
    @pytest.mark.parametrize("projs", [
        "none",
        "default_proj",
        "matrix_spec",
        "some_projs_no_default",
        "some_projs_and_matrix_spec",
        "some_projs_and_default_proj"
    ])
    @pytest.mark.parametrize("set_or_list", [
        "set",
        "list"
    ])
    def test_composition_pathways_arg_with_various_set_or_list_configurations(self, nodes_config, projs, set_or_list):
        import itertools

        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        # FIX: 4/9/22 - INCLUDE TWO PORT MECHANISM:
        # B_comparator = ComparatorMechanism(name='B COMPARATOR')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        E = ProcessingMechanism(name='E')
        F = ProcessingMechanism(name='F')
        M = ProcessingMechanism(name='M')
        # C = A.input_port
        # proj = MappingProjection(sender=A, receiver=B)

        default_proj = MappingProjection(matrix=[2])
        default_matrix = [1]
        # For many_many:
        A_D = MappingProjection(sender=A, receiver=D, matrix=[2])
        B_D = MappingProjection(sender=B, receiver=D, matrix=[3])
        B_E = MappingProjection(sender=B, receiver=E, matrix=[4])
        C_E = MappingProjection(sender=C, receiver=E, matrix=[5])
        # For many_one_many:
        A_M = MappingProjection(sender=A, receiver=M, matrix=[6])
        C_M = MappingProjection(sender=C, receiver=M, matrix=[7])
        M_D = MappingProjection(sender=M, receiver=D, matrix=[8])
        M_F = MappingProjection(sender=M, receiver=F, matrix=[9])
        B_M = MappingProjection(sender=B, receiver=M, matrix=[100])

        nodes_1 = {A,B,C}
        nodes_2 = {D,E,F}
        # FIX: 4/9/22 - MODIFY TO INCLUDE many to first (set->list) and last to many(list->set)
        # FIX: 4/9/22 - INCLUDE PORT SPECS:
        # nodes_1 = {A.output_port,B,C} if set_or_list == 'set' else [A.output_port,B,C]
        # nodes_2 = {D,E,F.input_port} if set_or_list == 'set' else [D,E,F.input_port]

        if projs != "none":
            if nodes_config == "many_many":
                projections = {
                    "default_proj": default_proj,
                    "matrix_spec": [10],
                    "some_projs_no_default": {A_D, B_E} if set_or_list == 'set' else [A_D, B_E],
                    "some_projs_and_matrix_spec":  [A_D, C_E, default_matrix], # matrix spec requires list
                    "some_projs_and_default_proj":
                        {B_D, B_E, default_proj} if set_or_list == 'set' else [B_D, B_E, default_proj]
                }
            elif nodes_config == "many_one_many":
                # Tuples with first item for nodes_1 -> M and second item M -> nodes_2
                projections = {
                    "default_proj": (default_proj, default_proj),
                    "matrix_spec": ([11], [12]),
                    "some_projs_no_default":
                        ({A_M, C_M}, {M_D, M_F}) if set_or_list == 'set' else ([A_M, C_M], [M_D, M_F]),
                    "some_projs_and_matrix_spec":  ([A_M, C_M, default_matrix],
                                              [M_D, M_F, default_matrix]),  # matrix spec requires list
                    "some_projs_and_default_proj":
                        ({A_M, C_M, default_proj}, {M_D, M_F, default_proj})
                        if set_or_list == 'set' else ([A_M, C_M, default_proj], [M_D, M_F, default_proj])
                }
            else:
                assert False, f"TEST ERROR: No handling for '{nodes_config}' condition."

        if projs in {'default_proj', 'some_projs_and_default_proj'}:
            matrix_val = default_proj._init_args['matrix']
        elif projs == 'matrix_spec':
            matrix_val = projections[projs]
        elif projs == "some_projs_and_matrix_spec":
            matrix_val = default_matrix

        if nodes_config == "many_many":

            if projs == 'none':
                comp = Composition([nodes_1, nodes_2])
                matrix_val = default_matrix
            else:
                comp = Composition([nodes_1, projections[projs], nodes_2])

            if projs == "some_projs_no_default":
                assert A_D in comp.projections
                assert B_E in comp.projections
                # Pre-specified Projections that were not included in pathways should not be in Composition:
                assert B_D not in comp.projections
                assert C_E not in comp.projections
                assert C in comp.get_nodes_by_role(NodeRole.SINGLETON)
                assert F in comp.get_nodes_by_role(NodeRole.SINGLETON)

            else:
                # If there is no Projection specification or a default one, then there should be all-to-all Projections
                # Each sender projects to all three 3 receivers
                assert all(len([p for p in node.efferents if p in comp.projections])==3 for node in {A,B,C})
                # Each receiver gets Projections from all 3 senders
                assert all(len([p for p in node.path_afferents if p in comp.projections])==3 for node in {D,E,F})
                for sender,receiver in itertools.product([A,B,C],[D,E,F]):
                    # Each sender projects to each of the receivers
                    assert sender in {p.sender.owner for p in receiver.path_afferents if p in comp.projections}
                    # Each receiver receives a Projection from each of the senders
                    assert receiver in {p.receiver.owner for p in sender.efferents if p in comp.projections}

                # Matrices for pre-specified Projections should preserve their specified value:
                A_D.parameters.matrix.get() == [2]
                B_D.parameters.matrix.get() == [3]
                B_E.parameters.matrix.get() == [4]
                C_E.parameters.matrix.get() == [5]
                # Matrices for pairs without pre-specified Projections should be assigned value of default
                assert [p.parameters.matrix.get() for p in A.efferents if p.receiver.owner.name == 'E'] == matrix_val
                assert [p.parameters.matrix.get() for p in A.efferents if p.receiver.owner.name == 'F'] == matrix_val
                assert [p.parameters.matrix.get() for p in B.efferents if p.receiver.owner.name == 'F'] == matrix_val
                assert [p.parameters.matrix.get() for p in C.efferents if p.receiver.owner.name == 'D'] == matrix_val
                assert [p.parameters.matrix.get() for p in C.efferents if p.receiver.owner.name == 'F'] == matrix_val

        elif nodes_config == 'many_one_many':
            if projs == 'none':
                comp = Composition([nodes_1, M, nodes_2])
                matrix_val = default_matrix

            else:
                comp = Composition([nodes_1, projections[projs][0], M, projections[projs][1], nodes_2])
                if projs == 'matrix_spec':
                    matrix_val = projections[projs][1]

            if projs == "some_projs_no_default":
                assert all(p in comp.projections for p in {A_M, C_M, M_D, M_F})
                # Pre-specified Projections that were not included in pathways should not be in Composition:
                assert B_M not in comp.projections
                assert B in comp.get_nodes_by_role(NodeRole.SINGLETON)
                assert E in comp.get_nodes_by_role(NodeRole.SINGLETON)

            else:
                # Each sender projects to just one receiver
                assert all(len([p for p in node.efferents if p in comp.projections])==1 for node in {A,B,C})
                # Each receiver receives from just one sender
                assert all(len([p for p in node.path_afferents if p in comp.projections])==1 for node in {D,E,F})
                for sender,receiver in itertools.product([A,B,C],[M]):
                    # Each sender projects to M:
                    assert sender in {p.sender.owner for p in receiver.path_afferents if p in comp.projections}
                    # Each receiver receives from M:
                    assert receiver in {p.receiver.owner for p in sender.efferents if p in comp.projections}
                # Matrices for pre-specified Projections should preserve their specified value:
                A_M.parameters.matrix.get() == [6]
                C_M.parameters.matrix.get() == [7]
                M_D.parameters.matrix.get() == [8]
                M_F.parameters.matrix.get() == [9]
                # Matrices for pairs without pre-specified Projections should be assigned value of default
                assert [p.parameters.matrix.get() for p in B.efferents if p.receiver.owner.name == 'M'] == [100]
                assert [p.parameters.matrix.get() for p in M.efferents if p.receiver.owner.name == 'E'] == matrix_val

        else:
            assert False, f"TEST ERROR: No handling for '{nodes_config}' condition."

    def test_pathways_examples(self):
        pass

    def test_composition_pathways_arg_dict_and_list_and_pathway_roles(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        c = Composition(pathways=[{'P1':[A,B]}, [C,D]])
        assert all(n in {A, C} for n in c.get_nodes_by_role(NodeRole.INPUT))
        assert all(n in {B, D} for n in c.get_nodes_by_role(NodeRole.OUTPUT))
        assert c.pathways['P1'].name == 'P1'
        assert set(c.pathways['P1'].roles) == {PathwayRole.ORIGIN,
                                               PathwayRole.INPUT,
                                               PathwayRole.OUTPUT,
                                               PathwayRole.TERMINAL}
        assert set(c.pathways['P1'].roles).isdisjoint({PathwayRole.SINGLETON,
                                                       PathwayRole.CYCLE,
                                                       PathwayRole.CONTROL,
                                                       PathwayRole.LEARNING})
        assert set(c.pathways[1].roles) == {PathwayRole.ORIGIN,
                                            PathwayRole.INPUT,
                                            PathwayRole.OUTPUT,
                                            PathwayRole.TERMINAL}
        assert set(c.pathways[1].roles).isdisjoint({PathwayRole.SINGLETON,
                                                    PathwayRole.CYCLE,
                                                    PathwayRole.CONTROL,
                                                    PathwayRole.LEARNING})

    def test_composition_pathways_arg_dict_and_node(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        c = Composition(pathways=[{'P1':[A,B]}, C])
        assert all(n in {B, C} for n in c.get_nodes_by_role(NodeRole.OUTPUT))
        assert c.pathways['P1'].name == 'P1'

    def test_composition_pathways_arg_two_dicts(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        c = Composition(pathways=[{'P1':[A,B]}, {'P2':[C,D]}])
        assert all(n in {B, D} for n in c.get_nodes_by_role(NodeRole.OUTPUT))
        assert c.pathways['P1'].name == 'P1'
        assert c.pathways['P2'].name == 'P2'

    def test_composition_pathways_arg_two_dicts_one_with_node(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        c = Composition(pathways=[{'P1':[A,B]}, {'P2':C}])
        assert all(n in {B, C} for n in c.get_nodes_by_role(NodeRole.OUTPUT))
        assert c.pathways['P1'].name == 'P1'
        assert c.pathways['P2'].name == 'P2'

    def test_composition_pathways_bad_arg_error(self):
        I = InputPort(name='I')
        with pytest.raises(pnl.CompositionError) as error_text:
            c = Composition(pathways=I)
        assert ("The \'pathways\' arg of the constructor" in str(error_text.value) and
                "must be a Node, list, tuple, dict or Pathway object" in str(error_text.value))

    def test_composition_pathways_arg_pathways_list_and_item_not_list_or_dict_or_node_error(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        with pytest.raises(pnl.CompositionError) as error_text:
            c = Composition(pathways=[[A,B], 'C'])
        assert ("Every item in the 'pathways' arg of the constructor for Composition-0 must be "
                "a Node, list, set, tuple or dict; the following are not: 'C'" in str(error_text.value))

    def test_composition_pathways_arg_pathways_dict_and_item_not_list_dict_or_node_error(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        with pytest.raises(pnl.CompositionError) as error_text:
            c = Composition(pathways=[{'P1':[A,B]}, 'C'])
        assert ("Every item in the 'pathways' arg of the constructor for Composition-0 must be "
                "a Node, list, set, tuple or dict; the following are not: 'C'" in str(error_text.value))

    def test_composition_pathways_arg_dict_with_more_than_one_entry_error(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        with pytest.raises(pnl.CompositionError) as error_text:
            c = Composition(pathways=[{'P1':[A,B], 'P2':[C,D]}])
        assert ("A dict specified in the \'pathways\' arg of the constructor" in str(error_text.value)
                and "contains more than one entry:" in str(error_text.value))

    def test_composition_pathways_arg_dict_with_non_string_key_error(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        with pytest.raises(pnl.CompositionError) as error_text:
            c = Composition(pathways=[{A:[B,C]}])
        assert ("The key in a dict specified in the \'pathways\' arg of the constructor" in str(error_text.value) and
                "must be a str (to be used as its name):" in str(error_text.value))

    def test_composition_pathways_arg_dict_with_non_list_or_node_value_error(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        with pytest.raises(pnl.CompositionError) as error_text:
            c = Composition(pathways=[{'P1':'A'}])
        assert ("The value in a dict specified in the \'pathways\' arg of the constructor" in str(error_text.value) and
                "must be a pathway specification (Node, list or tuple): A." in str(error_text.value))

    def test_composition_pathways_Pathway_in_learning_tuples(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        E = ProcessingMechanism(name='E')
        P1 = Pathway(pathway=[A,B,C], name='P1')
        P2 = Pathway(pathway=[D,E], name='P2')
        c = Composition(pathways=[(P1, BackPropagation), (P2, BackPropagation)])
        assert c.pathways['P1'].name == 'P1'
        assert c.pathways['P2'].name == 'P2'
        assert c.pathways['P1'].learning_components[OUTPUT_MECHANISM] is C
        assert c.pathways['P2'].learning_components[OUTPUT_MECHANISM] is E

    def test_composition_processing_and_learning_pathways_pathwayroles_learning_components(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        c = Composition(pathways=[{'P1':[A,B]}, {'P2':([C,D], pnl.BackPropagation)}])
        assert set(c.get_nodes_by_role(NodeRole.OUTPUT)) == {B, D}
        assert c.pathways['P1'].name == 'P1'
        assert c.pathways['P2'].name == 'P2'
        assert c.pathways['P2'].target == c.nodes['Target']
        assert set(c.pathways['P1'].roles) == {PathwayRole.ORIGIN,
                                               PathwayRole.INPUT,
                                               PathwayRole.OUTPUT,
                                               PathwayRole.TERMINAL}
        assert set(c.pathways['P1'].roles).isdisjoint({PathwayRole.SINGLETON,
                                                       PathwayRole.CYCLE,
                                                       PathwayRole.CONTROL,
                                                       PathwayRole.LEARNING})
        assert set(c.pathways['P2'].roles)  == {PathwayRole.ORIGIN,
                                                PathwayRole.INPUT,
                                                PathwayRole.OUTPUT,
                                                PathwayRole.LEARNING}
        assert set(c.pathways['P2'].roles).isdisjoint({PathwayRole.SINGLETON,
                                                       PathwayRole.CYCLE,
                                                       PathwayRole.CONTROL})
        assert isinstance(c.pathways['P2'].learning_components[OBJECTIVE_MECHANISM], ObjectiveMechanism)
        assert isinstance(c.pathways['P2'].learning_components[TARGET_MECHANISM], ProcessingMechanism)
        assert (len(c.pathways['P2'].learning_components[LEARNING_MECHANISMS])
                and all(isinstance(lm, LearningMechanism)
                        for lm in c.pathways['P2'].learning_components[LEARNING_MECHANISMS]))
        assert (len(c.pathways['P2'].learning_components[LEARNED_PROJECTIONS])
                and all(isinstance(lm, MappingProjection)
                        for lm in c.pathways['P2'].learning_components[LEARNED_PROJECTIONS]))

    def test_composition_learning_pathway_dict_and_tuple(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        c = Composition(pathways=[{'P1':([A,B], pnl.BackPropagation)}, ([C,D], pnl.BackPropagation)])
        assert all(n in {B, D} for n in c.get_nodes_by_role(NodeRole.OUTPUT))
        assert c.pathways['P1'].name == 'P1'
        assert c.pathways['P1'].target == c.nodes['Target']

    def test_composition_pathways_bad_arg_error(self):
        I = InputPort(name='I')
        with pytest.raises(pnl.CompositionError) as error_text:
            c = Composition(pathways=I)
        assert ("The 'pathways' arg of the constructor for Composition-0 must be a Node, list, "
                "set, tuple, dict or Pathway object: (InputPort I [Deferred Init])."
                in str(error_text.value))

    def test_composition_arg_pathways_list_and_item_not_list_or_dict_or_node_error(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        with pytest.raises(pnl.CompositionError) as error_text:
            c = Composition(pathways=[[A,B], 'C'])
        assert ("Every item in the 'pathways' arg of the constructor for Composition-0 must be a "
                "Node, list, set, tuple or dict; the following are not: 'C'" in str(error_text.value))

    def test_composition_learning_pathway_dict_and_list_error(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        with pytest.raises(pnl.CompositionError) as error_text:
            c = Composition(pathways=[{'P1':([A,B], pnl.BackPropagation)}, [C,D]])
        assert ("An item" in str(error_text.value) and "is not a dict or tuple." in str(error_text.value))

    def test_composition_learning_pathway_dict_and_list_error(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        with pytest.raises(pnl.CompositionError) as error_text:
            c = Composition(pathways=[{'P1':([A,B], pnl.BackPropagation),
                                                'P2':([C,D], pnl.BackPropagation)}])
        assert ("A dict" in str(error_text.value) and "contains more than one entry" in str(error_text.value))

    def test_composition_learning_pathways_arg_dict_with_non_str_key_error(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        with pytest.raises(pnl.CompositionError) as error_text:
            c = Composition(pathways={C:([A,B], pnl.BackPropagation)})
        assert ("The key" in str(error_text.value) and "must be a str" in str(error_text.value))

    def test_composition_learning_pathway_to_few_mechs_error(self):
        A = ProcessingMechanism(name='A')
        with pytest.raises(pnl.CompositionError) as error_text:
            c = Composition(pathways=[{'P1': (A, pnl.BackPropagation)}])
        assert ("Backpropagation pathway specification does not have enough components:" in str(error_text.value))

    def test_composition_learning_pathway_dict_with_no_learning_fct_in_tuple_error(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        with pytest.raises(pnl.CompositionError) as error_text:
            c = Composition(pathways=[{'P1': ([A,B],C)}])
        assert ("The 2nd item" in str(error_text.value) and "must be a LearningFunction" in str(error_text.value))


class TestProperties:

    @pytest.mark.control
    @pytest.mark.parametrize("control_spec", [CONTROL, PROJECTIONS])
    def test_properties(self, control_spec):

        Input = pnl.TransferMechanism(name='Input')
        Reward = pnl.TransferMechanism(output_ports=[pnl.RESULT, pnl.MEAN, pnl.VARIANCE], name='reward')
        Decision = pnl.DDM(function=pnl.DriftDiffusionAnalytical,
                           output_ports=[pnl.DECISION_VARIABLE,
                                         pnl.RESPONSE_TIME,
                                         pnl.PROBABILITY_UPPER_THRESHOLD],
                           name='Decision')
        task_execution_pathway = [Input, pnl.IDENTITY_MATRIX, (Decision, NodeRole.OUTPUT)]
        comp = pnl.Composition(name="evc", retain_old_simulation_data=True, pathways=[task_execution_pathway,
                                                                                      Reward])
        comp.add_controller(
            controller=pnl.OptimizationControlMechanism(
                agent_rep=comp,
                num_estimates=2,
                state_features=[Input.input_port, Reward.input_port],
                state_feature_function=pnl.AdaptiveIntegrator(rate=0.1),
                monitor_for_control=[Reward,
                                     Decision.output_ports[pnl.PROBABILITY_UPPER_THRESHOLD],
                                     Decision.output_ports[pnl.RESPONSE_TIME]],
                function=pnl.GridSearch(),
                control_signals=[{control_spec: ("drift_rate", Decision),
                                  ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)},
                                 {control_spec: ("threshold", Decision),
                                  ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)},
                                 {control_spec: ("slope", Reward),
                                  ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)}]))

        assert len(comp.nodes) == len(comp.mechanisms) == 3
        assert len(list(comp._all_nodes)) == 7
        assert len(comp.input_ports) == len(comp.get_nodes_by_role(NodeRole.INPUT)) == 2
        assert len(comp.external_input_ports) == len(comp.input_ports)
        assert len(comp.output_ports) / 3  == len(comp.get_nodes_by_role(NodeRole.OUTPUT)) == 2
        assert len(comp.shadowing_dict) == 2
        assert len(comp.stateful_nodes) == 0
        assert len(comp.stateful_parameters) == 10
        assert len(comp.random_variables) == 2
        assert len(comp._dependent_components) == 25
        assert len(comp.afferents) == len(comp.efferents) == 0
        assert isinstance(comp.controller, OptimizationControlMechanism)
        assert len(comp.projections) == 18
        assert len([proj for proj in comp.projections if isinstance(proj, MappingProjection)]) == 14
        assert len([proj for proj in comp.projections if isinstance(proj, ControlProjection)]) == 4


class TestAnalyzeGraph:

    def test_empty_call(self):
        comp = Composition()
        comp._analyze_graph()

    def test_singleton(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        comp.add_node(A)
        comp._analyze_graph()
        assert A in comp.get_nodes_by_role(NodeRole.ORIGIN)
        assert A in comp.get_nodes_by_role(NodeRole.TERMINAL)

    def test_two_independent(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B')
        comp.add_node(A)
        comp.add_node(B)
        comp._analyze_graph()
        assert A in comp.get_nodes_by_role(NodeRole.ORIGIN)
        assert B in comp.get_nodes_by_role(NodeRole.ORIGIN)
        assert A in comp.get_nodes_by_role(NodeRole.TERMINAL)
        assert B in comp.get_nodes_by_role(NodeRole.TERMINAL)

    def test_two_in_a_row(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B')
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(), A, B)
        comp._analyze_graph()
        assert A in comp.get_nodes_by_role(NodeRole.ORIGIN)
        assert B not in comp.get_nodes_by_role(NodeRole.ORIGIN)
        assert A not in comp.get_nodes_by_role(NodeRole.TERMINAL)
        assert B in comp.get_nodes_by_role(NodeRole.TERMINAL)

    # (A)<->(B)
    def test_two_recursive(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B')
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(), A, B)

        comp.add_projection(MappingProjection(), B, A)
        comp._analyze_graph()
        assert A in comp.get_nodes_by_role(NodeRole.ORIGIN)
        assert B in comp.get_nodes_by_role(NodeRole.ORIGIN)
        assert A in comp.get_nodes_by_role(NodeRole.TERMINAL)
        assert B in comp.get_nodes_by_role(NodeRole.TERMINAL)

    # (A)->(B)<->(C)<-(D)
    @pytest.mark.skip
    def test_two_origins_pointing_to_recursive_pair(self):
        comp = Composition()
        A = TransferMechanism(name='composition-pytests-A')
        B = TransferMechanism(name='composition-pytests-B')
        C = TransferMechanism(name='composition-pytests-C')
        D = TransferMechanism(name='composition-pytests-D')
        comp.add_node(A)
        comp.add_node(B)
        comp.add_node(C)
        comp.add_node(D)
        comp.add_projection(MappingProjection(), A, B)
        comp.add_projection(MappingProjection(), C, B)
        comp.add_projection(MappingProjection(), B, C)
        comp.add_projection(MappingProjection(), D, C)
        comp._analyze_graph()
        assert A in comp.get_nodes_by_role(NodeRole.ORIGIN)
        assert D in comp.get_nodes_by_role(NodeRole.ORIGIN)
        assert B in comp.get_nodes_by_role(NodeRole.CYCLE)
        assert C in comp.get_nodes_by_role(NodeRole.RECURRENT_INIT)

    @pytest.mark.control
    def test_controller_objective_mech_not_terminal(self):
        comp = Composition()
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        comp.add_linear_processing_pathway([A, B])
        comp.add_controller(controller=pnl.OptimizationControlMechanism(agent_rep=comp,
                                                                        state_features=[A.input_port],
                                                                        objective_mechanism=pnl.ObjectiveMechanism(
                                                                                function=pnl.LinearCombination(
                                                                                        operation=pnl.PRODUCT),
                                                                                monitor=[A]),
                                                                        function=pnl.GridSearch(),
                                                                        control_signals=[
                                                                            {PROJECTIONS:("slope", B),
                                                                             ALLOCATION_SAMPLES:np.arange(0.1,
                                                                                                          1.01,
                                                                                                          0.3)}]
                                                                        )
                                       )
        # # MODIFIED 4/25/20 OLD:
        # comp._analyze_graph()
        # MODIFIED 4/25/20 END
        assert comp.controller.objective_mechanism not in comp.get_nodes_by_role(NodeRole.OUTPUT)

        # disable controller
        comp.enable_controller = False
        comp._analyze_graph()
        # assert comp.controller.objective_mechanism in comp.get_nodes_by_role(NodeRole.OUTPUT)
        assert comp.controller.objective_mechanism not in comp.get_nodes_by_role(NodeRole.OUTPUT)

    @pytest.mark.control
    def test_controller_objective_mech_not_terminal_fall_back(self):
        comp = Composition()
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        comp.add_linear_processing_pathway([A, B])

        comp.add_controller(controller=pnl.OptimizationControlMechanism(agent_rep=comp,
                                                                        state_features=[A.input_port],
                                                                        objective_mechanism=pnl.ObjectiveMechanism(
                                                                                function=pnl.LinearCombination(
                                                                                        operation=pnl.PRODUCT),
                                                                                monitor=[A, B]),
                                                                        function=pnl.GridSearch(),
                                                                        control_signals=[
                                                                            {PROJECTIONS:("slope", B),
                                                                             ALLOCATION_SAMPLES:np.arange(0.1,
                                                                                                          1.01,
                                                                                                          0.3)}]
                                                                        )
                                       )
        comp._analyze_graph()
        # ObjectiveMechanism associated with controller should not be considered an OUTPUT node
        assert comp.controller.objective_mechanism not in comp.get_nodes_by_role(NodeRole.OUTPUT)
        assert B in comp.get_nodes_by_role(NodeRole.OUTPUT)

        # disable controller
        comp.enable_controller = False
        comp._analyze_graph()

        # assert comp.controller.objective_mechanism in comp.get_nodes_by_role(NodeRole.OUTPUT)
        # assert B not in comp.get_nodes_by_role(NodeRole.OUTPUT)

        # ObjectiveMechanism associated with controller should be treated the same (i.e., not be an OUTPUT node)
        #    irrespective of whether the controller is enabled or disabled
        assert comp.controller.objective_mechanism not in comp.get_nodes_by_role(NodeRole.OUTPUT)
        assert B in comp.get_nodes_by_role(NodeRole.OUTPUT)


class TestGraph:

    class TestProcessingGraph:

        def test_all_mechanisms(self):
            comp = Composition()
            A = TransferMechanism(function=Linear(slope=5.0, intercept=2.0), name='composition-pytests-A')
            B = TransferMechanism(function=Linear(intercept=4.0), name='composition-pytests-B')
            C = TransferMechanism(function=Linear(intercept=1.5), name='composition-pytests-C')
            mechs = [A, B, C]
            for m in mechs:
                comp.add_node(m)

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
                comp.add_node(m)
            comp.add_projection(MappingProjection(), A, B)
            comp.add_projection(MappingProjection(), B, C)

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
                comp.add_node(m)
            comp.add_projection(MappingProjection(), A, C)
            comp.add_projection(MappingProjection(), B, C)
            comp.add_projection(MappingProjection(), C, D)
            comp.add_projection(MappingProjection(), C, E)

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
                comp.add_node(m)
            comp.add_projection(MappingProjection(), A, B)
            comp.add_projection(MappingProjection(), B, C)
            comp.add_projection(MappingProjection(), C, A)

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
                comp.add_node(m)
            comp.add_projection(MappingProjection(), A, C)
            comp.add_projection(MappingProjection(), B, C)
            comp.add_projection(MappingProjection(), C, D)
            comp.add_projection(MappingProjection(), C, E)
            comp.add_projection(MappingProjection(), D, A)
            comp.add_projection(MappingProjection(), E, B)

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
                comp.add_node(m)
            comp.add_projection(MappingProjection(), A, C)
            comp.add_projection(MappingProjection(), B, C)
            comp.add_projection(MappingProjection(), C, D)
            comp.add_projection(MappingProjection(), C, E)
            comp.add_projection(MappingProjection(), D, A)
            comp.add_projection(MappingProjection(), D, B)
            comp.add_projection(MappingProjection(), E, A)
            comp.add_projection(MappingProjection(), E, B)

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


class TestGraphCycles:

    def test_recurrent_transfer_mechanisms(self):
        R1 = RecurrentTransferMechanism(auto=1.0)
        R2 = RecurrentTransferMechanism(auto=1.0,
                                        function=Linear(slope=2.0))
        comp = Composition()
        comp.add_linear_processing_pathway(pathway=[R1, R2])

        # Trial 0:
        # input to R1 = 1.0, output from R1 = 1.0
        # input to R2 = 1.0, output from R2 = 2.0

        # Trial 1:
        # input to R1 = 1.0 + 1.0, output from R1 = 2.0
        # input to R2 = 2.0 + 2.0, output from R2 = 8.0

        # Trial 2:
        # input to R1 = 1.0 + 2.0, output from R1 = 3.0
        # input to R2 = 3.0 + 8.0, output from R2 = 22.0


        output = comp.run(inputs={R1: [1.0]}, num_trials=3)
        assert np.allclose(output, [[np.array([22.])]])

@pytest.mark.pathways
class TestExecutionOrder:
    def test_2_node_loop(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        D = ProcessingMechanism(name="D")

        comp = Composition(name="comp")
        comp.add_linear_processing_pathway([A, B, C, D])
        comp.add_linear_processing_pathway([C, B])

        comp.run(inputs={A: 1.0})

    def test_double_loop(self):
        A1 = ProcessingMechanism(name="A1")
        A2 = ProcessingMechanism(name="A2")
        B1 = ProcessingMechanism(name="B1")
        B2 = ProcessingMechanism(name="B2")
        C1 = ProcessingMechanism(name="C1")
        C2 = ProcessingMechanism(name="C2")
        D = ProcessingMechanism(name="D")

        comp = Composition(name="comp")
        comp.add_linear_processing_pathway([A1, A2, D])
        comp.add_linear_processing_pathway([B1, B2, D])
        comp.add_linear_processing_pathway([C1, C2, D])
        comp.add_linear_processing_pathway([A2, B2])
        comp.add_linear_processing_pathway([B2, A2])
        comp.add_linear_processing_pathway([C2, B2])
        comp.add_linear_processing_pathway([B2, C2])

        comp.run(inputs={A1: 1.0,
                         B1: 1.0,
                         C1: 1.0})

    def test_feedback_pathway_spec(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")

        comp = Composition()
        comp.add_linear_processing_pathway([A, B, MappingProjection(matrix=2.0), C, MappingProjection(matrix=3.0), D, E])
        # comp.add_linear_processing_pathway([D, MappingProjection(matrix=4.0), B], feedback=True)
        comp.add_linear_processing_pathway([D, (MappingProjection(matrix=4.0), True), B])

        comp.run(inputs={A: 1.0})

        expected_consideration_queue = [{A}, {B}, {C}, {D}, {E}]
        assert all(expected_consideration_queue[i] == comp.scheduler.consideration_queue[i]
                   for i in range(len(comp.nodes)))

        expected_results = {A: 1.0,
                            B: 1.0,
                            C: 2.0,
                            D: 6.0,
                            E: 6.0}

        assert all(expected_results[mech] == mech.parameters.value.get(comp) for mech in expected_results)

        comp.run(inputs={A: 1.0})

        expected_results_2 = {A: 1.0,
                              B: 25.0,
                              C: 50.0,
                              D: 150.0,
                              E: 150.0}

        assert all(expected_results_2[mech] == mech.parameters.value.get(comp) for mech in expected_results_2)

    def test_feedback_projection_spec(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")

        comp = Composition()
        comp.add_linear_processing_pathway([A, B, MappingProjection(matrix=2.0), C, MappingProjection(matrix=3.0), D, E])
        comp.add_projection(projection=MappingProjection(matrix=4.0), sender=D, receiver=B, feedback=True)

        comp.run(inputs={A: 1.0})

        expected_consideration_queue = [{A}, {B}, {C}, {D}, {E}]
        assert all(expected_consideration_queue[i] == comp.scheduler.consideration_queue[i]
                   for i in range(len(comp.nodes)))

        expected_results = {A: 1.0,
                            B: 1.0,
                            C: 2.0,
                            D: 6.0,
                            E: 6.0}

        assert all(expected_results[mech] == mech.parameters.value.get(comp) for mech in expected_results)

        comp.run(inputs={A: 1.0})

        expected_results_2 = {A: 1.0,
                              B: 25.0,
                              C: 50.0,
                              D: 150.0,
                              E: 150.0}

        assert all(expected_results_2[mech] == mech.parameters.value.get(comp) for mech in expected_results_2)

    def test_outer_feedback_inner_loop(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")

        comp = Composition()
        comp.add_linear_processing_pathway([A, B, MappingProjection(matrix=2.0), C, MappingProjection(matrix=3.0), D, E])
        comp.add_projection(projection=MappingProjection(matrix=4.0), sender=D, receiver=B, feedback=True)
        comp.add_projection(projection=MappingProjection(matrix=1.0), sender=D, receiver=C, feedback=False)

        expected_consideration_queue = [{A}, {B}, {C, D}, {E}]
        assert all(expected_consideration_queue[i] == comp.scheduler.consideration_queue[i]
                   for i in range(len(expected_consideration_queue)))

    def test_inner_feedback_outer_loop(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")

        comp = Composition()
        comp.add_linear_processing_pathway([A, B, MappingProjection(matrix=2.0), C, MappingProjection(matrix=3.0), D, E])
        comp.add_projection(projection=MappingProjection(matrix=1.0), sender=D, receiver=B, feedback=False)
        comp.add_projection(projection=MappingProjection(matrix=4.0), sender=D, receiver=C, feedback=True)

        expected_consideration_queue = [{A}, {B, C, D}, {E}]
        assert all(expected_consideration_queue[i] == comp.scheduler.consideration_queue[i]
                   for i in range(len(expected_consideration_queue)))

    def test_origin_loop(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")

        comp = Composition()
        comp.add_linear_processing_pathway([A, B, MappingProjection(matrix=2.0), C, MappingProjection(matrix=3.0), D, E])
        comp.add_projection(projection=MappingProjection(matrix=1.0), sender=B, receiver=A, feedback=False)
        comp.add_projection(projection=MappingProjection(matrix=1.0), sender=C, receiver=B, feedback=False)

        expected_consideration_queue = [{A, B, C}, {D}, {E}]
        assert all(expected_consideration_queue[i] == comp.scheduler.consideration_queue[i]
                   for i in range(len(expected_consideration_queue)))

        comp._analyze_graph()
        assert set(comp.get_nodes_by_role(NodeRole.ORIGIN)) == expected_consideration_queue[0]

        new_origin = ProcessingMechanism(name="new_origin")
        comp.add_linear_processing_pathway([new_origin, B])

        expected_consideration_queue = [{new_origin}, {A, B, C}, {D}, {E}]
        assert all(expected_consideration_queue[i] == comp.scheduler.consideration_queue[i]
                   for i in range(len(expected_consideration_queue)))

        comp._analyze_graph()
        assert set(comp.get_nodes_by_role(NodeRole.ORIGIN)) == expected_consideration_queue[0]

    def test_terminal_loop(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")

        comp = Composition()
        comp.add_linear_processing_pathway([A, B, MappingProjection(matrix=2.0), C, MappingProjection(matrix=3.0), D, E])
        comp.add_projection(projection=MappingProjection(matrix=1.0), sender=E, receiver=D, feedback=False)
        comp.add_projection(projection=MappingProjection(matrix=1.0), sender=D, receiver=C, feedback=False)

        expected_consideration_queue = [{A}, {B}, {C, D, E}]
        assert all(expected_consideration_queue[i] == comp.scheduler.consideration_queue[i]
                   for i in range(len(expected_consideration_queue)))

        comp._analyze_graph()
        assert set(comp.get_nodes_by_role(NodeRole.TERMINAL)) == expected_consideration_queue[-1]

        new_terminal = ProcessingMechanism(name="new_terminal")
        comp.add_linear_processing_pathway([D, new_terminal])

        expected_consideration_queue = [{A}, {B}, {C, D, E}, {new_terminal}]
        assert all(expected_consideration_queue[i] == comp.scheduler.consideration_queue[i]
                   for i in range(len(expected_consideration_queue)))

        comp._analyze_graph()
        assert set(comp.get_nodes_by_role(NodeRole.TERMINAL)) == expected_consideration_queue[-1]

    def test_simple_loop(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")

        comp = Composition()
        comp.add_linear_processing_pathway([A, B, MappingProjection(matrix=2.0), C, MappingProjection(matrix=3.0), D, E])
        comp.add_linear_processing_pathway([D, MappingProjection(matrix=4.0), B])

        D.set_log_conditions("OutputPort-0")
        cycle_nodes = [B, C, D]
        for cycle_node in cycle_nodes:
            cycle_node.output_ports[0].parameters.value.set([1.0], override=True)

        comp.run(inputs={A: [1.0]})
        expected_values = {A: 1.0,
                           B: 5.0,
                           C: 2.0,
                           D: 3.0,
                           E: 3.0}

        for node in expected_values:
            assert np.allclose(expected_values[node], node.parameters.value.get(comp))

        comp.run(inputs={A: [1.0]})
        expected_values_2 = {A: 1.0,
                             B: 13.0,
                             C: 10.0,
                             D: 6.0,
                             E: 6.0}

        print(D.log.nparray_dictionary(["OutputPort-0"]))
        for node in expected_values:
            assert np.allclose(expected_values_2[node], node.parameters.value.get(comp))

    def test_loop_with_extra_node(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        C2 = ProcessingMechanism(name="C2")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")

        comp = Composition()

        cycle_nodes = [B, C, D, C2]
        for cycle_node in cycle_nodes:
            cycle_node.output_ports[0].parameters.value.set([1.0], override=True)

        comp.add_linear_processing_pathway([A, B, MappingProjection(matrix=2.0), C, D, MappingProjection(matrix=5.0), E])
        comp.add_linear_processing_pathway([D, MappingProjection(matrix=3.0), C2, MappingProjection(matrix=4.0), B])

        expected_consideration_queue = [{A}, {B, C, D, C2}, {E}]

        assert all(expected_consideration_queue[i] == comp.scheduler.consideration_queue[i] for i in range(3))
        comp.run(inputs={A: [1.0]})

        expected_values = {A: 1.0,
                           B: 5.0,
                           C: 2.0,
                           D: 1.0,
                           C2: 3.0,
                           E: 5.0}

        for node in expected_values:
            assert np.allclose(expected_values[node], node.parameters.value.get(comp))

        comp.run(inputs={A: [1.0]})
        expected_values_2 = {A: 1.0,
                             B: 13.0,
                             C: 10.0,
                             D: 2.0,
                             C2: 3.0,
                             E: 10.0}

        for node in expected_values:
            assert np.allclose(expected_values_2[node], node.parameters.value.get(comp))

    def test_two_overlapping_loops(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        C2 = ProcessingMechanism(name="C2")
        C3 = ProcessingMechanism(name="C3")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")

        comp = Composition()
        comp.add_linear_processing_pathway([A, B, C, D, E])
        comp.add_linear_processing_pathway([D, C2, B])
        comp.add_linear_processing_pathway([D, C3, B])

        comp.run(inputs={A: [1.0]})

        assert comp.scheduler.consideration_queue[0] == {A}
        assert comp.scheduler.consideration_queue[1] == {B, C, D, C2, C3}
        assert comp.scheduler.consideration_queue[2] == {E}

    def test_three_overlapping_loops(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        C2 = ProcessingMechanism(name="C2")
        C3 = ProcessingMechanism(name="C3")
        C4 = ProcessingMechanism(name="C4")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")

        comp = Composition()
        comp.add_linear_processing_pathway([A, B, C, D, E])
        comp.add_linear_processing_pathway([D, C2, B])
        comp.add_linear_processing_pathway([D, C3, B])
        comp.add_linear_processing_pathway([D, C4, B])

        comp.run(inputs={A: [1.0]})

        assert comp.scheduler.consideration_queue[0] == {A}
        assert comp.scheduler.consideration_queue[1] == {B, C, D, C2, C3, C4}
        assert comp.scheduler.consideration_queue[2] == {E}

    def test_two_separate_loops(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        L1 = ProcessingMechanism(name="L1")
        L2 = ProcessingMechanism(name="L2")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")
        F = ProcessingMechanism(name="F")

        comp = Composition()
        comp.add_linear_processing_pathway([A, B, C, D, E, F])
        comp.add_linear_processing_pathway([E, L1, D])
        comp.add_linear_processing_pathway([C, L2, B])

        comp.run(inputs={A: [1.0]})

        assert comp.scheduler.consideration_queue[0] == {A}
        assert comp.scheduler.consideration_queue[1] == {C, L2, B}
        assert comp.scheduler.consideration_queue[2] == {E, L1, D}
        assert comp.scheduler.consideration_queue[3] == {F}

    def test_two_paths_converge(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")

        comp = Composition()
        comp.add_linear_processing_pathway([A, B, C, D])
        comp.add_linear_processing_pathway([E, D])

        comp.run(inputs={A: 1.0,
                         E: 1.0})

        assert comp.scheduler.consideration_queue[0] == {A, E}
        assert comp.scheduler.consideration_queue[1] == {B}
        assert comp.scheduler.consideration_queue[2] == {C}
        assert comp.scheduler.consideration_queue[3] == {D}

    def test_diverge_and_reconverge(self):
        S = ProcessingMechanism(name="START")
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")

        comp = Composition()
        comp.add_linear_processing_pathway([S, A, B, C, D])
        comp.add_linear_processing_pathway([S, E, D])

        comp.run(inputs={S: 1.0})

        assert comp.scheduler.consideration_queue[0] == {S}
        assert comp.scheduler.consideration_queue[1] == {A, E}
        assert comp.scheduler.consideration_queue[2] == {B}
        assert comp.scheduler.consideration_queue[3] == {C}
        assert comp.scheduler.consideration_queue[4] == {D}

    def test_diverge_and_reconverge_2(self):
        S = ProcessingMechanism(name="START")
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")
        F = ProcessingMechanism(name="F")
        G = ProcessingMechanism(name="G")

        comp = Composition()
        comp.add_linear_processing_pathway([S, A, B, C, D])
        comp.add_linear_processing_pathway([S, E, F, G, D])

        comp.run(inputs={S: 1.0})

        assert comp.scheduler.consideration_queue[0] == {S}
        assert comp.scheduler.consideration_queue[1] == {A, E}
        assert comp.scheduler.consideration_queue[2] == {B, F}
        assert comp.scheduler.consideration_queue[3] == {C, G}
        assert comp.scheduler.consideration_queue[4] == {D}

    def test_figure_eight(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C1 = ProcessingMechanism(name="C1")
        D1 = ProcessingMechanism(name="D1")
        C2 = ProcessingMechanism(name="C2")
        D2 = ProcessingMechanism(name="D2")

        comp = Composition()

        comp.add_linear_processing_pathway([A, B])
        comp.add_linear_processing_pathway([B, C1, D1])
        comp.add_linear_processing_pathway([B, C2, D2])
        comp.add_linear_processing_pathway([D1, B])
        comp.add_linear_processing_pathway([D2, B])

        assert comp.scheduler.consideration_queue[0] == {A}
        assert comp.scheduler.consideration_queue[1] == {B, C1, D1, C2, D2}

    def test_many_loops(self):

        comp = Composition()

        start = ProcessingMechanism(name="start")
        expected_consideration_sets = [{start}]
        for i in range(10):
            A = ProcessingMechanism(name='A' + str(i))
            B = ProcessingMechanism(name='B' + str(i))
            C = ProcessingMechanism(name='C' + str(i))
            D = ProcessingMechanism(name='D' + str(i))

            comp.add_linear_processing_pathway([start, A, B, C, D])
            comp.add_linear_processing_pathway([C, B])

            expected_consideration_sets.append({A})
            expected_consideration_sets.append({B, C})
            expected_consideration_sets.append({D})

            start = D

        for i in range(len(comp.scheduler.consideration_queue)):
            assert comp.scheduler.consideration_queue[i] == expected_consideration_sets[i]

    def test_multiple_projections_along_pathway(self):

        comp = Composition()
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")
        D = ProcessingMechanism(name="D")
        E = ProcessingMechanism(name="E")

        comp.add_linear_processing_pathway([A, B, C, D, E])
        comp.add_linear_processing_pathway([A, C])
        comp.add_linear_processing_pathway([C, E])

        expected_consideration_queue = [{A}, {B}, {C}, {D}, {E}]

        assert expected_consideration_queue == comp.scheduler.consideration_queue

    @pytest.mark.composition
    @pytest.mark.benchmark(group="Frozen values")
    def test_3_mechanisms_frozen_values(self, benchmark, comp_mode):
        #
        #   B
        #  /|\
        # A-+-D
        #  \|/
        #   C
        #
        # A: 4 x 5 = 20
        # B: (20 + 0) x 4 = 80
        # C: (20 + 0) x 3 = 60
        # D: (20 + 80 + 60) x 2 = 320

        comp = Composition()
        A = TransferMechanism(name="A", function=Linear(slope=5.0))
        B = TransferMechanism(name="B", function=Linear(slope=4.0))
        C = TransferMechanism(name="C", function=Linear(slope=3.0))
        D = TransferMechanism(name="D", function=Linear(slope=2.0))
        comp.add_linear_processing_pathway([A, D])
        comp.add_linear_processing_pathway([B, C])
        comp.add_linear_processing_pathway([C, B])
        comp.add_linear_processing_pathway([A, B, D])
        comp.add_linear_processing_pathway([A, C, D])

        inputs_dict = {A: [4.0]}
        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched, execution_mode=comp_mode)
        assert np.allclose(output, 320)

        if benchmark.enabled:
            benchmark(comp.run, inputs=inputs_dict, scheduler=sched, execution_mode=comp_mode)

    @pytest.mark.control
    @pytest.mark.composition
    @pytest.mark.benchmark(group="Control composition scalar")
    def test_3_mechanisms_2_origins_1_multi_control_1_terminal(self, benchmark, comp_mode):
        #
        #   A--LC
        #  /    \
        # B------C
        #  \     |
        #   -----+-> D
        #
        # B: 4 x 5 = 20
        # A: 20 x 1 = 20
        # LC: f(20)[0] = 0.50838675
        # C: 20 x 5 x 0.50838675 = 50.83865743
        # D: (20 + 50.83865743) x 5 = 354.19328716

        comp = Composition()
        B = TransferMechanism(name="B", function=Linear(slope=5.0))
        C = TransferMechanism(name="C", function=Linear(slope=5.0))
        A = ObjectiveMechanism(function=Linear,
                               monitor=[B],
                               name="A")
        LC = LCControlMechanism(name="LC",
                               modulated_mechanisms=C,
                               objective_mechanism=A)
        D = TransferMechanism(name="D", function=Linear(slope=5.0))
        comp.add_linear_processing_pathway([B, C, D])
        comp.add_linear_processing_pathway([B, D])
        comp.add_node(A)
        comp.add_node(LC)


        inputs_dict = {B: [4.0]}
        output = comp.run(inputs=inputs_dict, execution_mode=comp_mode)
        assert np.allclose(output, 354.19328716)

        if benchmark.enabled:
            benchmark(comp.run, inputs=inputs_dict, execution_mode=comp_mode)

    @pytest.mark.control
    @pytest.mark.composition
    @pytest.mark.benchmark(group="Control composition scalar")
    def test_3_mechanisms_2_origins_1_additive_control_1_terminal(self, benchmark, comp_mode):
        #
        #   A--LC
        #  /    \
        # B------C
        #  \     |
        #   -----+-> D
        #
        # B: 4 x 5 = 20
        # A: 20 x 1 = 20
        # LC: f(20)[0] = 0.50838675
        # C: 20 x 5 + 0.50838675 = 100.50838675
        # D: (20 + 100.50838675) x 5 = 650.83865743

        comp = Composition()
        B = TransferMechanism(name="B", function=Linear(slope=5.0))
        C = TransferMechanism(name="C", function=Linear(slope=5.0))
        A = ObjectiveMechanism(function=Linear,
                               monitor=[B],
                               name="A")
        LC = LCControlMechanism(name="LC", modulation=ADDITIVE,
                               modulated_mechanisms=C,
                               objective_mechanism=A)
        D = TransferMechanism(name="D", function=Linear(slope=5.0))
        comp.add_linear_processing_pathway([B, C, D])
        comp.add_linear_processing_pathway([B, D])
        comp.add_node(A)
        comp.add_node(LC)

        inputs_dict = {B: [4.0]}
        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched, execution_mode=comp_mode)
        assert np.allclose(output, 650.83865743)

        if benchmark.enabled:
            benchmark(comp.run, inputs=inputs_dict, scheduler=sched, execution_mode=comp_mode)

    @pytest.mark.control
    @pytest.mark.composition
    @pytest.mark.benchmark(group="Control composition scalar")
    def test_3_mechanisms_2_origins_1_override_control_1_terminal(self, benchmark, comp_mode):
        #
        #   A--LC
        #  /    \
        # B------C
        #  \     |
        #   -----+-> D
        #
        # B: 4 x 5 = 20
        # A: 20 x 1 = 20
        # LC: f(20)[0] = 0.50838675
        # C: 20 x 0.50838675 = 10.167735
        # D: (20 + 10.167735) x 5 = 150.83865743

        comp = Composition()
        B = TransferMechanism(name="B", function=Linear(slope=5.0))
        C = TransferMechanism(name="C", function=Linear(slope=5.0))
        A = ObjectiveMechanism(function=Linear,
                               monitor=[B],
                               name="A")
        LC = LCControlMechanism(name="LC", modulation=OVERRIDE,
                               modulated_mechanisms=C,
                               objective_mechanism=A)
        D = TransferMechanism(name="D", function=Linear(slope=5.0))
        comp.add_linear_processing_pathway([B, C, D])
        comp.add_linear_processing_pathway([B, D])
        comp.add_node(A)
        comp.add_node(LC)


        inputs_dict = {B: [4.0]}
        output = comp.run(inputs=inputs_dict, execution_mode=comp_mode)
        assert np.allclose(output, 150.83865743)
        if benchmark.enabled:
            benchmark(comp.run, inputs=inputs_dict, execution_mode=comp_mode)

    @pytest.mark.control
    @pytest.mark.composition
    @pytest.mark.benchmark(group="Control composition scalar")
    def test_3_mechanisms_2_origins_1_disable_control_1_terminal(self, benchmark, comp_mode):
        #
        #   A--LC
        #  /    \
        # B------C
        #  \     |
        #   -----+-> D
        #
        # B: 4 x 5 = 20
        # A: 20 x 1 = 20
        # LC: f(20)[0] = 0.50838675
        # C: 20 x 5 = 100
        # D: (20 + 100) x 5 = 600

        comp = Composition()
        B = TransferMechanism(name="B", function=Linear(slope=5.0))
        C = TransferMechanism(name="C", function=Linear(slope=5.0))
        A = ObjectiveMechanism(function=Linear,
                               monitor=[B],
                               name="A")
        LC = LCControlMechanism(name="LC", modulation=DISABLE,
                               modulated_mechanisms=C,
                               objective_mechanism=A)
        D = TransferMechanism(name="D", function=Linear(slope=5.0))
        comp.add_linear_processing_pathway([B, C, D])
        comp.add_linear_processing_pathway([B, D])
        comp.add_node(A)
        comp.add_node(LC)


        inputs_dict = {B: [4.0]}
        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched, execution_mode=comp_mode)
        assert np.allclose(output, 600)

        if benchmark.enabled:
            benchmark(comp.run, inputs=inputs_dict, scheduler=sched, execution_mode=comp_mode)

    @pytest.mark.composition
    @pytest.mark.benchmark(group="Transfer")
    def xtest_transfer_mechanism(self, benchmark, comp_mode):

        # mechanisms
        C = TransferMechanism(name="C",
                              function=Logistic,
                              integration_rate=0.1,
                              integrator_mode=True)

        # comp2 uses a TransferMechanism in integrator mode
        comp2 = Composition(name="comp2")
        comp2.add_node(C)

        # pass same 3 trials of input to comp1 and comp2
        benchmark(comp2.run, inputs={C: [1.0, 2.0, 3.0]}, execution_mode=comp_mode)

        assert np.allclose(comp2.results[:3], [[[0.52497918747894]], [[0.5719961329315186]], [[0.6366838893983633]]])

    @pytest.mark.composition
    @pytest.mark.benchmark(group="Transfer")
    def test_transfer_mechanism_split(self, benchmark, comp_mode):

        # mechanisms
        A = ProcessingMechanism(name="A",
                                function=AdaptiveIntegrator(rate=0.1))
        B = ProcessingMechanism(name="B",
                                function=Logistic)

        # comp1 separates IntegratorFunction fn and Logistic fn into mech A and mech B
        comp1 = Composition(name="comp1")
        comp1.add_linear_processing_pathway([A, B])

        benchmark(comp1.run, inputs={A: [1.0, 2.0, 3.0]}, execution_mode=comp_mode)

        assert np.allclose(comp1.results[:3], [[[0.52497918747894]], [[0.5719961329315186]], [[0.6366838893983633]]])

    @pytest.mark.composition
    def test_exact_time(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")

        comp = Composition(name="comp")
        comp.add_linear_processing_pathway([A, B])

        # these cannot run at same execution set unless in EXACT_TIME
        comp.scheduler.add_condition(A, TimeInterval(start=1))
        comp.scheduler.add_condition(B, TimeInterval(start=1))

        comp.run(inputs={A: [1.0]}, scheduling_mode=SchedulingMode.EXACT_TIME)

        assert comp.scheduler.mode == SchedulingMode.EXACT_TIME
        assert comp.scheduler.execution_list[comp.default_execution_id] == [{A, B}]
        assert comp.scheduler.execution_timestamps[comp.default_execution_id][0].absolute == 1 * pnl._unit_registry.ms

@pytest.mark.pathways
class TestGetMechanismsByRole:

    def test_multiple_roles(self):

        comp = Composition()
        mechs = [TransferMechanism() for x in range(4)]

        for mech in mechs:
            comp.add_node(mech)

        comp._add_node_role(mechs[0], NodeRole.ORIGIN)
        comp._add_node_role(mechs[1], NodeRole.INTERNAL)
        comp._add_node_role(mechs[2], NodeRole.INTERNAL)

        for role in list(NodeRole):
            if role is NodeRole.ORIGIN:
                assert comp.get_nodes_by_role(role) == [mechs[0]]
            elif role is NodeRole.INTERNAL:
                assert comp.get_nodes_by_role(role) == [mechs[1], mechs[2]]
            else:
                assert comp.get_nodes_by_role(role) == []

    @pytest.mark.xfail(raises=CompositionError)
    def test_nonexistent_role(self):
        comp = Composition()
        comp.get_nodes_by_role(None)


class TestInputPortSpecifications:

    def test_two_input_ports_created_with_dictionaries(self):

        comp = Composition()
        A = ProcessingMechanism(
            name="composition-pytests-A",
            default_variable=[[0], [0]],
            # input_ports=[
            #     {NAME: "Input Port 1", },
            #     {NAME: "Input Port 2", }
            # ],
            function=Linear(slope=1.0)
            # specifying default_variable on the function doesn't seem to matter?
        )

        comp.add_node(A)


        inputs_dict = {A: [[2.], [4.]]}
        sched = Scheduler(composition=comp)
        comp.run(inputs=inputs_dict, scheduler=sched)

        assert np.allclose(A.input_ports[0].parameters.value.get(comp), [2.0])
        assert np.allclose(A.input_ports[1].parameters.value.get(comp), [4.0])
        assert np.allclose(A.parameters.variable.get(comp.default_execution_id), [[2.0], [4.0]])

    def test_recurrent_transfer_origin(self):
        R = RecurrentTransferMechanism(has_recurrent_input_port=True)
        C = Composition(pathways=[R])

        result = C.run(inputs={R: [[1.0], [2.0], [3.0]]})
        assert np.allclose(result, [[3.0]])

    def test_two_input_ports_created_first_with_deferred_init(self):
        comp = Composition()

        # create mechanism A
        I1 = InputPort(
            name="Input Port 1",
            reference_value=[0]
        )
        I2 = InputPort(
            name="Input Port 2",
            reference_value=[0]
        )
        A = TransferMechanism(
            name="composition-pytests-A",
            default_variable=[[0], [0]],
            input_ports=[I1, I2],
            function=Linear(slope=1.0)
        )

        # add mech A to composition
        comp.add_node(A)

        # get comp ready to run (identify roles, create sched, assign inputs)
        inputs_dict = { A: [[2.],[4.]]}

        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched)

        assert np.allclose(A.input_ports[0].parameters.value.get(comp), [2.0])
        assert np.allclose(A.input_ports[1].parameters.value.get(comp), [4.0])
        assert np.allclose(A.parameters.variable.get(comp.default_execution_id), [[2.0], [4.0]])

    def test_two_input_ports_created_with_keyword(self):
        comp = Composition()

        # create mechanism A

        A = TransferMechanism(
            name="composition-pytests-A",
            default_variable=[[0], [0]],
            input_ports=[INPUT_PORT, INPUT_PORT],
            function=Linear(slope=1.0)
        )

        # add mech A to composition
        comp.add_node(A)

        # get comp ready to run (identify roles, create sched, assign inputs)
        inputs_dict = {A: [[2.], [4.]]}

        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched)

        assert np.allclose(A.input_ports[0].parameters.value.get(comp), [2.0])
        assert np.allclose(A.input_ports[1].parameters.value.get(comp), [4.0])
        assert np.allclose(A.parameters.variable.get(comp.default_execution_id), [[2.0], [4.0]])

        assert np.allclose([[2], [4]], output)

    def test_two_input_ports_created_with_strings(self):
        comp = Composition()

        # create mechanism A

        A = TransferMechanism(
            name="composition-pytests-A",
            default_variable=[[0], [0]],
            input_ports=["Input Port 1", "Input Port 2"],
            function=Linear(slope=1.0)
        )

        # add mech A to composition
        comp.add_node(A)

        # get comp ready to run (identify roles, create sched, assign inputs)

        inputs_dict = {A: [[2.], [4.]]}

        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched)

        assert np.allclose(A.input_ports[0].parameters.value.get(comp), [2.0])
        assert np.allclose(A.input_ports[1].parameters.value.get(comp), [4.0])
        assert np.allclose(A.parameters.variable.get(comp.default_execution_id), [[2.0], [4.0]])

    def test_two_input_ports_created_with_values(self):
        comp = Composition()

        # create mechanism A

        A = TransferMechanism(
            name="composition-pytests-A",
            default_variable=[[0], [0]],
            input_ports=[[0.], [0.]],
            function=Linear(slope=1.0)
        )

        # add mech A to composition
        comp.add_node(A)

        # get comp ready to run (identify roles, create sched, assign inputs)
        inputs_dict = {A: [[2.], [4.]]}

        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched)

        assert np.allclose(A.input_ports[0].parameters.value.get(comp), [2.0])
        assert np.allclose(A.input_ports[1].parameters.value.get(comp), [4.0])
        assert np.allclose(A.parameters.variable.get(comp.default_execution_id), [[2.0], [4.0]])


class TestRunInputSpecifications:

    # def test_2_mechanisms_default_input_1(self):
    #     comp = Composition()
    #     A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
    #     B = TransferMechanism(function=Linear(slope=5.0))
    #     comp.add_node(A)
    #     comp.add_node(B)
    #     comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
    #     sched = Scheduler(composition=comp)
    #     output = comp.run(
    #         scheduler=sched
    #     )
    #     assert 25 == output[0][0]

    def test_input_not_provided_to_run(self):
        T = TransferMechanism(name='T',
                              default_variable=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        T2 = TransferMechanism(name='T2',
                               function=Linear(slope=2.0),
                               default_variable=[[0.0, 0.0]])
        C = Composition(pathways=[T, T2])
        run_result = C.run(inputs={})
        assert np.allclose(T.parameters.value.get(C), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert np.allclose(run_result, [[np.array([2.0, 4.0])]])

    input_args = [
        ('non_input_node',
         '"The following items specified in the \'inputs\' arg of the run() method for \'Composition-1\' '
         'are not INPUT Nodes of that Composition (nor InputPorts of them): \'OB\'."'
         ),
        ('non_input_port',
         '"The following items specified in the \'inputs\' arg of the run() method for \'Composition-1\' '
         'that are not a Mechanism, Composition, or an InputPort of one: \'OA[OutputPort-0]\'."'
         ),
        ('nested_non_input_node',
         '"The following items specified in the \'inputs\' arg of the run() method for \'Composition-1\' '
         'are not INPUT Nodes of that Composition (nor InputPorts of them): \'IB\'."'
         ),
        ('nested_non_input_port',
        '"The following items specified in the \'inputs\' arg of the run() method for \'Composition-1\' '
         'that are not a Mechanism, Composition, or an InputPort of one: \'IA[OutputPort-0]\'."'
         ),
        ('input_port_and_mech',
         '"The \'inputs\' arg of the run() method for \'Composition-1\' includes specifications of '
         'the following InputPorts *and* the Mechanisms to which they belong; '
         'only one or the other can be specified as inputs to run():  OA[InputPort-0]."'
         ),
        ('nested_input_port_and_comp',
         '"The \'inputs\' arg of the run() method for \'Composition-1\' includes specifications of '
         'the following InputPorts or Mechanisms *and* the Composition within which they are nested: '
         '[(\'IA[InputPort-0]\', \'Composition-0\')]."'
         ),
        ('nested_mech_and_comp',
         '"The \'inputs\' arg of the run() method for \'Composition-1\' includes specifications of '
         'the following InputPorts or Mechanisms *and* the Composition within which they are nested: '
         '[(\'IA\', \'Composition-0\')]."'
         ),
        ('run_nested_with_inputs', None)
    ]
    @pytest.mark.parametrize('input_args', input_args, ids=[x[0] for x in input_args])
    def test_inputs_key_errors(self, input_args):

        condition = input_args[0]
        expected_error_text = input_args[1]

        ia = ProcessingMechanism(name='IA')
        ib = ProcessingMechanism(name='IB')
        oa = ProcessingMechanism(name='OA')
        ob = ProcessingMechanism(name='OB')
        icomp = Composition([ia, ib])
        ocomp = Composition([oa, ob])
        ocomp.add_node(icomp)

        if condition in {'nested_non_input_node', 'nested_non_input_port'}:
            X = ia
            Y = ib
        else:
            X = oa
            Y = ob

        if 'non_input_node' in condition:
            inputs={X:[1], Y:[1]}
        elif 'non_input_port' in condition:
            inputs={X.output_port:[1]}
        elif condition == 'input_port_and_mech':
            inputs={X.input_port:[1], X:[1]}
        elif condition == 'nested_input_port_and_comp':
            inputs={ia.input_port:[1], icomp:[1]}
        elif condition == 'nested_mech_and_comp':
            inputs={ia:[1], icomp:[1]}
        elif condition == 'run_nested':
            inputs={ia:[1]}

        if expected_error_text:
            with pytest.raises(RunError) as error_text:
                ocomp.run(inputs=inputs)
            assert expected_error_text in str(error_text.value)
        else:
            ocomp._analyze_graph()
            icomp.run(inputs={ia:[1]})

    def test_some_inputs_not_specified(self):
        comp = Composition()

        A = TransferMechanism(name="composition-pytests-A",
                              default_variable=[[1.0, 2.0], [3.0, 4.0]],
                              function=Linear(slope=2.0))
        B = TransferMechanism(name="composition-pytests-B",
                              default_variable=[[0.0, 0.0, 0.0]],
                              function=Linear(slope=3.0))
        C = TransferMechanism(name="composition-pytests-C")
        D = TransferMechanism(name="composition-pytests-D")

        comp.add_node(A)
        comp.add_node(B)
        comp.add_node(C)
        comp.add_node(D)

        inputs = {B: [[1., 2., 3.]],
                  D: [[4.]]}
        sched = Scheduler(composition=comp)
        comp.run(inputs=inputs, scheduler=sched)[0]

        assert np.allclose(A.get_output_values(comp), [[2.0, 4.0], [6.0, 8.0]])
        assert np.allclose(B.get_output_values(comp), [[3., 6., 9.]])
        assert np.allclose(C.get_output_values(comp), [[0.]])
        assert np.allclose(D.get_output_values(comp), [[4.]])
        for i,j in zip(comp.results[0], [[2., 4.], [6., 8.], [3., 6., 9.],[0.], [4.]]):
            assert np.allclose(i,j)

    def test_some_inputs_not_specified_origin_node_is_composition(self):

        compA = Composition()
        A = TransferMechanism(name="composition-pytests-A",
                              default_variable=[[1.0, 2.0], [3.0, 4.0]],
                              function=Linear(slope=2.0))
        compA.add_node(A)

        comp = Composition()

        B = TransferMechanism(name="composition-pytests-B",
                              default_variable=[[0.0, 0.0, 0.0]],
                              function=Linear(slope=3.0))

        C = TransferMechanism(name="composition-pytests-C")

        D = TransferMechanism(name="composition-pytests-D")

        comp.add_node(compA)
        comp.add_node(B)
        comp.add_node(C)
        comp.add_node(D)


        inputs = {B: [[1., 2., 3.]],
                  D: [[4.]]}

        sched = Scheduler(composition=comp)
        comp.run(inputs=inputs, scheduler=sched)[0]

        assert np.allclose(A.get_output_values(comp), [[2.0, 4.0], [6.0, 8.0]])
        assert np.allclose(compA.get_output_values(comp), [[2.0, 4.0], [6.0, 8.0]])
        assert np.allclose(B.get_output_values(comp), [[3., 6., 9.]])
        assert np.allclose(C.get_output_values(comp), [[0.]])
        assert np.allclose(D.get_output_values(comp), [[4.]])

    def test_heterogeneous_variables_drop_outer_list(self):
        # from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
        A = TransferMechanism(name='a', default_variable=[[0.0], [0.0,0.0]])
        C = Composition(pathways=[A])
        output = C.run(inputs={A: [[1.0], [2.0, 2.0]]})
        for i,j in zip(output,[[1.0],[2.0,2.0]]):
            np.allclose(i,j)

    def test_heterogeneous_variables_two_trials(self):
        # from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
        A = TransferMechanism(name='a', default_variable=[[0.0], [0.0,0.0]])
        C = Composition(pathways=[A])
        C.run(inputs={A: [[[1.1], [2.1, 2.1]], [[1.2], [2.2, 2.2]]]})
        for i,j in zip(C.results,[[[1.1], [2.1, 2.1]], [[1.2], [2.2, 2.2]]]):
            for k,l in zip(i,j):
                np.allclose(k,l)

    def test_3_origins(self):
        comp = Composition()
        I1 = InputPort(
                        name="Input Port 1",
                        reference_value=[0]
        )
        I2 = InputPort(
                        name="Input Port 2",
                        reference_value=[0]
        )
        A = TransferMechanism(
                            name="composition-pytests-A",
                            default_variable=[[0], [0]],
                            input_ports=[I1, I2],
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
        comp.add_node(A)
        comp.add_node(B)
        comp.add_node(C)
        comp.add_node(D)
        comp.add_projection(MappingProjection(sender=A, receiver=D), A, D)
        comp.add_projection(MappingProjection(sender=B, receiver=D), B, D)
        comp.add_projection(MappingProjection(sender=C, receiver=D), C, D)
        inputs = {A: [[[0], [0]], [[1], [1]], [[2], [2]]],
                  B: [[0, 0], [1, 1], [2, 2]],
                  C: [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        }

        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs, scheduler=sched)

        assert np.allclose(np.array([[12.]]), output)

    def test_2_mechanisms_input_5(self):
        comp = Composition()
        A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
        B = TransferMechanism(function=Linear(slope=5.0))
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        inputs_dict = {A: [[5]]}
        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched)
        assert np.allclose([125], output)

    def test_run_2_mechanisms_reuse_input(self):
        comp = Composition()
        A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
        B = TransferMechanism(function=Linear(slope=5.0))
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        inputs_dict = {A: [[5]]}
        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched, num_trials=5)
        assert np.allclose([125], output)

    def test_function_as_input(self):
        c = pnl.Composition()

        m1 = pnl.TransferMechanism()
        m2 = pnl.TransferMechanism()

        c.add_linear_processing_pathway([m1, m2])

        def test_function(trial_num):
            stimuli = list(range(10))
            return {
                m1: stimuli[trial_num]
            }

        c.run(inputs=test_function,
              num_trials=10)
        assert c.parameters.results.get(c) == [[np.array([0.])], [np.array([1.])], [np.array([2.])], [np.array([3.])],
                                               [np.array([4.])], [np.array([5.])], [np.array([6.])], [np.array([7.])],
                                               [np.array([8.])], [np.array([9.])]]

    @pytest.mark.parametrize("mode", [pnl.ExecutionMode.Python,
                                      pytest.param(pnl.ExecutionMode.LLVMRun, marks=pytest.mark.llvm),
                                      pytest.param(pnl.ExecutionMode.PTXRun, marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                     ])
    def test_generator_as_input(self, mode):
        c = pnl.Composition()

        m1 = pnl.TransferMechanism()
        m2 = pnl.TransferMechanism()

        c.add_linear_processing_pathway([m1, m2])

        def test_generator():
            for i in range(10):
                yield {
                    m1: i
                }

        t_g = test_generator()

        c.run(inputs=t_g, execution_mode=mode)
        assert c.parameters.results.get(c) == [[np.array([0.])], [np.array([1.])], [np.array([2.])], [np.array([3.])],
                                               [np.array([4.])], [np.array([5.])], [np.array([6.])], [np.array([7.])],
                                               [np.array([8.])], [np.array([9.])]]

    @pytest.mark.parametrize("mode", [pnl.ExecutionMode.Python,
                                      pytest.param(pnl.ExecutionMode.LLVMRun, marks=pytest.mark.llvm),
                                      pytest.param(pnl.ExecutionMode.PTXRun, marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                     ])
    def test_generator_as_input_with_num_trials(self, mode):
        c = pnl.Composition()

        m1 = pnl.TransferMechanism()
        m2 = pnl.TransferMechanism()

        c.add_linear_processing_pathway([m1, m2])

        def test_generator():
            for i in range(10):
                yield {
                    m1: i
                }

        t_g = test_generator()

        c.run(inputs=t_g, num_trials=1, execution_mode=mode)
        assert c.parameters.results.get(c) == [[np.array([0.])]]

    def test_error_on_malformed_generator(self):
        c = pnl.Composition()

        m1 = pnl.TransferMechanism()
        m2 = pnl.TransferMechanism()

        c.add_linear_processing_pathway([m1, m2])

        def test_generator():
            yield {
                m1: [[1],[2]]
            }

        t_g = test_generator()

        try:
            c.run(inputs=t_g)
        except Exception as e:
            assert isinstance(e, pnl.CompositionError)


class TestRun:

    # def test_run_2_mechanisms_default_input_1(self):
    #     comp = Composition()
    #     A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
    #     B = TransferMechanism(function=Linear(slope=5.0))
    #     comp.add_node(A)
    #     comp.add_node(B)
    #     comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
    #     sched = Scheduler(composition=comp)
    #     output = comp.run(
    #         scheduler=sched
    #     )
    #     assert 25 == output[0][0]

    @pytest.mark.projection
    @pytest.mark.composition
    def test_run_2_mechanisms_input_grow(self, comp_mode):
        comp = Composition()
        A = IntegratorMechanism(default_variable=[1.0, 2.0], function=Linear(slope=5.0))
        B = TransferMechanism(default_variable=[1.0, 2.0, 3.0], function=Linear(slope=5.0))
        P = MappingProjection(sender=A, receiver=B)
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(P, A, B)
        inputs_dict = {A: [5, 4]}
        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched, execution_mode=comp_mode)
        assert np.allclose(output, [[225, 225, 225]])

    @pytest.mark.projection
    @pytest.mark.composition
    def test_run_2_mechanisms_input_shrink(self, comp_mode):
        comp = Composition()
        A = IntegratorMechanism(default_variable=[1.0, 2.0, 3.0], function=Linear(slope=5.0))
        B = TransferMechanism(default_variable=[4.0, 5.0], function=Linear(slope=5.0))
        P = MappingProjection(sender=A, receiver=B)
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(P, A, B)
        inputs_dict = {A: [5, 4, 3]}
        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched, execution_mode=comp_mode
        )
        assert np.allclose(output, [[300, 300]])

    @pytest.mark.composition
    def test_run_2_mechanisms_input_5(self, comp_mode):
        comp = Composition()
        A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
        B = TransferMechanism(function=Linear(slope=5.0))
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        inputs_dict = {A: [5]}
        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched, execution_mode=comp_mode)
        assert np.allclose(125, output[0][0])

    def test_projection_assignment_mistake_swap(self):

        comp = Composition()
        A = TransferMechanism(name="composition-pytests-A", function=Linear(slope=1.0))
        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=1.0))
        C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=5.0))
        D = TransferMechanism(name="composition-pytests-D", function=Linear(slope=5.0))
        comp.add_node(A)
        comp.add_node(B)
        comp.add_node(C)
        comp.add_node(D)
        comp.add_projection(MappingProjection(sender=A, receiver=C), A, C)
        with pytest.raises(CompositionError) as error_text:
            comp.add_projection(MappingProjection(sender=B, receiver=D), B, C)
        assert '"Receiver (\'composition-pytests-C\') assigned to ' \
               '\'MappingProjection from composition-pytests-B[RESULT] to composition-pytests-D[InputPort-0] ' \
               'is incompatible with the positions of these Components in \'Composition-0\'."' == str(error_text.value)

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
        comp.add_node(A)
        comp.add_node(B)
        comp.add_node(C)
        comp.add_node(D)
        comp.add_projection(MappingProjection(sender=A, receiver=C), A, C)
        with pytest.raises(CompositionError) as error_text:
            comp.add_projection(MappingProjection(sender=B, receiver=C), B, D)
        assert '"Receiver (\'composition-pytests-D\') assigned to ' \
               '\'MappingProjection from composition-pytests-B[RESULT] to composition-pytests-C[InputPort-0] ' \
               'is incompatible with the positions of these Components in \'Composition-0\'."' == str(error_text.value)

    @pytest.mark.composition
    def test_run_5_mechanisms_2_origins_1_terminal(self, comp_mode):
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
        comp.add_node(A)
        comp.add_node(B)
        comp.add_node(C)
        comp.add_node(D)
        comp.add_projection(MappingProjection(sender=A, receiver=C), A, C)
        comp.add_projection(MappingProjection(sender=B, receiver=D), B, D)
        comp.add_node(E)
        comp.add_projection(MappingProjection(sender=C, receiver=E), C, E)
        comp.add_projection(MappingProjection(sender=D, receiver=E), D, E)
        inputs_dict = {A: [5],
                       B: [5]}
        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched, execution_mode=comp_mode)

        assert np.allclose([250], output)

    @pytest.mark.composition
    @pytest.mark.parametrize("mode", [pnl.ExecutionMode.Python]) # LLVM doesn't support EveryNCalls fr N > 1
    def test_run_2_mechanisms_with_scheduling_AAB_integrator(self, mode):
        comp = Composition()

        A = IntegratorMechanism(name="A [integrator]", default_variable=2.0, function=SimpleIntegrator(rate=1.0))
        # (1) value = 0 + (5.0 * 1.0) + 0  --> return 5.0
        # (2) value = 5.0 + (5.0 * 1.0) + 0  --> return 10.0
        B = TransferMechanism(name="B [transfer]", function=Linear(slope=5.0))
        # value = 10.0 * 5.0 --> return 50.0
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        inputs_dict = {A: [5]}
        sched = Scheduler(composition=comp)
        sched.add_condition(B, EveryNCalls(A, 2))
        output = comp.run(inputs=inputs_dict, scheduler=sched, execution_mode=mode)

        assert np.allclose(50.0, output[0][0])

    @pytest.mark.composition
    def test_run_2_mechanisms_with_scheduling_AAB_transfer(self, comp_mode):
        comp = Composition()

        A = TransferMechanism(name="A [transfer]", function=Linear(slope=2.0))
        # (1) value = 5.0 * 2.0  --> return 10.0
        # (2) value = 5.0 * 2.0  --> return 10.0
        # ** TransferMechanism runs with the SAME input **
        B = TransferMechanism(name="B [transfer]", function=Linear(slope=5.0))
        # value = 10.0 * 5.0 --> return 50.0
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        inputs_dict = {A: [5]}
        sched = Scheduler(composition=comp)
        sched.add_condition(B, EveryNCalls(A, 2))
        output = comp.run(inputs=inputs_dict, scheduler=sched, execution_mode=comp_mode)
        assert np.allclose(50.0, output[0][0])

    @pytest.mark.composition
    def test_run_2_mechanisms_with_multiple_trials_of_input_values(self, comp_mode):
        comp = Composition()

        A = TransferMechanism(name="A [transfer]", function=Linear(slope=2.0))
        B = TransferMechanism(name="B [transfer]", function=Linear(slope=5.0))
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        inputs_dict = {A: [1, 2, 3, 4]}
        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched, execution_mode=comp_mode)

        assert np.allclose([[[40.0]]], output)

    @pytest.mark.composition
    def test_sender_receiver_not_specified(self, comp_mode):
        comp = Composition()

        A = TransferMechanism(name="A [transfer]", function=Linear(slope=2.0))
        B = TransferMechanism(name="B [transfer]", function=Linear(slope=5.0))
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(), A, B)
        inputs_dict = {A: [1, 2, 3, 4]}
        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched, execution_mode=comp_mode)

        assert np.allclose([[40.0]], output)

    @pytest.mark.composition
    def test_run_2_mechanisms_reuse_input(self, comp_mode):
        comp = Composition()
        A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
        B = TransferMechanism(function=Linear(slope=5.0))
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        inputs_dict = {A: [5]}
        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched, num_trials=5, execution_mode=comp_mode)
        assert np.allclose([125], output)

    @pytest.mark.composition
    def test_run_2_mechanisms_double_trial_specs(self, comp_mode):
        comp = Composition()
        A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
        B = TransferMechanism(function=Linear(slope=5.0))
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        inputs_dict = {A: [[5], [4], [3]]}
        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched, num_trials=3, execution_mode=comp_mode)

        assert np.allclose(np.array([[75.]]), output)

    @pytest.mark.composition
    @pytest.mark.parametrize("mode", [pnl.ExecutionMode.Python,
                                      pytest.param(pnl.ExecutionMode.LLVM, marks=pytest.mark.llvm),
                                      pytest.param(pnl.ExecutionMode.LLVMExec, marks=pytest.mark.llvm),
                                      pytest.param(pnl.ExecutionMode.PTXExec, marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                     ])
    def test_execute_composition(self, mode):
        comp = Composition()
        A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
        B = TransferMechanism(function=Linear(slope=5.0))
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        comp._analyze_graph()
        inputs_dict = {A: 3}
        sched = Scheduler(composition=comp)
        output = comp.execute(inputs=inputs_dict, scheduler=sched, execution_mode=mode)
        assert np.allclose([75], output)

    @pytest.mark.composition
    @pytest.mark.benchmark(group="LPP")
    def test_LPP(self, benchmark, comp_mode):

        comp = Composition()
        A = TransferMechanism(name="composition-pytests-A", function=Linear(slope=2.0, intercept=1.0))   # 1 x 2 + 1 = 3
        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=2.0, intercept=2.0))   # 3 x 2 + 2 = 8
        C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=2.0, intercept=3.0))   # 8 x 2 + 3 = 19
        D = TransferMechanism(name="composition-pytests-D", function=Linear(slope=2.0, intercept=4.0))   # 19 x 2 + 4 = 42
        E = TransferMechanism(name="composition-pytests-E", function=Linear(slope=2.0, intercept=5.0))   # 42 x 2 + 5 = 89
        comp.add_linear_processing_pathway([A, B, C, D, E])
        comp._analyze_graph()
        inputs_dict = {A: [[1]]}
        sched = Scheduler(composition=comp)
        output = benchmark(comp.run, inputs=inputs_dict, scheduler=sched, execution_mode=comp_mode)
        assert np.allclose(89., output)

    @pytest.mark.composition
    def test_LPP_with_projections(self, comp_mode):
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
        output = comp.run(inputs=inputs_dict, scheduler=sched, execution_mode=comp_mode)
        assert np.allclose(32., output)

    def test_LPP_end_with_projection(self):
        comp = Composition()
        A = TransferMechanism(name="composition-pytests-A", function=Linear(slope=2.0))
        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=2.0))
        C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=2.0))
        D = TransferMechanism(name="composition-pytests-D", function=Linear(slope=2.0))
        E = TransferMechanism(name="composition-pytests-E", function=Linear(slope=2.0))
        A_to_B = MappingProjection(sender=A, receiver=B)
        C_to_E = MappingProjection(sender=C, receiver=E)
        with pytest.raises(CompositionError) as error_text:
            comp.add_linear_processing_pathway([A, A_to_B, B, C, D, E, C_to_E])

        assert ("The last item in the \'pathway\' arg for add_linear_procesing_pathway method" in str(error_text.value)
                and "cannot be a Projection:" in str(error_text.value))

    def test_LPP_two_projections_in_a_row(self):
        comp = Composition()
        A = TransferMechanism(name="composition-pytests-A", function=Linear(slope=2.0))
        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=2.0))
        C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=2.0))
        A_to_B = MappingProjection(sender=A, receiver=B)
        B_to_C = MappingProjection(sender=B, receiver=C)
        with pytest.raises(CompositionError) as error_text:
            comp.add_linear_processing_pathway([A, B_to_C, A_to_B, B, C])
        assert ("A Projection specified in \'pathway\' arg for add_linear_procesing_pathway" in str(error_text.value)
                and "is not between two Nodes:" in str(error_text.value))

    def test_LPP_start_with_projection(self):
        comp = Composition()
        Nonsense_Projection = MappingProjection()
        A = TransferMechanism(name="composition-pytests-A", function=Linear(slope=2.0))
        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=2.0))
        with pytest.raises(CompositionError) as error_text:
            comp.add_linear_processing_pathway([Nonsense_Projection, A, B])
        assert ("First item in 'pathway' arg for add_linear_procesing_pathway method" in str(error_text.value)
                and "must be a Node (Mechanism or Composition)" in str(error_text.value))

    def test_LPP_wrong_component(self):
        from psyneulink.core.components.ports.inputport import InputPort
        comp = Composition()
        Nonsense = InputPort() # Note:  ports are OK in general, but this one was unassigned
        A = TransferMechanism(name="composition-pytests-A", function=Linear(slope=2.0))
        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=2.0))
        with pytest.raises(CompositionError) as error_text:
            comp.add_linear_processing_pathway([A, Nonsense, B])
        assert ("Bad Projection specification in \'pathway\' arg " in str(error_text.value)
                and "for add_linear_procesing_pathway method" in str(error_text.value)
                and "Attempt to assign Projection" in str(error_text.value)
                and "using InputPort" in str(error_text.value)
                and "that is in deferred init" in str(error_text.value))

    @pytest.mark.composition
    @pytest.mark.parametrize("mode", [pnl.ExecutionMode.Python,
                                      pytest.param(pnl.ExecutionMode.LLVM, marks=pytest.mark.llvm),
                                      pytest.param(pnl.ExecutionMode.LLVMExec, marks=pytest.mark.llvm),
                                      pytest.param(pnl.ExecutionMode.PTXExec, marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                     ])
    def test_execute_no_inputs(self, mode):
        m_inner = ProcessingMechanism(size=2)
        inner_comp = Composition(pathways=[m_inner])
        m_outer = ProcessingMechanism(size=2)
        outer_comp = Composition(pathways=[m_outer, inner_comp])
        result = outer_comp.run(execution_mode=mode)
        assert np.allclose(result, [[0.0],[0.0]])

    @pytest.mark.composition
    def test_run_no_inputs(self, comp_mode):
        m_inner = ProcessingMechanism(size=2)
        inner_comp = Composition(pathways=[m_inner])
        m_outer = ProcessingMechanism(size=2)
        outer_comp = Composition(pathways=[m_outer, inner_comp])
        result = outer_comp.run(execution_mode=comp_mode)
        assert np.allclose(result, [[0.0],[0.0]])

    def test_lpp_invalid_matrix_keyword(self):
        comp = Composition()
        A = TransferMechanism(name="composition-pytests-A", function=Linear(slope=2.0))
        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=2.0))
        with pytest.raises(CompositionError) as error_text:
        # Typo in IdentityMatrix
            comp.add_linear_processing_pathway([A, "IdntityMatrix", B])
        assert (f"The following entries in a pathway specified for 'Composition-0' are not a Node "
                f"(Mechanism or Composition) or a Projection nor a set of either: 'IdntityMatrix'"
                in str(error_text.value))

    @pytest.mark.composition
    def test_LPP_two_origins_one_terminal(self, comp_mode):
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
        inputs_dict = {A: [5],
                       B: [5]}
        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched, execution_mode=comp_mode)
        assert np.allclose([250], output)

    @pytest.mark.composition
    @pytest.mark.benchmark(group="LinearComposition")
    def test_run_composition(self, benchmark, comp_mode):
        comp = Composition()
        A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
        B = TransferMechanism(function=Linear(slope=5.0))
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        sched = Scheduler(composition=comp)
        output = benchmark(comp.run, inputs={A: [[1.0]]}, scheduler=sched, execution_mode=comp_mode)
        assert np.allclose(25, output)


    @pytest.mark.skip
    @pytest.mark.composition
    @pytest.mark.benchmark(group="LinearComposition")
    def test_run_composition_default(self, benchmark, comp_mode):
        comp = Composition()
        A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
        B = TransferMechanism(function=Linear(slope=5.0))
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        sched = Scheduler(composition=comp)
        output = benchmark(comp.run, scheduler=sched, execution_mode=comp_mode)
        assert 25 == output[0][0]

    @pytest.mark.composition
    @pytest.mark.benchmark(group="LinearComposition Vector")
    @pytest.mark.parametrize("vector_length", [2**x for x in range(1)])
    def test_run_composition_vector(self, benchmark, comp_mode, vector_length):
        var = [1.0 for x in range(vector_length)]
        comp = Composition()
        A = IntegratorMechanism(default_variable=var, function=Linear(slope=5.0))
        B = TransferMechanism(default_variable=var, function=Linear(slope=5.0))
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        sched = Scheduler(composition=comp)
        output = benchmark(comp.run, inputs={A: [var]}, scheduler=sched, execution_mode=comp_mode)
        assert np.allclose([25.0 for x in range(vector_length)], output[0])

    @pytest.mark.composition
    @pytest.mark.benchmark(group="Merge composition scalar")
    def test_3_mechanisms_2_origins_1_terminal(self, benchmark, comp_mode):
        # C --
        #              ==> E
        # D --

        # 5 x 5 = 25 --
        #                25 + 25 = 50  ==> 50 * 5 = 250
        # 5 x 5 = 25 --

        comp = Composition()
        C = TransferMechanism(name="C", function=Linear(slope=5.0))
        D = TransferMechanism(name="D", function=Linear(slope=5.0))
        E = TransferMechanism(name="E", function=Linear(slope=5.0))
        comp.add_node(C)
        comp.add_node(D)
        comp.add_node(E)
        comp.add_projection(MappingProjection(sender=C, receiver=E), C, E)
        comp.add_projection(MappingProjection(sender=D, receiver=E), D, E)
        inputs_dict = {C: [5.0],
                       D: [5.0]}
        sched = Scheduler(composition=comp)
        output = benchmark(comp.run, inputs=inputs_dict, scheduler=sched, execution_mode=comp_mode)
        assert np.allclose(250, output)

    @pytest.mark.composition
    @pytest.mark.benchmark(group="Merge composition scalar")
    def test_3_mechanisms_1_origin_2_terminals(self, benchmark, comp_mode):
        #       ==> D
        # C
        #       ==> E

        #                25 * 4 = 100
        # 5 x 5 = 25 --
        #                25 * 6 = 150

        comp = Composition()
        C = TransferMechanism(name="C", function=Linear(slope=5.0))
        D = TransferMechanism(name="D", function=Linear(slope=4.0))
        E = TransferMechanism(name="E", function=Linear(slope=6.0))
        comp.add_node(C)
        comp.add_node(D)
        comp.add_node(E)
        comp.add_projection(MappingProjection(sender=C, receiver=D), C, D)
        comp.add_projection(MappingProjection(sender=C, receiver=E), C, E)
        inputs_dict = {C: [5.0]}
        sched = Scheduler(composition=comp)
        output = benchmark(comp.run, inputs=inputs_dict, scheduler=sched, execution_mode=comp_mode)
        assert np.allclose([[100], [150]], output)

    @pytest.mark.composition
    @pytest.mark.benchmark(group="Merge composition scalar MIMO")
    def test_3_mechanisms_2_origins_1_terminal_mimo_last(self, benchmark, comp_mode):
        # C --
        #              ==> E
        # D --

        # [6] x 5 = [30] --
        #                            [30, 40] * 5 = [150, 200]
        # [8] x 5 = [40] --

        comp = Composition()
        C = TransferMechanism(name="C", function=Linear(slope=5.0))
        D = TransferMechanism(name="D", function=Linear(slope=5.0))
        E = TransferMechanism(name="E", input_ports=['a', 'b'], function=Linear(slope=5.0))
        comp.add_node(C)
        comp.add_node(D)
        comp.add_node(E)
        comp.add_projection(MappingProjection(sender=C, receiver=E.input_ports['a']), C, E)
        comp.add_projection(MappingProjection(sender=D, receiver=E.input_ports['b']), D, E)
        inputs_dict = {C: [6.0],
                       D: [8.0]}
        sched = Scheduler(composition=comp)
        output = benchmark(comp.run, inputs=inputs_dict, scheduler=sched, execution_mode=comp_mode)
        assert np.allclose([[150], [200]], output)


    @pytest.mark.composition
    @pytest.mark.benchmark(group="Merge composition scalar MIMO")
    def test_3_mechanisms_2_origins_1_terminal_mimo_parallel(self, benchmark, comp_mode):
        # C --
        #              ==> E
        # D --

        # [5, 6] x 5 = [25, 30] --
        #                            [25 + 35, 30 + 40] = [60, 70]  ==> [60, 70] * 5 = [300, 350]
        # [7, 8] x 5 = [35, 40] --

        comp = Composition()
        C = TransferMechanism(name="C", input_ports=['a', 'b'], function=Linear(slope=5.0))
        D = TransferMechanism(name="D", input_ports=['a', 'b'], function=Linear(slope=5.0))
        E = TransferMechanism(name="E", input_ports=['a', 'b'], function=Linear(slope=5.0))
        comp.add_node(C)
        comp.add_node(D)
        comp.add_node(E)
        comp.add_projection(MappingProjection(sender=C.output_ports[0], receiver=E.input_ports['a']), C, E)
        comp.add_projection(MappingProjection(sender=C.output_ports[1], receiver=E.input_ports['b']), C, E)
        comp.add_projection(MappingProjection(sender=D.output_ports[0], receiver=E.input_ports['a']), D, E)
        comp.add_projection(MappingProjection(sender=D.output_ports[1], receiver=E.input_ports['b']), D, E)
        inputs_dict = {C: [[5.0], [6.0]],
                       D: [[7.0], [8.0]]}
        sched = Scheduler(composition=comp)
        output = benchmark(comp.run, inputs=inputs_dict, scheduler=sched, execution_mode=comp_mode)
        assert np.allclose([[300], [350]], output)


    @pytest.mark.composition
    @pytest.mark.benchmark(group="Merge composition scalar MIMO")
    def test_3_mechanisms_2_origins_1_terminal_mimo_all_sum(self, benchmark, comp_mode):
        # C --
        #              ==> E
        # D --

        # [5, 6] x 5 = [25, 30] --
        #                            [25 + 35 + 30 + 40] = 130  ==> 130 * 5 = 650
        # [7, 8] x 5 = [35, 40] --

        comp = Composition()
        C = TransferMechanism(name="C", input_ports=['a', 'b'], function=Linear(slope=5.0))
        D = TransferMechanism(name="D", input_ports=['a', 'b'], function=Linear(slope=5.0))
        E = TransferMechanism(name="E", function=Linear(slope=5.0))
        comp.add_node(C)
        comp.add_node(D)
        comp.add_node(E)
        comp.add_projection(MappingProjection(sender=C.output_ports[0], receiver=E), C, E)
        comp.add_projection(MappingProjection(sender=C.output_ports[1], receiver=E), C, E)
        comp.add_projection(MappingProjection(sender=D.output_ports[0], receiver=E), D, E)
        comp.add_projection(MappingProjection(sender=D.output_ports[1], receiver=E), D, E)
        inputs_dict = {C: [[5.0], [6.0]],
                       D: [[7.0], [8.0]]}
        sched = Scheduler(composition=comp)
        output = benchmark(comp.run, inputs=inputs_dict, scheduler=sched, execution_mode=comp_mode)
        assert np.allclose([[650]], output)

    @pytest.mark.composition
    @pytest.mark.benchmark(group="Recurrent")
    def test_run_recurrent_transfer_mechanism(self, benchmark, comp_mode):
        comp = Composition()
        A = RecurrentTransferMechanism(size=3, function=Linear(slope=5.0), name="A")
        comp.add_node(A)
        sched = Scheduler(composition=comp)
        output1 = comp.run(inputs={A: [[1.0, 2.0, 3.0]]}, scheduler=sched, execution_mode=comp_mode)
        assert np.allclose([5.0, 10.0, 15.0], output1)
        output2 = comp.run(inputs={A: [[1.0, 2.0, 3.0]]}, scheduler=sched, execution_mode=comp_mode)
        # Using the hollow matrix: (10 + 15 + 1) * 5 = 130,
        #                          ( 5 + 15 + 2) * 5 = 110,
        #                          ( 5 + 10 + 3) * 5 = 90
        assert np.allclose([130.0, 110.0, 90.0], output2)
        if benchmark.enabled:
            benchmark(comp.run, inputs={A: [[1.0, 2.0, 3.0]]}, scheduler=sched, execution_mode=comp_mode)

    @pytest.mark.composition
    @pytest.mark.benchmark(group="Recurrent")
    @pytest.mark.parametrize("mode", [pnl.ExecutionMode.Python,
                                      pytest.param(pnl.ExecutionMode.LLVM, marks=pytest.mark.llvm),
                                      pytest.param(pnl.ExecutionMode.LLVMExec, marks=pytest.mark.llvm),
                                      pytest.param(pnl.ExecutionMode.PTXExec, marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                     ])
    def test_run_recurrent_transfer_mechanism_hetero(self, benchmark, mode):
        comp = Composition()
        R = RecurrentTransferMechanism(size=1,
                                       function=Logistic(),
                                       hetero=-2.0,
                                       output_ports = [RESULT])
        comp.add_node(R)
        comp._analyze_graph()
        sched = Scheduler(composition=comp)
        val = comp.execute(inputs={R: [[3.0]]}, execution_mode=mode)
        assert np.allclose(val, [[0.95257413]])
        val = comp.execute(inputs={R: [[4.0]]}, execution_mode=mode)
        assert np.allclose(val, [[0.98201379]])

        # execute 10 times
        for i in range(10):
            val = comp.execute(inputs={R: [[5.0]]}, execution_mode=mode)

        assert np.allclose(val, [[0.99330715]])

        if benchmark.enabled:
            benchmark(comp.execute, inputs={R: [[1.0]]}, execution_mode=mode)

    @pytest.mark.composition
    @pytest.mark.benchmark(group="Recurrent")
    @pytest.mark.parametrize("mode", [pnl.ExecutionMode.Python,
                                      pytest.param(pnl.ExecutionMode.LLVM, marks=pytest.mark.llvm),
                                      pytest.param(pnl.ExecutionMode.LLVMExec, marks=pytest.mark.llvm),
                                      pytest.param(pnl.ExecutionMode.PTXExec, marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                     ])
    def test_run_recurrent_transfer_mechanism_integrator(self, benchmark, mode):
        comp = Composition()
        R = RecurrentTransferMechanism(size=1,
                                       function=Logistic(),
                                       hetero=-2.0,
                                       integrator_mode=True,
                                       integration_rate=0.01,
                                       output_ports = [RESULT])
        comp.add_node(R)
        comp._analyze_graph()
        sched = Scheduler(composition=comp)
        val = comp.execute(inputs={R: [[3.0]]}, execution_mode=mode)
        assert np.allclose(val, [[0.50749944]])
        val = comp.execute(inputs={R: [[4.0]]}, execution_mode=mode)
        assert np.allclose(val, [[0.51741795]])

        # execute 10 times
        for i in range(10):
            val = comp.execute(inputs={R: [[5.0]]}, execution_mode=mode)

        assert np.allclose(val, [[0.6320741]])

        if benchmark.enabled:
            benchmark(comp.execute, inputs={R: [[1.0]]}, execution_mode=mode)

    @pytest.mark.composition
    @pytest.mark.benchmark(group="Recurrent")
    @pytest.mark.parametrize("mode", [pnl.ExecutionMode.Python,
                                      pytest.param(pnl.ExecutionMode.LLVM, marks=pytest.mark.llvm),
                                      pytest.param(pnl.ExecutionMode.LLVMExec, marks=pytest.mark.llvm),
                                      pytest.param(pnl.ExecutionMode.PTXExec, marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                     ])
    def test_run_recurrent_transfer_mechanism_vector_2(self, benchmark, mode):
        comp = Composition()
        R = RecurrentTransferMechanism(size=2, function=Logistic())
        comp.add_node(R)
        comp._analyze_graph()
        sched = Scheduler(composition=comp)
        val = comp.execute(inputs={R: [[1.0, 2.0]]}, execution_mode=mode)
        assert np.allclose(val, [[0.81757448, 0.92414182]])
        val = comp.execute(inputs={R: [[1.0, 2.0]]}, execution_mode=mode)
        assert np.allclose(val, [[0.87259959,  0.94361816]])

        # execute 10 times
        for i in range(10):
            val = comp.execute(inputs={R: [[1.0, 2.0]]}, execution_mode=mode)

        assert np.allclose(val, [[0.87507549,  0.94660049]])

        if benchmark.enabled:
            benchmark(comp.execute, inputs={R: [[1.0, 2.0]]}, execution_mode=mode)

    @pytest.mark.composition
    @pytest.mark.benchmark(group="Recurrent")
    @pytest.mark.parametrize("mode", [pnl.ExecutionMode.Python,
                                      pytest.param(pnl.ExecutionMode.LLVM, marks=pytest.mark.llvm),
                                      pytest.param(pnl.ExecutionMode.LLVMExec, marks=pytest.mark.llvm),
                                      pytest.param(pnl.ExecutionMode.PTXExec, marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                     ])
    def test_run_recurrent_transfer_mechanism_hetero_2(self, benchmark, mode):
        comp = Composition()
        R = RecurrentTransferMechanism(size=2,
                                       function=Logistic(),
                                       hetero=-2.0,
                                       output_ports = [RESULT])
        comp.add_node(R)
        comp._analyze_graph()
        sched = Scheduler(composition=comp)
        val = comp.execute(inputs={R: [[1.0, 2.0]]}, execution_mode=mode)
        assert np.allclose(val, [[0.5, 0.73105858]])
        val = comp.execute(inputs={R: [[1.0, 2.0]]}, execution_mode=mode)
        assert np.allclose(val, [[0.3864837, 0.73105858]])

        # execute 10 times
        for i in range(10):
            val = comp.execute(inputs={R: [[1.0, 2.0]]}, execution_mode=mode)

        assert np.allclose(val, [[0.36286875, 0.78146724]])

        if benchmark.enabled:
            benchmark(comp.execute, inputs={R: [[1.0, 2.0]]}, execution_mode=mode)

    @pytest.mark.composition
    @pytest.mark.benchmark(group="Recurrent")
    @pytest.mark.parametrize("mode", [pnl.ExecutionMode.Python,
                                      pytest.param(pnl.ExecutionMode.LLVM, marks=pytest.mark.llvm),
                                      pytest.param(pnl.ExecutionMode.LLVMExec, marks=pytest.mark.llvm),
                                      pytest.param(pnl.ExecutionMode.PTXExec, marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                     ])
    def test_run_recurrent_transfer_mechanism_integrator_2(self, benchmark, mode):
        comp = Composition()
        R = RecurrentTransferMechanism(size=2,
                                       function=Logistic(),
                                       hetero=-2.0,
                                       integrator_mode=True,
                                       integration_rate=0.01,
                                       output_ports = [RESULT])
        comp.add_node(R)
        comp._analyze_graph()
        sched = Scheduler(composition=comp)
        val = comp.execute(inputs={R: [[1.0, 2.0]]}, execution_mode=mode)
        assert np.allclose(val, [[0.5, 0.50249998]])
        val = comp.execute(inputs={R: [[1.0, 2.0]]}, execution_mode=mode)
        assert np.allclose(val, [[0.4999875, 0.50497484]])

        # execute 10 times
        for i in range(10):
            val = comp.execute(inputs={R: [[1.0, 2.0]]}, execution_mode=mode)

        assert np.allclose(val, [[0.49922843, 0.52838607]])

        if benchmark.enabled:
            benchmark(comp.execute, inputs={R: [[1.0, 2.0]]}, execution_mode=mode)

    def test_run_termination_condition_custom_context(self):
        D = pnl.DDM(function=pnl.DriftDiffusionIntegrator, execute_until_finished=False)
        comp = pnl.Composition()

        comp.add_node(node=D)

        comp.run(
            inputs={D: 0},
            termination_processing={pnl.TimeScale.RUN: pnl.WhenFinished(D)},
            context='custom'
        )

    def test_manual_context(self):
        t = pnl.TransferMechanism()
        comp = pnl.Composition()

        comp.add_node(t)

        comp.run({t: [1]})
        assert comp.results == [[[1]]]

        context = pnl.Context()
        t.function.parameters.slope._set(2, context)

        comp.run({t: [1]}, context=context)
        assert comp.results == [[[2]]]


class TestCallBeforeAfterTimescale:

    def test_call_before_record_timescale(self):

        comp = Composition()

        A = TransferMechanism(name="A [transfer]", function=Linear(slope=2.0))
        B = TransferMechanism(name="B [transfer]", function=Linear(slope=5.0))
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        inputs_dict = {A: [1, 2, 3, 4]}
        sched = Scheduler(composition=comp)

        time_step_array = []
        trial_array = []
        pass_array = []

        def cb_timestep(scheduler, arr):

            def record_timestep():

                arr.append(scheduler.clocks[comp.default_execution_id].get_total_times_relative(TimeScale.TIME_STEP, TimeScale.TRIAL))

            return record_timestep

        def cb_pass(scheduler, arr):

            def record_pass():

                arr.append(scheduler.clocks[comp.default_execution_id].get_total_times_relative(TimeScale.PASS, TimeScale.RUN))

            return record_pass

        def cb_trial(scheduler, arr):

            def record_trial():

                arr.append(scheduler.clocks[comp.default_execution_id].get_total_times_relative(TimeScale.TRIAL, TimeScale.LIFE))

            return record_trial

        comp.run(inputs=inputs_dict, scheduler=sched,
                 call_after_time_step=cb_timestep(sched, time_step_array), call_before_pass=cb_pass(sched, pass_array),
                 call_before_trial=cb_trial(sched, trial_array))
        assert time_step_array == [0, 1, 0, 1, 0, 1, 0, 1]
        assert trial_array == [0, 1, 2, 3]
        assert pass_array == [0, 1, 2, 3]

    def test_call_beforeafter_values_onepass(self):
        comp = Composition()

        A = TransferMechanism(name="A [transfer]", function=Linear(slope=2.0))
        B = TransferMechanism(name="B [transfer]", function=Linear(slope=5.0))
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        inputs_dict = {A: [1, 2, 3, 4]}
        sched = Scheduler(composition=comp)

        before = {}
        after = {}

        before_expected = {
            TimeScale.TIME_STEP: {
                A: [0, 2, 2, 4, 4, 6, 6, 8],
                B: [0, 0, 10, 10, 20, 20, 30, 30]
            },
            TimeScale.PASS: {
                A: [0, 2, 4, 6],
                B: [0, 10, 20, 30]
            },
            TimeScale.TRIAL: {
                A: [0, 2, 4, 6],
                B: [0, 10, 20, 30]
            },
        }

        after_expected = {
            TimeScale.TIME_STEP: {
                A: [2, 2, 4, 4, 6, 6, 8, 8],
                B: [0, 10, 10, 20, 20, 30, 30, 40]
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
            scheduler=sched,
            call_before_time_step=functools.partial(record_values, before, TimeScale.TIME_STEP, A, B, comp=comp),
            call_after_time_step=functools.partial(record_values, after, TimeScale.TIME_STEP, A, B, comp=comp),
            call_before_pass=functools.partial(record_values, before, TimeScale.PASS, A, B, comp=comp),
            call_after_pass=functools.partial(record_values, after, TimeScale.PASS, A, B, comp=comp),
            call_before_trial=functools.partial(record_values, before, TimeScale.TRIAL, A, B, comp=comp),
            call_after_trial=functools.partial(record_values, after, TimeScale.TRIAL, A, B, comp=comp),
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
                    except (TypeError, IndexError):
                        comp.append(x)
                np.testing.assert_allclose(comp, after_expected[ts][mech], err_msg='Failed on after[{0}][{1}]'.format(ts, mech))

    def test_call_beforeafter_values_twopass(self):
        comp = Composition()

        A = IntegratorMechanism(name="A [transfer]", function=SimpleIntegrator(rate=1))
        B = IntegratorMechanism(name="B [transfer]", function=SimpleIntegrator(rate=2))
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        inputs_dict = {A: [1, 2]}
        sched = Scheduler(composition=comp)
        sched.add_condition(B, EveryNCalls(A, 2))

        before = {}
        after = {}

        before_expected = {
            TimeScale.TIME_STEP: {
                A: [
                    0, 1, 2,
                    2, 4, 6,
                ],
                B: [
                    0, 0, 0,
                    4, 4, 4,
                ]
            },
            TimeScale.PASS: {
                A: [
                    0, 1,
                    2, 4,
                ],
                B: [
                    0, 0,
                    4, 4,
                ]
            },
            TimeScale.TRIAL: {
                A: [0, 2],
                B: [0, 4]
            },
        }

        after_expected = {
            TimeScale.TIME_STEP: {
                A: [
                    1, 2, 2,
                    4, 6, 6,
                ],
                B: [
                    0, 0, 4,
                    4, 4, 16,
                ]
            },
            TimeScale.PASS: {
                A: [
                    1, 2,
                    4, 6,
                ],
                B: [
                    0, 4,
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
            scheduler=sched,
            call_before_time_step=functools.partial(record_values, before, TimeScale.TIME_STEP, A, B, comp=comp),
            call_after_time_step=functools.partial(record_values, after, TimeScale.TIME_STEP, A, B, comp=comp),
            call_before_pass=functools.partial(record_values, before, TimeScale.PASS, A, B, comp=comp),
            call_after_pass=functools.partial(record_values, after, TimeScale.PASS, A, B, comp=comp),
            call_before_trial=functools.partial(record_values, before, TimeScale.TRIAL, A, B, comp=comp),
            call_after_trial=functools.partial(record_values, after, TimeScale.TRIAL, A, B, comp=comp),
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
                    except (TypeError, IndexError):
                        comp.append(x)
                np.testing.assert_allclose(comp, after_expected[ts][mech], err_msg='Failed on after[{0}][{1}]'.format(ts, mech))


    # when self.sched is ready:
    # def test_run_default_scheduler(self):
    #     comp = Composition()
    #     A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
    #     B = TransferMechanism(function=Linear(slope=5.0))
    #     comp.add_node(A)
    #     comp.add_node(B)
    #     comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
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
    #     Output_Layerrecord_values = TransferMechanism(
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
    #     comp.add_node(Input_Layer)
    #     comp.add_node(Hidden_Layer_1)
    #     comp.add_node(Hidden_Layer_2)
    #     comp.add_node(Output_Layer)
    #
    #     comp.add_projection(Input_Layer, Input_Weights, Hidden_Layer_1)
    #     comp.add_projection(Hidden_Layer_1, MappingProjection(), Hidden_Layer_2)
    #     comp.add_projection(Hidden_Layer_2, MappingProjection(), Output_Layer)
    #
    #     stim_list = {Input_Layer: [[-1, 30]]}
    #     sched = Scheduler(composition=comp)
    #     output = comp.run(
    #         inputs=stim_list,
    #         scheduler=sched,
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
#         comp.add_node(A)
#         comp.add_node(B)
#         comp.add_node(C)
#         comp.add_node(D)
#         comp.add_projection(A, MappingProjection(sender=A, receiver=C), C)
#         comp.add_projection(B, MappingProjection(sender=B, receiver=D), D)
#         comp.add_node(E)
#         comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
#         comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
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
#             scheduler=sched,
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
#         comp.add_node(A)
#         comp.add_node(B)
#         comp.add_node(C)
#         comp.add_node(D)
#         comp.add_projection(A, MappingProjection(sender=A, receiver=C), C)
#         comp.add_projection(B, MappingProjection(sender=B, receiver=D), D)
#         comp.add_node(E)
#         comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
#         comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
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
#             scheduler=sched,
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
#         comp.add_node(A)
#         comp.add_node(B)
#         comp.add_node(C)
#         comp.add_node(D)
#         comp.add_projection(A, MappingProjection(sender=A, receiver=C), C)
#         comp.add_projection(B, MappingProjection(sender=B, receiver=D), D)
#         comp.add_node(E)
#         comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
#         comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
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
#             scheduler=sched,
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
#         comp.add_node(A)
#         comp.add_node(B)
#         comp.add_node(C)
#         comp.add_node(D)
#         comp.add_projection(A, MappingProjection(sender=A, receiver=C), C)
#         comp.add_projection(B, MappingProjection(sender=B, receiver=D), D)
#         comp.add_node(E)
#         comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
#         comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
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
#             scheduler=sched,
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
#         comp.add_node(A)
#         comp.add_node(B)
#         comp.add_node(C)
#         comp.add_node(D)
#         comp.add_projection(A, MappingProjection(sender=A, receiver=C), C)
#         comp.add_projection(B, MappingProjection(sender=B, receiver=D), D)
#         comp.add_node(E)
#         comp.add_projection(C, MappingProjection(sender=C, receiver=E), E)
#         comp.add_projection(D, MappingProjection(sender=D, receiver=E), E)
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
#             scheduler=sched,
#             clamp_input=NO_CLAMP
#         )
#         # FIX: This value is correct given that there is a BUG in Recurrent Transfer Mech --
#         # Recurrent projection BEGINS with a value leftover from initialization
#         # (only shows up if the function has an additive component or default variable is not zero)
#         assert 925 == output[0][0]


class TestNestedCompositions:

    @pytest.mark.composition
    def test_transfer_mechanism_composition(self, comp_mode):

        # mechanisms
        A = ProcessingMechanism(name="A",
                                function=AdaptiveIntegrator(rate=0.1))
        B = ProcessingMechanism(name="B",
                                function=Logistic)
        C = TransferMechanism(name="C",
                              function=Logistic,
                              integration_rate=0.1,
                              integrator_mode=True)

        # comp1 separates IntegratorFunction fn and Logistic fn into mech A and mech B
        comp1 = Composition(name="comp1")
        comp1.add_linear_processing_pathway([A, B])

        # comp2 uses a TransferMechanism in integrator comp_mode
        comp2 = Composition(name="comp2")
        comp2.add_node(C)

        # pass same 3 trials of input to comp1 and comp2
        comp1.run(inputs={A: [1.0, 2.0, 3.0]}, execution_mode=comp_mode)
        comp2.run(inputs={C: [1.0, 2.0, 3.0]}, execution_mode=comp_mode)

        assert np.allclose(comp1.results, comp2.results)
        assert np.allclose(comp2.results, [[[0.52497918747894]], [[0.5719961329315186]], [[0.6366838893983633]]])

    @pytest.mark.nested
    @pytest.mark.composition
    def test_nested_transfer_mechanism_composition(self, comp_mode):

        # mechanisms
        A = ProcessingMechanism(name="A",
                                function=AdaptiveIntegrator(rate=0.1))
        B = ProcessingMechanism(name="B",
                                function=Logistic)

        inner_comp = Composition(name="inner_comp")
        inner_comp.add_linear_processing_pathway([A, B])
        sched = Scheduler(composition=inner_comp)

        outer_comp = Composition(name="outer_comp")
        outer_comp.add_node(inner_comp)

        sched = Scheduler(composition=outer_comp)
        ret = outer_comp.run(inputs=[1.0], execution_mode=comp_mode)

        assert np.allclose(ret, [[[0.52497918747894]]])


    @pytest.mark.nested
    @pytest.mark.composition
    def test_nested_transfer_mechanism_composition_parallel(self, comp_mode):

        # mechanisms
        A = ProcessingMechanism(name="A",
                                function=AdaptiveIntegrator(rate=0.1))
        B = ProcessingMechanism(name="B",
                                function=Logistic)

        inner_comp1 = Composition(name="inner_comp1")
        inner_comp1.add_linear_processing_pathway([A, B])
        sched = Scheduler(composition=inner_comp1)

        C = TransferMechanism(name="C",
                              function=Logistic,
                              integration_rate=0.1,
                              integrator_mode=True)

        inner_comp2 = Composition(name="inner_comp2")
        inner_comp2.add_node(C)
        sched = Scheduler(composition=inner_comp2)

        outer_comp = Composition(name="outer_comp")
        outer_comp.add_node(inner_comp1)
        outer_comp.add_node(inner_comp2)

        sched = Scheduler(composition=outer_comp)
        ret = outer_comp.run(inputs={inner_comp1: [[1.0]], inner_comp2: [[1.0]]}, execution_mode=comp_mode)
        assert np.allclose(ret, [[[0.52497918747894]],[[0.52497918747894]]])

    @pytest.mark.nested
    @pytest.mark.composition
    def test_nested_run_differing_num_trials(self, comp_mode):
        # Test for case where nested composition is ran with inputs of differing but valid sizes
        outer = pnl.Composition(name="outer")

        inner = pnl.Composition(name="inner")
        inner_mech_A = pnl.TransferMechanism(name="inner_mech_A", default_variable=[[0,0]])
        inner_mech_B = pnl.TransferMechanism(name="inner_mech_B", default_variable=[[0,0]])
        inner.add_nodes([inner_mech_A, inner_mech_B])
        outer.add_node(inner)
        input = {
            inner : {
                inner_mech_A : [[[0,0]]],
                inner_mech_B : [[[0,0]],[[0,0]]],
            }
        }

        outer.run(inputs=input, execution_mode=comp_mode)

    @pytest.mark.control
    def test_invalid_projection_deletion_when_nesting_comps(self):
        oa = pnl.TransferMechanism(name='oa')
        ob = pnl.TransferMechanism(name='ob')
        ocomp = pnl.Composition(name='ocomp', controller_mode=pnl.BEFORE)
        ia = pnl.TransferMechanism(name='ia')
        ib = pnl.ProcessingMechanism(name='ib',
                                     function=lambda x: abs(x - 75))
        icomp = pnl.Composition(name='icomp', controller_mode=pnl.BEFORE)
        ocomp.add_node(oa, required_roles=pnl.NodeRole.INPUT)
        ocomp.add_node(ob)
        ocomp.add_node(icomp)
        icomp.add_node(ia, required_roles=pnl.NodeRole.INPUT)
        icomp.add_node(ib)
        ocomp._analyze_graph()
        icomp._analyze_graph()
        ocomp.add_projection(pnl.MappingProjection(), sender=oa, receiver=ia)
        icomp.add_projection(pnl.MappingProjection(), sender=ia, receiver=ib)
        ocomp.add_projection(pnl.MappingProjection(), sender=ib, receiver=ob)

        ocomp_objective_mechanism = pnl.ObjectiveMechanism(
                    monitor=ib.output_port,
                    function=pnl.SimpleIntegrator,
                    name="oController Objective Mechanism"
                )

        ocomp.add_controller(
            pnl.OptimizationControlMechanism(
                agent_rep=ocomp,
                state_features=[oa.input_port],
                # state_feature_function=pnl.Buffer(history=2),
                name="Controller",
                objective_mechanism=ocomp_objective_mechanism,
                function=pnl.GridSearch(direction=pnl.MINIMIZE),
                control_signals=[pnl.ControlSignal(projections=[(pnl.SLOPE, ia)],
                                                   variable=1.0,
                                                   intensity_cost_function=pnl.Linear(slope=0.0),
                                                   allocation_samples=pnl.SampleSpec(start=1.0, stop=5.0, num=5))])
        )

        icomp_objective_mechanism = pnl.ObjectiveMechanism(
                    monitor=ib.output_port,
                    function=pnl.SimpleIntegrator,
                    name="iController Objective Mechanism"
                )

        icomp.add_controller(
            pnl.OptimizationControlMechanism(
                agent_rep=icomp,
                state_features=[ia.input_port],
                # state_feature_function=pnl.Buffer(history=2),
                name="Controller",
                objective_mechanism=icomp_objective_mechanism,
                function=pnl.GridSearch(direction=pnl.MAXIMIZE),
                control_signals=[pnl.ControlSignal(projections=[(pnl.SLOPE, ia)],
                                                   variable=1.0,
                                                   intensity_cost_function=pnl.Linear(slope=0.0),
                                                   allocation_samples=pnl.SampleSpec(start=1.0, stop=5.0, num=5))])
        )
        assert not ocomp._check_for_existing_projections(sender=ib, receiver=ocomp_objective_mechanism)
        return ocomp
    # # Does not work yet due to initialize_cycle_values bug that causes first recurrent projection to pass different values
    # # to TranfserMechanism version vs Logistic fn + AdaptiveIntegrator fn version
    # def test_recurrent_transfer_mechanism_composition(self):
    #
    #     # mechanisms
    #     A = ProcessingMechanism(name="A",
    #                             function=AdaptiveIntegrator(rate=0.1))
    #     B = ProcessingMechanism(name="B",
    #                             function=Logistic)
    #     C = RecurrentTransferMechanism(name="C",
    #                                    function=Logistic,
    #                                    integration_rate=0.1,
    #                                    integrator_mode=True)
    #
    #     # comp1 separates IntegratorFunction fn and Logistic fn into mech A and mech B and uses a "feedback" proj for recurrence
    #     comp1 = Composition(name="comp1")
    #     comp1.add_linear_processing_pathway([A, B])
    #     comp1.add_linear_processing_pathway([B, A], feedback=True)
    #
    #     # comp2 uses a RecurrentTransferMechanism in integrator mode
    #     comp2 = Composition(name="comp2")
    #     comp2.add_node(C)
    #
    #     # pass same 3 trials of input to comp1 and comp2
    #     comp1.run(inputs={A: [1.0, 2.0, 3.0]})
    #     comp2.run(inputs={C: [1.0, 2.0, 3.0]})
    #
    #     assert np.allclose(comp1.results, comp2.results)

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

        tree1.add_node(myMech1)
        tree1.add_node(myMech2)
        tree1.add_node(myMech3)
        tree1.add_projection(MappingProjection(sender=myMech1, receiver=myMech3), myMech1, myMech3)
        tree1.add_projection(MappingProjection(sender=myMech2, receiver=myMech3), myMech2, myMech3)

        # validate first composition ---------------------------------------------

        tree1._analyze_graph()
        origins = tree1.get_nodes_by_role(NodeRole.ORIGIN)
        assert len(origins) == 2
        assert myMech1 in origins
        assert myMech2 in origins
        terminals = tree1.get_nodes_by_role(NodeRole.TERMINAL)
        assert len(terminals) == 1
        assert myMech3 in terminals

        # create second composition ----------------------------------------------

        # Mech4 --
        #           --> Mech6
        # Mech5 --

        tree2 = Composition()
        tree2.add_node(myMech4)
        tree2.add_node(myMech5)
        tree2.add_node(myMech6)
        tree2.add_projection(MappingProjection(sender=myMech4, receiver=myMech6), myMech4, myMech6)
        tree2.add_projection(MappingProjection(sender=myMech5, receiver=myMech6), myMech5, myMech6)

        # validate second composition ----------------------------------------------

        tree2._analyze_graph()
        origins = tree2.get_nodes_by_role(NodeRole.ORIGIN)
        assert len(origins) == 2
        assert myMech4 in origins
        assert myMech5 in origins
        terminals = tree2.get_nodes_by_role(NodeRole.TERMINAL)
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

        origins = tree1.get_nodes_by_role(NodeRole.ORIGIN)
        assert len(origins) == 4
        assert myMech1 in origins
        assert myMech2 in origins
        assert myMech4 in origins
        assert myMech5 in origins
        terminals = tree1.get_nodes_by_role(NodeRole.TERMINAL)
        assert len(terminals) == 2
        assert myMech3 in terminals
        assert myMech6 in terminals

        # AFTER linking via 3 --> 4 projection ------------------------------------
        # Mech1 --
        #          --> Mech3 ----> Mech4 --
        # Mech2 --                          --> Mech6
        #                          Mech5 --

        tree1.add_projection(MappingProjection(sender=myMech3, receiver=myMech4), myMech3, myMech4)
        tree1._analyze_graph()

        origins = tree1.get_nodes_by_role(NodeRole.ORIGIN)
        assert len(origins) == 3
        assert myMech1 in origins
        assert myMech2 in origins
        assert myMech5 in origins
        terminals = tree1.get_nodes_by_role(NodeRole.TERMINAL)
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

        tree1.add_node(myMech1)
        tree1.add_node(myMech2)
        tree1.add_node(myMech3)
        tree1.add_projection(MappingProjection(sender=myMech1, receiver=myMech3), myMech1, myMech3)
        tree1.add_projection(MappingProjection(sender=myMech2, receiver=myMech3), myMech2, myMech3)

        # validate first composition ---------------------------------------------

        tree1._analyze_graph()
        origins = tree1.get_nodes_by_role(NodeRole.ORIGIN)
        assert len(origins) == 2
        assert myMech1 in origins
        assert myMech2 in origins
        terminals = tree1.get_nodes_by_role(NodeRole.TERMINAL)
        assert len(terminals) == 1
        assert myMech3 in terminals

        # create second composition ----------------------------------------------

        # Mech3 --
        #           --> Mech5
        # Mech4 --

        tree2 = Composition()
        tree2.add_node(myMech3)
        tree2.add_node(myMech4)
        tree2.add_node(myMech5)
        tree2.add_projection(MappingProjection(sender=myMech3, receiver=myMech5), myMech3, myMech5)
        tree2.add_projection(MappingProjection(sender=myMech4, receiver=myMech5), myMech4, myMech5)

        # validate second composition ----------------------------------------------

        tree2._analyze_graph()
        origins = tree2.get_nodes_by_role(NodeRole.ORIGIN)
        assert len(origins) == 2
        assert myMech3 in origins
        assert myMech4 in origins
        terminals = tree2.get_nodes_by_role(NodeRole.TERMINAL)
        assert len(terminals) == 1
        assert myMech5 in terminals

        # combine the compositions -------------------------------------------------

        tree1.add_pathway(tree2)
        tree1._analyze_graph()
        # no need for a projection connecting the two compositions because they share myMech3

        origins = tree1.get_nodes_by_role(NodeRole.ORIGIN)
        assert len(origins) == 3
        assert myMech1 in origins
        assert myMech2 in origins
        assert myMech4 in origins
        terminals = tree1.get_nodes_by_role(NodeRole.TERMINAL)
        assert len(terminals) == 1
        assert myMech5 in terminals

    @pytest.mark.pathways
    def test_three_level_deep_pathway_routing_single_mech(self):
        p2 = ProcessingMechanism(name='p2')
        p0 = ProcessingMechanism(name='p0')

        c2 = Composition(name='c2', pathways=[p2])
        c1 = Composition(name='c1', pathways=[c2])
        c0 = Composition(name='c0', pathways=[[c1], [p0]])

        c0.add_projection(MappingProjection(), sender=p0, receiver=p2)
        result = c0.run([5])
        assert result == [5]

    @pytest.mark.pathways
    def test_three_level_deep_pathway_routing_two_mech(self):
        p3a = ProcessingMechanism(name='p3a')
        p3b = ProcessingMechanism(name='p3b')
        p1 = ProcessingMechanism(name='p1')

        c3 = Composition(name='c3', pathways=[[p3a], [p3b]])
        c2 = Composition(name='c2', pathways=[c3])
        c1 = Composition(name='c1', pathways=[[c2], [p1]])

        c1.add_projection(MappingProjection(), sender=p1, receiver=p3a)
        c1.add_projection(MappingProjection(), sender=p1, receiver=p3b)

        result = c1.run([5])
        assert result == [5, 5]

    @pytest.mark.pathways
    def test_three_level_deep_modulation_routing_single_mech(self):
        p3 = ProcessingMechanism(name='p3')
        ctrl1 = ControlMechanism(name='ctrl1',
                                 control=ControlSignal(modulates=(SLOPE, p3)))

        c3 = Composition(name='c3', pathways=[p3])
        c2 = Composition(name='c2', pathways=[c3])
        c1 = Composition(name='c1', pathways=[[(c2, NodeRole.INPUT)], [ctrl1]])

        result = c1.run({c2: 2, ctrl1: 5})
        assert result == [10]

    @pytest.mark.pathways
    def test_three_level_deep_modulation_routing_two_mech(self):
        p3a = ProcessingMechanism(name='p3a')
        p3b = ProcessingMechanism(name='p3b')
        ctrl1 = ControlMechanism(name='ctrl1',
                                 control=[
                                     ControlSignal(modulates=(SLOPE, p3a)),
                                     ControlSignal(modulates=(SLOPE, p3b))
                                 ])

        c3 = Composition(name='c3', pathways=[[p3a], [p3b]])
        c2 = Composition(name='c2', pathways=[c3])
        c1 = Composition(name='c1', pathways=[[(c2, NodeRole.INPUT)], [ctrl1]])

        result = c1.run({c2: [[2], [2]], ctrl1: [5]})
        assert result == [10, 10]

    @pytest.mark.pathways
    @pytest.mark.state_features
    def test_four_level_nested_transfer_mechanism_composition_parallel(self):
        # mechanisms
        A = ProcessingMechanism(name="A",
                                function=AdaptiveIntegrator(rate=0.1))
        B = ProcessingMechanism(name="B",
                                function=Logistic)

        comp_lvl3a = Composition(name="comp_lvl3a", pathways=[A, B])

        C = TransferMechanism(name="C",
                              function=Logistic,
                              integration_rate=0.1,
                              integrator_mode=True)

        comp_lvl3b = Composition(name="comp_lvl3b", pathways=[C])
        comp_lvl2 = Composition(name="comp_lvl2", pathways=[[comp_lvl3a], [comp_lvl3b]])
        comp_lvl1 = Composition(name="comp_lvl2", pathways=[comp_lvl2])
        comp_lvl0 = Composition(name="outer_comp", pathways=[comp_lvl1])
        ret = comp_lvl0.run(inputs={comp_lvl1: {comp_lvl2: {comp_lvl3a: [[1.0]], comp_lvl3b: [[1.0]]}}})
        assert np.allclose(ret, [[[0.52497918747894]], [[0.52497918747894]]])

    @pytest.mark.state_features
    @pytest.mark.control
    def test_four_level_nested_OCM_control(self):
        p_lvl3 = ProcessingMechanism(name='p_lvl3')

        c_lvl3 = Composition(name='c_lvl3', pathways=[p_lvl3])
        c_lvl2 = Composition(name='c_lvl2', pathways=[c_lvl3])
        c_lvl1 = Composition(name='c_lvl1', pathways=[c_lvl2])
        c_lvl0 = Composition(name='c_lvl0', pathways=[c_lvl1], controller_mode=BEFORE)

        c_lvl0.add_controller(OptimizationControlMechanism(
            name='c_top_controller',
            agent_rep=c_lvl0,
            state_features=[c_lvl1.input_port],
            objective_mechanism=ObjectiveMechanism(monitor=[p_lvl3]),
            function=GridSearch(),
            control_signals=ControlSignal(
                intensity_cost_function=lambda _: 0,
                modulates=(SLOPE, p_lvl3),
                allocation_samples=[10, 20, 30])))
        result = c_lvl0.run([5])
        assert result == [150]

    @pytest.mark.state_features
    @pytest.mark.control
    def test_four_level_nested_dual_OCM_control(self):
        p_lvl3 = ProcessingMechanism(name='p_lvl3')

        c_lvl3 = Composition(name='c_lvl3', pathways=[p_lvl3])
        c_lvl2 = Composition(name='c_lvl2', pathways=[c_lvl3])
        c_lvl1 = Composition(name='c_lvl1', pathways=[c_lvl2], controller_mode=BEFORE)

        c_lvl1.add_controller(OptimizationControlMechanism(
            name='c_lvl1_controller',
            agent_rep=c_lvl1,
            state_features=[c_lvl2.input_port],
            objective_mechanism=ObjectiveMechanism(monitor=[p_lvl3]),
            function=GridSearch(),
            control_signals=ControlSignal(
                intensity_cost_function=lambda _: 0,
                modulates=(SLOPE, p_lvl3),
                allocation_samples=[10, 20, 30])))

        c_lvl0 = Composition(name='c_lvl0', pathways=[c_lvl1], controller_mode=BEFORE)

        c_lvl0.add_controller(OptimizationControlMechanism(
            name='c_lvl0_controller',
            agent_rep=c_lvl0,
            state_features=[c_lvl1.input_port],
            objective_mechanism=ObjectiveMechanism(monitor=[p_lvl3]),
            function=GridSearch(),
            control_signals=ControlSignal(
                intensity_cost_function=lambda _: 0,
                modulates=(SLOPE, p_lvl3),
                allocation_samples=[10, 20, 30])))
        result = c_lvl0.run([5])
        assert result == [4500]

    @pytest.mark.control
    @pytest.mark.parametrize('nesting', ("unnested", "nested"))
    def test_partially_overlapping_local_and_control_mech_control_specs_in_unnested_and_nested_comp(self, nesting):
        pnl.clear_registry()
        samples = np.arange(0.1, 1.01, 0.3)
        Input = pnl.TransferMechanism(name='Input')
        reward = pnl.TransferMechanism(output_ports=[pnl.RESULT, pnl.MEAN, pnl.VARIANCE],
                                       name='reward',
                                       )
        Decision = pnl.DDM(function=pnl.DriftDiffusionAnalytical(drift_rate=(1.0,
                                                                     pnl.ControlProjection(function=pnl.Linear,
                                                                                       control_signal_params={
                                                                                           pnl.ALLOCATION_SAMPLES: samples,
                                                                                       })),
                                                         threshold=(1.0,
                                                                    pnl.ControlProjection(function=pnl.Linear,
                                                                                      control_signal_params={
                                                                                          pnl.ALLOCATION_SAMPLES: samples,
                                                                                      })),
                                                         noise=0.5,
                                                         starting_value=0,
                                                         non_decision_time=0.45),
                       output_ports=[pnl.DECISION_VARIABLE,
                                     pnl.RESPONSE_TIME,
                                     pnl.PROBABILITY_UPPER_THRESHOLD],
                       name='Decision')
        Response = pnl.DDM(function=pnl.DriftDiffusionAnalytical(drift_rate=1.0,
                                                          threshold=1.0,
                                                          noise=0.5,
                                                          starting_value=0,
                                                          non_decision_time=0.45),
                        output_ports=[pnl.DECISION_VARIABLE,
                                      pnl.RESPONSE_TIME,
                                      pnl.PROBABILITY_UPPER_THRESHOLD],
                        name='Response')

        icomp = pnl.Composition(name="EVC (inner comp)", retain_old_simulation_data=True)
        icomp.add_node(reward, required_roles=[pnl.NodeRole.OUTPUT])
        icomp.add_node(Decision, required_roles=[pnl.NodeRole.OUTPUT])
        icomp.add_node(Response, required_roles=[pnl.NodeRole.OUTPUT])
        icomp.add_linear_processing_pathway([Input, pnl.IDENTITY_MATRIX, Decision, Response])
        if nesting == 'nested':
            comp = Composition(nodes=icomp, name="Outer Composition")
        else:
            comp = icomp
        ocm=OptimizationControlMechanism(
            agent_rep=comp,
            monitor_for_control=[Decision.output_ports[pnl.DECISION_VARIABLE],
                                 Decision.output_ports[pnl.RESPONSE_TIME]],
            num_estimates=1,
            function=GridSearch,
            control_signals=[
                ControlSignal(modulates=('drift_rate',Decision), # OVERLAPS WITH CONTROL SPEC ON Decision
                              allocation_samples=[1,2]),
                ControlSignal(modulates=('threshold',Response), # ADDS CONTROL SPEC FOR Response
                              allocation_samples=[1,2]),
            ]
        )
        comp.add_controller(ocm)

        assert len(comp.controller.input_ports[pnl.OUTCOME].path_afferents) == 2
        if nesting == 'nested':
            # All Projections to controller's OUTCOME InputPort should be from input_CIM
            assert all(isinstance(comp.controller.input_ports[pnl.OUTCOME].path_afferents[i].sender.owner,
                                  pnl.CompositionInterfaceMechanism) for i in range(2))

        assert len(comp.controller.control_signals) == 4  # Should be 4:  Decision threshold (spec'd locally on mech)
                                                          #               Decision drift_rate (spec'd on mech and OCM)
                                                          #               Response threshold (spec'd on OCM)
                                                          #               RANDOMIZATION
        ctl_sig_names = ['Decision[drift_rate] ControlSignal', 'Decision[threshold] ControlSignal',
                         'Response[threshold] ControlSignal', 'RANDOMIZATION_CONTROL_SIGNAL']
        assert all([name in ctl_sig_names for name in comp.controller.control_signals.names])
        if nesting == 'nested':
            # All of the controller's ControlSignals should project to the ParameterCIM for the nested comp
            assert all(isinstance(comp.controller.control_signals[i].efferents[0].receiver.owner,
                                  pnl.CompositionInterfaceMechanism) for i in range(4))


class TestOverloadedCompositions:
    def test_mechanism_different_inputs(self):
        a = TransferMechanism(name='a', function=Linear(slope=2))
        b = TransferMechanism(name='b')
        c = TransferMechanism(name='c', function=Linear(slope=3))
        p = MappingProjection(sender=a, receiver=b)

        comp = Composition()
        comp2 = Composition()

        comp.add_node(a)
        comp.add_node(b)
        comp.add_projection(p, a, b)

        comp2.add_node(a)
        comp2.add_node(b)
        comp2.add_node(c)
        comp2.add_projection(p, a, b)
        comp2.add_projection(MappingProjection(sender=c, receiver=b), c, b)

        comp.run({a: 1})
        comp2.run({a: 1, c: 1})

        np.testing.assert_allclose(comp.results, [[np.array([2])]])
        np.testing.assert_allclose(comp2.results, [[np.array([5])]])


class TestCompositionInterface:

    def test_one_input_port_per_origin_two_origins(self):

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
        comp.add_node(A)
        comp.add_node(B)
        comp.add_node(C)
        comp.add_node(D)
        comp.add_projection(MappingProjection(sender=A, receiver=C), A, C)
        comp.add_projection(MappingProjection(sender=B, receiver=D), B, D)
        comp.add_node(E)
        comp.add_projection(MappingProjection(sender=C, receiver=E), C, E)
        comp.add_projection(MappingProjection(sender=D, receiver=E), D, E)
        inputs_dict = {
            A: [[5.]],
            # two trials of one InputPort each
            #        TRIAL 1     TRIAL 2
            # A : [ [ [0,0] ] , [ [0, 0] ]  ]

            # two trials of multiple input ports each
            #        TRIAL 1     TRIAL 2

            #       TRIAL1 IS1      IS2      IS3     TRIAL2    IS1      IS2
            # A : [ [     [0,0], [0,0,0], [0,0,0,0] ] ,     [ [0, 0],   [0] ]  ]
            B: [[5.]]
        }
        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched)

        assert np.allclose(250, output)

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
        comp.add_node(A)
        comp.add_node(B)
        comp.add_node(C)
        comp.add_node(D)
        comp.add_projection(MappingProjection(sender=A, receiver=C), A, C)
        comp.add_projection(MappingProjection(sender=B, receiver=D), B, D)
        comp.add_node(E)
        comp.add_projection(MappingProjection(sender=C, receiver=E), C, E)
        comp.add_projection(MappingProjection(sender=D, receiver=E), D, E)
        inputs_dict = {
            A: [[5.]],
            B: [[5.]]
        }
        sched = Scheduler(composition=comp)

        output = comp.run(inputs=inputs_dict, scheduler=sched)
        assert np.allclose(250, output)

        inputs_dict2 = {
            A: [[2.]],
            B: [[5.]],
            # two trials of one InputPort each
            #        TRIAL 1     TRIAL 2
            # A : [ [ [0,0] ] , [ [0, 0] ]  ]

            # two trials of multiple input ports each
            #        TRIAL 1     TRIAL 2

            #       TRIAL1 IS1      IS2      IS3     TRIAL2    IS1      IS2
            # A : [ [     [0,0], [0,0,0], [0,0,0,0] ] ,     [ [0, 0],   [0] ]  ]
            B: [[5.]]
        }
        sched = Scheduler(composition=comp)

        output = comp.run(inputs=inputs_dict, scheduler=sched)

        assert np.allclose([np.array([[250.]]), np.array([[250.]])], output)

        # add a new branch to the composition
        F = TransferMechanism(name="composition-pytests-F", function=Linear(slope=2.0))
        G = TransferMechanism(name="composition-pytests-G", function=Linear(slope=2.0))
        comp.add_node(F)
        comp.add_node(G)
        comp.add_projection(projection=MappingProjection(sender=F, receiver=G), sender=F, receiver=G)
        comp.add_projection(projection=MappingProjection(sender=G, receiver=E), sender=G, receiver=E)

        # execute the updated composition
        inputs_dict2 = {
            A: [[1.]],
            B: [[2.]],
            F: [[3.]]
        }

        sched = Scheduler(composition=comp)
        output2 = comp.run(inputs=inputs_dict2, scheduler=sched)

        assert np.allclose(np.array([[135.]]), output2)

    def test_changing_origin_for_second_execution(self):

        comp = Composition()
        A = TransferMechanism(name="composition-pytests-A",
                              function=Linear(slope=1.0)
                              )

        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=1.0))
        C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=5.0))
        comp.add_node(A)
        comp.add_node(B)
        comp.add_node(C)
        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        comp.add_projection(MappingProjection(sender=B, receiver=C), B, C)
        inputs_dict = {A: [[5.]]}
        sched = Scheduler(composition=comp)

        output = comp.run(inputs=inputs_dict, scheduler=sched)

        assert np.allclose(25, output)

        # add a new origin to the composition
        F = TransferMechanism(name="composition-pytests-F", function=Linear(slope=2.0))
        comp.add_node(F)
        comp.add_projection(projection=MappingProjection(sender=F, receiver=A), sender=F, receiver=A)


        # execute the updated composition
        inputs_dict2 = {F: [[3.]]}

        sched = Scheduler(composition=comp)
        output2 = comp.run(inputs=inputs_dict2, scheduler=sched)

        connections_to_A = []
        expected_connections_to_A = [(F.output_ports[0], A.input_ports[0])]
        for input_port in A.input_ports:
            for p_a in input_port.path_afferents:
                connections_to_A.append((p_a.sender, p_a.receiver))

        assert connections_to_A == expected_connections_to_A
        assert np.allclose(np.array([[30.]]), output2)

    def test_two_input_ports_new_inputs_second_trial(self):

        comp = Composition()
        my_fun = Linear(
            # default_variable=[[0], [0]],
            # ^ setting default_variable on the function actually does not matter -- does the mechanism update it?
            slope=1.0)
        A = TransferMechanism(name="composition-pytests-A",
                              default_variable=[[0], [0]],
                              input_ports=[{NAME: "Input Port 1",
                                             },
                                            {NAME: "Input Port 2",
                                             }],
                              function=my_fun
                              )
        comp.add_node(A)
        inputs_dict = {A: [[5.], [5.]]}

        sched = Scheduler(composition=comp)
        comp.run(inputs=inputs_dict, scheduler=sched)

        inputs_dict2 = {A: [[2.], [4.]]}

        output = comp.run(inputs=inputs_dict2, scheduler=sched)

        assert np.allclose(A.input_ports[0].parameters.value.get(comp), [2.])
        assert np.allclose(A.input_ports[1].parameters.value.get(comp), [4.])
        assert np.allclose(A.parameters.variable.get(comp.default_execution_id), [[2.], [4.]])
        assert np.allclose(output, np.array([[2.], [4.]]))

    def test_two_input_ports_new_origin_second_trial(self):

        # A --> B --> C

        comp = Composition()
        my_fun = Linear(
            # default_variable=[[0], [0]],
            # ^ setting default_variable on the function actually does not matter -- does the mechanism update it?
            slope=1.0)
        A = TransferMechanism(
            name="composition-pytests-A",
            default_variable=[[0], [0]],
            input_ports=[
                {NAME: "Input Port 1", },
                {NAME: "Input Port 2", }
            ],
            function=my_fun
        )

        B = TransferMechanism(name="composition-pytests-B", function=Linear(slope=2.0))
        C = TransferMechanism(name="composition-pytests-C", function=Linear(slope=5.0))
        comp.add_node(A)
        comp.add_node(B)
        comp.add_node(C)
        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        comp.add_projection(MappingProjection(sender=B, receiver=C), B, C)

        inputs_dict = {A: [[5.], [5.]]}

        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched)
        assert np.allclose(A.input_ports[0].parameters.value.get(comp), [5.])
        assert np.allclose(A.input_ports[1].parameters.value.get(comp), [5.])
        assert np.allclose(A.parameters.variable.get(comp.default_execution_id), [[5.], [5.]])
        assert np.allclose(output, [[50.]])

        # A --> B --> C
        #     ^
        # D __|

        D = TransferMechanism(
            name="composition-pytests-D",
            default_variable=[[0], [0]],
            input_ports=[
                {NAME: "Input Port 1", },
                {NAME: "Input Port 2", }
            ],
            function=my_fun
        )
        comp.add_node(D)
        comp.add_projection(MappingProjection(sender=D, receiver=B), D, B)
        # Need to analyze graph again (identify D as an origin so that we can assign input) AND create the scheduler
        # again (sched, even though it is tied to comp, will not update according to changes in comp)
        sched = Scheduler(composition=comp)

        inputs_dict2 = {A: [[2.], [4.]],
                        D: [[2.], [4.]]}
        output2 = comp.run(inputs=inputs_dict2, scheduler=sched)
        assert np.allclose(A.input_ports[0].parameters.value.get(comp), [2.])
        assert np.allclose(A.input_ports[1].parameters.value.get(comp), [4.])
        assert np.allclose(A.parameters.variable.get(comp.default_execution_id), [[2.], [4.]])

        assert np.allclose(D.input_ports[0].parameters.value.get(comp), [2.])
        assert np.allclose(D.input_ports[1].parameters.value.get(comp), [4.])
        assert np.allclose(D.parameters.variable.get(comp.default_execution_id), [[2.], [4.]])

        assert np.allclose(np.array([[40.]]), output2)

    def test_output_cim_one_terminal_mechanism_multiple_output_ports(self):

        comp = Composition()
        A = TransferMechanism(name="composition-pytests-A",
                              function=Linear(slope=1.0))
        B = TransferMechanism(name="composition-pytests-B",
                              function=Linear(slope=1.0))
        C = TransferMechanism(name="composition-pytests-C",
                              function=Linear(slope=2.0),
                              output_ports=[RESULT, VARIANCE])
        comp.add_node(A)
        comp.add_node(B)
        comp.add_node(C)

        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        comp.add_projection(MappingProjection(sender=B, receiver=C), B, C)

        comp.run(inputs={A: [1.0]})

        for terminal_port in comp.output_CIM_ports:
            # all CIM OutputPort keys in the CIM --> Terminal mapping dict are on the actual output CIM
            assert (comp.output_CIM_ports[terminal_port][0] in comp.output_CIM.input_ports) and \
                   (comp.output_CIM_ports[terminal_port][1] in comp.output_CIM.output_ports)

        # all Terminal Output ports are in the CIM --> Terminal mapping dict
        assert C.output_ports[0] in comp.output_CIM_ports.keys()
        assert C.output_ports[1] in comp.output_CIM_ports.keys()

        assert len(comp.output_CIM.output_ports) == 2

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
                              output_ports=[RESULT, VARIANCE])
        comp.add_node(A)
        comp.add_node(B)
        comp.add_node(C)
        comp.add_node(D)
        comp.add_node(E)
        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        comp.add_projection(MappingProjection(sender=B, receiver=C), B, C)
        comp.add_projection(MappingProjection(sender=B, receiver=D), B, D)
        comp.add_projection(MappingProjection(sender=B, receiver=E), B, E)
        comp.run(inputs={A: [1.0]})

        for terminal_port in comp.output_CIM_ports:
            # all CIM OutputPort keys in the CIM --> Terminal mapping dict are on the actual output CIM
            assert (comp.output_CIM_ports[terminal_port][0] in comp.output_CIM.input_ports) and \
                   (comp.output_CIM_ports[terminal_port][1] in comp.output_CIM.output_ports)

        # all Terminal Output ports are in the CIM --> Terminal mapping dict
        assert C.output_port in comp.output_CIM_ports.keys()
        assert D.output_port in comp.output_CIM_ports.keys()
        assert E.output_ports[0] in comp.output_CIM_ports.keys()
        assert E.output_ports[1] in comp.output_CIM_ports.keys()

        assert len(comp.output_CIM.output_ports) == 4

    def test_default_variable_shape_of_output_CIM(self):
        comp = Composition(name='composition')
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')

        comp.add_node(A)
        comp.add_node(B)

        comp.run(inputs={A: [1.0],
                         B: [2.0]})

        out = comp.output_CIM

        assert np.allclose(np.shape(out.defaults.variable), (2,1))
        assert np.allclose(out.parameters.variable.get(comp), [[1.0], [2.0]])

        C = ProcessingMechanism(name='C')
        comp.add_node(C)

        comp.run(inputs={A: [1.0],
                         B: [2.0],
                         C: [3.0]})

        out = comp.output_CIM

        assert np.allclose(np.shape(out.defaults.variable), (3, 1))
        assert np.allclose(out.parameters.variable.get(comp), [[1.0], [2.0], [3.0]])

        T = ProcessingMechanism(name='T')
        comp.add_linear_processing_pathway([A, T])
        comp.add_linear_processing_pathway([B, T])
        comp.add_linear_processing_pathway([C, T])

        comp.run(inputs={A: [1.0],
                         B: [2.0],
                         C: [3.0]})

        out = comp.output_CIM
        print(out.input_values)
        print(out.variable)
        print(out.defaults.variable)
        assert np.allclose(np.shape(out.defaults.variable), (1, 1))
        assert np.allclose(out.parameters.variable.get(comp), [[6.0]])

    def test_inner_composition_change_before_run(self):
        outer_comp = Composition(name="Outer Comp")
        inner_comp = Composition(name="Inner Comp")

        A = pnl.TransferMechanism(name='A')
        B = pnl.TransferMechanism(name='B')
        C = pnl.TransferMechanism(name='C')

        inner_comp.add_nodes([B, C])
        outer_comp.add_nodes([A, inner_comp])

        outer_comp.add_projection(pnl.MappingProjection(), A, inner_comp)
        inner_comp.add_projection(pnl.MappingProjection(), B, C)

        # comp.show_graph()
        outer_comp.run(inputs={A: 1})

        # inner_comp is updated to make B not an OUTPUT node
        # after being added to comp
        assert len(outer_comp.output_CIM.output_ports) == 1
        assert len(outer_comp.results[0]) == 1


class TestInputPortSpecifications:

    def test_two_input_ports_created_with_dictionaries(self):

        comp = Composition()
        A = ProcessingMechanism(
            name="composition-pytests-A",
            default_variable=[[0], [0]],
            # input_ports=[
            #     {NAME: "Input Port 1", },
            #     {NAME: "Input Port 2", }
            # ],
            function=Linear(slope=1.0)
            # specifying default_variable on the function doesn't seem to matter?
        )

        comp.add_node(A)


        inputs_dict = {A: [[2.], [4.]]}
        sched = Scheduler(composition=comp)
        comp.run(inputs=inputs_dict, scheduler=sched)

        assert np.allclose(A.input_ports[0].parameters.value.get(comp), [2.0])
        assert np.allclose(A.input_ports[1].parameters.value.get(comp), [4.0])
        assert np.allclose(A.parameters.variable.get(comp.default_execution_id), [[2.0], [4.0]])

    def test_two_input_ports_created_first_with_deferred_init(self):
        comp = Composition()

        # create mechanism A
        I1 = InputPort(
            name="Input Port 1",
            reference_value=[0]
        )
        I2 = InputPort(
            name="Input Port 2",
            reference_value=[0]
        )
        A = TransferMechanism(
            name="composition-pytests-A",
            default_variable=[[0], [0]],
            input_ports=[I1, I2],
            function=Linear(slope=1.0)
        )

        # add mech A to composition
        comp.add_node(A)

        # get comp ready to run (identify roles, create sched, assign inputs)
        inputs_dict = { A: [[2.],[4.]]}

        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched)

        assert np.allclose(A.input_ports[0].parameters.value.get(comp), [2.0])
        assert np.allclose(A.input_ports[1].parameters.value.get(comp), [4.0])
        assert np.allclose(A.parameters.variable.get(comp.default_execution_id), [[2.0], [4.0]])

    def test_two_input_ports_created_with_keyword(self):
        comp = Composition()

        # create mechanism A

        A = TransferMechanism(
            name="composition-pytests-A",
            default_variable=[[0], [0]],
            input_ports=[INPUT_PORT, INPUT_PORT],
            function=Linear(slope=1.0)
        )

        # add mech A to composition
        comp.add_node(A)

        # get comp ready to run (identify roles, create sched, assign inputs)
        inputs_dict = {A: [[2.], [4.]]}

        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched)

        assert np.allclose(A.input_ports[0].parameters.value.get(comp), [2.0])
        assert np.allclose(A.input_ports[1].parameters.value.get(comp), [4.0])
        assert np.allclose(A.parameters.variable.get(comp.default_execution_id), [[2.0], [4.0]])

        assert np.allclose([[2], [4]], output)

    def test_two_input_ports_created_with_strings(self):
        comp = Composition()

        # create mechanism A

        A = TransferMechanism(
            name="composition-pytests-A",
            default_variable=[[0], [0]],
            input_ports=["Input Port 1", "Input Port 2"],
            function=Linear(slope=1.0)
        )

        # add mech A to composition
        comp.add_node(A)

        # get comp ready to run (identify roles, create sched, assign inputs)

        inputs_dict = {A: [[2.], [4.]]}

        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched)

        assert np.allclose(A.input_ports[0].parameters.value.get(comp), [2.0])
        assert np.allclose(A.input_ports[1].parameters.value.get(comp), [4.0])
        assert np.allclose(A.parameters.variable.get(comp.default_execution_id), [[2.0], [4.0]])

    def test_two_input_ports_created_with_values(self):
        comp = Composition()

        # create mechanism A

        A = TransferMechanism(
            name="composition-pytests-A",
            default_variable=[[0], [0]],
            input_ports=[[0.], [0.]],
            function=Linear(slope=1.0)
        )

        # add mech A to composition
        comp.add_node(A)

        # get comp ready to run (identify roles, create sched, assign inputs)
        inputs_dict = {A: [[2.], [4.]]}

        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched)

        assert np.allclose(A.input_ports[0].parameters.value.get(comp), [2.0])
        assert np.allclose(A.input_ports[1].parameters.value.get(comp), [4.0])
        assert np.allclose(A.parameters.variable.get(comp.default_execution_id), [[2.0], [4.0]])


class TestInputSpecifications:

    # def test_2_mechanisms_default_input_1(self):
    #     comp = Composition()
    #     A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
    #     B = TransferMechanism(function=Linear(slope=5.0))
    #     comp.add_node(A)
    #     comp.add_node(B)
    #     comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
    #     sched = Scheduler(composition=comp)
    #     output = comp.run(
    #         scheduler=sched
    #     )
    #     assert 25 == output[0][0]

    def test_3_origins(self):
        comp = Composition()
        I1 = InputPort(
                        name="Input Port 1",
                        reference_value=[0]
        )
        I2 = InputPort(
                        name="Input Port 2",
                        reference_value=[0]
        )
        A = TransferMechanism(
                            name="composition-pytests-A",
                            default_variable=[[0], [0]],
                            input_ports=[I1, I2],
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
        comp.add_node(A)
        comp.add_node(B)
        comp.add_node(C)
        comp.add_node(D)
        comp.add_projection(MappingProjection(sender=A, receiver=D), A, D)
        comp.add_projection(MappingProjection(sender=B, receiver=D), B, D)
        comp.add_projection(MappingProjection(sender=C, receiver=D), C, D)
        inputs = {A: [[[0], [0]], [[1], [1]], [[2], [2]]],
                  B: [[0, 0], [1, 1], [2, 2]],
                  C: [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        }

        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs, scheduler=sched)

        assert np.allclose(np.array([[12.]]), output)

    def test_2_mechanisms_input_5(self):
        comp = Composition()
        A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
        B = TransferMechanism(function=Linear(slope=5.0))
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        inputs_dict = {A: [[5]]}
        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched)
        assert np.allclose([125], output)

    def test_run_2_mechanisms_reuse_input(self):
        comp = Composition()
        A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
        B = TransferMechanism(function=Linear(slope=5.0))
        comp.add_node(A)
        comp.add_node(B)
        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        inputs_dict = {A: [[5]]}
        sched = Scheduler(composition=comp)
        output = comp.run(inputs=inputs_dict, scheduler=sched, num_trials=5)
        assert np.allclose([125], output)

    def test_some_inputs_not_specified(self):
        comp = Composition()

        A = TransferMechanism(name="composition-pytests-A",
                              default_variable=[[1.0, 2.0], [3.0, 4.0]],
                              function=Linear(slope=2.0))

        B = TransferMechanism(name="composition-pytests-B",
                              default_variable=[[0.0, 0.0, 0.0]],
                              function=Linear(slope=3.0))

        C = TransferMechanism(name="composition-pytests-C")

        D = TransferMechanism(name="composition-pytests-D")

        comp.add_node(A)
        comp.add_node(B)
        comp.add_node(C)
        comp.add_node(D)


        inputs = {B: [[1., 2., 3.]],
                  D: [[4.]]}

        sched = Scheduler(composition=comp)
        comp.run(inputs=inputs, scheduler=sched)[0]

        assert np.allclose(A.get_output_values(comp), [[2.0, 4.0], [6.0, 8.0]])
        assert np.allclose(B.get_output_values(comp), [[3., 6., 9.]])
        assert np.allclose(C.get_output_values(comp), [[0.]])
        assert np.allclose(D.get_output_values(comp), [[4.]])

    def test_some_inputs_not_specified_origin_node_is_composition(self):

        compA = Composition()
        A = TransferMechanism(name="composition-pytests-A",
                              default_variable=[[1.0, 2.0], [3.0, 4.0]],
                              function=Linear(slope=2.0))
        compA.add_node(A)

        comp = Composition()

        B = TransferMechanism(name="composition-pytests-B",
                              default_variable=[[0.0, 0.0, 0.0]],
                              function=Linear(slope=3.0))

        C = TransferMechanism(name="composition-pytests-C")

        D = TransferMechanism(name="composition-pytests-D")

        comp.add_node(compA)
        comp.add_node(B)
        comp.add_node(C)
        comp.add_node(D)


        inputs = {B: [[1., 2., 3.]],
                  D: [[4.]]}

        sched = Scheduler(composition=comp)
        comp.run(inputs=inputs, scheduler=sched)[0]

        assert np.allclose(A.get_output_values(comp), [[2.0, 4.0], [6.0, 8.0]])
        assert np.allclose(compA.get_output_values(comp), [[2.0, 4.0], [6.0, 8.0]])
        assert np.allclose(B.get_output_values(comp), [[3., 6., 9.]])
        assert np.allclose(C.get_output_values(comp), [[0.]])
        assert np.allclose(D.get_output_values(comp), [[4.]])

    def test_function_as_input(self):
        c = pnl.Composition()

        m1 = pnl.TransferMechanism()
        m2 = pnl.TransferMechanism()

        c.add_linear_processing_pathway([m1, m2])

        def test_function(trial_num):
            stimuli = list(range(10))
            return {
                m1: stimuli[trial_num]
            }

        c.run(inputs=test_function,
              num_trials=10)
        assert c.parameters.results.get(c) == [[np.array([0.])], [np.array([1.])], [np.array([2.])], [np.array([3.])],
                                               [np.array([4.])], [np.array([5.])], [np.array([6.])], [np.array([7.])],
                                               [np.array([8.])], [np.array([9.])]]

    def test_function_as_learning_input(self):
        num_epochs=2

        xor_inputs = np.array(  # the inputs we will provide to the model
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]])

        xor_targets = np.array(  # the outputs we wish to see from the model
            [[0],
             [1],
             [1],
             [0]])

        in_to_hidden_matrix = np.random.rand(2,10)
        hidden_to_out_matrix = np.random.rand(10,1)

        input_comp = pnl.TransferMechanism(name='input_comp',
                                    default_variable=np.zeros(2))

        hidden_comp = pnl.TransferMechanism(name='hidden_comp',
                                    default_variable=np.zeros(10),
                                    function=pnl.Logistic())

        output_comp = pnl.TransferMechanism(name='output_comp',
                                    default_variable=np.zeros(1),
                                    function=pnl.Logistic())

        in_to_hidden_comp = pnl.MappingProjection(name='in_to_hidden_comp',
                                    matrix=in_to_hidden_matrix.copy(),
                                    sender=input_comp,
                                    receiver=hidden_comp)

        hidden_to_out_comp = pnl.MappingProjection(name='hidden_to_out_comp',
                                    matrix=hidden_to_out_matrix.copy(),
                                    sender=hidden_comp,
                                    receiver=output_comp)

        xor_comp = pnl.Composition()

        backprop_pathway = xor_comp.add_backpropagation_learning_pathway([input_comp,
                                                                            in_to_hidden_comp,
                                                                            hidden_comp,
                                                                            hidden_to_out_comp,
                                                                            output_comp],
                                                                            learning_rate=10)
        def test_function(trial_num):
            return {
                input_comp: xor_inputs[trial_num]
            }

        xor_comp.learn(inputs=test_function,
              num_trials=4)

    @pytest.mark.control
    @pytest.mark.parametrize(
        "controllers, results",[
            ('none', [[-2], [1]]),
            ('inner',  [[-2], [10]]),
            ('outer',  [[-2], [10]]),
            ('inner_and_outer', [[-2], [100]]),
        ]
    )
    @pytest.mark.parametrize(
        "inputs_arg",['inputs_dict','generator_function','generator_instance']
    )
    def test_input_type_equivalence(self, controllers, results, inputs_arg):
        # instantiate mechanisms and inner comp
        ia = pnl.TransferMechanism(name='ia')
        ib = pnl.TransferMechanism(name='ib')
        icomp = pnl.Composition(name='icomp', controller_mode=pnl.BEFORE)

        # set up structure of inner comp
        icomp.add_node(ia, required_roles=pnl.NodeRole.INPUT)
        icomp.add_node(ib, required_roles=pnl.NodeRole.OUTPUT)
        icomp.add_projection(pnl.MappingProjection(), sender=ia, receiver=ib)

        # add controller to inner comp
        if controllers in {'inner', 'inner_and_outer'}:
            icomp.add_controller(
                    pnl.OptimizationControlMechanism(
                            agent_rep=icomp,
                            state_features=[ia.input_port],
                            name="iController",
                            objective_mechanism=pnl.ObjectiveMechanism(
                                    monitor=ib.output_port,
                                    function=pnl.SimpleIntegrator,
                                    name="iController Objective Mechanism"
                            ),
                            function=pnl.GridSearch(direction=pnl.MAXIMIZE),
                            control_signals=[pnl.ControlSignal(projections=[(pnl.SLOPE, ia)],
                                                               variable=1.0,
                                                               intensity_cost_function=pnl.Linear(slope=0.0),
                                                               allocation_samples=pnl.SampleSpec(start=1.0,
                                                                                                 stop=10.0,
                                                                                                 num=2))])
            )

        # instantiate outer comp
        ocomp = pnl.Composition(name='ocomp', controller_mode=pnl.BEFORE)

        # setup structure of outer comp
        ocomp.add_node(icomp)

        # add controller to outer comp
        if controllers in {'outer', 'inner_and_outer'}:
            ocomp.add_controller(
                    pnl.OptimizationControlMechanism(
                            agent_rep=ocomp,
                            state_features=[ia.input_port],
                            name="oController",
                            objective_mechanism=pnl.ObjectiveMechanism(
                                    monitor=ib.output_port,
                                    function=pnl.SimpleIntegrator,
                                    name="oController Objective Mechanism"
                            ),
                            function=pnl.GridSearch(direction=pnl.MAXIMIZE),
                            control_signals=[pnl.ControlSignal(projections=[(pnl.SLOPE, ia)],
                                                               variable=1.0,
                                                               intensity_cost_function=pnl.Linear(slope=0.0),
                                                               allocation_samples=pnl.SampleSpec(start=1.0,
                                                                                                 stop=10.0,
                                                                                                 num=2))
                                             ]),
            )

        # set up input using three different formats:
        #  1) inputs dict
        #  2) generator function
        #  3) instance of generator function
        inputs_dict = {
            icomp:
                {
                    ia: [[-2], [1]]
                }
        }

        def inputs_generator_function():
            for i in range(2):
                yield {
                    icomp:
                        {
                            ia: inputs_dict[icomp][ia][i]
                        }
                }

        inputs_generator_instance = inputs_generator_function()

        inputs_source = {
            'inputs_dict': inputs_dict,
            'generator_function': inputs_generator_function,
            'generator_instance': inputs_generator_instance
        }[inputs_arg]

        # # Version with reporting for debugging purposes:
        # ib.reportOutputPref=[pnl.VALUE, pnl.VARIABLE]
        # icomp.controller.reportOutputPref = pnl.ReportOutput.ON
        # ocomp.controller.reportOutputPref = pnl.ReportOutput.FULL
        # ocomp.run(inputs=inputs_dict,
        #           report_output=pnl.ReportOutput.FULL,
        #           report_progress=pnl.ReportProgress.ON,
        #           report_simulations=pnl.ReportSimulations.ON,
        #           report_to_devices=pnl.ReportDevices.DIVERT
        #           )
        # actual_output = ocomp.rich_diverted_reports

        # run Composition with all three input types and assert that results are as expected.
        ocomp.run(inputs=inputs_source)
        assert ocomp.results == results

    expected_format_strings = \
        [
        '{\n\tX: [ [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]] ],'
        '\n\tICOMP: [ [[0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]], [[0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]] ],'
        '\n\tY: [ [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]] ]\n}',

        "\nInputs to (nested) INPUT Nodes of OCOMP for 3 trials:\n\tX: [ [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]], [[0.0, "
        "0.0, 0.0],[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]] ]"
        "\n\tICOMP: \n\t\tA: [ ['red'], ['green'], ['red'] ]"
        "\n\t\tC: [ ['red','blue'], ['green','yellow'], ['orange','purple'] ]"
        "\n\tY: [ ['red','red'], ['green','green'], ['red','red'] "
        "\n\nFormat as follows for inputs to run():"
        "\n{\n\tX: [ [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]], "
        "[[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]], "
        "[[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]] ],"
        "\n\tICOMP: [ [[0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]], "
        "[[0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]], "
        "[[0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]] ],"
        "\n\tY: [ [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]], "
        "[[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]], "
        "[[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]] ]\n}"
        ]
    test_args = [
        # form, labels, nested, num_trials, expected_format_string
        (pnl.TEXT, False, False, 2, expected_format_strings[0]),
        (pnl.TEXT, True, True, pnl.FULL, expected_format_strings[1]),
        (pnl.DICT, False, False, 1, None),
        (pnl.DICT, False, False, pnl.FULL, None),
        (pnl.DICT, True, True, 1, None),
        (pnl.DICT, True, False, pnl.FULL, None)
    ]
    @pytest.mark.parametrize('form, use_labels, show_nested, num_trials, expected_format_string', test_args,
                             ids = [f"{'dict' if x[0] else 'string'} "
                                    f"{'use_labels_true' if x[1] else 'use_labels_false'} "
                                    f"{'show_nested_true' if x[2] else 'show_nested_false'} "
                                    f"num_trials-{x[3]} "
                                    f"{'expected_format_string' if x[4] else 'None'}" for x in test_args]
                             )
    def test_get_input_format(self, form, use_labels, show_nested, num_trials, expected_format_string):
        """Also tests input_labels_dict"""

        A = pnl.ProcessingMechanism(size=1, name='A',
                                input_labels={0:{'red':0, 'green':1},
                                              1:{'blue':2, 'yellow':3}})
        B = pnl.ProcessingMechanism(size=2, name='B')
        C = pnl.ProcessingMechanism(size=[3,3],
                                input_ports=['C INPUT 1', 'C INPUT 2'],
                                input_labels={'C INPUT 1':{'red':[0,0,0], 'green':[1,1,1], 'orange':[2,2,2]},
                                              'C INPUT 2':{'blue':[3,3,3], 'yellow':[4,4,4], 'purple':[5,5,5]}},
                                name='C')
        assert C.variable.shape == (2,3)
        X = ProcessingMechanism(size=[3,3],
                                input_ports=['X INPUT 1', 'X INPUT 2'],
                                name='X',
                                # input_labels={0:{'red':[0,0,0], 'green':[1,1,1]}}  # Specify dict for only one port
                                )
        # Use TransferMechanism so that 2nd OutputPort uses 2nd item of Mechanism's value
        #    (i.e. ,without having to specify that explicitly, as would be the case for ProcessingMechanism)
        Y = pnl.TransferMechanism(input_ports=[{NAME:'Y INPUT 1', pnl.SIZE: 3, pnl.FUNCTION: pnl.Reduce},
                                                 {NAME:'Y INPUT 2', pnl.SIZE: 3}],
                                    # Test specification of labels for all InputPorts of Mechanism:
                                    input_labels={'red':[0,0,0], 'green':[1,1,1]},
                                    name='Y')
        assert len(Y.input_ports[0].variable) == 3
        assert len(Y.input_ports[0].value) == 1
        assert len(Y.input_ports[1].variable) == 3
        assert len(Y.input_ports[1].value) == 3
        icomp = Composition(pathways=[[A,B],[C]], name='ICOMP')
        ocomp = Composition(nodes=[X, icomp, Y], name='OCOMP')

        inputs_dict = ocomp.get_input_format(form=form,
                                             num_trials=num_trials,
                                             use_labels=use_labels,
                                             show_nested_input_nodes=show_nested)
        if form == pnl.TEXT:
            assert inputs_dict == expected_format_string
        else:
            ocomp.run(inputs=inputs_dict)
            if num_trials == pnl.FULL:
                num_trials = 2
            len(ocomp.results)==num_trials


class TestProperties:
    @pytest.mark.composition
    @pytest.mark.parametrize("mode", [pnl.ExecutionMode.Python, pnl.ExecutionMode.Auto,
                                      pytest.param(pnl.ExecutionMode.LLVM, marks=[pytest.mark.xfail, pytest.mark.llvm]),
                                      pytest.param(pnl.ExecutionMode.LLVMExec, marks=[pytest.mark.xfail, pytest.mark.llvm]),
                                      pytest.param(pnl.ExecutionMode.LLVMRun, marks=[pytest.mark.xfail, pytest.mark.llvm]),
                                      pytest.param(pnl.ExecutionMode.PTXExec, marks=[pytest.mark.xfail, pytest.mark.llvm, pytest.mark.cuda]),
                                      pytest.param(pnl.ExecutionMode.PTXRun, marks=[pytest.mark.xfail, pytest.mark.llvm, pytest.mark.cuda]),
                                     ])
    def test_llvm_fallback(self, mode):
        comp = Composition()
        # FIXME: using num_executions is a hack. The name collides with
        #        a stateful param of every component and thus it's not supported
        def myFunc(variable, params, context, num_executions):
            return variable * 2
        U = UserDefinedFunction(custom_function=myFunc, default_variable=[[0, 0], [0, 0]], num_executions=0)
        A = TransferMechanism(name="composition-pytests-A",
                              default_variable=[[1.0, 2.0], [3.0, 4.0]],
                              function=U)
        inputs = {A: [[10., 20.], [30., 40.]]}
        comp.add_node(A)

        res = comp.run(inputs=inputs, execution_mode=mode)
        assert np.allclose(res, [[20.0, 40.0], [60.0, 80.0]])

    def test_get_output_values_prop(self):
        A = pnl.ProcessingMechanism()
        c = pnl.Composition()
        c.add_node(A)
        result = c.run(inputs={A: [1]}, num_trials=2)
        assert result == c.output_values == [np.array([1])]


class TestAuxComponents:
    def test_two_transfer_mechanisms(self):
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B')

        A.aux_components = [B, MappingProjection(sender=A, receiver=B)]

        comp = Composition(name='composition')
        comp.add_node(A)

        comp.run(inputs={A: [[1.0]]})

        assert np.allclose(B.parameters.value.get(comp), [[1.0]])
        # First Run:
        # Input to A = 1.0 | Output = 1.0
        # Input to B = 1.0 | Output = 1.0

        comp.run(inputs={A: [[2.0]]})
        # Second Run:
        # Input to A = 2.0 | Output = 2.0
        # Input to B = 2.0 | Output = 2.0

        assert np.allclose(B.parameters.value.get(comp), [[2.0]])

    def test_two_transfer_mechanisms_with_feedback_proj(self):
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B')

        A.aux_components = [B, (MappingProjection(sender=A, receiver=B), True)]

        comp = Composition(name='composition')
        comp.add_node(A)

        comp.run(inputs={A: [[1.0]],
                         B: [[2.0]]})

        assert np.allclose(B.parameters.value.get(comp), [[2.0]])
        # First Run:
        # Input to A = 1.0 | Output = 1.0
        # Input to B = 2.0 | Output = 2.0

        comp.run(inputs={A: [[1.0]],
                         B: [[2.0]]})
        # Second Run:
        # Input to A = 1.0 | Output = 1.0
        # Input to B = 2.0 + 1.0 | Output = 3.0

        assert np.allclose(B.parameters.value.get(comp), [[3.0]])

    def test_aux_component_with_required_role(self):
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B')
        C = TransferMechanism(name='C',
                              function=Linear(slope=2.0))

        A.aux_components = [(B, NodeRole.TERMINAL), MappingProjection(sender=A, receiver=B)]

        comp = Composition(name='composition')
        comp.add_node(A)
        comp.add_linear_processing_pathway([B, C])

        comp.run(inputs={A: [[1.0]]})

        assert np.allclose(B.parameters.value.get(comp), [[1.0]])
        # First Run:
        # Input to A = 1.0 | Output = 1.0
        # Input to B = 1.0 | Output = 1.0

        comp.run(inputs={A: [[2.0]]})
        # Second Run:
        # Input to A = 2.0 | Output = 2.0
        # Input to B = 2.0 | Output = 2.0

        assert np.allclose(B.parameters.value.get(comp), [[2.0]])

        assert B in comp.get_nodes_by_role(NodeRole.TERMINAL)
        assert np.allclose(C.parameters.value.get(comp), [[4.0]])
        assert np.allclose(comp.get_output_values(comp), [[2.0], [4.0]])

    def test_stateful_nodes(self):
        A = TransferMechanism(name='A')
        B1 = TransferMechanism(name='B1',
                               integrator_mode=True)
        B2 = IntegratorMechanism(name='B2')
        C = TransferMechanism(name='C')


        inner_composition1 = Composition(name="inner-composition-1")
        inner_composition1.add_linear_processing_pathway([A, B1])

        inner_composition2 = Composition(name="inner-composition2")
        inner_composition2.add_linear_processing_pathway([A, B2])

        outer_composition1 = Composition(name="outer-composition-1")
        outer_composition1.add_node(inner_composition1)
        outer_composition1.add_node(C)
        outer_composition1.add_projection(sender=inner_composition1, receiver=C)

        outer_composition2 = Composition(name="outer-composition-2")
        outer_composition2.add_node(inner_composition2)
        outer_composition2.add_node(C)
        outer_composition2.add_projection(sender=inner_composition2, receiver=C)

        expected_stateful_nodes = {inner_composition1: [B1],
                                   inner_composition2: [B2],
                                   outer_composition1: [inner_composition1],
                                   outer_composition2: [inner_composition2]}

        for comp in expected_stateful_nodes:
            assert comp.stateful_nodes == expected_stateful_nodes[comp]


class TestShadowInputs:

    def test_two_origins(self):
        comp = Composition(name='comp')
        A = ProcessingMechanism(name='A')
        comp.add_node(A)
        B = ProcessingMechanism(name='B',
                                input_ports=[A.input_port])

        comp.add_node(B)
        comp.run(inputs={A: [[1.0]]})

        assert A.value == [[1.0]]
        assert B.value == [[1.0]]

        # Since B is both an INPUT Node and also shadows A, it should have two afferent Projections,
        #   one from it own OutputPort of the Composition's input_CIM, and another from the one for A
        # assert len(B.path_afferents)==2
        # assert B.input_port.path_afferents[1].sender is A.input_port.path_afferents[0].sender

        C = ProcessingMechanism(name='C')
        comp.add_linear_processing_pathway([C, A])

        comp.run(inputs={C: 1.5})
        assert A.value == [[1.5]]
        assert B.value == [[1.5]]
        assert C.value == [[1.5]]

        # Since B is shadowing A, its old projection from the CIM should be deleted,
        # and a new projection from C should be added
        assert len(B.path_afferents) == 1
        assert B.path_afferents[0].sender.owner == C

    def test_shadow_internal_projectionstest_two_origins_two_input_ports(self):
        comp = Composition(name='comp')
        A = ProcessingMechanism(name='A',
                                function=Linear(slope=2.0))
        B = ProcessingMechanism(name='B',
                                input_ports=[A.input_port, A.output_port])
        comp.add_node(A)
        comp.add_node(B)
        comp.run(inputs={A: [[1.0]]})

        assert A.value == [[2.0]]
        assert np.allclose(B.value, [[1.0], [2.0]])

        assert len(B.input_ports)==2
        assert len(B.input_ports[0].path_afferents)==1
        assert len(B.input_ports[1].path_afferents)==1
        assert B.input_ports[0].path_afferents[0].sender is A.input_ports[0].path_afferents[0].sender
        assert B.input_ports[1].path_afferents[0].sender is A.output_ports[0]

        C = ProcessingMechanism(name='C')
        comp.add_linear_processing_pathway([C, A])

        comp.run(inputs={C: 1.5})
        assert A.value == [[3.0]]
        assert C.value == [[1.5]]
        assert np.allclose(B.value, [[1.5], [3.0]])

        # Since B is shadowing A, its old projection from the CIM should be deleted,
        # and a new projection from C should be added
        assert len(B.path_afferents) == 2
        for proj in B.path_afferents:
            assert proj.sender.owner in {A, C}

    def test_shadow_internal_projections(self):
        comp = Composition(name='comp')

        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C',
                                input_ports=[B.input_port])

        comp.add_linear_processing_pathway([A, B])
        comp.add_node(C)
        comp.run(inputs={A: [[1.0]]})
        assert A.value == [[1.0]]
        assert B.value == [[1.0]]
        assert C.value == [[1.0]]

        input_nodes = comp.get_nodes_by_role(NodeRole.INPUT)
        output_nodes = comp.get_nodes_by_role(NodeRole.OUTPUT)
        assert A in input_nodes
        assert B in output_nodes
        assert C not in input_nodes
        assert C in output_nodes
        A2 = ProcessingMechanism(name='A2')
        comp.add_linear_processing_pathway([A2, B])
        comp.run(inputs={A: [[1.0]],
                         A2: [[1.0]]})
        assert A.value == [[1.0]]
        assert A2.value == [[1.0]]
        assert B.value == [[2.0]]
        assert C.value == [[2.0]]

    _test_shadow_nested_nodes_arg =\
        [
            ('shadow_nodes_one_and_two_levels_deep', 0),
            ('shadow_nested_internal_node', 1),
         ],

    @pytest.mark.parametrize(
        'condition',
        ['shadow_nodes_one_and_two_levels_deep',
         'shadow_nested_internal_node'],
    )
    def test_shadow_nested_nodes(self, condition):

        I = ProcessingMechanism(name='I')
        icomp = Composition(nodes=I, name='INNER COMP')

        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        mcomp = Composition(pathways=[[A,B,C],icomp], name='MIDDLE COMP')

        if condition == 'shadow_nodes_one_and_two_levels_deep':

            # Confirm that B's shadow of I comes from the same ocomp_input_CIM that serves I
            O = ProcessingMechanism(name='O',input_ports=[I.input_port, A.input_port])
            ocomp = Composition(nodes=[mcomp,O], name='OUTER COMP')
            ocomp._analyze_graph()
            assert len(O.afferents)==2
            assert O.input_ports[0].shadow_inputs.owner is I
            receiver = icomp.input_CIM.port_map[I.input_port][0]
            receiver = receiver.path_afferents[0].sender.owner.port_map[receiver][0]
            assert O.input_ports[0].path_afferents[0].sender is \
                   ocomp.input_CIM.port_map[receiver][1]

            # Confirm that B's shadow of A comes from the same ocomp_input_CIM that serves A
            assert O.input_ports[1].shadow_inputs.owner is A
            assert O.input_ports[1].path_afferents[0].sender is \
                   mcomp.input_CIM.port_map[A.input_port][0].path_afferents[0].sender

        elif condition == 'shadow_nested_internal_node':
            with pytest.raises(CompositionError) as err:
                O = ProcessingMechanism(name='O',input_ports=[B.input_port])
                ocomp = Composition(nodes=[mcomp,O], name='OUTER COMP')
            assert 'Attempt to shadow the input to a node (B) in a nested Composition of OUTER COMP ' \
                   'that is not an INPUT Node of that Composition is not currently supported.' \
                   in err.value.error_value

    def test_failure_to_find_node_to_shadow(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C', input_ports=[A.input_port])
        D = ProcessingMechanism(name='D')
        icomp = Composition(name='iCOMP', nodes=[A, B])
        comp1 = Composition(name='COMP_1', pathways=[icomp])
        err_msg = "Unable to find port specified to be shadowed by 'C' (A[InputPort-0]) within the same Composition"
        with pytest.raises(CompositionError) as error_value:
            comp2 = Composition(name='COMP_2', pathways=[C, D])
            ocomp = Composition(name='OUTER COMPOSITION',pathways=[comp1, comp2])
        assert err_msg in str(error_value)

    def test_monitor_input_ports(self):
        comp = Composition(name='comp')

        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')

        obj = ObjectiveMechanism(name='A_input_plus_B_input',
                                 monitor=[A.input_port, B.input_port],
                                 function=LinearCombination())

        comp.add_node(A)
        comp.add_node(B)
        comp.add_node(obj)

        comp.run(inputs={A: 10.0,
                         B: 15.0})
        assert obj.value == [[25.0]]


class TestInitialize:

    def test_initialize_cycle_values(self):
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B')
        C = RecurrentTransferMechanism(name='C',
                                       auto=1.0)

        abc_Composition = Composition(pathways=[[A, B, C]])

        abc_Composition.run(inputs={A: [1.0, 2.0, 3.0]},
                            initialize_cycle_values={C: 2.0})

        abc_Composition.run(inputs={A: [1.0, 2.0, 3.0]})

        # Run 1 --> Execution 1: 1 + 2 = 3    |    Execution 2: 3 + 2 = 5    |    Execution 3: 5 + 3 = 8
        # Run 2 --> Execution 1: 8 + 1 = 9    |    Execution 2: 9 + 2 = 11    |    Execution 3: 11 + 3 = 14
        assert abc_Composition.results == [[[3]], [[5]], [[8]], [[9]], [[11]], [[14]]]

    def test_initialize_cycle_values_warning(self):
        A = ProcessingMechanism(name='A')
        a_Composition = Composition(name='a_Composition',
                                    pathways=[[A]])
        err = f"A value is specified for {A.name} of {a_Composition.name} in the 'initialize_cycle_values' " \
              f"argument of call to run, but it is neither part of a cycle nor a FEEDBACK_SENDER. " \
              f"Its value will be overwritten when the node first executes, and therefore not used."
        with pytest.warns(UserWarning, match=err):
            a_Composition.run(inputs={A:[1]},
                              initialize_cycle_values={A:[1]})

    @pytest.mark.parametrize("context_specified", [True, False])
    def test_initialize_cycles(self, context_specified):
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B')
        C = RecurrentTransferMechanism(name='C',
                                       auto=1.0)

        context = Context(execution_id='a') if context_specified else None

        abc_Composition = Composition(pathways=[[A, B, C]])

        abc_Composition.initialize({C: 2.0}, context=context)

        abc_Composition.run(inputs={A: [1.0, 2.0, 3.0]}, context=context)

        if not context_specified:
            abc_Composition.run(context=Context(execution_id='b'))

        abc_Composition.run(inputs={A: [1.0, 2.0, 3.0]}, context=context)

        # Run 1 --> Execution 1: 1 + 2 = 3    |    Execution 2: 3 + 2 = 5    |    Execution 3: 5 + 3 = 8
        # Run 2 --> Execution 1: 8 + 1 = 9    |    Execution 2: 9 + 2 = 11    |    Execution 3: 11 + 3 = 14
        assert abc_Composition.results == [[[3]], [[5]], [[8]], [[9]], [[11]], [[14]]]

    def test_initialize_cycles_excluding_unspecified_nodes(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')

        comp = Composition(pathways=[A, B, C, A])
        comp.run({A: 1, B: 1, C: 1})
        context = comp.most_recent_context

        assert A.parameters.value._get(context) == 1
        assert B.parameters.value._get(context) == 1
        assert C.parameters.value._get(context) == 1

        # ALL: value of preceding node + value from input CIM == 0 + 1 == 1

        # initialize B to 0
        comp.initialize({B: 0}, include_unspecified_nodes=False)

        assert A.parameters.value._get(context) == 1
        assert B.parameters.value._get(context) == 0
        assert C.parameters.value._get(context) == 1

        comp.run({A: 0, B: 0, C: 0})

        assert A.parameters.value._get(context) == 1
        assert B.parameters.value._get(context) == 1
        assert C.parameters.value._get(context) == 0

        # A and B: value of preceding node + value from input CIM == 1 + 0 == 1
        # C: value of preceding node + value from input CIM == 0 + 0 == 0

    def test_initialize_cycles_using_default_keyword(self):
        A = ProcessingMechanism(name='A', default_variable=1)
        B = ProcessingMechanism(name='B', default_variable=1)
        C = ProcessingMechanism(name='C', default_variable=1)

        comp = Composition(pathways=[A, B, C, A])
        comp.run({A: 1, B: 1, C: 1})
        context = comp.most_recent_context

        assert A.parameters.value._get(context) == 2
        assert B.parameters.value._get(context) == 2
        assert C.parameters.value._get(context) == 2

        # initialize all nodes to their default values
        comp.initialize({A: DEFAULT, B: DEFAULT, C: DEFAULT})

        assert A.parameters.value._get(context) == 1
        assert B.parameters.value._get(context) == 1
        assert C.parameters.value._get(context) == 1

    def test_initialize_cycles_error(self):
        a = ProcessingMechanism(name='mech_a')
        b = ProcessingMechanism(name='mech_b')
        comp = Composition(nodes=[b])
        error_text = (
            f"{a.name} [(]entry in initialize values arg[)] is not a node in '{comp.name}'"
        )
        with pytest.raises(CompositionError, match=error_text):
            comp.initialize({a: 1})

    def test_initialize_cycles_warning(self):
        a = ProcessingMechanism(name='mech_a')
        comp = Composition(nodes=[a])
        warning_text = (
            f"A value is specified for {a.name} of {comp.name} in the 'initialize_cycle_values' "
            f"argument of call to run, but it is neither part of a cycle nor a FEEDBACK_SENDER. "
            f"Its value will be overwritten when the node first executes, and therefore not used."
        )
        with pytest.warns(UserWarning, match=warning_text):
            comp.run(initialize_cycle_values={a: 1})


class TestResetValues:

    def test_reset_one_mechanism_through_run(self):
        A = TransferMechanism(name='A')
        B = TransferMechanism(
            name='B',
            integrator_mode=True,
            integration_rate=0.5
        )
        C = TransferMechanism(name='C')

        comp = Composition()
        comp.add_linear_processing_pathway([A, B, C])

        C.log.set_log_conditions('value')

        comp.run(
            inputs={A: [1.0]},
            num_trials=5,
            reset_stateful_functions_when=AtTimeStep(0)
        )

        # Trial 0: 0.5, Trial 1: 0.75, Trial 2: 0.5, Trial 3: 0.75. Trial 4: 0.875
        assert np.allclose(
            C.log.nparray_dictionary('value')[comp.default_execution_id]['value'],
            [
                [np.array([0.5])],
                [np.array([0.5])],
                [np.array([0.5])],
                [np.array([0.5])],
                [np.array([0.5])]
            ]
        )

    def test_reset_one_mechanism_at_trial_2_condition(self):
        A = TransferMechanism(name='A')
        B = TransferMechanism(
            name='B',
            integrator_mode=True,
            integration_rate=0.5
        )
        C = TransferMechanism(name='C')

        comp = Composition()
        comp.add_linear_processing_pathway([A, B, C])

        # Set reinitialization condition
        B.reset_stateful_function_when = AtTrial(2)

        C.log.set_log_conditions('value')

        comp.run(
            inputs={A: [1.0]},
            reset_stateful_functions_to={B: [0.]},
            num_trials=5
        )

        # Trial 0: 0.5, Trial 1: 0.75, Trial 2: 0.5, Trial 3: 0.75. Trial 4: 0.875
        assert np.allclose(
            C.log.nparray_dictionary('value')[comp.default_execution_id]['value'],
            [
                [np.array([0.5])],
                [np.array([0.75])],
                [np.array([0.5])],
                [np.array([0.75])],
                [np.array([0.875])]
            ]
        )

    def test_reset_two_mechanisms_at_different_trials_with_dict(self):
        A = TransferMechanism(
            name='A',
            integrator_mode=True,
            integration_rate=0.5
        )
        B = TransferMechanism(
            name='B',
            integrator_mode=True,
            integration_rate=0.5
        )
        C = TransferMechanism(name='C')

        comp = Composition(
            pathways=[[A, C], [B, C]]
        )

        A.log.set_log_conditions('value')
        B.log.set_log_conditions('value')
        C.log.set_log_conditions('value')

        comp.run(
            inputs={A: [1.0],
                    B: [1.0]},
            reset_stateful_functions_when = {
                A: AtTrial(1),
                B: AtTrial(2)
            },
            num_trials=5
        )

        # Mechanisms A and B should have their original reset_integrator_when
        # Conditions after the call to run has completed
        assert isinstance(A.reset_stateful_function_when, Never)
        assert isinstance(B.reset_stateful_function_when, Never)

        # Mechanism A - resets on Trial 1
        # Trial 0: 0.5, Trial 1: 0.5, Trial 2: 0.75, Trial 3: 0.875, Trial 4: 0.9375
        assert np.allclose(
            A.log.nparray_dictionary('value')[comp.default_execution_id]['value'],
            [
                [np.array([0.5])],
                [np.array([0.5])],
                [np.array([0.75])],
                [np.array([0.875])],
                [np.array([0.9375])]
            ]
        )

        # Mechanism B - resets on Trial 2
        # Trial 0: 0.5, Trial 1: 0.75, Trial 2: 0.5, Trial 3: 0.75. Trial 4: 0.875
        assert np.allclose(
            B.log.nparray_dictionary('value')[comp.default_execution_id]['value'],
            [
                [np.array([0.5])],
                [np.array([0.75])],
                [np.array([0.5])],
                [np.array([0.75])],
                [np.array([0.875])]
            ]
        )

        # Mechanism C - sum of A and B
        # Trial 0: 1.0, Trial 1: 1.25, Trial 2: 1.25, Trial 3: 1.625, Trial 4: 1.8125
        assert np.allclose(
            C.log.nparray_dictionary('value')[comp.default_execution_id]['value'],
            [
                [np.array([1.0])],
                [np.array([1.25])],
                [np.array([1.25])],
                [np.array([1.625])],
                [np.array([1.8125])]
            ]
        )

    def test_save_state_before_simulations(self):

        A = TransferMechanism(
            name='A',
            integrator_mode=True,
            integration_rate=0.2
        )

        B = IntegratorMechanism(name='B', function=DriftDiffusionIntegrator(rate=0.1, time_step_size=1.0))
        C = TransferMechanism(name='C')

        comp = Composition()
        comp.add_linear_processing_pathway([A, B, C])

        comp.run(inputs={A: [[1.0], [1.0]]})

        run_1_values = [
            A.parameters.value.get(comp),
            B.parameters.value.get(comp)[0],
            C.parameters.value.get(comp)
        ]

        # "Save state" code from EVCaux

        # Get any values that need to be reset for each run
        reinitialization_values = {}
        for mechanism in comp.stateful_nodes:
            # "save" the current state of each stateful mechanism by storing the values of each of its stateful
            # attributes in the reinitialization_values dictionary; this gets passed into run and used to call
            # the reset method on each stateful mechanism.
            reinitialization_value = {}

            if isinstance(mechanism.function, IntegratorFunction):
                for attr in mechanism.function.stateful_attributes:
                    reinitialization_value[attr] = getattr(mechanism.function.parameters, attr).get(comp)
            elif hasattr(mechanism, "integrator_function"):
                if isinstance(mechanism.integrator_function, IntegratorFunction):
                    for attr in mechanism.integrator_function.stateful_attributes:
                        reinitialization_value[attr] = getattr(mechanism.integrator_function.parameters, attr).get(comp)

            reinitialization_values[mechanism] = reinitialization_value

        # Allow values to continue accumulating so that we can set them back to the saved state
        comp.run(inputs={A: [[1.0], [1.0]]})

        run_2_values = [A.parameters.value.get(comp),
                        B.parameters.value.get(comp)[0],
                        C.parameters.value.get(comp)]

        comp.run(
            inputs={A: [[1.0], [1.0]]},
            reset_stateful_functions_to=reinitialization_values
        )

        run_3_values = [A.parameters.value.get(comp),
                        B.parameters.value.get(comp)[0],
                        C.parameters.value.get(comp)]

        assert np.allclose(np.asfarray(run_2_values),
                           np.asfarray(run_3_values))
        assert np.allclose(np.asfarray(run_1_values),
                           [np.array([0.36]), np.array([0.056]), np.array([0.056])])
        assert np.allclose(np.asfarray(run_2_values),
                           [np.array([0.5904]), np.array([0.16384]), np.array([0.16384])])


class TestNodeRoles:

    def test_INPUT_and_OUTPUT_and_SINGLETON(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        comp = Composition(pathways=[[A],[B,C]], name='comp')
        comp._analyze_graph()

        assert set(comp.get_nodes_by_role(NodeRole.INPUT)) == {A,B}
        assert set(comp.get_nodes_by_role(NodeRole.OUTPUT)) == {A,C}
        assert set(comp.get_nodes_by_role(NodeRole.SINGLETON)) == {A}

    def test_INTERNAL(self):
        comp = Composition(name='comp')
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        comp.add_linear_processing_pathway([A, B, C])

        assert comp.get_nodes_by_role(NodeRole.INTERNAL) == [B]

    def test_input_labels_and_results_by_node_and_no_orphaning_of_nested_output_nodes(self):
        """
        Test that nested Composition with two outputs, one of which Projects to a node in the outer Composition is,
        by virtue of its other output, still assigned as an OUTPUT Node of the outer Composition
        Also test get_input_format and get_results_by_nodes methods
        """
        input_labels_dict = {0:{'red':0, 'green':1}}
        output_labels_dict = {0:{'red':0, 'green':1}}
        A = ProcessingMechanism(name='A', input_labels=input_labels_dict)
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        icomp = Composition(pathways=[[A,B,C]], name='INNER COMP')

        X = ProcessingMechanism(name='X')
        Y = ProcessingMechanism(name='Y')
        Z = ProcessingMechanism(name='Z', output_labels=output_labels_dict)
        mcomp = Composition(pathways=[[X,Y,Z],icomp], name='MIDDLE COMP')

        Q = ProcessingMechanism(name='Q', input_labels=input_labels_dict)
        O = ProcessingMechanism(name='O', input_ports=[Z])
        ocomp = Composition(name='OUTER COMP', nodes=[O, mcomp,Q])

        len(ocomp.output_values)==3
        result = ocomp.run(inputs={mcomp:[[0],[0]]})
        assert len(result)==4

        input_format = ocomp.get_input_format(form=pnl.TEXT)
        assert repr(input_format) == '\'{\\n\\tMIDDLE COMP: [[0.0],[0.0]],\\n\\tQ: [[0.0]]\\n}\''
        input_format = ocomp.get_input_format(form=pnl.TEXT, num_trials=3, use_labels=True)
        assert repr(input_format) == '"{\\n\\tMIDDLE COMP: [ [[0.0],[\'red\']], [[0.0],[\'green\']], [[0.0],[\'red\']] ],\\n\\tQ: [ [\'red\'], [\'green\'], [\'red\'] ]\\n}"'
        input_format = ocomp.get_input_format(form=pnl.TEXT, num_trials=2, show_nested_input_nodes=True)
        assert input_format == '\nInputs to (nested) INPUT Nodes of OUTER COMP for 2 trials:\n\tMIDDLE COMP: \n\t\tX: [ [[0.0]], [[0.0]] ]\n\t\tINNER COMP: \n\t\t\tA: [ [[0.0]], [[0.0]] ]\n\tQ: [ [[0.0]], [[0.0]] \n\nFormat as follows for inputs to run():\n{\n\tMIDDLE COMP: [ [[0.0],[0.0]], [[0.0],[0.0]] ],\n\tQ: [ [[0.0]], [[0.0]] ]\n}'
        input_format = ocomp.get_input_format(form=pnl.TEXT, num_trials=2, show_nested_input_nodes=True, use_labels=True)
        assert input_format == "\nInputs to (nested) INPUT Nodes of OUTER COMP for 2 trials:\n\tMIDDLE COMP: \n\t\tX: [ [[0.0]], [[0.0]] ]\n\t\tINNER COMP: \n\t\t\tA: [ ['red'], ['green'] ]\n\tQ: [ ['red'], ['green'] \n\nFormat as follows for inputs to run():\n{\n\tMIDDLE COMP: [ [[0.0],[0.0]], [[0.0],[0.0]] ],\n\tQ: [ [[0.0]], [[0.0]] ]\n}"

        result = ocomp.run(inputs={mcomp:[[.2],['green']], Q:[4.6]})
        assert result == [[0.2], [0.2], [1.],[4.6]]
        results_by_node = ocomp.get_results_by_nodes()
        assert results_by_node[O] == [0.2]
        assert results_by_node[Z] == [0.2]
        assert results_by_node[C] == [1.0]
        assert results_by_node[Q] == [4.6]
        results_by_node = ocomp.get_results_by_nodes(use_names=True)
        assert repr(results_by_node) == '{\'O\': [0.2], \'Z\': [0.2], \'C\': [1.0], \'Q\': [4.6]}'
        results_by_node = ocomp.get_results_by_nodes(use_names=True, use_labels=True)
        assert repr(results_by_node) == '{\'O\': [[0.2]], \'Z\': [\'red\'], \'C\': [[1.0]], \'Q\': [[4.6]]}'
        results_by_node = ocomp.get_results_by_nodes(nodes=[Q, Z])
        assert repr(results_by_node) == '{(ProcessingMechanism Z): [0.2], (ProcessingMechanism Q): [4.6]}'
        results_by_node = ocomp.get_results_by_nodes(nodes=Q, use_names=True)
        assert repr(results_by_node) == '{\'Q\': [4.6]}'
        results_by_node = ocomp.get_results_by_nodes(nodes=Z, use_labels=True)
        assert repr(results_by_node) == '{(ProcessingMechanism Z): [\'red\']}'

        label_not_in_dict_error_msg = '"Inappropriate use of \'purple\' as a stimulus for A in MIDDLE COMP: ' \
                                      'it is not a label in its input_labels_dict."'
        with pytest.raises(CompositionError) as error_text:
            ocomp.run(inputs={mcomp:[[0],['purple']],Q:['red']})
        assert label_not_in_dict_error_msg in str(error_text.value)

        no_label_dict_error_msg = '"Inappropriate use of str (\'red\') as a stimulus for X in MIDDLE COMP: ' \
                                  'it does not have an input_labels_dict."'
        with pytest.raises(CompositionError) as error_text:
            ocomp.run(inputs={mcomp:[['red'],['red']],Q:['red']})
        assert no_label_dict_error_msg in str(error_text.value)

        no_such_node_error_msg = '"Nodes specified in get_results_by_nodes() method not found in OUTER COMP ' \
                                 'nor any Compositions nested within it: [\'N\']"'
        with pytest.raises(CompositionError) as error_text:
            ocomp.get_results_by_nodes(nodes=['N'])
        assert no_such_node_error_msg in str(error_text.value)

    def test_unnested_PROBE(self):
        """Assigning PROBE to a Node should add OutputPort to output_CIM even if Node is not in a nested Composition"""
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        comp = Composition(pathways=[A,(B, NodeRole.PROBE), C], name='COMP')
        assert B.output_port in comp.output_CIM.port_map

    params = [  # id     allow_probes  include_probes_in_output  err_msg
        (
            "allow_probes_True", True, False, None
         ),
        (
            "allow_probes_True", True, True, None
         ),
        (
            "allow_probes_False", False, False,
            "B found in nested Composition of OUTER COMP (MIDDLE COMP) but without required NodeRole.OUTPUT."
         ),
        (
            "allow_probes_CONTROL", "CONTROL", True,
            "B found in nested Composition of OUTER COMP (MIDDLE COMP) but without required NodeRole.OUTPUT."
         )
    ]
    @pytest.mark.parametrize('id, allow_probes, include_probes_in_output, err_msg', params, ids=[x[0] for x in params])
    def test_nested_PROBES(self, id, allow_probes, include_probes_in_output, err_msg):
        """Test use of allow_probes, include_probes_in_output and orphaned output from nested comp"""

        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        icomp = Composition(pathways=[[A,B,C]], name='INNER COMP')

        X = ProcessingMechanism(name='X')
        Y = ProcessingMechanism(name='Y')
        Z = ProcessingMechanism(name='Z')
        mcomp = Composition(pathways=[[X,Y,Z],icomp], name='MIDDLE COMP')

        O = ProcessingMechanism(name='O',
                                input_ports=[B, Y]
                                )

        if not err_msg:
            ocomp = Composition(name='OUTER COMP',
                                # node=[0,mcomp],   # <- CRASHES
                                nodes=[mcomp,O],
                                allow_probes=allow_probes,
                                include_probes_in_output=include_probes_in_output
                                )
            # ocomp.show_graph(show_cim=True, show_node_structure=True)

            assert B.output_port in icomp.output_CIM.port_map
            # assert B.output_port in mcomp.output_CIM.port_map
            assert Y.output_port in mcomp.output_CIM.port_map
            if include_probes_in_output is False:
                assert len(ocomp.output_values)==4  # Should only be outputs from mcomp (C, Z) and O
                result = ocomp.run(inputs={mcomp:[[0],[0]]})
                assert len(result)==4  # Should only be outputs from mcomp (C, Z) and O
            elif include_probes_in_output is True:
                assert len(ocomp.output_values)==6           # Outputs from mcomp (C and Z) and O (from B and Y)
                result = ocomp.run(inputs={mcomp:[[0],[0]]}) # This tests that outputs from mcomp (C and Z) are included
                assert len(result)==6                        # even though mcomp also projects to O (for B and Y)
        else:
            with pytest.raises(CompositionError) as err:
                ocomp = Composition(name='OUTER COMP',
                                nodes=[mcomp,O],
                                allow_probes=allow_probes,
                                include_probes_in_output=include_probes_in_output)
                ocomp._analyze_graph()
            assert err.value.error_value == err_msg

    def test_two_node_cycle(self):
        A = TransferMechanism()
        B = TransferMechanism()
        comp = Composition(pathways=[A, B, A])
        assert set(comp.get_nodes_by_role(NodeRole.ORIGIN))=={A,B}
        assert set(comp.get_nodes_by_role(NodeRole.TERMINAL))=={A,B}
        assert set(comp.get_nodes_by_role(NodeRole.CYCLE))=={A,B}
        # # THE FOLLOWING FAIL:
        # assert set(comp.get_nodes_by_role(NodeRole.FEEDBACK_RECEIVER))=={A}
        # assert set(comp.get_nodes_by_role(NodeRole.FEEDBACK_SENDER))=={B}

    def test_three_node_cycle(self):
        A = TransferMechanism(name='MECH A')
        B = TransferMechanism(name='MECH B')
        C = TransferMechanism(name='MECH C')
        comp = Composition(pathways=[A, B, C])
        comp.add_projection(sender=C, receiver=A)
        comp._analyze_graph()
        assert set(comp.get_nodes_by_role(NodeRole.CYCLE)) == {A,B,C}
        # Test that order of output_CIM.output ports follows order of Nodes in self.nodes
        assert 'MECH A' in comp.output_CIM.output_ports.names[0]
        assert 'MECH B' in comp.output_CIM.output_ports.names[1]
        assert 'MECH C' in comp.output_CIM.output_ports.names[2]
        # THE FOLLOWING PASS:
        assert not set(comp.get_nodes_by_role(NodeRole.FEEDBACK_SENDER))
        assert not set(comp.get_nodes_by_role(NodeRole.FEEDBACK_RECEIVER))

    def test_three_node_cycle_with_FEEDBACK(self):
        A = TransferMechanism()
        B = TransferMechanism()
        C = TransferMechanism()
        comp = Composition(pathways=[A, B, C])
        comp.add_projection(sender=C, receiver=A, feedback=True)
        comp._analyze_graph()
        assert not set(comp.get_nodes_by_role(NodeRole.CYCLE))
        assert set(comp.get_nodes_by_role(NodeRole.FEEDBACK_SENDER)) == {C}
        assert set(comp.get_nodes_by_role(NodeRole.INTERNAL)) == {B}
        assert set(comp.get_nodes_by_role(NodeRole.FEEDBACK_RECEIVER)) == {A}

    def test_branch(self):
        a = TransferMechanism(default_variable=[0, 0])
        b = TransferMechanism()
        c = TransferMechanism()
        d = TransferMechanism()
        comp = Composition(pathways=[[a, b, c], [a, b, d]])
        comp.run(inputs={a: [[2, 2]]})
        assert set(comp.get_nodes_by_role(NodeRole.ORIGIN))=={a}
        assert set(comp.get_nodes_by_role(NodeRole.INTERNAL))=={b}
        assert set(comp.get_nodes_by_role(NodeRole.TERMINAL))=={c,d}

    def test_bypass(self):
        a = TransferMechanism(default_variable=[0, 0])
        b = TransferMechanism(default_variable=[0, 0])
        c = TransferMechanism()
        d = TransferMechanism()
        comp = Composition(pathways=[[a, b, c, d],[a, b, d]])
        comp.run(inputs = {a: [[2, 2], [0, 0]]})
        assert set(comp.get_nodes_by_role(NodeRole.ORIGIN))=={a}
        assert set(comp.get_nodes_by_role(NodeRole.INTERNAL))=={b,c}
        assert set(comp.get_nodes_by_role(NodeRole.TERMINAL))=={d}

    def test_chain(self):
        a = TransferMechanism(default_variable=[0, 0, 0])
        b = TransferMechanism()
        c = TransferMechanism()
        d = TransferMechanism()
        e = TransferMechanism()
        comp = Composition(pathways=[[a, b, c],[c, d, e]])
        comp.run(inputs = {a: [[2, 2, 2], [0, 0, 0]]})
        assert set(comp.get_nodes_by_role(NodeRole.ORIGIN))=={a}
        assert set(comp.get_nodes_by_role(NodeRole.INTERNAL))=={b,c,d}
        assert set(comp.get_nodes_by_role(NodeRole.TERMINAL))=={e}

    def test_convergent(self):
        a = TransferMechanism(default_variable=[0, 0])
        b = TransferMechanism()
        c = TransferMechanism()
        c = TransferMechanism(default_variable=[0])
        d = TransferMechanism()
        e = TransferMechanism()
        comp = Composition(pathways=[[a, b, e],[c, d, e]])
        comp.run(inputs = {a: [[2, 2]], c: [[0]]})
        assert set(comp.get_nodes_by_role(NodeRole.ORIGIN))=={a,c}
        assert set(comp.get_nodes_by_role(NodeRole.INTERNAL))=={b,d}
        assert set(comp.get_nodes_by_role(NodeRole.TERMINAL))=={e}

    def test_one_pathway_cycle(self):
        a = TransferMechanism(default_variable=[0, 0])
        b = TransferMechanism(default_variable=[0, 0])
        comp = Composition(pathways=[a, b, a])
        comp.run(inputs={a: [1, 1]})
        assert set(comp.get_nodes_by_role(NodeRole.ORIGIN))=={a,b}
        assert set(comp.get_nodes_by_role(NodeRole.TERMINAL))=={a,b}
        assert set(comp.get_nodes_by_role(NodeRole.CYCLE))=={a,b}
        # FIX 4/25/20 [JDC]: THESE SHOULD BE OK:
        # assert set(comp.get_nodes_by_role(NodeRole.FEEDBACK_RECEIVER))=={a}
        # assert set(comp.get_nodes_by_role(NodeRole.FEEDBACK_SENDER))=={b}

    def test_two_pathway_cycle(self):
        a = TransferMechanism(default_variable=[0, 0])
        b = TransferMechanism(default_variable=[0, 0])
        c = TransferMechanism(default_variable=[0, 0])
        comp = Composition(pathways=[[a, b, a],[a, c, a]])
        comp.run(inputs={a: [1, 1]})
        assert set(comp.get_nodes_by_role(NodeRole.ORIGIN))=={a,b,c}
        assert set(comp.get_nodes_by_role(NodeRole.TERMINAL))=={a,b,c}
        assert set(comp.get_nodes_by_role(NodeRole.CYCLE))=={a,b,c}

    def test_extended_loop(self):
        a = TransferMechanism(default_variable=[0, 0])
        b = TransferMechanism()
        c = TransferMechanism()
        d = TransferMechanism()
        e = TransferMechanism(default_variable=[0])
        f = TransferMechanism()
        comp = Composition(pathways=[[a, b, c, d],[e, c, f, b, d]])
        comp.run(inputs={a: [2, 2], e: [0]})
        assert set(comp.get_nodes_by_role(NodeRole.ORIGIN))=={a,e}
        assert set(comp.get_nodes_by_role(NodeRole.TERMINAL))=={d}
        assert set(comp.get_nodes_by_role(NodeRole.CYCLE))=={b,c,f}

    def test_two_node_cycle(self):
        A = TransferMechanism()
        B = TransferMechanism()
        comp = Composition(pathways=[A, B, A])
        assert set(comp.get_nodes_by_role(NodeRole.ORIGIN))=={A,B}
        assert set(comp.get_nodes_by_role(NodeRole.TERMINAL))=={A,B}
        assert set(comp.get_nodes_by_role(NodeRole.CYCLE))=={A,B}
        # # THE FOLLOWING FAIL:
        # assert set(comp.get_nodes_by_role(NodeRole.FEEDBACK_RECEIVER))=={A}
        # assert set(comp.get_nodes_by_role(NodeRole.FEEDBACK_SENDER))=={B}

    def test_three_node_cycle(self):
        A = TransferMechanism()
        B = TransferMechanism()
        C = TransferMechanism()
        comp = Composition(pathways=[A, B, C])
        comp.add_projection(sender=C, receiver=A)
        comp._analyze_graph()
        assert set(comp.get_nodes_by_role(NodeRole.CYCLE)) == {A,B,C}
        assert not set(comp.get_nodes_by_role(NodeRole.FEEDBACK_SENDER))
        result = comp.run(inputs={A:[3]})
        assert True

    def test_three_node_cycle_with_FEEDBACK(self):
        A = TransferMechanism()
        B = TransferMechanism()
        C = TransferMechanism()
        comp = Composition(pathways=[A, B, C])
        comp.add_projection(sender=C, receiver=A, feedback=True)
        comp._analyze_graph()
        assert not set(comp.get_nodes_by_role(NodeRole.CYCLE))
        assert set(comp.get_nodes_by_role(NodeRole.FEEDBACK_SENDER)) == {C}
        assert set(comp.get_nodes_by_role(NodeRole.INTERNAL)) == {B}
        assert set(comp.get_nodes_by_role(NodeRole.FEEDBACK_RECEIVER)) == {A}
        result = comp.run(inputs={A:[3]})
        assert True

    def test_FEEDBACK_no_CYCLE(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        comp = Composition(pathways=[A, B, C])
        comp.add_projection(sender=C, receiver=A, feedback=True)
        comp._analyze_graph()

        assert comp.get_nodes_by_role(NodeRole.FEEDBACK_SENDER) == [C]
        assert comp.get_nodes_by_role(NodeRole.FEEDBACK_RECEIVER) == [A]
        assert comp.get_nodes_by_role(NodeRole.INPUT) == [A]
        assert comp.get_nodes_by_role(NodeRole.INTERNAL) == [B]
        assert comp.get_nodes_by_role(NodeRole.OUTPUT) == [C]
        assert not comp.get_nodes_by_role(NodeRole.CYCLE)

    def test_CYCLE_no_FEEDBACK(self):
        comp = Composition(name='comp')
        A = ProcessingMechanism(name='MECH A')
        B = ProcessingMechanism(name='MECH B')
        C = ProcessingMechanism(name='MECH C')
        comp.add_linear_processing_pathway([A, B, C])
        comp.add_projection(sender=C, receiver=A)
        comp._analyze_graph()

        # Test that order of output_CIM.output ports follows order of Nodes in self.nodes
        assert 'MECH A' in comp.output_CIM.output_ports.names[0]
        assert 'MECH B' in comp.output_CIM.output_ports.names[1]
        assert 'MECH C' in comp.output_CIM.output_ports.names[2]
        assert set(comp.get_nodes_by_role(NodeRole.CYCLE)) == {A,B,C}
        assert not set(comp.get_nodes_by_role(NodeRole.FEEDBACK_SENDER))
        assert not set(comp.get_nodes_by_role(NodeRole.FEEDBACK_RECEIVER))
        assert set(comp.get_nodes_by_role(NodeRole.SINGLETON)) == {A,B,C}

    def test_CYCLE_in_pathway_spec_no_FEEDBACK(self):
        comp = Composition(name='comp')
        A = ProcessingMechanism(name='MECH A')
        B = ProcessingMechanism(name='MECH B')
        C = ProcessingMechanism(name='MECH C')
        comp.add_linear_processing_pathway([A, B, C, A])
        comp._analyze_graph()

        # Test that order of output_CIM.output ports follows order of Nodes in self.nodes
        assert 'MECH A' in comp.output_CIM.output_ports.names[0]
        assert 'MECH B' in comp.output_CIM.output_ports.names[1]
        assert 'MECH C' in comp.output_CIM.output_ports.names[2]
        assert set(comp.get_nodes_by_role(NodeRole.CYCLE)) == {A,B,C}
        assert not set(comp.get_nodes_by_role(NodeRole.FEEDBACK_SENDER))
        assert not set(comp.get_nodes_by_role(NodeRole.FEEDBACK_RECEIVER))
        assert set(comp.get_nodes_by_role(NodeRole.SINGLETON)) == {A,B,C}

    # def test_CYCLE_no_FEEDBACK(self):
    #     comp = Composition(name='comp')
    #     A = ProcessingMechanism(name='A')
    #     B = ProcessingMechanism(name='B')
    #     C = ProcessingMechanism(name='C')
    #     comp.add_linear_processing_pathway([A, B, C])
    #     comp.add_projection(sender=C, receiver=A)
    #     comp._analyze_graph()
    #
    #     assert set(comp.get_nodes_by_role(NodeRole.CYCLE)) == {A,B,C}
    #     assert not set(comp.get_nodes_by_role(NodeRole.FEEDBACK_SENDER))
    #     assert not set(comp.get_nodes_by_role(NodeRole.FEEDBACK_RECEIVER))
    #     assert set(comp.get_nodes_by_role(NodeRole.SINGLETON)) == {A,B,C}

    # def test_CONTROL_OBJECTIVE(self):
    #     pass
    #
    # def test_CONTROLLER_OBJECTIVE(self):
    #     pass

    def test_OUTPUT_asymmetric_with_learning_short_first(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        c = Composition(pathways=[[A], {'LEARNING_PATHWAY':([B,C], BackPropagation)}])
        assert {A,C} == set(c.get_nodes_by_role(NodeRole.OUTPUT))

    def test_OUTPUT_asymmetric_with_learning_short_last(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        c = Composition(pathways=[{'LEARNING_PATHWAY':([B,C], BackPropagation)},[A]])
        assert {A,C} == set(c.get_nodes_by_role(NodeRole.OUTPUT))

    def test_OUTPUT_required_node_roles_override(self):
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B', function=Linear(slope=2.0))
        comp = Composition(name='composition')
        comp.add_node(A, required_roles=[NodeRole.OUTPUT])
        comp.add_linear_processing_pathway([A, B])
        result = comp.run(inputs={A: [[1.0]]})
        output_mechanisms = comp.get_nodes_by_role(NodeRole.OUTPUT)
        assert set(output_mechanisms) == {A,B}
        assert np.allclose(result, [[1.0],[2.0]])

    def test_OUTPUT_required_node_roles_both(self):
        A = TransferMechanism(name='A')
        B = TransferMechanism(name='B', function=Linear(slope=2.0))
        comp = Composition(name='composition')
        comp.add_node(A, required_roles=[NodeRole.OUTPUT])
        comp.add_linear_processing_pathway([A, (B, NodeRole.OUTPUT)])
        result = comp.run(inputs={A: [[1.0]]})
        terminal_mechanisms = comp.get_nodes_by_role(NodeRole.OUTPUT)
        assert A in terminal_mechanisms and B in terminal_mechanisms
        assert np.allclose(result, [[1.0],[2.0]])

    def test_exclude_control_mechanisms_as_OUTPUT(self):
        mech = ProcessingMechanism(name='my_mech')
        ctl_mech_A = ControlMechanism(monitor_for_control=mech,
                                      control_signals=ControlSignal(modulates=(INTERCEPT,mech),
                                                                    cost_options=CostFunctions.INTENSITY))
        ctl_mech_B = ControlMechanism(monitor_for_control=mech,
                                      control_signals=ControlSignal(modulates=ctl_mech_A.control_signals[0],
                                                                    modulation=INTENSITY_COST_FCT_MULTIPLICATIVE_PARAM))
        comp = Composition(pathways=[mech, ctl_mech_A, ctl_mech_B])
        # mech (and not either ControlMechanism) should be the OUTPUT Nodd
        assert {mech} == set(comp.get_nodes_by_role(NodeRole.OUTPUT))
        # # There should be only one TERMINAL node (one -- but not both -- of the ControlMechanisms
        # assert len(comp.get_nodes_by_role(NodeRole.TERMINAL)) == 1
        assert isinstance(list(comp.get_nodes_by_role(NodeRole.TERMINAL))[0], ControlMechanism)

    def test_force_one_control_mechanisms_as_OUTPUT(self):
        mech = ProcessingMechanism(name='my_mech')
        ctl_mech_A = ControlMechanism(monitor_for_control=mech,
                                      control_signals=ControlSignal(modulates=(INTERCEPT,mech),
                                                                    cost_options=CostFunctions.INTENSITY))
        ctl_mech_B = ControlMechanism(monitor_for_control=mech,
                                      control_signals=ControlSignal(modulates=ctl_mech_A.control_signals[0],
                                                                    modulation=INTENSITY_COST_FCT_MULTIPLICATIVE_PARAM))
        comp = Composition(pathways=[mech, (ctl_mech_A, NodeRole.OUTPUT), ctl_mech_B])
        assert {mech, ctl_mech_A} == set(comp.get_nodes_by_role(NodeRole.OUTPUT))
        # Current instantiation always assigns ctl_mech_A as TERMINAL (presumably since it was forced to be OUTPUT);
        # However, ctl_mech_B might also be;  depends on where feedback was assigned?
        assert ctl_mech_A in set(comp.get_nodes_by_role(NodeRole.TERMINAL))

    def test_force_two_control_mechanisms_as_OUTPUT(self):
        mech = ProcessingMechanism(name='my_mech')
        ctl_mech_A = ControlMechanism(monitor_for_control=mech,
                                      control_signals=ControlSignal(modulates=(INTERCEPT,mech),
                                                                    cost_options=CostFunctions.INTENSITY))
        ctl_mech_B = ControlMechanism(monitor_for_control=mech,
                                      control_signals=ControlSignal(modulates=ctl_mech_A.control_signals[0],
                                                                    modulation=INTENSITY_COST_FCT_MULTIPLICATIVE_PARAM))
        comp = Composition(pathways=[mech, (ctl_mech_A, NodeRole.OUTPUT), (ctl_mech_B, NodeRole.OUTPUT)])
        assert {mech, ctl_mech_A, ctl_mech_B} == set(comp.get_nodes_by_role(NodeRole.OUTPUT))
        # Current instantiation always assigns ctl_mech_B as TERMINAL in this case;
        # this is here to flag any violation of this in the future, in case that is not intended
        assert {ctl_mech_B} == set(comp.get_nodes_by_role(NodeRole.TERMINAL))

    def test_LEARNING_hebbian(self):
        A = RecurrentTransferMechanism(name='A', size=2, enable_learning=True)
        comp = Composition(pathways=A)
        pathway = comp.pathways[0]
        assert pathway.target is None
        assert pathway.learning_objective is None
        assert pathway.learning_components == {}
        roles = {NodeRole.INPUT, NodeRole.CYCLE, NodeRole.OUTPUT
            # , NodeRole.FEEDBACK_RECEIVER
                 }
        assert roles.issubset(set(comp.get_roles_by_node(A)))
        assert set(comp.get_nodes_by_role(NodeRole.LEARNING)) == {A.learning_mechanism}

    def test_LEARNING_rl(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        comp = Composition(pathways=([A,B], Reinforcement))
        learning_pathway = comp.pathways[0]
        target = learning_pathway.target
        objective= learning_pathway.learning_objective
        learning_mech = learning_pathway.learning_components[LEARNING_MECHANISMS]
        learning = {learning_mech}
        learning.add(target)
        learning.add(objective)
        assert set(comp.get_nodes_by_role(NodeRole.INPUT)) == {A, target}
        assert set(comp.get_nodes_by_role(NodeRole.OUTPUT)) == {B}
        assert set(comp.get_nodes_by_role(NodeRole.LEARNING)) == learning
        assert set(comp.get_nodes_by_role(NodeRole.LEARNING_OBJECTIVE)) == {objective}
        # Validate that objective projects to LearningMechanism (allowed to have other user-assigned Projections)
        assert any([isinstance(proj.receiver.owner, LearningMechanism) for proj in objective.efferents])
        # Validate that TERMINAL is LearningMechanism that projects to first MappingProjection in learning pathway
        (comp.get_nodes_by_role(NodeRole.TERMINAL))[0].efferents[0].receiver.owner.sender.owner == A

    def test_LEARNING_bp(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')
        comp = Composition(pathways=([A,B,C,D], BackPropagation))
        learning_pathway = comp.pathways[0]
        target = learning_pathway.target
        objective= learning_pathway.learning_objective
        learning_mechs = learning_pathway.learning_components[LEARNING_MECHANISMS]
        learning = set(learning_mechs)
        learning.add(target)
        learning.add(objective)
        assert set(comp.get_nodes_by_role(NodeRole.INPUT)) == {A, target}
        assert set(comp.get_nodes_by_role(NodeRole.OUTPUT)) == {D}
        assert set(comp.get_nodes_by_role(NodeRole.LEARNING)) == set(learning)
        assert set(comp.get_nodes_by_role(NodeRole.LEARNING_OBJECTIVE)) == {objective}
        # Validate that objective projects to LearningMechanism  (allowed to have other user-assigned Projections)
        assert any([isinstance(proj.receiver.owner, LearningMechanism) for proj in objective.efferents])
        # Validate that TERMINAL is LearningMechanism that Projects to first MappingProjection in learning_pathway
        (comp.get_nodes_by_role(NodeRole.TERMINAL))[0].efferents[0].receiver.owner.sender.owner == A

    @pytest.mark.control
    def test_controller_role(self):
        comp = Composition()
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        comp.add_linear_processing_pathway([A, B])
        comp.add_controller(
            controller=pnl.OptimizationControlMechanism(
                agent_rep=comp,
                state_features=[A.input_port],
                objective_mechanism=pnl.ObjectiveMechanism(
                    function=pnl.LinearCombination(
                        operation=pnl.PRODUCT),
                    monitor=[A]
                ),
                function=pnl.GridSearch(),
                control_signals=[
                    {
                        PROJECTIONS: ("slope", B),
                        ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                    }
                ]
            )
        )

        assert comp.get_nodes_by_role(NodeRole.CONTROLLER) == [comp.controller]
        assert comp.nodes_to_roles[comp.controller] == {NodeRole.CONTROLLER}

    def test_inactive_terminal_projection(self):
        A = pnl.ProcessingMechanism(name='A')
        B = pnl.ProcessingMechanism(name='B')
        C = pnl.ProcessingMechanism(name='C')
        D = pnl.ProcessingMechanism(name='D')

        pnl.MappingProjection(sender=A, receiver=D)
        comp = pnl.Composition([[A],[B,C]])

        assert comp.nodes_to_roles[A] == {NodeRole.INPUT, NodeRole.OUTPUT, NodeRole.SINGLETON, NodeRole.ORIGIN, NodeRole.TERMINAL}

    def test_feedback_projection_added_by_pathway(self):
        A = pnl.ProcessingMechanism(name='A')
        B = pnl.ProcessingMechanism(name='B')
        C = pnl.ProcessingMechanism(name='C')

        icomp = pnl.Composition(pathways=[C])
        ocomp = pnl.Composition(pathways=[A, icomp, (B, pnl.NodeRole.FEEDBACK_SENDER), A])

        assert ocomp.nodes_to_roles == {
            A: {NodeRole.ORIGIN, NodeRole.INPUT, NodeRole.FEEDBACK_RECEIVER},
            icomp: {NodeRole.INTERNAL},
            B: {NodeRole.TERMINAL, NodeRole.OUTPUT, NodeRole.FEEDBACK_SENDER},
        }


class TestMisc:

    def test_disable_all_history(self):
        comp = Composition(name='comp')
        A = ProcessingMechanism(name='A')

        comp.add_node(A)
        comp.disable_all_history()
        comp.run(inputs={A: [2]})

        assert len(A.parameters.value.history[comp.default_execution_id]) == 0

    def test_danglingControlledMech(self):
        #
        #   first section is from Stroop Demo
        #
        Color_Input = TransferMechanism(
            name='Color Input',
            function=Linear(slope=0.2995)
        )
        Word_Input = TransferMechanism(
            name='Word Input',
            function=Linear(slope=0.2995)
        )

        # Processing Mechanisms (Control)
        Color_Hidden = TransferMechanism(
            name='Colors Hidden',
            function=Logistic(gain=(1.0, pnl.ControlProjection)),
        )
        Word_Hidden = TransferMechanism(
            name='Words Hidden',
            function=Logistic(gain=(1.0, pnl.ControlProjection)),
        )
        Output = TransferMechanism(
            name='Output',
            function=Logistic(gain=(1.0, pnl.ControlProjection)),
        )

        # Decision Mechanisms
        Decision = pnl.DDM(
            function=pnl.DriftDiffusionAnalytical(
                drift_rate=(1.0),
                threshold=(0.1654),
                noise=(0.5),
                starting_value=(0),
                non_decision_time=0.25,
            ),
            name='Decision',
        )
        # Outcome Mechanisms:
        Reward = TransferMechanism(name='Reward')

        # add another DDM but do not add to system
        second_DDM = pnl.DDM(
            function=pnl.DriftDiffusionAnalytical(
                drift_rate=(
                    1.0,
                    pnl.ControlProjection(
                        function=Linear,
                        control_signal_params={
                            ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                        },
                    ),
                ),
                threshold=(
                    1.0,
                    pnl.ControlProjection(
                        function=Linear,
                        control_signal_params={
                            ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)
                        },
                    ),
                ),
                noise=(0.5),
                starting_value=(0),
                non_decision_time=0.45
            ),
            name='second_DDM',
        )

        comp = Composition(enable_controller=True)
        comp.add_linear_processing_pathway([
            Color_Input,
            Color_Hidden,
            Output,
            Decision
        ])
        comp.add_linear_processing_pathway([
            Word_Input,
            Word_Hidden,
            Output,
            Decision
        ])
        comp.add_node(Reward)
        # no assert, should only complete without error

    @pytest.mark.parametrize(
        'removed_nodes, expected_dependencies',
        [
            (['A'], {'B': set(), 'C': set('B'), 'D': set('C'), 'E': set('C')}),
            (['C'], {'A': set(), 'B': set(), 'D': set(), 'E': set()}),
            (['E'], {'A': set(), 'B': set(), 'C': {'A', 'B'}, 'D': set('C')}),
            (['A', 'B'], {'C': set(), 'D': set('C'), 'E': set('C')}),
            (['D', 'E'], {'A': set(), 'B': set(), 'C': {'A', 'B'}}),
            (['A', 'B', 'C', 'D', 'E'], {}),
        ]
    )
    def test_remove_node(self, removed_nodes, expected_dependencies):
        def stringify_dependency_dict(dd):
            return {node.name: {n.name for n in deps} for node, deps in dd.items()}

        A = pnl.TransferMechanism(name='A')
        B = pnl.TransferMechanism(name='B')
        C = pnl.TransferMechanism(name='C')
        D = pnl.TransferMechanism(name='D')
        E = pnl.TransferMechanism(name='E')

        locs = locals()
        removed_nodes = [locs[n] for n in removed_nodes]

        comp = pnl.Composition(
            pathways=[
                [A, C, D],
                [A, C, D],
                [B, C, D],
                [B, C, E],
            ]
        )

        comp.remove_nodes(removed_nodes)

        assert stringify_dependency_dict(comp.scheduler.dependency_dict) == expected_dependencies
        assert stringify_dependency_dict(comp.graph_processing.dependency_dict) == expected_dependencies

        proj_dependencies = collections.defaultdict(set)
        for node in comp.nodes:
            if node not in removed_nodes:
                for proj in node.afferents:
                    if (
                        proj.sender.owner not in removed_nodes
                        and proj.receiver.owner not in removed_nodes
                        and not isinstance(proj.sender.owner, CompositionInterfaceMechanism)
                        and not isinstance(proj.receiver.owner, CompositionInterfaceMechanism)
                    ):
                        proj_dependencies[node.name].add(proj.name)
                        proj_dependencies[proj.name].add(proj.sender.owner.name)
        assert stringify_dependency_dict(comp.graph.dependency_dict) == {**expected_dependencies, **proj_dependencies}

        for node in removed_nodes:
            assert node not in comp.nodes
            assert node not in comp.nodes_to_roles
            assert node not in comp.graph.comp_to_vertex
            assert node not in comp.graph_processing.comp_to_vertex
            assert node not in comp.scheduler.conditions

            for proj in node.afferents + node.efferents:
                assert proj not in comp.projections
                assert not proj.is_active_in_composition(comp)

        comp.run(inputs={n: [0] for n in comp.get_nodes_by_role(pnl.NodeRole.INPUT)})

    @pytest.mark.parametrize('slope_A', [1, (1, pnl.CONTROL)])
    @pytest.mark.parametrize('slope_B', [1, (1, pnl.CONTROL)])
    @pytest.mark.parametrize('removed_nodes', [['A'], ['B'], ['A', 'B']])
    def test_remove_node_control(self, slope_A, slope_B, removed_nodes):
        A = pnl.TransferMechanism(name='A', function=pnl.Linear(slope=slope_A))
        B = pnl.TransferMechanism(name='B', function=pnl.Linear(slope=slope_B))

        locs = locals()
        removed_nodes = [locs[n] for n in removed_nodes]

        search_space_len = sum(1 if isinstance(s, tuple) else 0 for s in [slope_A, slope_B])

        comp = pnl.Composition(pathways=[A, B])
        comp.add_controller(
            pnl.OptimizationControlMechanism(
                agent_rep=comp, search_space=[0, 1] * search_space_len
            )
        )
        comp.remove_nodes(removed_nodes)

        for n in removed_nodes:
            for proj in n.parameter_ports['slope'].all_afferents:
                assert not proj.is_active_in_composition(comp), f'{n.name} {proj.name}'

    def test_remove_node_from_conditions(self):
        def assert_conditions_do_not_contain(*args):
            conds_queue = list(comp.scheduler.conditions.conditions.values())
            while len(conds_queue) > 0:
                cur_cond = conds_queue.pop()
                deps = []

                try:
                    deps = [cur_cond.dependency]
                except AttributeError:
                    try:
                        deps = cur_cond.dependencies
                    except AttributeError:
                        pass

                for d in deps:
                    if isinstance(d, pnl.Condition):
                        conds_queue.append(d)
                    assert d not in args

        A = pnl.TransferMechanism(name='A')
        B = pnl.TransferMechanism(name='B')
        C = pnl.TransferMechanism(name='C')
        D = pnl.TransferMechanism(name='D')

        comp = pnl.Composition(pathways=[[A, D], [B, D], [C, D]])

        comp.run(inputs={A: [0], B: [0], C: [0]})
        comp.remove_node(A)
        comp.run(inputs={B: [0], C: [0]})
        assert_conditions_do_not_contain(A)

        comp.remove_node(B)
        comp.run(inputs={C: [0]})
        assert_conditions_do_not_contain(A, B)

    def test_remove_node_learning(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')
        D = ProcessingMechanism(name='D')

        comp = Composition()
        comp.add_linear_learning_pathway(pathway=[A, B], learning_function=BackPropagation)
        comp.add_linear_learning_pathway(pathway=[C, D], learning_function=Reinforcement)

        comp.remove_node(A)
        comp.learn(inputs={n: [0] for n in comp.get_nodes_by_role(pnl.NodeRole.INPUT)})

        comp.remove_node(D)
        comp.learn(inputs={n: [0] for n in comp.get_nodes_by_role(pnl.NodeRole.INPUT)})

    def test_rebuild_scheduler_after_add_node(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')

        comp = Composition(pathways=[A, C])

        comp.scheduler.add_condition(C, pnl.EveryNCalls(A, 2))
        comp.add_node(B)
        comp.scheduler.add_condition(B, pnl.EveryNCalls(A, 2))

        comp.run(inputs={A: [0], B: [0]})

        assert type(comp.scheduler.conditions[A]) is pnl.Always
        assert type(comp.scheduler.conditions[B]) is pnl.EveryNCalls
        assert comp.scheduler.conditions[B].args == (A, 2)

        assert type(comp.scheduler.conditions[C]) is pnl.EveryNCalls
        assert comp.scheduler.conditions[C].args == (A, 2)

        assert comp.scheduler.execution_list[comp.default_execution_id] == [{A}, {A, B}, {C}]
        assert set(comp.scheduler._user_specified_conds.keys()) == {B, C}

    def test_rebuild_scheduler_after_remove_node(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C')

        comp = Composition(pathways=[[A, C], [B, C]])

        comp.scheduler.add_condition(C, pnl.EveryNCalls(A, 2))
        comp.remove_node(B)

        comp.run(inputs={A: [0]})

        assert type(comp.scheduler.conditions[A]) is pnl.Always
        assert B not in comp.scheduler.conditions
        assert type(comp.scheduler.conditions[C]) is pnl.EveryNCalls
        assert comp.scheduler.conditions[C].args == (A, 2)

        assert comp.scheduler.execution_list[comp.default_execution_id] == [{A}, {A}, {C}]
        assert set(comp.scheduler._user_specified_conds.keys()) == {C}


class TestInputSpecsDocumentationExamples:

    @pytest.mark.parametrize(
        "variable_a, num_trials, inputs, expected_inputs", [
            # "If num_trials is in use, run will iterate over the inputs
            # until num_trials is reached. For example, if five inputs
            # are provided for each ORIGIN mechanism, and
            # num_trials = 7, the system will execute seven times. The
            # first two items in the list of inputs will be used on the
            # 6th and 7th trials, respectively."
            pytest.param(
                None,
                7,
                [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]],
                [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]], [[1.0]], [[2.0]]],
                id='example_2'
            ),
            # Origin mechanism has only one InputPort
            # COMPLETE specification
            pytest.param(
                None,
                None,
                [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.1]]],
                [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.1]]],
                id='example_3'
            ),
            # Origin mechanism has only one InputPort
            # SHORTCUT: drop the outer list on each input because 'a'
            # only has one InputPort
            pytest.param(
                None,
                None,
                [[1.0], [2.0], [3.0], [4.0], [5.2]],
                [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.2]]],
                id='example_4'
            ),
            # Origin mechanism has only one InputPort
            # SHORTCUT: drop the remaining list on each input because
            # 'a' only has one element
            pytest.param(
                None,
                None,
                [1.0, 2.0, 3.0, 4.0, 5.3],
                [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.3]]],
                id='example_5'
            ),
            # Only one input is provided for the mechanism
            # [single trial]
            # COMPLETE input specification
            pytest.param(
                [[0.0], [0.0]],
                None,
                [[[1.0], [2.0]]],
                [[[1.0], [2.0]]],
                id='example_6'
            ),
            # Only one input is provided for the mechanism
            # [single trial]
            # SHORTCUT: Remove outer list because we only have one trial
            pytest.param(
                [[0.0], [0.0]],
                None,
                [[1.0], [2.0]],
                [[[1.0], [2.0]]],
                id='example_7'
            ),
            # Only one input is provided for the mechanism [repeat]
            # COMPLETE SPECIFICATION
            pytest.param(
                [[0.0], [0.0]],
                None,
                [
                    [[1.0], [2.0]],
                    [[1.0], [2.0]],
                    [[1.0], [2.0]],
                    [[1.0], [2.0]],
                    [[1.0], [2.0]]
                ],
                [
                    [[1.0], [2.0]],
                    [[1.0], [2.0]],
                    [[1.0], [2.0]],
                    [[1.0], [2.0]],
                    [[1.0], [2.0]]
                ],
                id='example_8'
            ),
            # Only one input is provided for the mechanism [REPEAT]
            # SHORTCUT: Remove outer list because we want to use the
            # same input on every trial
            pytest.param(
                [[0.0], [0.0]],
                5,
                [[1.0], [2.0]],
                [
                    [[1.0], [2.0]],
                    [[1.0], [2.0]],
                    [[1.0], [2.0]],
                    [[1.0], [2.0]],
                    [[1.0], [2.0]]
                ],
                id='example_9'
            ),
            # There is only one origin mechanism in the system
            # COMPLETE SPECIFICATION
            pytest.param(
                [[1.0, 2.0, 3.0]],
                None,
                [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
                [[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]],
                id='example_10',
                marks=pytest.mark.xfail(
                    reason='System version used np.allclose for inputs'
                    + ' comparison, resulting in hiding the failure of this'
                    + ' test. (resulting inputs are only 2d, instead of 3d)'
                )
            )
        ]
    )
    def test_documentation_example_two_mechs(
        self,
        variable_a,
        num_trials,
        inputs,
        expected_inputs
    ):
        a = pnl.TransferMechanism(name='a', default_variable=variable_a)
        b = pnl.TransferMechanism(name='b')

        comp = pnl.Composition()
        comp.add_linear_processing_pathway([a, b])

        check_inputs = []

        def store_inputs():
            check_inputs.append(a.get_input_values(comp))

        comp.run(
            inputs={a: inputs},
            num_trials=num_trials,
            call_after_trial=store_inputs
        )

        assert check_inputs == expected_inputs

    def test_example_1(self):
        # "If num_trials is not in use, the number of inputs provided
        # determines the number of trials in the run. For example, if
        # five inputs are provided for each origin mechanism, and
        # num_trials is not specified, the system will execute five
        # times."

        import psyneulink as pnl

        a = pnl.TransferMechanism(name="a", default_variable=[[0.0, 0.0]])
        b = pnl.TransferMechanism(name="b", default_variable=[[0.0], [0.0]])
        c = pnl.TransferMechanism(name="c")

        comp = pnl.Composition()
        comp.add_linear_processing_pathway([a, c])
        comp.add_linear_processing_pathway([b, c])

        input_dictionary = {
            a: [[[1.0, 1.0]], [[1.0, 1.0]]],
            b: [[[2.0], [3.0]], [[2.0], [3.0]]],
        }

        check_inputs_dictionary = {a: [], b: []}

        def store_inputs():
            check_inputs_dictionary[a].append(a.get_input_values(comp))
            check_inputs_dictionary[b].append(b.get_input_values(comp))

        comp.run(inputs=input_dictionary, call_after_trial=store_inputs)

        for mech in input_dictionary:
            assert np.allclose(
                check_inputs_dictionary[mech], input_dictionary[mech]
            )

    def test_example_11(self):
        # There is only one origin mechanism in the system
        # SHORT CUT - specify inputs as a list instead of a dictionary

        a = pnl.TransferMechanism(name='a', default_variable=[[1.0, 2.0, 3.0]])
        b = pnl.TransferMechanism(name='b')

        comp = pnl.Composition()
        comp.add_linear_processing_pathway([a, b])

        check_inputs = []

        def store_inputs():
            check_inputs.append(a.get_input_values(comp))

        input_list = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]

        comp.run(
            inputs=input_list,
            call_after_trial=store_inputs
        )

        assert np.allclose(check_inputs, [[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]])
