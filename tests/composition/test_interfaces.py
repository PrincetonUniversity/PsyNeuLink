import numpy as np
import pytest

import psyneulink.core.llvm as pnlvm

from psyneulink.core.components.functions.transferfunctions import Identity, Linear
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.modulatory.control.optimizationcontrolmechanism import OptimizationControlMechanism
from psyneulink.core.components.ports.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.compositions.composition import Composition, CompositionError, RunError
from psyneulink.core.scheduling.scheduler import Scheduler
from psyneulink.core.globals.utilities import convert_all_elements_to_np_array
from psyneulink.core.globals.keywords import INTERCEPT, NOISE, SLOPE


class TestExecuteCIM:

    def test_identity_function(self):

        I = Identity()

        output = I.execute(2.0)
        assert output == 2.0

        output = I.execute([1.0, 2.0,3.0])
        assert np.allclose([1.0, 2.0,3.0], output)

        output = I.execute([[1.0, 2.0], [3.0]])

        assert np.allclose([1.0, 2.0], output[0])
        assert np.allclose([3.0], output[1])

    def test_standalone_CIM(self):

        cim = CompositionInterfaceMechanism()
        cim.execute(2.0)
        assert np.allclose(cim.value, [2.0])

    def test_assign_value(self):
        cim = CompositionInterfaceMechanism()
        cim.defaults.variable = [2.0]
        cim.execute()
        assert np.allclose(cim.value, [2.0])

    def test_standalone_CIM_multiple_input_ports(self):

        cim = CompositionInterfaceMechanism(default_variable=[[0.0], [0.0], [0.0]])
        cim.execute([[1.0], [2.0], [3.0]])
        assert np.allclose(cim.value, [[1.0], [2.0], [3.0]])

    def test_standalone_processing_multiple_input_ports(self):

        processing_mech = ProcessingMechanism(default_variable=[[0.0], [0.0], [0.0]])
        processing_mech.execute([[1.0], [2.0], [3.0]])
        assert np.allclose(processing_mech.value, [[1.0], [2.0], [3.0]])


    def test_one_input_port_one_output_port(self):

        comp = Composition()

        A = TransferMechanism(name="A",
                              function=Linear(slope=2.0))

        B = TransferMechanism(name="B",
                              function=Linear(slope=3.0))

        comp.add_node(A)
        comp.add_node(B)

        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)

        inputs_dict = {
            A: [[5.]],
        }
        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs_dict,
            scheduler=sched
        )

        assert np.allclose([30], output)

    def test_two_input_ports_two_output_ports(self):

        comp = Composition()

        A = TransferMechanism(name="A",
                              default_variable=[[0.0], [0.0]],
                              function=Linear(slope=2.0))

        B = TransferMechanism(name="B",
                              default_variable=[[0.0], [0.0]],
                              function=Linear(slope=3.0))

        comp.add_node(A)
        comp.add_node(B)

        comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        comp.add_projection(MappingProjection(sender=A.output_ports[1], receiver=B.input_ports[1]), A, B)

        inputs_dict = {
            A: [[5.], [6.]],
        }
        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs_dict,
            scheduler=sched
        )

        assert np.allclose([[30.], [36.]], output)


        # assert np.allclose([30.], comp.output_CIM.output_ports[1].value)
        # assert np.allclose([36.], comp.output_CIM.output_ports[2].value)

class TestConnectCompositionsViaCIMS:

    @pytest.mark.nested
    @pytest.mark.composition
    @pytest.mark.parametrize("mode", ['Python',
                             pytest.param('LLVM', marks=pytest.mark.llvm),
                             pytest.param('LLVMExec', marks=pytest.mark.llvm),
                             pytest.param('LLVMRun', marks=pytest.mark.llvm),
                             pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                             pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])
                             ])
    def test_connect_compositions_with_simple_states(self, mode):

        comp1 = Composition(name="first_composition")

        A = TransferMechanism(name="A",
                              function=Linear(slope=2.0))

        B = TransferMechanism(name="B",
                              function=Linear(slope=3.0))

        comp1.add_node(A)
        comp1.add_node(B)

        comp1.add_projection(MappingProjection(sender=A, receiver=B), A, B)

        inputs_dict = {
            A: [[5.]],
        }


        sched = Scheduler(composition=comp1)

        comp2 = Composition(name="second_composition")

        A2 = TransferMechanism(name="A2",
                              function=Linear(slope=2.0))

        B2 = TransferMechanism(name="B2",
                              function=Linear(slope=3.0))

        comp2.add_node(A2)
        comp2.add_node(B2)

        comp2.add_projection(MappingProjection(sender=A2, receiver=B2), A2, B2)

        sched = Scheduler(composition=comp2)

        comp3 = Composition(name="outer_composition")
        comp3.add_node(comp1)
        comp3.add_node(comp2)
        comp3.add_projection(MappingProjection(), comp1, comp2)

        # comp1:
        # input = 5.0
        # mechA: 2.0*5.0 = 10.0
        # mechB: 3.0*10.0 = 30.0
        # output = 30.0

        # comp2:
        # input = 30.0
        # mechA2: 2.0*30.0 = 60.0
        # mechB2: 3.0*60.0 = 180.0
        # output = 180.0

        # comp3:
        # input = 5.0
        # output = 180.0
        res = comp3.run(inputs={comp1: [[5.]]}, bin_execute=mode)
        assert np.allclose(res, [[[180.0]]])
        if mode == 'Python':
            assert np.allclose(comp1.output_port.parameters.value.get(comp3), [30.0])
            assert np.allclose(comp2.output_port.parameters.value.get(comp3), [180.0])
            assert np.allclose(comp3.output_port.parameters.value.get(comp3), [180.0])

    @pytest.mark.nested
    @pytest.mark.composition
    @pytest.mark.parametrize("mode", ['Python',
                             pytest.param('LLVM', marks=pytest.mark.llvm),
                             pytest.param('LLVMExec', marks=pytest.mark.llvm),
                             pytest.param('LLVMRun', marks=pytest.mark.llvm),
                             pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                             pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])
                             ])
    def test_connect_compositions_with_complicated_states(self, mode):

        inner_composition_1 = Composition(name="comp1")

        A = TransferMechanism(name="A1",
                              default_variable=[[0.0], [0.0]],
                              function=Linear(slope=2.0))

        B = TransferMechanism(name="B1",
                              default_variable=[[0.0], [0.0]],
                              function=Linear(slope=3.0))

        inner_composition_1.add_node(A)
        inner_composition_1.add_node(B)

        inner_composition_1.add_projection(MappingProjection(sender=A, receiver=B), A, B)
        inner_composition_1.add_projection(MappingProjection(sender=A.output_ports[1], receiver=B.input_ports[1]), A,
                                           B)

        inner_composition_2 = Composition(name="comp2")

        A2 = TransferMechanism(name="A2",
                              default_variable=[[0.0], [0.0]],
                              function=Linear(slope=2.0))

        B2 = TransferMechanism(name="B2",
                              default_variable=[[0.0], [0.0]],
                              function=Linear(slope=3.0))

        inner_composition_2.add_node(A2)
        inner_composition_2.add_node(B2)

        inner_composition_2.add_projection(MappingProjection(sender=A2, receiver=B2), A2, B2)
        inner_composition_2.add_projection(MappingProjection(sender=A2.output_ports[1], receiver=B2.input_ports[1]),
                                           A2, B2)

        outer_composition = Composition(name="outer_composition")

        outer_composition.add_node(inner_composition_1)
        outer_composition.add_node(inner_composition_2)

        outer_composition.add_projection(projection=MappingProjection(), sender=inner_composition_1,
                                         receiver=inner_composition_2)
        outer_composition.add_projection(
            projection=MappingProjection(sender=inner_composition_1.output_CIM.output_ports[1],
                                         receiver=inner_composition_2.input_CIM.input_ports[1]),
            sender=inner_composition_1, receiver=inner_composition_2)

        sched = Scheduler(composition=outer_composition)
        output = outer_composition.run(
            inputs={inner_composition_1: [[[5.0], [50.0]]]},
            scheduler=sched,
            bin_execute=mode
        )

        assert np.allclose(output, [[[180.], [1800.]]])
        if mode == 'Python':
            assert np.allclose(inner_composition_1.get_output_values(outer_composition), [[30.], [300.]])
            assert np.allclose(inner_composition_2.get_output_values(outer_composition), [[180.], [1800.]])
            assert np.allclose(outer_composition.get_output_values(outer_composition), [[180.], [1800.]])

    @pytest.mark.nested
    @pytest.mark.composition
    @pytest.mark.parametrize("mode", ['Python',
                             pytest.param('LLVM', marks=pytest.mark.llvm),
                             pytest.param('LLVMExec', marks=pytest.mark.llvm),
                             pytest.param('LLVMRun', marks=pytest.mark.llvm),
                             pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                             pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])
                             ])
    def test_compositions_as_origin_nodes(self, mode):

        inner_composition_1 = Composition(name="inner_composition_1")

        A = TransferMechanism(name="A",
                              function=Linear(slope=0.5))

        B = TransferMechanism(name="B",
                              function=Linear(slope=2.0))

        C = TransferMechanism(name="C",
                              function=Linear(slope=3.0))

        inner_composition_1.add_node(A)
        inner_composition_1.add_node(B)
        inner_composition_1.add_node(C)

        inner_composition_1.add_projection(MappingProjection(), A, C)
        inner_composition_1.add_projection(MappingProjection(), B, C)

        inner_composition_2 = Composition(name="inner_composition_2")

        A2 = TransferMechanism(name="A2",
                               function=Linear(slope=0.25))

        B2 = TransferMechanism(name="B2",
                               function=Linear(slope=1.0))

        inner_composition_2.add_node(A2)
        inner_composition_2.add_node(B2)

        inner_composition_2.add_projection(MappingProjection(), A2, B2)

        mechanism_d = TransferMechanism(name="D",
                                        function=Linear(slope=3.0))

        outer_composition = Composition(name="outer_composition")

        outer_composition.add_node(inner_composition_1)
        outer_composition.add_node(inner_composition_2)
        outer_composition.add_node(mechanism_d)

        outer_composition.add_projection(projection=MappingProjection(), sender=inner_composition_1,
                                         receiver=mechanism_d)
        outer_composition.add_projection(projection=MappingProjection(), sender=inner_composition_2,
                                         receiver=mechanism_d)

        sched = Scheduler(composition=outer_composition)

        # FIX: order of InputPorts on inner composition 1 is not stable
        output = outer_composition.run(
            inputs={
                # inner_composition_1: [[2.0], [1.0]],
                inner_composition_1: {A: [2.0],
                                          B: [1.0]},
                inner_composition_2: [[12.0]]},
            scheduler=sched,
            bin_execute=mode
        )
        assert np.allclose(output, [[[36.]]])

        if mode == 'Python':
            assert np.allclose(A.get_output_values(outer_composition), [[1.0]])
            assert np.allclose(B.get_output_values(outer_composition), [[2.0]])
            assert np.allclose(C.get_output_values(outer_composition), [[9.0]])
            assert np.allclose(A2.get_output_values(outer_composition), [[3.0]])
            assert np.allclose(B2.get_output_values(outer_composition), [[3.0]])
            assert np.allclose(inner_composition_1.get_output_values(outer_composition), [[9.0]])
            assert np.allclose(inner_composition_2.get_output_values(outer_composition), [[3.0]])
            assert np.allclose(mechanism_d.get_output_values(outer_composition), [[36.0]])
            assert np.allclose(outer_composition.get_output_values(outer_composition), [[36.0]])

    @pytest.mark.nested
    @pytest.mark.composition
    @pytest.mark.parametrize("mode", ['Python',
                             pytest.param('LLVM', marks=pytest.mark.llvm),
                             pytest.param('LLVMExec', marks=pytest.mark.llvm),
                             pytest.param('LLVMRun', marks=pytest.mark.llvm),
                             pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                             pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])
                             ])
    def test_compositions_as_origin_nodes_multiple_trials(self, mode):

        inner_composition_1 = Composition(name="inner_composition_1")

        A = TransferMechanism(name="A",
                              function=Linear(slope=0.5))

        B = TransferMechanism(name="B",
                              function=Linear(slope=2.0))

        C = TransferMechanism(name="C",
                              function=Linear(slope=3.0))

        inner_composition_1.add_node(A)
        inner_composition_1.add_node(B)
        inner_composition_1.add_node(C)

        inner_composition_1.add_projection(MappingProjection(), A, C)
        inner_composition_1.add_projection(MappingProjection(), B, C)

        inner_composition_2 = Composition(name="inner_composition_2")

        A2 = TransferMechanism(name="A2",
                               function=Linear(slope=0.25))

        B2 = TransferMechanism(name="B2",
                               function=Linear(slope=1.0))

        inner_composition_2.add_node(A2)
        inner_composition_2.add_node(B2)

        inner_composition_2.add_projection(MappingProjection(), A2, B2)

        mechanism_d = TransferMechanism(name="D",
                                        function=Linear(slope=3.0))

        outer_composition = Composition(name="outer_composition")

        outer_composition.add_node(inner_composition_1)


        outer_composition.add_node(inner_composition_2)


        outer_composition.add_node(mechanism_d)

        inner_composition_1._analyze_graph()

        outer_composition.add_projection(projection=MappingProjection(), sender=inner_composition_1,
                                         receiver=mechanism_d)



        inner_composition_2._analyze_graph()

        outer_composition.add_projection(projection=MappingProjection(), sender=inner_composition_2,
                                         receiver=mechanism_d)

        sched = Scheduler(composition=outer_composition)

        # FIX: order of InputPorts on inner composition 1 is not stable
        output = outer_composition.run(
            inputs={
                inner_composition_1: {A: [[2.0], [1.5], [2.5]],
                                      B: [[1.0], [1.5], [1.5]]},
                inner_composition_2: [[12.0], [11.5], [12.5]]},
            scheduler=sched,
            bin_execute=mode
        )

        # trial 0:
        # inner composition 1 = (0.5*2.0 + 2.0*1.0) * 3.0 = 9.0
        # inner composition 2 = 0.25*12.0 = 3.0
        # outer composition = (3.0 + 9.0) * 3.0 = 36.0

        # trial 1:
        # inner composition 1 = (0.5*1.5 + 2.0*1.5) * 3.0 = 11.25
        # inner composition 2 = 0.25*11.5 = 2.875
        # outer composition = (2.875 + 11.25) * 3.0 = 42.375

        # trial 2:
        # inner composition 1 = (0.5*2.5 + 2.0*1.5) * 3.0 = 12.75
        # inner composition 2 = 0.25*12.5 = 3.125
        # outer composition = (3.125 + 12.75) * 3.0 = 47.625

        assert np.allclose(output, np.array([47.625]))

    def test_input_specification_multiple_nested_compositions(self):

        # level_0 composition --------------------------------- innermost composition
        level_0 = Composition(name="level_0")

        A0 = TransferMechanism(name="A0",
                               default_variable=[[0.], [0.]],
                               function=Linear(slope=1.))
        B0 = TransferMechanism(name="B0",
                               function=Linear(slope=2.))

        level_0.add_node(A0)
        level_0.add_node(B0)
        level_0.add_projection(MappingProjection(), A0, B0)
        level_0.add_projection(MappingProjection(sender=A0.output_ports[1], receiver=B0), A0, B0)

        # level_1 composition ---------------------------------
        level_1 = Composition(name="level_1")

        A1 = TransferMechanism(name="A1",
                              function=Linear(slope=1.))
        B1 = TransferMechanism(name="B1",
                              function=Linear(slope=2.))

        level_1.add_node(level_0)
        level_1.add_node(A1)
        level_1.add_node(B1)
        level_1.add_projection(MappingProjection(), level_0, B1)
        level_1.add_projection(MappingProjection(), A1, B1)

        # level_2 composition --------------------------------- outermost composition
        level_2 = Composition(name="level_2")

        A2 = TransferMechanism(name="A2",
                               size=2,
                               function=Linear(slope=1.))
        B2 = TransferMechanism(name="B2",
                               function=Linear(slope=2.))

        level_2.add_node(level_1)
        level_2.add_node(A2)
        level_2.add_node(B2)
        level_2.add_projection(MappingProjection(), level_1, B2)
        level_2.add_projection(MappingProjection(), A2, B2)

        sched = Scheduler(composition=level_2)

        # FIX: order of InputPorts in each inner composition (level_0 and level_1)
        level_2.run(inputs={A2: [[1.0, 2.0]],
                            level_1: {A1: [[1.0]],
                                      level_0: {A0: [[1.0], [2.0]]}}},
                    scheduler=sched)

        # level_0 output = 2.0 * (1.0 + 2.0) = 6.0
        assert np.allclose(level_0.get_output_values(level_2), [6.0])
        # level_1 output = 2.0 * (1.0 + 6.0) = 14.0
        assert np.allclose(level_1.get_output_values(level_2), [14.0])
        # level_2 output = 2.0 * (1.0 + 2.0 + 14.0) = 34.0
        assert np.allclose(level_2.get_output_values(level_2), [34.0])

    def test_warning_on_custom_cim_ports(self):

        comp = Composition()
        mech = ProcessingMechanism()
        comp.add_node(mech)
        warning_text = ('You are attempting to add custom ports to a CIM, which can result in unpredictable behavior '
                        'and is therefore recommended against. If suitable, you should instead add ports to the '
                       r'mechanism\(s\) that project to or are projected to from the CIM.')
        with pytest.warns(UserWarning, match=warning_text):
            # KDM 7/22/20: previously was OutputPort, but that produces
            # an invalid CIM state that cannot be executed, and will
            # throw an error due to new _update_default_variable call
            comp.input_CIM.add_ports(InputPort())

        with pytest.warns(None) as w:
            comp._analyze_graph()
            comp.run({mech: [[1]]})

        assert len(w) == 0

    def test_user_added_ports(self):

        comp = Composition()
        mech = ProcessingMechanism()
        comp.add_node(mech)
        # instantiate custom input and output ports
        inp = InputPort(size=2)
        out = OutputPort(size=2)
        # add custom input and output ports to CIM
        comp.input_CIM.add_ports([inp, out])
        # verify the ports have been added to the user_added_ports set
        # and that no extra ports were added
        assert inp in comp.input_CIM.user_added_ports['input_ports']
        assert len(comp.input_CIM.user_added_ports['input_ports']) == 1
        assert out in comp.input_CIM.user_added_ports['output_ports']
        assert len(comp.input_CIM.user_added_ports['output_ports']) == 1
        comp.input_CIM.remove_ports([inp, out])
        # verify that call to remove ports succesfully removed the ports from user_added_ports
        assert len(comp.input_CIM.user_added_ports['input_ports']) == 0
        assert len(comp.input_CIM.user_added_ports['output_ports']) == 0

    def test_parameter_CIM_port_order(self):
        # Note:  CIM_port order is also tested in TestNodes and test_simplified_necker_cube()

        # Inner Composition
        ia = TransferMechanism(name='ia')
        icomp = Composition(name='icomp', pathways=[ia])

        # Outer Composition
        ocomp = Composition(name='ocomp', pathways=[icomp])
        ocm = OptimizationControlMechanism(name='ic',
                                           agent_rep=ocomp,
                                           control_signals=[
                                               ControlSignal(projections=[(NOISE, ia)]),
                                               ControlSignal(projections=[(INTERCEPT, ia)]),
                                               ControlSignal(projections=[(SLOPE, ia)]),
                                           ]
                                           )
        ocomp.add_controller(ocm)

        assert INTERCEPT in icomp.parameter_CIM.output_ports.names[0]
        assert NOISE in icomp.parameter_CIM.output_ports.names[1]
        assert SLOPE in icomp.parameter_CIM.output_ports.names[2]

    def test_parameter_CIM_routing_from_ControlMechanism(self):
        # Inner Composition
        ia = TransferMechanism(name='ia')
        ib = TransferMechanism(name='ib')
        icomp = Composition(name='icomp', pathways=[ia])
        # Outer Composition
        ocomp = Composition(name='ocomp', pathways=[icomp])
        cm = ControlMechanism(
            name='control_mechanism',
            control_signals=
            ControlSignal(projections=[(SLOPE, ib)])
        )
        icomp.add_linear_processing_pathway([ia, ib])
        ocomp.add_linear_processing_pathway([cm, icomp])
        res = ocomp.run([[2], [2], [2]])
        assert np.allclose(res, [[4], [4], [4]])
        assert len(ib.mod_afferents) == 1
        assert ib.mod_afferents[0].sender == icomp.parameter_CIM.output_port
        assert icomp.parameter_CIM_ports[ib.parameter_ports['slope']][0].path_afferents[0].sender == cm.output_port

    def test_nested_control_projection_count_controller(self):
        # Inner Composition
        ia = TransferMechanism(name='ia')
        icomp = Composition(name='icomp', pathways=[ia])
        # Outer Composition
        ocomp = Composition(name='ocomp', pathways=[icomp])
        ocm = OptimizationControlMechanism(name='ocm',
                                           agent_rep=ocomp,
                                           control_signals=[
                                               ControlSignal(projections=[(NOISE, ia)]),
                                               ControlSignal(projections=[(INTERCEPT, ia)]),
                                               ControlSignal(projections=[(SLOPE, ia)]),
                                           ]
                                           )
        ocomp.add_controller(ocm)
        assert len(ocm.efferents) == 3
        assert all([proj.receiver.owner == icomp.parameter_CIM for proj in ocm.efferents])
        assert len(ia.mod_afferents) == 3
        assert all([proj.sender.owner == icomp.parameter_CIM for proj in ia.mod_afferents])

    def test_nested_control_projection_count_control_mech(self):
        # Inner Composition
        ia = TransferMechanism(name='ia')
        icomp = Composition(name='icomp', pathways=[ia])
        # Outer Composition
        oa = TransferMechanism(name='oa')
        cm = ControlMechanism(name='cm',
            control=[
            ControlSignal(projections=[(NOISE, ia)]),
            ControlSignal(projections=[(INTERCEPT, ia)]),
            ControlSignal(projections=[(SLOPE, ia)])
            ]
        )
        ocomp = Composition(name='ocomp', pathways=[[oa, icomp], [cm]])
        assert len(cm.efferents) == 3
        assert all([proj.receiver.owner == icomp.parameter_CIM for proj in cm.efferents])
        assert len(ia.mod_afferents) == 3
        assert all([proj.sender.owner == icomp.parameter_CIM for proj in ia.mod_afferents])


class TestInputCIMOutputPortToOriginOneToMany:

    def test_one_to_two(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C',
                                input_ports=[A.input_port])

        comp = Composition(name='comp')

        comp.add_linear_processing_pathway([A, B])
        comp.add_node(C)

        comp.run(inputs={A: [[1.23]]})

        assert np.allclose(A.parameters.value.get(comp), [[1.23]])
        assert np.allclose(B.parameters.value.get(comp), [[1.23]])
        assert np.allclose(C.parameters.value.get(comp), [[1.23]])

    def test_origin_input_source_true_no_input(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B')
        C = ProcessingMechanism(name='C',
                                default_variable=[[4.56]])

        comp = Composition(name='comp')

        comp.add_linear_processing_pathway([A, B])
        comp.add_node(C)

        comp.run(inputs={A: [[1.23]]})

        assert np.allclose(A.parameters.value.get(comp), [[1.23]])
        assert np.allclose(B.parameters.value.get(comp), [[1.23]])
        assert np.allclose(C.parameters.value.get(comp), [[4.56]])

    def test_mix_and_match_input_sources(self):
        A = ProcessingMechanism(name='A')
        B = ProcessingMechanism(name='B',
                                default_variable=[[0.], [0.]])
        C = ProcessingMechanism(name='C',
                                input_ports=[B.input_ports[1], A.input_port, B.input_ports[0]])

        input_dict = {A: [[2.0]],
                      B: [[3.0], [1.0]]}

        comp = Composition(name="comp")

        comp.add_node(A)
        comp.add_node(B)
        comp.add_node(C)

        comp.run(inputs=input_dict)

        assert np.allclose(A.parameters.value.get(comp), [[2.]])
        assert np.allclose(B.parameters.value.get(comp), [[3.], [1.]])
        assert np.allclose(C.parameters.value.get(comp), [[1.], [2.], [3.]])

    def test_non_origin_partial_input_spec(self):
        A = ProcessingMechanism(name='A',
                                function=Linear(slope=2.0))
        B = ProcessingMechanism(name='B',
                                input_ports=[[0.], A.input_port])

        comp = Composition(name='comp')

        comp.add_linear_processing_pathway([A, B])

        comp.run(inputs={A: [[1.23]]})
        assert np.allclose(B.get_input_values(comp), [[2.46], [1.23]])

class TestInputSpec:

    def test_valid_mismatched_input_lens(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")

        comp = Composition(name="COMP")

        comp.add_linear_processing_pathway([A, C])
        comp.add_linear_processing_pathway([B, C])

        inputs_to_A = [[1.0]]                           # same (1.0) on every trial
        inputs_to_B = [[1.0], [2.0], [3.0], [4.0]]      # increment on every trial

        results_A = []
        results_B = []
        results_C = []

        def call_after_trial():
            results_A.append(A.parameters.value.get(comp))
            results_B.append(B.parameters.value.get(comp))
            results_C.append(C.parameters.value.get(comp))

        comp.run(inputs={A: inputs_to_A,
                         B: inputs_to_B},
                 call_after_trial=call_after_trial)

        assert np.allclose(results_A, [[[1.0]], [[1.0]], [[1.0]], [[1.0]]])
        assert np.allclose(results_B, [[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
        assert np.allclose(results_C, [[[2.0]], [[3.0]], [[4.0]], [[5.0]]])

    def test_valid_only_one_node_provides_input_spec(self):
        A = ProcessingMechanism(name="A",
                                default_variable=[[1.5]])   # default variable will be used as input to this INPUT node
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")

        comp = Composition(name="COMP")

        comp.add_linear_processing_pathway([A, C])
        comp.add_linear_processing_pathway([B, C])

        inputs_to_B = [[1.0], [2.0], [3.0], [4.0]]      # increment on every trial

        results_A = []
        results_B = []
        results_C = []

        def call_after_trial():
            results_A.append(A.parameters.value.get(comp))
            results_B.append(B.parameters.value.get(comp))
            results_C.append(C.parameters.value.get(comp))

        comp.run(inputs={B: inputs_to_B},
                 call_after_trial=call_after_trial)

        assert np.allclose(results_A, [[[1.5]], [[1.5]], [[1.5]], [[1.5]]])
        assert np.allclose(results_B, [[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
        assert np.allclose(results_C, [[[2.5]], [[3.5]], [[4.5]], [[5.5]]])

    def test_invalid_mismatched_input_lens(self):
        A = ProcessingMechanism(name="A")
        B = ProcessingMechanism(name="B")
        C = ProcessingMechanism(name="C")

        comp = Composition(name="COMP")

        comp.add_linear_processing_pathway([A, C])
        comp.add_linear_processing_pathway([B, C])

        inputs_to_A = [[1.0], [2.0]]                    # 2 input specs
        inputs_to_B = [[1.0], [2.0], [3.0], [4.0]]      # 4 input specs

        with pytest.raises(CompositionError) as error_text:
            comp.run(inputs={A: inputs_to_A,
                             B: inputs_to_B})
        assert "input dictionary for COMP contains input specifications of different lengths" in str(error_text.value)

    def test_valid_input_float(self):
        A = ProcessingMechanism(name="A")
        comp = Composition(name="comp")
        comp.add_node(A)

        comp.run(inputs={A: 5.0})
        assert np.allclose(comp.results, [[5.0]])

        comp.run(inputs={A: [5.0, 10.0, 15.0]})
        assert np.allclose(comp.results, [[[5.0]], [[5.0]], [[10.0]], [[15.0]]])


class TestSimplifedNestedCompositionSyntax:
    def test_connect_outer_composition_to_only_input_node_in_inner_comp_option1(self):

        inner1 = Composition(name="inner")

        A1 = TransferMechanism(name="A1",
                               function=Linear(slope=2.0))

        B1 = TransferMechanism(name="B1",
                               function=Linear(slope=3.0))

        inner1.add_linear_processing_pathway([A1, B1])

        inner2 = Composition(name="inner2")

        A2 = TransferMechanism(name="A2",
                               function=Linear(slope=2.0))

        B2 = TransferMechanism(name="B2",
                               function=Linear(slope=3.0))

        inner2.add_linear_processing_pathway([A2, B2])

        outer = Composition(name="outer")
        outer.add_nodes([inner1, inner2])
        outer.add_projection(sender=B1, receiver=inner2)

        # comp1:  input = 5.0   |  mechA: 2.0*5.0 = 10.0    |  mechB: 3.0*10.0 = 30.0    |  output = 30.0
        # comp2:  input = 30.0  |  mechA2: 2.0*30.0 = 60.0  |  mechB2: 3.0*60.0 = 180.0  |  output = 180.0
        # comp3:  input = 5.0   |  output = 180.0

        res = outer.run(inputs={inner1: [[5.]]})
        assert np.allclose(res, [[[180.0]]])

        assert np.allclose(inner1.output_port.parameters.value.get(outer), [30.0])
        assert np.allclose(inner2.output_port.parameters.value.get(outer), [180.0])
        assert np.allclose(outer.output_port.parameters.value.get(outer), [180.0])

    def test_connect_outer_composition_to_only_input_node_in_inner_comp_option2(self):
        inner1 = Composition(name="inner")

        A1 = TransferMechanism(name="A1",
                               function=Linear(slope=2.0))

        B1 = TransferMechanism(name="B1",
                               function=Linear(slope=3.0))

        inner1.add_linear_processing_pathway([A1, B1])

        inner2 = Composition(name="inner2")

        A2 = TransferMechanism(name="A2",
                               function=Linear(slope=2.0))

        B2 = TransferMechanism(name="B2",
                               function=Linear(slope=3.0))

        inner2.add_linear_processing_pathway([A2, B2])

        outer = Composition(name="outer")
        outer.add_nodes([inner1, inner2])
        outer.add_projection(sender=inner1, receiver=A2)

        # CRASHING WITH: FIX 6/1/20
        # subprocess.CalledProcessError: Command '['dot', '-Tpdf', '-O', 'outer']' returned non-zero exit status 1.
        # outer.show_graph(show_node_structure=True,
        #                  show_nested=True)

        # comp1:  input = 5.0   |  mechA: 2.0*5.0 = 10.0    |  mechB: 3.0*10.0 = 30.0    |  output = 30.0
        # comp2:  input = 30.0  |  mechA2: 2.0*30.0 = 60.0  |  mechB2: 3.0*60.0 = 180.0  |  output = 180.0
        # comp3:  input = 5.0   |  output = 180.0

        res = outer.run(inputs={inner1: [[5.]]})
        assert np.allclose(res, [[[180.0]]])

        assert np.allclose(inner1.output_port.parameters.value.get(outer), [30.0])
        assert np.allclose(inner2.output_port.parameters.value.get(outer), [180.0])
        assert np.allclose(outer.output_port.parameters.value.get(outer), [180.0])

    def test_connect_outer_composition_to_only_input_node_in_inner_comp_option3(self):

        inner1 = Composition(name="inner")

        A1 = TransferMechanism(name="A1",
                               function=Linear(slope=2.0))

        B1 = TransferMechanism(name="B1",
                               function=Linear(slope=3.0))

        inner1.add_linear_processing_pathway([A1, B1])

        inner2 = Composition(name="inner2")

        A2 = TransferMechanism(name="A2",
                               function=Linear(slope=2.0))

        B2 = TransferMechanism(name="B2",
                               function=Linear(slope=3.0))

        inner2.add_linear_processing_pathway([A2, B2])

        outer = Composition(name="outer")
        outer.add_nodes([inner1, inner2])
        outer.add_projection(sender=B1, receiver=A2)

        # comp1:  input = 5.0   |  mechA: 2.0*5.0 = 10.0    |  mechB: 3.0*10.0 = 30.0    |  output = 30.0
        # comp2:  input = 30.0  |  mechA2: 2.0*30.0 = 60.0  |  mechB2: 3.0*60.0 = 180.0  |  output = 180.0
        # comp3:  input = 5.0   |  output = 180.0

        res = outer.run(inputs={inner1: [[5.]]})
        assert np.allclose(res, [[[180.0]]])

        assert np.allclose(inner1.output_port.parameters.value.get(outer), [30.0])
        assert np.allclose(inner2.output_port.parameters.value.get(outer), [180.0])
        assert np.allclose(outer.output_port.parameters.value.get(outer), [180.0])

    def test_connect_outer_composition_to_all_input_nodes_in_inner_comp(self):

        inner1 = Composition(name="inner")
        A1 = TransferMechanism(name="A1",
                               function=Linear(slope=2.0))
        B1 = TransferMechanism(name="B1",
                               function=Linear(slope=3.0))

        inner1.add_linear_processing_pathway([A1, B1])

        inner2 = Composition(name="inner2")
        A2 = TransferMechanism(name="A2")
        B2 = TransferMechanism(name="B2")
        C2 = TransferMechanism(name="C2")

        inner2.add_nodes([A2, B2, C2])

        outer1 = Composition(name="outer1")
        outer1.add_nodes([inner1, inner2])

        # Spec 1: add projection *node in* inner1 --> inner 2 (implies first InputPort -- corresponding to A2)
        outer1.add_projection(sender=B1, receiver=inner2)
        # Spec 2:  add projection *node in* inner1 --> *node in* inner2
        outer1.add_projection(sender=B1, receiver=B2)
        # Spec 3: add projection inner1 --> *node in* inner2
        outer1.add_projection(sender=inner1, receiver=C2)
        eid = "eid"
        outer1.run(inputs={inner1: [[1.]]},
                   context=eid)

        assert np.allclose(A1.parameters.value.get(eid), [[2.0]])
        assert np.allclose(B1.parameters.value.get(eid), [[6.0]])

        for node in [A2, B2, C2]:
            assert np.allclose(node.parameters.value.get(eid), [[6.0]])

@pytest.mark.parametrize(
    # expected_input_shape: one input per source input_port
    # expected_output_shape: one output per terminal output_port
    # TODO: change mechanisms to input_ports and output_ports args?
    'mechanisms, expected_input_shape, expected_output_shape',
    [
        (
            [TransferMechanism()], [[0]], [[0]]
        ),
        (
            [
                TransferMechanism(),
                TransferMechanism(output_ports=['RESULT', 'MEAN']),
            ],
            [[0], [0]],
            [[0], [0], [0]],
        ),
        (
            [
                TransferMechanism(),
                TransferMechanism(output_ports=['RESULT', 'MEAN']),
                TransferMechanism(output_ports=['RESULT', 'MEAN', 'MEDIAN']),
            ],
            [[0], [0], [0]],
            [[0], [0], [0], [0], [0], [0]],
        ),
        (
            [
                TransferMechanism(input_ports=['Port1', 'Port2']),
                TransferMechanism(output_ports=['RESULT', 'MEAN']),
            ],
            [[0], [0], [0]],
            # an output_port generated for each custom input_port
            # overwrites output_ports arg - intentional?
            [[0], [0], [0], [0]],
        ),
        (
            [
                TransferMechanism(),
                TransferMechanism(
                    input_ports=['Port1', 'Port2'],
                ),
                TransferMechanism(
                    input_ports=['Port1', 'Port2', 'Port3'],
                ),
            ],
            [[0], [0], [0], [0], [0], [0]],
            [[0], [0], [0], [0], [0], [0]],
        ),
    ]
)
def test_CIM_shapes(mechanisms, expected_input_shape, expected_output_shape):
    comp = Composition()

    for i in range(len(mechanisms)):
        comp.add_node(mechanisms[i])

    comp._analyze_graph()

    for cim, expected_shape in [
        (comp.input_CIM, expected_input_shape),
        (comp.output_CIM, expected_output_shape),
    ]:
        np.testing.assert_array_equal(
            cim.defaults.variable.shape,
            cim.defaults.value.shape,
            err_msg=f'{cim}:',
            verbose=True
        )
        np.testing.assert_array_equal(
            cim.defaults.variable.shape,
            cim.parameters.variable.get().shape,
            err_msg=f'{cim}:',
            verbose=True
        )
        np.testing.assert_array_equal(
            cim.defaults.value.shape,
            cim.parameters.value.get().shape,
            err_msg=f'{cim}:',
            verbose=True
        )
        np.testing.assert_array_equal(
            cim.defaults.variable,
            convert_all_elements_to_np_array(expected_shape),
            err_msg=f'{cim}:',
            verbose=True
        )
