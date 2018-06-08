import functools
import logging
from timeit import timeit

import numpy as np
import pytest

from psyneulink.components.functions.function import Linear, SimpleIntegrator, Identity
from psyneulink.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism, TRANSFER_OUTPUT
from psyneulink.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.library.mechanisms.processing.transfer.recurrenttransfermechanism import RecurrentTransferMechanism
from psyneulink.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
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

class TestExecuteCIM:

    def test_identity_function(self):

        I = Identity()

        output = I.execute(2.0)
        assert output == 2.0

        output = I.execute([1.0, 2.0,3.0])
        assert np.allclose([1.0, 2.0,3.0], output)

        output = I.execute([[1.0, 2.0], [3.0]])
        print(output)
        assert np.allclose([1.0, 2.0], output[0])
        assert np.allclose([3.0], output[1])

    def test_standalone_CIM(self):

        cim = CompositionInterfaceMechanism()
        cim.execute(2.0)
        assert np.allclose(cim.value, [2.0])

    def test_assign_value(self):
        cim = CompositionInterfaceMechanism()
        cim.instance_defaults.variable = [2.0]
        cim.execute()
        assert np.allclose(cim.value, [2.0])

    def test_standalone_CIM_multiple_input_states(self):

        cim = CompositionInterfaceMechanism(default_variable=[[0.0], [0.0], [0.0]])
        cim.execute([[1.0], [2.0], [3.0]])
        assert np.allclose(cim.value, [[1.0], [2.0], [3.0]])
        print(cim.output_states)
        print(cim.output_states[0].value)
        print(cim.output_states[1].value)
        print(cim.output_states[0].variable)
        print(cim.output_states[1].variable)

    def test_standalone_processing_multiple_input_states(self):

        processing_mech = ProcessingMechanism(default_variable=[[0.0], [0.0], [0.0]])
        processing_mech.execute([[1.0], [2.0], [3.0]])
        assert np.allclose(processing_mech.value, [[1.0], [2.0], [3.0]])


    def test_one_input_state_one_output_state(self):

        comp = Composition()

        A = TransferMechanism(name="composition-pytests-A",
                              function=Linear(slope=2.0))

        B = TransferMechanism(name="composition-pytests-B",
                              function=Linear(slope=3.0))

        comp.add_mechanism(A)
        comp.add_mechanism(B)

        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)

        comp._analyze_graph()

        inputs_dict = {
            A: [[5.]],
        }
        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )
        print("B.value = ", B.value)
        print("B.output_states = ", B.output_states)
        print("B.output_values = ", B.output_values)
        # print("B.output_states[0].efferents[0].receiver.variable = ", B.output_states[0].efferents[0].receiver.variable)
        print("Input States: \n", comp.output_CIM.input_states)
        print("Input Values: \n", comp.output_CIM.input_values)
        print("\n\n Output States: \n", comp.output_CIM.output_states)
        print("Output Values: \n", comp.output_CIM.output_values)
        print("output = ", output)
        assert np.allclose([30], output)

    def test_two_input_states_two_output_states(self):

        comp = Composition()

        A = TransferMechanism(name="composition-pytests-A",
                              default_variable=[[0.0], [0.0]],
                              function=Linear(slope=2.0))

        B = TransferMechanism(name="composition-pytests-B",
                              default_variable=[[0.0], [0.0]],
                              function=Linear(slope=3.0))

        comp.add_mechanism(A)
        comp.add_mechanism(B)

        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp.add_projection(A, MappingProjection(sender=A.output_states[1], receiver=B.input_states[1]), B)

        comp._analyze_graph()
        inputs_dict = {
            A: [[5.], [6.]],
        }
        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )
        print(B.value)
        print(comp.output_CIM.output_states)
        print(comp.output_CIM.output_values)
        assert np.allclose([[30.], [36.]], output)


        # assert np.allclose([30.], comp.output_CIM.output_states[1].value)
        # assert np.allclose([36.], comp.output_CIM.output_states[2].value)

class TestConnectCompositionsViaCIMS:

    def test_connect_compositions_with_simple_states(self):

        comp1 = Composition(name="first_composition")

        A = TransferMechanism(name="composition-pytests-A",
                              function=Linear(slope=2.0))

        B = TransferMechanism(name="composition-pytests-B",
                              function=Linear(slope=3.0))

        comp1.add_mechanism(A)
        comp1.add_mechanism(B)

        comp1.add_projection(A, MappingProjection(sender=A, receiver=B), B)

        comp1._analyze_graph()
        inputs_dict = {
            A: [[5.]],
        }
        # comp1.run(inputs_dict)

        sched = Scheduler(composition=comp1)

        comp2 = Composition(name="second_composition")

        A2 = TransferMechanism(name="composition-pytests-A2",
                              function=Linear(slope=2.0))

        B2 = TransferMechanism(name="composition-pytests-B2",
                              function=Linear(slope=3.0))

        comp2.add_mechanism(A2)
        comp2.add_mechanism(B2)

        comp2.add_projection(A2, MappingProjection(sender=A2, receiver=B2), B2)

        comp2._analyze_graph()
        sched = Scheduler(composition=comp2)

        comp3 = Composition(name="outer_composition")
        comp3.add_mechanism(comp1)

        comp3.run(inputs={comp1: [[5.]]})
        # m=MappingProjection(sender=comp1.output_states[0], receiver=comp2.input_states[0])
        #
        # comp1.run(inputs=inputs_dict)
        # # m.execute()
        # comp2.execute()

    def test_connect_compositions_with_complicated_states(self):

        comp = Composition()

        A = TransferMechanism(name="composition-pytests-A",
                              default_variable=[[0.0], [0.0]],
                              function=Linear(slope=2.0))

        B = TransferMechanism(name="composition-pytests-B",
                              default_variable=[[0.0], [0.0]],
                              function=Linear(slope=3.0))

        comp.add_mechanism(A)
        comp.add_mechanism(B)

        comp.add_projection(A, MappingProjection(sender=A, receiver=B), B)
        comp.add_projection(A, MappingProjection(sender=A.output_states[1], receiver=B.input_states[1]), B)

        comp._analyze_graph()
        inputs_dict = {
            A: [[5.], [6.]],
        }
        sched = Scheduler(composition=comp)
        output = comp.run(
            inputs=inputs_dict,
            scheduler_processing=sched
        )
        print(B.value)
        print(comp.output_CIM.output_states)
        print(comp.output_CIM.output_values)
        assert np.allclose([[30.], [36.]], output)


        # assert np.allclose([30.], comp.output_CIM.output_states[1].value)
        # assert np.allclose([36.], comp.output_CIM.output_states[2].value)
