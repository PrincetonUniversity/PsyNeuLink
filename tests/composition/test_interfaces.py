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

class TestExecuteCIM():

    def test_identity_function(self):

        I = Identity()

        output = I.execute(2.0)
        assert output == 2.0

        output = I.execute([1.0, 2.0,3.0])
        assert np.allclose([1.0, 2.0,3.0], output)

        output = I.execute([[1.0, 2.0], [3.0]])
        assert np.allclose([1.0, 2.0], [3.0], output)

    def test_standalone_CIM(self):
        cim = CompositionInterfaceMechanism()
        cim.execute(2.0)

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
        print(comp.input_CIM.output_states)
        print(output)
        print(comp.output_CIM.output_states)
        assert np.allclose(30, output)
