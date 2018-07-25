import functools
import logging
from timeit import timeit

import numpy as np
import pytest

from psyneulink.components.functions.function import Linear, SimpleIntegrator
from psyneulink.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism, TRANSFER_OUTPUT
from psyneulink.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.library.mechanisms.processing.transfer.recurrenttransfermechanism import RecurrentTransferMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.states.inputstate import InputState
from psyneulink.compositions.composition import Composition, CompositionError, CNodeRole
from psyneulink.compositions.parsingautodiffcomposition import ParsingAutodiffComposition, ParsingAutodiffCompositionError
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

# Unit tests for functions of ParsingAutodiffComposition class that are new (not in Composition)
# or override functions in Composition



class TestPACConstructor:
    
    def test_no_args(self):
        comp = ParsingAutodiffComposition()
        assert isinstance(comp, ParsingAutodiffComposition)
    
    def test_two_calls_no_args(self):
        comp = ParsingAutodiffComposition()
        assert isinstance(comp, ParsingAutodiffComposition)
        
        comp_2 = ParsingAutodiffComposition()
        assert isinstance(comp, ParsingAutodiffComposition)
        assert isinstance(comp_2, ParsingAutodiffComposition)
    
    def test_target_CIM(self):
        comp = ParsingAutodiffComposition()
        assert isinstance(comp.target_CIM, CompositionInterfaceMechanism)
        assert comp.target_CIM.composition == comp
        assert comp.target_CIM_states == {}
    
    def test_model(self):
        comp = ParsingAutodiffComposition()
        assert comp.model == None
    
    def test_report_prefs(self):
        comp = ParsingAutodiffComposition()
        assert comp.input_CIM.reportOutputPref == False
        assert comp.output_CIM.reportOutputPref == False
        assert comp.target_CIM.reportOutputPref == False



class TestCIMStateCreation:
    
    def test_
        






























