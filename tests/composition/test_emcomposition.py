import logging
import timeit as timeit
import os
import numpy as np

import pytest

import psyneulink as pnl

from psyneulink.core.components.functions.nonstateful.transferfunctions import Logistic
from psyneulink.core.components.functions.nonstateful.learningfunctions import BackPropagation
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.globals import Context
from psyneulink.core.globals.keywords import TRAINING_SET, Loss
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.library.compositions.emcomposition import EMComposition, EMCompositionError
from psyneulink.core.compositions.report import ReportOutput

logger = logging.getLogger(__name__)


# All tests are set to run. If you need to skip certain tests,
# see http://doc.pytest.org/en/latest/skipping.html

# Unit tests for functions of EMComposition class that are new (not in Composition)
# or override functions in Composition

def _single_learn_results(composition, *args, **kwargs):
    composition.learn(*args, **kwargs)
    return composition.learning_results

@pytest.mark.pytorch
@pytest.mark.acconstructor
class TestACConstructor:

    def test_no_args(self):
        comp = EMComposition()
        assert isinstance(comp, EMComposition)

    def test_two_calls_no_args(self):
        comp = EMComposition()
        comp_2 = EMComposition()
        assert isinstance(comp, EMComposition)
        assert isinstance(comp_2, EMComposition)

    @pytest.mark.skip(reason="no pytorch representation of EMComposition yet")
    def test_pytorch_representation(self):
        comp = EMComposition()
        assert comp.pytorch_representation is None

    def test_report_prefs(self):
        comp = EMComposition()
        assert comp.input_CIM.reportOutputPref == ReportOutput.OFF
        assert comp.output_CIM.reportOutputPref == ReportOutput.OFF
        # assert comp.target_CIM.reportOutputPref == False


@pytest.mark.skip(reason="no pytorch representation of EMComposition yet")
@pytest.mark.pytorch
@pytest.mark.composition
def test_autodiff_forward(autodiff_mode):
    # create xor model mechanisms and projections
    xor_in = TransferMechanism(name='xor_in',
                               default_variable=np.zeros(2))

    xor_hid = TransferMechanism(name='xor_hid',
                                default_variable=np.zeros(10),
                                function=Logistic())

    xor_out = TransferMechanism(name='xor_out',
                                default_variable=np.zeros(1),
                                function=Logistic())

    hid_map = MappingProjection(matrix=np.random.rand(2,10))
    out_map = MappingProjection(matrix=np.random.rand(10,1))

    # put the mechanisms and projections together in an autodiff composition (AC)
    xor = EMComposition()

    xor.add_node(xor_in)
    xor.add_node(xor_hid)
    xor.add_node(xor_out)

    xor.add_projection(sender=xor_in, projection=hid_map, receiver=xor_hid)
    xor.add_projection(sender=xor_hid, projection=out_map, receiver=xor_out)

    outputs = xor.run(inputs=[0,0], execution_mode=autodiff_mode)
    np.testing.assert_allclose(outputs, [[0.9479085241082691]])


@pytest.mark.skip(reason="no pytorch representation of EMComposition yet")
@pytest.mark.pytorch
@pytest.mark.accorrectness
@pytest.mark.composition
class TestTrainingCorrectness:
    """FIX: SHOULD IMPLEMENT CORRESPONDING TESTS FROM AutodiffComposition"""
    pass


@pytest.mark.skip(reason="no pytorch representation of EMComposition yet")
@pytest.mark.pytorch
@pytest.mark.acidenticalness
class TestTrainingIdenticalness():
    """FIX: SHOULD IMPLEMENT CORRESPONDING TESTS FROM AutodiffComposition"""
    pass


@pytest.mark.skip(reason="no pytorch representation of EMComposition yet")
@pytest.mark.pytorch
@pytest.mark.acmisc
@pytest.mark.composition
class TestMiscTrainingFunctionality:
    """FIX: SHOULD IMPLEMENT CORRESPONDING TESTS FROM AutodiffComposition"""
    pass

@pytest.mark.skip(reason="no pytorch representation of EMComposition yet")
@pytest.mark.pytorch
@pytest.mark.actime
class TestTrainingTime:
    """FIX: SHOULD IMPLEMENT CORRESPONDING TESTS FROM AutodiffComposition"""
    pass


@pytest.mark.skip(reason="no pytorch representation of EMComposition yet")
@pytest.mark.pytorch
def test_autodiff_saveload(tmp_path):
    """FIX: SHOULD IMPLEMENT CORRESPONDING TESTS FROM AutodiffComposition"""
    pass


@pytest.mark.skip(reason="no pytorch representation of EMComposition yet")
@pytest.mark.pytorch
@pytest.mark.aclogging
class TestACLogging:
    """FIX: SHOULD IMPLEMENT CORRESPONDING TESTS FROM AutodiffComposition"""
    pass


@pytest.mark.skip(reason="no pytorch representation of EMComposition yet")
@pytest.mark.pytorch
@pytest.mark.acnested
@pytest.mark.composition
class TestNested:
    """FIX: SHOULD IMPLEMENT CORRESPONDING TESTS FROM AutodiffComposition"""
    pass


@pytest.mark.skip(reason="no pytorch representation of EMComposition yet")
@pytest.mark.pytorch
class TestBatching:
    """FIX: SHOULD IMPLEMENT CORRESPONDING TESTS FROM AutodiffComposition"""
    pass
