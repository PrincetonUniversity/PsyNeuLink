import psyneulink as pnl
import pytest

from psyneulink.core.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
from psyneulink.core.components.projections.pathway.pathwayprojection import PathwayProjection_Base


@pytest.mark.parametrize(
    'class_type',
    [
        pnl.Mechanism_Base,
        pnl.Function_Base,
        pnl.State_Base,
        ModulatoryProjection_Base,
        PathwayProjection_Base,
    ]
)
def test_abstract_classes(class_type):
    with pytest.raises(TypeError) as err:
        class_type()
    assert 'abstract class' in str(err.value)
