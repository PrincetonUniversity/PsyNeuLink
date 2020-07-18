import inspect
import psyneulink as pnl
import pytest

from psyneulink.core.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
from psyneulink.core.components.projections.pathway.pathwayprojection import PathwayProjection_Base

# gather all Component classes (a set to ensure no duplicates)
component_classes = []
component_class_constructor_arguments = {}
for item in pnl.__all__:
    evaled = eval(f'pnl.{item}')

    if isinstance(
        evaled,
        pnl.core.components.component.ComponentsMeta
    ):
        component_classes.append(evaled)
        component_class_constructor_arguments[evaled] = inspect.signature(
            evaled.__init__
        ).parameters

component_classes.sort(key=lambda x: x.__name__)


@pytest.mark.parametrize(
    'class_type',
    [
        pnl.Mechanism_Base,
        pnl.Function_Base,
        pnl.Port_Base,
        ModulatoryProjection_Base,
        PathwayProjection_Base,
    ]
)
def test_abstract_classes(class_type):
    with pytest.raises(TypeError) as err:
        class_type()
    assert 'abstract class' in str(err.value)


@pytest.mark.parametrize(
    'class_',
    component_classes
)
def test_function_parameters_stateless(class_):
    try:
        assert class_.parameters.function.stateful is False, (
            f'{class_.__name__}.parameters.function.stateful is True. '
            'The function Parameter for Components is currently '
            'expected to be stateless (defined as stateful=False)'
        )
    except AttributeError:
        pass


@pytest.mark.parametrize(
    'class_',
    component_classes
)
def test_parameters_user_specified(class_):
    violators = set()
    constructor_parameters = inspect.signature(class_.__init__).parameters
    for name, param in constructor_parameters.items():
        if (
            param.kind in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY
            }
            and name in class_.parameters.names()
            and param.default is not inspect.Parameter.empty
            and param.default is not None
        ):
            violators.add(name)

    message = (
        "If a value other than None is used as the default value in a class's"
        + ' constructor/__init__, for an argument corresponding to a Parameter,'
        + ' _user_specified will always be True. The default value should be'
        + " specified in the class's Parameters inner class. Violators for"
        + f' {class_.__name__}: {violators}'
    )
    assert violators == set(), message
