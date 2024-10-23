import inspect
import psyneulink as pnl
import numpy as np
import pytest
import re

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


@pytest.mark.parametrize("class_", component_classes)
def test_constructors_have_check_user_specified(class_):
    assert "check_user_specified" in inspect.getsource(class_.__init__), (
        f"The __init__ method of Component {class_.__name__} must be wrapped by"
        f" check_user_specified in {pnl.core.globals.parameters.check_user_specified.__module__}"
    )


def _numeric_parameter_value_check(class_, param_name, value, descriptor):
    descriptor = f'{class_}.parameters.{param_name}{descriptor}'
    assert isinstance(value, np.ndarray) or not pnl.is_numeric(value), (
        f'{descriptor} is a numeric value but is not wrapped in a'
        f' numpy array:\n\t{value}\n\t{type(value)}'
    )


# could parametrize over each parameter for each class instead of
# looping, but this would greatly increase the overall number of tests
# for minimal benefit
@pytest.mark.parametrize('class_', component_classes)
def test_numeric_parameter_values_are_numpy_defaults(class_):
    for parameter in class_.parameters:
        _numeric_parameter_value_check(
            class_, parameter.name, parameter.default_value, ' default_value'
        )


@pytest.mark.parametrize('class_', component_classes)
def test_numeric_parameter_values_are_numpy_values(class_):
    for parameter in class_.parameters:
        for eid, v in parameter.values.items():
            _numeric_parameter_value_check(class_, parameter.name, v, f'.values[{eid}]')


@pytest.fixture(scope='module')
def nested_compositions():
    comp = pnl.Composition(name='comp')
    inner_comp = pnl.Composition(name='Inner Composition')
    A = pnl.TransferMechanism(
        function=pnl.Linear(slope=5.0, intercept=2.0),
        name='A'
    )
    B = pnl.TransferMechanism(function=pnl.Logistic, name='B')
    C = pnl.RecurrentTransferMechanism(name='C')
    D = pnl.IntegratorMechanism(
        function=pnl.SimpleIntegrator(noise=pnl.NormalDist()),
        name='D'
    )
    E = pnl.TransferMechanism(name='E')
    F = pnl.TransferMechanism(name='F')

    for m in [E, F]:
        inner_comp.add_node(m)

    for m in [A, B, C, D, inner_comp]:
        comp.add_node(m)

    comp.add_projection(pnl.MappingProjection(), A, B)
    comp.add_projection(pnl.MappingProjection(), A, C)
    comp.add_projection(pnl.MappingProjection(), B, D)
    comp.add_projection(pnl.MappingProjection(), C, D)
    comp.add_projection(pnl.MappingProjection(), C, inner_comp)

    inner_comp.add_projection(pnl.MappingProjection(), E, F)

    yield comp, inner_comp


@pytest.mark.parametrize(
    'filter_name, filter_regex, unknown_param_names',
    [
        (None, None, []),
        (None, 'slo$', ['slope']),
        ('slo', None, ['slope']),
        ('slo', 'slo$', ['slope']),
        (['slope', 'seed'], None, []),
        (None, ['slope', 'seed'], []),
        (None, ['.*_param'], ['slope']),
    ]
)
def test_all_dependent_parameters(
    nested_compositions,
    filter_name,
    filter_regex,
    unknown_param_names
):
    comp, inner_comp = nested_compositions

    params_comp = comp.all_dependent_parameters(filter_name, filter_regex)
    params_inner_comp = inner_comp.all_dependent_parameters(
        filter_name, filter_regex
    )

    params_comp_keys = set(params_comp.keys())
    params_inner_comp_keys = set(params_inner_comp.keys())

    assert params_inner_comp_keys.issubset(params_comp_keys)
    assert (
        len(params_comp_keys) == 0
        or not params_comp_keys.issubset(params_inner_comp_keys)
    )

    if filter_name is not None:
        if isinstance(filter_name, str):
            filter_name = [filter_name]

    if filter_regex is not None:
        if isinstance(filter_regex, str):
            filter_regex = [filter_regex]

    for item, comp_name in [
        (params_comp, 'comp'),
        (params_inner_comp, 'inner_comp')
    ]:
        for p in item:
            assert p._owner._owner is item[p], (p.name, comp_name)

            matches = True
            try:
                matches = matches and p.name in filter_name
            except TypeError:
                pass

            try:
                for pattern in filter_regex:
                    matches = matches or re.match(pattern, p.name)
            except TypeError:
                pass

            assert matches

        for p in unknown_param_names:
            assert p not in item, (p.name, comp_name)
