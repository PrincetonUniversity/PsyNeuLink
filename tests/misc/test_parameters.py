import psyneulink as pnl
import pytest

from psyneulink.core.globals.utilities import unproxy_weakproxy

# (ancestor, child, should_override)
ancestor_child_data = [
    (pnl.Component, pnl.TransferMechanism, False),
    (pnl.Component, pnl.OutputState, True),
    (pnl.Component, pnl.InputState, True),
    (pnl.Component, pnl.SimpleIntegrator, False),
    (pnl.Function_Base, pnl.SimpleIntegrator, True),
    (pnl.TransferMechanism, pnl.RecurrentTransferMechanism, True)
]

# (obj, param_name, alias_name)
param_alias_data = [
    (pnl.Linear, 'slope', 'multiplicative_param'),
    (pnl.Linear, 'intercept', 'additive_param'),
    (pnl.ControlMechanism, 'value', 'allocation_policy'),
]


@pytest.fixture(scope='function')
def reset_variable(*args):
    yield
    # pytest cannot provide the exact parametrized arguments to fixtures
    # so just reset all of the possibilities
    # this must be used when altering class level defaults
    for item in ancestor_child_data:
        item[0].parameters.variable.reset()
        item[1].parameters.variable.reset()


@pytest.mark.parametrize('ancestor, child', [(item[0], item[1]) for item in ancestor_child_data])
def test_parameter_propagation(ancestor, child):
    for param in ancestor.parameters:
        child_params = child.parameters.values(show_all=True)

        assert param.name in child_params


@pytest.mark.parametrize('ancestor, child, should_override', ancestor_child_data)
def test_parameter_values_overriding(ancestor, child, should_override, reset_variable):
    original_child_variable = child.parameters.variable.default_value

    # ancestor updates
    ancestor.parameters.variable = -1
    assert ancestor.parameters.variable.default_value == -1

    if should_override:
        assert child.parameters.variable.default_value == -1
    else:
        assert child.parameters.variable.default_value == original_child_variable

    # child updates and ancestor does not update
    child.parameters.variable = -2
    assert child.parameters.variable.default_value == -2
    assert ancestor.parameters.variable.default_value == -1

    # child should not get overridden because it is explicitly specified
    ancestor.parameters.variable = -3
    assert child.parameters.variable.default_value == -2

    # revert to original behavior
    child.parameters.variable.reset()
    if should_override:
        assert child.parameters.variable.default_value == -3
    else:
        assert child.parameters.variable.default_value == original_child_variable


@pytest.mark.parametrize('obj, param_name, alias_name', param_alias_data)
def test_aliases(obj, param_name, alias_name):
    obj = obj()
    assert unproxy_weakproxy(obj.parameters._owner) is obj
    assert unproxy_weakproxy(getattr(obj.parameters, alias_name)._owner._owner) is obj
    assert getattr(obj.defaults, param_name) == getattr(obj.defaults, alias_name)
    assert unproxy_weakproxy(getattr(obj.parameters, alias_name).source) is getattr(obj.parameters, param_name)


@pytest.mark.parametrize('obj, param_name, alias_name', param_alias_data)
def test_aliases_set_source(obj, param_name, alias_name):
    obj = obj()

    setattr(obj.defaults, param_name, -100)
    assert getattr(obj.defaults, param_name) == getattr(obj.defaults, alias_name)


@pytest.mark.parametrize('obj, param_name, alias_name', param_alias_data)
def test_aliases_set_alias(obj, param_name, alias_name):
    obj = obj()

    setattr(obj.defaults, alias_name, -1)
    assert getattr(obj.defaults, param_name) == getattr(obj.defaults, alias_name)


def test_parameter_getter():
    f = pnl.Linear()
    f.parameters.slope.getter = lambda x: x ** 2

    assert f.parameters.slope.get(x=3) == 9


def test_parameter_setter():
    f = pnl.Linear()
    f.parameters.slope.setter = lambda x: x ** 2

    f.parameters.slope.set(3)

    assert f.parameters.slope.get() == 9
