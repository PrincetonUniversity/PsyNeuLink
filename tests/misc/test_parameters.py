import copy
import numpy as np
import psyneulink as pnl
import pytest


# (ancestor, child, should_override)
ancestor_child_data = [
    (pnl.Component, pnl.TransferMechanism, False),
    (pnl.Component, pnl.OutputPort, False),
    (pnl.Component, pnl.InputPort, True),
    (pnl.Component, pnl.SimpleIntegrator, False),
    (pnl.Function_Base, pnl.SimpleIntegrator, True),
    (pnl.TransferMechanism, pnl.RecurrentTransferMechanism, True)
]

# (obj, param_name, alias_name)
param_alias_data = [
    (pnl.Linear, 'slope', 'multiplicative_param'),
    (pnl.Linear, 'intercept', 'additive_param'),
    (pnl.ControlMechanism, 'value', 'control_allocation'),
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
    assert obj.parameters._owner is obj
    assert getattr(obj.parameters, alias_name)._owner._owner is obj
    assert getattr(obj.defaults, param_name) == getattr(obj.defaults, alias_name)
    # if hasattr(getattr(obj.parameters, alias_name), 'source'):
    assert getattr(obj.parameters, alias_name).source is getattr(obj.parameters, param_name)


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


def test_history():
    t = pnl.TransferMechanism()
    assert t.parameters.value.get_previous() is None
    t.execute(10)
    assert t.parameters.value.get_previous() == 0
    t.execute(100)
    assert t.parameters.value.get_previous() == 10


@pytest.mark.parametrize(
    'index, range_start, range_end, expected',
    [
        (1, None, None, 4),
        (6, None, None, None),
        (None, 2, None, [3, 4]),
        (None, 2, 0, [3, 4]),
        (1, 2, 0, [3, 4]),
        (None, 5, 2, [0, 1, 2]),
        (None, 10, 2, [0, 1, 2])
    ]
)
def test_get_previous(index, range_start, range_end, expected):
    t = pnl.TransferMechanism()
    t.parameters.value.history_max_length = 10

    for i in range(1, 6):
        t.execute(i)

    previous = t.parameters.value.get_previous(
        index=index,
        range_start=range_start,
        range_end=range_end,
    )

    assert previous == expected


def test_delta():
    t = pnl.TransferMechanism()

    t.execute(10)
    assert t.parameters.value.get_delta() == 10
    t.execute(100)
    assert t.parameters.value.get_delta() == 90


def test_delta_fail():
    t = pnl.TransferMechanism()
    t.parameters.value.set(None, override=True)

    t.execute(10)
    with pytest.raises(TypeError) as error:
        t.parameters.value.get_delta()

    assert "Parameter 'value' value mismatch between current" in str(error.value)


def test_validation():
    class NewTM(pnl.TransferMechanism):
        class Parameters(pnl.TransferMechanism.Parameters):
            variable = pnl.Parameter(np.array([[0], [0], [0]]), read_only=True)

            def _validate_variable(self, variable):
                if not isinstance(variable, np.ndarray) or not variable.shape == np.array([[0], [0], [0]]).shape:
                    return 'must be 2d numpy array of shape (3, 1)'

    t = NewTM()

    t.defaults.variable = np.array([[1], [2], [3]])
    t.parameters.variable.default_value = np.array([[1], [2], [3]])

    with pytest.raises(pnl.ParameterError):
        t.defaults.variable = 0

    with pytest.raises(pnl.ParameterError):
        t.defaults.variable = np.array([0])

    with pytest.raises(pnl.ParameterError):
        t.parameters.variable.default_value = 0

    with pytest.raises(pnl.ParameterError):
        t.parameters.variable.default_value = np.array([[0]])


def test_dot_notation():
    c = pnl.Composition()
    d = pnl.Composition()
    t = pnl.TransferMechanism()
    c.add_node(t)
    d.add_node(t)

    t.execute(1)
    assert t.value == 1
    c.run({t: 5})
    assert t.value == 5
    d.run({t: 10})
    assert t.value == 10
    c.run({t: 20}, context='custom execution id')
    assert t.value == 20

    # context None
    assert t.parameters.value.get() == 1
    assert t.parameters.value.get(c) == 5
    assert t.parameters.value.get(d) == 10
    assert t.parameters.value.get('custom execution id') == 20


def test_copy():
    f = pnl.Linear()
    g = copy.deepcopy(f)

    assert isinstance(g.parameters.additive_param, pnl.ParameterAlias)
    assert g.parameters.additive_param.source is g.parameters.intercept


@pytest.mark.parametrize(
    'cls_, kwargs, parameter, is_user_specified',
    [
        (pnl.AdaptiveIntegrator, {'rate': None}, 'rate', False),
        (pnl.AdaptiveIntegrator, {'rate': None}, 'multiplicative_param', False),
        (pnl.AdaptiveIntegrator, {'rate': 0.5}, 'rate', True),
        (pnl.AdaptiveIntegrator, {'rate': 0.5}, 'multiplicative_param', True),
    ]
)
def test_user_specified(cls_, kwargs, parameter, is_user_specified):
    c = cls_(**kwargs)
    assert getattr(c.parameters, parameter)._user_specified == is_user_specified
