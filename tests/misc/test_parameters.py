import copy
import numpy as np
import psyneulink as pnl
import pytest
import re
import warnings


NO_PARAMETERS = "NO_PARAMETERS"
NO_INIT = "NO_INIT"
NO_VALUE = "NO_VALUE"


def shared_parameter_warning_regex(param_name, shared_name=None):
    if shared_name is None:
        shared_name = param_name

    return (
        f'Specification of the "{param_name}" parameter.*conflicts'
        f' with specification of its shared parameter "{shared_name}"'
    )


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


def test_unspecified_inheritance():
    class NewTM(pnl.TransferMechanism):
        class Parameters(pnl.TransferMechanism.Parameters):
            pass

    assert NewTM.parameters.variable._inherited
    NewTM.parameters.variable.default_value = -1
    assert not NewTM.parameters.variable._inherited

    NewTM.parameters.variable.reset()
    assert NewTM.parameters.variable._inherited


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
        (pnl.AdaptiveIntegrator, {'rate': 0.5}, 'additive_param', False),
        (pnl.AdaptiveIntegrator, {'rate': 0.5}, 'rate', True),
        (pnl.AdaptiveIntegrator, {'rate': 0.5}, 'multiplicative_param', True),
        (pnl.TransferMechanism, {'integration_rate': None}, 'integration_rate', False),
        (pnl.TransferMechanism, {'integration_rate': 0.5}, 'integration_rate', True),
        (pnl.TransferMechanism, {'initial_value': 0}, 'initial_value', True),
        (pnl.TransferMechanism, {'initial_value': None}, 'initial_value', False),
        (pnl.TransferMechanism, {}, 'initial_value', False),
    ],
)
def test_user_specified(cls_, kwargs, parameter, is_user_specified):
    c = cls_(**kwargs)
    assert getattr(c.parameters, parameter)._user_specified == is_user_specified


@pytest.mark.parametrize(
    'kwargs, parameter, is_user_specified',
    [
        ({'function': pnl.Linear}, 'slope', False),
        ({'function': pnl.Linear()}, 'slope', False),
        ({'function': pnl.Linear(slope=1)}, 'slope', True),
    ]
)
def test_function_user_specified(kwargs, parameter, is_user_specified):
    t = pnl.TransferMechanism(**kwargs)
    assert getattr(t.function.parameters, parameter)._user_specified == is_user_specified


# sort param names or pytest-xdist may cause failure
# see https://github.com/pytest-dev/pytest/issues/4101
@pytest.mark.parametrize('attr', sorted(pnl.Parameter._additional_param_attr_properties))
def test_additional_param_attrs(attr):
    assert hasattr(pnl.Parameter, f'_set_{attr}'), (
        f'To include {attr} in Parameter._additional_param_attr_properties, you'
        f' must add a _set_{attr} method on Parameter. If this is unneeded,'
        ' remove it from Parameter._additional_param_attr_properties.'
    )


class TestSharedParameters:

    recurrent_mech = pnl.RecurrentTransferMechanism(default_variable=[0, 0], enable_learning=True)
    recurrent_mech_no_learning = pnl.RecurrentTransferMechanism(default_variable=[0, 0])
    transfer_with_costs = pnl.TransferWithCosts(default_variable=[0, 0])

    test_values = [
        (
            recurrent_mech,
            'learning_function',
            recurrent_mech.learning_mechanism.parameters.function
        ),
        (
            recurrent_mech,
            'learning_rate',
            recurrent_mech.learning_mechanism.parameters.learning_rate
        ),
        (
            transfer_with_costs,
            'transfer_fct_mult_param',
            transfer_with_costs.transfer_fct.parameters.multiplicative_param
        )
    ]

    @pytest.mark.parametrize(
        'obj, parameter_name, source',
        test_values + [
            (recurrent_mech_no_learning, 'learning_function', None),
        ]
    )
    def test_sources(self, obj, parameter_name, source):
        assert getattr(obj.parameters, parameter_name).source is source

    @pytest.mark.parametrize(
        'obj, parameter_name, source',
        test_values
    )
    def test_values(self, obj, parameter_name, source):
        obj_param = getattr(obj.parameters, parameter_name)
        eids = range(5)

        for eid in eids:
            obj.execute(np.array([eid, eid]), context=eid)

        assert all([
            obj_param.get(eid) is source.get(eid)
            for eid in eids
        ])

    @pytest.mark.parametrize(
        'obj, parameter_name, attr_name',
        [
            (transfer_with_costs, 'intensity_cost_fct_mult_param', 'modulable'),
            (recurrent_mech, 'learning_function', 'stateful'),
            (recurrent_mech, 'learning_function', 'loggable'),
            (recurrent_mech.recurrent_projection, 'auto', 'modulable'),
            (recurrent_mech, 'integration_rate', 'modulable'),
            (recurrent_mech, 'noise', 'modulable'),
        ]
    )
    def test_param_attrs_match(self, obj, parameter_name, attr_name):
        shared_param = getattr(obj.parameters, parameter_name)
        source_param = shared_param.source

        assert getattr(shared_param, attr_name) == getattr(source_param, attr_name)

        orig_values = shared_param.stateful, source_param.stateful

        # change value of shared attribute on source parameter
        source_param.stateful = not source_param.stateful
        assert getattr(shared_param, attr_name) == getattr(source_param, attr_name)

        shared_param.stateful, source_param.stateful = orig_values

        # change value of shared attribute on sharedparameter
        shared_param.stateful = not shared_param.stateful
        assert getattr(shared_param, attr_name) == getattr(source_param, attr_name)

        shared_param.stateful, source_param.stateful = orig_values

    @pytest.mark.parametrize(
        'integrator_function, expected_rate',
        [
            (pnl.AdaptiveIntegrator, pnl.TransferMechanism.defaults.integration_rate),
            (pnl.AdaptiveIntegrator(), pnl.TransferMechanism.defaults.integration_rate),
            (pnl.AdaptiveIntegrator(rate=.75), .75)
        ]
    )
    def test_override_tmech(self, integrator_function, expected_rate):
        t = pnl.TransferMechanism(integrator_function=integrator_function)
        assert t.integrator_function.defaults.rate == expected_rate
        assert t.integration_rate.modulated == t.integration_rate.base == expected_rate

    def test_conflict_warning(self):
        with pytest.warns(
            UserWarning,
            match=shared_parameter_warning_regex('integration_rate', 'rate')
        ):
            pnl.TransferMechanism(
                integration_rate=.1,
                integrator_function=pnl.AdaptiveIntegrator(rate=.2)
            )

    @pytest.mark.parametrize(
        'mech_type, param_name, shared_param_name, param_value',
        [
            (pnl.LCAMechanism, 'noise', 'noise', pnl.GaussianDistort),
            (pnl.LCAMechanism, 'noise', 'noise', pnl.GaussianDistort()),
            (pnl.TransferMechanism, 'noise', 'noise', pnl.NormalDist),
            (pnl.TransferMechanism, 'noise', 'noise', pnl.NormalDist()),
            (pnl.TransferMechanism, 'noise', 'noise', [pnl.NormalDist()]),
        ]
    )
    def test_conflict_no_warning(
        self,
        mech_type,
        param_name,
        shared_param_name,
        param_value
    ):
        # pytest doesn't support inverse warning assertion for specific
        # warning only
        with warnings.catch_warnings():
            warnings.simplefilter(action='error', category=UserWarning)
            try:
                mech_type(**{param_name: param_value})
            except UserWarning as w:
                if re.match(shared_parameter_warning_regex(param_name, shared_param_name), str(w)):
                    raise

    def test_conflict_no_warning_parser(self):
        # replace with different class/parameter if _parse_noise ever implemented
        assert not hasattr(pnl.AdaptiveIntegrator.Parameters, '_parse_noise')
        pnl.AdaptiveIntegrator.Parameters._parse_noise = lambda self, noise: 2 * noise

        # pytest doesn't support inverse warning assertion for specific
        # warning only
        with warnings.catch_warnings():
            warnings.simplefilter(action='error', category=UserWarning)
            try:
                pnl.TransferMechanism(
                    noise=2,
                    integrator_function=pnl.AdaptiveIntegrator(noise=1)
                )
            except UserWarning as w:
                if re.match(shared_parameter_warning_regex('noise'), str(w)):
                    raise

        delattr(pnl.AdaptiveIntegrator.Parameters, '_parse_noise')


class TestSpecificationType:
    @staticmethod
    def _create_params_class_variant(cls_param, init_param, parent_class=pnl.Component):
        # init_param as Parameter doesn't make sense, only check cls_param
        if cls_param is pnl.Parameter:
            cls_param = pnl.Parameter()

        if cls_param is NO_PARAMETERS:
            if init_param is NO_INIT:

                class TestComponent(parent_class):
                    pass

            else:

                class TestComponent(parent_class):
                    @pnl.core.globals.parameters.check_user_specified
                    def __init__(self, p=init_param):
                        super().__init__(p=p)

        elif cls_param is NO_VALUE:
            if init_param is NO_INIT:

                class TestComponent(parent_class):
                    class Parameters(parent_class.Parameters):
                        pass

            else:

                class TestComponent(parent_class):
                    class Parameters(parent_class.Parameters):
                        pass

                    @pnl.core.globals.parameters.check_user_specified
                    def __init__(self, p=init_param):
                        super().__init__(p=p)

        else:
            if init_param is NO_INIT:

                class TestComponent(parent_class):
                    class Parameters(parent_class.Parameters):
                        p = cls_param

            else:

                class TestComponent(parent_class):
                    class Parameters(parent_class.Parameters):
                        p = cls_param

                    @pnl.core.globals.parameters.check_user_specified
                    def __init__(self, p=init_param):
                        super().__init__(p=p)

        return TestComponent

    @pytest.mark.parametrize(
        "cls_param, init_param, param_default",
        [
            (1, 1, 1),
            (1, None, 1),
            (None, 1, 1),
            (1, NO_INIT, 1),
            ("foo", "foo", "foo"),
            (np.array(1), np.array(1), np.array(1)),
            (np.array([1]), np.array([1]), np.array([1])),
        ],
    )
    def test_valid_assignment(self, cls_param, init_param, param_default):
        TestComponent = TestSpecificationType._create_params_class_variant(cls_param, init_param)
        assert TestComponent.defaults.p == param_default
        assert TestComponent.parameters.p.default_value == param_default

    @pytest.mark.parametrize(
        "cls_param, init_param",
        [
            (1, 2),
            (2, 1),
            (1, 1.0),
            (np.array(1), 1),
            (np.array([1]), 1),
            (np.array([1]), np.array(1)),
            ("foo", "bar"),
        ],
    )
    def test_conflicting_assignments(self, cls_param, init_param):
        with pytest.raises(AssertionError, match="Conflicting default parameter"):
            TestSpecificationType._create_params_class_variant(cls_param, init_param)

    @pytest.mark.parametrize(
        "child_cls_param, child_init_param, parent_value, child_value",
        [
            (NO_PARAMETERS, NO_INIT, 1, 1),
            (NO_VALUE, NO_INIT, 1, 1),
            (2, NO_INIT, 1, 2),
            (NO_PARAMETERS, 2, 1, 2),
            (NO_VALUE, 2, 1, 2),
            (2, 2, 1, 2),
        ],
    )
    @pytest.mark.parametrize(
        "parent_cls_param, parent_init_param",
        [(1, 1), (1, None), (None, 1), (pnl.Parameter, 1)],
    )
    def test_inheritance(
        self,
        parent_cls_param,
        parent_init_param,
        child_cls_param,
        child_init_param,
        parent_value,
        child_value,
    ):
        TestParent = TestSpecificationType._create_params_class_variant(
            parent_cls_param, parent_init_param
        )
        TestChild = TestSpecificationType._create_params_class_variant(
            child_cls_param, child_init_param, parent_class=TestParent
        )

        assert TestParent.defaults.p == parent_value
        assert TestParent.parameters.p.default_value == parent_value

        assert TestChild.defaults.p == child_value
        assert TestChild.parameters.p.default_value == child_value

    @pytest.mark.parametrize("set_from_defaults", [True, False])
    @pytest.mark.parametrize(
        "child_cls_param, child_init_param",
        [(1, 1), (1, None), (None, 1), (NO_PARAMETERS, 1), (1, NO_INIT)],
    )
    @pytest.mark.parametrize("parent_cls_param, parent_init_param", [(0, 0), (0, None)])
    def test_set_and_reset(
        self,
        parent_cls_param,
        parent_init_param,
        child_cls_param,
        child_init_param,
        set_from_defaults,
    ):
        def set_p_default(obj, val):
            if set_from_defaults:
                obj.defaults.p = val
            else:
                obj.parameters.p.default_value = val

        TestParent = TestSpecificationType._create_params_class_variant(
            parent_cls_param, parent_init_param
        )
        TestChild = TestSpecificationType._create_params_class_variant(
            child_cls_param, child_init_param, parent_class=TestParent
        )
        TestGrandchild = TestSpecificationType._create_params_class_variant(
            NO_PARAMETERS, NO_INIT, parent_class=TestChild
        )

        set_p_default(TestChild, 10)
        assert TestParent.defaults.p == 0
        assert TestChild.defaults.p == 10
        assert TestGrandchild.defaults.p == 10

        set_p_default(TestGrandchild, 20)
        assert TestParent.defaults.p == 0
        assert TestChild.defaults.p == 10
        assert TestGrandchild.defaults.p == 20

        TestChild.parameters.p.reset()
        assert TestParent.defaults.p == 0
        assert TestChild.defaults.p == 1
        assert TestGrandchild.defaults.p == 20

        TestGrandchild.parameters.p.reset()
        assert TestParent.defaults.p == 0
        assert TestChild.defaults.p == 1
        assert TestGrandchild.defaults.p == 1

        set_p_default(TestGrandchild, 20)
        assert TestParent.defaults.p == 0
        assert TestChild.defaults.p == 1
        assert TestGrandchild.defaults.p == 20
