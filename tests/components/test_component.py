import numpy as np
import psyneulink as pnl
import pytest


def nest_dictionary(elem, keys=NotImplemented):
    """
    Args:
        elem
        keys (object, iterable, optional)

    Returns:
        dict: **elem** if **keys** is NotImplemented, or a dictionary
        containing **elem** nested by **keys**
    """
    if keys is NotImplemented:
        return elem

    if isinstance(keys, str):
        keys = [keys]

    try:
        iter(keys)
    except TypeError:
        keys = [keys]

    res = elem
    for k in reversed(keys):
        res = {k: res}
    return res


class TestComponent:

    def test_detection_of_legal_arg_in_kwargs(self):
        assert isinstance(pnl.ProcessingMechanism().reset_stateful_function_when, pnl.Never)
        assert isinstance(pnl.ProcessingMechanism(reset_stateful_function_when=pnl.AtTrialStart()).reset_stateful_function_when,
                          pnl.AtTrialStart)

    def test_detection_of_illegal_arg_in_kwargs(self):
        with pytest.raises(pnl.ComponentError) as error_text:
            pnl.ProcessingMechanism(flim_flam=1)
        assert "ProcessingMechanism-0: Illegal argument in constructor (type: ProcessingMechanism):" in str(error_text.value)
        assert "'flim_flam'" in str(error_text.value)

    def test_detection_of_illegal_args_in_kwargs(self):
        with pytest.raises(pnl.ComponentError) as error_text:
            pnl.ProcessingMechanism(name='MY_MECH', flim_flam=1, grumblabble=2)
        assert "MY_MECH: Illegal arguments in constructor (type: ProcessingMechanism):" in str(error_text.value)
        assert "'flim_flam'" in str(error_text.value)
        assert "'grumblabble'" in str(error_text.value)

    def test_component_execution_counts_for_standalone_mechanism(self):

        T = pnl.TransferMechanism()

        T.execute()
        assert T.execution_count == 1
        assert T.input_port.execution_count == 1 # incremented by Mechanism.get_variable_from_input()

        # skipped (0 executions) because execution is bypassed when no afferents, and
        # function._is_identity is satisfied (here, Linear function with slope 0 and intercept 1)
        # This holds true for each below
        assert T.parameter_ports[pnl.SLOPE].execution_count == 0
        assert T.output_port.execution_count == 0

        T.execute()
        assert T.execution_count == 2
        assert T.input_port.execution_count == 2
        assert T.parameter_ports[pnl.SLOPE].execution_count == 0
        assert T.output_port.execution_count == 0

        T.execute()
        assert T.execution_count == 3
        assert T.input_port.execution_count == 3
        assert T.parameter_ports[pnl.SLOPE].execution_count == 0
        assert T.output_port.execution_count == 0

    def test_component_execution_counts_for_mechanisms_in_composition(self):

        T1 = pnl.TransferMechanism()
        T2 = pnl.TransferMechanism()
        c = pnl.Composition()
        c.add_node(T1)
        c.add_node(T2)
        c.add_projection(sender=T1, receiver=T2)

        input_dict = {T1:[[0]]}

        c.run(input_dict)
        assert T2.execution_count == 1
        assert T2.input_port.execution_count == 1
        assert T2.parameter_ports[pnl.SLOPE].execution_count == 0
        assert T2.output_port.execution_count == 0

        c.run(input_dict)
        assert T2.execution_count == 2
        assert T2.input_port.execution_count == 2
        assert T2.parameter_ports[pnl.SLOPE].execution_count == 0
        assert T2.output_port.execution_count == 0

        c.run(input_dict)
        assert T2.execution_count == 3
        assert T2.input_port.execution_count == 3
        assert T2.parameter_ports[pnl.SLOPE].execution_count == 0
        assert T2.output_port.execution_count == 0

    def test__set_all_parameter_properties_recursively(self):
        A = pnl.ProcessingMechanism(name='A')
        A._set_all_parameter_properties_recursively(history_max_length=0)

        for c in A._dependent_components:
            for param in c.parameters:
                assert param.history_max_length == 0

    @pytest.mark.parametrize(
        'component_type', [
            pnl.ProcessingMechanism,
            pnl.TransferMechanism,
            pnl.Linear,
            pnl.DDM
        ]
    )
    def test_execute_manual_context(self, component_type):
        c = component_type()
        default_result = c.execute(5)

        assert pnl.safe_equals(c.execute(5, context='new'), default_result)


class TestConstructorArguments:
    class NewTestMech(pnl.Mechanism_Base):
        deprecated_constructor_args = {
            **pnl.Mechanism_Base.deprecated_constructor_args,
            **{
                'deprecated_param': 'new_param',
            }
        }

        class Parameters(pnl.Mechanism_Base.Parameters):
            cca_param = pnl.Parameter('A', constructor_argument='cca_constr')
            param_with_alias = pnl.Parameter(None, constructor_argument='pwa_constr_arg', aliases=['pwa_alias'])
            param_with_alias_spec_none = pnl.Parameter(None, aliases=['pwasn_alias'], specify_none=True)

        def __init__(self, default_variable=None, **kwargs):
            super().__init__(default_variable=default_variable, **kwargs)

    @pytest.mark.parametrize(
        'cls_', [pnl.ProcessingMechanism, pnl.TransferMechanism, pnl.IntegratorMechanism]
    )
    @pytest.mark.parametrize(
        'input_shapes, expected_variable',
        [
            (1, [[0]]),
            (2, [[0, 0]]),
            (3, [[0, 0, 0]]),
            ((1, 1), [[0], [0]]),
            ((2, 2), [[0, 0], [0, 0]]),
            ((3, 3), [[0, 0, 0], [0, 0, 0]]),
        ]
    )
    @pytest.mark.parametrize('params_dict_entry', [NotImplemented, 'params'])
    def test_input_shapes(self, cls_, params_dict_entry, input_shapes, expected_variable):
        c = cls_(**nest_dictionary({'input_shapes': input_shapes}, params_dict_entry))
        np.testing.assert_array_equal(c.defaults.variable, expected_variable)

    @pytest.mark.parametrize(
        'cls_, function_params, expected_values',
        [
            (pnl.ProcessingMechanism, {'slope': 2}, NotImplemented),
        ]
    )
    @pytest.mark.parametrize('params_dict_entry', [NotImplemented, 'params'])
    def test_function_params(self, cls_, function_params, expected_values, params_dict_entry):
        m = cls_(**nest_dictionary({'function_params': function_params}, params_dict_entry))

        if expected_values is NotImplemented:
            expected_values = function_params

        for k, v in expected_values.items():
            np.testing.assert_array_equal(getattr(m.function.defaults, k), v)

    @pytest.mark.parametrize(
        'cls_, function, function_params, err_msg',
        [
            (pnl.ProcessingMechanism, pnl.DriftDiffusionIntegrator, {'invalid_arg': 0}, 'Illegal argument in constructor (type: DriftDiffusionIntegrator)'),
            (pnl.ProcessingMechanism, pnl.DriftDiffusionIntegrator, {'initializer': 0, 'starting_value': 1}, 'starting_value is an alias of initializer'),
            pytest.param(
                pnl.ProcessingMechanism, pnl.LeakyCompetingIntegrator, {'invalid_arg': 0}, {},
                marks=pytest.mark.xfail(reason='kwargs are not passed up __init__, are just for backward compatibility for single parameter')
            ),
            (pnl.ProcessingMechanism, pnl.Linear, {'invalid_arg': 0}, "unexpected keyword argument 'invalid_arg'"),
        ]
    )
    @pytest.mark.parametrize('params_dict_entry', [NotImplemented, 'params'])
    def test_function_params_invalid(self, cls_, function, function_params, err_msg, params_dict_entry):
        with pytest.raises(pnl.ComponentError) as err:
            cls_(
                function=function,
                **nest_dictionary({'function_params': function_params}, params_dict_entry)
            )
        assert err_msg in str(err.value)

    @pytest.mark.parametrize(
        'cls_, param_name, argument_name, param_value',
        [
            (pnl.TransferMechanism, 'variable', 'default_variable', [[10]]),
            (NewTestMech, 'cca_param', 'cca_constr', 1),
            (NewTestMech, 'param_with_alias', 'pwa_constr_arg', 1),
            (NewTestMech, 'param_with_alias', 'pwa_alias', 1),
        ]
    )
    @pytest.mark.parametrize('params_dict_entry', [NotImplemented, 'params'])
    def test_valid_argument(self, cls_, param_name, argument_name, param_value, params_dict_entry):
        obj = cls_(**nest_dictionary({argument_name: param_value}, params_dict_entry))
        np.testing.assert_array_equal(getattr(obj.defaults, param_name), param_value)

    @pytest.mark.parametrize(
        'cls_, argument_name, param_value',
        [
            (NewTestMech, 'cca_param', 1),
            (NewTestMech, 'param_with_alias', 1),
            (pnl.TransferMechanism, 'variable', [[10]]),
        ]
    )
    @pytest.mark.parametrize('params_dict_entry', [NotImplemented, 'params'])
    def test_invalid_argument(self, cls_, argument_name, param_value, params_dict_entry):
        with pytest.raises(pnl.ComponentError) as err:
            cls_(**nest_dictionary({argument_name: param_value}, params_dict_entry))
        assert 'Illegal argument in constructor' in str(err.value)
        assert f'(type: {cls_.__name__})' in str(err.value)

        constr_arg = getattr(cls_.parameters, argument_name).constructor_argument
        assert f"'{argument_name}': must use '{constr_arg}' instead" in str(err.value)

    @pytest.mark.parametrize(
        'cls_, argument_name, new_name',
        [
            (NewTestMech, 'deprecated_param', 'new_param'),
            (NewTestMech, 'size', 'input_shapes'),
            (pnl.TransferMechanism, 'size', 'input_shapes'),
        ]
    )
    @pytest.mark.parametrize('params_dict_entry', [NotImplemented, 'params'])
    def test_invalid_argument_deprecated(self, cls_, argument_name, new_name, params_dict_entry):
        with pytest.raises(
            pnl.ComponentError,
            match=(
                rf".*Illegal argument in constructor \(type: {cls_.__name__}\):"
                f"\n\t'{argument_name}' is deprecated. Use '{new_name}' instead"
            )
        ):
            cls_(**nest_dictionary({argument_name: new_name}, params_dict_entry))

    @pytest.mark.parametrize(
        'cls_, param_name, param_value, alias_name, alias_value',
        [
            (pnl.DriftDiffusionIntegrator, 'initializer', 1, 'starting_value', 2),
            (NewTestMech, 'pwa_constr_arg', 1, 'pwa_alias', 2),
            (NewTestMech, 'param_with_alias_spec_none', 1, 'pwasn_alias', None),
            (NewTestMech, 'param_with_alias_spec_none', None, 'pwasn_alias', 1),
        ]
    )
    @pytest.mark.parametrize('params_dict_entry', [NotImplemented, 'params'])
    def test_conflicting_aliases(
        self, cls_, param_name, param_value, alias_name, alias_value, params_dict_entry
    ):
        with pytest.raises(pnl.ComponentError) as err:
            cls_(
                **nest_dictionary(
                    {param_name: param_value, alias_name: alias_value}, params_dict_entry
                )
            )

        assert 'Multiple values' in str(err.value)
        assert f'{param_name}: {param_value}' in str(err.value)
        assert f'{alias_name}: {alias_value}' in str(err.value)
        assert f'{alias_name} is an alias of {param_name}' in str(err.value)

    @pytest.mark.parametrize(
        'cls_, param_name, alias_name',
        [
            (pnl.DriftDiffusionIntegrator, 'initializer', 'starting_value'),
            (NewTestMech, 'pwa_constr_arg', 'pwa_alias'),
        ]
    )
    @pytest.mark.parametrize('param_value', [None, 1])
    @pytest.mark.parametrize('alias_value', [None, 1])
    @pytest.mark.parametrize('params_dict_entry', [NotImplemented, 'params'])
    def test_nonconflicting_aliases(
        self, cls_, param_name, alias_name, param_value, alias_value, params_dict_entry
    ):
        cls_(
            **nest_dictionary(
                {param_name: param_value, alias_name: alias_value}, params_dict_entry
            )
        )

    @pytest.mark.parametrize('param_value, alias_value', [(None, None), (1, 1)])
    @pytest.mark.parametrize('params_dict_entry', [NotImplemented, 'params'])
    def test_nonconflicting_aliases_specify_none(self, param_value, alias_value, params_dict_entry):
        self.NewTestMech(
            **nest_dictionary(
                {'param_with_alias_spec_none': param_value, 'pwasn_alias': alias_value},
                params_dict_entry
            )
        )
