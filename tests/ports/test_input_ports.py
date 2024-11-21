import numpy as np
import pytest

import psyneulink as pnl
import psyneulink.core.components.functions.nonstateful.transformfunctions
import psyneulink.core.components.functions.nonstateful.transferfunctions


class TestInputPorts:

    def test_combine_param_alone(self):
        t1 = pnl.TransferMechanism(input_shapes=2)
        t2 = pnl.TransferMechanism(input_shapes=2)
        t3 = pnl.TransferMechanism(
                input_shapes=2,
                input_ports=pnl.InputPort(
                        combine=pnl.PRODUCT))
        c = pnl.Composition(pathways=[[t1, t3], [t2, t3]])
        input_dict = {t1:[1,2],t2:[3,4]}
        val = c.run(inputs=input_dict)
        np.testing.assert_allclose(val, [[3, 8]])

    def test_combine_param_redundant_fct_class_spec(self):
        t1 = pnl.TransferMechanism(input_shapes=2)
        t2 = pnl.TransferMechanism(input_shapes=2)
        t3 = pnl.TransferMechanism(
                input_shapes=2,
                input_ports=pnl.InputPort(function=psyneulink.core.components.functions.nonstateful.transformfunctions
                                           .LinearCombination,
                                           combine=pnl.PRODUCT))
        c = pnl.Composition(pathways=[[t1, t3],[t2, t3]])
        input_dict = {t1:[1,2],t2:[3,4]}
        val = c.run(inputs=input_dict)
        np.testing.assert_allclose(val, [[3, 8]])

    def test_combine_param_redundant_fct_constructor_spec(self):
        t1 = pnl.TransferMechanism(input_shapes=2)
        t2 = pnl.TransferMechanism(input_shapes=2)
        t3 = pnl.TransferMechanism(
                input_shapes=2,
                input_ports=pnl.InputPort(function=psyneulink.core.components.functions.nonstateful.transformfunctions.LinearCombination(operation=pnl.PRODUCT),
                                          combine=pnl.PRODUCT))
        c = pnl.Composition(pathways=[[t1, t3],[t2, t3]])
        input_dict = {t1:[1,2],t2:[3,4]}
        val = c.run(inputs=input_dict)
        np.testing.assert_allclose(val, [[3, 8]])

    def test_combine_param_conflicting_fct_operation_spec(self):
        with pytest.raises(pnl.InputPortError) as error_text:
            t = pnl.TransferMechanism(input_ports=pnl.InputPort(function=psyneulink.core.components.functions.nonstateful.transformfunctions.LinearCombination(operation=pnl.SUM),
                                                                combine=pnl.PRODUCT))
        assert "Specification of 'combine' argument (PRODUCT) conflicts with specification of 'operation' (SUM) " \
               "for LinearCombination in 'function' argument for InputPort" in str(error_text.value)

    def test_combine_param_conflicting_function_spec(self):
        with pytest.raises(pnl.InputPortError) as error_text:
            t = pnl.TransferMechanism(input_ports=pnl.InputPort(function=psyneulink.core.components.functions.nonstateful.transferfunctions.Linear(), combine=pnl.PRODUCT))
        assert "Specification of 'combine' argument (PRODUCT) conflicts with Function specified " \
               "in 'function' argument (Linear Function" in str(error_text.value)

    def test_combine_param_conflicting_fct_class_spec(self):
        with pytest.raises(pnl.InputPortError) as error_text:
            t = pnl.TransferMechanism(input_ports=pnl.InputPort(function=psyneulink.core.components.functions.nonstateful.transferfunctions.Linear, combine=pnl.PRODUCT))
        assert "Specification of 'combine' argument (PRODUCT) conflicts with Function specified " \
               "in 'function' argument (Linear) for InputPort" in str(error_text.value)

    def test_combine_dict_spec(self):
        t = pnl.TransferMechanism(input_ports={pnl.COMBINE: pnl.PRODUCT})
        assert t.input_port.function.operation == pnl.PRODUCT

    def test_equivalent_function_dict_spec(self):
        t = pnl.TransferMechanism(input_ports={pnl.FUNCTION:pnl.LinearCombination(operation=pnl.PRODUCT)})
        assert t.input_port.function.operation == pnl.PRODUCT

    def test_combine_dict_spec_redundant_with_function(self):
        with pytest.warns(UserWarning) as warnings:  # Warn, since default_input is NOT set
            t = pnl.TransferMechanism(input_ports={pnl.COMBINE:pnl.PRODUCT,
                                                   pnl.FUNCTION:pnl.LinearCombination(operation=pnl.PRODUCT)})
        assert any(w.message.args[0] == "Both COMBINE ('product') and FUNCTION (LinearCombination Function) "
                                              "specifications found in InputPort specification dictionary for "
                                              "'InputPort' of 'TransferMechanism-0'; no need to specify both."
                   for w in warnings)
        assert t.input_port.function.operation == pnl.PRODUCT

    def test_combine_dict_spec_conflicts_with_function(self):
        with pytest.raises(pnl.InputPortError) as error_text:
            t = pnl.TransferMechanism(input_ports={pnl.COMBINE:pnl.PRODUCT,
                                                   pnl.FUNCTION:pnl.LinearCombination})
        assert "COMBINE entry (='product') of InputPort specification dictionary for 'InputPort' of " \
               "'TransferMechanism-0' conflicts with FUNCTION entry (LinearCombination Function); " \
               "remove one or the other." in str(error_text.value)

    def test_single_projection_variable(self):
        a = pnl.TransferMechanism()
        b = pnl.TransferMechanism()

        pnl.MappingProjection(sender=a, receiver=b)

        assert b.input_port.defaults.variable.shape == np.array([0]).shape
        assert b.input_port.function.defaults.variable.shape == np.array([0]).shape

    @pytest.mark.parametrize('num_incoming_projections', [2, 3, 4])
    def test_adding_projections_modifies_variable(self, num_incoming_projections):
        mechs = [pnl.TransferMechanism() for _ in range(num_incoming_projections + 1)]
        [pnl.MappingProjection(sender=mechs[i], receiver=mechs[-1]) for i in range(num_incoming_projections)]

        receiver_input_port_variable = np.array([[0] for _ in range(num_incoming_projections)])

        assert mechs[-1].input_port.defaults.variable.shape == receiver_input_port_variable.shape
        assert mechs[-1].input_port.function.defaults.variable.shape == receiver_input_port_variable.shape

    def test_input_port_variable_shapes(self):
        t = pnl.TransferMechanism(input_ports=[{pnl.VARIABLE: [[0], [0]]}])

        assert t.input_port.defaults.variable.shape == np.array([[0], [0]]).shape
        assert t.input_port.defaults.value.shape == np.array([0]).shape

        assert t.input_port.function.defaults.variable.shape == np.array([[0], [0]]).shape
        assert t.input_port.function.defaults.value.shape == np.array([0]).shape

    def test_internal_only(self):
        m = pnl.TransferMechanism(input_ports=['EXTERNAL', pnl.InputPort(name='INTERNAL_ONLY', internal_only=True)])
        assert m.input_values == [[ 0.],[ 0.]]
        assert m.external_input_values == [[0.]]

    @pytest.mark.parametrize('default_input', [None, pnl.DEFAULT_VARIABLE])
    def test_default_input(self, default_input):
        variable = [22]
        m = pnl.TransferMechanism(input_ports=[pnl.InputPort(name='INTERNAL_NODE',
                                                             default_input=default_input,
                                                             variable=variable)])
        m.execute()
        assert m.input_port.value == variable
        if default_input:
            assert m.input_port.internal_only is True
        else:
            assert m.input_port.internal_only is False
        comp = pnl.Composition()
        comp.add_node(
            m,
            required_roles=pnl.NodeRole.INTERNAL,
            context=pnl.Context(source=pnl.ContextFlags.METHOD)
        )
        comp._analyze_graph()
        assert pnl.NodeRole.INTERNAL in comp.get_roles_by_node(m)
        assert pnl.NodeRole.INPUT not in comp.get_roles_by_node(m)

        assert not m.path_afferents  # No path_afferents since internal_only is set by default_input


        if default_input is None:
            with pytest.warns(UserWarning) as warnings:  # Warn, since default_input is NOT set
                comp.run()
            assert any(repr(w.message.args[0]) == '"InputPort (\'INTERNAL_NODE\') of \'TransferMechanism-0\' '
                                                  'doesn\'t have any afferent Projections."'
                       for w in warnings)
        else:
            comp.run()                   # No warning since default_input is set

        assert m.input_port.value == variable # For Mechanisms other than controller, default_variable seems
        assert m.value == variable            #     to still be used even though default_input is NOT set

    def test_no_efferents(self):
        A = pnl.InputPort()
        with pytest.raises(pnl.PortError) as error:
            A.efferents
        assert 'InputPorts do not have \'efferents\'; (access attempted for Deferred Init InputPort).' \
               in str(error.value)
        with pytest.raises(pnl.PortError) as error:
            A.efferents = ['test']
        assert 'InputPorts are not allowed to have \'efferents\' ' \
               '(assignment attempted for Deferred Init InputPort).' in str(error.value)


class TestDefaultInput:
    def test_default_input_standalone(self):
        a = pnl.ProcessingMechanism(
            input_ports=[
                {pnl.VARIABLE: [2], pnl.PARAMS: {pnl.DEFAULT_INPUT: pnl.DEFAULT_VARIABLE}}
            ]
        )
        np.testing.assert_array_equal(a.execute(), [[2]])

        a.input_ports[0].defaults.variable = [3]
        np.testing.assert_array_equal(a.execute(), [[3]])

    def test_default_input_standalone_two_ports(self):
        a = pnl.ProcessingMechanism(default_variable=[[1], [1]])
        a.input_ports[0].defaults.variable = [2]
        a.input_ports[1].defaults.variable = [3]

        # no port default_input set, use mechanism.defaults.variable
        np.testing.assert_array_equal(a.execute(), [[1], [1]])

        # port has default_input set, so get variable from input ports.
        # second port has no afferents, so it doesn't have a value
        a.input_ports[0].parameters.default_input.set(pnl.DEFAULT_VARIABLE, override=True)
        with pytest.raises(
            pnl.FunctionError, match="may be due to missing afferent projection"
        ):
            a.execute()

        # both ports have default_input set, so use it
        a.input_ports[1].parameters.default_input.set(pnl.DEFAULT_VARIABLE, override=True)
        np.testing.assert_array_equal(a.execute(), [[2], [3]])

        # as second check above. one port has default_input, other does
        # not and has no afferents
        a.input_ports[0].parameters.default_input.set(None, override=True)
        with pytest.raises(
            pnl.FunctionError, match="may be due to missing afferent projection"
        ):
            a.execute()

        # as first check above. no port default_input set, use
        # mechanism.defaults.variable
        a.input_ports[1].parameters.default_input.set(None, override=True)
        np.testing.assert_array_equal(a.execute(), [[1], [1]])

    def test_default_input_with_projection(self):
        a = pnl.ProcessingMechanism(default_variable=[[1]])
        b = pnl.ProcessingMechanism(
            input_ports=[
                {pnl.VARIABLE: [2], pnl.PARAMS: {pnl.DEFAULT_INPUT: pnl.DEFAULT_VARIABLE}}
            ]
        )
        comp = pnl.Composition(pathways=[a, b])

        inputs = {a: [[10]]}

        np.testing.assert_array_equal(comp.execute(), [[2]])
        np.testing.assert_array_equal(comp.run(inputs), [[2]])

        b.input_ports[0].defaults.variable = [3]
        np.testing.assert_array_equal(comp.execute(), [[3]])
        np.testing.assert_array_equal(comp.run(inputs), [[3]])

        b.input_ports[0].parameters.default_input.set(None, override=True)
        np.testing.assert_array_equal(comp.execute(), [[1]])
        np.testing.assert_array_equal(comp.run(inputs), [[10]])

    def test_default_input_with_projections_two_ports(self):
        a = pnl.ProcessingMechanism(default_variable=[[1]])
        b = pnl.ProcessingMechanism(
            input_ports=[
                {pnl.VARIABLE: [2]},
                {pnl.VARIABLE: [3], pnl.PARAMS: {pnl.DEFAULT_INPUT: pnl.DEFAULT_VARIABLE}},
            ]
        )
        comp = pnl.Composition()
        comp.add_nodes([a, b])
        comp.add_projection(sender=a, receiver=b.input_ports[0])

        inputs = {a: [[10]]}

        np.testing.assert_array_equal(comp.run(inputs), [[10], [3]])

        b.input_ports[0].parameters.default_input.set(pnl.DEFAULT_VARIABLE, override=True)
        np.testing.assert_array_equal(comp.run(inputs), [[2], [3]])

        b.input_ports[0].defaults.variable = [4]
        np.testing.assert_array_equal(comp.run(inputs), [[4], [3]])

        b.input_ports[1].defaults.variable = [5]
        np.testing.assert_array_equal(comp.run(inputs), [[4], [5]])

        b.input_ports[0].parameters.default_input.set(None, override=True)
        np.testing.assert_array_equal(comp.run(inputs), [[10], [5]])

        b.input_ports[1].parameters.default_input.set(None, override=True)
        with pytest.raises(
            pnl.FunctionError, match="may be due to missing afferent projection"
        ):
            with pytest.warns(
                UserWarning, match="'ProcessingMechanism-1' doesn't have any afferent Projections"
            ):
                comp.run(inputs)
