import numpy as np
import psyneulink as pnl
import pytest

import psyneulink.core.components.functions.combinationfunctions
import psyneulink.core.components.functions.transferfunctions


class TestInputPorts:

    def test_combine_param_alone(self):
        t1 = pnl.TransferMechanism(size=2)
        t2 = pnl.TransferMechanism(size=2)
        t3 = pnl.TransferMechanism(
                size=2,
                input_ports=pnl.InputPort(
                        combine=pnl.PRODUCT))
        c = pnl.Composition(pathways=[[t1, t3], [t2, t3]])
        input_dict = {t1:[1,2],t2:[3,4]}
        val = c.run(inputs=input_dict)
        assert np.allclose(val, [[3, 8]])

    def test_combine_param_redundant_fct_class_spec(self):
        t1 = pnl.TransferMechanism(size=2)
        t2 = pnl.TransferMechanism(size=2)
        t3 = pnl.TransferMechanism(
                size=2,
                input_ports=pnl.InputPort(function=psyneulink.core.components.functions.combinationfunctions
                                           .LinearCombination,
                                           combine=pnl.PRODUCT))
        c = pnl.Composition(pathways=[[t1, t3],[t2, t3]])
        input_dict = {t1:[1,2],t2:[3,4]}
        val = c.run(inputs=input_dict)
        assert np.allclose(val, [[3, 8]])

    def test_combine_param_redundant_fct_constructor_spec(self):
        t1 = pnl.TransferMechanism(size=2)
        t2 = pnl.TransferMechanism(size=2)
        t3 = pnl.TransferMechanism(
                size=2,
                input_ports=pnl.InputPort(function=psyneulink.core.components.functions.combinationfunctions.LinearCombination(operation=pnl.PRODUCT),
                                           combine=pnl.PRODUCT))
        c = pnl.Composition(pathways=[[t1, t3],[t2, t3]])
        input_dict = {t1:[1,2],t2:[3,4]}
        val = c.run(inputs=input_dict)
        assert np.allclose(val, [[3, 8]])

    def test_combine_param_conflicting_fct_operation_spec(self):
        with pytest.raises(pnl.InputPortError) as error_text:
            t = pnl.TransferMechanism(input_ports=pnl.InputPort(function=psyneulink.core.components.functions.combinationfunctions.LinearCombination(operation=pnl.SUM),
                                                                 combine=pnl.PRODUCT))
        assert "Specification of 'combine' argument (PRODUCT) conflicts with specification of 'operation' (SUM) " \
               "for LinearCombination in 'function' argument for InputPort" in str(error_text.value)

    def test_combine_param_conflicting_function_spec(self):
        with pytest.raises(pnl.InputPortError) as error_text:
            t = pnl.TransferMechanism(input_ports=pnl.InputPort(function=psyneulink.core.components.functions
                                                                 .transferfunctions.Linear(), combine=pnl.PRODUCT))
        assert "Specification of 'combine' argument (PRODUCT) conflicts with Function specified " \
               "in 'function' argument (Linear Function" in str(error_text.value)

    def test_combine_param_conflicting_fct_class_spec(self):
        with pytest.raises(pnl.InputPortError) as error_text:
            t = pnl.TransferMechanism(input_ports=pnl.InputPort(function=psyneulink.core.components.functions.transferfunctions.Linear, combine=pnl.PRODUCT))
        assert "Specification of 'combine' argument (PRODUCT) conflicts with Function specified " \
               "in 'function' argument (Linear) for InputPort" in str(error_text.value)

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
