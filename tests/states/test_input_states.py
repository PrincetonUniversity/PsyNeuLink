import psyneulink as pnl
import numpy as np
import pytest

class TestInputStates:

    def test_combine_param_alone(self):
        t1 = pnl.TransferMechanism(size=2)
        t2 = pnl.TransferMechanism(size=2)
        t3 = pnl.TransferMechanism(
                size=2,
                input_states=pnl.InputState(
                        combine=pnl.PRODUCT))
        p1 = pnl.Process(pathway=[t1, t3])
        p2 = pnl.Process(pathway=[t2, t3])
        s = pnl.System(processes=[p1,p2])
        input_dict = {t1:[1,2],t2:[3,4]}
        val = s.run(inputs=input_dict)
        assert np.allclose(val, [[3, 8]])

    def test_combine_param_redundant_fct_class_spec(self):
        t1 = pnl.TransferMechanism(size=2)
        t2 = pnl.TransferMechanism(size=2)
        t3 = pnl.TransferMechanism(
                size=2,
                input_states=pnl.InputState(function=pnl.LinearCombination,
                                            combine=pnl.PRODUCT))
        p1 = pnl.Process(pathway=[t1, t3])
        p2 = pnl.Process(pathway=[t2, t3])
        s = pnl.System(processes=[p1,p2])
        input_dict = {t1:[1,2],t2:[3,4]}
        val = s.run(inputs=input_dict)
        assert np.allclose(val, [[3, 8]])

    def test_combine_param_redundant_fct_constructor_spec(self):
        t1 = pnl.TransferMechanism(size=2)
        t2 = pnl.TransferMechanism(size=2)
        t3 = pnl.TransferMechanism(
                size=2,
                input_states=pnl.InputState(function=pnl.LinearCombination(operation=pnl.PRODUCT),
                                            combine=pnl.PRODUCT))
        p1 = pnl.Process(pathway=[t1, t3])
        p2 = pnl.Process(pathway=[t2, t3])
        s = pnl.System(processes=[p1,p2])
        input_dict = {t1:[1,2],t2:[3,4]}
        val = s.run(inputs=input_dict)
        assert np.allclose(val, [[3, 8]])

    def test_combine_param_conflicting_fct_operation_spec(self):
        with pytest.raises(pnl.InputStateError) as error_text:
            t = pnl.TransferMechanism(input_states=pnl.InputState(function=pnl.LinearCombination(operation=pnl.SUM),
                                                                  combine=pnl.PRODUCT))
        assert "Specification of 'combine' argument (PRODUCT) conflicts with specification of 'operation' (SUM) " \
               "for LinearCombination in 'function' argument for InputState" in str(error_text.value)

    def test_combine_param_conflicting_function_spec(self):
        with pytest.raises(pnl.InputStateError) as error_text:
            t = pnl.TransferMechanism(input_states=pnl.InputState(function=pnl.Linear(), combine=pnl.PRODUCT))
        assert "Specification of 'combine' argument (PRODUCT) conflicts with Function specified " \
               "in 'function' argument (Linear Function" in str(error_text.value)

    def test_combine_param_conflicting_fct_class_spec(self):
        with pytest.raises(pnl.InputStateError) as error_text:
            t = pnl.TransferMechanism(input_states=pnl.InputState(function=pnl.Linear, combine=pnl.PRODUCT))
        assert "Specification of 'combine' argument (PRODUCT) conflicts with Function specified " \
               "in 'function' argument (Linear) for InputState" in str(error_text.value)
