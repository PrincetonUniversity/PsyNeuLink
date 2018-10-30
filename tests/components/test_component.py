import numpy as np
import psyneulink as pnl
import pytest

class TestComponent:

    def test_component_execution_counts_for_standalone_mechanism(self):
        '''Note: input_state should not update execution count, since it has no afferents'''

        T = pnl.TransferMechanism()

        T.execute()
        assert T.current_execution_count == 1
        assert T.input_state.current_execution_count == 0
        assert T.parameter_states[pnl.SLOPE].current_execution_count == 1
        assert T.output_state.current_execution_count == 1

        T.execute()
        assert T.current_execution_count == 2
        assert T.input_state.current_execution_count == 0
        assert T.parameter_states[pnl.SLOPE].current_execution_count == 2
        assert T.output_state.current_execution_count == 2

        T.execute()
        assert T.current_execution_count == 3
        assert T.input_state.current_execution_count == 0
        assert T.parameter_states[pnl.SLOPE].current_execution_count == 3
        assert T.output_state.current_execution_count == 3

    def test_component_execution_counts_for_mechanisms_in_composition(self):

        T1 = pnl.TransferMechanism()
        T2 = pnl.TransferMechanism()
        c = pnl.Composition()
        c.add_c_node(T1)
        c.add_c_node(T2)
        c.add_projection(sender=T1, receiver=T2)

        input_dict = {T1:[[0]]}

        c.run(input_dict)
        assert T2.current_execution_count == 1
        assert T2.input_state.current_execution_count == 1
        assert T2.parameter_states[pnl.SLOPE].current_execution_count == 1
        assert T2.output_state.current_execution_count == 1

        c.run(input_dict)
        assert T2.current_execution_count == 2
        assert T2.input_state.current_execution_count == 2
        assert T2.parameter_states[pnl.SLOPE].current_execution_count == 2
        assert T2.output_state.current_execution_count == 2

        c.run(input_dict)
        assert T2.current_execution_count == 3
        assert T2.input_state.current_execution_count == 3
        assert T2.parameter_states[pnl.SLOPE].current_execution_count == 3
        assert T2.output_state.current_execution_count == 3
