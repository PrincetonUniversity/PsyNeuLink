import numpy as np
import psyneulink as pnl
import pytest

class TestComponent:

    def test_component_execution_counts(self):

        T = pnl.TransferMechanism()

        T.execute()
        assert T.current_execution_count == 1
        # assert T.input_state.current_execution_count == 1
        assert T.parameter_states[pnl.SLOPE].current_execution_count == 1
        assert T.output_state.current_execution_count == 1

        T.execute()
        assert T.current_execution_count == 2
        # assert T.input_state.current_execution_count == 2
        assert T.parameter_states[pnl.SLOPE].current_execution_count == 2
        assert T.output_state.current_execution_count == 2

        T.execute()
        assert T.current_execution_count == 3
        # assert T.input_state.current_execution_count == 3
        assert T.parameter_states[pnl.SLOPE].current_execution_count == 3
        assert T.output_state.current_execution_count == 3
