import numpy as np
import psyneulink as pnl
import pytest

import psyneulink.core.components.functions.combinationfunctions
import psyneulink.core.components.functions.transferfunctions


class TestOutputStates:

    def test_output_state_variable_spec(self):
        udf = pnl.UserDefinedFunction(custom_function=lambda x: np.array([[1], [2], [3]]))
        mech = pnl.ProcessingMechanism(function=udf, name='MyMech',
                                       output_states=[
                                           pnl.OutputState(name='z', variable=(pnl.OWNER_VALUE, 2)),
                                           pnl.OutputState(name='y', variable=(pnl.OWNER_VALUE, 1)),
                                           pnl.OutputState(name='x', variable=(pnl.OWNER_VALUE, 0))
                                       ]
                                       )
        C = pnl.Composition(name='MyComp')
        C.add_node(node=mech)
        assert mech.output_values == [[3.],[2.],[1.]]
        mech.execute([0])
        assert mech.output_values == [[3.],[2.],[1.]]
        outs = C.run(inputs={mech: np.array([[0]])})
        assert np.array_equal(outs, np.array([[3], [2], [1]]))
