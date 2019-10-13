import numpy as np
import psyneulink as pnl
import pytest

import psyneulink.core.components.functions.combinationfunctions
import psyneulink.core.components.functions.transferfunctions


class TestOutputPorts:

    def test_output_port_variable_spec(self):
        # Test specification of OutputPort's variable
        udf = pnl.UserDefinedFunction(custom_function=lambda x: np.array([[1], [2], [3]]))
        mech = pnl.ProcessingMechanism(function=udf, name='MyMech',
                                       output_ports=[
                                           pnl.OutputPort(name='z', variable=(pnl.OWNER_VALUE, 2)),
                                           pnl.OutputPort(name='y', variable=(pnl.OWNER_VALUE, 1)),
                                           pnl.OutputPort(name='x', variable=(pnl.OWNER_VALUE, 0)),
                                           pnl.OutputPort(name='all', variable=(pnl.OWNER_VALUE)),
                                           pnl.OutputPort(name='execution count', variable=(pnl.OWNER_EXECUTION_COUNT))
                                       ])
        expected = [[3.],[2.],[1.],[[1.],[2.],[3.]], [0]]
        for i, e in zip(mech.output_values, expected):
            assert np.array_equal(i, e)
        mech.execute([0])
        expected = [[3.],[2.],[1.],[[1.],[2.],[3.]], [1]]
        for i, e in zip(mech.output_values, expected):
            assert np.array_equal(i, e)
        # OutputPort mech.output_ports['all'] has a different dimensionality than the other OutputPorts;
        #    as a consequence, when added as a terminal node, the Composition can't construct an IDENTITY_MATRIX
        #    from the mech's OutputPorts to the Composition's output_CIM.
        # FIX: Remove the following line and correct assertions below once above condition is resolved
        mech.remove_ports(ports=mech.output_ports['all'])
        C = pnl.Composition(name='MyComp')
        C.add_node(node=mech)
        outs = C.run(inputs={mech: np.array([[0]])})
        assert np.array_equal(outs, np.array([[3], [2], [1], [2]]))
        outs = C.run(inputs={mech: np.array([[0]])})
        assert np.array_equal(outs, np.array([[3], [2], [1], [3]]))
