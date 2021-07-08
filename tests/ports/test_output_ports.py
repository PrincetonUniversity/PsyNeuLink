import numpy as np
import psyneulink as pnl
import psyneulink.core.llvm as pnlvm
import pytest


class TestOutputPorts:

    @pytest.mark.mechanism
    @pytest.mark.lca_mechanism
    def test_output_port_variable_spec(self, mech_mode):
        # Test specification of OutputPort's variable
        mech = pnl.ProcessingMechanism(default_variable=[[1.],[2.],[3.]],
                                       name='MyMech',
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
        EX = pytest.helpers.get_mech_execution(mech, mech_mode)

        EX([[1.],[2.],[3.]])
        EX([[1.],[2.],[3.]])
        EX([[1.],[2.],[3.]])
        res = EX([[1.],[2.],[3.]])
        expected = [[3.],[2.],[1.],[[1.],[2.],[3.]], [4]]
        for i, e in zip(res, expected):
            assert np.array_equal(i, e)

    @pytest.mark.mechanism
    @pytest.mark.lca_mechanism
    def test_output_port_variable_spec_composition(self, comp_mode):
        # Test specification of OutputPort's variable
        # OutputPort mech.output_ports['all'] has a different dimensionality than the other OutputPorts;
        #    as a consequence, when added as a terminal node, the Composition can't construct an IDENTITY_MATRIX
        #    from the mech's OutputPorts to the Composition's output_CIM.
        # FIX: Remove the following line and correct assertions below once above condition is resolved
        mech = pnl.ProcessingMechanism(default_variable=[[1.],[2.],[3.]],
                                       name='MyMech',
                                       output_ports=[
                                           pnl.OutputPort(name='z', variable=(pnl.OWNER_VALUE, 2)),
                                           pnl.OutputPort(name='y', variable=(pnl.OWNER_VALUE, 1)),
                                           pnl.OutputPort(name='x', variable=(pnl.OWNER_VALUE, 0)),
#                                           pnl.OutputPort(name='all', variable=(pnl.OWNER_VALUE)),
                                           pnl.OutputPort(name='execution count', variable=(pnl.OWNER_EXECUTION_COUNT))
                                       ])
        C = pnl.Composition(name='MyComp')
        C.add_node(node=mech)
        outs = C.run(inputs={mech: [[1.],[2.],[3.]]}, execution_mode=comp_mode)
        assert np.array_equal(outs, [[3], [2], [1], [1]])
        outs = C.run(inputs={mech: [[1.],[2.],[3.]]}, execution_mode=comp_mode)
        assert np.array_equal(outs, [[3], [2], [1], [2]])
