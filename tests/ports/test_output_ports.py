import numpy as np
import psyneulink as pnl
import psyneulink.core.llvm as pnlvm
import pytest


class TestOutputPorts:

    @pytest.mark.mechanism
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
    @pytest.mark.parametrize('spec, expected1, expected2',
                             [((pnl.OWNER_VALUE, 0), [1], [1]),
                              ((pnl.OWNER_VALUE, 1), [2], [2]),
                              ((pnl.OWNER_VALUE, 2), [3], [3]),
                              pytest.param((pnl.OWNER_VALUE, 3), [3], [3], marks=[pytest.mark.xfail()]),
                              ((pnl.OWNER_EXECUTION_COUNT), [4], [8]),
                              (("num_executions", pnl.TimeScale.LIFE), [4], [8]),
                              (("num_executions", pnl.TimeScale.RUN), [4], [4]),
                              (("num_executions", pnl.TimeScale.TRIAL), [2], [2]),
                              (("num_executions", pnl.TimeScale.PASS), [1], [1]),
                              (("num_executions", pnl.TimeScale.TIME_STEP), [1], [1]),
                             ], ids=lambda x: str(x) if len(x) != 1 else '')
    @pytest.mark.usefixtures("comp_mode_no_llvm")
    def tests_output_port_variable_spec_composition(self, comp_mode, spec, expected1, expected2):
        if (len(spec) == 2) and (spec[1] == pnl.TimeScale.RUN) and \
           ((comp_mode & pnl.ExecutionMode._Exec) == pnl.ExecutionMode._Exec):
            pytest.skip("{} is not supported in {}".format(spec[1], comp_mode))

        # Test specification of OutputPort's variable
        # OutputPort mech.output_ports['all'] has a different dimensionality than the other OutputPorts;
        #    as a consequence, when added as a terminal node, the Composition can't construct an IDENTITY_MATRIX
        #    from the mech's OutputPorts to the Composition's output_CIM.
        # FIX: Remove the following line and correct assertions below once above condition is resolved
        var = [[1], [2], [3]]
        mech = pnl.ProcessingMechanism(default_variable=var, name='MyMech',
                                       output_ports=[pnl.OutputPort(variable=spec)])
        C = pnl.Composition(name='MyComp')
        C.add_node(node=mech)
        C.termination_processing[pnl.TimeScale.TRIAL] = pnl.AtPass(2)
        outs = C.run(inputs={mech: var}, num_trials=2, execution_mode=comp_mode)
        assert np.allclose(outs, expected1)
        outs = C.run(inputs={mech: var}, num_trials=2, execution_mode=comp_mode)
        assert np.allclose(outs, expected2)

    def test_no_path_afferents(self):
        A = pnl.OutputPort()
        with pytest.raises(pnl.PortError) as error:
            A.path_afferents
        assert '"OutputPorts do not have \'path_afferents\'; (access attempted for Deferred Init OutputPort)."' \
               in str(error.value)
        with pytest.raises(pnl.PortError) as error:
            A.path_afferents = ['test']
        assert '"OutputPorts are not allowed to have \'path_afferents\' ' \
               '(assignment attempted for Deferred Init OutputPort)."' in str(error.value)
