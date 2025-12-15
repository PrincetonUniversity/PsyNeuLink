import numpy as np
import psyneulink as pnl
import pytest


class TestOutputPorts:

    @pytest.mark.mechanism
    @pytest.mark.parametrize("spec,expected1,expected2",
                             [((pnl.OWNER_VALUE, 2), [[3]], [[3]]),
                              ((pnl.OWNER_VALUE, 1), [[2]], [[2]]),
                              ((pnl.OWNER_VALUE, 0), [[1]], [[1]]),
                              ((pnl.OWNER_VALUE), [[[1], [2], [3]]], [[[1], [2], [3]]]),
                              ([(pnl.OWNER_VALUE, 0), (pnl.OWNER_VALUE, 1)], [[[1], [2]]], [[[1], [2]]]),
                              ((pnl.OWNER_EXECUTION_COUNT), [[0]], [[4]]),
                              ([(pnl.PREVIOUS_VALUE, 1, 0), pnl.RATE], [[0, 1]], [[2, 1]]), # From AdaptiveIntegrator
                              ], ids=lambda x: str(x) if len(x) != 1 else '')
    def test_output_port_variable_spec(self, spec, expected1, expected2, mech_mode):
        mech = pnl.ProcessingMechanism(default_variable=[[1.], [2.], [3.]],
                                       name='MyMech',
                                       function=pnl.AdaptiveIntegrator,
                                       output_ports=[pnl.OutputPort(variable=spec)])

        np.testing.assert_allclose(mech.output_values, expected1)

        EX = pytest.helpers.get_mech_execution(mech, mech_mode)

        EX([[1.], [2.], [3.]])
        EX([[1.], [2.], [3.]])
        EX([[1.], [2.], [3.]])
        res = EX([[1.], [2.], [3.]])

        np.testing.assert_allclose(res, expected2)

    @pytest.mark.composition
    @pytest.mark.mechanism
    @pytest.mark.parametrize('spec, expected1, expected2',
                             [((pnl.OWNER_VALUE, 0), [1], [1]),
                              ((pnl.OWNER_VALUE, 1), [2], [2]),
                              ((pnl.OWNER_VALUE, 2), [3], [3]),
                              pytest.param((pnl.OWNER_VALUE, 3), [3], [3], marks=[pytest.mark.xfail(raises=IndexError, match="list index out of range")]),
                              ((pnl.OWNER_EXECUTION_COUNT), [4], [8]),
                              (("num_executions", pnl.TimeScale.LIFE), [4], [8]),
                              (("num_executions", pnl.TimeScale.RUN), [4], [4]),
                              (("num_executions", pnl.TimeScale.TRIAL), [2], [2]),
                              (("num_executions", pnl.TimeScale.PASS), [1], [1]),
                              (("num_executions", pnl.TimeScale.TIME_STEP), [1], [1]),
                              ([pnl.SLOPE, pnl.SCALE], [1, 1], [1, 1]), # From Linear function
                             ], ids=lambda x: str(x) if len(x) != 1 else '')
    @pytest.mark.usefixtures("comp_mode_no_per_node")
    def test_output_port_variable_spec_composition(self, comp_mode, spec, expected1, expected2):
        # TimeScale.RUN is not supported in LLVMExec/PTXExec mode since the necessary
        # counter maintenance is skipped
        if (len(spec) == 2) and (spec[1] == pnl.TimeScale.RUN) and (comp_mode & pnl.ExecutionMode._Exec):
            pytest.skip("{} is not supported in {}".format(spec[1], comp_mode))

        var = [[1], [2], [3]]
        mech = pnl.ProcessingMechanism(default_variable=var, name='MyMech',output_ports=[pnl.OutputPort(variable=spec)])

        C = pnl.Composition(name='MyComp')
        C.add_node(node=mech)
        C.termination_processing[pnl.TimeScale.TRIAL] = pnl.AtPass(2)

        # outs is entire mechanism value, expected is output port value
        outs = C.run(inputs={mech: var}, num_trials=2, execution_mode=comp_mode)
        np.testing.assert_allclose(outs, [expected1])

        outs = C.run(inputs={mech: var}, num_trials=2, execution_mode=comp_mode)
        np.testing.assert_allclose(outs, [expected2])

    def test_no_path_afferents(self):
        A = pnl.OutputPort()
        with pytest.raises(pnl.PortError) as error:
            A.path_afferents
        assert 'OutputPorts do not have \'path_afferents\'; (access attempted for Deferred Init OutputPort).' \
               in str(error.value)
        with pytest.raises(pnl.PortError) as error:
            A.path_afferents = ['test']
        assert 'OutputPorts are not allowed to have \'path_afferents\' ' \
               '(assignment attempted for Deferred Init OutputPort).' in str(error.value)
