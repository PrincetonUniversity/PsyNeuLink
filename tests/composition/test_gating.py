import numpy as np
import pytest
import psyneulink as pnl


@pytest.mark.composition
@pytest.mark.benchmark(group="Gating")
@pytest.mark.parametrize("mode", ['Python',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.param('LLVMExec', marks=pytest.mark.llvm),
                                  pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                  pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                  pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])
                                  ])
def test_gating(benchmark, mode):

    Input_Layer = pnl.TransferMechanism(
        name='Input_Layer',
        default_variable=np.zeros((2,)),
        function=pnl.Logistic()
    )

    Output_Layer = pnl.TransferMechanism(
        name='Output_Layer',
        default_variable=[0, 0, 0],
        function=pnl.Linear(),
        output_ports={
            pnl.NAME: 'RESULTS USING UDF',
            pnl.FUNCTION: pnl.Linear(slope=pnl.GATING)
        }
    )

    Gating_Mechanism = pnl.GatingMechanism(
        size=[1],
        gating_signals=[Output_Layer.output_port]
    )

    p_pathway = [Input_Layer, Output_Layer]

    stim_list = {
        Input_Layer: [[-1, 30], [-1, 30], [-1, 30], [-1, 30]],
        Gating_Mechanism: [[0.0], [0.5], [1.0], [2.0]]
    }

    comp = pnl.Composition(name="comp")
    comp.add_linear_processing_pathway(p_pathway)
    comp.add_node(Gating_Mechanism)

    comp.run(num_trials=4, inputs=stim_list, bin_execute=mode)

    expected_results = [
        [np.array([0., 0., 0.])],
        [np.array([0.63447071, 0.63447071, 0.63447071])],
        [np.array([1.26894142, 1.26894142, 1.26894142])],
        [np.array([2.53788284, 2.53788284, 2.53788284])]
    ]

    np.testing.assert_allclose(comp.results, expected_results)
    benchmark(comp.run, num_trials=4, inputs=stim_list, bin_execute=mode)

# DEPRECATED FUNCTIONALITY 9/26/19
# @pytest.mark.composition
# @pytest.mark.benchmark(group="Gating")
# @pytest.mark.parametrize("mode", ['Python',
#                                   pytest.param('LLVM', marks=pytest.mark.llvm),
#                                   pytest.param('LLVMExec', marks=pytest.mark.llvm),
#                                   pytest.param('LLVMRun', marks=pytest.mark.llvm),
#                                   pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
#                                   pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])
#                                   ])
# def test_gating_using_ControlMechanism(benchmark, mode):
#
#     Input_Layer = pnl.TransferMechanism(
#         name='Input_Layer',
#         default_variable=np.zeros((2,)),
#         function=pnl.Logistic()
#     )
#
#     Output_Layer = pnl.TransferMechanism(
#         name='Output_Layer',
#         default_variable=[0, 0, 0],
#         function=pnl.Linear(),
#         output_ports={
#             pnl.NAME: 'RESULTS USING UDF',
#             pnl.FUNCTION: pnl.Linear(slope=pnl.GATING)
#         }
#     )
#
#     Gating_Mechanism = pnl.ControlMechanism(
#         size=[1],
#         control_signals=[Output_Layer.output_port]
#     )
#
#     p_pathway = [Input_Layer, Output_Layer]
#
#     stim_list = {
#         Input_Layer: [[-1, 30], [-1, 30], [-1, 30], [-1, 30]],
#         Gating_Mechanism: [[0.0], [0.5], [1.0], [2.0]]
#     }
#
#     comp = pnl.Composition(name="comp")
#     comp.add_linear_processing_pathway(p_pathway)
#     comp.add_node(Gating_Mechanism)
#
#     comp.run(num_trials=4, inputs=stim_list, bin_execute=mode)
#
#     expected_results = [
#         [np.array([0., 0., 0.])],
#         [np.array([0.63447071, 0.63447071, 0.63447071])],
#         [np.array([1.26894142, 1.26894142, 1.26894142])],
#         [np.array([2.53788284, 2.53788284, 2.53788284])]
#     ]
#
#     np.testing.assert_allclose(comp.results, expected_results)
#     benchmark(comp.run, num_trials=4, inputs=stim_list, bin_execute=mode)
