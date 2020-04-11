"""

bistable percepts

"""



import numpy as np
import psyneulink as pnl
import pytest
from itertools import product




@pytest.mark.model
@pytest.mark.benchmark(group="Simplified Necker Cube")
@pytest.mark.parametrize("mode", ['Python',
                                  pytest.param('LLVM', marks=[pytest.mark.llvm]),
                                  pytest.param('LLVMExec', marks=[pytest.mark.llvm]),
                                  pytest.param('LLVMRun', marks=[pytest.mark.llvm]),
                                  pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                  pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                  pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                  ])
def test_simplified_necker_cube(benchmark, mode):
    # this code only works for N_PERCEPTS == 2
    ALL_PERCEPTS = ['a', 'b']

    # variables
    n_nodes_per_percepts = 3
    excit_level = 1
    inhib_level = 1
    node_dict = {percept: None for percept in ALL_PERCEPTS}

    def get_node(percept, node_id):
        """helper func for creating a node"""
        tm_function = pnl.Linear(slope=1, intercept=0)
        tm_integrator_mode = True
        tm_integration_rate = .5
        node_ = pnl.TransferMechanism(
            name='{percept}-{node_id}'.format(percept=percept, node_id=node_id),
            function=tm_function,
            integrator_mode=tm_integrator_mode,
            integration_rate=tm_integration_rate,
            default_variable=np.zeros((1,)),
        )
        return node_

    # init all nodes, save them in list and dict form
    for percept in ALL_PERCEPTS:
        node_dict[percept] = [
            get_node(percept, i) for i in range(n_nodes_per_percepts)
        ]

    # init composition
    bp_comp = pnl.Composition()

    # MODIFIED 4/11/20 OLD:  PASSES IN PYTHON, BUT NEEDS RESULTS B BELOW
    # within-percept excitation
    for percept in ALL_PERCEPTS:
        for node_i, node_j in product(node_dict[percept], node_dict[percept]):
            if node_i is not node_j:
                bp_comp.add_linear_processing_pathway(
                    pathway=(node_i, [excit_level], node_j))

    # inter-percepts inhibition
    for node_i, node_j in zip(node_dict[ALL_PERCEPTS[0]],
                              node_dict[ALL_PERCEPTS[1]]):
        bp_comp.add_linear_processing_pathway(
            pathway=(node_i, [-inhib_level], node_j))
        bp_comp.add_linear_processing_pathway(
            pathway=(node_j, [-inhib_level], node_i))

    # turn off report
    reportOutputPref = False

    # make sure all nodes are both input and outputs
    for node in bp_comp.nodes:
        bp_comp.add_required_node_role(node, pnl.NodeRole.INPUT)
        bp_comp.add_required_node_role(node, pnl.NodeRole.OUTPUT)
        # turn off report
        node.reportOutputPref = reportOutputPref

    # # MODIFIED 4/11/20 NEW:  [PASSES ALL TESTS, BUT NEEDS RSEULTS A BELOW]
    # # within-percept excitation
    # for percept in ALL_PERCEPTS:
    #     for node_i, node_j in product(node_dict[percept], node_dict[percept]):
    #         if node_i is not node_j:
    #             bp_comp.add_linear_processing_pathway(
    #                 pathway=((node_i, [pnl.NodeRole.INPUT, pnl.NodeRole.OUTPUT]), [excit_level], (node_j, [pnl.NodeRole.INPUT,
    #                                                                                                pnl.NodeRole.OUTPUT])))
    #
    # # inter-percepts inhibition
    # for node_i, node_j in zip(node_dict[ALL_PERCEPTS[0]],
    #                           node_dict[ALL_PERCEPTS[1]]):
    #     bp_comp.add_linear_processing_pathway(
    #         pathway=((node_i, [pnl.NodeRole.INPUT, pnl.NodeRole.OUTPUT]), [-inhib_level], (node_j, [pnl.NodeRole.INPUT, pnl.NodeRole.OUTPUT])))
    #     bp_comp.add_linear_processing_pathway(
    #         pathway=((node_j, [pnl.NodeRole.INPUT, pnl.NodeRole.OUTPUT]), [-inhib_level], (node_i, [pnl.NodeRole.INPUT,
    #                                                                                         pnl.NodeRole.OUTPUT])))

    # turn off report
    reportOutputPref = False

    # MODIFIED 4/11/20 END:


    # bp_comp.show_graph()

    # init the inputs
    # # MODIFIED 4/4/20 NEW:
    # np.random.seed(12345)
    # # MODIFIED 4/4/20 END:
    n_time_steps = 10
    input_dict = {
        node_: np.random.normal(size=(n_time_steps,))
        for node_ in bp_comp.nodes
    }

    # run the model
    res = bp_comp.run(input_dict, num_trials=10, bin_execute=mode)
    np.testing.assert_allclose(res,
                               # [[3127.65559899], [3610.74194658],  # A) original:  no seed and
                               #  [6468.6978669], [-4615.15074428],  #            no_analyze_graph in Composition:3776
                               #  [-7369.73302025], [-11190.45001744]])
                               [[-11190.45001744], [3127.65559899],  # B) no seed, but with with_analyze_graph in
                                [3610.74194658], [6468.6978669],     #  Composition:3776; passes for Python but not LLVM
                                [-4615.15074428], [-7369.73302025]])
                               # [[4380.19172585], [5056.09548856],   # C) seed but no _analyze_graph in Composition:3776
                               #  [9058.54210893], [-6465.3497555],   # passes for Python abd LLVM
                               #  [-10322.33734752], [-15673.99046508]])
                               # [[-15673.99046508], [4380.19172585],  # D) seed + _analyze_graph in Composition:3776
                               #  [5056.09548856], [9058.54210893],    # passes for Python but not LLVM
                               #  [-6465.3497555], [-10322.33734752]])


    benchmark(bp_comp.run, input_dict, num_trials=10, bin_execute=mode)

@pytest.mark.model
@pytest.mark.benchmark(group="Necker Cube")
@pytest.mark.parametrize("mode", ['Python',
                                  pytest.param('LLVM', marks=[pytest.mark.llvm]),
                                  pytest.param('LLVMExec', marks=[pytest.mark.llvm]),
                                  pytest.param('LLVMRun', marks=[pytest.mark.llvm]),
                                  pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                  pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                  pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                  ])
def test_necker_cube(benchmark, mode):

    Build_N_Matrix = np.zeros((16,5))
    Build_N_Matrix[0,:] = [0, 1, 3, 4, 8]
    Build_N_Matrix[1,:] = [1, 0, 2, 5, 9]
    Build_N_Matrix[2,:] = [2, 1, 3, 6, 10]
    Build_N_Matrix[3,:] = [3, 0, 2, 7, 11]
    Build_N_Matrix[4,:] = [4, 5, 7, 0, 12]
    Build_N_Matrix[5,:] = [5, 4, 6, 1, 13]
    Build_N_Matrix[6,:] = [6, 5, 7, 2, 14]
    Build_N_Matrix[7,:] = [7, 4, 6, 3, 15]
    Build_N_Matrix[8,:] = [8, 9, 11, 12, 0]
    Build_N_Matrix[9,:] = [9, 8, 10, 13, 1]
    Build_N_Matrix[10,:] = [10, 9, 11, 14, 2]
    Build_N_Matrix[11,:] = [11, 8, 10, 15, 3]
    Build_N_Matrix[12,:] = [12, 13, 15, 8, 4]
    Build_N_Matrix[13,:] = [13, 12, 14, 9, 5]
    Build_N_Matrix[14,:] = [14, 13, 15, 10, 6]
    Build_N_Matrix[15,:] = [15, 12, 14, 11, 7]

    Build_N_Matrix = Build_N_Matrix.astype(int)

    Necker_Matrix = np.zeros((16,16))
    Necker_Matrix = Necker_Matrix.astype(int)


    excite = 1
    inhibit = -2

    for x in range(0,16):
        Necker_Matrix[Build_N_Matrix[x,0], Build_N_Matrix[x,1]] = excite
        Necker_Matrix[Build_N_Matrix[x,0], Build_N_Matrix[x,2]] = excite
        Necker_Matrix[Build_N_Matrix[x,0], Build_N_Matrix[x,3]] = excite
        Necker_Matrix[Build_N_Matrix[x,0], Build_N_Matrix[x,4]] = inhibit

    comp2 = pnl.Composition()

    node3 = pnl.TransferMechanism(
        name='node3',
        function=pnl.Linear(slope = 1, intercept = 0),
        integrator_mode = True,
        integration_rate = .5,
        default_variable=np.zeros((1,16)),
    )


    #integrator function ((1-rate)*previous_value + rate*current_input) * mechanism_function

    node4 = pnl.TransferMechanism(
        name='node4',
        function=pnl.Linear(slope = 1, intercept = 0),
        integrator_mode = True,
        integration_rate = .5,
        default_variable=np.zeros((1,16)),
    )


    connect_3_4 = Necker_Matrix
    connect_4_3 = Necker_Matrix

    weights_3_4 = pnl.MappingProjection(
        name='connect_3_4',
        matrix=connect_3_4,
    )

    weights_4_3 = pnl.MappingProjection(
        name='connect_4_3',
        matrix=connect_3_4,
    )

    comp2.add_linear_processing_pathway(pathway = (node3, connect_3_4, node4, connect_4_3, node3))
    # MODIFIED 4/4/20 NEW:
    np.random.seed(12345)
    # MODIFIED 4/4/20 END
    input_dict = {node3: np.random.random((1,16)),
                  node4: np.random.random((1,16))
                 }

    result = comp2.run(input_dict, num_trials=10, bin_execute=mode)
    assert np.allclose(result,
            # [[ 2636.29181172,  -662.53579899,  2637.35386946,  -620.15550833,
            #    -595.55319772,  2616.74310649,  -442.74286574,  2588.4778162 ,
            #     725.33941441, -2645.25148476,   570.96811513, -2616.80319979,
            #   -2596.82097419,   547.30466563, -2597.99430789,   501.50648114],
            #  [ -733.2213593 ,  2638.81033464,  -578.76439993,  2610.55912376,
            #    2590.69244696,  -555.19824432,  2591.63200098,  -509.58072358,
            #   -2618.88711219,   682.65814776, -2620.18294962,   640.09719335,
            #     615.39758884, -2599.45663784,   462.67291695, -2570.99427346]])
            [[ 753.49687364,  380.1835271 ,  526.71129889,  253.30439596,
                    335.33291717,  796.34470018,  504.94661527,  664.84397208,
                   -228.29889962, -699.72265243, -395.45414321, -568.29933106,
                   -837.38658858, -477.94765341, -612.70717468, -348.86306586],
             [ 217.19651713,  708.59009834,  384.29837558,  577.37836065,
                    846.10421744,  466.68904807,  621.40583149,  337.60282732,
                   -750.45164969, -357.23030678, -523.68504698, -230.35280883,
                   -312.48416776, -793.679849  , -482.15125099, -661.96753723]])

    benchmark(comp2.run, input_dict, num_trials=10, bin_execute=mode)
