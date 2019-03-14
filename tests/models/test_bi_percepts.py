"""

bistable percepts

"""



import numpy as np
import psyneulink as pnl
import pytest
from itertools import product

# constants

# this code only works for N_PERCEPTS == 2

ALL_PERCEPTS = ['a', 'b']

N_PERCEPTS = len(ALL_PERCEPTS)

assert N_PERCEPTS == 2



# variables

n_nodes_per_percepts = 3

node_dict = {percept: None for percept in ALL_PERCEPTS}


@pytest.mark.model
@pytest.mark.benchmark(group="BiPercept")
@pytest.mark.parametrize("mode", ['Python',
    pytest.param('LLVM', marks=[pytest.mark.llvm]),
    pytest.param('LLVMExec', marks=[pytest.mark.llvm]),
    pytest.param('LLVMRun', marks=[pytest.mark.llvm]),
    pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda]),
    pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
    pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda]),
])
def test_bi_precepts(benchmark, mode):
    excit_level = 1
    inhib_level = 1

    def get_node(percept, node_id):

        # helper func for creating a node

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

    # make sure all nodes are both input and outputs
    reportOutputPref = False

    for node in bp_comp.nodes:
        bp_comp.add_required_node_role(node, pnl.NodeRole.INPUT)
        bp_comp.add_required_node_role(node, pnl.NodeRole.OUTPUT)

        # turn off report
        node.reportOutputPref = reportOutputPref

#    bp_comp.show_graph()

    # init the inputs
    n_time_steps = 10
    input_dict = {
        node_: np.random.normal(size=(n_time_steps,))
        for node_ in bp_comp.nodes
    }

    # run the model
    res = bp_comp.run(input_dict, num_trials=10, bin_execute=mode)
    np.testing.assert_allclose(res, [[3127.65559899], [3610.74194658],
                                     [6468.6978669], [-4615.15074428],
                                     [-7369.73302025], [-11190.45001744]])

    benchmark(bp_comp.run, input_dict, num_trials=10, bin_execute=mode)
