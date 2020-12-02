##################################################################################################
#
# network_utils.py contains utility functions for the agent model -- specifically, a neural network
# instantiated with PsyNeuLink.
#
# Author: Yotam Sagiv
#
##################################################################################################

import numpy as np
import psyneulink as pnl
import networkx as nx
import sys
import warnings
warnings.filterwarnings("error", category=UserWarning)
warnings.filterwarnings("ignore", "Pathway specified in 'pathway' arg for add_backpropagation_learning_pathway method", category=UserWarning)
warnings.filterwarnings("ignore", "Unable to copy weight matrix for", category=UserWarning)
###################### Convenience functions for testing script #################################

# read in bipartite graph, return graph object, number of possible tasks, number of
# input dimensions and number of output dimensions.
# file format Ni No (input dimension number, output dimension number)
def read_bipartite_adjlist(filename):
    g = nx.Graph()
    with open(filename) as f:
        for line in f:
            inode, onode = line.split()
            g.add_node(inode, bipartite=1)
            g.add_node(onode, bipartite=0)
            g.add_edge(inode, onode, task_id=(inode, onode))

    onodes = { n for n, d in g.nodes(data=True) if d['bipartite'] == 0 }
    inodes = set(g) - onodes

    return g, g.number_of_edges(), len(inodes), len(onodes)

############################## Network utility functions #######################################

# Given LCA parameters and environment spec, return the LCA connectivity matrix
def get_LCA_matrix(num_output_dims, num_features, self_excitation, competition):
    output_layer_size = num_output_dims * num_features

    # Set up unit block matrix
    unit_dim_mat = np.zeros((num_features, num_features)) - competition
    np.fill_diagonal(unit_dim_mat, self_excitation)

    # Build overall matrix in terms of unit block matrices on the diagonal, zeros elsewhere
    for output_dim in range(num_output_dims):
        if output_dim == 0:
            lca_mat = unit_dim_mat
        else:
            lca_mat = np.block([
                    [lca_mat, np.zeros((lca_mat.shape[0], unit_dim_mat.shape[1]))],
                    [np.zeros((unit_dim_mat.shape[0], lca_mat.shape[1])), unit_dim_mat],
            ])

    return lca_mat

# Given a bipartite graph object, parse out a list of all the tasks defined on
# that graph.
def get_all_tasks(env_bipartite_graph):
    graph_edges = env_bipartite_graph.edges()
    all_tasks = []

    # Check if the nodes are the right way around (input, output)
    for edge in graph_edges:
        if edge[1][-1] == 'i' or edge[0][-1] == 'o': # hopefully equivalent
            inode = edge[1]
            onode = edge[0]
        else:
            inode = edge[0]
            onode = edge[1]

        # Strip suffix and convert to ints
        all_tasks.append((int(inode[:-1]), int(onode[:-1])))

    return all_tasks

# Train a multitasking network using PsyNeuLink and return the trained network, with optional attached
# LCAMechanism at the end for performance evaluation
# Params:
#     bipartite_graph: bipartite graph representing the task environment (NetworkX object)
#     num_features: number of particular features per dimension (e.g. number of colours)
#     num_hidden: number of hidden units in the network
#     epochs: number of training iterations
#     learning_rate: learning rate for SGD or (however pnl train their networks)
#     attach_LCA: if True, will attach an LCAMechanism to evaluate network performance
#     rest: LCA parameters
def get_trained_network(bipartite_graph, num_features=3, num_hidden=200, epochs=10, learning_rate=20, attach_LCA=True, competition=0.2, self_excitation=0.2, leak=0.4, threshold=1e-4):
    # Get all tasks from bipartite graph (edges) and strip 'i/o' suffix
    all_tasks = get_all_tasks(bipartite_graph)

    # Analyze bipartite graph for network properties
    onodes = [ n for n, d in bipartite_graph.nodes(data=True) if d['bipartite'] == 0 ]
    inodes = [ n for n, d in bipartite_graph.nodes(data=True) if d['bipartite'] == 1 ]
    input_dims = len(inodes)
    output_dims = len(onodes)
    num_tasks = len(all_tasks)

    # Start building network as PsyNeuLink object
    # Layer parameters
    nh = num_hidden
    D_i = num_features * input_dims
    D_c = num_tasks
    D_h = nh
    D_o = num_features * output_dims

    # Weight matrices (defaults provided by Dillon)
    wih = np.random.rand(D_i, D_h) * 0.02 - 0.01
    wch = np.random.rand(D_c, D_h) * 0.02 - 0.01
    wco = np.random.rand(D_c, D_o) * 0.02 - 0.01
    who = np.random.rand(D_h, D_o) * 0.02 - 0.01

    # Training params (defaults provided by Dillon)
    patience = 10
    min_delt = 0.00001
    lr = learning_rate

    # Instantiate layers and projections
    il = pnl.TransferMechanism(size=D_i, name='input')
    cl = pnl.TransferMechanism(size=D_c, name='control')

    hl = pnl.TransferMechanism(size=D_h, name='hidden',
                                               function=pnl.Logistic(bias=-2))

    ol = pnl.TransferMechanism(size=D_o, name='output',
                                               function=pnl.Logistic(bias=-2))

    pih = pnl.MappingProjection(matrix=wih)
    pch = pnl.MappingProjection(matrix=wch)
    pco = pnl.MappingProjection(matrix=wco)
    pho = pnl.MappingProjection(matrix=who)

    # Create training data for network
    # We train across all possible inputs, one task at a time
    input_examples, output_examples, control_examples = generate_training_data(all_tasks, num_features, input_dims, output_dims)

    # Training parameter set
    input_set = {
            'inputs': {
                    il: input_examples.tolist(),
                    cl: control_examples.tolist()
            },
            'targets': {
                    ol: output_examples.tolist()
            },
            'epochs': epochs
    }

    mnet = pnl.AutodiffComposition(learning_rate=learning_rate,
                                   name='mnet')

    mnet.output_CIM.parameters.value._set_history_max_length(100000)
    mnet.add_node(il)
    mnet.add_node(cl)
    mnet.add_node(hl)
    mnet.add_node(ol)
    mnet.add_projection(projection=pih, sender=il, receiver=hl)
    mnet.add_projection(projection=pch, sender=cl, receiver=hl)
    mnet.add_projection(projection=pco, sender=cl, receiver=ol)
    mnet.add_projection(projection=pho, sender=hl, receiver=ol)

    # Train network
    mnet.learn(inputs=input_set,
               minibatch_size=1,
               bin_execute=True,
               patience=patience,
               min_delta=min_delt)

    for projection in mnet.projections:
        weights = projection.parameters.matrix.get(mnet)
        projection.parameters.matrix.set(weights, None)


    # Apply LCA transform (values from Sebastian's code -- supposedly taken from the original LCA paper from Marius & Jay)
    if attach_LCA:
        lca = pnl.LCAMechanism(size=D_o,
                               leak=leak,
                               competition=competition,
                               self_excitation=self_excitation,
                               time_step_size=0.01,
                               threshold=threshold,
                               threshold_criterion=pnl.CONVERGENCE,
                               reset_stateful_function_when=pnl.AtTrialStart(),
                               name='lca')

        # Wrapper composition used to pass values between mnet (AutodiffComposition) and lca (LCAMechanism)
        wrapper_composition = pnl.Composition()

        # Add mnet and lca to outer_composition
        wrapper_composition.add_linear_processing_pathway([mnet, lca])

        return wrapper_composition

    return mnet

# Train a multitasking network using PsyNeuLink and return the trained network, with optional attached
# RecurrentTransferMechanism at the end for performance evaluation
# Params:
#     bipartite_graph: bipartite graph representing the task environment (NetworkX object)
#     num_features: number of particular features per dimension (e.g. number of colours)
#     num_hidden: number of hidden units in the network
#     epochs: number of training iterations
#     learning_rate: learning rate for SGD or (however pnl train their networks)
#     attach_LCA: if True, will attach an LCAMechanism to evaluate network performance
#     rest: LCA parameters
def get_trained_network_multLCA(bipartite_graph, num_features=3, num_hidden=200, epochs=10, learning_rate=20,
                                                                attach_LCA=True, competition=0.2, self_excitation=0.2, leak=0.4, threshold=1e-4,
                                                                exec_limit=10000):
    # Get all tasks from bipartite graph (edges) and strip 'i/o' suffix
    all_tasks = get_all_tasks(bipartite_graph)

    # Analyze bipartite graph for network properties
    onodes = [ n for n, d in bipartite_graph.nodes(data=True) if d['bipartite'] == 0 ]
    inodes = [ n for n, d in bipartite_graph.nodes(data=True) if d['bipartite'] == 1 ]
    input_dims = len(inodes)
    output_dims = len(onodes)
    num_tasks = len(all_tasks)

    # Start building network as PsyNeuLink object
    # Layer parameters
    nh = num_hidden
    D_i = num_features * input_dims
    D_c = num_tasks
    D_h = nh
    D_o = num_features * output_dims

    # Weight matrices (defaults provided by Dillon)
    wih = np.random.rand(D_i, D_h) * 0.02 - 0.01
    wch = np.random.rand(D_c, D_h) * 0.02 - 0.01
    wco = np.random.rand(D_c, D_o) * 0.02 - 0.01
    who = np.random.rand(D_h, D_o) * 0.02 - 0.01

    # Training params (defaults provided by Dillon)
    patience = 10
    min_delt = 0.00001
    lr = learning_rate

    # Instantiate layers and projections
    il = pnl.TransferMechanism(size=D_i, name='input')
    cl = pnl.TransferMechanism(size=D_c, name='control')

    hl = pnl.TransferMechanism(size=D_h,
                               name='hidden',
                               function=pnl.Logistic(bias=-2))

    ol = pnl.TransferMechanism(size=D_o,
                               name='output',
                               function=pnl.Logistic(bias=-2))

    pih = pnl.MappingProjection(matrix=wih)
    pch = pnl.MappingProjection(matrix=wch)
    pco = pnl.MappingProjection(matrix=wco)
    pho = pnl.MappingProjection(matrix=who)

    # Create training data for network
    # We train across all possible inputs, one task at a time
    input_examples, output_examples, control_examples = generate_training_data(all_tasks, num_features, input_dims, output_dims)

    # Training parameter set
    input_set = {
            'inputs': {
                    il: input_examples.tolist(),
                    cl: control_examples.tolist()
            },
            'targets': {
                    ol: output_examples.tolist()
            },
            'epochs': epochs
    }

    # Build network
    mnet = pnl.AutodiffComposition(learning_rate=learning_rate,
                                   name='mnet')

    mnet.output_CIM.parameters.value._set_history_max_length(1000)
    mnet.add_node(il)
    mnet.add_node(cl)
    mnet.add_node(hl)
    mnet.add_node(ol)
    mnet.add_projection(projection=pih, sender=il, receiver=hl)
    mnet.add_projection(projection=pch, sender=cl, receiver=hl)
    mnet.add_projection(projection=pco, sender=cl, receiver=ol)
    mnet.add_projection(projection=pho, sender=hl, receiver=ol)

    # Train network
    mnet.learn(inputs=input_set,
               minibatch_size=1,
               bin_execute=True,
               patience=patience,
               min_delta=min_delt)

    for projection in mnet.projections:
        try:
            weights = projection.parameters.matrix.get(mnet)
            projection.parameters.matrix.set(weights, None)
        except AttributeError as e:
            warnings.warn(f"Unable to copy weight matrix for {projection}")

    # Apply LCA transform (values from Sebastian's code -- supposedly taken from the original LCA paper from Marius & Jay)
    if attach_LCA:
        lci = pnl.LeakyCompetingIntegrator(rate=leak,
                                           time_step_size=0.01)

        lca_matrix = get_LCA_matrix(output_dims, num_features, self_excitation, competition)

        lca = pnl.RecurrentTransferMechanism(size=D_o,
                                             matrix=lca_matrix,
                                             integrator_mode=True,
                                             integrator_function=lci,
                                             name='lca',
                                             termination_threshold=threshold,
                                             reset_stateful_function_when=pnl.AtTrialStart())

        # Wrapper composition used to pass values between mnet (AutodiffComposition) and lca (LCAMechanism)
        wrapper_composition = pnl.Composition()

        # Add mnet and lca to outer_composition
        wrapper_composition.add_linear_processing_pathway([mnet, lca])

        # Set execution limit
        lca.parameters.max_executions_before_finished.set(exec_limit, wrapper_composition)

        # # Logging/Debugging
        # lca.set_log_conditions('value', pnl.LogCondition.EXECUTION)

        return wrapper_composition

    return mnet


# Generate data for the network to train on. This means single-task training on all available
# tasks within the environment, under a uniform task distribution. As data we generate all possible
# mappings within each task. To specify a mapping, we use the rule that input feature nodes map to
# equal ordinal output feature nodes (i.e. 1st feature input maps to 1st feature output).
# Params:
#     all_tasks: list containing all tasks in the environment
#     num_features: number of particular features per dimension (e.g. number of colours)
#     num_input_dims: number of input dimensions in the environment
#     num_output_dims: number of output dimensions in the environment
#     samples_per_feature: how many stimuli will be sampled per feature to be trained within a task
#               (even though the input-output feature mapping is fixed for a given task, the values
#                of all the other input dimensions are not, so we can sample many stimuli for a given
#                feature association)
def generate_training_data(all_tasks, num_features, num_input_dims, num_output_dims, samples_per_feature=100):
    # Extract relevant parameters
    num_tasks = len(all_tasks)
    input_layer_size = num_features * num_input_dims
    output_layer_size = num_features * num_output_dims
    control_layer_size = num_tasks
    num_examples = num_features * num_tasks * samples_per_feature

    # Instantiate example matrices
    input_examples = np.zeros((num_examples, input_layer_size))
    output_examples = np.zeros((num_examples, output_layer_size))
    control_examples = np.zeros((num_examples, control_layer_size))

    # Create examples, task by task
    row_count = 0
    for task in all_tasks:
        # Load parameters
        task_input_dim, task_output_dim = task
        control_idx = task_id_to_control_idx(task, all_tasks)

        # Generate feature maps (we arbitrarily pick the redundant mapping within each dimension)
        # and also randomly sample input values for the other dimensions of the stimulus
        for _ in range(samples_per_feature):
            # Set feature of relevant task
            for i in range(num_features):
                input_examples[row_count, task_input_dim * num_features + i] = 1
                output_examples[row_count, task_output_dim * num_features + i] = 1
                control_examples[row_count, control_idx] = 1

                # Set all other stimulus dimensions randomly
                for input_dim in range(num_input_dims):
                    if input_dim == task_input_dim:
                        continue

                    input_examples[row_count, input_dim * num_features + np.random.choice(num_features)] = 1

                row_count += 1

    return input_examples, output_examples, control_examples

# Generate data for the network to test on. test_tasks is a performance set (i.e. a multitasking set of tasks to execute).
# As data we generate random features for all input dimensions. To specify a mapping, we use the rule that input
# feature nodes map to equal ordinal output feature nodes (i.e. 1st feature input maps to 1st feature output).
# Params:
#     test_tasks: list containing set of tasks to multitask
#     all_tasks: list containing all tasks in the environment
#     num_features: number of particular features per dimension (e.g. number of colours)
#     num_input_dims: number of input dimensions in the environment
#     num_output_dims: number of output dimensions in the environment
#     num_test_points: number of test points to generate
def generate_testing_data(test_tasks, all_tasks, num_features, num_input_dims, num_output_dims, num_test_points):
    # Extract relevant parameters
    num_tasks = len(all_tasks)
    input_layer_size = num_features * num_input_dims
    output_layer_size = num_features * num_output_dims
    control_layer_size = num_tasks

    # Instantiate example matrices
    input_examples = np.zeros((num_test_points, input_layer_size))
    output_examples = np.zeros((num_test_points, output_layer_size))
    control_examples = np.zeros((num_test_points, control_layer_size))

    # Create examples
    for i in range(num_test_points):
        for input_dim in range(num_input_dims):
            # Input
            feature = np.random.choice(num_features)
            input_idx = input_dim * num_features + feature
            input_examples[i, input_idx] = 1

            # Output & Control
            for output_dim in range(num_output_dims):
                # If there is not a task with this input/output pair, move on
                if (input_dim, output_dim) not in test_tasks:
                    continue

                # Output
                output_idx = output_dim * num_features + feature
                output_examples[i, output_idx] = 1

                # Control
                control_idx = task_id_to_control_idx((input_dim, output_dim), all_tasks)
                control_examples[i, control_idx] = 1

    return input_examples, output_examples, control_examples

# We define the control layer activation as just the index of task within the global all_tasks list.
def task_id_to_control_idx(task, all_tasks):
    return all_tasks.index(task)

# Use the LCA to evaluate network performance on a given set of tasks (performance set)
# Params:
#     mnet_lca: Multitasking network AutodiffComposition with attached LCAMechanism (PNL)
#     test_tasks: Set of tasks to be simultaneously executed. List of tuples indicating (input dim, output dim)
#     all_tasks: Global list of all possible tasks within the environment. Formatted as test_tasks.
#     num_features, num_output_dims, num_input_dims: ints encoding the relevant size
#     num_test_points: number of self-generated test points to test performance on
#     threshold: how many points correct to consider task correctly completed
def evaluate_net_perf_lca(mnet_lca, test_tasks, all_tasks, num_features, num_input_dims, num_output_dims, num_test_points, threshold=0.8):
    # Extract relevant parameters
    input_layer_size = num_features * num_input_dims
    output_layer_size = num_features * num_output_dims
    control_layer_size = len(all_tasks)
    num_tasks = len(test_tasks)

    # Instantiate test matrices
    input_test_pts, output_true_pts, control_test_pts = generate_testing_data(test_tasks,
                                                                              all_tasks,
                                                                              num_features,
                                                                              num_input_dims,
                                                                              num_output_dims,
                                                                              num_test_points=num_test_points)

    # Run the outer composition, one point at a time (for debugging purposes)
    for i in range(num_test_points):
        # Construct input dict
        input_set = {
                'inputs' : {
                        mnet_lca.nodes['mnet'].nodes['input'] : input_test_pts[i, :].tolist(),
                        mnet_lca.nodes['mnet'].nodes['control'] : control_test_pts[i, :].tolist()
                }
        }

        try:
            mnet_lca.run( { mnet_lca.nodes['mnet'] : input_set } )
        except Warning:
            print('input: ', input_test_pts[i, :])
            print('control: ', control_test_pts[i, :])
            print('true: ', output_true_pts[i, :])
            mnet = mnet_lca.nodes['mnet']
            lca = mnet_lca.nodes['lca']
            print('net out: ', np.array([mnet.output_CIM.parameters.value.get(mnet_lca)]).reshape(1, output_layer_size))
            print('num executions: ', mnet_lca.nodes['lca'].num_executions_before_finished)


            # mnet_lca.nodes['lca'].log.print_entries()
            sys.exit(0)

    # Retrieve LCA results
    lca_out = np.array(mnet_lca.parameters.results.get(mnet_lca)[-num_test_points:]).reshape(num_test_points, output_layer_size)

    # Retrieve mnet results
    mnet = mnet_lca.nodes['mnet']

    # Brutal line of code provided by Dillon
    # Intuition: Part in list() expression is stored history. Part afterwards is the current value.
    # This distinction is important because PsyNeuLink objects update history only on next trial call.
    if num_test_points > 1:
        mnet_out = np.array(list(mnet.output_CIM.parameters.value.history[mnet_lca.default_execution_id])[
                           -num_test_points +1:] +[mnet.output_CIM.parameters.value.get(mnet_lca)]).reshape(num_test_points, output_layer_size)
    else:
        mnet_out = np.array([mnet.output_CIM.parameters.value.get(mnet_lca)]).reshape(num_test_points, output_layer_size)

    # Compare correctness, get a bunch of useful stats
    stats_dict = {
        'tasks' : test_tasks,
        'num_tasks_correct' : 0,
        'inputs' : input_test_pts,
        'outputs' : lca_out,
        'true_outputs' : output_true_pts,
        'mnet_out' : mnet_out,
        'num_correct_per_task' : {},
        'wrong_idxs' : set()
    }

    for task in test_tasks:
        input_dim, output_dim = task

        num_test_pts_correct = 0
        for i in range(num_test_points):
            test_pt = input_test_pts[i, :]
            test_output = lca_out[i, :]

            # get input feature
            input_feature_idx = np.argmax(test_pt[input_dim * num_features : (input_dim + 1) * num_features])

            # get output feature
            output_feature_idx = np.argmax(test_output[output_dim * num_features : (output_dim + 1) * num_features])

            if input_feature_idx == output_feature_idx:
                num_test_pts_correct += 1
            else:
                stats_dict['wrong_idxs'].add(i)

        if num_test_pts_correct / num_test_points >= threshold:
            stats_dict['num_tasks_correct'] += 1

        stats_dict['num_correct_per_task'][task] = num_test_pts_correct

    return stats_dict

# Use Mean Squared Error (MSE) to evaluate network performance without an LCA
# Params:
#     mnet_lca: Multitasking network AutodiffComposition
#     test_tasks: Set of tasks to be simultaneously executed. List of tuples indicating (input dim, output dim)
#     all_tasks: Global list of all possible tasks within the environment. Formatted as test_tasks.
#     num_features, num_output_dims, num_input_dims: ints encoding the relevant size
#     num_test_points: number of self-generated test points to test performance on
#     threshold: how many points correct to consider task correctly completed
def evaluate_net_perf_mse(mnet, test_tasks, all_tasks, num_features, num_input_dims, num_output_dims, num_test_points, threshold=0.8):
    # Extract relevant parameters
    input_layer_size = num_features * num_input_dims
    output_layer_size = num_features * num_output_dims
    control_layer_size = len(all_tasks)
    num_tasks = len(test_tasks)

    # Instantiate test matrices
    input_test_pts, output_true_pts, control_test_pts = generate_testing_data(test_tasks,
                                                                              all_tasks,
                                                                              num_features,
                                                                              num_input_dims,
                                                                              num_output_dims,
                                                                              num_test_points=num_test_points)

    # Run the composition
    input_set = {
            'inputs' : {
                    mnet.nodes['input'] : input_test_pts.tolist(),
                    mnet.nodes['control'] : control_test_pts.tolist()
            }
    }

    mnet.run(input_set)

    # Retrieve results
    output_test_pts = np.array(mnet.parameters.results.get(mnet)[-num_test_points:]).reshape(num_test_points, output_layer_size)

    # Analyze accuracy
    MSE = 0
    plotted_examples = 0
    for i in range(num_test_points):
        test_out = output_test_pts[i, :]
        true_out = output_true_pts[i, :]

        MSE += np.sum(np.square(test_out - true_out))

        # Plot some examples arbitrarily at random
        if np.random.uniform() < 0.05 and plotted_examples < 3 and len(test_tasks) > 1:
            print('all tasks: ', all_tasks)
            print('test tasks: ', test_tasks)
            print('input: ', input_test_pts[i, :])
            print('control: ', control_test_pts[i, :])
            print('true output: ', true_out)
            print('test output: ', test_out)
            print()
            plotted_examples += 1

    MSE /= (num_test_points * output_layer_size)

    return MSE

############################## TESTING SCRIPT ##############################

# Trivial testing script
if __name__ == '__main__':
    np.set_printoptions(precision=7, threshold=sys.maxsize, suppress=True, linewidth=np.nan)
    verbose = False
    np.random.seed(12345)

    # Train and evaluate an mnet-LCA combo on single-tasking and multitasking

    # Params
    num_test_points = 100
    num_features = 3
    g, num_tasks, num_input_dims, num_output_dims = read_bipartite_adjlist('./data/bipartite_graphs/7-tasks.txt')
    all_tasks = get_all_tasks(g)

    # Get trained network (@ Jon: This won't work)
    mnet_lca = get_trained_network_multLCA(g, learning_rate=0.3, epochs=200, attach_LCA = True, exec_limit=10000)

    # (@ Jon: This is the global LCA (i.e. no within dimension effects) and it does work, using LCAMechanism)
    # mnet_lca = get_trained_network(g, learning_rate=0.3, epochs=1000, attach_LCA = True)

    # Run some simulations

    # Save stats dictionaries to list
    perf_dicts = []

    # single tasking
    # for task in all_tasks:
    # 	perf_dicts.append(evaluate_net_perf_lca(mnet_lca, [task], all_tasks, num_features, num_input_dims, num_output_dims, num_test_points))

    # perf_dicts.append(evaluate_net_perf_lca(mnet_lca, [(4,3)], all_tasks, num_features, num_input_dims, num_output_dims, num_test_points))

    # # multitasking
    for task_i in all_tasks[:2]:
        for task_j in all_tasks[:2]:
            if task_i == task_j:
                continue

            perf_dicts.append(evaluate_net_perf_lca(mnet_lca, [task_i, task_j], all_tasks, num_features, num_input_dims, num_output_dims, num_test_points))

    # perf_dicts.append(evaluate_net_perf_lca(mnet_lca, [(0, 0), (1, 1)], all_tasks, num_features, num_input_dims, num_output_dims, num_test_points))

    # Print summary stats
    for perf_dict in perf_dicts:
        print('tasks: ', perf_dict['tasks'])
        print('num_tasks_correct: ', perf_dict['num_tasks_correct'])
        print('number of test points incorrect:', len(perf_dict['wrong_idxs']))
        print('number of test points correct per task:')
        for task in perf_dict['num_correct_per_task'].keys():
            print('\t', task, ' ', perf_dict['num_correct_per_task'][task])

        if verbose:
            print('all points:')
            for idx in range(num_test_points):
                print('\tidx ', idx)
                print('\tinput: ', perf_dict['inputs'][idx, :])
                print('\ttrue out: ', perf_dict['true_outputs'][idx, :])
                print('\tnet out: ', perf_dict['mnet_out'][idx, :])
                print('\tlca out: ', perf_dict['outputs'][idx, :])
                print()


        print('*****')

    # MSE Testing
    # np.random.seed(2)
    # g, num_tasks, num_input_dims, num_output_dims = read_bipartite_adjlist('./data/bipartite_graphs/7-tasks.txt')

    # mnet = get_trained_network(g, learning_rate=0.3, epochs=5000, attach_LCA = False)

    # all_tasks = get_all_tasks(g)

    #

    # # single tasking (MSE)
    # print('******** single-tasking: ********')
    # print('%.2f' % evaluate_net_perf_mse(mnet, [(1, 1)], all_tasks, 3, num_input_dims, num_output_dims, 100))
    # print('%.2f' % evaluate_net_perf_mse(mnet, [(0, 0)], all_tasks, 3, num_input_dims, num_output_dims, 100))
    # print('%.2f' % evaluate_net_perf_mse(mnet, [(2, 3)], all_tasks, 3, num_input_dims, num_output_dims, 100))
    # print()

    # # multitasking (MSE)
    # print('******** multitasking: ********')
    # print('%.2f\n\n\n' % evaluate_net_perf_mse(mnet, [(0, 0), (1, 1)], all_tasks, 3, num_input_dims, num_output_dims, 100))
    # print('%.2f\n\n\n' % evaluate_net_perf_mse(mnet, [(0, 0), (4, 4)], all_tasks, 3, num_input_dims, num_output_dims, 100))
    # print('%.2f\n\n\n' % evaluate_net_perf_mse(mnet, [(1, 1), (2, 2)], all_tasks, 3, num_input_dims, num_output_dims, 100))

    # print('%.2f\n\n\n' % evaluate_net_perf_mse(mnet, [(1, 1), (2, 2), (3, 3), (4, 4)], all_tasks, 3, num_input_dims, num_output_dims, 100))
