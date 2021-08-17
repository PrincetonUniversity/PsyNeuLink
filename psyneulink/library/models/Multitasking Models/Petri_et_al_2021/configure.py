import numpy as np
from ..Base_Models import NNModel
from ..Tools import *

# Network parameters
init_scale = 0.1        # scales for initialized random weights
learning_rate = 0.3     # learning rate
decay = 0.0             # weight penalization parameter
bias = -5               # weight from bias units to hidden & output units
thresh = 0.001          # mean-squared error stopping criterion
hidden_path_size = 1    # group size of hidden units that receive the same weights from the task layer
output_path_size = 1    # group size of output units that receive the same weights from the task layer

# Training environment parameters
subsample_graphs = 1
n_subsamples = 5
sd_scale = 0
same_stimuli_across_tasks = 1
samples_per_task_train = 100
samples_per_task_test = 1000
fix_n_features = False
n_features_fixed = 6


def main(graph_input, hidden_arg, silence):
    # Get graph
    a = graph_input
    # Set number of features per dimension
    if not fix_n_features:
        n_features = np.min(np.size(a))
    else:
        n_features = n_features_fixed
    # Compute dependency graph and MIS of dependency graph
    a_dual = get_dependency_graph(a)
    # TODO
    samples_per_task = samples_per_task_train
    n_pathways = np.max(np.size(a))
    input_data, tasks_data, train_data = generate_environment_to_graph(a, n_features,
        samples_per_task, sd_scale, same_stimuli_across_tasks)
    # TEMP: Read input/task/train/data
    input_data = np.loadtxt(open('input_sglt.csv', 'rb'), delimiter=',', skiprows=1)
    tasks_data = np.loadtxt(open('tasks_sglt.csv', 'rb'), delimiter=',', skiprows=1)
    train_data = np.loadtxt(open('train_sglt.csv', 'rb'), delimiter=',', skiprows=1)
    # Build network
    n_hidden = hidden_arg
    task_net = NNModel(n_hidden, learning_rate, bias, init_scale, thresh, decay,
        hidden_path_size, output_path_size)
    task_net.n_pathways = n_pathways
    task_net.silence = silence
    # Initialize network according to bipartite graph
    task_net.set_data(input_data, tasks_data, train_data)
    task_net.configure()
    return task_net
