import numpy as np
from ...Base_Models import NNModel
from ...Tools import *

# Network parameters
num_hidden_units = 100  # number of hidden units
init_scale = 0.1        # scales for initialized random weights
learning_rate = 0.3     # learning rate
decay = 0               # weight penalization parameter
bias = -4               # weight from bias units to hidden & output units
thresh = 0.001          # mean-squared error stopping criterion
hidden_path_size = 1    # group size of hidden units that receive the same weights from the task layer
output_path_size = 1    # group size of output units that receive the same weights from the task layer

# Training environment parameters
sd_scale = 0            # max. variance for around each stimulus input unit activation
n_pathways = 6          # number of pathways (i.e. number of feature dimensions % output dimensions)
n_features = 3          # number of feature units per stimulus input dimension
same_stimuli_across_tasks = 1 # use same stimuli across tasks? (this parameter is only relevant if sdScale > 0)
samples_per_task = []


def config():
    # Create 3 different initial training environments
    input_shared1, tasks_shared1, train_shared1 = \
        create_task_patterns(n_pathways, n_features, samples_per_task, sd_scale,
            same_stimuli_across_tasks, [1])
    input_shared2, tasks_shared2, train_shared2 = \
        create_task_patterns(n_pathways, n_features, samples_per_task, sd_scale,
            same_stimuli_across_tasks, [1, 8])
    input_shared3, tasks_shared3, train_shared3 = \
        create_task_patterns(n_pathways, n_features, samples_per_task, sd_scale,
            same_stimuli_across_tasks, [1, 8, 15])
    # Create test training environment
    input_tested, tasks_tested, train_tested = \
        create_task_patterns(n_pathways, n_features, samples_per_task, sd_scale,
            same_stimuli_across_tasks, [4, 11, 18])
    # Create full envirnments
    tasks_to_perform = range(1, n_pathways*n_pathways)
    input, tasks, train = \
        create_task_patterns(n_pathways, n_features, samples_per_task, sd_scale,
            same_stimuli_across_tasks, tasks_to_perform)
    # Initialize network
    tasknet_shared0 = NNModel(num_hidden_units, learning_rate, bias, init_scale, thresh, decay,
        hidden_path_size, output_path_size)
    tasknet_shared0.set_data(input_shared3, tasks_shared3, train_shared3)
    tasknet_shared0.configure()
    tasknet_shared0.input_tested = input_tested
    tasknet_shared0.tasks_tested = tasks_tested
    tasknet_shared0.train_tested = train_tested
    # Generate 3 network conditions
    tasknet_shared1 = NNModel(num_hidden_units, learning_rate, bias, init_scale, thresh, decay,
        hidden_path_size, output_path_size)
    tasknet_shared1.n_pathways = n_pathways
    tasknet_shared2 = NNModel(num_hidden_units, learning_rate, bias, init_scale, thresh, decay,
        hidden_path_size, output_path_size)
    tasknet_shared2.n_pathways = n_pathways
    tasknet_shared3 = NNModel(num_hidden_units, learning_rate, bias, init_scale, thresh, decay,
        hidden_path_size, output_path_size)
    tasknet_shared3.n_pathways = n_pathways
    # Set data to pretrain all three networks
    tasknet_shared1.set_data(input_shared1, tasks_shared1, train_shared1)
    tasknet_shared1.configure()
    tasknet_shared2.set_data(input_shared2, tasks_shared2, train_shared2)
    tasknet_shared2.configure()
    tasknet_shared3.set_data(input_shared3, tasks_shared3, train_shared3)
    tasknet_shared3.configure()
    return tasknet_shared0, tasknet_shared1, tasknet_shared2, tasknet_shared3
