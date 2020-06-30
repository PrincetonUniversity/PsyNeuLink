# The following script briefly demos the LeabraMechanism in PsyNeuLink by comparing its output with a corresponding
# network from the leabra package.
# Before running this, please make sure you are using Python 3.5, and that you have installed the leabra package in
# your Python 3.5 environment.

# Installation notes:
#
# If you see an error such as:
#  "Runtime warning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88"
# then this may be an issue with scipy (or other similar modules such as scikit-learn or sklearn).
#
# To resolve this, if you have pip, then use PyCharm to uninstall scipy (or other packages if they continue
# to give you trouble) and then use "pip install scipy --no-use-wheel". Or, if you can figure out how to get PyCharm
#  to ignore warnings, that's fine too.
#
# More info here: https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate
# -binary-incompatibility

import numpy as np
import psyneulink as pnl
import random
import time
import warnings

# suppress warning as described above
import psyneulink.core.components.functions.transferfunctions

warnings.filterwarnings("ignore", message=r".*numpy.dtype size changed.*")

random_seed_value = 1  # feel free to change this value
random.seed(random_seed_value)
num_trials = 10  # how many trials should we run?
hidden_layers = 4  # how many hidden layers are there?
hidden_sizes = [2, 3, 4, 5]  # how big is each hidden layer?
input_pattern = np.repeat(np.array([[0, 1, 3, 4]]), num_trials, axis=0)  # the input
print("inputs to the networks will be: ", input_pattern)
# similar example: input_pattern = [[0, 1, 3, 4]] * int(num_trials/2) + [[0, 0, 0, 0]] * int(num_trials/2)
training_pattern = np.repeat(np.array([[0, 0, 0]]), num_trials, axis=0)  # the training pattern
print("training inputs to the networks will be: ", training_pattern)
input_size = len(input_pattern[0])  # how big is the input layer of the network?
output_size = len(training_pattern[0])  # how big is the output layer of the network?
train_flag = False  # should the LeabraMechanism and leabra network learn?

# NOTE: there is currently a bug with training, in which the output may differ between trials, randomly
# ending up in one of two possible outputs. Running this script repeatedly will make this behavior clear.
# The leabra network and LeabraMechanism experience this bug equally.

# NOTE: The reason TransferMechanisms are used below is because there is currently a bug where LeabraMechanism
# (and other `Mechanism` with multiple input ports) cannot be used as origin Mechanisms for a Composition. If you desire
# to use a LeabraMechanism as an origin Mechanism, you can work around this bug by creating two `TransferMechanism`s
# as origin Mechanisms instead, and have these two TransferMechanisms pass their output to the InputPorts of
# the LeabraMechanism.

# create a LeabraMechanism in PsyNeuLink
L = pnl.LeabraMechanism(
    input_size=input_size,
    output_size=output_size,
    hidden_layers=hidden_layers,
    hidden_sizes=hidden_sizes,
    name='L',
    training_flag=train_flag
)


T1 = pnl.TransferMechanism(name='T1', size=input_size, function=psyneulink.core.components.functions
                           .transferfunctions.Linear)
T2 = pnl.TransferMechanism(name='T2', size=output_size, function=psyneulink.core.components.functions.transferfunctions.Linear)

proj = pnl.MappingProjection(sender=T2, receiver=L.input_ports[1])
comp = pnl.Composition(pathways=[[T1, L], [T2, proj, L]])

print('Running Leabra in PsyNeuLink...')
start_time = time.process_time()
outputs = comp.run(inputs={T1: input_pattern.copy(), T2: training_pattern.copy()})
end_time = time.process_time()

print('Time to run LeabraMechanism in PsyNeuLink: ', end_time - start_time, "seconds")
print('LeabraMechanism Outputs Over Time: ', outputs, type(outputs))
print('LeabraMechanism Final Output: ', outputs[-1], type(outputs[-1]))


random.seed(random_seed_value)
leabra_net = pnl.build_leabra_network(
    n_input=input_size,
    n_output=output_size,
    n_hidden=hidden_layers,
    hidden_sizes=hidden_sizes,
    training_flag=train_flag
)

print('\nRunning Leabra in Leabra...')
start_time = time.process_time()
for i in range(num_trials):
    if train_flag is True:
        pnl.train_leabra_network(leabra_net, input_pattern[i].copy(), training_pattern[i].copy())
    else:
        pnl.run_leabra_network(leabra_net, input_pattern[i].copy())
end_time = time.process_time()
print('Time to run Leabra on its own: ', end_time - start_time, "seconds")
print('Leabra Output: ', [unit.act_m for unit in leabra_net.layers[-1].units], type([unit.act_m for unit in leabra_net.layers[-1].units][0]))
