# This script demos the LeabraMechanism by training it to learn a simple linear rule.
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

import psyneulink as pnl
import numpy as np

### building the LeabraMechanism
n_input = 4  # don't change this!
n_output = 2  # don't change this!
n_hidden = 0
Leab = pnl.LeabraMechanism(input_size=n_input, output_size=n_output, hidden_layers=n_hidden,
                        hidden_sizes=None, training_flag=True, quarter_size=20)

### building the PsyNeuLink network
T_input = pnl.TransferMechanism(size=n_input)
T_target = pnl.TransferMechanism(size=n_output)
# target_projection connects T_target to the TARGET InputPort of Leab
target_projection = pnl.MappingProjection(sender=T_target, receiver = Leab.input_ports[1])
comp = pnl.Composition(pathways=[[T_input, Leab], [T_target, target_projection, Leab]])

### building the learning data
n = 1000
inputs = [None] * n
targets = [None] * n
print("here's what the inputs/targets will look like:")
for i in range(n):
    nums = np.random.randint(0, 7, size=2) * 0.4
    a = nums[0]
    b = nums[1]
    inputs[i] = [a, a, b, b]
    if a > b:
        targets[i] = [1, 0]
    elif b > a:
        targets[i] = [0, 1]
    else:
        targets[i] = [0.5, 0.5]

    if i < 4:
        print("example input", i, ":", inputs[i])
        print("target", i, ":", targets[i])

### do the training
n_trials = 50
for i in range(n_trials):
    if i % 10 == 0:
        print("trial", i, "out of", n_trials)
    comp_output = comp.run(inputs = {T_input: inputs, T_target: targets})

Leab.training_flag = False

print(Leab.execute([[.3, .3, .1, .1], [0, 0]]))
print(Leab.execute([[.1, .1, .4, .4], [0, 0]]))
print(Leab.execute([[.2, .2, .6, .6], [0, 0]]))
print(Leab.execute([[.3, .3, .4, .4], [0, 0]]))