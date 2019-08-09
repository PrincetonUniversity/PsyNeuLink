import psyneulink as pnl
import numpy as np
# import pytest

# imports specific to the lab:

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import matplotlib.axes as ax

from mpl_toolkits import mplot3d
import psyneulink as pnl

# This couple of lines sets our color palette
import seaborn as sb

sb.palplot(sb.color_palette("RdBu_r", 7))
sb.set_palette("RdBu_r", 7)
sb.set_style("whitegrid")

print("Imports Successful")
print("This will be our color palette")



# Stimuli and Relations

nouns = ['oak', 'pine', 'rose', 'daisy', 'canary', 'robin', 'salmon', 'sunfish']
relations = ['is', 'has', 'can']
is_list = ['living', 'living thing', 'plant', 'animal', 'tree', 'flower', 'bird', 'fish', 'big', 'green', 'red',
           'yellow']
has_list = ['roots', 'leaves', 'bark', 'branches', 'skin', 'feathers', 'wings', 'gills', 'scales']
can_list = ['grow', 'move', 'swim', 'fly', 'breathe', 'breathe underwater', 'breathe air', 'walk', 'photosynthesize']
descriptors = [nouns, is_list, has_list, can_list]

truth_nouns = np.identity(len(nouns))

truth_is = np.zeros((len(nouns), len(is_list)))

truth_is[0, :] = [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
truth_is[1, :] = [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
truth_is[2, :] = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
truth_is[3, :] = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
truth_is[4, :] = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
truth_is[5, :] = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
truth_is[6, :] = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0]
truth_is[7, :] = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0]

truth_has = np.zeros((len(nouns), len(has_list)))

truth_has[0, :] = [1, 1, 1, 1, 0, 0, 0, 0, 0]
truth_has[1, :] = [1, 1, 1, 1, 0, 0, 0, 0, 0]
truth_has[2, :] = [1, 1, 0, 0, 0, 0, 0, 0, 0]
truth_has[3, :] = [1, 1, 0, 0, 0, 0, 0, 0, 0]
truth_has[4, :] = [0, 0, 0, 0, 1, 1, 1, 0, 0]
truth_has[5, :] = [0, 0, 0, 0, 1, 1, 1, 0, 0]
truth_has[6, :] = [0, 0, 0, 0, 0, 0, 0, 1, 1]
truth_has[7, :] = [0, 0, 0, 0, 0, 0, 0, 1, 1]

truth_can = np.zeros((len(nouns), len(can_list)))

truth_can[0, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
truth_can[1, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
truth_can[2, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
truth_can[3, :] = [1, 0, 0, 0, 0, 0, 0, 0, 1]
truth_can[4, :] = [1, 1, 0, 1, 1, 0, 1, 1, 0]
truth_can[5, :] = [1, 1, 0, 1, 1, 0, 1, 1, 0]
truth_can[6, :] = [1, 1, 1, 0, 1, 1, 0, 0, 0]
truth_can[7, :] = [1, 1, 1, 0, 1, 1, 0, 0, 0]

truths = [[truth_nouns], [truth_is], [truth_has], [truth_can]]


def gen_input_vals(nouns, relations):
    rumel_nouns_bias = np.vstack((np.identity(len(nouns)), np.ones((1, len(nouns)))))
    rumel_nouns_bias = rumel_nouns_bias.T

    rumel_rels_bias = np.vstack((np.identity(len(relations)), np.ones((1, len(relations)))))
    rumel_rels_bias = rumel_rels_bias.T
    return (rumel_nouns_bias, rumel_rels_bias)


nouns_onehot, rels_onehot = gen_input_vals(nouns, relations)

r_nouns = np.shape(nouns_onehot)[0]
c_nouns = np.shape(nouns_onehot)[1]
r_rels = np.shape(rels_onehot)[0]
c_rels = np.shape(rels_onehot)[1]

# Constructing the network itself:

rep_in = pnl.TransferMechanism(size=c_nouns, name='rep_in')
rel_in = pnl.TransferMechanism(size=c_rels, name='rel_in')
rep_hidden = pnl.TransferMechanism(size=4,
                                   function=pnl.Logistic,
                                   name='rep_hidden')
rel_hidden = pnl.TransferMechanism(size=5, function=pnl.Logistic, name='rel_hidden')
rep_out = pnl.TransferMechanism(size=len(nouns), function=pnl.Logistic, name='rep_out')
prop_out = pnl.TransferMechanism(size=len(is_list), function=pnl.Logistic, name='prop_out') #'is'
qual_out = pnl.TransferMechanism(size=len(has_list), function=pnl.Logistic, name='qual_out') # 'has'
act_out = pnl.TransferMechanism(size=len(can_list), function=pnl.Logistic, name='act_out') # 'can'

comp = pnl.Composition()

learning_path_1 = comp.add_backpropagation_pathway(pathway=[rel_in, rel_hidden], learning_rate= 2)
learning_path_2 = comp.add_backpropagation_pathway(pathway=[rel_hidden, rep_out], learning_rate= 2)
learning_path_3 = comp.add_backpropagation_pathway(pathway=[rel_hidden, prop_out], learning_rate= 2)
learning_path_4 = comp.add_backpropagation_pathway(pathway=[rel_hidden, qual_out], learning_rate= 2)
learning_path_5 = comp.add_backpropagation_pathway(pathway=[rel_hidden, act_out], learning_rate= 2)
learning_path_6 = comp.add_backpropagation_pathway(pathway=[rep_in, rep_hidden, rel_hidden], learning_rate= 2)

rep_out_targ = learning_path_2[pnl.TARGET_MECHANISM]
prop_out_targ = learning_path_3[pnl.TARGET_MECHANISM]
qual_out_targ = learning_path_4[pnl.TARGET_MECHANISM]
act_out_targ = learning_path_5[pnl.TARGET_MECHANISM]

# comp.show_graph(show_learning=True)
# validate_learning_mechs(comp)

# Creates the targets that will be assigned to outputs irrelevant
# to the input pairs.

irrel_is = np.ones((len(nouns), len(is_list))) * .5
irrel_has = np.ones((len(nouns), len(has_list))) * .5
irrel_can = np.ones((len(nouns), len(can_list))) * .5

# This block of code trains the network using a set of three loops. The innermost
# pair of loops takes each noun and creates the appropriate training inputs and outputs associated
# with its "is", "has", and "can" relations. It will also be associated with an
# identity output.

# After constructing the dictionaries, the middle loop, associated with the nouns,
# trains the network on the dictionaries for n_epochs.

# The outermost loop simply repeats the training on each noun for a set number of
# repetitions.

# You are encouraged to experiment with changing the number of repetitions and
# epochs to see how the network learns best.

# You will find that this code takes a few minutes to run. We have placed flags
# in the loops so you can see that it's not stuck.

num_trials = 1
tot_reps = 50

for reps in range(tot_reps):
    print('Training rep: ', reps + 1, ' of: ', tot_reps)
    for noun in range(len(nouns)):

        inputs_dict = {rep_in: [], rel_in: [], rep_out_targ: [], prop_out_targ: [], qual_out_targ: [], act_out_targ: []}


        for i in range(len(relations)):

            if i == 0:
                rel = 'is'

                targ_is = truth_is[noun],
                targ_has = irrel_has[noun],
                targ_can = irrel_can[noun],

                targ_is = np.reshape(targ_is, np.amax(np.shape(targ_is)))
                targ_has = np.reshape(targ_has, np.amax(np.shape(targ_has)))
                targ_can = np.reshape(targ_can, np.amax(np.shape(targ_can)))

            elif i == 1:
                rel = 'has'

                targ_is = irrel_is[noun],
                targ_has = truth_has[noun],
                targ_can = irrel_can[noun],

                targ_is = np.reshape(targ_is, np.amax(np.shape(targ_is)))
                targ_has = np.reshape(targ_has, np.amax(np.shape(targ_has)))
                targ_can = np.reshape(targ_can, np.amax(np.shape(targ_can)))


            else:
                rel = 'can'

                targ_is = irrel_is[noun],
                targ_has = irrel_has[noun],
                targ_can = truth_can[noun],

                targ_is = np.reshape(targ_is, np.amax(np.shape(targ_is)))
                targ_has = np.reshape(targ_has, np.amax(np.shape(targ_has)))
                targ_can = np.reshape(targ_can, np.amax(np.shape(targ_can)))

            inputs_dict[prop_out_targ].append(targ_is)
            inputs_dict[qual_out_targ].append(targ_has)
            inputs_dict[act_out_targ].append(targ_can)

            inputs_dict[rep_in].append(nouns_onehot[noun])
            inputs_dict[rep_out_targ].append(truth_nouns[noun])
            inputs_dict[rel_in].append(rels_onehot[i])

            print(inputs_dict)

            result = comp.run(
                                inputs=inputs_dict
                              )

            print(result)
