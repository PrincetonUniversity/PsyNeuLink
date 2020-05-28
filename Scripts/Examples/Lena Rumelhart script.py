import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import matplotlib.axes as ax
#matplotlib inline
import psyneulink as pnl
import psyneulink.core.components.functions.transferfunctions
from psyneulink.core.components.functions.learningfunctions import BackPropagation

nouns = ['oak', 'pine', 'rose', 'daisy', 'canary', 'robin', 'salmon', 'sunfish']
relations = ['is', 'has', 'can']
is_list = ['living', 'living thing', 'plant', 'animal', 'tree', 'flower', 'bird', 'fish', 'big', 'green', 'red',
           'yellfsow']
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

    X_1=np.vstack((np.identity(len(nouns)),np.ones((1,len(nouns)))))
    X_1=X_1.T

    X_2=np.vstack((np.identity(len(relations)),np.ones((1,len(relations)))))
    X_2=X_2.T
    return (X_1, X_2)

nouns_onehot, rels_onehot = gen_input_vals(nouns, relations)

r_1 = np.shape(nouns_onehot)[0]
c_1 = np.shape(nouns_onehot)[1]
r_2 = np.shape(rels_onehot)[0]
c_2 = np.shape(rels_onehot)[1]

############### THIS IS THE PART WHERE WE START BUILDING TRANSFER MECHANISMS ########################

#In order to build in biases, add an extra node to every layer, including the inputs

nouns_in = pnl.TransferMechanism(name="nouns_input",
                                 default_variable=np.zeros(c_1)
                                )

rels_in = pnl.TransferMechanism(name="rels_input",
                                default_variable=np.zeros(c_2)
                               )

h1 = pnl.TransferMechanism(name="hidden_nouns",
                           size=9,
                           function=psyneulink.core.components.functions.transferfunctions.Logistic()
                            )

h2 = pnl.TransferMechanism(name="hidden_mixed",
                           size=16,
                           function=psyneulink.core.components.functions.transferfunctions.Logistic()
                               )

out_sig_I = pnl.TransferMechanism(name="sig_outs_I",
                                  size=len(nouns),
                                  function=psyneulink.core.components.functions.transferfunctions.Logistic()
                                  )

out_sig_is = pnl.TransferMechanism(name="sig_outs_is",
                                   size=len(is_list),
                                   function=psyneulink.core.components.functions.transferfunctions.Logistic()
 )

out_sig_has = pnl.TransferMechanism(name="sig_outs_has",
                                    size=len(has_list),
                                    function=psyneulink.core.components.functions.transferfunctions.Logistic()
                                    )

out_sig_can = pnl.TransferMechanism(name="sig_outs_can",
                                    size=len(can_list),
                                    function=psyneulink.core.components.functions.transferfunctions.Logistic()
                                    )

###################### THIS IS THE PART WHERE I PUT IN THE FORCED RANDOM MATRICES #########################

#alla de maps

map_nouns_h1 = pnl.MappingProjection(matrix=np.random.rand(c_1,c_1),
                                name="map_nouns_h1"
                                )

map_rel_h2 = pnl.MappingProjection(matrix=np.random.rand(c_2,16),
                                name="map_relh2"
                                )

map_h1_h2 = pnl.MappingProjection(matrix=np.random.rand(c_1,16),
                                name="map_h1_h2"
                                )

map_h2_I = pnl.MappingProjection(matrix=np.random.rand(16,len(nouns)),
                                name="map_h2_I"
                                )

map_h2_is = pnl.MappingProjection(matrix=np.random.rand(16,len(is_list)),
                                name="map_h2_is"
                                )

map_h2_has = pnl.MappingProjection(matrix=np.random.rand(16,len(has_list)),
                                name="map_h2_has"
                                )

map_h2_can = pnl.MappingProjection(matrix=np.random.rand(16,len(can_list)),
                                name="map_h2_can"
                                )

#################### THIS IS THE PART WHERE WE START BUILDING OUT ALL THE PROCESSES ########################

p11 = pnl.Pathway(pathway=[nouns_in,
                           map_nouns_h1,
                           h1,
                           map_h1_h2,
                           h2])

p12 = pnl.Pathway(pathway=[rels_in,
                           map_rel_h2,
                           h2])

p21 = pnl.Pathway(pathway=[h2,
                           map_h2_I,
                           out_sig_I])

p22 = pnl.Pathway(pathway=[h2,
                           map_h2_is,
                           out_sig_is])

p23 = pnl.Pathway(pathway=[h2,
                           map_h2_has,
                           out_sig_has])

p24 = pnl.Pathway(pathway=[h2,
                           map_h2_can,
                           out_sig_can])

############################# THIS IS WHERE WE BUILD OUT THE COMPOSITION ###################################

rumel_comp = pnl.Composition(pathways=[(p11, BackPropagation),
                                       (p12, BackPropagation),
                                       (p21, BackPropagation),
                                       (p22, BackPropagation),
                                       (p23, BackPropagation),
                                       (p24, BackPropagation),
                                       ],
                             learning_rate=.5)

rumel_comp.show_graph(output_fmt='jupyter')

############################## THIS IS WHERE WE SETUP THE LOOP VARIABLES #########################################

# THESE ARRAYS STORE THE ERROR VALUES FROM THE SIG OUTPUT AND BIN OUTPUT
delta_bin_array=[]
delta_sig_array=[]

# SET NUMBER OF EPOCHS:
epochs=1000

# CREATE CONDITIONAL:
div = epochs / 100
spits=np.arange(0,epochs,div)

#CREATE KILLSWITCH:
kill=0

############################## THIS IS WHERE WE RUN THE COMPOSITION #########################################

for epoch in range(epochs):
    print("epoch number", epoch)
    for noun in range(len(nouns)):
        for rel_out in range(3):
            # K GIVES THE OUTPUT OF THE COMPOSITION

            k = rumel_comp.learn(inputs={nouns_in: nouns_onehot[noun],
                                         rels_in: rels_onehot[rel_out],
                                         },
                                 targets={out_sig_I: truth_nouns[noun],
                                          out_sig_is: truth_is[noun],
                                          out_sig_has: truth_has[noun],
                                          out_sig_can: truth_can[noun]
                                          },
                                 )

            # PUT K INTO AN ARRAY SO WE CAN MANIPULATE ITS VALUES

            k_array = np.array(k)

            # IT_K GIVES THE OUTPUT FROM THIS SPECIFIC RUN

            it_k = k[np.shape(k_array)[0] - 1]

            # THE DELTAS ADD UP THE SQUARED ERROR FROM EVERY OUTPUT OF K (I, IS, HAS, AND CAN)

            delta = 0
            delta = np.sum((truth_nouns[noun] - np.round(it_k[0])) ** 2)
            delta = delta + np.sum((truth_is[noun] - np.round(it_k[1])) ** 2)
            delta = delta + np.sum((truth_has[noun] - np.round(it_k[2])) ** 2)
            delta = delta + np.sum((truth_can[noun] - np.round(it_k[3])) ** 2)
            delta = delta / (len(nouns) + len(is_list) + len(has_list) + len(can_list))

            delta_sig = 0
            delta_sig = np.sum((truth_nouns[noun] - (it_k[0])) ** 2)
            delta_sig = delta_sig + np.sum((truth_is[noun] - (it_k[1])) ** 2)
            delta_sig = delta_sig + np.sum((truth_has[noun] - (it_k[2])) ** 2)
            delta_sig = delta_sig + np.sum((truth_can[noun] - (it_k[3])) ** 2)
            delta_sig = delta_sig / (len(nouns) + len(is_list) + len(has_list) + len(can_list))

            # THE ARRAYS STORE THE ERROR FROM EVERY RUN. TO SMOOTH THESE, WE CAN AVERAGE THEM OVER EPOCHS.

            delta_bin_array = np.append(delta_bin_array, delta)
            delta_sig_array = np.append(delta_sig_array, delta_sig)

    # PRINT PROGRESS INFORMATION

    if np.isin(epoch, spits):
        print('the average sum squared error on sigmoids for this epoch was', np.sum(delta_sig_array[-25:]) / 24)
        print('the average sum squared error on binaries for this epoch was', np.sum(delta_bin_array[-25:]) / 24)

        # KILL THE LOOP ONCE THE LABELS CONVERGE TO ZERO ERROR FOR A CERTAIN NUMBER OF EPOCHS

    if (np.sum(delta_bin_array[-25:]) / 24) == 0.0:
        kill = kill + 1
        if kill >= 99:
            break

######################## SETUP THE LABEL ERRORS TO BE GRAPHED #######################################

delta_bin_array=np.array(delta_bin_array)
delta_bin_array_trunc = np.array(delta_bin_array[0:int(np.floor(len(delta_bin_array) / 24) * 24)])
height = len(delta_bin_array_trunc) / 24
delta_bin_array_trunc=np.reshape(delta_bin_array_trunc,(int(height),24))
delta_bin_epochs = np.sum(delta_bin_array_trunc, axis=1) / 24

######################## SETUP THE SIGMOID ERRORS TO BE GRAPHED #######################################

delta_sig_array=np.array(delta_sig_array)
delta_sig_array_trunc = delta_sig_array[0:int(np.floor(len(delta_sig_array) / 24) * 24)]
delta_sig_array_trunc=np.reshape(delta_sig_array_trunc,(int(height),24))
delta_sig_epochs = np.sum(delta_sig_array_trunc, axis=1) / 24

######################## DO THE PLOTTING #######################################

plt.plot(delta_bin_epochs)
plt.title('Label error as a function of epochs')
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show()

plt.plot(delta_sig_epochs)
plt.title('sigmoid error as a function of epochs')
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show()
