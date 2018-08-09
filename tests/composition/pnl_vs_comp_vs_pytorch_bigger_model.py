import numpy as np
import torch
from torch import nn

import psyneulink as pnl
from psyneulink.components.system import System
from psyneulink.components.process import Process
from psyneulink.components.functions.function import Linear, Logistic, ReLU, SimpleIntegrator
from psyneulink.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism, TRANSFER_OUTPUT
from psyneulink.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.library.mechanisms.processing.transfer.recurrenttransfermechanism import RecurrentTransferMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.projections.projection import Projection
from psyneulink.components.states.inputstate import InputState
from psyneulink.compositions.composition import Composition, CompositionError, CNodeRole
from psyneulink.compositions.parsingautodiffcomposition import ParsingAutodiffComposition, ParsingAutodiffCompositionError

import timeit



# FIRST UP - BIGGER XOR TYPE THING #######################################################################################

# SET UP PYTORCH CLASS, MODEL OBJECT OF THIS CLASS

class PT_rand(torch.nn.Module):
    
    def __init__(self):
        super(PT_rand, self).__init__()
        self.w1 = nn.Parameter(torch.ones(10,15).float()*0.01)
        self.w2 = nn.Parameter(torch.ones(15,10).float()*0.01)
    
    def forward(self, x):
        q = nn.Sigmoid()
        x = torch.matmul(x, self.w1)
        x = q(x)
        x = torch.matmul(x, self.w2)
        x = q(x)
        return x

rand_pt = PT_rand()



# SET UP MECHANISMS AND PROJECTIONS FOR SYSTEM & COMPOSITION

rand_in = TransferMechanism(name='rand_in',
                            default_variable=np.zeros(10))

'''
rand_hid = TransferMechanism(name='rand_hid',
                             default_variable=np.zeros(40),
                             function=Logistic())
'''

rand_hid1 = TransferMechanism(name='rand_hid1',
                             default_variable=np.zeros(20),
                             function=Logistic())

rand_hid2 = TransferMechanism(name='rand_hid2',
                             default_variable=np.zeros(20),
                             function=Logistic())


rand_out = TransferMechanism(name='rand_out',
                             default_variable=np.zeros(10),
                             function=Logistic())

'''
hid_map = MappingProjection(name='hid_map',
                            matrix=np.ones((10,40))*0.01,
                            sender=rand_in,
                            receiver=rand_hid)
'''

hid_map1 = MappingProjection(name='hid_map1',
                            matrix=np.ones((10,20))*0.01,
                            sender=rand_in,
                            receiver=rand_hid1)

hid_map2 = MappingProjection(name='hid_map2',
                            matrix=np.ones((10,20))*0.01,
                            sender=rand_in,
                            receiver=rand_hid2)

'''
out_map = MappingProjection(name='out_map',
                            matrix=np.ones((40,10))*0.01,
                            sender=rand_hid,
                            receiver=rand_out)
'''

out_map1 = MappingProjection(name='out_map1',
                            matrix=np.ones((20,10))*0.01,
                            sender=rand_hid1,
                            receiver=rand_out)

out_map2 = MappingProjection(name='out_map2',
                            matrix=np.ones((20,10))*0.01,
                            sender=rand_hid2,
                            receiver=rand_out)



# SET UP COMPOSITION

rand_comp = ParsingAutodiffComposition(param_init_from_pnl=True)

rand_comp.add_c_node(rand_in)
# rand_comp.add_c_node(rand_hid)
rand_comp.add_c_node(rand_hid1)
rand_comp.add_c_node(rand_hid2)
rand_comp.add_c_node(rand_out)

# rand_comp.add_projection(sender=rand_in, projection=hid_map, receiver=rand_hid)
# rand_comp.add_projection(sender=rand_hid, projection=out_map, receiver=rand_out)
rand_comp.add_projection(sender=rand_in, projection=hid_map1, receiver=rand_hid1)
rand_comp.add_projection(sender=rand_in, projection=hid_map2, receiver=rand_hid2)
rand_comp.add_projection(sender=rand_hid1, projection=out_map1, receiver=rand_out)
rand_comp.add_projection(sender=rand_hid2, projection=out_map2, receiver=rand_out)




# SET UP INPUTS AND OUTPUTS - GONNA BE 8 TRIAL SETS

np.random.seed(10)

rand_inputs = np.random.randint(2, size=240).reshape(24,10)
rand_targets = np.random.randint(2, size=240).reshape(24,10)



# SET UP COMPOSITION, CHECK PARAMS OF ALL

results_comp = rand_comp.run(inputs={rand_in:rand_inputs[0]})

# weights, biases = rand_comp.get_parameters()
'''
print("weights of system before training: ")
print("\n")
print(hid_map.matrix)
print("\n")
print(out_map.matrix)
print("\n")
print("weights of composition before training: ")
print("\n")
# print(weights[hid_map])
print(rand_comp.model.params[0])
print("\n")
# print(weights[out_map])
print(rand_comp.model.params[1])
print("\n")
print("weights of pytorch before training: ")
print("\n")
print(rand_pt.w1)
print("\n")
print(rand_pt.w2)
print("\n")
print("\n")
'''



# SET SOME TRAINING PARAMETERS

eps = 100
learny = 0.1
ep_size = 24
trials_for_sys = eps*ep_size + 1

print("printing params for this round: ")
print(eps)
print(learny)
print(ep_size)
print(trials_for_sys)
print("\n")
print("\n")



# TRAIN COMPOSITION

print("starting composition training: ")
print("\n")
start = timeit.default_timer()
results_comp = rand_comp.run(inputs={rand_in:rand_inputs},
                             targets={rand_out:rand_targets}, epochs=eps, learning_rate=learny, optimizer='sgd')
end = timeit.default_timer()
comp_time = end - start
print("Composition time: ", comp_time)
print("\n")


# TRAIN PYTORCH

loss = nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(rand_pt.parameters(), lr=learny)

'''
for i in range(eps):
    inp = torch.from_numpy(rand_inputs[0].copy()).float()
    targ = torch.from_numpy(rand_targets[0].copy()).float()
    output = rand_pt.forward(inp)
    l = loss(output, targ)
    l = l/2
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
'''

'''
for i in range(eps):
    for j in range(ep_size):
        inp = torch.from_numpy(rand_inputs[j].copy()).float()
        targ = torch.from_numpy(rand_targets[j].copy()).float()
        output = rand_pt.forward(inp)
        l = loss(output, targ)
        l = l/2
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
'''


# CHECK PARAMS

# weights, biases = rand_comp.get_parameters()
'''
print("weights of system after composition, pytorch training: ")
print("\n")
print(hid_map.matrix)
print("\n")
print(out_map.matrix)
print("\n")
print("weights of composition after composition, pytorch training: ")
print("\n")
# print(weights[hid_map])
print(rand_comp.model.params[0])
print("\n")
# print(weights[out_map])
print(rand_comp.model.params[1])
print("\n")
print("weights of pytorch after composition, pytorch training: ")
print("\n")
print(rand_pt.w1)
print("\n")
print(rand_pt.w2)
print("\n")
print("\n")
'''


# CREATE SYSTEM

'''
rand_process = Process(pathway=[rand_in,
                                hid_map,
                                rand_hid,
                                out_map,
                                rand_out],
                       learning=pnl.LEARNING)
'''

'''
rand_process = Process(pathway=[rand_in,
                                hid_map,
                                rand_hid],
                       learning=pnl.LEARNING)

rand_process1 = Process(pathway=[rand_hid,
                                 out_map,
                                 rand_out],
                        learning=pnl.LEARNING)
'''

rand_process = Process(pathway=[rand_in,
                                hid_map1,
                                rand_hid1,
                                out_map1,
                                rand_out],
                       learning=pnl.LEARNING)

rand_process1 = Process(pathway=[rand_in,
                                hid_map2,
                                rand_hid2,
                                out_map2,
                                rand_out],
                       learning=pnl.LEARNING)

rand_system = System(processes=[rand_process, rand_process1],
                     learning_rate=learny)

# rand_system.show_graph()
# rand_system.show()
# rand_system.show_graph(show_dimensions=pnl.ALL) # , output_fmt = 'jupyter')
rand_system.show_graph(show_projection_labels=True) # , output_fmt = 'jupyter')

start = timeit.default_timer()
results_sys = rand_system.run(inputs={rand_in:rand_inputs}, 
                              targets={rand_out:rand_targets},
                              num_trials=trials_for_sys)
end = timeit.default_timer()
sys_time = end - start

print("System training time: ", sys_time)
print("\n")

print("Speedup: ", (sys_time/comp_time))

# weights, biases = rand_comp.get_parameters()
'''
print("weights of system after all training: ")
print("\n")
print(hid_map.matrix)
print("\n")
print(out_map.matrix)
print("\n")
print("weights of composition after all training: ")
print("\n")
# print(weights[hid_map])
print(rand_comp.model.params[0])
print("\n")
# print(weights[out_map])
print(rand_comp.model.params[1])
print("\n")
print("weights of pytorch after all training: ")
print("\n")
print(rand_pt.w1)
print("\n")
print(rand_pt.w2)
print("\n")
print("\n")
'''





'''
dot = make_dot(output, params=dict(model.named_parameters()))
dot.format = 'svg'
dot.render()


my_Stroop.show()
my_Stroop.show_graph(show_dimensions=pnl.ALL, output_fmt = 'jupyter')
'''


[24.48, 52.770000000000003, 78.140000000000001, 112.48, 163.84, 218.56999999999999, 363.94, 549.29999999999995, 814.97000000000003, 1253.77]
[  60.99   63.63   68.84   68.51   69.3    72.94   71.37   76.91  101.78
  110.05]