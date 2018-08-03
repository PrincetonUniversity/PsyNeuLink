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





# SET UP PYTORCH AND GATE, MODEL OBJECT OF THIS CLASS
        
class PT_and(torch.nn.Module):
    
    def __init__(self):
        super(PT_and, self).__init__()
        self.w = nn.Parameter(torch.ones(2,1).float())
    
    def forward(self, x):
        q = nn.Sigmoid()
        x = torch.matmul(x, self.w)
        x = q(x)
        return x
    
and_pt = PT_and()



# SET UP MECHANISMS AND PROJECTIONS FOR AND SYSTEM & COMPOSITION

and_in = TransferMechanism(name='and_in',
                           default_variable=np.zeros(2))

and_out = TransferMechanism(name='and_out',
                            default_variable=np.zeros(1),
                            function=Logistic())

and_map = MappingProjection(matrix=np.ones((2,1)))



# SET UP SYSTEM

and_process = Process(pathway=[and_in,
                               and_map,
                               and_out],
                      learning=pnl.LEARNING)

and_system = System(processes=[and_process],
                    learning_rate=10)



# SET UP COMPOSITION

and_comp = ParsingAutodiffComposition(param_init_from_pnl=True)

and_comp.add_c_node(and_in)
and_comp.add_c_node(and_out)

and_comp.add_projection(sender=and_in, projection=and_map, receiver=and_out)



# SET UP INPUTS AND OUTPUTS

and_inputs = np.zeros((6,2))
and_inputs[0] = [0, 0]
and_inputs[1] = [0, 1]
and_inputs[2] = [1, 0]
and_inputs[3] = [1, 1]
and_inputs[4] = [0, 0]
and_inputs[5] = [0, 1]

and_targets = np.zeros((6,1))
and_targets[0] = [0]
and_targets[1] = [0]
and_targets[2] = [0]
and_targets[3] = [1]
and_targets[4] = [0]
and_targets[5] = [0]

print("\n")



# SET UP COMPOSITION (PYTORCH BACKEND), CHECK PARAMS OF ALL

results_comp = and_comp.run(inputs={and_in:and_inputs[3]})

weights, biases = and_comp.get_parameters()

print("weights of system before training: ")
print(and_map.matrix)
print("\n")
print("weights of composition before training: ")
print(weights[and_map])
print("\n")
print("weights of pytorch before training: ")
print(and_pt.w)
print("\n")



# TRAIN COMPOSITION AND PYTORCH, CHECK PARAMS OF ALL

print("starting composition training: ")
print("\n")

results_comp = and_comp.run(inputs={and_in:and_inputs[0:2]},
                            targets={and_out:and_targets[0:2]}, epochs=1, learning_rate=10, optimizer='sgd')

print("starting basic pytorch training: ")
print("\n")

loss = nn.MSELoss(size_average=True)
optimizer = torch.optim.SGD(and_pt.parameters(), lr=10)
'''
for i in range(len(and_inputs[i])):
    inp = torch.from_numpy(and_inputs[i].copy()).float()
    targ = torch.from_numpy(and_targets[i].copy()).float()
    output = and_basicpt.forward(inp)
    l = loss(output, targ)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
'''
for i in range(1):
    inp = torch.from_numpy(and_inputs[0].copy()).float()
    targ = torch.from_numpy(and_targets[0].copy()).float()
    output = and_pt.forward(inp)
    l = loss(output, targ)
    l = l/2
    optimizer.zero_grad()
    print(l)
    l.backward()
    optimizer.step()
    inp = torch.from_numpy(and_inputs[1].copy()).float()
    targ = torch.from_numpy(and_targets[1].copy()).float()
    output = and_pt.forward(inp)
    l = loss(output, targ)
    l = l/2
    optimizer.zero_grad()
    print(l)
    l.backward()
    optimizer.step()
    

weights, biases = and_comp.get_parameters()
print("weights of system after composition, basic pytorch training: ")
print(and_map.matrix)
print("\n")
print("weights of composition after composition, basic pytorch training: ")
print(weights[and_map])
print("\n")
print("weights of basic pytorch after composition, basic pytorch training: ")
print(and_pt.w)
print("\n")

for i in range(1):
    results_sys = and_system.run(inputs={and_in:and_inputs[0:3]}, 
                                 targets={and_out:and_targets[0:3]})

weights, biases = and_comp.get_parameters()
print("weights of system after both training: ")
print(and_map.matrix)
print("\n")
print("weights of composition after both training: ")
print(weights[and_map])
print("\n")
print("weights of basic pytorch after composition, basic pytorch training: ")
print(and_pt.w)
print("\n")




