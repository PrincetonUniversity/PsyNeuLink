from psyneulink.components.functions.function import Linear, Logistic
import logging
logger = logging.getLogger(__name__)
import torch
from torch import nn
import numpy as np

# Class that creates pytorch model objects from processing graphs

class PytorchCreator(torch.nn.Module):
    
    # helper methods ----------------------------------------------------------------------------------------
    
    # returns activation function for node
    def activation_function_creator(self, node):
        
        # linear function
        if isinstance(node.component.function_object, Linear):
            slope = node.component.function_object.params['slope']
            intercept = node.component.function_object.params['intercept']
            return lambda x: x * slope + intercept
        
        # logistic function
        elif isinstance(node.component.function_object, Logistic):
            gain = node.component.function_object.params['gain']
            bias = node.component.function_object.params['bias']
            offset = node.component.function_object.params['offset']
            return lambda x: 1 / (1 + torch.exp(-gain * (x - bias) + offset))
        
        # relu function
        else:
            gain = node.component.function_object.params['gain']
            bias = node.component.function_object.params['bias']
            leak = node.component.function_object.params['leak']
            return lambda x: (torch.max(input=(x-bias), other=torch.tensor([0]).double()) * gain + 
                              torch.min(input=(x-bias), other=torch.tensor([0]).double()) * leak)
    
    # returns dict mapping psyneulink projections to corresponding pytorch weights (in numpy arrays, not torch tensors)
    def get_weights_for_projections(self):
        copied_to_numpy = {}
        for projection, weights in self.projections_to_torch_weights.items():
            copied_to_numpy[projection] = weights.detach().numpy().copy()
        return copied_to_numpy
    
    # returns dict mapping psyneulink mechanisms to corresponding pytorch biases (in numpy arrays, not torch tensors)
    def get_biases_for_mechanisms(self):
        copied_to_numpy = {}
        for mechanism, biases in self.mechanisms_to_torch_biases.items():
            copied_to_numpy[mechanism] = biases.detach().numpy().copy()
        return copied_to_numpy
    
    # init and forward methods ------------------------------------------------------------------------------
    
    # sets up parameters of model, information for performing feedfoward step
    def __init__(self, processing_graph, param_init_from_pnl, ordered_execution_sets):
        
        super(PytorchCreator, self).__init__()
        
        # instance variables
        self.ordered_execution_sets = ordered_execution_sets
        self.processing_graph = processing_graph # processing graph
        self.node_to_feedforward_info = {} # dict mapping PNL nodes to feedforward info
        self.projections_to_torch_weights = {} # dict mapping PNL projections to pytorch parameters
        self.mechanisms_to_torch_biases = {} # dict mapping PNL mechanisms to pytorch biases
        self.params = nn.ParameterList() # list of parameters for Pytorch to keep track of
        
        # iterate over nodes in execution sets, set up feedforward info for each
        for i in range(len(self.ordered_execution_sets)):
            for j in range(len(self.ordered_execution_sets[i])):
                
                # get current node, create feedforward info list for node
                node = self.ordered_execution_sets[i][j]
                node_feedforward_info = []
                
                # set up node's tensor, biases, activation function, afferent inputs info
                layer = None
                biases = None
                activation_function = self.activation_function_creator(node)
                afferents = {}
                
                # if we don't have origin node: set up biases, add to afferent inputs info
                if len(node.parents) > 0:
                    
                    # if not copying params from psyneulink, set up biases for node/mechanism,
                    # add biases to params list, mechanisms_to_torch_biases dict
                    if param_init_from_pnl == False:
                        biases = nn.Parameter(torch.zeros(len(node.component.input_states[0].value)).double())
                        self.params.append(biases)
                        self.mechanisms_to_torch_biases[node.component] = biases
                    
                    # iterate over projections to node
                    for k in range(len(node.component.path_afferents)):
                        
                        # get projection, sender component/node for projection
                        mapping_proj = node.component.path_afferents[k]
                        input_component = mapping_proj.sender.owner
                        input_node = self.processing_graph.comp_to_vertex[input_component]
                        
                        # set up pytorch weights that correspond to projection. If copying params from psyneulink,
                        # copy weight values from projection. Otherwise, use random values.
                        if param_init_from_pnl == True:
                            weights = nn.Parameter(torch.tensor(mapping_proj.matrix.copy()).double())
                        else:
                            weights = nn.Parameter(torch.rand(np.shape(mapping_proj.matrix)).double())
                        
                        # add node-weights mapping to afferent inputs info, add weights to params list,
                        # add weights to projections_to_torch_weights dict
                        afferents[input_node] = weights
                        self.params.append(weights)
                        self.projections_to_torch_weights[mapping_proj] = weights
                
                # append node's tensor, biases, activation function, afferent inputs info to node's feedforward info
                node_feedforward_info.append(layer)
                node_feedforward_info.append(biases)
                node_feedforward_info.append(activation_function)
                node_feedforward_info.append(afferents)
                
                # add node's feedforward info to feedforward info dict
                self.node_to_feedforward_info[node] = node_feedforward_info
    
    # performs feedfoward computation for the model, creating its computational graph
    def forward(self, inputs):
        
        # set up output list
        outputs = []
        
        # iterate over nodes in execution sets
        for i in range(len(self.ordered_execution_sets)):
            for j in range(len(self.ordered_execution_sets[i])):
                
                # get current node, node's biases, activation function & afferent info
                node = self.ordered_execution_sets[i][j]
                biases = self.node_to_feedforward_info[node][1]
                activation_function = self.node_to_feedforward_info[node][2]
                afferents = self.node_to_feedforward_info[node][3]
                
                # feedforward step if we have origin node
                if (i == 0):
                    layer = activation_function(inputs[j])
                
                # feedforward step if we do not have origin node
                else:
                    layer = torch.zeros(len(node.component.input_states[0].value)).double()
                    for input_node, weights in afferents.items():
                        layer += torch.matmul(self.node_to_feedforward_info[input_node][0], weights)
                    if biases is not None:
                        layer = layer + biases
                    layer = activation_function(layer)
                
                # put layer in correct place in the feedforward info dict
                self.node_to_feedforward_info[node][0] = layer
                
                # save outputs if we're at a node in the last execution set
                if i == len(self.ordered_execution_sets)-1:
                    outputs.append(layer)
        
        return outputs
