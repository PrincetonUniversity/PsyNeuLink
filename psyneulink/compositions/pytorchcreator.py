
# IMPORTS

from psyneulink.components.functions.function import Linear, Logistic
import logging
logger = logging.getLogger(__name__)
import torch
from torch import nn




# NN MODULE SUBCLASS THAT CREATES PYTORCH MODEL OBJECTS FROM PROCESSING GRAPHS

class PytorchCreator(torch.nn.Module):
    
    
    
    # HELPER METHODS
    
    # creates ordered execution sets from processing graph
    def get_ordered_exec_sets(self, processing_graph):
        
        # set up lists of ordered execution sets, terminal nodes
        ordered_exec_sets = []
        terminal_nodes = []
        
        # create list of terminal nodes in processing graph
        for i in range(len(processing_graph.vertices)):
            node = processing_graph.vertices[i]
            if len(node.children) == 0:
                terminal_nodes.append(node)
        
        # iterate over terminal nodes, call recursive function to create ordered execution sets
        for i in range(len(terminal_nodes)):
            node = terminal_nodes[i]
            ordered_exec_sets, node_pos = self.get_node_pos(node, ordered_exec_sets)
        
        return ordered_exec_sets
    
    # recursive helper method for get_ordered_exec_sets
    def get_node_pos(self, node, ordered_exec_sets):
        
        # if node has already been put in execution sets
        for i in range(len(ordered_exec_sets)):
            if (node in ordered_exec_sets[i]):
                return ordered_exec_sets, i
            
        # if node has no parents
        if len(node.parents) == 0:
            if len(ordered_exec_sets) < 1:
                ordered_exec_sets.append([node])
            else:
                ordered_exec_sets[0].append(node)
            return ordered_exec_sets, 0
            
        # if node has parents
        else:
            
            # call function on parents, find parent path with max length
            max_dist = -1
            for i in range(len(node.parents)):
                parent = node.parents[i]
                ordered_exec_sets, dist = self.get_node_pos(parent, ordered_exec_sets)
                dist += 1
                if dist > max_dist: 
                    max_dist = dist
            
            # set node at position = max_dist in the ordered execution sets list
            if len(ordered_exec_sets) < (max_dist+1):
                ordered_exec_sets.append([node])
            else:
                ordered_exec_sets[max_dist].append(node)
            return ordered_exec_sets, max_dist
    
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
            return lambda x: (torch.max(input=(x-bias), other=torch.tensor([0]).float()) * gain + 
                              torch.min(input=(x-bias), other=torch.tensor([0]).float()) * leak)
    
    # returns dict mapping psyneulink projections to corresponding pytorch parameters
    def get_parameters_for_projections(self):
        return self.projections_to_torch_weights
    
    # returns ordered execution sets
    def get_ordered_execution_sets(self):
        return self.ordered_execution_sets
    
    
    
    # INIT AND FEEDFORWARD
    
    def __init__(self, processing_graph):
        
        super(PytorchCreator, self).__init__()
        
        # instance variables
        self.ordered_execution_sets = self.get_ordered_exec_sets(processing_graph) # execution sets
        self.processing_graph = processing_graph # processing graph
        self.node_to_feedforward_info = {} # dict mapping PNL nodes to feedforward info
        self.projections_to_torch_weights = {} # dict mapping PNL projections to pytorch parameters
        self.params = nn.ParameterList() # list of parameters for Pytorch to keep track of
        
        # iterate over nodes in execution sets, set up feedforward info for each
        for i in range(len(self.ordered_execution_sets)):
            for j in range(len(self.ordered_execution_sets[i])):
                
                # get current node, create feedforward info list for node
                node = self.ordered_execution_sets[i][j]
                node_feedforward_info = []
                
                # set up node's tensor, activation function, afferent inputs info
                layer = None
                activation_function = self.activation_function_creator(node)
                afferents = {}
                
                # add tensor, activation function to node's feedforward info
                node_feedforward_info.append(layer)
                node_feedforward_info.append(activation_function)
                
                # add to afferent inputs info if we don't have origin node
                if len(node.parents) > 0:
                    
                    # iterate over projections to node
                    for k in range(len(node.component.path_afferents)):
                        
                        # get projection, sender component/node for projection
                        mapping_proj = node.component.path_afferents[k]
                        input_component = mapping_proj.sender.owner
                        input_node = self.processing_graph.comp_to_vertex[input_component]
                        
                        # set up pytorch parameters that correspond to projection
                        weights = nn.Parameter(torch.randn(len(input_component.input_states[0].value), 
                                                           len(node.component.input_states[0].value)).float())
                        biases = nn.Parameter(torch.randn(len(node.component.input_states[0].value)).float())
                        
                        # add parameters to params list, afferent inputs info, projection to parameter dict
                        self.params.append(weights)
                        self.params.append(biases)
                        afferents[input_node] = [weights, biases]
                        self.projections_to_torch_weights[mapping_proj] = [weights, biases]
                
                # add afferent inputs info to node's feedforward info, node's feedforward info to feedforward info dict
                node_feedforward_info.append(afferents)
                self.node_to_feedforward_info[node] = node_feedforward_info
    
    def forward(self, inputs):
        
        # set up output list
        outputs = []
        
        # iterate over nodes in execution sets
        for i in range(len(self.ordered_execution_sets)):
            for j in range(len(self.ordered_execution_sets[i])):
                
                # get current node, node's activation function & afferent info
                node = self.ordered_execution_sets[i][j]
                activation_function = self.node_to_feedforward_info[node][1]
                afferents = self.node_to_feedforward_info[node][2]
                
                # feedforward step if we have origin node
                if (i == 0):
                    layer = activation_function(inputs[j])
                
                # feedforward step if we do not have origin node
                else:
                    layer = torch.zeros(len(node.component.input_states[0].value))
                    for input_node, param_list in afferents.items():
                        layer += (torch.matmul(self.node_to_feedforward_info[input_node][0], param_list[0]) + param_list[1])
                    layer = activation_function(layer)
                
                # put layer in correct place in the feedforward info dict
                self.node_to_feedforward_info[node][0] = layer
                
                # save outputs if we're at a node in the last execution set
                if i == len(self.ordered_execution_sets)-1:
                    outputs.append(layer)
        
        return outputs
    
    
    
    