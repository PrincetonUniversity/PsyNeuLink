
# Create an abstraction very similar to the psyneulink graph, and another very similar
# to the scheduler's execution list

# Write a Pytorch neural network module subclass that takes a graph and an execution list
# and creates a model's parameters and feedforward step with them. 










# imports

from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.functions.function import Linear, Logistic
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection

import collections
from collections import Iterable, OrderedDict
import logging

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F 
import torch.optim as optim

logger = logging.getLogger(__name__)











# ERRORS
class CompositionError(Exception):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)











# VERTEX ABSTRACTION (copied from composition)

class Vertex(object):
    '''
        Stores a Component for use with a `Graph`

        Arguments
        ---------

        component : Component
            the `Component <Component>` represented by this Vertex

        parents : list[Vertex]
            the `Vertices <Vertex>` corresponding to the incoming edges of this `Vertex`

        children : list[Vertex]
            the `Vertices <Vertex>` corresponding to the outgoing edges of this `Vertex`

        Attributes
        ----------

        component : Component
            the `Component <Component>` represented by this Vertex

        parents : list[Vertex]
            the `Vertices <Vertex>` corresponding to the incoming edges of this `Vertex`

        children : list[Vertex]
            the `Vertices <Vertex>` corresponding to the outgoing edges of this `Vertex`
    '''

    def __init__(self, component, parents=None, children=None):
        self.component = component
        if parents is not None:
            self.parents = parents
        else:
            self.parents = []
        if children is not None:
            self.children = children
        else:
            self.children = []

    def __repr__(self):
        return '(Vertex {0} {1})'.format(id(self), self.component)











# GRAPH ABSTRACTION (copied from composition)

class Graph(object):
    '''
        A Graph of vertices and edges/

        Attributes
        ----------

        comp_to_vertex : Dict[`Component <Component>` : `Vertex`]
            maps `Component` in the graph to the `Vertices <Vertex>` that represent them.

        vertices : List[Vertex]
            the `Vertices <Vertex>` contained in this Graph.

    '''

    def __init__(self):
        self.comp_to_vertex = collections.OrderedDict()  # Translate from mechanisms to related vertex
        self.vertices = []  # List of vertices within graph

    def copy(self):
        '''
            Returns
            -------

            A copy of the Graph. `Vertices <Vertex>` are distinct from their originals, and point to the same
            `Component <Component>` object : `Graph`
        '''
        g = Graph()

        for vertex in self.vertices:
            g.add_vertex(Vertex(vertex.component))

        for i in range(len(self.vertices)):
            g.vertices[i].parents = [g.comp_to_vertex[parent_vertex.component] for parent_vertex in self.vertices[i].parents]
            g.vertices[i].children = [g.comp_to_vertex[parent_vertex.component] for parent_vertex in self.vertices[i].children]

        return g

    def add_component(self, component):
        if component in [vertex.component for vertex in self.vertices]:
            logger.info('Component {1} is already in graph {0}'.format(component, self))
        else:
            vertex = Vertex(component)
            self.comp_to_vertex[component] = vertex
            self.add_vertex(vertex)

    def add_vertex(self, vertex):
        if vertex in self.vertices:
            logger.info('Vertex {1} is already in graph {0}'.format(vertex, self))
        else:
            self.vertices.append(vertex)
            self.comp_to_vertex[vertex.component] = vertex

    def remove_component(self, component):
        try:
            self.remove_vertex(self.comp_to_vertex(component))
        except KeyError as e:
            raise CompositionError('Component {1} not found in graph {2}: {0}'.format(e, component, self))

    def remove_vertex(self, vertex):
        try:
            self.vertices.remove(vertex)
            del self.comp_to_vertex[vertex.component]
            # TODO:
            #   check if this removal puts the graph in an inconsistent state
        except ValueError as e:
            raise CompositionError('Vertex {1} not found in graph {2}: {0}'.format(e, vertex, self))

    def connect_components(self, parent, child):
        self.connect_vertices(self.comp_to_vertex[parent], self.comp_to_vertex[child])

    def connect_vertices(self, parent, child):
        if child not in parent.children:
            parent.children.append(child)
        if parent not in child.parents:
            child.parents.append(parent)

    def get_parents_from_component(self, component):
        '''
            Arguments
            ---------

            component : Component
                the Component whose parents will be returned

            Returns
            -------

            A list[Vertex] of the parent `Vertices <Vertex>` of the Vertex associated with **component** : list[`Vertex`]
        '''
        return self.comp_to_vertex[component].parents

    def get_children_from_component(self, component):
        '''
            Arguments
            ---------

            component : Component
                the Component whose children will be returned

            Returns
            -------

            A list[Vertex] of the child `Vertices <Vertex>` of the Vertex associated with **component** : list[`Vertex`]
        '''
        return self.comp_to_vertex[component].children












# Pytorch nn module subclass
class ModelInPytorch(torch.nn.Module):
    
    
    
    # HELPER METHODS
    
    
    # method for creating ordered execution sets from processing graph
    def get_ordered_exec_sets(self, processing_graph):
        
        # initialize lists of ordered execution sets, terminal nodes
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
    
    
    # helper recursive method for creating ordered execution sets from processing graph
    def get_node_pos(self, node, ordered_exec_sets):
        
        # check if current node has already been put in an execution set
        for i in range(len(ordered_exec_sets)):
            if (node in ordered_exec_sets[i]):
                return ordered_exec_sets, i
            
        # check if we are at a node with no parents
        if len(node.parents) == 0:
            if len(ordered_exec_sets) < 1:
                ordered_exec_sets.append([node])
            else:
                ordered_exec_sets[0].append(node)
            return ordered_exec_sets, 0
            
        # if we are at a node with parents
        else:
            
            # iterate over parents, find parent path with maximum length
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
    
    
    # helper function creator
    def activation_function_creator(self, node):
        
        # linear function case
        if isinstance(node.component.function_object, Linear):
            slope = node.component.function_object.params['slope']
            intercept = node.component.function_object.params['intercept']
            return lambda x: x * slope + intercept
        
        # logistic function case
        elif isinstance(node.component.function_object, Logistic):
            gain = node.component.function_object.params['gain']
            bias = node.component.function_object.params['bias']
            offset = node.component.function_object.params['offset']
            return lambda x: 1 / (1 + torch.exp(-gain * (x - bias) + offset))
        
        # relu function case
        else:
            gain = node.component.function_object.params['gain']
            bias = node.component.function_object.params['bias']
            leak = node.component.function_object.params['leak']
            return lambda x: (torch.max(input=(x-bias), other=torch.tensor([0]).float()) * gain + 
                              torch.min(input=(x-bias), other=torch.tensor([0]).float()) * leak)
    
    
    
    # initialization
    def __init__(self, processing_graph):
        
        # do superclass init
        super(ModelInPytorch, self).__init__()
        
        # instance variables
        self.ordered_execution_sets = self.get_ordered_exec_sets(processing_graph) # execution sets
        self.processing_graph = processing_graph # processing graph of model
        self.node_to_feedforward_info = {} # map from PNL node to feedforward information
        self.projections_to_torch_weights = {} # map from PNL projections to corresponding params in pytorch
        self.layer_list = [] # list of layers in model
        self.params = nn.ParameterList() # list of parameters for Pytorch to take note of
        
        # go through nodes in the execution sets one by one, set up above dictionaries for each
        for i in range(len(self.ordered_execution_sets)):
            for j in range(len(self.ordered_execution_sets[i])):
                
                # get current node
                node = self.ordered_execution_sets[i][j]
                
                # create feedforward information list for node
                node_feedforward_info = []
                
                # set up node's tensor and activation function
                layer = torch.zeros(len(node.component.input_states[0].variable)).float()
                activation_function = self.activation_function_creator(node)
                
                # add them to relevant places
                node_feedforward_info.append(layer)
                node_feedforward_info.append(activation_function)
                self.layer_list.append(layer)
                
                # set up node's afferents dictionary. If we have an origin node:
                if len(node.parents) == 0:
                    
                    afferents = {}
                    afferents[j] = None
                    node_feedforward_info.append(afferents)
                
                # if we don't have origin node
                else:
                    
                    afferents = {}
                    for k in range(len(node.component.path_afferents)):
                        mapping_proj = node.component.path_afferents[k]
                        input_component = mapping_proj.sender.owner
                        input_node = self.processing_graph.comp_to_vertex[input_component]
                        weights = nn.Parameter(torch.randn(len(input_component.input_states[0].variable), 
                                                           len(node.component.input_states[0].variable)).float())
                        biases = nn.Parameter(torch.randn(len(node.component.input_states[0].variable)).float())
                        self.params.append(weights)
                        self.params.append(biases)
                        afferents[input_node] = [weights, biases]
                        self.projections_to_torch_weights[mapping_proj] = [weights, biases]
                    node_feedforward_info.append(afferents)
                
                self.node_to_feedforward_info[node] = node_feedforward_info
    
    
    # feedforward method
    def forward(self, inputs):
        
        # zero activations of the layers 
        for i in range(len(self.layer_list)):
            layer = self.layer_list[i]
            layer = torch.zeros(len(layer)).float()
        
        # set up output list
        outputs = []
        
        # iterate over nodes in execution sets
        for i in range(len(self.ordered_execution_sets)):
            for j in range(len(self.ordered_execution_sets[i])):
                
                # get current node, feedforward information for it
                node = self.ordered_execution_sets[i][j]
                node_feedforward_info = self.node_to_feedforward_info[node]
                layer = node_feedforward_info[0]
                activation_function = node_feedforward_info[1]
                afferents = node_feedforward_info[2]
                
                # check if we have origin node (we are in the 1st execution set)
                if (i == 0):
                    
                    # perform feedforward step for node
                    layer = inputs[j]
                    layer = activation_function(layer)
                
                # if we do not have origin node (we are not in the 1st execution set)
                else:
                    
                    # perform feedforward step for node
                    for input_node, param_list in afferents.items():
                        layer += (torch.matmul(self.node_to_feedforward_info[input_node][0], param_list[0]) + param_list[1])
                    layer = activation_function(layer)
                
                # if we're at a node in the last execution set
                if i == len(self.ordered_execution_sets)-1:
                    outputs.append(layer)
        
        return outputs
    
    
    
    # method for retreiving the weights, biases corresponding to psyneulink projections
    def get_parameters_for_projections(self):
        return self.projections_to_torch_weights
    
    
    
    
    
    
    
    





# RUMELHART'S SEMANTIC MODEL


# Mechanisms:

nouns_in = TransferMechanism(name="nouns_input", 
                             default_variable=np.zeros(8)
                             )

rels_in = TransferMechanism(name="rels_input", 
                            default_variable=np.zeros(3)
                            )

h1 = TransferMechanism(name="hidden_nouns",
                       default_variable=np.zeros(8),
                       function=Logistic()
                       )

h2 = TransferMechanism(name="hidden_mixed",
                       default_variable=np.zeros(15),
                       function=Logistic()
                       )

out_sig_I = TransferMechanism(name="sig_outs_I",
                              default_variable=np.zeros(8),
                              function=Logistic()
                              )

out_sig_is = TransferMechanism(name="sig_outs_is",
                               default_variable=np.zeros(12),
                               function=Logistic()
                               )

out_sig_has = TransferMechanism(name="sig_outs_has",
                                default_variable=np.zeros(9),
                                function=Logistic()
                                )

out_sig_can = TransferMechanism(name="sig_outs_can",
                                default_variable=np.zeros(9),
                                function=Logistic()
                                )


# Projections:

map_nouns_h1 = MappingProjection(matrix=np.random.rand(8,8),
                                 name="map_nouns_h1",
                                 sender=nouns_in,
                                 receiver=h1
                                 )

map_rel_h2 = MappingProjection(matrix=np.random.rand(3,15),
                               name="map_relh2",
                               sender=rels_in,
                               receiver=h2
                               )

map_h1_h2 = MappingProjection(matrix=np.random.rand(8,15),
                              name="map_h1_h2",
                              sender=h1,
                              receiver=h2
                              )

map_h2_I = MappingProjection(matrix=np.random.rand(15,8),
                             name="map_h2_I",
                             sender=h2,
                             receiver=out_sig_I
                             )

map_h2_is = MappingProjection(matrix=np.random.rand(15,12),
                              name="map_h2_is",
                              sender=h2,
                              receiver=out_sig_is
                              )

map_h2_has = MappingProjection(matrix=np.random.rand(15,9),
                               name="map_h2_has",
                               sender=h2,
                               receiver=out_sig_has
                               )

map_h2_can = MappingProjection(matrix=np.random.rand(15,9),
                               name="map_h2_can",
                               sender=h2,
                               receiver=out_sig_can
                               )



# Graph with Semantic Model parts

rumel_processing_graph = Graph()

rumel_processing_graph.add_component(nouns_in)
rumel_processing_graph.add_component(rels_in)
rumel_processing_graph.add_component(h1)
rumel_processing_graph.add_component(h2)
rumel_processing_graph.add_component(out_sig_I)
rumel_processing_graph.add_component(out_sig_is)
rumel_processing_graph.add_component(out_sig_has)
rumel_processing_graph.add_component(out_sig_can)

rumel_processing_graph.connect_components(nouns_in, h1)
rumel_processing_graph.connect_components(rels_in, h2)
rumel_processing_graph.connect_components(h1, h2)
rumel_processing_graph.connect_components(h2, out_sig_I)
rumel_processing_graph.connect_components(h2, out_sig_is)
rumel_processing_graph.connect_components(h2, out_sig_has)
rumel_processing_graph.connect_components(h2, out_sig_can)


# test the graph

print("Checking the vertices of the processing graph for the semantic model: ")
print("\n")
for i in range(len(rumel_processing_graph.vertices)):
    vertex = rumel_processing_graph.vertices[i]
    print(vertex)
    print(vertex.component)
    print(vertex.parents)
    print(vertex.children)
    print("\n")


# Create Pytorch model by parsing the processing graph, exec sets 

rumel_parsed_pytorch = ModelInPytorch(rumel_processing_graph)

print("\n")
print("Checking the parameters of the pytorch object representing the semantic model: ")
print("\n")
print(rumel_parsed_pytorch.parameters())
print(rumel_parsed_pytorch)
print("\n")

print("\n")
print("Checking the execution sets created by the pytorch object: ")
print("\n")
for i in range(len(rumel_parsed_pytorch.ordered_execution_sets)):
    print(rumel_parsed_pytorch.ordered_execution_sets[i])
    print("\n")


        
        
        
        
    






# create inputs, outputs for semantic model

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

'''
def gen_input_vals(nouns, relations):

    X_1=np.vstack((np.identity(len(nouns)),np.ones((1,len(nouns)))))
    X_1=X_1.T

    X_2=np.vstack((np.identity(len(relations)),np.ones((1,len(relations)))))
    X_2=X_2.T
    return (X_1, X_2)

nouns_onehot, rels_onehot = gen_input_vals(nouns, relations)
print(nouns_onehot)
print(rels_onehot)

r_1 = np.shape(nouns_onehot)[0]
c_1 = np.shape(nouns_onehot)[1]
r_2 = np.shape(rels_onehot)[0]
c_2 = np.shape(rels_onehot)[1]

print(r_1)
print(c_1)
print(r_2)
print(c_2)
'''

# set up the inputs for pytorch for both models

torch.set_default_tensor_type(torch.FloatTensor)

PT_nouns = torch.eye(len(nouns)).float()
PT_rels = torch.eye(len(relations)).float()
PT_truth_nouns = torch.from_numpy(truth_nouns).float()
PT_truth_is = torch.from_numpy(truth_is).float()
PT_truth_has = torch.from_numpy(truth_has).float()
PT_truth_can = torch.from_numpy(truth_can).float()

print("\n")
print("\n")
print("\n")

'''
print("Checking tensors for inputs, outputs: ")
print(PT_nouns)
print(PT_nouns.shape)
print("\n")

print(PT_rels)
print(PT_rels.shape)
print("\n")

print(PT_truth_nouns)
print(PT_truth_nouns.shape)
print("\n")

print(PT_truth_is)
print(PT_truth_is.shape)
print("\n")

print(PT_truth_has)
print(PT_truth_has.shape)
print("\n")

print(PT_truth_can)
print(PT_truth_can.shape)
print("\n")
'''







'''
# takes inputs and targets for pytorch model and trains them 
    def autodiff_training(self, inputs, targets, epochs_or_stop_learning_condition=None):
        
        # convert inputs, targets to torch tensors
        tensor_inputs = np.empty([len(inputs)], dtype=object)
        tensor_targets = np.empty([len(targets)], dtype=object)
        
        for t in range(len(inputs)):
            
            curr_tensor_inputs = []
            curr_tensor_targets = []
            
            for i in len(inputs[t]):
                curr_tensor_inputs.append(torch.from_numpy(np.asarray(inputs[t][i])).float())
            tensor_inputs[t] = curr_tensor_inputs
            
            for i in len(targets[t]):
                curr_tensor_targets.append(torch.from_numpy(np.asarray(targets[t][i])).float())
            tensor_targets[t] = curr_tensor_targets
        
        # train model
        
        # NOTE 1: currently, learning conditions are not supported - first implementation defaults
        # to 50 epochs, using mean-squared-error loss and the Adam optimizer with certain default settings
        
        # NOTE 2: currently, tensor_inputs/targets is a numpy array of lists of tensors - this should be simplified
        
        # set loss criterion, optimizer, output list for holding outputs on final iteration
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters)
        outputs = np.empty([len(inputs)], dtype=object)
        
        # iterate over epochs
        for epoch in range(epochs_or_stop_learning_condition):
            
            # set a random number seed
            torch.manual_seed(epoch)
            
            # get a random permutation of inputs/targets
            rand_train_order = np.random.permutation(len(tensor_inputs))
            
            # iterate over inputs/targets
            for t in range(len(tensor_inputs)):
                
                # get current inputs, targets
                curr_tensor_inputs = tensor_inputs[rand_train_order[t]]
                curr_tensor_targets = tensor_targets[rand_train_order[t]]
                
                # run the model on inputs
                output_tuple = self.model.forward(*curr_tensor_inputs)
                
                # compute loss
                loss = torch.zeros(1).float()
                for i in range(len(output_tuple)):
                    loss += criterion(output_tuple[i], curr_tensor_targets[i])
                
                # compute gradients and perform parameter update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # save outputs of model if this is final epoch
                if epoch == epochs_or_stop_learning_condition - 1:
                    curr_output_list = []
                    for i in range(len(output_tuple)):
                        curr_output_list.append(output_tuple[i].numpy())
                    outputs[rand_train_order[t]] = curr_output_list
            
            return outputs
'''












print("Tryna run this bitch: ")
print("\n")

ready_inputs = []
ready_targets = []

for i in range(len(PT_nouns)):
    for j in range(len(PT_rels)):
        
        ready_inputs.append([PT_nouns[i], PT_rels[j]])
        ready_targets.append([PT_truth_nouns[i], PT_truth_is[i], PT_truth_has[i], PT_truth_can[i]])

print(len(ready_inputs))
print(len(ready_targets))

for i in range(24):
    print(len(ready_inputs[i]))
    print(len(ready_targets[i]))



























