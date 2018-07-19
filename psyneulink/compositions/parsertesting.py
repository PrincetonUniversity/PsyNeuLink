
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
from torchviz import make_dot

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
        self.params = nn.ParameterList() # list of parameters for Pytorch to take note of
        
        # go through nodes in the execution sets one by one, set up above dictionaries for each
        for i in range(len(self.ordered_execution_sets)):
            for j in range(len(self.ordered_execution_sets[i])):
                
                # get current node
                node = self.ordered_execution_sets[i][j]
                
                # create feedforward information list for node
                node_feedforward_info = []
                
                # set up node's tensor, activation function, afferent inputs information
                layer = None
                activation_function = self.activation_function_creator(node)
                afferents = {}
                
                # add tensor, activation function to node's feedforward information
                node_feedforward_info.append(layer)
                node_feedforward_info.append(activation_function)
                
                # add to afferent inputs information if we don't have origin node
                if len(node.parents) > 0:
                    
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
                
                # add afferent inputs information to node's feedforward information
                node_feedforward_info.append(afferents)
                
                # add feedforward information to dictionary
                self.node_to_feedforward_info[node] = node_feedforward_info
    
    
    # feedforward method
    def forward(self, inputs):
        
        # set up output list
        outputs = []
        
        # iterate over nodes in execution sets
        for i in range(len(self.ordered_execution_sets)):
            for j in range(len(self.ordered_execution_sets[i])):
                
                # get current node, feedforward information for it
                node = self.ordered_execution_sets[i][j]
                activation_function = self.node_to_feedforward_info[node][1]
                afferents = self.node_to_feedforward_info[node][2]
                
                # feedforward step if we have origin node
                if (i == 0):
                    layer = activation_function(inputs[j])
                
                # feedforward step if we do not have origin node
                else:
                    layer = torch.zeros(len(node.component.input_states[0].variable))
                    for input_node, param_list in afferents.items():
                        layer += (torch.matmul(self.node_to_feedforward_info[input_node][0], param_list[0]) + param_list[1])
                    layer = activation_function(layer)
                
                # put layer in correct place in the dictionary
                self.node_to_feedforward_info[node][0] = layer
                
                # if we're at a node in the last execution set
                if i == len(self.ordered_execution_sets)-1:
                    outputs.append(layer)
        
        return outputs
    
    
    
    # method for retreiving the weights, biases corresponding to psyneulink projections
    def get_parameters_for_projections(self):
        return self.projections_to_torch_weights















# takes inputs and targets for pytorch model and trains them 
def autodiff_training(model, inputs, targets, learning_rate, epochs_or_stop_learning_condition=None):
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    outputs = np.empty([len(inputs)], dtype=object)
    
    # iterate over epochs
    for epoch in range(epochs_or_stop_learning_condition):
        
        if (epoch % 10 == 0):
            print(epoch)
            print("\n")
        
        # set a random number seed
        torch.manual_seed(epoch)
        
        # get a random permutation of inputs/targets
        rand_train_order = np.random.permutation(len(inputs))
        
        # iterate over inputs/targets
        for t in range(len(inputs)):
            
            # print(t)
            # print("\n")
            
            # get current inputs, targets
            curr_tensor_inputs = inputs[rand_train_order[t]]
            curr_tensor_targets = targets[rand_train_order[t]]
            
            # run the model on inputs
            curr_tensor_outputs = model.forward(curr_tensor_inputs)
            
            # compute loss
            loss = torch.zeros(1).float()
            for i in range(len(curr_tensor_outputs)):
                loss += criterion(curr_tensor_outputs[i], curr_tensor_targets[i])
            
            
            # dot = make_dot(loss)
            # dot.format = 'svg'
            # dot.render()
            
            
            # if (t == 1):
                # dot = make_dot(loss)
                # dot.format = 'svg'
                # dot.render()
            
            
            # compute gradients and perform parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # save outputs of model if this is final epoch
            if epoch == epochs_or_stop_learning_condition - 1:
                curr_output_list = []
                for i in range(len(curr_tensor_outputs)):
                    curr_output_list.append(curr_tensor_outputs[i])
                outputs[rand_train_order[t]] = curr_output_list
    
    
    outputs_list = []
    for i in range(len(outputs)):
        outputs_list.append(outputs[i])
    
    return outputs_list















# takes inputs and targets for pytorch model and checks the model outputs against the targets
def autodiff_checking(model, inputs, targets):
    
    with torch.no_grad():
    
        # keep track of stuff
        num_correct_rounded = 0 
        total_MSE = torch.zeros(1).float()
        criterion = nn.MSELoss()
    
        # iterate over inputs
        for t in range(len(inputs)):
            
            # get current inputs, targets
            curr_inputs = inputs[t]
            curr_targets = targets[t]
            
            # get model outputs for current inputs
            curr_outputs = model.forward(curr_inputs)
            
            # compute MSE loss on current output
            for i in range(len(curr_outputs)):
                total_MSE += criterion(curr_outputs[i], curr_targets[i])
                print("Rounded model output: ")
                print(torch.round(curr_outputs[i]))
                print("Actual value: ")
                print(curr_targets[i])
                print("\n")
            
            # iterate over output entries to get correctness of rounded output
            correct_or_nah = 0
            for i in range(len(curr_outputs)):
                for j in range(len(curr_outputs[i])):
                    if (torch.round(curr_outputs[i][j]) != curr_targets[i][j]):
                        correct_or_nah = 1
                        break
            if (correct_or_nah == 0):
                num_correct_rounded += 1
        
        # compute average MSE across all inputs
        avg_MSE = (total_MSE / len(inputs)).numpy()[0]
        
        # compute percentage accuracy based on rounded output
        rounded_output_accuracy = (num_correct_rounded / len(inputs)) * 100
        
        # compute percentage error based on rounded output
        rounded_output_error = 100 - rounded_output_accuracy
        
        return rounded_output_accuracy, rounded_output_error, avg_MSE
    
        
            
        
    
    
    






'''
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
    print("\n")

# Create Pytorch model by parsing the processing graph, exec sets 

rumel_parsed_pytorch = ModelInPytorch(rumel_processing_graph)

print("\n")
print("Checking the parameters of the pytorch object representing the semantic model: ")
print("\n")
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


print("Tryna run this bitch: ")
print("\n")
print("\n")
print("\n")

ready_inputs = []
ready_targets = []

for i in range(len(PT_nouns)):
    for j in range(len(PT_rels)):
        
        ready_inputs.append([PT_rels[j], PT_nouns[i]])
        ready_targets.append([PT_truth_nouns[i], PT_truth_is[i], PT_truth_has[i], PT_truth_can[i]])

test_training = autodiff_training(rumel_parsed_pytorch, ready_inputs, ready_targets, learning_rate=0.001, epochs_or_stop_learning_condition=1000)
# print(test_training)

percentage_acc, percentage_err, avg_MSE = autodiff_checking(rumel_parsed_pytorch, ready_inputs, ready_targets)

print(percentage_acc)
print(percentage_err)
print(avg_MSE)
'''











# AND GATE

print("\n")
print("\n")
print("\n")

# mechanisms and projection
and_input = TransferMechanism(name = 'AND gate input',
                              default_variable = np.zeros(4)
                              )

and_output = TransferMechanism(name = 'AND gate output',
                               default_variable = np.zeros(1)
                               )

and_map = MappingProjection(matrix = np.zeros((4,1)),
                            name = 'AND map',
                            sender = and_input,
                            receiver = and_output
                            )

# create inputs and outputs for model
PT_and_ins = []
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                input_val = torch.tensor([i, j, k, l]).float()
                PT_and_ins.append([input_val])




print(PT_and_ins)
print("\n")

PT_and_outs = []
for i in range(16):
    if (i < 15): PT_and_outs.append([torch.tensor([0]).float()])
    else: PT_and_outs.append([torch.tensor([1]).float()])

print(PT_and_outs)
print("\n")

# create graph
and_graph = Graph()
and_graph.add_component(and_input)
and_graph.add_component(and_output)
and_graph.connect_components(and_input, and_output)

# test graph
print("Testing graph for and gate: ")
print("\n")
for i in range(len(and_graph.vertices)):
    print(and_graph.vertices[i])
    print("\n")

# create model
and_parsed_pytorch = ModelInPytorch(and_graph)

# test model parameters
print("Testing parameters for and gate: ")
print("\n")
print(and_parsed_pytorch)
print("\n")

# test exec list
print("Testing execution sets: ")
for i in range(len(and_parsed_pytorch.ordered_execution_sets)):
    print(and_parsed_pytorch.ordered_execution_sets[i])
    print("\n")

# test model.forward
test_and = and_parsed_pytorch.forward(PT_and_ins[0])
for i in range(len(test_and)):
    print(test_and[i])
    print("\n")

# test training the model
test_and_training = autodiff_training(and_parsed_pytorch, PT_and_ins, PT_and_outs, learning_rate=0.01, epochs_or_stop_learning_condition=5000)
for i in range(len(test_and_training)):
    print(test_and_training[i])
    print("\n")

print("\n")
print("\n")
# test testing the model
percentage_acc, percentage_err, avg_MSE = autodiff_checking(and_parsed_pytorch, PT_and_ins, PT_and_outs)

print(percentage_acc)
print(percentage_err)
print(avg_MSE)



























