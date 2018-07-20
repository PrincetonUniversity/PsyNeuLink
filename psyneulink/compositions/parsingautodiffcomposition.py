
# imports
import collections
from collections import Iterable, OrderedDict
from enum import Enum
import logging
import numpy as np
import uuid

from composition import Composition

from psyneulink.components.component import function_type
from psyneulink.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.shellclasses import Mechanism, Projection
from psyneulink.components.states.outputstate import OutputState
from psyneulink.components.functions.function import InterfaceStateMap
from psyneulink.components.states.inputstate import InputState
from psyneulink.globals.context import ContextFlags
from psyneulink.globals.keywords import OWNER_VALUE, SYSTEM, EXECUTING, HARD_CLAMP, IDENTITY_MATRIX, NO_CLAMP, PULSE_CLAMP, SOFT_CLAMP
from psyneulink.scheduling.condition import Always
from psyneulink.scheduling.scheduler import Scheduler
from psyneulink.scheduling.time import TimeScale

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F 
import torch.optim as optim

logger = logging.getLogger(__name__)







# Parsing Autodiff Composition class
class ParsingAutodiffComposition(Composition):
    
    
    
    # init
    def __init__(self):
        super(ParsingAutodiffComposition, self).__init__()
        
        self.model = None # reference to instance of pytorch nn module subclass object
    
    
    '''
    # ordered execution set creator
    def get_ordered_exec_sets(self, processing_graph):
        
        if (processing_graph is None):
            processing_graph = self.graph_processing
        
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
    
    
    
    # helper recursive method for creating ordered execution sets
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
    '''
    
    
    # method to create pytorch model
    def model_creator():
        self.model = ModelInPytorch(processing_graph)




# Pytorch nn module subclass
class ModelInPytorch(torch.nn.Module):
    
    
    
    # initialization
    def __init__(self, processing_graph, execution_list):
        
        self.node_to_execute = {} # map from PNL node to feedforward information
        self.projections_to_torch_weights = {} # map from PNL projections to corresponding weights in pytorch
        self.torch_weights_to_projections = {} # map from weights in pytorch to corresponding PNL projections
    
    
    
    # feedforward method
    def forward(self):
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    