
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
    






# Pytorch nn module subclass
class ModelInPytorch(torch.nn.Module):
    
    
    
    # initialization
    def __init__(self, processing_graph, execution_list):
        
        self.node_to_execute = {} # map from PNL node to feedforward information
        self.projections_to_torch_weights = {} # map from PNL projections to corresponding weights in pytorch
        self.torch_weights_to_projections = {} # map from weights in pytorch to corresponding PNL projections
    
    
    
    # feedforward method
    def forward(self):
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    