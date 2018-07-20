
# imports

from psyneulink.compositions.composition import Composition
from psyneulink.compositions.composition import CNodeRole
from psyneulink.compositions.pytorchcreator import PytorchCreator
from psyneulink.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.components.functions.function import Linear, Logistic, ReLU
from psyneulink.components.states.outputstate import OutputState
from psyneulink.components.functions.function import InterfaceStateMap
from psyneulink.components.states.inputstate import InputState
from psyneulink.globals.keywords import OWNER_VALUE

from toposort import toposort

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F 
import torch.optim as optim

import logging
logger = logging.getLogger(__name__)




__all__ = [
    'ParsingAutodiffComposition', 'ParsingAutodiffCompositionError'
]




class ParsingAutodiffCompositionError(Exception):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)




class ParsingAutodiffComposition(Composition):
    
    # init
    def __init__(self):
        
        super(ParsingAutodiffComposition, self).__init__()
        
        # set up target CIM
        self.target_CIM = CompositionInterfaceMechanism(name=self.name + " Target_CIM",
                                                        composition=self)
        self.target_CIM_states = {}
        
        # model instance variable
        self.model = None
    
    
    
    def _create_CIM_states(self):
        
        
        if not self.input_CIM.connected_to_composition:
            self.input_CIM.input_states.remove(self.input_CIM.input_state)
            self.input_CIM.output_states.remove(self.input_CIM.output_state)
            self.input_CIM.connected_to_composition = True

        if not self.output_CIM.connected_to_composition:
            self.output_CIM.input_states.remove(self.output_CIM.input_state)
            self.output_CIM.output_states.remove(self.output_CIM.output_state)
            self.output_CIM.connected_to_composition = True
        
        
        #  INPUT CIM
        # loop over all origin nodes

        current_origin_input_states = set()
        
        for node in self.get_c_nodes_by_role(CNodeRole.ORIGIN):

            for input_state in node.external_input_states:
                current_origin_input_states.add(input_state)

                # if there is not a corresponding CIM output state, add one
                if input_state not in set(self.input_CIM_states.keys()):

                    interface_input_state = InputState(owner=self.input_CIM,
                                                       variable=input_state.value,
                                                       reference_value=input_state.value,
                                                       name="INPUT_CIM_" + node.name + "_" + input_state.name)

                    interface_output_state = OutputState(owner=self.input_CIM,
                                                         variable=OWNER_VALUE,
                                                         default_variable=self.input_CIM.variable,
                                                         function=InterfaceStateMap(corresponding_input_state=interface_input_state),
                                                         name="INPUT_CIM_" + node.name + "_" + input_state.name)

                    self.input_CIM_states[input_state] = [interface_input_state, interface_output_state]
        
        
        sends_to_input_states = set(self.input_CIM_states.keys())
        
        # For any states still registered on the CIM that does not map to a corresponding ORIGIN node I.S.:
        for input_state in sends_to_input_states.difference(current_origin_input_states):

            # remove the CIM input and output states associated with this Origin node input state
            self.input_CIM.input_states.remove(self.input_CIM_states[input_state][0])
            self.input_CIM.output_states.remove(self.input_CIM_states[input_state][1])
            
            # and from the dictionary of CIM output state/input state pairs
            del self.input_CIM_states[input_state]
        
        
        # OUTPUT AND TARGET CIM
        # loop over all terminal nodes
        
        current_terminal_output_states = set()
        
        for node in self.get_c_nodes_by_role(CNodeRole.TERMINAL):
            
            for output_state in node.output_states:
                current_terminal_output_states.add(output_state)
                
                # if there is not a corresponding CIM output state, add one
                if output_state not in set(self.output_CIM_states.keys()):
                    
                    interface_input_state = InputState(owner=self.output_CIM,
                                                       variable=output_state.value,
                                                       reference_value=output_state.value,
                                                       name="OUTPUT_CIM_" + node.name + "_" + output_state.name)
                    
                    interface_output_state = OutputState(
                            owner=self.output_CIM,
                            variable=OWNER_VALUE,
                            function=InterfaceStateMap(corresponding_input_state=interface_input_state,
                                                       default_variable=self.output_CIM.value),
                            reference_value=output_state.value,
                            name="OUTPUT_CIM_" + node.name + "_" + output_state.name)
                    
                    self.output_CIM_states[output_state] = [interface_input_state, interface_output_state]
                
                if output_state not in set(self.target_CIM_states.keys()):
                    
                    interface_input_state = InputState(owner=self.target_CIM,
                                                       variable=output_state.value,
                                                       reference_value=output_state.value,
                                                       name="TARGET_CIM_" + node.name + "_" + output_state.name)
                    
                    interface_output_state = OutputState(
                            owner=self.target_CIM,
                            variable=OWNER_VALUE,
                            function=InterfaceStateMap(corresponding_input_state=interface_input_state,
                                                       default_variable=self.target_CIM.value),
                            reference_value=output_state.value,
                            name="TARGET_CIM_" + node.name + "_" + output_state.name)
                    
                    self.target_CIM_states[output_state] = [interface_input_state, interface_output_state]
        
        
        previous_terminal_output_states = set(self.output_CIM_states.keys())
        
        # For any states still registered on the CIM that does not map to a corresponding ORIGIN node I.S.:
        for output_state in previous_terminal_output_states.difference(current_terminal_output_states):
            
            # remove the CIM input and output states associated with this Terminal Node output state
            self.output_CIM.input_states.remove(self.output_CIM_states[output_state][0])
            self.output_CIM.output_states.remove(self.output_CIM_states[output_state][1])
            
            # and from the dictionary of CIM output state/input state pairs
            del self.output_CIM_states[output_state]
        
        previous_terminal_target_states = set(self.target_CIM_states.keys())
        
        # For any states still registered on the CIM that does not map to a corresponding ORIGIN node I.S.:
        for output_state in previous_terminal_target_states.difference(current_terminal_output_states):
            
            # remove the CIM input and output states associated with this Terminal Node output state
            self.target_CIM.input_states.remove(self.target_CIM_states[output_state][0])
            self.target_CIM.output_states.remove(self.target_CIM_states[output_state][1])
            
            # and from the dictionary of CIM output state/input state pairs
            del self.target_CIM_states[output_state]
    
    
    '''
    @property
    def model(self):
        
        processing_graph = self.graph_processing
        
        if self.model is None:
            return PytorchCreator(processing_graph)
        
        else:
            return self.model
    '''
    
    def model(self):
        
        processing_graph = self.graph_processing
        
        if self.model is None:
            return PytorchCreator(processing_graph)
        
        else:
            return self.model
    
    
    
    
    # method to validate params of parsing autodiff composition
    def validate_params(self):
        
        # get processing graph
        processing_graph = self.graph_processing
        
        # iterate over nodes in processing graph
        for node in processing_graph.vertices:
            
            # raise error if node is a composition
            if isinstance(node.component, Composition):
                raise ParsingAutodiffCompositionError("Composition {0} was added as a node to {1}. Compositions cannot be "
                                                      "added as nodes to Parsing Autodiff Compositions."
                                                      .format(node.component, self.name))
            
            # raise error if node's mechanism doesn't have Linear, Logistic, or ReLU functions
            if not isinstance(node.component.function_object, (Linear, Logistic, ReLU)):
                raise ParsingAutodiffCompositionError("Function {0} of mechanism {1} in {2} is not a valid function "
                                                      "for a Parsing Autodiff Composition. Functions of mechanisms in "
                                                      "Parsing Autodiff Compositions can only be Linear, Logistic, or ReLU."
                                                      .format(node.component.function, node.component, self.name))
            
            # raise error if node has more than one input state
            if len(node.component.input_states) > 1:
                raise ParsingAutodiffCompositionError("Mechanism {0} of {1} has more than one input state. Parsing Autodiff "
                                                      "Compositions only allow mechanisms to have one input state. The "
                                                      "dimensionality of this input state will become the dimensionality of "
                                                      "the tensor representing the state's mechanism in the underlying "
                                                      "Pytorch model."
                                                      .format(node.component, self.name))
        
        # raise error if any recurrent paths detected in graph by toposort
        topo_dict = {}
        for node in processing_graph.vertices:
            topo_dict[node.component] = set()
            for parent in processing_graph.get_parents_from_component(node.component):
                topo_dict[node.component].add(parent.component)
                try:
                    list(toposort(topo_dict))
                except ValueError:
                    raise ParsingAutodiffCompositionError("Mechanisms {0} and {1} are part of a recurrent path in {2}. "
                                                          "Parsing Autodiff Compositions currently do not support recurrence. "
                                                          .format(node.component, parent.component, self.name))
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    