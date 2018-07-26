
# imports

from psyneulink.compositions.composition import Composition
from psyneulink.compositions.composition import CompositionError
from psyneulink.compositions.composition import RunError
from psyneulink.compositions.composition import CNodeRole
from psyneulink.compositions.pytorchcreator import PytorchCreator
from psyneulink.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.functions.function import Linear, Logistic, ReLU
from psyneulink.components.states.outputstate import OutputState
from psyneulink.components.functions.function import InterfaceStateMap
from psyneulink.components.states.inputstate import InputState
from psyneulink.globals.keywords import OWNER_VALUE
from psyneulink.scheduling.time import TimeScale

import numpy as np
from toposort import toposort
from collections import Iterable

import timeit as timeit

import torch
from torch import nn
import torch.optim as optim
from torchviz import make_dot

import logging
logger = logging.getLogger(__name__)





__all__ = [
    'ParsingAutodiffComposition', 'ParsingAutodiffCompositionError'
]





class ParsingAutodiffCompositionError(CompositionError):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)





class ParsingAutodiffComposition(Composition):
    
    # setup added for target CIM and model, output reporting switched off by default for CIM's
    def __init__(self):
        
        super(ParsingAutodiffComposition, self).__init__()
        
        self.target_CIM = CompositionInterfaceMechanism(name=self.name + " Target_CIM",
                                                        composition=self)
        self.target_CIM_states = {}
        
        self.input_CIM.reportOutputPref = False
        self.output_CIM.reportOutputPref = False
        self.target_CIM.reportOutputPref = False
        
        self.model = None
    
    
    
    # overriden to create target CIM as well
    def _create_CIM_states(self):
        
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
                    # print("hello somebody")
                    # print("\n")
        
        
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
        current_terminal_input_states = set()
        
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
                    # print("Hello nobody")
                    # print("\n")
                    
            for input_state in node.input_states:
                current_terminal_input_states.add(input_state)
                
                # if there is not a corresponding CIM output state, add one
                if input_state not in set(self.target_CIM_states.keys()):
                    
                    interface_input_state = InputState(owner=self.target_CIM,
                                                       variable=input_state.value,
                                                       reference_value=input_state.value,
                                                       name="TARGET_CIM_" + node.name + "_" + input_state.name)
                    
                    interface_output_state = OutputState(
                            owner=self.target_CIM,
                            variable=OWNER_VALUE,
                            function=InterfaceStateMap(corresponding_input_state=interface_input_state,
                                                       default_variable=self.target_CIM.value),
                            reference_value=input_state.value,
                            name="TARGET_CIM_" + node.name + "_" + input_state.name)
                    
                    self.target_CIM_states[input_state] = [interface_input_state, interface_output_state]
                    
        
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
        for target_state in previous_terminal_target_states.difference(current_terminal_input_states):
            
            # remove the CIM input and output states associated with this Terminal Node output state
            self.target_CIM.input_states.remove(self.target_CIM_states[target_state][0])
            self.target_CIM.output_states.remove(self.target_CIM_states[target_state][1])
            
            # and from the dictionary of CIM output state/input state pairs
            del self.target_CIM_states[target_state]
        
        
        # set CIM's as connected to composition, remove their default input/output states
        if not self.input_CIM.connected_to_composition:
            self.input_CIM.input_states.remove(self.input_CIM.input_state)
            self.input_CIM.output_states.remove(self.input_CIM.output_state)
            self.input_CIM.connected_to_composition = True
        
        if not self.output_CIM.connected_to_composition:
            self.output_CIM.input_states.remove(self.output_CIM.input_state)
            self.output_CIM.output_states.remove(self.output_CIM.output_state)
            self.output_CIM.connected_to_composition = True
        
        if not self.target_CIM.connected_to_composition:
            self.target_CIM.input_states.remove(self.target_CIM.input_state)
            self.target_CIM.output_states.remove(self.target_CIM.output_state)
            self.target_CIM.connected_to_composition = True
    
    
    
    # similar method to _assign_values_to_input_CIM - however, this 
    # assigns inputs to input & target CIM of parsing autodiff composition,
    # executes them, and puts them in appropriate form for underlying pytorch model
    def _throw_through_input_CIM(self, stimuli, inputs_or_targets):
        '''
        print("\n")
        print("\n")
        print("\n")
        print("Printing the list of values going into CIM in \"throw through input CIM's\" before mod'ing: ")
        print("\n")
        print(stimuli)
        print("\n")
        print("\n")
        print("\n")
        '''
        # get execution sets
        exec_sets = self.model.get_ordered_execution_sets()
        
        # set some variables based on whether we have inputs or targets
        if inputs_or_targets == 'inputs':
            CIM = self.input_CIM
            states = self.input_CIM_states
            order = exec_sets[0]
        else:
            CIM = self.target_CIM
            states = self.target_CIM_states
            order = exec_sets[len(exec_sets)-1]
        
        # set up list that will hold inputs for CIM
        CIM_list = []
        
        # add inputs to CIM list
        for input_state in CIM.input_states:
            
            for key in states:
                if states[key][0] == input_state:
                    node_state = key
                    node = key.owner
                    index = node.input_states.index(node_state)
                    
                    if node in stimuli:
                        value = stimuli[node][index]
                    
                    else:
                        value = node.instance_defaults.variable[index]
            
            CIM_list.append(value)
        '''
        print("\n")
        print("\n")
        print("\n")
        print("Printing the list of values going into CIM in \"throw through input CIM's\": ")
        print("\n")
        print(CIM_list)
        print("\n")
        print("\n")
        print("\n")
        '''
        # execute CIM with CIM list
        CIM.execute(CIM_list)
        '''
        # check CIM's
        print("\n")
        print("\n")
        print("\n")
        print("After executing input/target CIM in \"throw through input CIM's \": ")
        print("\n")
        print("For the input CIM: ")
        print("\n")
        print(self.input_CIM.variable)
        print(self.input_CIM.input_states)
        print(self.input_CIM.value)
        print(self.input_CIM.output_states)
        print(self.input_CIM.output_values)
        print("\n")
        print("For the output CIM: ")
        print("\n")
        print(self.output_CIM.variable)
        print(self.output_CIM.input_states)
        print(self.output_CIM.value)
        print(self.output_CIM.output_states)
        print(self.output_CIM.output_values)
        print("\n")
        print("For the target CIM: ")
        print("\n")
        print(self.target_CIM.variable)
        print(self.target_CIM.input_states)
        print(self.target_CIM.value)
        print(self.target_CIM.output_states)
        print(self.target_CIM.output_values)
        print("\n")
        print("\n")
        print("\n")
        '''
        # set up list that will hold inputs for pytorch
        pytorch_list = []
        
        # iterate over nodes in pytorch's desired order, add corresponding inputs from CIM 
        # output to pytorch list in that order 
        for i in range(len(order)):
            
            # get output state corresponding to ith node in pytorch's desired order, add
            # the value of the output state to pytorch list at position i 
            node = order[i]
            value = states[node.component.input_states[0]][1].value
            pytorch_list.append(torch.from_numpy(np.asarray(value).copy()).float())
        
        return pytorch_list
    
    
    
    def _throw_through_output_CIM(self, outputs):
        '''
        print("\n")
        print("\n")
        print("\n")
        print("Printing the list of outputs going into CIM in throw through output CIM before mod'ing: ")
        print("\n")
        print(outputs)
        print("\n")
        print("\n")
        print("\n")
        '''
        # get order
        exec_sets = self.model.get_ordered_execution_sets()
        order = exec_sets[len(exec_sets)-1]
        
        # set up arry that will hold inputs for output CIM
        output_CIM_list = []
        
        # iterate over CIM input states - for each CIM input state, find mechanism in final execution set
        # whose only output state maps to the CIM input state, add pytorch output for this mechanism
        # to output CIM list
        for input_state in self.output_CIM.input_states:
            
            for i in range(len(order)):
                node = order[i]
                if self.output_CIM_states[node.component.output_states[0]][0] == input_state:
                    value = outputs[i]
            
            output_CIM_list.append(value)
        
        '''
        print("\n")
        print("\n")
        print("\n")
        print("Printing the list of values going into output CIM in throw through output CIM: ")
        print("\n")
        print(output_CIM_list)
        print("\n")
        print("\n")
        print("\n")
        '''
        # execute output CIM
        self.output_CIM.execute(output_CIM_list)
        '''
        # check CIM's
        print("\n")
        print("\n")
        print("\n")
        print("After executing output CIM in throw through output CIM: ")
        print("\n")
        print("For the input CIM: ")
        print("\n")
        print(self.input_CIM.variable)
        print(self.input_CIM.input_states)
        print(self.input_CIM.value)
        print(self.input_CIM.output_states)
        print(self.input_CIM.output_values)
        print("\n")
        print("For the output CIM: ")
        print("\n")
        print(self.output_CIM.variable)
        print(self.output_CIM.input_states)
        print(self.output_CIM.value)
        print(self.output_CIM.output_states)
        print(self.output_CIM.output_values)
        print("\n")
        print("For the target CIM: ")
        print("\n")
        print(self.target_CIM.variable)
        print(self.target_CIM.input_states)
        print(self.target_CIM.value)
        print(self.target_CIM.output_states)
        print(self.target_CIM.output_values)
        print("\n")
        print("\n")
        print("\n")
        '''
        # collect CIM output, return it
        output_values = []
        for i in range(len(self.output_CIM.output_states)):
            output_values.append(self.output_CIM.output_states[i].value)
        
        return output_values
    
    
    
    # overriden to provide execution id for target CIM
    def _assign_execution_ids(self, execution_id=None):
        '''
            assigns the same uuid to each Node in the composition's processing graph as well as the CIMs. The uuid is
            either specified in the user's call to run(), or generated randomly at run time.
        '''
        
        # Traverse processing graph and assign one uuid to all of its nodes
        if execution_id is None:
            execution_id = self._get_unique_id()
        
        if execution_id not in self.execution_ids:
            self.execution_ids.append(execution_id)
        
        for v in self._graph_processing.vertices:
            v.component._execution_id = execution_id
        
        self.input_CIM._execution_id = execution_id
        self.output_CIM._execution_id = execution_id
        self.target_CIM._execution_id = execution_id
        
        self._execution_id = execution_id
        return execution_id
    
    
    
    # NOTES:
    # got rid of machinery like user defined "hook" functions, learning scheduler, for now
    # nested functionality not currently supported
    def execute(
        self,
        inputs=None,
        targets=None,
        epochs=None,
        scheduler_processing=None,
        execution_id=None,
    ):
        
        # set up execution id, processing scheduler
        execution_id = self._assign_execution_ids(execution_id)
        
        if scheduler_processing is None:
            scheduler_processing = self.scheduler_processing
        
        
        # if step-by-step processing
        if targets is None:
            
            # push inputs through CIM, get them in correct form for pytorch
            autodiff_inputs = self._throw_through_input_CIM(inputs, 'inputs')
            
            # call processing function
            outputs = self.autodiff_processing(autodiff_inputs)
            
            # get outputs in correct form for output CIM, push them through output CIM
            output_values = self._throw_through_output_CIM(outputs)
            
            return output_values
        
        
        # if batch learning
        else:
            
            # create empty arrays to hold inputs/targets in correct form for pytorch
            autodiff_inputs = []
            autodiff_targets = []
            
            # iterate over trial sets
            for i in range(len(next(iter(inputs.values())))):
                
                # create input/target dictionary for current trial set
                input_stimuli = {}
                for node in inputs:
                    input_stimuli[node] = inputs[node][i]
                target_stimuli = {}
                for node in targets:
                    target_stimuli[node] = targets[node][i]
                
                # send inputs/targets through CIM's, pick them up on other side
                autodiff_inputs.append(self._throw_through_input_CIM(input_stimuli, 'inputs'))
                autodiff_targets.append(self._throw_through_input_CIM(target_stimuli, 'targets'))
                
            # call learning function
            outputs = self.autodiff_training(autodiff_inputs, autodiff_targets, epochs)
            
            # get outputs in correct form for output CIM, push them through output CIM
            output_values = []
            for i in range(len(outputs)):
                output_values.append(self._throw_through_output_CIM(outputs[i]))
            '''
            print("\n")
            print("\n")
            print("Printing output values at the end of execute: ")
            print("\n")
            print(output_values)
            print("\n")
            print("\n")
            '''
            return output_values
    
    
    
    # NOTES:
    # got rid of run machinery like user defined "hook" functions, learning scheduler, for now
    def run(
        self,
        inputs=None,
        targets=None,
        epochs=None,
        scheduler_processing=None,
        execution_id=None,
    ):
        '''
        print("\n")
        print("\n")
        print("\n")
        print("Printing the inputs as they come in: ")
        print("\n")
        print(inputs)
        print("\n")
        print("\n")
        print("\n")
        print("Printing the targets as they come in: ")
        print("\n")
        print(targets)
        print("\n")
        print("\n")
        print("\n")
        '''
        # set up the pytorch model, if not set up yet
        if self.model is None:
            self.model = PytorchCreator(self.graph_processing)
        
        # make sure the presence/absence of targets and number of training epochs are consistent
        if targets is None:
            if epochs is not None:
                raise ParsingAutodiffCompositionError("Number of training epochs specified for Parsing Autodiff "
                                                      "Composition \"{0}\" but no targets."
                                                      .format(self.name))
        else:
            if epochs is None:
                raise ParsingAutodiffCompositionError("Targets specified for Parsing Autodiff Composition \"{0}\" "
                                                      "but no number of training epochs."
                                                      .format(self.name))
            
            if len(self.model.get_weights_for_projections()) == 0:
                raise ParsingAutodiffCompositionError("Targets specified for training Parsing Autodiff Composition \"{0}\" "
                                                      "but Composition has no trainable parameters."
                                                      .format(self.name))
        
        # set up processing scheduler
        if scheduler_processing is None:
            scheduler_processing = self.scheduler_processing
        
        # validate properties of parsing autodiff composition, get node roles, set up CIM's
        self._validate_params()
        self._analyze_graph()
        '''
        # check CIM's
        print("\n")
        print("\n")
        print("\n")
        print("After setting up CIM's: ")
        print("\n")
        print("For the input CIM: ")
        print("\n")
        print(self.input_CIM.variable)
        print(self.input_CIM.input_states)
        print(self.input_CIM.value)
        print(self.input_CIM.output_states)
        print(self.input_CIM.output_values)
        print("\n")
        print("For the output CIM: ")
        print("\n")
        print(self.output_CIM.variable)
        print(self.output_CIM.input_states)
        print(self.output_CIM.value)
        print(self.output_CIM.output_states)
        print(self.output_CIM.output_values)
        print("\n")
        print("For the target CIM: ")
        print("\n")
        print(self.target_CIM.variable)
        print(self.target_CIM.input_states)
        print(self.output_CIM.value)
        print(self.target_CIM.output_states)
        print(self.target_CIM.output_values)
        print("\n")
        print("\n")
        print("\n")
        '''
        # get execution id, do some stuff with processing scheduler
        execution_id = self._assign_execution_ids(execution_id)
        scheduler_processing._init_counts(execution_id=execution_id)
        
        # if there is only one origin mechanism, allow inputs to be specified in a list
        origin_nodes = self.get_c_nodes_by_role(CNodeRole.ORIGIN)
        if isinstance(inputs, (list, np.ndarray)):
            if len(origin_nodes) == 1:
                inputs = {next(iter(origin_nodes)): inputs}
            else:
                raise CompositionError("Inputs to {} must be specified in a dictionary with a key for each of "
                                       "its {} origin nodes."
                                       .format(self.name, len(origin_nodes)))
        elif not isinstance(inputs, dict):
            if len(origin_nodes) == 1:
                raise CompositionError("Inputs to {} must be specified in a list or in a dictionary with the "
                                       "origin mechanism({}) as its only key"
                                       .format(self.name, next(iter(origin_nodes)).name))
            else:
                raise CompositionError("Inputs to {} must be specified in a dictionary with a key for each of "
                                       "its {} origin nodes."
                                       .format(self.name, len(origin_nodes)))
        
        # validate inputs, get adjusted inputs, number of input trial sets
        inputs, num_input_sets = self._adjust_stimulus_dict(inputs, 'inputs')
        
        # reset counts on processing scheduler
        scheduler_processing._reset_counts_total(TimeScale.RUN, execution_id)
        
        
        # if we're just doing step-by-step processing
        if targets is None:
            
            # --- RESET FOR NEXT TRIAL ---
            for trial_num in range(num_input_sets):
                
                # PROCESSING ------------------------------------------------------------------------
                
                # prepare stimuli
                execution_stimuli = {}
                for node in inputs:
                    execution_stimuli[node] = inputs[node][trial_num]
                
                # call execute method
                trial_output = self.execute(inputs=execution_stimuli,
                                            scheduler_processing=scheduler_processing,
                                            execution_id=execution_id)
                
                # ---------------------------------------------------------------------------------
                
                # store the result of this execute in case it will be the final result
                if isinstance(trial_output, Iterable):
                    result_copy = trial_output.copy()
                else:
                    result_copy = trial_output
                self.results.append(result_copy)
        
        
        # if we're doing batch learning
        else: 
            
            # if there is only one terminal mechanism, allow targets to be specified in a list
            terminal_nodes = self.get_c_nodes_by_role(CNodeRole.ORIGIN)
            if isinstance(targets, (list, np.ndarray)):
                if len(terminal_nodes) == 1:
                    targets = {next(iter(terminal_nodes)): targets}
                else:
                    raise CompositionError("Targets to {} must be specified in a dictionary with a key for each "
                                           "of its {} terminal nodes."
                                           .format(self.name, len(terminal_nodes)))
            elif not isinstance(targets, dict):
                if len(terminal_nodes) == 1:
                    raise CompositionError("Targets to {} must be specified in a list or in a dictionary with "
                                           "the terminal mechanism({}) as its only key"
                                           .format(self.name, next(iter(terminal_nodes)).name))
                else:
                    raise CompositionError("Targets to {} must be specified in a dictionary with a key for each "
                                           "of its {} terminal nodes."
                                           .format(self.name, len(terminal_nodes)))
        
            # validate targets, get adjusted targets, number of target trial sets
            targets, num_target_sets = self._adjust_stimulus_dict(targets, 'targets')
            
            # check that number of target trial sets and number of input trial sets are the same
            if num_input_sets != num_target_sets:
                raise ParsingAutodiffCompositionError("Number of input trial sets ({}) provided and number of "
                                                      "target trial sets ({}) provided are different."
                                               .format(num_input_sets, num_target_sets))
            '''
            print("\n")
            print("\n")
            print("\n")
            print("Printing the inputs just after adjusting: ")
            print("\n")
            print(inputs)
            print("\n")
            print("\n")
            print("\n")
            print("Printing the targets just after adjusting: ")
            print("\n")
            print(targets)
            print("\n")
            print("\n")
            print("\n")
            '''
            
            # LEARNING --------------------------------------------------------------------------
            
            # call execute method
            trial_output = self.execute(inputs=inputs,
                                        targets=targets,
                                        epochs=epochs,
                                        scheduler_processing=scheduler_processing,
                                        execution_id=execution_id)
            
            # ---------------------------------------------------------------------------------
            
            # store the result of this execute
            if isinstance(trial_output, Iterable):
                result_copy = trial_output.copy()
            else:
                result_copy = trial_output
            self.results.append(result_copy)
        
        
        # increment clock, return result
        scheduler_processing.clocks[execution_id]._increment_time(TimeScale.RUN)
        return self.results
    
    
    
    # performs feedforward step for one input
    def autodiff_processing(self, inputs):
        
        # run the model on inputs - switch autograd off for this (we don't need it)
        with torch.no_grad():
            tensor_outputs = self.model.forward(inputs)
        
        # get outputs back into numpy
        outputs = []
        for i in range(len(tensor_outputs)):
            outputs.append(tensor_outputs[i].numpy().copy())
        
        return outputs
    
    
    
    # uses inputs and targets to train model for given number of epochs
    def autodiff_training(self, inputs, targets, epochs):
        
        learning_rate = 0.001
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        outputs = []
        rand_train_order_reverse = np.zeros(len(inputs))
        '''
        print("\n")
        print("\n")
        print("Inputs as they come into autodiff_training: ")
        print("\n")
        print(inputs)
        print("\n")
        print("Targets as they come into autodiff_training: ")
        print("\n")
        print(targets)
        print("\n")
        print("\n")
        '''
        # iterate over epochs
        for epoch in range(epochs):
            
            '''
            print("Epoch number: ", epoch)
            '''
            
            # set a random number seed
            torch.manual_seed(epoch)
            
            # get a random permutation of input/target trial sets
            rand_train_order = np.random.permutation(len(inputs))
            
            # if we're on final epoch, get mapping back to original order of input/target trial sets
            if epoch == epochs-1:
                rand_train_order_reverse[rand_train_order] = np.arange(len(inputs))
            
            outputs = []
            
            # iterate over trial sets
            for t in range(len(inputs)):
                
                # get current inputs, targets
                curr_tensor_inputs = inputs[rand_train_order[t]]
                curr_tensor_targets = targets[rand_train_order[t]]
                
                # run the model on inputs
                curr_tensor_outputs = self.model.forward(curr_tensor_inputs)
                
                # compute loss
                loss = torch.zeros(1).float()
                for i in range(len(curr_tensor_outputs)):
                    loss += criterion(curr_tensor_outputs[i], curr_tensor_targets[i])
                
                
                '''
                if (epoch == 5 and t == 12):
                    dot = make_dot(loss)
                    dot.format = 'svg'
                    dot.render()
                '''
                
                '''
                if ((epoch%5 == 0 or epoch == epochs-1) and t == len(inputs)-1):
                    print("\n")
                    print("Loss on this epoch: ")
                    print("\n")
                    print(loss)
                    print("\n")
                '''
                
                # compute gradients and perform parameter update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # save outputs of model if this is final epoch
                if epoch%5 == 0 or epoch == epochs-1:
                    curr_output_list = []
                    for i in range(len(curr_tensor_outputs)):
                        curr_output_list.append(curr_tensor_outputs[i].detach().numpy().copy())
                    outputs.append(curr_output_list)
            
            '''
            if (epoch%5 == 0 or epoch == epochs-1):
                num_correct = 0
                for i in range(len(outputs)):
                    correct_or_nah = 0
                    for j in range(len(outputs[i])):
                        # print("Where we are: (", i, ", ", j, ")")
                        # print("Correct answer: ")
                        # print(targets[int(rand_train_order[i])][j])
                        # print("Predicted answer: ")
                        # print(np.round(outputs[i][j]))
                        # print("\n")
                        
                        # if (np.round(outputs[i][j]) != targets[int(rand_train_order[i])][j]):
                            # correct_or_nah = 1
                            # break
                    # if (correct_or_nah == 0):
                        # num_correct += 1
                        
                        for k in range(len(outputs[i][j])):
                            if (np.round(outputs[i][j][k]) != targets[int(rand_train_order[i])][j][k]):
                                correct_or_nah = 1
                                break
                        if (correct_or_nah == 1):
                            break
                    if (correct_or_nah == 0):
                        num_correct += 1
                
                print("Number correct on this epoch: ")
                print(num_correct)
                print("\n")
                print("\n")
            '''
        
        
        # save outputs in a list in correct order, return this list
        outputs_list = []
        for i in range(len(outputs)):
            outputs_list.append(outputs[int(rand_train_order_reverse[i])])
        '''
        print("\n")
        print("\n")
        print("Outputs after being collected at the end of autodiff training: ")
        print("\n")
        print(outputs_list)
        print("\n")
        print("\n")
        '''
        return outputs_list
    
    
    
    # method for giving user pytorch weights and biases
    def get_parameters(self):
        
        if self.model is None:
            self.model = PytorchCreator(self.graph_processing)
        
        weights = self.model.get_weights_for_projections()
        biases = self.model.get_biases_for_mechanisms()
        
        return weights, biases
    
    
    
    # overriden to permit only homogenous inputs - parsing autodiff compositions
    # cannot have mechanisms with multiple input states of different lengths
    def _input_matches_variable(self, input_value, var):
        
        # var already guaranteed to only have 1 row/input state by validate params method
        if np.shape(np.atleast_2d(input_value)) == np.shape(var):
            return "homogeneous"
        
        return False
    
    
    
    # overriden to adjust dictionary for inputs or targets - the adjusting is exactly the same though.
    # Note: some comments removed from the function that are present in composition.py
    def _adjust_stimulus_dict(self, stimuli, inputs_or_targets):
        
        
        # check if we're dealing with inputs or targets, set variables accordingly
        if inputs_or_targets == 'inputs':
            nodes = self.get_c_nodes_by_role(CNodeRole.ORIGIN)
        else:
            nodes = self.get_c_nodes_by_role(CNodeRole.TERMINAL)
        
        
        # STEP 1: Validate that there is a one-to-one mapping of input entries to origin nodes
        
        # Check that all of the nodes listed in the stimuli dict are ORIGIN/TERMINAL nodes in self
        for node in stimuli.keys():
            if not node in nodes:
                if inputs_or_targets == 'inputs':
                    raise CompositionError("{} in inputs dict for {} is not one of its ORIGIN nodes".
                                           format(node.name, self.name))
                else:
                    raise CompositionError("{} in inputs dict for {} is not one of its ORIGIN nodes".
                                           format(node.name, self.name))
        
        # Check that all of the ORIGIN/TERMINAL nodes are represented - if not, use default_variable
        for node in nodes:
            if not node in stimuli:
                stimuli[node] = node.default_external_input_values
        
        
        # STEP 2: Loop over all dictionary entries to validate their content and adjust any convenience notations:
        
        adjusted_stimuli = {}
        num_sets = -1
        
        for node, stim_list in stimuli.items():
            
            input_must_match = node.external_input_values
            
            # check if we have 1 trial's worth of correct inputs
            check_spec_type = self._input_matches_variable(stim_list, input_must_match)
            if check_spec_type == "homogeneous":
                adjusted_stimuli[node] = [np.atleast_2d(stim_list)]
                
                # verify that all nodes have provided the same number of inputs
                if num_sets == -1:
                    num_sets = 1
                elif num_sets != 1:
                    raise RunError("Input specification for {} is not valid. The number of inputs (1) provided for {}"
                                   "conflicts with at least one other node's input specification.".format(self.name,
                                                                                                               node.name))
            
            else:
                adjusted_stimuli[node] = []
                for stim in stimuli[node]:
                    
                    # check if we have 1 trial's worth of correct inputs
                    check_spec_type = self._input_matches_variable(stim, input_must_match)
                    if check_spec_type == False:
                        err_msg = "Input stimulus ({}) for {} is incompatible with its external_input_values ({}).".\
                            format(stim, node.name, input_must_match)
                        if "KWTA" in str(type(node)):
                            err_msg = err_msg + " For KWTA mechanisms, remember to append an array of zeros (or other values)" \
                                                " to represent the outside stimulus for the inhibition input state, and " \
                                                "for systems, put your inputs"
                        raise RunError(err_msg)
                    else:
                        adjusted_stimuli[node].append(np.atleast_2d(stim))
                
                # verify that all nodes have provided the same number of inputs
                if num_sets == -1:
                    num_sets = len(stimuli[node])
                elif num_sets != len(stimuli[node]):
                    raise RunError("Input specification for {} is not valid. The number of inputs ({}) provided for {}"
                                   "conflicts with at least one other node's input specification."
                                   .format(self.name, (stimuli[node]), node.name))
        
        return adjusted_stimuli, num_sets
    
    
    
    # method to validate params of parsing autodiff composition
    def _validate_params(self):
        
        processing_graph = self.graph_processing
        topo_dict = {}
        
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
                                                      "dimensionality of this state's value will become the dimensionality of "
                                                      "the tensor representing the state's mechanism in the underlying "
                                                      "Pytorch model."
                                                      .format(node.component, self.name))
            
            topo_dict[node.component] = set()
            for parent in processing_graph.get_parents_from_component(node.component):
                topo_dict[node.component].add(parent.component)
                try:
                    list(toposort(topo_dict))
                except ValueError:
                    raise ParsingAutodiffCompositionError("Mechanisms {0} and {1} are part of a recurrent path in {2}. "
                                                          "Parsing Autodiff Compositions currently do not support recurrence."
                                                          .format(node.component, parent.component, self.name))




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

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
'''
map_nouns_h1 = MappingProjection(matrix=np.random.rand(8,8),
                                 name="map_nouns_h1",
                                 sender=nouns_in,
                                 receiver=h1
                                 )

map_rels_h2 = MappingProjection(matrix=np.random.rand(3,15),
                                name="map_rels_h2",
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
'''

'''
# DOING SHIT DIRECTLY WITH A COMPOSITION
rumel_composition = ParsingAutodiffComposition()

rumel_composition.add_c_node(nouns_in)
rumel_composition.add_c_node(rels_in)
rumel_composition.add_c_node(h1)
rumel_composition.add_c_node(h2)
rumel_composition.add_c_node(out_sig_I)
rumel_composition.add_c_node(out_sig_is)
rumel_composition.add_c_node(out_sig_has)
rumel_composition.add_c_node(out_sig_can)

rumel_composition.add_projection(nouns_in, map_nouns_h1, h1)
rumel_composition.add_projection(rels_in, map_rels_h2, h2)
rumel_composition.add_projection(h1, map_h1_h2, h2)
rumel_composition.add_projection(h2, map_h2_I, out_sig_I)
rumel_composition.add_projection(h2, map_h2_is, out_sig_is)
rumel_composition.add_projection(h2, map_h2_has, out_sig_has)
rumel_composition.add_projection(h2, map_h2_can, out_sig_can)
'''

# DOING SHIT DIRECTLY WITH A COMPOSITION (AGAIN)
rumel_composition = ParsingAutodiffComposition()

rumel_composition.add_c_node(nouns_in)
rumel_composition.add_c_node(rels_in)
rumel_composition.add_c_node(h1)
rumel_composition.add_c_node(h2)
rumel_composition.add_c_node(out_sig_I)
rumel_composition.add_c_node(out_sig_is)
rumel_composition.add_c_node(out_sig_has)
rumel_composition.add_c_node(out_sig_can)

rumel_composition.add_projection(nouns_in, MappingProjection(sender=nouns_in, receiver=h1), h1)
rumel_composition.add_projection(rels_in, MappingProjection(sender=rels_in, receiver=h2), h2)
rumel_composition.add_projection(h1, MappingProjection(sender=h1, receiver=h2), h2)
rumel_composition.add_projection(h2, MappingProjection(sender=h2, receiver=out_sig_I), out_sig_I)
rumel_composition.add_projection(h2, MappingProjection(sender=h2, receiver=out_sig_is), out_sig_is)
rumel_composition.add_projection(h2, MappingProjection(sender=h2, receiver=out_sig_has), out_sig_has)
rumel_composition.add_projection(h2, MappingProjection(sender=h2, receiver=out_sig_can), out_sig_can)

'''
rumel_composition._update_processing_graph()

rumel_processing_graph = rumel_composition._graph_processing

# test the graph

print("Checking the vertices of the processing graph for the semantic model: ")
print("\n")
for i in range(len(rumel_processing_graph.vertices)):
    vertex = rumel_processing_graph.vertices[i]
    component = vertex.component
    print(vertex)
    print(component)
    print(component.variable)
    print(np.shape(component.variable))
    print(component.value)
    print(np.shape(component.value))
    print(component.input_states)
    print(component.input_states[0])
    print(component.input_states[0].variable)
    print(component.input_states[0].value)
    print(np.shape(component.input_states[0].variable))
    print(np.shape(component.input_states[0].value))
    print("\n")
'''

'''
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

print("\n")
print("Checking the execution sets created by the scheduler: ")
rumel_parsed_sched = Scheduler(graph=rumel_processing_graph)
print(rumel_parsed_sched.consideration_queue)
print("\n")
'''


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

nouns_input = np.identity(len(nouns))
rels_input = np.identity(len(relations))

inputs_dict = {}
inputs_dict[nouns_in] = []
inputs_dict[rels_in] = []

targets_dict = {}
targets_dict[out_sig_I] = []
targets_dict[out_sig_is] = []
targets_dict[out_sig_has] = []
targets_dict[out_sig_can] = []



# Training on all input-output pairs

for i in range(len(nouns)):
    for j in range(len(relations)):
        inputs_dict[nouns_in].append(nouns_input[i])
        inputs_dict[rels_in].append(rels_input[j])
        targets_dict[out_sig_I].append(truth_nouns[i])
        targets_dict[out_sig_is].append(truth_is[i])
        targets_dict[out_sig_has].append(truth_has[i])
        targets_dict[out_sig_can].append(truth_can[i])


'''
# Training on one input-output pair
inputs_dict[nouns_in].append(nouns_input[0])
inputs_dict[rels_in].append(rels_input[0])
targets_dict[out_sig_I].append(truth_nouns[0])
targets_dict[out_sig_is].append(truth_is[0])
targets_dict[out_sig_has].append(truth_has[0])
targets_dict[out_sig_can].append(truth_can[0])
'''
'''
print("\n")
print(inputs_dict[nouns_in])
print(np.shape(inputs_dict[nouns_in]))
print("\n")
print(inputs_dict[rels_in])
print(np.shape(inputs_dict[rels_in]))
print("\n")
'''

'''
# Try running this shit
result = rumel_composition.run(inputs=inputs_dict)

print("\n")
print("\n")
print("\n")
print("\n")
print("Checking result: ")
print("\n")
print(result)
print("\n")
print(len(result))
print("\n")
for i in range(len(result)):
    print(len(result[i]))
    print(result[i])
    print("\n")
    for j in range(len(result[i])):
        print(len(result[i][j]))
        print("\n")
'''

'''
# Try training this shit
start = timeit.default_timer()
result = rumel_composition.run(inputs=inputs_dict, targets=targets_dict, epochs=400)
end = timeit.default_timer()
print(end - start)
# print(result[0][0])
'''

'''
print("\n")
print("\n")
print("Printing the result after everything: ")
print("\n")
print(result)
print("\n")
print(len(result))
'''

'''
result = rumel_composition.run(inputs=inputs_dict, targets=targets_dict, epochs=1)
result = rumel_composition.run(inputs=inputs_dict, targets=targets_dict, epochs=1)
result = rumel_composition.run(inputs=inputs_dict, targets=targets_dict, epochs=1)
result = rumel_composition.run(inputs=inputs_dict, targets=targets_dict, epochs=1)
result = rumel_composition.run(inputs=inputs_dict, targets=targets_dict, epochs=1)
result = rumel_composition.run(inputs=inputs_dict, targets=targets_dict, epochs=1)

print("\n")
print("\n")
print("Printing the result: ")
print("\n")
print(result)
print(len(result))
'''

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


print("Tryna run this thing: ")
print("Running parsing autodiff composition: ")
print("\n")
print("\n")
print("\n")

ready_inputs = []
ready_targets = []

for i in range(len(PT_nouns)):
    for j in range(len(PT_rels)):
        ready_inputs.append([PT_rels[j], PT_nouns[i]])
        ready_targets.append([PT_truth_nouns[i], PT_truth_is[i], PT_truth_has[i], PT_truth_can[i]])

# start_time = timeit.default_timer()

test_training = autodiff_training(rumel_parsed_pytorch, ready_inputs, ready_targets, learning_rate=0.001, epochs_or_stop_learning_condition=1000)

# end_time = timeit.default_timer()
# print(end_time - start_time)

print("\n")

# print(times[:20])
# print(np.shape(times))
# print(np.mean(times))
# print("\n")

# print(test_training)

percentage_acc, percentage_err, avg_MSE = autodiff_checking(rumel_parsed_pytorch, ready_inputs, ready_targets)

print(percentage_acc)
print(percentage_err)
print(avg_MSE)
    
'''
    
    
    
    