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
    
    def __init__(self, param_init_from_pnl=False, name=None):
        
        # set up name
        if (name is None):
            name = "parsing_autodiff_composition"
        self.name = name
        
        super(ParsingAutodiffComposition, self).__init__()
        
        # set up target CIM
        self.target_CIM = CompositionInterfaceMechanism(name=self.name + " Target_CIM",
                                                        composition=self)
        self.target_CIM_states = {}
        
        # default is to switch off output reporting for CIM's
        self.input_CIM.reportOutputPref = False
        self.output_CIM.reportOutputPref = False
        self.target_CIM.reportOutputPref = False
        
        # model and associated parameters
        self.model = None
        self.learning_rate = None
        self.optimizer = None
        self.loss = None
        self.param_init_from_pnl = param_init_from_pnl
        
        # keeps track of average loss per epoch
        self.losses = []
    
    
    
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
        
        # execute CIM with CIM list
        CIM.execute(CIM_list)
        
        # set up list that will hold inputs for pytorch
        pytorch_list = []
        
        # iterate over nodes in pytorch's desired order, add corresponding inputs from CIM 
        # output to pytorch list in that order 
        for i in range(len(order)):
            
            # get output state corresponding to ith node in pytorch's desired order, add
            # the value of the output state to pytorch list at position i 
            node = order[i]
            value = states[node.component.input_states[0]][1].value
            pytorch_list.append(torch.from_numpy(np.asarray(value).copy()).double())
        
        '''
        print("\n")
        print(pytorch_list)
        print("\n")
        '''
        
        return pytorch_list
    
    
    # similar method to _assign_values_to_input_CIM - however, this gets pytorch output from execute,
    # assigns it to output CIM of parsing autodiff composition, executes the CIM, and sends
    # its output in a list back to execute
    def _throw_through_output_CIM(self, outputs):
        
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
        
        # execute output CIM
        self.output_CIM.execute(output_CIM_list)
        
        # collect CIM output, return it
        output_values = []
        for i in range(len(self.output_CIM.output_states)):
            output_values.append(self.output_CIM.output_states[i].value)
        
        return output_values
    
    
    
    # overriden to provide execution id for target CIM
    def _assign_execution_ids(self, execution_id=None):
        
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
        randomize=False,
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
            outputs = self.autodiff_training(autodiff_inputs, autodiff_targets, epochs, randomize=randomize)
            
            # get outputs in correct form for output CIM, push them through output CIM
            output_values = []
            for i in range(len(outputs)):
                output_values.append(self._throw_through_output_CIM(outputs[i]))
            
            return output_values
    
    
    
    # NOTES:
    # got rid of run machinery like user defined "hook" functions, learning scheduler, for now
    def run(
        self,
        inputs=None,
        targets=None,
        epochs=None,
        learning_rate=None,
        optimizer=None,
        loss=None,
        randomize=False,
        refresh_losses=False,
        scheduler_processing=None,
        execution_id=None,
    ):
        
        # set up model/training parameters, check that arguments provided are consistent
        if self.model is None:
            self.model = PytorchCreator(self.graph_processing, self.param_init_from_pnl)
        
        if learning_rate is None:
            if self.learning_rate is None:
                self.learning_rate = 0.001
        else:
            if not isinstance(learning_rate, (int, float)):
                raise ParsingAutodiffCompositionError("Learning rate must be an integer or float value.")
            self.learning_rate = learning_rate
        
        if optimizer is None:
            if self.optimizer is None:
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            if optimizer not in ['sgd', 'adam']:
                raise ParsingAutodiffCompositionError("Invalid optimizer specified. Optimizer argument must be a string. "
                                                      "Currently, Stochastic Gradient Descent and Adam are the only available "
                                                      "optimizers (specified as 'sgd' or 'adam').")
            if optimizer == 'sgd':
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
            else:
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        if loss is None:
            if self.loss is None:
                self.loss = nn.MSELoss(size_average=False)
        else:
            if loss not in ['mse', 'crossentropy']:
                raise ParsingAutodiffCompositionError("Invalid loss specified. Loss argument must be a string. "
                                                      "Currently, Mean Squared Error and Cross Entropy are the only "
                                                      "available loss functions (specified as 'mse' or 'crossentropy').")
            if loss == 'mse':
                self.loss = nn.MSELoss(size_average=False)
            else:
                self.loss = nn.CrossEntropyLoss(size_average=False)
        
        if refresh_losses == True:
            self.losses = []
        
        if targets is None:
            if epochs is not None:
                raise ParsingAutodiffCompositionError("Number of training epochs specified for {0} but no targets given."
                                                      .format(self.name))
        else:
            if epochs is None:
                raise ParsingAutodiffCompositionError("Targets specified for {0}, but no number of training epochs given."
                                                      .format(self.name))
            
            if len(self.model.get_weights_for_projections()) == 0:
                raise ParsingAutodiffCompositionError("Targets specified for training {0}, but {0} has no trainable "
                                                      "parameters."
                                                      .format(self.name))
        
        
        # set up processing scheduler
        if scheduler_processing is None:
            scheduler_processing = self.scheduler_processing
        
        # validate properties of parsing autodiff composition, get node roles, set up CIM's
        self._validate_params(targets)
        self._analyze_graph()
        
        # get execution id, do some stuff with processing scheduler
        execution_id = self._assign_execution_ids(execution_id)
        scheduler_processing._init_counts(execution_id=execution_id)
        
        # if there is only one origin mechanism, allow inputs to be specified in a list
        origin_nodes = self.get_c_nodes_by_role(CNodeRole.ORIGIN)
        if isinstance(inputs, (list, np.ndarray)):
            if len(origin_nodes) == 1:
                inputs = {next(iter(origin_nodes)): inputs}
            else:
                raise ParsingAutodiffCompositionError("Inputs to {0} must be specified in a dictionary with a "
                                                      "key for each of its {1} origin nodes."
                                                      .format(self.name, len(origin_nodes)))
        elif not isinstance(inputs, dict):
            if len(origin_nodes) == 1:
                raise ParsingAutodiffCompositionError("Inputs to {0} must be specified in a list or in a "
                                                      "dictionary with the origin mechanism({1}) as its only key."
                                                      .format(self.name, next(iter(origin_nodes)).name))
            else:
                raise ParsingAutodiffCompositionError("Inputs to {0} must be specified in a dictionary with a "
                                                      "key for each of its {1} origin nodes."
                                                      .format(self.name, len(origin_nodes)))
        
        # validate inputs, get adjusted inputs, number of input trial sets
        inputs, num_input_sets = self._adjust_stimulus_dict(inputs, 'inputs')
        
        # reset counts on processing scheduler
        scheduler_processing._reset_counts_total(TimeScale.RUN, execution_id)
        
        
        # if we're just doing step-by-step processing
        if targets is None:
            
            results = []
            
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
                
                # -----------------------------------------------------------------------------------
                
                # store the result of this execute in case it will be the final result
                if isinstance(trial_output, Iterable):
                    result_copy = trial_output.copy()
                else:
                    result_copy = trial_output
                results.append(result_copy)
            
            self.results.append(results)
        
        
        # if we're doing batch learning
        else: 
            
            # if there is only one terminal mechanism, allow targets to be specified in a list
            terminal_nodes = self.get_c_nodes_by_role(CNodeRole.ORIGIN)
            if isinstance(targets, (list, np.ndarray)):
                if len(terminal_nodes) == 1:
                    targets = {next(iter(terminal_nodes)): targets}
                else:
                    raise ParsingAutodiffCompositionError("Targets to {0} must be specified in a dictionary with a "
                                                          "key for each of its {1} terminal nodes."
                                                          .format(self.name, len(terminal_nodes)))
            elif not isinstance(targets, dict):
                if len(terminal_nodes) == 1:
                    raise ParsingAutodiffCompositionError("Targets to {0} must be specified in a list or in a "
                                                          "dictionary with the terminal mechanism({1}) as its only key."
                                                          .format(self.name, next(iter(terminal_nodes)).name))
                else:
                    raise ParsingAutodiffCompositionError("Targets to {0} must be specified in a dictionary with a "
                                                          "key for each of its {1} terminal nodes."
                                                          .format(self.name, len(terminal_nodes)))
            
            # validate targets, get adjusted targets, number of target trial sets
            targets, num_target_sets = self._adjust_stimulus_dict(targets, 'targets')
            
            # check that number of target trial sets and number of input trial sets are the same
            if num_input_sets != num_target_sets:
                raise ParsingAutodiffCompositionError("Number of input trial sets ({0}) provided and number of "
                                                      "target trial sets ({1}) provided to {2} are different."
                                                      .format(num_input_sets, num_target_sets, self.name))
            
            # LEARNING ------------------------------------------------------------------------------
            
            # call execute method
            trial_output = self.execute(inputs=inputs,
                                        targets=targets,
                                        epochs=epochs,
                                        randomize=randomize,
                                        scheduler_processing=scheduler_processing,
                                        execution_id=execution_id)
            
            # ---------------------------------------------------------------------------------------
            
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
    def autodiff_training(self, inputs, targets, epochs, randomize):
        
        # training over trial sets in random order, set up array for mapping random order back to original order
        if randomize == True:
            rand_train_order_reverse = np.zeros(len(inputs))
        
        # iterate over epochs
        for epoch in range(epochs):
            
            # if training in random order, set random number seed, generate random order
            if randomize == True:
                torch.manual_seed(epoch)
                rand_train_order = np.random.permutation(len(inputs))
            
            # set up array to keep track of losses on epoch
            curr_losses = np.zeros(len(inputs))
            
            # if we're on final epoch, set up temporary list to keep track of outputs,
            # and if training in random order, set up mapping from random order back to original order
            if epoch == epochs-1:
                outputs = []
                if randomize == True:
                    rand_train_order_reverse[rand_train_order] = np.arange(len(inputs))
            
            # iterate over trial sets
            for t in range(len(inputs)):
                
                # get current inputs, targets
                if randomize == True:
                    curr_tensor_inputs = inputs[rand_train_order[t]]
                    curr_tensor_targets = targets[rand_train_order[t]]
                else:
                    curr_tensor_inputs = inputs[t]
                    curr_tensor_targets = targets[t]
                    
                # get total number of output neurons
                out_size = 0
                for i in range(len(curr_tensor_targets)):
                    out_size += len(curr_tensor_targets[i])
                
                # run the model on inputs
                curr_tensor_outputs = self.model.forward(curr_tensor_inputs)
                
                '''
                if epoch == 0 and t == 10:
                    print("\n")
                    print(curr_tensor_outputs)
                    print("\n")
                '''
                
                # compute loss
                curr_loss = torch.zeros(1).double()
                for i in range(len(curr_tensor_outputs)):
                    nowloss = self.loss(curr_tensor_outputs[i], curr_tensor_targets[i])
                    # print(nowloss)
                    # print("\n")
                    curr_loss += nowloss
                
                # save loss on current trial
                curr_losses[t] = (curr_loss[0].item())/out_size
                
                
                # print model computational graph
                if epoch == 0 and t == 0:
                    dot = make_dot(curr_loss)
                    dot.format = 'svg'
                    dot.render()
                
                
                # compute gradients and perform parameter update
                self.optimizer.zero_grad()
                curr_loss = curr_loss/2 # *len(curr_tensor_outputs))
                curr_loss.backward()
                self.optimizer.step()
                
                # save outputs of model if this is final epoch
                if epoch == epochs-1:
                    curr_output_list = []
                    for i in range(len(curr_tensor_outputs)):
                        curr_output_list.append(curr_tensor_outputs[i].detach().numpy().copy())
                    outputs.append(curr_output_list)
            
            # save loss on current epoch
            self.losses.append(np.mean(curr_losses))
        
        # save outputs in a list in correct order, return this list
        outputs_list = []
        for i in range(len(outputs)):
            if randomize == True:
                outputs_list.append(outputs[int(rand_train_order_reverse[i])])
            else:
                outputs_list.append(outputs[i])
        
        return outputs_list
    
    
    
    # method for giving user pytorch weights and biases
    def get_parameters(self):
        
        if self.model is None:
            self.model = PytorchCreator(self.graph_processing)
            print("\n")
            print("\n")
            print("HI SHIT IS HAPPENING!")
            print("\n")
            print("\n")
        
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
                    raise ParsingAutodiffCompositionError("{0} in inputs dict for {1} is not one of its ORIGIN nodes".
                                                          format(node.name, self.name))
                else:
                    raise ParsingAutodiffCompositionError("{0} in inputs dict for {1} is not one of its TERMINAL nodes".
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
                    raise RunError("Input specification for {0} is not valid. The number of inputs (1) provided for {1}"
                                   "conflicts with at least one other node's input specification."
                                   .format(self.name, node.name))
            
            else:
                adjusted_stimuli[node] = []
                for stim in stimuli[node]:
                    
                    # check if we have 1 trial's worth of correct inputs
                    check_spec_type = self._input_matches_variable(stim, input_must_match)
                    if check_spec_type == False:
                        err_msg = "Input stimulus ({0}) for {1} is incompatible with its external_input_values ({2}).".\
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
                    raise RunError("Input specification for {0} is not valid. The number of inputs ({1}) provided for {2}"
                                   "conflicts with at least one other node's input specification."
                                   .format(self.name, len(stimuli[node]), node.name))
        
        return adjusted_stimuli, num_sets
    
    
    
    # method to validate params of parsing autodiff composition
    def _validate_params(self, targets):
        
        # set up processing graph, dictionary for checking recurrence using topological sort
        processing_graph = self.graph_processing
        topo_dict = {}
        
        # STEP 1: ENSURE THAT COMPOSITION HAS SOMETHING INSIDE IT
        
        if len([vert.component for vert in self.graph.vertices]) == 0:
            raise ParsingAutodiffCompositionError("{0} has no mechanisms or projections to execute."
                                                  .format(self.name))
        
        # STEP 2: CHECK PROPERTIES OF EACH NODE/MECHANISM IN PARSING AUTODIFF COMPOSITION
        
        # iterate over nodes in processing graph
        for node in processing_graph.vertices:
            
            # raise error if node is a composition
            if isinstance(node.component, Composition):
                raise ParsingAutodiffCompositionError("{0} was added as a node to {1}. Compositions cannot be "
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
            
            # raise error if any parent of current node creates a cycle in the composition
            topo_dict[node.component] = set()
            for parent in processing_graph.get_parents_from_component(node.component):
                topo_dict[node.component].add(parent.component)
                try:
                    list(toposort(topo_dict))
                except ValueError:
                    raise ParsingAutodiffCompositionError("Mechanisms {0} and {1} are part of a recurrent path in {2}. "
                                                          "Parsing Autodiff Compositions currently do not support recurrence."
                                                          .format(node.component, parent.component, self.name))
            
        # STEP 3: CHECK PROPERTIES THAT MUST APPLY IF TRAINING IS TO TAKE PLACE
            
        if targets is not None:
                
            # raise error if no trainable parameters are present
            if len([vert.component for vert in self.graph.vertices if isinstance(vert.component, MappingProjection)]) == 0:
                raise ParsingAutodiffCompositionError("Targets specified for {0}, but {0} has no trainable parameters."
                                                      .format(self.name))
    



