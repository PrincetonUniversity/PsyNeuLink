# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* AutodiffComposition *************************************************

"""

.. _AutodiffComposition_Overview:

Overview
--------

AutodiffComposition is a subclass of `Composition <Composition>` that trains models more quickly by integrating with
`PyTorch <https://pytorch.org/>`_, a popular machine learning library. In situations with training,
AutodiffComposition is used similarly to a Composition, but is much faster.

The `xor_in_psyneulink_and_pytorch.py` script (in the Scripts folder of the PsyNeuLink source code) is an example of
how to use AutodiffComposition. The script also gives a comparison of runtimes.

.. _AutodiffComposition_Creation:

Creating an AutodiffComposition
-------------------------------

An AutodiffComposition can be created by calling the constructor, and then adding `Components <Component>` using the
add methods of its parent class `Composition`. The most significant argument in initialization is
**param_init_from_pnl**, which controls how parameters are set up for the internal PyTorch representation of the model.

If set to True:

* Only weight parameters that correspond to projections are created. No trainable bias parameters are created, as they don’t exist for the autodiff composition’s mechanisms.

* The weight parameters are initialized to be perfectly identical to the autodiff composition’s projections - the tensor of the parameter object corresponding to a particular projection not only has the same dimensionality as the projection’s matrix, it has the same exact values.

* Pytorch functions representing mechanism functions incorporate their scalar, untrainable biases.

If set to False:

* Both weight parameters corresponding to projections and trainable bias parameters for mechanisms are created.

* Weight parameters have the same dimensionality as their corresponding projections. However, their values - and those of the bias parameters - are sampled from a random distribution.

* Though trainable biases now exist, Pytorch functions representing mechanism functions still incorporate their scalar, untrainable biases.

.. warning:: Do not add or remove Mechanisms or Projections to an AutodiffComposition after it has been run for the
    first time. Unlike an ordinary Composition, AutodiffComposition does not support this functionality.

Two other initialization arguments are **patience** and **min_delta**, allow the model to halt training early. The
model tracks how many consecutive 'bad' epochs of training have failed to significantly reduce the model's loss. Once
this number exceeds **patience**, the model stops training. By default, **patience** is ``None``, and the model
will train for the number of specified epochs and will not stop training early.

**min_delta** defines what threshold counts as a significant reduction in model loss. By default it is zero, in which
case any reduction in loss counts as a significant reduction. If **min_delta** is large and positive, the model tends to
stop earlier because it views fewer epochs as 'good'.

.. _AutodiffComposition_Structure:

Structure
---------

AutodiffComposition has all the attributes of its parent class `Composition`, in addition to several more.

The `target_CIM <AutodiffComposition.target_CIM>` attribute is analogous to the `input_CIM <Composition.input_CIM>` of
any Composition, but instead of providing inputs, provides targets for the AutodiffComposition.

The `pytorch_representation <AutodiffComposition.pytorch_representation>` attribute holds the PyTorch representation
of the PsyNeuLink model that AutodiffComposition contains. The `losses <AutodiffComposition.losses>` attribute tracks
the average loss for each training epoch.

.. _AutodiffComposition_Execution:

Execution
---------

Execute an AutodiffComposition with its `run` method. During training, both **inputs** and **targets** must be
specified. If you wish to run the AutodiffComposition without training it, only specify **inputs**. Some arguments to an
AutodiffComposition's `run` method (such as **inputs** and **execution_id**) are the same as in a Composition's
`run <Composition.run>` method. There are several different arguments as well:

**epochs** specifies the number of times the entire input set will be run. This is in contrast to **num_trials** in
Composition's `run <Composition.run>` method, which specifies the number of inputs that will be run. For example, if
your input set has size 3, setting **epochs** to 1 in AutodiffComposition is equivalent to setting **num_trials** to 3
in Composition.

**learning_rate** specifies the learning rate for this run (default 0.001), which is passed to the **optimizer**
argument. **optimizer** specifies the kind of optimizer used in training. The current options are 'sgd' (the default)
or 'adam'.

**loss** specifies the loss function for training. The current options are 'mse' (the default) and 'crossentropy'.

**randomize** specifies whether the order of inputs will be randomized in each epoch. (In each epoch, all inputs are
run, but if **randomize** is True then the order in which inputs are within an epoch is random.)

If **refresh_losses** is set to True, the AutodiffComposition resets the self.losses attribute before running.

.. _Composition_Class_Reference:

Class Reference
---------------

"""

from psyneulink.core.components.functions.function import InterfaceStateMap
from psyneulink.core.components.functions.function import Linear, Logistic, ReLU
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.states.inputstate import InputState
from psyneulink.core.components.states.outputstate import OutputState
from psyneulink.core.compositions.composition import CNodeRole
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.compositions.composition import CompositionError
from psyneulink.core.compositions.composition import RunError
from psyneulink.core.globals.keywords import OWNER_VALUE

import numpy as np

from collections import Iterable
from toposort import toposort

import logging
try:
    import torch
    from torch import nn
    import torch.optim as optim
    from psyneulink.library.compositions.pytorchmodelcreator import PytorchModelCreator
    torch_available = True
except ImportError:
    torch_available = False

logger = logging.getLogger(__name__)


__all__ = [
    'AutodiffComposition', 'AutodiffCompositionError'
]


class AutodiffCompositionError(CompositionError):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class AutodiffComposition(Composition):
    """
    AutodiffComposition(            \
    param_init_from_pnl=True,       \
    patience=None,                  \
    min_delta=0,
    name="autodiff_composition")

    Subclass of `Composition` that trains models more quickly by integrating with PyTorch.

    Arguments
    ---------

    param_init_from_pnl : boolean : default True
        a Boolean specifying how parameters are initialized. (See
        `Creating an AutodiffComposition <AutodiffComposition_Creation>` for details)

    patience : int or None : default None
        **patience** allows the model to stop training early, if training stops reducing loss. The model tracks how many
        consecutive epochs of training have failed to reduce the model's loss. When this number exceeds **patience**,
        the model stops training early. If **patience** is ``None``, the model will train for the number
        of specified epochs and will not stop training early.

    min_delta : float : default 0
        the minimum reduction in average loss that an epoch must provide in order to qualify as a 'good' epoch.
        Used for early stopping of training, in combination with **patience**.

    Attributes
    ----------

    target_CIM : CompositionInterfaceMechanism
        analogous to the input_CIM attribute, except it provides targets

    pytorch_representation : PytorchModelCreator
        the PyTorch representation of the PsyNeuLink model

    losses : list of floats
        tracks the average loss for each training epoch

    patience : int or None : default None
        allows the model to stop training early, if training stops reducing loss. The model tracks how many
        consecutive epochs of training have failed to reduce the model's loss. When this number exceeds **patience**,
        the model stops training early. If **patience** is ``None``, the model will train for the number
        of specified epochs and will not stop training early.

    min_delta : float : default 0
        the minimum reduction in average loss that an epoch must provide in order to qualify as a 'good' epoch.
        Used for early stopping of training, in combination with **patience**.

    name : str : default LeabraMechanism-<index>
        the name of the Mechanism.
        Specified in the **name** argument of the constructor for the Projection;
        if not specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    Returns
    -------
    instance of AutodiffComposition : AutodiffComposition
    """

    # TODO (CW 9/28): add compositions to registry so default arg for name is no longer needed
    def __init__(self,
                 param_init_from_pnl=True,
                 patience=None,
                 min_delta=0,
                 name="autodiff_composition"):

        if not torch_available:
            raise AutodiffCompositionError('Pytorch python module (torch) is not installed. Please install it with '
                                           '`pip install torch` or `pip3 install torch`')

        self.name = name

        super(AutodiffComposition, self).__init__()

        # set up target CIM
        self.target_CIM = CompositionInterfaceMechanism(name=self.name + " Target_CIM",
                                                        composition=self)
        self.target_CIM_states = {}

        # prevent CIM's from printing their input & output whenever an autodiff composition is run
        self.input_CIM.reportOutputPref = False
        self.output_CIM.reportOutputPref = False
        self.target_CIM.reportOutputPref = False

        # pytorch representation of model and associated training parameters
        self.pytorch_representation = None
        self.learning_rate = None
        self.optimizer = None
        self.loss = None

        # user indication of how to initialize pytorch parameters
        self.param_init_from_pnl = param_init_from_pnl

        # keeps track of average loss per epoch
        self.losses = []

        # ordered execution sets for the pytorch model
        self.ordered_execution_sets = None

        # patience is the "bad" epochs (with no progress in average loss) the model tolerates in one training session
        # before ending training
        self.patience = patience

        self.min_delta = min_delta


    # TODO (CW 9/28): this mirrors _create_CIM_states() in Composition but doesn't call super().
    #   This is not ideal, but I don't see a simple way to rewrite this function to call super().
    # Very similar to the same function for the normal composition.
    # Overriden to create target CIM as well as input and output CIM's
    def _create_CIM_states(self):

        #  INPUT CIM
        # loop over all origin nodes

        current_origin_input_states = set()

        for node in self.get_c_nodes_by_role(CNodeRole.ORIGIN):

            for input_state in node.external_input_states:
                current_origin_input_states.add(input_state)

                # if there are no CIM input and output states for the origin node input state, add them
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

        # For any state still registered on the CIM that does not map to a corresponding ORIGIN node I.S.:
        for input_state in sends_to_input_states.difference(current_origin_input_states):

            # remove the CIM input and output states associated with this Origin node input state
            self.input_CIM.input_states.remove(self.input_CIM_states[input_state][0])
            self.input_CIM.output_states.remove(self.input_CIM_states[input_state][1])

            # and from the dictionary of CIM output state/input state pairs
            del self.input_CIM_states[input_state]


        # OUTPUT AND TARGET CIM's
        # loop over all terminal nodes

        current_terminal_output_states = set()
        current_terminal_input_states = set()

        for node in self.get_c_nodes_by_role(CNodeRole.TERMINAL):

            for output_state in node.output_states:
                current_terminal_output_states.add(output_state)

                # if there are no CIM input and output states for the origin node output state, add them
                if output_state not in set(self.output_CIM_states.keys()):

                    interface_input_state = InputState(owner=self.output_CIM,
                                                       variable=output_state.value,
                                                       reference_value=output_state.value,
                                                       name="OUTPUT_CIM_" + node.name + "_" + output_state.name)

                    interface_output_state = OutputState(
                            owner=self.output_CIM,
                            variable=OWNER_VALUE,
                            function=InterfaceStateMap(corresponding_input_state=interface_input_state),
                            reference_value=output_state.value,
                            name="OUTPUT_CIM_" + node.name + "_" + output_state.name)

                    self.output_CIM_states[output_state] = [interface_input_state, interface_output_state]

            for input_state in node.input_states:
                current_terminal_input_states.add(input_state)

                # if there are no CIM input and output states for the origin node input state, add them
                if input_state not in set(self.target_CIM_states.keys()):

                    interface_input_state = InputState(owner=self.target_CIM,
                                                       variable=input_state.value,
                                                       reference_value=input_state.value,
                                                       name="TARGET_CIM_" + node.name + "_" + input_state.name)

                    interface_output_state = OutputState(
                            owner=self.target_CIM,
                            variable=OWNER_VALUE,
                            function=InterfaceStateMap(corresponding_input_state=interface_input_state),
                            reference_value=input_state.value,
                            name="TARGET_CIM_" + node.name + "_" + input_state.name)

                    self.target_CIM_states[input_state] = [interface_input_state, interface_output_state]


        previous_terminal_output_states = set(self.output_CIM_states.keys())

        # For any states still registered on the CIM that does not map to a corresponding ORIGIN node I.S.:
        for output_state in previous_terminal_output_states.difference(current_terminal_output_states):

            # remove the CIM input and output states associated with this Terminal Node output state
            self.output_CIM.remove_states(self.output_CIM_states[output_state][0])
            self.output_CIM.remove_states(self.output_CIM_states[output_state][1])

            # and from the dictionary of CIM output state/input state pairs
            del self.output_CIM_states[output_state]

        previous_terminal_target_states = set(self.target_CIM_states.keys())

        # For any states still registered on the CIM that does not map to a corresponding ORIGIN node I.S.:
        for target_state in previous_terminal_target_states.difference(current_terminal_input_states):

            # remove the CIM input and output states associated with this Terminal Node output state
            self.target_CIM.remove_states(self.output_CIM_states[output_state][0])
            self.target_CIM.remove_states(self.output_CIM_states[output_state][1])

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



    def _assign_execution_ids(self, execution_id=None):

        exec_id = super()._assign_execution_ids(execution_id=execution_id)

        self.target_CIM._execution_id = exec_id

        return exec_id


    # similar function to _assign_values_to_input_CIM from the normal composition - however, this
    # assigns values to the input or target CIM of autodiff composition,
    # executes the CIM's, and puts the values in appropriate form for pytorch
    def _throw_through_input_CIM(self, stimuli, inputs_or_targets):

        # set up some variables to use based on whether we have inputs or targets
        if inputs_or_targets == 'inputs':
            CIM = self.input_CIM
            states = self.input_CIM_states
            order = self.ordered_execution_sets[0]
        else:
            CIM = self.target_CIM
            states = self.target_CIM_states
            order = self.ordered_execution_sets[len(self.ordered_execution_sets)-1]

        # set up list that will hold values for CIM
        CIM_list = []

        # add values to CIM list in correct order
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

        # execute CIM
        CIM.execute(CIM_list)

        # set up list that will hold values for pytorch
        pytorch_list = []

        # iterate over nodes in pytorch's desired order, add corresponding inputs from CIM
        # output to pytorch list in that order. convert them to torch tensors in the process
        for i in range(len(order)):

            # get output state corresponding to ith node in pytorch's desired order, add
            # the value of the output state to pytorch list at position i
            node = order[i]
            value = states[node.component.input_states[0]][1].value
            value_for_pytorch = torch.from_numpy(np.asarray(value).copy()).double()
            pytorch_list.append(value_for_pytorch)

        return pytorch_list



    # similar function to _throw_through_input_CIM - however, this gets pytorch output from execute,
    # assigns it to the output CIM of autodiff composition, executes the CIM, and sends
    # its output in a list back to execute
    def _throw_through_output_CIM(self, outputs):

        order = self.ordered_execution_sets[len(self.ordered_execution_sets)-1]

        output_CIM_list = []

        # iterate over CIM input states - for each CIM input state, find mechanism in final execution set
        # whose output state maps to the CIM input state, add pytorch output for this mechanism
        # to output CIM list
        for input_state in self.output_CIM.input_states:

            for i in range(len(order)):
                node = order[i]
                if self.output_CIM_states[node.component.output_states[0]][0] == input_state:
                    value = outputs[i]

            output_CIM_list.append(value)

        self.output_CIM.execute(output_CIM_list)

        output_values = []
        for i in range(len(self.output_CIM.output_states)):
            output_values.append(self.output_CIM.output_states[i].value)

        return output_values



    # performs forward computation for one input
    def autodiff_processing(self, inputs):

        # run the model on inputs - switch autograd off for this (we don't need it)
        with torch.no_grad():
            tensor_outputs = self.pytorch_representation.forward(inputs)

        # get outputs back into numpy
        outputs = []
        for i in range(len(tensor_outputs)):
            outputs.append(tensor_outputs[i].numpy().copy())

        return outputs



    # performs learning/training on all input-target pairs it recieves for given number of epochs
    def autodiff_training(self, inputs, targets, epochs, randomize):

        if self.patience is not None:
            # set up object for early stopping
            early_stopper = EarlyStopping(patience=self.patience, min_delta=self.min_delta)

        # if training over trial sets in random order, set up array for mapping random order back to original order
        if randomize:
            rand_train_order_reverse = np.zeros(len(inputs))

        # get total number of output neurons from the dimensionality of targets on the first trial
        # (this is for computing average loss across neurons on each trial later)
        out_size = 0
        for i in range(len(targets[0])):
            out_size += len(targets[0][i])

        # iterate over epochs
        for epoch in range(epochs):

            # if training in random order, generate random order and set up mapping
            # from random order back to original order
            if randomize:
                rand_train_order = np.random.permutation(len(inputs))
                rand_train_order_reverse[rand_train_order] = np.arange(len(inputs))

            # set up array to keep track of losses on epoch
            curr_losses = np.zeros(len(inputs))

            # reset temporary list to keep track of most recent outputs
            outputs = []

            # iterate over inputs, targets
            for t in range(len(inputs)):

                # get current inputs, targets
                if randomize:
                    curr_tensor_inputs = inputs[rand_train_order[t]]
                    curr_tensor_targets = targets[rand_train_order[t]]
                else:
                    curr_tensor_inputs = inputs[t]
                    curr_tensor_targets = targets[t]

                # do forward computation on current inputs
                curr_tensor_outputs = self.pytorch_representation.forward(curr_tensor_inputs)

                # compute total loss across output neurons for current trial
                curr_loss = torch.zeros(1).double()
                for i in range(len(curr_tensor_outputs)):
                    curr_loss += self.loss(curr_tensor_outputs[i], curr_tensor_targets[i])

                # save average loss across all output neurons on current trial
                curr_losses[t] = (curr_loss[0].item())/out_size

                # backpropagate to compute gradients and perform learning update for parameters
                self.optimizer.zero_grad()
                curr_loss = curr_loss/2
                curr_loss.backward()
                self.optimizer.step()

                # save outputs of model if this is final epoch
                curr_output_list = []
                for i in range(len(curr_tensor_outputs)):
                    curr_output_list.append(curr_tensor_outputs[i].detach().numpy().copy())
                outputs.append(curr_output_list)

            # save average loss on the current epoch
            average_loss = np.mean(curr_losses)
            self.losses.append(average_loss)

            # update early stopper with most recent average loss
            if self.patience is not None:
                should_stop = early_stopper.step(average_loss)
                if should_stop:
                    if randomize:
                        outputs_list = [None] * len(outputs)
                        for i in range(len(outputs)):
                            outputs_list[i] = outputs[int(rand_train_order_reverse[i])]
                        return outputs_list
                    else:
                        return outputs

        if randomize:  # save outputs in a list in correct order, return them
            outputs_list = [None] * len(outputs)
            for i in range(len(outputs)):
                outputs_list[i] = outputs[int(rand_train_order_reverse[i])]
            return outputs_list
        else:
            return outputs



    # orchestrates either processing for one input or training for a given set of
    # inputs and targets for a provided number of epochs. Sends values through input
    # CIM's, calls functions that perform processing/training with the pytorch
    # representation of the model, then sends values through output CIM's.
    def execute(
        self,
        inputs=None,
        targets=None,
        epochs=None,
        randomize=False,
        execution_id=None
    ):

        # set up execution id
        execution_id = self._assign_execution_ids(execution_id)

        # if we're doing step-by-step processing
        if targets is None:

            # push values through input CIM, get them in correct form for pytorch
            autodiff_inputs = self._throw_through_input_CIM(inputs, 'inputs')

            # call function to do all processing
            outputs = self.autodiff_processing(autodiff_inputs)

            # get outputs in correct form for output CIM, push them through output CIM
            output_values = self._throw_through_output_CIM(outputs)

            return output_values


        # if we're doing learning
        else:

            # create empty arrays to hold inputs/targets in correct form for pytorch
            autodiff_inputs = []
            autodiff_targets = []

            # iterate over trials
            for i in range(len(next(iter(inputs.values())))):

                # create input/target dictionary for inputs/targets on current trial
                input_stimuli = {}
                for node in inputs:
                    input_stimuli[node] = inputs[node][i]
                target_stimuli = {}
                for node in targets:
                    target_stimuli[node] = targets[node][i]

                # send inputs/targets through CIM's, get them in correct form for pytorch
                autodiff_inputs.append(self._throw_through_input_CIM(input_stimuli, 'inputs'))
                autodiff_targets.append(self._throw_through_input_CIM(target_stimuli, 'targets'))

            # call function to do all learning/training
            outputs = self.autodiff_training(autodiff_inputs, autodiff_targets, epochs, randomize=randomize)

            # get outputs in correct form for output CIM, push them through output CIM
            output_values = []
            for i in range(len(outputs)):
                output_values.append(self._throw_through_output_CIM(outputs[i]))

            return output_values



    # what the user calls for doing processing/training, similar to the run function of the normal composition
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
        execution_id=None
    ):

        # validate arguments, and properties of the autodiff composition
        self._validate_params(targets, epochs)

        # set up mechanism execution order
        if self.ordered_execution_sets is None:
            self.ordered_execution_sets = self.get_ordered_exec_sets(self.graph_processing)

        # set up pytorch representation of the autodiff composition's model
        if self.pytorch_representation is None:
            self.pytorch_representation = PytorchModelCreator(self.graph_processing, self.param_init_from_pnl, self.ordered_execution_sets)

        # if we're doing learning/training, set up learning rate, optimizer, and loss
        if (targets is not None):

            if learning_rate is None: # FIX DCW 10/8/18: I think this logic is wrong!
                if self.learning_rate is None:
                    self.learning_rate = 0.001
            else:
                if not isinstance(learning_rate, (int, float)):
                    raise AutodiffCompositionError("Learning rate must be an integer or float value.")
                self.learning_rate = learning_rate

            if optimizer is None:
                if self.optimizer is None:
                    self.optimizer = optim.SGD(self.pytorch_representation.parameters(), lr=self.learning_rate)
            else:
                if optimizer not in ['sgd', 'adam']:
                    raise AutodiffCompositionError("Invalid optimizer specified. Optimizer argument must be a string. "
                                                   "Currently, Stochastic Gradient Descent and Adam are the only available "
                                                   "optimizers (specified as 'sgd' or 'adam').")
                if optimizer == 'sgd':
                    self.optimizer = optim.SGD(self.pytorch_representation.parameters(), lr=self.learning_rate)
                else:
                    self.optimizer = optim.Adam(self.pytorch_representation.parameters(), lr=self.learning_rate)

            if loss is None:
                if self.loss is None:
                    self.loss = nn.MSELoss(reduction='sum')
            else:
                if loss not in ['mse', 'crossentropy']:
                    raise AutodiffCompositionError("Invalid loss specified. Loss argument must be a string. "
                                                   "Currently, Mean Squared Error and Cross Entropy are the only "
                                                   "available loss functions (specified as 'mse' or 'crossentropy').")
                if loss == 'mse':
                    self.loss = nn.MSELoss(reduction='sum')
                else:
                    self.loss = nn.CrossEntropyLoss(reduction='sum')

        # allow user to refresh the list tracking loss on every epoch in the autodiff composition's training history
        if refresh_losses:
            self.losses = []

        # get node roles, set up CIM's
        self._analyze_graph()

        # get execution id
        execution_id = self._assign_execution_ids(execution_id)

        # validate how inputs are specified - if there is only one origin mechanism,
        # allow inputs to be specified in a list
        origin_nodes = self.get_c_nodes_by_role(CNodeRole.ORIGIN)
        if isinstance(inputs, (list, np.ndarray)):
            if len(origin_nodes) == 1:
                inputs = {next(iter(origin_nodes)): inputs}
            else:
                raise AutodiffCompositionError("Inputs to {0} must be specified in a dictionary with a "
                                               "key for each of its {1} origin nodes."
                                               .format(self.name, len(origin_nodes)))
        elif not isinstance(inputs, dict):
            if len(origin_nodes) == 1:
                raise AutodiffCompositionError("Inputs to {0} must be specified in a list or in a "
                                               "dictionary with the origin mechanism({1}) as its only key."
                                               .format(self.name, next(iter(origin_nodes)).name))
            else:
                raise AutodiffCompositionError("Inputs to {0} must be specified in a dictionary with a "
                                               "key for each of its {1} origin nodes."
                                               .format(self.name, len(origin_nodes)))

        # validate inputs, get adjusted inputs, number of input trial sets
        inputs, num_input_sets = self._adjust_stimulus_dict(inputs, 'inputs')


        # if we're just doing step-by-step processing
        if targets is None:

            results = []

            # iterate over inputs
            for trial_num in range(num_input_sets):

                # PROCESSING ------------------------------------------------------------------------

                # prepare current input
                execution_stimuli = {}
                for node in inputs:
                    execution_stimuli[node] = inputs[node][trial_num]

                # call execute function to process current input
                trial_output = self.execute(inputs=execution_stimuli,
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

            # validate how targets are specified - if there is only one terminal mechanism,
            # allow targets to be specified in a list
            terminal_nodes = self.get_c_nodes_by_role(CNodeRole.TERMINAL)
            if isinstance(targets, (list, np.ndarray)):
                if len(terminal_nodes) == 1:
                    targets = {next(iter(terminal_nodes)): targets}
                else:
                    raise AutodiffCompositionError("Targets to {0} must be specified in a dictionary with a "
                                                   "key for each of its {1} terminal nodes."
                                                   .format(self.name, len(terminal_nodes)))
            elif not isinstance(targets, dict):
                if len(terminal_nodes) == 1:
                    raise AutodiffCompositionError("Targets to {0} must be specified in a list or in a "
                                                   "dictionary with the terminal mechanism({1}) as its only key."
                                                   .format(self.name, next(iter(terminal_nodes)).name))
                else:
                    raise AutodiffCompositionError("Targets to {0} must be specified in a dictionary with a "
                                                   "key for each of its {1} terminal nodes."
                                                   .format(self.name, len(terminal_nodes)))

            # validate targets, get adjusted targets, number of target trial sets
            targets, num_target_sets = self._adjust_stimulus_dict(targets, 'targets')

            # check that number of target trial sets and number of input trial sets are the same
            if num_input_sets != num_target_sets:
                raise AutodiffCompositionError("Number of input trial sets ({0}) provided and number of "
                                               "target trial sets ({1}) provided to {2} are different."
                                               .format(num_input_sets, num_target_sets, self.name))

            # LEARNING ------------------------------------------------------------------------------

            # call execute function to do learning for desired number of epochs on all input-target pairs
            trial_output = self.execute(inputs=inputs,
                                        targets=targets,
                                        epochs=epochs,
                                        randomize=randomize,
                                        execution_id=execution_id)

            # ---------------------------------------------------------------------------------------

            # store the result of this execute
            if isinstance(trial_output, Iterable):
                result_copy = trial_output.copy()
            else:
                result_copy = trial_output
            self.results.append(result_copy)


        # return result
        return self.results



    # validates properties of the autodiff composition, and arguments to run, when run is called
    def _validate_params(self, targets, epochs):

        # set up processing graph and dictionary (for checking if recurrence is present later)
        processing_graph = self.graph_processing
        topo_dict = {}

        # raise error if composition is empty
        if len([vert.component for vert in self.graph.vertices]) == 0:
            raise AutodiffCompositionError("{0} has no mechanisms or projections to execute."
                                           .format(self.name))

        # iterate over nodes in processing graph
        for node in processing_graph.vertices:

            # raise error if a node is a composition
            if isinstance(node.component, Composition):
                raise AutodiffCompositionError("{0} was added as a node to {1}. Compositions cannot be "
                                               "added as nodes to Autodiff Compositions."
                                               .format(node.component, self.name))

            # raise error if a node's mechanism doesn't have a Linear, Logistic, or ReLU function
            if not isinstance(node.component.function_object, (Linear, Logistic, ReLU)):
                raise AutodiffCompositionError("Function {0} of mechanism {1} in {2} is not a valid function "
                                               "for a Autodiff Composition. Functions of mechanisms in "
                                               "Autodiff Compositions can only be Linear, Logistic, or ReLU."
                                               .format(node.component.function, node.component, self.name))

            # raise error if a node has more than one input state
            if len(node.component.input_states) > 1:
                raise AutodiffCompositionError("Mechanism {0} of {1} has more than one input state. Autodiff "
                                               "Compositions only allow mechanisms to have one input state. The "
                                               "dimensionality of this state's value will become the dimensionality of "
                                               "the tensor representing the state's mechanism in the underlying "
                                               "Pytorch model."
                                               .format(node.component, self.name))

            # raise error if any parent of current node creates a cycle in the composition (ie. if there's recurrence)
            topo_dict[node.component] = set()
            for parent in processing_graph.get_parents_from_component(node.component):
                topo_dict[node.component].add(parent.component)
                try:
                    list(toposort(topo_dict))
                except ValueError:
                    raise AutodiffCompositionError("Mechanisms {0} and {1} are part of a recurrent path in {2}. "
                                                   "Autodiff Compositions currently do not support recurrence."
                                                   .format(node.component, parent.component, self.name))

        # raise errors if arguments to run are not consistent or we're doing training but there are
        # no trainable parameters
        if targets is None:
            if epochs is not None:
                raise AutodiffCompositionError("Number of training epochs specified for {0} but no targets given."
                                               .format(self.name))

        else:
            if epochs is None:
                raise AutodiffCompositionError("Targets specified for {0}, but no number of training epochs given."
                                               .format(self.name))

            if len([vert.component for vert in self.graph.vertices if isinstance(vert.component, MappingProjection)]) == 0:
                raise AutodiffCompositionError("Targets specified for {0}, but {0} has no trainable parameters."
                                               .format(self.name))



    # defines order in which mechanisms can be executed from processing graph
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



    # recursive helper function for get_ordered_exec_sets
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


    # TODO (CW 9/28): this mirrors _adjust_stimulus_dict() in Composition but doesn't call super().
    #   This is not ideal, but I don't see a simple way to rewrite this function to call super().
    # validates inputs/targets. overriden to be able to adjust a dictionary of inputs or targets
    # (not just inputs). the adjusting is exactly the same though.
    def _adjust_stimulus_dict(self, stimuli, inputs_or_targets):


        # check if we're dealing with inputs or targets, set variables accordingly
        if inputs_or_targets == 'inputs':
            nodes = self.get_c_nodes_by_role(CNodeRole.ORIGIN)
        else:
            nodes = self.get_c_nodes_by_role(CNodeRole.TERMINAL)


        # STEP 1: Validate that there is a one-to-one mapping of input/target entries to origin/terminal nodes

        # Check that all of the nodes listed in the stimuli dict are ORIGIN/TERMINAL nodes in self
        for node in stimuli.keys():
            if not node in nodes:
                if inputs_or_targets == 'inputs':
                    raise AutodiffCompositionError("{0} in inputs dict for {1} is not one of its ORIGIN nodes"
                                                   .format(node.name, self.name))
                else:
                    raise AutodiffCompositionError("{0} in inputs dict for {1} is not one of its TERMINAL nodes"
                                                   .format(node.name, self.name))

        # Check that all of the ORIGIN/TERMINAL nodes are represented - if not, use default_variable
        for node in nodes:
            if not node in stimuli:
                stimuli[node] = node.default_external_input_values


        # STEP 2: Loop over all dictionary entries to validate their content and adjust any convenience notations:

        adjusted_stimuli = {}
        num_sets = -1

        for node, stim_list in stimuli.items():

            input_must_match = node.external_input_values

            # check if we have 1 trial's worth of correct inputs/targets
            check_spec_type = self._input_matches_variable(stim_list, input_must_match)
            if check_spec_type == "homogeneous":
                adjusted_stimuli[node] = [np.atleast_2d(stim_list)]

                # verify that all nodes have provided the same number of inputs/targets
                if num_sets == -1:
                    num_sets = 1
                elif num_sets != 1:
                    raise RunError("Input specification for {0} is not valid. The number of inputs (1) provided for {1}"
                                   "conflicts with at least one other node's input specification."
                                   .format(self.name, node.name))

            else:
                adjusted_stimuli[node] = []
                for stim in stimuli[node]:

                    # check if we have 1 trial's worth of correct inputs/targets
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

                # verify that all nodes have provided the same number of inputs/targets
                if num_sets == -1:
                    num_sets = len(stimuli[node])
                elif num_sets != len(stimuli[node]):
                    raise RunError("Input specification for {0} is not valid. The number of inputs ({1}) provided for {2}"
                                   "conflicts with at least one other node's input specification."
                                   .format(self.name, len(stimuli[node]), node.name))

        return adjusted_stimuli, num_sets



    # helper function for _adjust_stimulus_dict. overriden to permit only homogenous inputs -
    # autodiff compositions cannot have mechanisms with multiple input states of different lengths
    def _input_matches_variable(self, input_value, var):

        if np.shape(np.atleast_2d(input_value)) == np.shape(var):
            return "homogeneous"

        return False



    # gives user weights and biases of the model (from the pytorch representation)
    def get_parameters(self):

        if self.pytorch_representation is None:
            raise AutodiffCompositionError("{0} has not been run yet so parameters have not been created "
                                           "in Pytorch."
                                           .format(self.name))

        weights = self.pytorch_representation.get_weights_for_projections()
        biases = self.pytorch_representation.get_biases_for_mechanisms()

        return weights, biases

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta


