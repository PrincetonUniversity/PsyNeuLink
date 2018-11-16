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

If **refresh_losses** is set to True, the AutodiffComposition resets the self.losses attribute before running.

.. _Composition_Class_Reference:

Class Reference
---------------

"""

from psyneulink.core.components.component import Param
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
from psyneulink.core.globals.keywords import OWNER_VALUE, SOFT_CLAMP
from psyneulink.core.scheduling.scheduler import Scheduler

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
    min_delta=0,                    \
    randomize=False,                \
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

    randomize: boolean : default False
        specifies whether the order of inputs will be randomized in each epoch. (In each epoch, all inputs are run, but
        if **randomize** is True then the order of inputs within an epoch is random.)

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

    class Params(Composition.Params):
        loss = Param(nn.MSELoss(reduction='sum'), stateful=False, loggable=False)

        optimizer = None
        learning_rate = .5
        losses = None
        patience = None
        min_delta = 0
        pytorch_representation = None

    # TODO (CW 9/28): add compositions to registry so default arg for name is no longer needed
    def __init__(self,
                 param_init_from_pnl=True,
                 patience=None,
                 min_delta=0,
                 learning_rate=0.5,
                 learning_enabled=True,
                 optimizer_type=None,
                 loss_type=None,
                 randomize=None,
                 refresh_losses=False,
                 name="autodiff_composition"):

        self.learning_enabled = True
        if not torch_available:
            raise AutodiffCompositionError('Pytorch python module (torch) is not installed. Please install it with '
                                           '`pip install torch` or `pip3 install torch`')

        # since this does not pass params argument, defaults will not be automatically set..
        super(AutodiffComposition, self).__init__(name=name)

        self.learning_enabled = learning_enabled
        self.optimizer_type = optimizer_type
        self.loss_type = loss_type
        self.randomize = randomize
        self.refresh_losses = refresh_losses

        # pytorch representation of model and associated training parameters
        self.pytorch_representation = None
        self.learning_rate = learning_rate
        self.optimizer = None
        self.loss = self.defaults.loss

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

        # CW 11/1/18: maybe we should make scheduler a property, like in Composition
        self.scheduler = None

    # CLEANUP: move some of what's done in the method below to a "validate_params" type of method
    def _build_pytorch_representation(self):
        if self.scheduler is None:  # if learning_enabled has never been run yet
            self.scheduler = Scheduler(graph=self.graph_processing)
        if self.ordered_execution_sets is None:
            self.ordered_execution_sets = list(self.scheduler.run())
        if self.pytorch_representation is None:
            self.pytorch_representation = PytorchModelCreator(self.graph_processing,
                                                              self.param_init_from_pnl,
                                                              self.ordered_execution_sets)
         # Set up optimizer function
        if self.optimizer_type is None:
            if self.optimizer is None:
                self.optimizer = optim.SGD(self.pytorch_representation.parameters(), lr=self.learning_rate)
        else:
            if self.optimizer_type not in ['sgd', 'adam']:
                raise AutodiffCompositionError("Invalid optimizer specified. Optimizer argument must be a string. "
                                               "Currently, Stochastic Gradient Descent and Adam are the only available "
                                               "optimizers (specified as 'sgd' or 'adam').")
            if self.optimizer_type == 'sgd':
                self.optimizer = optim.SGD(self.pytorch_representation.parameters(), lr=self.learning_rate)
            else:
                self.optimizer = optim.Adam(self.pytorch_representation.parameters(), lr=self.learning_rate)
        if self.loss_type is None:
            if self.loss is None:
                self.loss = nn.MSELoss(reduction='sum')
        else:
            if self.loss_type not in ['mse', 'crossentropy']:
                raise AutodiffCompositionError("Invalid loss specified. Loss argument must be a string. "
                                               "Currently, Mean Squared Error and Cross Entropy are the only "
                                               "available loss functions (specified as 'mse' or 'crossentropy').")
            if self.loss_type == 'mse':
                self.loss = nn.MSELoss(reduction='sum')
            else:
                self.loss = nn.CrossEntropyLoss(reduction='sum')
        if not isinstance(self.learning_rate, (int, float)):
            raise AutodiffCompositionError("Learning rate must be an integer or float value.")

    def _reshape_for_autodiff(self, stimuli):
        TEMP = self.get_ordered_exec_sets(self.graph_processing)
        order = {"inputs": TEMP[0],
                 "targets": TEMP[-1]}
        pytorch_stimuli = {}
        for stimulus_type in order:
            pytorch_list = []
            # iterate over nodes in pytorch's desired order, add corresponding inputs from CIM
            # output to pytorch list in that order. convert them to torch tensors in the process
            for i in range(len(order[stimulus_type])):
                # get output state corresponding to ith node in pytorch's desired order, add
                # the value of the output state to pytorch list at position i
                node = order[stimulus_type][i].component
                value = stimuli[stimulus_type][node]
                value_for_pytorch = torch.from_numpy(np.asarray(value).copy()).double()
                pytorch_list.append(value_for_pytorch)
            pytorch_stimuli[stimulus_type] = pytorch_list
        return pytorch_stimuli

    def _has_required_keys(self, input_dict):
        required_keys = set(["inputs", "targets"])
        return required_keys.issubset(set(input_dict.keys()))

    def _adjust_stimulus_dict(self, inputs):
        if self.learning_enabled:
            if isinstance(inputs, dict):
                if self._has_required_keys(inputs):
                    return [inputs]
                raise AutodiffCompositionError("Invalid input specification.")
            elif isinstance(inputs, list):
                for input_dict in inputs:
                    if not self._has_required_keys(input_dict):
                        raise AutodiffCompositionError("Invalid input specification.")
                return inputs
        return super(AutodiffComposition, self)._adjust_stimulus_dict(inputs)

    # performs forward computation for one input
    def autodiff_processing(self, inputs, execution_id=None):
        pytorch_representation = self.parameters.pytorch_representation.get(execution_id)
        # run the model on inputs - switch autograd off for this (we don't need it)
        with torch.no_grad():
            tensor_outputs = pytorch_representation.forward(inputs, execution_id=execution_id)

        # get outputs back into numpy
        outputs = []
        for i in range(len(tensor_outputs)):
            outputs.append(tensor_outputs[i].numpy().copy())

        return outputs

    # performs learning/training on all input-target pairs it recieves for given number of epochs
    def autodiff_training(self, inputs, targets, epochs, execution_id=None):

        # FIX CW 11/1/18: this value of num_inputs assumes all inputs have same length, and that the length of
        # the input for an origin component equals the number of desired trials. We could clean this up
        # by perhaps using modular arithmetic on t, or by being more explicit about number of desired trials
        first_input_value = list(inputs.values())[0]
        num_inputs = len(first_input_value)

        patience = self.parameters.patience.get(execution_id)

        if patience is not None:
            # set up object for early stopping
            early_stopper = EarlyStopping(patience=patience, min_delta=self.parameters.min_delta.get(execution_id))

        # if training over trial sets in random order, set up array for mapping random order back to original order
        if self.randomize:
            rand_train_order_reverse = np.zeros(num_inputs)

        # get total number of output neurons from the dimensionality of targets on the first trial
        # (this is for computing average loss across neurons on each trial later)
        out_size = 0
        for target in targets.values():
            out_size += len(target)

        # iterate over epochs
        for epoch in range(epochs):

            # if training in random order, generate random order and set up mapping
            # from random order back to original order
            if self.randomize:
                rand_train_order = np.random.permutation(num_inputs)
                rand_train_order_reverse[rand_train_order] = np.arange(num_inputs)

            # set up array to keep track of losses on epoch
            curr_losses = np.zeros(num_inputs)

            # reset temporary list to keep track of most recent outputs
            outputs = []

            # iterate over inputs, targets
            for t in range(num_inputs):

                if self.randomize:
                    input_index = rand_train_order[t]
                else:
                    input_index = t
                curr_tensor_inputs = {}
                curr_tensor_targets = {}
                for component in inputs.keys():
                    input = inputs[component][input_index]
                    curr_tensor_inputs[component] = torch.tensor(input).double()
                for component in targets.keys():
                    target = targets[component][input_index]
                    curr_tensor_targets[component] = torch.tensor(target).double()

                # do forward computation on current inputs
                curr_tensor_outputs = self.parameters.pytorch_representation.get(execution_id).forward(
                    curr_tensor_inputs,
                    execution_id
                )

                # compute total loss across output neurons for current trial
                curr_loss = torch.zeros(1).double()
                for component in curr_tensor_outputs.keys():
                    curr_loss += self.loss(curr_tensor_outputs[component], curr_tensor_targets[component])

                # save average loss across all output neurons on current trial
                curr_losses[t] = (curr_loss[0].item())/out_size

                optimizer = self.parameters.optimizer.get(execution_id)

                # backpropagate to compute gradients and perform learning update for parameters
                optimizer.zero_grad()
                curr_loss = curr_loss/2
                curr_loss.backward()
                optimizer.step()

                # save outputs of model if this is final epoch
                curr_output_list = []
                for component in curr_tensor_outputs.keys():
                    curr_output_list.append(curr_tensor_outputs[component].detach().numpy().copy())
                outputs.append(curr_output_list)

            # save average loss on the current epoch
            average_loss = np.mean(curr_losses)
            self.parameters.losses.get(execution_id).append(average_loss)

            # update early stopper with most recent average loss
            if self.parameters.patience.get(execution_id) is not None:
                should_stop = early_stopper.step(average_loss)
                if should_stop:
                    logger.warning('Stopped training early after {} epochs'.format(epoch))
                    if self.randomize:
                        outputs_list = [None] * len(outputs)
                        for i in range(len(outputs)):
                            outputs_list[i] = outputs[int(rand_train_order_reverse[i])]
                        return outputs_list
                    else:
                        return outputs

        if self.randomize:  # save outputs in a list in correct order, return them
            outputs_list = [None] * len(outputs)
            for i in range(len(outputs)):
                outputs_list[i] = outputs[int(rand_train_order_reverse[i])]
            return outputs_list
        else:
            return outputs

    def execute(self,
                inputs=None,
                autodiff_stimuli=None,
                scheduler_processing=None,
                scheduler_learning=None,
                termination_processing=None,
                termination_learning=None,
                call_before_time_step=None,
                call_before_pass=None,
                call_after_time_step=None,
                call_after_pass=None,
                execution_id=None,
                base_execution_id=None,
                clamp_input=SOFT_CLAMP,
                targets=None,
                runtime_params=None,
                bin_execute=False,
                context=None
                ):
        if self.learning_enabled:
            # TBI: How are we supposed to use base_execution_id and statefulness here?
            # TBI: can we call _build_pytorch_representation in _analyze_graph so that pytorch
            # model may be modified between runs?
            self._build_pytorch_representation()

            if execution_id is None:
                execution_id = self._assign_execution_ids(execution_id)

            autodiff_inputs = inputs["inputs"]
            autodiff_targets = inputs["targets"]
            autodiff_epochs = 1
            if "epochs" in inputs:
                autodiff_epochs = inputs["epochs"]

            output = self.autodiff_training(autodiff_inputs, autodiff_targets, autodiff_epochs)
            
            return output

        # learning not enabled. execute as a normal composition
        return super(AutodiffComposition, self).execute(inputs=inputs,
                                                        scheduler_processing=scheduler_processing,
                                                        scheduler_learning=scheduler_learning,
                                                        termination_processing=termination_processing,
                                                        termination_learning=termination_learning,
                                                        call_before_time_step=call_before_time_step,
                                                        call_before_pass=call_before_pass,
                                                        call_after_time_step=call_after_time_step,
                                                        call_after_pass=call_after_pass,
                                                        execution_id=execution_id,
                                                        base_execution_id=base_execution_id,
                                                        clamp_input=clamp_input,
                                                        targets=targets,
                                                        runtime_params=runtime_params,
                                                        bin_execute=bin_execute,
                                                        context=context)

    # what the user calls for doing processing/training, similar to the run function of the normal composition
    def run(
        self,
        inputs=None,
        scheduler_processing=None,
        scheduler_learning=None,
        termination_processing=None,
        termination_learning=None,
        execution_id=None,
        num_trials=None,
        call_before_time_step=None,
        call_after_time_step=None,
        call_before_pass=None,
        call_after_pass=None,
        call_before_trial=None,
        call_after_trial=None,
        clamp_input=SOFT_CLAMP,
        targets=None,
        bin_execute=False,
        initial_values=None,
        runtime_params=None):
        # TBI: Handle trials, timesteps, etc
        if self.learning_enabled:

            self._analyze_graph()

            if self.refresh_losses:
                self.losses = []
            adjusted_stimuli = self._adjust_stimulus_dict(inputs)
            if num_trials is not None:
                for trial_num in range(num_trials):
                    stimulus_index = trial_num % len(adjusted_stimuli)
                    return self.execute(inputs=adjusted_stimuli[stimulus_index])
            else:
                for stimulus in adjusted_stimuli:
                    return self.execute(inputs=stimulus)
        return super(AutodiffComposition, self).run(inputs=inputs,
                                                    scheduler_processing=scheduler_processing,
                                                    scheduler_learning=scheduler_learning,
                                                    termination_processing=termination_processing,
                                                    termination_learning=termination_learning,
                                                    execution_id=execution_id,
                                                    num_trials=num_trials,
                                                    call_before_time_step=call_before_time_step,
                                                    call_after_time_step=call_after_time_step,
                                                    call_before_pass=call_before_pass,
                                                    call_after_pass=call_after_pass,
                                                    call_before_trial=call_before_trial,
                                                    call_after_trial=call_after_trial,
                                                    clamp_input=clamp_input,
                                                    targets=targets,
                                                    bin_execute=bin_execute,
                                                    initial_values=initial_values,
                                                    runtime_params=runtime_params)

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

    # gives user weights and biases of the model (from the pytorch representation)
    def get_parameters(self, execution_id=NotImplemented):
        if execution_id is NotImplemented:
            execution_id = self.default_execution_id

        pytorch_representation = self.parameters.pytorch_representation.get(execution_id)

        if pytorch_representation is None:
            raise AutodiffCompositionError("{0} has not been run yet so parameters have not been created "
                                           "in Pytorch."
                                           .format(self.name))

        weights = pytorch_representation.get_weights_for_projections()
        biases = pytorch_representation.get_biases_for_mechanisms()

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


