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
add methods of its parent class `Composition`. The most unusual argument in initialization is
**param_init_from_pnl**, which controls how parameters are set up for the internal PyTorch representation of the model.

If set to True:

* Only weight parameters that correspond to projections are created. No trainable bias parameters are created, as they
    don’t exist for the autodiff composition’s mechanisms.

* The weight parameters are initialized to be perfectly identical to the autodiff composition’s projections - the
    tensor of the parameter object corresponding to a particular projection not only has the same dimensionality as
    the projection’s matrix, it has the same exact values.

* Pytorch functions representing mechanism functions incorporate their scalar, untrainable biases.

If set to False:

* Both weight parameters corresponding to projections and trainable bias parameters for mechanisms are created.

* Weight parameters have the same dimensionality as their corresponding projections. However, their values - and those
    of the bias parameters - are sampled from a random distribution.

* Though trainable biases now exist, Pytorch functions representing mechanism functions still incorporate their scalar,
    untrainable biases.

.. warning:: Do not add or remove Mechanisms or Projections to an AutodiffComposition after it has been run for the
    first time. Unlike an ordinary Composition, AutodiffComposition does not support this functionality.

Two other initialization arguments are **patience** and **min_delta**, allow the model to halt training early. The
model tracks how many consecutive 'bad' epochs of training have failed to significantly reduce the model's loss. Once
this number exceeds **patience**, the model stops training. By default, **patience** is ``None``, and the model
will train for the number of specified epochs and will not stop training early.

**min_delta** defines what threshold counts as a significant reduction in model loss. By default it is zero, in which
case any reduction in loss counts as a significant reduction. If **min_delta** is large and positive, the model tends to
stop earlier because it views fewer epochs as 'good'.

**learning_rate** specifies the learning rate for this run (default 0.001), which is passed to the **optimizer**
argument. **optimizer** specifies the kind of optimizer used in training. The current options are 'sgd' (the default)
or 'adam'.

**learning_enabled** specifies whether the AutodiffComposition should learn, and it defaults to True. When True, the
AutodiffComposition trains using PyTorch, as normal. When False, the AutodiffComposition acts like an ordinary
Composition, which does not change weights. `learning_enabled <AutodiffComposition.learning_enabled>` is also an
attribute, which can be toggled between runs.

**optimizer_type** specifies the kind of optimizer used in training. The current options are 'sgd' (which is the
default) or 'adam'.

**weight_decay** specifies the L2 penalty (which discourages large weights) used by the optimizer. This defaults to 0.

**loss_spec** specifies the loss function for training. It can be a string or a PyTorch loss function. The current
options for strings are 'mse' (the default), 'crossentropy', 'l1', 'nll', 'poissonnll', and 'kldiv'. These refer to
Mean Squared Error, Cross Entropy, L1 loss, Negative Log Likelihood loss, Poisson Negative Log Likelihood, and KL
Divergence respectively. The **loss_spec** can also be any PyTorch loss function, including a custom-written one. For a
list of PyTorch loss functions, see https://pytorch.org/docs/stable/nn.html#loss-functions. For information on writing
a custom loss function, see https://pytorch.org/docs/master/notes/extending.html and
https://discuss.pytorch.org/t/build-your-own-loss-function-in-pytorch/235

**randomize** specifies whether the order of inputs will be randomized in each epoch. (In each epoch, all inputs are
run, but if **randomize** is True then the order in which inputs are within an epoch is random.)

**refresh_losses** specifies whether the `losses` attribute is refreshed for each call to `run()`. If False, the losses
of each run are appended to the `losses` attribute. If True, the losses of each run overwrite `losses` instead.

**force_no_retain_graph** defaults to False. If True, the AutodiffComposition does not use the `retain_graph` option
when computing PyTorch gradient. This can reduce memory usage. However, it breaks recurrent networks, so it should only
be used when the network is not recurrent.

.. note::
    The AutodiffComposition detachs all gradients between epochs of training. For more information on why this is done,
    see `here <bit.ly/2t2ZkyR>` or `here <bit.ly/2RGuMNg>`.

.. _AutodiffComposition_Structure:

Structure
---------

AutodiffComposition has all the attributes of its parent class `Composition`, in addition to several more.

The `target_CIM <AutodiffComposition.target_CIM>` attribute is analogous to the `input_CIM <Composition.input_CIM>` of
any Composition, but instead of providing inputs, provides targets for the AutodiffComposition.

The `pytorch_representation <AutodiffComposition.pytorch_representation>` attribute holds the PyTorch representation
of the PsyNeuLink model that AutodiffComposition contains.

The `losses <AutodiffComposition.losses>` attribute tracks the average loss for each training epoch.

As mentioned above, the `learning_enabled <AutodiffComposition.learning_enabled>` attribute can be toggled to determine
whether the AutodiffComposition learns or whether it executes like an ordinary Composition.

The `optimizer <AutodiffComposition.optimizer>` attribute contains the PyTorch optimizer function used for learning. It
is determined at initialization by the **optimizer_type**, **learning_rate**, and **weight_decay** arguments.

The `loss <AutodiffComposition.loss>` attribute contains the PyTorch loss function used for learning. It is determined
at initialization by the **loss_spec** argument.

.. _AutodiffComposition_Execution:

Execution
---------

Most arguments to AutodiffComposition's `run` or `execute` methods are the same as in a Composition. When
`learning_enabled <AutodiffComposition.learning_enabled>` is False, the arguments are the same, since in this
case the AutodiffComposition executes like a Composition.

However, if `learning_enabled <AutodiffComposition.learning_enabled>` is True, the **inputs** argument
format is different. If `learning_enabled <AutodiffComposition.learning_enabled>` is True, then **inputs** should be a
dictionary with required keys "inputs" and "targets", and optional key "epochs". The value at "inputs" should be a
dictionary relating origin mechanisms to their inputs. The value at "targets" should be a dictionary relating terminal
mechanisms to their inputs. The value at "epochs" is an integer stating the number of epochs of training (i.e. how many
times all inputs and targets are run). It defaults to 1. Here is an example of creating a simple AutodiffComposition
and specifying inputs and targets:

    >>> import psyneulink as pnl
    >>> # set up PsyNeuLink Components
    >>> my_mech_1 = pnl.TransferMechanism(function=pnl.Linear, size = 3)
    >>> my_mech_2 = pnl.TransferMechanism(function=pnl.Linear, size = 2)
    >>> my_projection = pnl.MappingProjection(matrix=np.random.randn(3,2),
    ...                     sender=my_mech_1,
    ...                     receiver=my_mech_2)
    >>> # create AutodiffComposition
    >>> my_autodiff = pnl.AutodiffComposition()
    >>> my_autodiff.add_node(my_mech_1)
    >>> my_autodiff.add_node(my_mech_1)
    >>> my_autodiff.add_projection(sender=my_mech_1, projection=my_projection, receiver=my_mech_2)
    >>> # input specification
    >>> my_inputs = {my_mech_1: [[1, 2, 3]]}
    >>> my_targets = {my_mech_2: [[4, 5]]}
    >>> input_dict = {"inputs": my_inputs, "targets": my_targets, "epochs": 2}
    >>> my_autodiff.run(inputs = input_dict)

Logging
-------

Logging currently works differently in AutodiffComposition than in Composition. In an AutodiffComposition, no logging
is done by default, because logging substantially (roughly by 30%) slows down AutodiffComposition. If you wish for all
projection weights and mechanism values to be logged during execution or training of AutodiffComposition, you must
set the **do_logging** argument of the ``run()`` method to ``True``. Logging with AutodiffComposition is slightly hacked
together, so the time and context in the log are not meaningful, only the logged value is meaningful.

Nested Execution
----------------
COMMENT:
    Need to add link to docs about nesting ordinary Compositions, once those docs are written.
COMMENT
In general, an AutodiffComposition may be nested inside another Composition, like ordinary Composition nesting. However,
there are a few differences. The input format of an AutodiffComposition with learning enabled is quite unusual. Thus,
when learning is enabled, the AutodiffComposition must be an origin mechanism of the Composition.

.. note::

    Like with all nested Compositions, you must call an AutodiffComposition's ``_analyze_graph()`` method
    (or execute the AutodiffComposition) before nesting it.

However, when learning is not enabled, AutodiffComposition works just like an ordinary Composition, in theory. Thus, an
AutodiffComposition with learning not enabled receives input in the same format as an ordinary Composition, and can
therefore be placed anywhere in a Composition.

.. note::

    Using an AutodiffComposition not as an origin mechanism is currently buggy, and might produce unexpected results.

Below is an example script showing how to nest an AutodiffComposition with learning enabled.

    >>> import psyneulink as pnl
    >>> # set up PsyNeuLink Components
    >>> my_mech_1 = pnl.TransferMechanism(function=pnl.Linear, size = 3)
    >>> my_mech_2 = pnl.TransferMechanism(function=pnl.Linear, size = 2)
    >>> my_projection = pnl.MappingProjection(matrix=np.random.randn(3,2),
    ...                     sender=my_mech_1,
    ...                     receiver=my_mech_2)
    >>> # create AutodiffComposition
    >>> my_autodiff = pnl.AutodiffComposition()
    >>> my_autodiff.add_node(my_mech_1)
    >>> my_autodiff.add_node(my_mech_1)
    >>> my_autodiff.add_projection(sender=my_mech_1, projection=my_projection, receiver=my_mech_2)
    >>> my_autodiff._analyze_graph()  # alternatively, my_autodiff.run( ... )
    >>>
    >>> # input specification
    >>> my_inputs = {my_mech_1: [[1, 2, 3]]}
    >>> my_targets = {my_mech_2: [[4, 5]]}
    >>> input_dict = {"inputs": my_inputs, "targets": my_targets, "epochs": 2}
    >>>
    >>> parentComposition = pnl.Composition()
    >>> parentComposition.add_node(my_autodiff)
    >>>
    >>> training_input = {my_autodiff: input_dict}
    >>> result1 = parentComposition.run(inputs=input)
    >>>
    >>> my_autodiff.learning_enabled = False
    >>> no_training_input = {my_autodiff: my_inputs}
    >>> result2 = parentComposition.run(inputs=no_training_input)


.. _Composition_Class_Reference:

Class Reference
---------------

"""

from psyneulink.core.components.functions.transferfunctions import Linear, Logistic, ReLU
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.compositions.composition import CompositionError
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import SOFT_CLAMP
from psyneulink.core.scheduling.scheduler import Scheduler

import numpy as np
import copy

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
    learning_rate=0.001,            \
    learning_enabled=True,          \
    optimizer_type=None,            \
    loss_spec=None,                 \
    randomize=False,                \
    refresh_losses=False,           \
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

    learning_rate : float : default 0.001
        the learning rate, which is passed to the optimizer.

    learning_enabled : boolean : default True
        specifies whether the AutodiffComposition should learn. When True, the AutodiffComposition trains using PyTorch.
        When False, the AutodiffComposition executes just like an ordinary Composition

    optimizer_type : str : default 'sgd'
        the kind of optimizer used in training. The current options are 'sgd' or 'adam'.

    weight_decay : float : default 0
        specifies the L2 penalty (which discourages large weights) used by the optimizer.

    loss_spec : str or PyTorch loss function : default 'mse'
        specifies the loss function for training. The current string options are 'mse' (the default), 'crossentropy',
        'l1', 'nll', 'poissonnll', and 'kldiv'. Any PyTorch loss function can work here, such as ones from
        https://pytorch.org/docs/stable/nn.html#loss-functions

    randomize: boolean : default False
        specifies whether the order of inputs will be randomized in each epoch. (In each epoch, all inputs are run, but
        if **randomize** is True then the order of inputs within an epoch is random.)

    refresh_losses : boolean: default False
        specifies whether the `losses` attribute is refreshed for each call to `run()`. If False, the losses of each run
        are appended to the `losses` attribute. If True, the losses of each run overwrite `losses` instead.

    Attributes
    ----------

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

    learning_enabled : boolean : default True
        specifies whether the AutodiffComposition should learn. When True, the AutodiffComposition trains using PyTorch.
        When False, the AutodiffComposition executes just like an ordinary Composition. This attribute can be toggled.

    learning_rate : float: default 0.001
        the learning rate for training. Currently only used to initialize the `optimizer` attribute.

    optimizer : PyTorch optimizer function
        the optimizer used for training. Depends on the **optimizer_type**, **learning_rate**, and **weight_decay**
        arguments from initialization.

    loss : PyTorch loss function
        the loss function used for training. Depends on the **loss_spec** argument from initialization.

    name : str : default LeabraMechanism-<index>
        the name of the Mechanism.
        Specified in the **name** argument of the constructor for the Projection;
        if not specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    Returns
    -------
    instance of AutodiffComposition : AutodiffComposition
    """

    class Parameters(Composition.Parameters):
        """
            Attributes
            ----------

                learning_rate
                    see `learning_rate <AutodiffComposition.learning_rate>`

                    :default value: 0.001
                    :type: float

                losses
                    see `losses <AutodiffComposition.losses>`

                    :default value: None
                    :type:

                min_delta
                    see `min_delta <AutodiffComposition.min_delta>`

                    :default value: 0
                    :type: int

                optimizer
                    see `optimizer <AutodiffComposition.optimizer>`

                    :default value: None
                    :type:

                patience
                    see `patience <AutodiffComposition.patience>`

                    :default value: None
                    :type:

                pytorch_representation
                    see `pytorch_representation <AutodiffComposition.pytorch_representation>`

                    :default value: None
                    :type:

        """
        optimizer = None
        learning_rate = .001
        losses = None
        patience = None
        min_delta = 0
        pytorch_representation = None

    # TODO (CW 9/28/18): add compositions to registry so default arg for name is no longer needed
    def __init__(self,
                 param_init_from_pnl=True,
                 patience=None,
                 min_delta=0,
                 learning_rate=0.001,
                 learning_enabled=True,
                 optimizer_type='sgd',
                 weight_decay=0,
                 loss_spec='mse',
                 randomize=None,
                 refresh_losses=False,
                 disable_cuda=False,
                 cuda_index=None,
                 force_no_retain_graph=False,
                 name="autodiff_composition"):

        self.learning_enabled = True
        if not torch_available:
            raise AutodiffCompositionError('Pytorch python module (torch) is not installed. Please install it with '
                                           '`pip install torch` or `pip3 install torch`')

        # params = self._assign_args_to_param_dicts(learning_rate=learning_rate)

        # since this does not pass params argument, defaults will not be automatically set..
        super(AutodiffComposition, self).__init__(name=name)
        # super(AutodiffComposition, self).__init__(params=params, name=name)

        self.learning_enabled = learning_enabled
        self.optimizer_type = optimizer_type
        self.loss_spec = loss_spec
        self.randomize = randomize
        self.refresh_losses = refresh_losses

        # pytorch representation of model and associated training parameters
        self.pytorch_representation = None
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = None
        self.loss = None
        self.force_no_retain_graph = force_no_retain_graph

        # user indication of how to initialize pytorch parameters
        self.param_init_from_pnl = param_init_from_pnl

        # keeps track of average loss per epoch
        self.losses = []

        # ordered execution sets for the pytorch model
        self.execution_sets = None

        # patience is the "bad" epochs (with no progress in average loss) the model tolerates in one training session
        # before ending training
        self.patience = patience

        self.min_delta = min_delta

        # CW 11/1/18: maybe we should make scheduler a property, like in Composition
        self.scheduler = None
        if not disable_cuda and torch.cuda.is_available():
            if cuda_index is None:
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cuda:' + cuda_index)
        else:
            self.device = torch.device('cpu')

    # CLEANUP: move some of what's done in the methods below to a "validate_params" type of method
    def _build_pytorch_representation(self, execution_id = None):
        if self.scheduler is None:  # if learning_enabled has never been run yet
            self.scheduler = Scheduler(graph=self.graph_processing)
        if self.execution_sets is None:
            self.execution_sets = list(self.scheduler.run())
        if self.parameters.pytorch_representation.get(execution_id) is None:
            model = PytorchModelCreator(self.graph_processing,
                                        self.param_init_from_pnl,
                                        self.execution_sets,
                                        self.device,
                                        execution_id)
            self.parameters.pytorch_representation.set(model, execution_id)

        # Set up optimizer function
        old_opt = self.parameters.optimizer.get(execution_id)
        if old_opt is not None:
            logger.warning("Overwriting optimizer for AutodiffComposition {}! Old optimizer: {}".format(
                self, old_opt))
        opt = self._make_optimizer(self.optimizer_type, self.learning_rate, self.weight_decay, execution_id)
        self.parameters.optimizer.set(opt, execution_id)

        # Set up loss function
        if self.loss is not None:
            logger.warning("Overwriting loss function for AutodiffComposition {}! Old loss function: {}".format(
                self, self.loss))
        self.loss = self._get_loss(self.loss_spec)

    def _make_optimizer(self, optimizer_type, learning_rate, weight_decay, execution_id):
        if not isinstance(learning_rate, (int, float)):
            raise AutodiffCompositionError("Learning rate must be an integer or float value.")
        if optimizer_type not in ['sgd', 'adam']:
            raise AutodiffCompositionError("Invalid optimizer specified. Optimizer argument must be a string. "
                                           "Currently, Stochastic Gradient Descent and Adam are the only available "
                                           "optimizers (specified as 'sgd' or 'adam').")
        params = self.parameters.pytorch_representation.get(execution_id).parameters()
        if optimizer_type == 'sgd':
            return optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            return optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

    def _get_loss(self, loss_spec):
        if not isinstance(self.loss_spec, str):
            return self.loss_spec
        elif loss_spec == 'mse':
            return nn.MSELoss(reduction='sum')
        elif loss_spec == 'crossentropy':
            return nn.CrossEntropyLoss(reduction='sum')
        elif loss_spec == 'l1':
            return nn.L1Loss(reduction='sum')
        elif loss_spec == 'nll':
            return nn.NLLLoss(reduction='sum')
        elif loss_spec == 'poissonnll':
            return nn.PoissonNLLLoss(reduction='sum')
        elif loss_spec == 'kldiv':
            return nn.KLDivLoss(reduction='sum')
        else:
            raise AutodiffCompositionError("Loss type {} not recognized. Loss argument must be a string or function. "
                                           "Currently, the recognized loss types are Mean Squared Error, Cross Entropy,"
                                           " L1 loss, Negative Log Likelihood loss, Poisson Negative Log Likelihood, "
                                           "and KL Divergence. These are specified as 'mse', 'crossentropy', 'l1', "
                                           "'nll', 'poissonnll', and 'kldiv' respectively.".format(loss_spec))

    def _has_required_keys(self, input_dict):
        required_keys = {"inputs", "targets"}
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
    def autodiff_processing(self, inputs, execution_id=None, do_logging=False):
        pytorch_representation = self.parameters.pytorch_representation.get(execution_id)
        # run the model on inputs - switch autograd off for this (we don't need it)
        with torch.no_grad():
            tensor_outputs = pytorch_representation.forward(inputs, execution_id=execution_id, do_logging=do_logging)

        # get outputs back into numpy
        outputs = []
        for i in range(len(tensor_outputs)):
            outputs.append(tensor_outputs[i].numpy().copy())

        return outputs

    # performs learning/training on all input-target pairs it recieves for given number of epochs
    def autodiff_training(self, inputs, targets, epochs, execution_id=None, do_logging=False):

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

            self.parameters.pytorch_representation.get(execution_id).detach_all()
            # self.parameters.pytorch_representation.get(execution_id).reset_all()

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
                    curr_tensor_inputs[component] = torch.tensor(input, device=self.device).double()
                for component in targets.keys():
                    target = targets[component][input_index]
                    curr_tensor_targets[component] = torch.tensor(target, device=self.device).double()

                # do forward computation on current inputs
                curr_tensor_outputs = self.parameters.pytorch_representation.get(execution_id).forward(
                    curr_tensor_inputs,
                    execution_id,
                    do_logging
                )

                # compute total loss across output neurons for current trial
                curr_loss = torch.zeros(1).double()
                for component in curr_tensor_outputs.keys():
                    # possibly add custom loss option, which is a loss function that takes many args
                    # (outputs, targets, weights, and more) and returns a scalar
                    curr_loss += self.loss(curr_tensor_outputs[component], curr_tensor_targets[component])

                # save average loss across all output neurons on current trial
                curr_losses[t] = (curr_loss[0].item())/out_size

                optimizer = self.parameters.optimizer.get(execution_id)

                # backpropagate to compute gradients and perform learning update for parameters
                optimizer.zero_grad()
                curr_loss = curr_loss/2
                if self.force_no_retain_graph:
                    curr_loss.backward(retain_graph=False)
                else:
                    curr_loss.backward(retain_graph=True)
                optimizer.step()

                # save outputs of model if this is final epoch
                curr_output_list = []
                for input_state in self.output_CIM.input_states:
                    assert(len(input_state.all_afferents) == 1)  # CW 12/05/18, this assert may eventually be outdated
                    component = input_state.all_afferents[0].sender.owner
                    curr_output_list.append(curr_tensor_outputs[component].detach().numpy().copy())
                # for component in curr_tensor_outputs.keys():
                #     curr_output_list.append(curr_tensor_outputs[component].detach().numpy().copy())
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
                do_logging=False,
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
                skip_initialization=False,
                bin_execute=False,
                context=None
                ):
        execution_id = self._assign_execution_ids(execution_id)
        if self.learning_enabled:
            # TBI: How are we supposed to use base_execution_id and statefulness here?
            # TBI: can we call _build_pytorch_representation in _analyze_graph so that pytorch
            # model may be modified between runs?

            self._analyze_graph()  # ADDED by CW 12/17/18: unsure if correct here

            self._build_pytorch_representation(execution_id)

            autodiff_inputs = inputs["inputs"]
            autodiff_targets = inputs["targets"]
            autodiff_epochs = 1
            if "epochs" in inputs:
                autodiff_epochs = inputs["epochs"]

            output = self.autodiff_training(autodiff_inputs, autodiff_targets, autodiff_epochs, execution_id, do_logging)
            ctx = self.output_CIM.parameters.context.get(execution_id)
            # new_ctx = copy.deepcopy(ctx)
            # new_ctx.execution_phase = ContextFlags.PROCESSING
            # self.output_CIM.parameters.context.set(new_ctx, execution_id=execution_id)
            if ctx is not None:  # HACK: CW 12/18/18 for some reason context isn't set correctly
                ctx.execution_phase = ContextFlags.PROCESSING
            # note that output[-1] might not be the truly most recent value
            # HACK CW 2/5/19: the line below is a hack. In general, the output_CIM of an AutodiffComposition
            # is not having its parameters populated correctly, and this should be fixed in the long run.
            self.output_CIM.execute(input=output[-1], execution_id=execution_id, context=ContextFlags.PROCESSING)

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
                                                        skip_initialization=skip_initialization,
                                                        bin_execute=bin_execute,
                                                        context=context)

    # what the user calls for doing processing/training, similar to the run function of the normal composition
    def run(
        self,
        inputs=None,
        do_logging=False,
        scheduler_processing=None,
        scheduler_learning=None,
        termination_processing=None,
        termination_learning=None,
        execution_id=None,
        num_trials=1,
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
        reinitialize_values=None,
        runtime_params=None,
        context=None):
        # TBI: Handle trials, timesteps, etc
        execution_id = self._assign_execution_ids(execution_id)
        if self.learning_enabled:

            self._analyze_graph()

            if self.refresh_losses or (self.parameters.losses.get(execution_id) is None):
                self.parameters.losses.set([], execution_id)
            adjusted_stimuli = self._adjust_stimulus_dict(inputs)
            if num_trials is None:
                num_trials = len(adjusted_stimuli)

            results = []
            for trial_num in range(num_trials):
                stimulus_index = trial_num % len(adjusted_stimuli)
                trial_output = self.execute(
                    inputs=adjusted_stimuli[stimulus_index],
                    execution_id=execution_id,
                    do_logging=do_logging,
                )
                results.append(trial_output)
            return results

        else:
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
                                                    reinitialize_values=reinitialize_values,
                                                    runtime_params=runtime_params,
                                                    context=context)

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
            if not isinstance(node.component.function, (Linear, Logistic, ReLU)):
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

