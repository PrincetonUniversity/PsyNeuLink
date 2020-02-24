# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* AutodiffComposition *************************************************

"""

Contents
--------

  * `AutodiffComposition_Overview`
  * `AutodiffComposition_Creation`
  * `AutodiffComposition_Structure`
  * `AutodiffComposition_Execution`
      - `Input <AutodiffComposition_Input>`
      - `Input <AutodiffComposition_Input>`
      - `AutodiffComposition_Logging`
      - `AutodiffComposition_Nested_Execution`
  * `AutodiffComposition_Class_Reference`


.. _AutodiffComposition_Overview:

Overview
--------

AutodiffComposition is a subclass of `Composition` used to train feedforward neural network models through integration
with `PyTorch <https://pytorch.org/>`_, a popular machine learning library, which executes considerably more quickly
than using the `standard implementation of learning <Composition_Learning_Standard>` in a Composition, using its
`learning methods <Composition_Learning_Methods>`. An AutodiffComposition is configured and run similarly to a standard
Composition, with some exceptions that are described below.  An example is provided in the
`xor_in_psyneulink_and_pytorch.py` script (in the Scripts folder of the PsyNeuLink source code), which also provides
a comparison of runtimes.
COMMENT:
FIX: UPDATE runtimes WITH COMPILED VERSION
COMMENT

.. _AutodiffComposition_Creation:

Creating an AutodiffComposition
-------------------------------

An AutodiffComposition can be created by calling its constructor, and then adding `Components <Component>` using the
standard `Composition methods <Composition_Creation>` for doing so.  The constructor also includes an number of
arguments that are specific to the AutodiffComposition, as described below.

.. warning:: Mechanisms or Projections should not be added to or deleted from an AutodiffComposition after it has
   been run for the first time. Unlike an ordinary Composition, AutodiffComposition does not support this
   functionality.

.. warning:: When comparing models built in PyTorch to those using AutodiffComposition, the `bias <https://www.pytorch.org/docs/stable/nn.html#torch.nn.Module>` parameter of PyTorch modules should be set to `False`, as AutodiffComposition does not currently support trainable biases.

* **param_init_from_pnl** argument -- determines how parameters are set up for the internal PyTorch representation of
  the model.  If it is set to True:

    COMMENT:
    FIX: THIS SHOULD BE ADDRESSED
    COMMENT
    * only weight parameters that correspond to the `value <Parameter.value>` of the `matrix <MappingProjection.matrix>`
      parameter of the `MappingProjections <MappingProjection>` in the Composition are created;  no bias parameters are
      created, as the bias parameters associated with Mechanisms are not trainable;

    * the weight parameters are initialized to be identical to the `value <Parameter.value>` of `matrix
      <MappingProjection.matrix>` parameters of the `MappingProjections <MappingProjection>` in the Composition;
      the tensor of the parameter object corresponding to a particular MappingProjection not only has the same
      dimensionality as its `matrix <MappingProjection.matrix>`, it also has the exact same values;

    * Pytorch functions representing the `function <Mechanism_Base.function>` of each `Mechanism <Mechanism>` in the
      Composition incorporate their scalar, untrainable biases.

    If it is set to False:

        * weight parameters have the same dimensionality as the `matrix <MappingProjection.matrix>` parameter of the
          corresponding `MappingProjections <MappingProjection>`;  however, their values -- and those of the bias
          parameters -- are sampled from a random distribution;

* **patience** -- allows the model to halt training early. The  model tracks how many consecutive 'bad' epochs of
  training have failed to significantly reduce the model's loss. When this number exceeds **patience**, the model stops
  training. By default, **patience** is ``None``, and the model will train for the number of specified epochs and
  will not stop training early.

* **min_delta** -- specifies the threshold used by **patience** used to determine a significant reduction in model
  loss. By default it is zero, in which case any reduction in loss counts as a significant reduction. If **min_delta**
  is large and positive, the model tends to stop earlier because it views fewer epochs as 'good'.

* **learning_rate** -- specifies the learning rate for the current run (default 0.001), which is passed to the
  optimized specified in the **optimizer** argument.

* **optimizer** -- specifies the kind of optimizer used in training. The current options are 'sgd' (the default) or
  'adam'.

* **optimizer_type** -- specifies the kind of optimizer used in training. The current options are 'sgd' (which is the
  default) or 'adam'.

* **learning_enabled** -- specifies whether the AutodiffComposition should learn (default is True). When True,
  the AutodiffComposition trains using PyTorch, as normal. When False, the AutodiffComposition run like an ordinary
  `Composition`, which does not change weights. `learning_enabled <AutodiffComposition.learning_enabled>` is also an
  attribute, which can be toggled between runs.

* **weight_decay** -- specifies the L2 penalty (which discourages large weights) used by the optimizer. This defaults
  to 0.

* **loss_spec** -- specifies the loss function for training. It can be a string or a PyTorch loss function. The current
  options for strings are 'mse' (the default), 'crossentropy', 'l1', 'nll', 'poissonnll', and 'kldiv'. These refer to
  Mean Squared Error, Cross Entropy, L1 loss, Negative Log Likelihood loss, Poisson Negative Log Likelihood, and KL
  Divergence respectively. The **loss_spec** can also be any PyTorch loss function, including a custom-written one.
  For a list of PyTorch loss functions, see `Loss function <https://pytorch.org/docs/stable/nn.html#loss-functions>`_.
  For information on writing a custom loss function, see `Extending PyTorch
  <https://pytorch.org/docs/master/notes/extending.html>`_, as well as `Build your own loss function in PyTorch
  <https://discuss.pytorch.org/t/build-your-own-loss-function-in-pytorch/235>`_.

* **randomize** -- specifies whether the order of inputs will be randomized in each epoch. All inputs are run in each
  epoch.  However, if **randomize** is True, then the order in which inputs are within an epoch is random.

* **refresh_losses** -- specifies whether the `losses` attribute is refreshed for each call to `run
  <AutodiffComposition.run>`. If False, the losses of each run are appended to the `losses
  <AutodiffComposition.losses>` attribute. If True, the losses of each run overwrite `losses
  <AutodiffComposition.losses>` instead.

* **force_no_retain_graph** -- False by default.  If True, the AutodiffComposition does not use PyTorch's `retain_graph
  <https://pytorch.org/docs/master/autograd.html?highlight=retain_graph>`_  option when computing the gradient. This
  can reduce memory usage; however, it breaks recurrent networks, so it should only be used when the network is not
  recurrent.

.. note::
    The AutodiffComposition detachs all gradients between epochs of training. For more information on why this is done,
    see `Trying to backward through a graph a second time <https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time/6795>`_
    and `Why we need to detach Variable which contains [a] hidden representation
    <https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426/4>`_.

.. _AutodiffComposition_Structure:

Structure
---------

An AutodiffComposition has all the attributes of its parent class `Composition`, as well ones corresponding to the
arguments described above, and the following.

* `target_CIM <AutodiffComposition.target_CIM>` -- analogous to the `input_CIM <Composition.input_CIM>` of
  a standard `Composition`, but instead of representing the input to the `ORIGIN` Mechanism(s) of the Composition
  represents the targets used to specify the target `value <Mechanism_Base.value>` of its `TERMINAL` Mechanism(s)
  (see `below <AutodiffComposition_Input>` for a description of how these are specified).

* `pytorch_representation <AutodiffComposition.pytorch_representation>` -- containsa the PyTorch representation
  of the PsyNeuLink model that AutodiffComposition contains.

* `losses <AutodiffComposition.losses>` -- tracks the average loss for each training epoch.

* `optimizer <AutodiffComposition.optimizer>` -- contains the PyTorch optimizer function used for learning. It
  is determined at initialization by the **optimizer_type**, **learning_rate**, and **weight_decay** arguments.

* `loss <AutodiffComposition.loss>` -- contains the PyTorch loss function used for learning. It is determined
  at initialization by the **loss_spec** argument.

.. _AutodiffComposition_Execution:

Execution
---------

Most arguments to AutodiffComposition's `run` or `execute` methods are the same as for a `Composition`. When
`learning_enabled <AutodiffComposition.learning_enabled>` is False, the arguments are the same, since in this
case the AutodiffComposition executes like a Composition.

.. _AutodiffComposition_Input:

However, if `learning_enabled <AutodiffComposition.learning_enabled>` is True, the **inputs** argument
format is different. If `learning_enabled <AutodiffComposition.learning_enabled>` is True, then **inputs**
argument must be a dictionary with at least two nested dictionaries within it, one for the inputs and the other
for the targets, as well as an additional entry specifying the number of training epochs to run.  Specifically,
the outer dictionary must have at least two entries with keys *"inputs"* and  *"targets"*.  The value of the
*"inputs"* entry must be a standard input dictionary, specifying the inputs for each `ORIGIN` Mechanism.  The value
of the *"targets"* entry must be a similar dictionary, in this case specifying the target values for the outputs of
each `TERMINAL` Mechanism in the Composition. In addition, an entry with the key *"epochs"* can be included, which
must then have as its value an integer specifying the number of epochs of training to run (i.e. how many times all
inputs and corresponding targets are run); it defaults to 1. The following is an example showing how to create a
simple AutodiffComposition, specify its inputs and targets, and run it with learning enabled and disabled.

    >>> import psyneulink as pnl
    >>> # Set up PsyNeuLink Components
    >>> my_mech_1 = pnl.TransferMechanism(function=pnl.Linear, size = 3)
    >>> my_mech_2 = pnl.TransferMechanism(function=pnl.Linear, size = 2)
    >>> my_projection = pnl.MappingProjection(matrix=np.random.randn(3,2),
    ...                     sender=my_mech_1,
    ...                     receiver=my_mech_2)
    >>> # Create AutodiffComposition
    >>> my_autodiff = pnl.AutodiffComposition()
    >>> my_autodiff.add_node(my_mech_1)
    >>> my_autodiff.add_node(my_mech_2)
    >>> my_autodiff.add_projection(sender=my_mech_1, projection=my_projection, receiver=my_mech_2)
    >>> # Specify inputs and targets
    >>> my_inputs = {my_mech_1: [[1, 2, 3]]}
    >>> my_targets = {my_mech_2: [[4, 5]]}
    >>> input_dict = {"inputs": my_inputs, "targets": my_targets, "epochs": 2}
    >>> # Run Composition with learning enabled
    >>> my_autodiff.learning_enabled=True # this is not strictly necessary, as learning_enabled is True by default
    >>> my_autodiff.run(inputs = input_dict)
    >>> # Run Composition with learning disabled
    >>> my_autodiff.learning_enabled=False
    >>> my_autodiff.run(inputs = input_dict)

As shown above (and for convenience), an AutodiffComposition with learning disabled can be run with the same input
format used for training.  In that case, the *"input"* entry is used as the inputs for the run, and the *"targets"*
and *"epochs"* entries (if present) are ignored. However, since an AutodiffComposition with learning disabled is
treated like any other Composition, it can also be run with the same `input format <Composition_Run_Inputs>` as a standard
`Composition`; that is, a single dictionary specifying the inputs for each `ORIGIN` Mechanism), such the one defined
in the exaple above, as follows::

    >>> my_autodiff.run(inputs = my_inputs)

or `using a function <Composition_Input_as_Function>`.

.. _AutodiffComposition_Logging:

Logging
~~~~~~~

Logging currently works differently in AutodiffComposition than in Composition. In an AutodiffComposition, no logging
is done by default, because logging substantially (roughly by 30%) slows down AutodiffComposition. If you wish for all
projection weights and mechanism values to be logged during execution or training of AutodiffComposition, you must
set the **do_logging** argument of the ``run()`` method to ``True``. Logging with AutodiffComposition is slightly hacked
together, so the time and context in the log are not meaningful, only the logged value is meaningful.

.. _AutodiffComposition_Nested_Execution:

Nested Execution
~~~~~~~~~~~~~~~~

Like any other `Composition`, an AutodiffComposition may be `nested inside another <Composition_Nested>`.  If learning
is not enabled, nesting is handled in the same way as any other Composition.  However, if learning is enabled for a
nested AutodiffComposition, its input format is different (see below);  as a consequence, a nested AutodiffComposition
with learning enabled must an `ORIGIN` Mchanism of the Composition in which it is nested.

.. note::

    As with all `nested Compositions <Composition_Nested>`, the AutodiffComposition's `_analyze_graph
    <Composition._analyze_graph>` method must be called (or the AutodiffComposition must be run) before nesting it.

COMMENT:
FIX:  IS THIS STILL TRUE:
.. note::

    Using an AutodiffComposition not as an origin mechanism is currently buggy, and might produce unexpected results.
COMMENT

The following shows how the AutodiffComposition created in the previous example can be nested and run inside another
Composition::

    >>> my_autodiff._analyze_graph()  # alternatively, my_autodiff.run( ... )
    >>>
    >>> # Create outer composition
    >>> my_outer_composition = pnl.Composition()
    >>> my_outer_composition.add_node(my_autodiff)
    >>> # Specify dict containing inputs and targets for nested Composition
    >>> training_input = {my_autodiff: input_dict}
    >>> # Run with learning enabled
    >>> result1 = my_outer_composition.run(inputs=training_input)
    COMMENT:
    >>> # Run with learning disabled (and standard input format)
    >>> my_autodiff.learning_enabled = False
    >>> no_training_input = {my_autodiff: my_inputs}
    >>> result2 = parentmy_outer_compositionComposition.run(inputs=no_training_input)
    COMMENT

.. _AutodiffComposition_Class_Reference:

Class Reference
---------------

"""

from psyneulink.core.components.functions.transferfunctions import Linear, Logistic, ReLU
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.compositions.composition import CompositionError
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import SOFT_CLAMP, TRAINING_SET
from psyneulink.core.globals.utilities import NodeRole
from psyneulink.core.scheduling.scheduler import Scheduler
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.core import llvm as pnlvm
import copy
import numpy as np
import ctypes
import warnings
from collections.abc import Iterable
from toposort import toposort
from inspect import isgenerator

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
    AutodiffComposition(             \
        param_init_from_pnl=True,    \
        patience=None,               \
        min_delta=0,                 \
        learning_rate=0.001,         \
        learning_enabled=True,       \
        optimizer_type=None,         \
        loss_spec=None,              \
        randomize=False,             \
        refresh_losses=False,        \
        name="autodiff_composition")

    Subclass of `Composition` that trains models using `PyTorch <https://pytorch.org>`_.
    See `Composition <Composition_Class_Reference>` for additional arguments and attributes.

    Arguments
    ---------

    param_init_from_pnl : boolean : default True
        a Boolean specifying how parameters are initialized. **WARNING: deprecated!** (See
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

    name : str : default AutodiffComposition-<index>
        the name of the Composition. Specified in the **name** argument of the constructor for the Projection;
        if not specified, a default is assigned by `CompositionRegistry` (see :doc:`Registry <LINK>` for conventions
        used in naming, including for default and duplicate names).

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
                    :type: ``float``

                losses
                    see `losses <AutodiffComposition.losses>`

                    :default value: None
                    :type:

                min_delta
                    see `min_delta <AutodiffComposition.min_delta>`

                    :default value: 0
                    :type: ``int``

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
        learning_rate = Parameter(.001, fallback_default=True)
        losses = None
        patience = None
        min_delta = 0
        pytorch_representation = None

    # TODO (CW 9/28/18): add compositions to registry so default arg for name is no longer needed
    def __init__(self,
                 param_init_from_pnl=True,
                 patience=None,
                 min_delta=0,
                 learning_rate=None,
                 optimizer_type='sgd',
                 weight_decay=0,
                 loss_spec='mse',
                 randomize=None,
                 refresh_losses=False,
                 disable_cuda=True,
                 cuda_index=None,
                 force_no_retain_graph=False,
                 name="autodiff_composition"):

        if not torch_available:
            raise AutodiffCompositionError('Pytorch python module (torch) is not installed. Please install it with '
                                           '`pip install torch` or `pip3 install torch`')

        super(AutodiffComposition, self).__init__(name = name,
                                                  patience = patience,
                                                  min_delta = min_delta,
                                                  learning_rate = learning_rate,
                                                  optimizer_type = optimizer_type,
                                                  weight_decay = weight_decay,
                                                  loss_spec = loss_spec,
                                                  randomize = randomize)

        self.optimizer_type = optimizer_type
        self.loss_spec = loss_spec
        self.randomize = randomize
        self.refresh_losses = refresh_losses

        self.weight_decay = weight_decay
        self.force_no_retain_graph = force_no_retain_graph
        self.loss = None

        # user indication of how to initialize pytorch parameters
        self.param_init_from_pnl = param_init_from_pnl
        
        if param_init_from_pnl is False:
            warnings.warn("WARNING: Autodiffcomposition.param_init_from_pnl is deprecated! Please do not use it!")
            
        # keeps track of average loss per epoch
        self.losses = []

        # ordered execution sets for the pytorch model
        self.execution_sets = None

        # patience is the "bad" epochs (with no progress in average loss) the model tolerates in one training session
        # before ending training
        self.patience = patience

        self.min_delta = min_delta

        if not disable_cuda and torch.cuda.is_available():
            if cuda_index is None:
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cuda:' + str(cuda_index))
        else:
            self.device = torch.device('cpu')

    # CLEANUP: move some of what's done in the methods below to a "validate_params" type of method
    @handle_external_context()
    def _build_pytorch_representation(self, context=None):
        if self.scheduler is None:  # if learning_enabled has never been run yet
            self.scheduler = Scheduler(graph=self.graph_processing)
        if self.execution_sets is None:
            self.execution_sets = [ x - set(self.get_nodes_by_role(NodeRole.LEARNING)) for x in self.scheduler.run(context=context)]
            self.execution_sets = [x for x in self.execution_sets if len(x) > 0]
        if self.parameters.pytorch_representation._get(context) is None:
            model = PytorchModelCreator(self.graph_processing,
                                        self.param_init_from_pnl,
                                        self.execution_sets,
                                        self.device,
                                        context=context,
                                        composition = self,
                                        )
            self.parameters.pytorch_representation._set(model, context)

        # Set up optimizer function
        old_opt = self.parameters.optimizer._get(context)
        if old_opt is None:
            opt = self._make_optimizer(self.optimizer_type, self.learning_rate, self.weight_decay, context)
            self.parameters.optimizer._set(opt, context)

        # Set up loss function
        if self.loss is not None:
            logger.warning("Overwriting loss function for AutodiffComposition {}! Old loss function: {}".format(
                self, self.loss))
        if callable(self.loss_spec):
            self.loss = self.loss_spec
        else:
            self.loss = self._get_loss(self.loss_spec)
        
        return self.parameters.pytorch_representation._get(context)

    def _make_optimizer(self, optimizer_type, learning_rate, weight_decay, context):
        if not isinstance(learning_rate, (int, float)):
            raise AutodiffCompositionError("Learning rate must be an integer or float value.")
        if optimizer_type not in ['sgd', 'adam']:
            raise AutodiffCompositionError("Invalid optimizer specified. Optimizer argument must be a string. "
                                           "Currently, Stochastic Gradient Descent and Adam are the only available "
                                           "optimizers (specified as 'sgd' or 'adam').")
        params = self.parameters.pytorch_representation._get(context).parameters()
        if optimizer_type == 'sgd':
            return optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            return optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

    def _get_loss(self, loss_spec):
        if not isinstance(self.loss_spec, str):
            return self.loss_spec
        elif loss_spec == 'mse':
            return nn.MSELoss(reduction='mean')
        elif loss_spec == 'sse':
            return nn.MSELoss(reduction='sum')
        elif loss_spec == 'crossentropy':
            # Cross entropy loss is used for multiclass categorization and needs inputs in shape
            # ((# minibatch_size, C), targets) where C is a 1-d vector of probabilities for each potential category
            # and where target is a 1d vector of type long specifying the index to the target category. This
            # formatting is different from most other loss functions available to autodiff compositions,
            # and therefore requires a wrapper function to properly package inputs.
            cross_entropy_loss = nn.CrossEntropyLoss()
            return lambda x, y: cross_entropy_loss(
                    x.unsqueeze(0),
                    y.type(torch.LongTensor)
            )
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

    # performs forward computation for one input
    def autodiff_processing(self, inputs, context=None, do_logging=False, scheduler=None):
        pytorch_representation = self.parameters.pytorch_representation._get(context)
        # run the model on inputs - switch autograd off for this (we don't need it)
        with torch.no_grad():
            tensor_outputs = pytorch_representation.forward(inputs, context=context, do_logging=do_logging, scheduler=scheduler)

        # get outputs back into numpy
        outputs = []
        for i in range(len(tensor_outputs)):
            outputs.append(tensor_outputs[i].detach().cpu().numpy().copy())

        return outputs

    # performs learning/training on all input-target pairs it recieves for given number of epochs
    def autodiff_training(self, inputs, targets, context=None, do_logging=False, scheduler=None):

        # FIX CW 11/1/18: this value of num_inputs assumes all inputs have same length, and that the length of
        # the input for an origin component equals the number of desired trials. We could clean this up
        # by perhaps using modular arithmetic on t, or by being more explicit about number of desired trials
        first_input_value = list(inputs.values())[0]
        num_inputs = len(first_input_value)

        # get total number of output neurons from the dimensionality of targets on the first trial
        # (this is for computing average loss across neurons on each trial later)
        out_size = 0
        for target in targets.values():
            out_size += len(target)

        # if training in random order, generate random order and set up mapping
        # from random order back to original order
        if self.randomize:
            rand_train_order = np.random.permutation(num_inputs)
            rand_train_order_reverse[rand_train_order] = np.arange(num_inputs)

        # set up array to keep track of losses on epoch
        curr_losses = np.zeros(num_inputs)

        # reset temporary list to keep track of most recent outputs
        outputs = []

        self.parameters.pytorch_representation._get(context).detach_all()
        # self.parameters.pytorch_representation._get(context).reset_all()

        # compute total loss across output neurons for current trial
        curr_loss = torch.zeros(1, device=self.device).double()

        # iterate over inputs, targets
        for t in range(num_inputs):

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
            curr_tensor_outputs = self.parameters.pytorch_representation._get(context).forward(
                curr_tensor_inputs,
                context,
                do_logging,
                scheduler=scheduler,
            )

            for component in curr_tensor_outputs.keys():
                # possibly add custom loss option, which is a loss function that takes many args
                # (outputs, targets, weights, and more) and returns a scalar
                new_loss = self.loss(curr_tensor_outputs[component], curr_tensor_targets[component])
                curr_loss += new_loss

            # save average loss across all output neurons on current trial
            curr_losses[t] = curr_loss[0].item() / num_inputs

            curr_output_list = []
            for input_port in self.output_CIM.input_ports:
                assert (len(input_port.all_afferents) == 1)  # CW 12/05/18, this assert may eventually be outdated
                component = input_port.all_afferents[0].sender.owner
                curr_output_list.append(curr_tensor_outputs[component].detach().cpu().numpy().copy())
            outputs.append(curr_output_list)

        optimizer = self.parameters.optimizer._get(context)

        # backpropagate to compute gradients and perform learning update for parameters
        optimizer.zero_grad()
        curr_loss = curr_loss / num_inputs
        printable = {}
        for component in curr_tensor_outputs.keys():
            printable[component] = curr_tensor_outputs[component].detach().numpy()
        if self.force_no_retain_graph:
            curr_loss.backward(retain_graph=False)
        else:
            curr_loss.backward(retain_graph=True)
        optimizer.step()
        self.parameters.pytorch_representation._get(context).copy_weights_to_psyneulink(context)

        # save average loss on the current epoch
        average_loss = np.mean(curr_losses)
        self.parameters.losses._get(context).append(average_loss)

        return outputs

    def _gen_llvm_function(self, *, tags:frozenset):
        with pnlvm.LLVMBuilderContext.get_global() as ctx:
            if "run" in tags:
                return ctx.gen_composition_run(self, tags=tags)
            elif "learning" in tags:
                return ctx.gen_autodiffcomp_learning_exec(self, tags=tags)
            else:
                return ctx.gen_autodiffcomp_exec(self, tags=tags)

    def _infer_output_nodes(self, nodes: dict):
        """
        Maps targets onto target mechanisms (as needed by learning)

        Returns
        ---------
        A dict mapping TargetMechanisms -> target values
        """
        ret = {}
        for node, values in nodes.items():
            if NodeRole.TARGET in self.get_roles_by_node(node) and NodeRole.LEARNING in self.get_roles_by_node(node):
                node_efferent_mechanisms = [x.receiver.owner for x in node.efferents]
                comparators = [x for x in node_efferent_mechanisms if (isinstance(x, ComparatorMechanism) and NodeRole.LEARNING in self.get_roles_by_node(x))]
                comparator_afferent_mechanisms = [x.sender.owner for c in comparators for x in c.afferents]
                output_nodes = [t for t in comparator_afferent_mechanisms if (NodeRole.OUTPUT in self.get_roles_by_node(t) and NodeRole.LEARNING not in self.get_roles_by_node(t))]
                
                if len(output_nodes) != 1:
                    # Invalid specification! Either we have no valid target nodes, or there is ambiguity in which target node to choose
                    raise Exception(f"Unable to infer learning target node from output node {node}!")
                
                ret[output_nodes[0]] = values
            elif NodeRole.OUTPUT in self.get_roles_by_node(node):
                ret[node] = values
        return ret
    
    def _infer_input_nodes(self, nodes: dict):
        """
        Maps targets onto target mechanisms (as needed by learning)

        Returns
        ---------
        A dict mapping TargetMechanisms -> target values
        """
        ret = {}
        for node, values in nodes.items():
            if NodeRole.INPUT in self.get_roles_by_node(node) and not NodeRole.TARGET in self.get_roles_by_node(node):
                ret[node] = values
        return ret

    @handle_external_context()
    def execute(self,
                inputs=None,
                autodiff_stimuli=None,
                num_trials=None,
                minibatch_size=1,
                do_logging=False,
                scheduler=None,
                termination_processing=None,
                call_before_minibatch=None,
                call_after_minibatch=None,
                call_before_time_step=None,
                call_before_pass=None,
                call_after_time_step=None,
                call_after_pass=None,
                context=None,
                base_context=Context(execution_id=None),
                clamp_input=SOFT_CLAMP,
                targets=None,
                runtime_params=None,
                bin_execute=False,
                skip_initialization=False,
                ):
        self._assign_execution_ids(context)
        context.composition = self
        context.source = ContextFlags.COMPOSITION

        if scheduler is None:
            scheduler = self.scheduler

        if self._is_learning(context):
            # TBI: How are we supposed to use base_context and statefulness here?
            # TBI: can we call _build_pytorch_representation in _analyze_graph so that pytorch
            # model may be modified between runs?


            autodiff_inputs = self._infer_input_nodes(inputs)
            autodiff_targets = self._infer_output_nodes(inputs)

            self._build_pytorch_representation(context)
            output = self.autodiff_training(autodiff_inputs,
                                            autodiff_targets,
                                            context,
                                            do_logging,
                                            scheduler)

            context.add_flag(ContextFlags.PROCESSING)
            
            self.output_CIM.execute(output[-1], context=context)
            context.remove_flag(ContextFlags.PROCESSING)

            # note that output[-1] might not be the truly most recent value
            # HACK CW 2/5/19: the line below is a hack. In general, the output_CIM of an AutodiffComposition
            # is not having its parameters populated correctly, and this should be fixed in the long run.
            scheduler.get_clock(context)._increment_time(TimeScale.TRIAL)
            return output

        return super(AutodiffComposition, self).execute(inputs=inputs,
                                                        scheduler=scheduler,
                                                        termination_processing=termination_processing,
                                                        call_before_time_step=call_before_time_step,
                                                        call_before_pass=call_before_pass,
                                                        call_after_time_step=call_after_time_step,
                                                        call_after_pass=call_after_pass,
                                                        context=context,
                                                        base_context=base_context,
                                                        clamp_input=clamp_input,
                                                        runtime_params=runtime_params,
                                                        bin_execute=bin_execute,
                                                        )

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

            # raise error if a node has more than one InputPort
            if len(node.component.input_ports) > 1:
                raise AutodiffCompositionError("Mechanism {0} of {1} has more than one InputPort. Autodiff "
                                               "Compositions only allow mechanisms to have one InputPort. The "
                                               "dimensionality of this port's value will become the dimensionality of "
                                               "the tensor representing the port's mechanism in the underlying "
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
    @handle_external_context(execution_id=NotImplemented)
    def get_parameters(self, context=None):
        if context.execution_id is NotImplemented:
            context.execution_id = self.default_execution_id

        pytorch_representation = self.parameters.pytorch_representation._get(context)

        if pytorch_representation is None:
            raise AutodiffCompositionError("{0} has not been run yet so parameters have not been created "
                                           "in Pytorch."
                                           .format(self.name))

        weights = pytorch_representation.get_weights_for_projections()

        return weights
    
    def _get_state_struct_type(self, ctx):
        """Gets state struct for compiled autodiff"""
        node_state_type_list = (ctx.get_state_struct_type(m) for m in self._all_nodes)
        proj_state_type_list = (ctx.get_state_struct_type(p) for p in self._inner_projections)
        pytorch_representation = self._build_pytorch_representation()
        optimizer_state_type = pytorch_representation._get_compiled_optimizer()._get_optimizer_struct_type(ctx)
        return pnlvm.ir.LiteralStructType((
            pnlvm.ir.LiteralStructType(node_state_type_list),
            pnlvm.ir.LiteralStructType(proj_state_type_list),
            optimizer_state_type),
            )

    def _get_state_initializer(self, context):
        node_states = (m._get_state_initializer(context=context) for m in self._all_nodes)
        proj_states = (p._get_state_initializer(context=context) for p in self._inner_projections)
        return (tuple(node_states), tuple(proj_states), tuple())
