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

    * Pytorch functions representing the `function <Mechanism_Base.function>` of each `Mechanism` in the Composition
      incorporate their scalar, untrainable biases.

    If it is set to False:

        * in addition to the weight parameters created for each MappingProjection, a trainable bias parameter
          is created for each for each Mechanism in the Composition;

        * weight parameters have the same dimensionality as the `matrix <MappingProjection.matrix>` parameter of the
          corresponding `MappingProjections <MappingProjection>`;  however, their values -- and those of the bias
          parameters -- are sampled from a random distribution;

        * in addition to the trainable biases created for each Mechanism, the Pytorch function implemented for each
          Mechanism's `function <Mechanism_Base.function>` still incorporates its scalar, untrainable bias.

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

Logging
-------

Logging currently works differently in AutodiffComposition than in Composition. In an AutodiffComposition, no logging
is done by default, because logging substantially (roughly by 30%) slows down AutodiffComposition. If you wish for all
projection weights and mechanism values to be logged during execution or training of AutodiffComposition, you must
set the **do_logging** argument of the ``run()`` method to ``True``. Logging with AutodiffComposition is slightly hacked
together, so the time and context in the log are not meaningful, only the logged value is meaningful.

Nested Execution
----------------

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

.. _Composition_Class_Reference:

Class Reference
---------------

"""

from psyneulink.core.components.functions.transferfunctions import Linear, Logistic, ReLU
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
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
from collections.abc import Iterable
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
                 learning_enabled=True,
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

        # params = self._assign_args_to_param_dicts(learning_rate=learning_rate)
        #
        # super(AutodiffComposition, self).__init__(params=params, name=name)
        super(AutodiffComposition, self).__init__(name = name,
                                                  patience = patience,
                                                  min_delta = min_delta,
                                                  learning_rate = learning_rate,
                                                  learning_enabled = learning_enabled,
                                                  optimizer_type = optimizer_type,
                                                  weight_decay = weight_decay,
                                                  loss_spec = loss_spec,
                                                  randomize = randomize)

        self.learning_enabled = learning_enabled
        self.optimizer_type = optimizer_type
        self.loss_spec = loss_spec
        self.randomize = randomize
        self.refresh_losses = refresh_losses

        self.weight_decay = weight_decay
        self.force_no_retain_graph = force_no_retain_graph
        self.loss = None


        self.__forward_bin_run_func = None
        self.__learning_bin_run_func = None
        # stores compiled binary execute function
        self.__forward_bin_exec_func = None
        self.__learning_bin_exec_func = None
        self.__generated_learning_run = None
        self.__generated_forward_run = None

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
                self.device = torch.device('cuda:' + str(cuda_index))
        else:
            self.device = torch.device('cpu')

    # CLEANUP: move some of what's done in the methods below to a "validate_params" type of method
    @handle_external_context()
    def _build_pytorch_representation(self, context=None):
        if self.scheduler is None:  # if learning_enabled has never been run yet
            self.scheduler = Scheduler(graph=self.graph_processing)
        if self.execution_sets is None:
            self.execution_sets = list(self.scheduler.run(context=context))
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
        if old_opt is not None:
            logger.warning("Overwriting optimizer for AutodiffComposition {}! Old optimizer: {}".format(
                self, old_opt))

        opt = self._make_optimizer(self.optimizer_type, self.learning_rate, self.weight_decay, context)
        self.parameters.optimizer._set(opt, context)

        # Set up loss function
        if self.loss is not None:
            logger.warning("Overwriting loss function for AutodiffComposition {}! Old loss function: {}".format(
                self, self.loss))
        self.loss = self._get_loss(self.loss_spec)

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

    def _adjust_stimulus_dict(self, inputs, bin_execute=False):
        # for bin executes, we manually parse out the autodiff stimuli
        if bin_execute is True or str(bin_execute).endswith('Run'):
            if not self.learning_enabled and isinstance(inputs, dict) and self._has_required_keys(inputs):
                inputs = inputs["inputs"]
            return super(AutodiffComposition, self)._adjust_stimulus_dict(inputs)

        if self.learning_enabled:
            if isinstance(inputs, dict):
                if self._has_required_keys(inputs):
                    return inputs
                raise AutodiffCompositionError("Invalid input specification.")
            elif isinstance(inputs, list):
                for input_dict in inputs:
                    if not self._has_required_keys(input_dict):
                        raise AutodiffCompositionError("Invalid input specification.")
                return inputs

        # If learning is disabled, but inputs are provided in the same format as used for learning,
        #    ignore dict in "targets" entry, and pass dict in "inputs" entry along as inputs
        elif isinstance(inputs, dict) and "inputs" in inputs.keys():
            inputs = inputs["inputs"]

        return super(AutodiffComposition, self)._adjust_stimulus_dict(inputs)

    # performs forward computation for one input
    def autodiff_processing(self, inputs, context=None, do_logging=False, scheduler=None,bin_execute=False):
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
    def autodiff_training(self, inputs, targets, total_epochs, curr_epoch, context=None, do_logging=False, scheduler=None, bin_execute=False):

        # FIX CW 11/1/18: this value of num_inputs assumes all inputs have same length, and that the length of
        # the input for an origin component equals the number of desired trials. We could clean this up
        # by perhaps using modular arithmetic on t, or by being more explicit about number of desired trials
        first_input_value = list(inputs.values())[0]
        num_inputs = len(first_input_value)

        patience = self.parameters.patience._get(context)

        if patience is not None:
            # set up object for early stopping
            early_stopper = EarlyStopping(patience=patience, min_delta=self.parameters.min_delta._get(context))

        # if training over trial sets in random order, set up array for mapping random order back to original order
        if self.randomize:
            rand_train_order_reverse = np.zeros(num_inputs)

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
            curr_losses[t] = curr_loss[0].item()/num_inputs

            # save outputs of model if this is final epoch or if using early stopping
            if patience is not None or curr_epoch == total_epochs - 1:
                curr_output_list = []
                for input_state in self.output_CIM.input_states:
                    assert (len(input_state.all_afferents) == 1)  # CW 12/05/18, this assert may eventually be outdated
                    component = input_state.all_afferents[0].sender.owner
                    curr_output_list.append(curr_tensor_outputs[component].detach().cpu().numpy().copy())
                outputs.append(curr_output_list)
                # outputs.extend(curr_output_list)

        optimizer = self.parameters.optimizer._get(context)

        # backpropagate to compute gradients and perform learning update for parameters
        optimizer.zero_grad()
        curr_loss = curr_loss / num_inputs / 2
        printable = {}
        for component in curr_tensor_outputs.keys():
            printable[component] = curr_tensor_outputs[component].detach().numpy()
        np.set_printoptions(precision=3)
        if self.force_no_retain_graph:
            curr_loss.backward(retain_graph=False)
        else:
            curr_loss.backward(retain_graph=True)
        self.parameters.pytorch_representation._get(context).copy_weights_to_psyneulink(context)
        optimizer.step()

        if curr_epoch == total_epochs - 1 and not do_logging:
            self.parameters.pytorch_representation._get(context).\
                copy_outputs_to_psyneulink(curr_tensor_outputs, context)

        scheduler.get_clock(context)._increment_time(TimeScale.TRIAL)

        # save average loss on the current epoch
        average_loss = np.mean(curr_losses)
        self.parameters.losses._get(context).append(average_loss)

        # update early stopper with most recent average loss
        if self.parameters.patience._get(context) is not None:
            should_stop = early_stopper.step(average_loss)
            if should_stop:
                logger.warning('Due to early stopping, stopped training early after {} epochs'.format(curr_epoch))
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

    @property
    def _bin_exec_func(self):
        if self.learning_enabled is True:
            if self.__learning_bin_exec_func is None:
                with pnlvm.LLVMBuilderContext.get_global() as ctx:
                    self.__learning_bin_exec_func = ctx.gen_autodiffcomp_learning_exec(self)
            return self.__learning_bin_exec_func
        else:
            if self.__forward_bin_exec_func is None:
                with pnlvm.LLVMBuilderContext.get_global() as ctx:
                    self.__forward_bin_exec_func = ctx.gen_autodiffcomp_exec(self)
            return self.__forward_bin_exec_func

    @property
    def _llvm_run(self):
        if self.learning_enabled is True:
            if self.__generated_learning_run is None:
                with pnlvm.LLVMBuilderContext.get_global() as ctx:
                    self.__generated_learning_run = ctx.gen_composition_run(self)
            return self.__generated_learning_run
        if self.__generated_forward_run is None:
            with pnlvm.LLVMBuilderContext.get_global() as ctx:
                self.__generated_forward_run = ctx.gen_composition_run(self)
        return self.__generated_forward_run

    def _gen_llvm_function(self):
        return self._bin_exec_func

    @handle_external_context()
    def execute(self,
                inputs=None,
                autodiff_stimuli=None,
                num_trials=None,
                minibatch_size=1,
                do_logging=False,
                scheduler_processing=None,
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
                skip_initialization=False,
                bin_execute=False,
                ):
        self._assign_execution_ids(context)
        context.composition = self
        context.source = ContextFlags.COMPOSITION

        if scheduler_processing is None:
            scheduler_processing = self.scheduler_processing

        scheduler_processing._init_clock(context.execution_id, base_context.execution_id)

        if self.learning_enabled:
            # TBI: How are we supposed to use base_context and statefulness here?
            # TBI: can we call _build_pytorch_representation in _analyze_graph so that pytorch
            # model may be modified between runs?

            self._analyze_graph()  # ADDED by CW 12/17/18: unsure if correct here

            autodiff_inputs = inputs["inputs"]
            autodiff_targets = inputs["targets"]
            autodiff_epochs = 1
            if "epochs" in inputs:
                autodiff_epochs = inputs["epochs"]

            minibatch = {
                'inputs': {
                    k: [] for k in inputs['inputs'].keys()
                },
                'targets': {
                    k: [] for k in inputs['targets'].keys()
                }
            }

            if num_trials is None:
                num_trials = len(list(inputs['inputs'].values())[0])
            if minibatch_size == TRAINING_SET:
                minibatch_size = num_trials

            results = []
            self._build_pytorch_representation(context)

            for current_epoch in range(autodiff_epochs):
                for trial_num in range(num_trials):
                    for k,v in inputs['inputs'].items():
                        minibatch['inputs'][k].append(v[trial_num])
                    for k, v in inputs['targets'].items():
                        minibatch['targets'][k].append(v[trial_num])
                    minibatch_results = []
                    if len(list(minibatch['inputs'].values())[0]) == minibatch_size or \
                            trial_num == num_trials-1:
                        if call_before_minibatch:
                            call_before_minibatch()
                        output = self.autodiff_training(minibatch['inputs'],
                                                        minibatch['targets'],
                                                        autodiff_epochs,
                                                        current_epoch,
                                                        context,
                                                        do_logging,
                                                        scheduler_processing)
                        if call_after_minibatch:
                            call_after_minibatch()
                        self.most_recent_context = context
                        for k, v in inputs['inputs'].items():
                            minibatch['inputs'][k] = []
                        for k, v in inputs['targets'].items():
                            minibatch['targets'][k] = []
                        if current_epoch == autodiff_epochs-1:
                            results.extend(output)

            context.add_flag(ContextFlags.PROCESSING)
            # note that output[-1] might not be the truly most recent value
            # HACK CW 2/5/19: the line below is a hack. In general, the output_CIM of an AutodiffComposition
            # is not having its parameters populated correctly, and this should be fixed in the long run.
            self.output_CIM.execute(input=results[-1], context=context)
            context.remove_flag(ContextFlags.PROCESSING)
            return results

        return super(AutodiffComposition, self).execute(inputs=inputs,
                                                        scheduler_processing=scheduler_processing,
                                                        termination_processing=termination_processing,
                                                        call_before_time_step=call_before_time_step,
                                                        call_before_pass=call_before_pass,
                                                        call_after_time_step=call_after_time_step,
                                                        call_after_pass=call_after_pass,
                                                        context=context,
                                                        base_context=base_context,
                                                        clamp_input=clamp_input,
                                                        runtime_params=runtime_params,
                                                        skip_initialization=skip_initialization,
                                                        bin_execute=bin_execute,
                                                        )

    # what the user calls for doing processing/training, similar to the run function of the normal composition
    @handle_external_context()
    def run(
        self,
        inputs=None,
        do_logging=False,
        scheduler_processing=None,
        termination_processing=None,
        context=None,
        num_trials=None,
        minibatch_size=1,
        call_before_time_step=None,
        call_after_time_step=None,
        call_before_pass=None,
        call_after_pass=None,
        call_before_trial=None,
        call_after_trial=None,
        call_before_minibatch=None,
        call_after_minibatch=None,
        clamp_input=SOFT_CLAMP,
        bin_execute=False,
        initial_values=None,
        reinitialize_values=None,
        runtime_params=None,
    ):
        """Passes inputs to AutodiffComposition, then execute sets of nodes that are eligible to run until termination
        conditions are met.

            Arguments
            ---------

            inputs: {'inputs': {`Mechanism <Mechanism>`: list}, 'targets': {`Mechanism <Mechanism>`: list}, 'epochs': int }
                a key-value pair with the keys "inputs", "targets", and "epochs". The value corresponding
                to the "inputs" key should itself be a key-value pair for each Node in the composition that receives
                inputs from the user. For each pair, the key is the Node and the value is a list of inputs. Each input
                in the list corresponds to one `TRIAL`. Analogously, the value corresponding with 'targets' should be a
                key-value pair with keys for each terminal Node in the composition and a corresponding list of the Node's
                target values for each trial. The value corresponding to the 'epochs' key is an int specifying how many
                times the Composition should run through the entire input set.

            scheduler_processing: Scheduler
                the scheduler object that owns the conditions that will instruct the execution of the Composition.
                If not specified, the Composition will use its automatically generated scheduler.

            context
                context will be set to self.default_execution_id if unspecified

            base_context
                the context corresponding to the execution context from which this execution will be initialized,
                if values currently do not exist for **context**

            num_trials: int
                typically, the composition will infer the number of trials from the length of its input specification.
                To reuse the same inputs across many trials, you may specify an input dictionary with lists of length 1,
                or use default inputs, and select a number of trials with num_trials.

            minibatch_size: int or `TRAINING_SET`
                if learning is enabled, the number of trials to be executed by the autodiff composition between weight
                updates. if set to `TRAINING_SET`, weights will be updated after each full traversal of the provided
                inputs (i.e. after each epoch).

            call_before_time_step: callable
                Not currently implemented for autodiff compositions.

            call_after_time_step: callable
                Not currently implemented for autodiff compositions.

            call_before_pass: callable
                Not currently implemented for autodiff compositions.

            call_after_pass: callable
                Not currently implemented for autodiff compositions.

            call_before_trial: callable
                Not currently implemented for autodiff compositions.

            call_after_trial: callable
                Not currently implemented for autodiff compositions.

            call_before_minibatch: callable
                will be called before each minibatch is executed.

            call_after_minibatch: callable
                will be called after each minibatch is executed.

            initial_values: Dict[Node: Node Value]
                sets the values of nodes before the start of the run. This is useful in cases where a node's value is
                used before that node executes for the first time (usually due to recurrence or control).

            runtime_params: Dict[Node: Dict[Parameter: Tuple(Value, Condition)]]
                nested dictionary of (value, `Condition`) tuples for parameters of Nodes (`Mechanisms <Mechanism>` or
                `Compositions <Composition>` of the Composition; specifies alternate parameter values to be used only
                during this `Run` when the specified `Condition` is met.

                Outer dictionary:
                    - *key* - Node
                    - *value* - Runtime Parameter Specification Dictionary

                Runtime Parameter Specification Dictionary:
                    - *key* - keyword corresponding to a parameter of the Node
                    - *value* - tuple in which the index 0 item is the runtime parameter value, and the index 1 item is
                      a `Condition`

                See `Run_Runtime_Parameters` for more details and examples of valid dictionaries.

            log: bool, LogCondition
                Sets the `log_condition <Parameter.log_condition>` for every primary `node <Composition.nodes>` and
                `projection <Composition.projections>` in this Composition, if it is not already set.

                .. note::
                   as when setting the `log_condition <Parameter.log_condition>` directly, a value of `True` will
                   correspond to the `EXECUTION LogCondition <LogCondition.EXECUTION>`.

        COMMENT:
        REPLACE WITH EVC/OCM EXAMPLE
        Examples
        --------

        This figure shows an animation of the Composition in the XXX example script, with
        the show_graph **show_learning** argument specified as *ALL*:

        .. _Composition_XXX_movie:

        .. figure:: _static/XXX_movie.gif
           :alt: Animation of Composition in XXX example script
           :scale: 50 %

        This figure shows an animation of the Composition in the XXX example script, with
        the show_graph **show_control** argument specified as *ALL* and *UNIT* specified as *EXECUTION_SET*:

        .. _Composition_XXX_movie:

        .. figure:: _static/XXX_movie.gif
           :alt: Animation of Composition in XXX example script
           :scale: 150 %
        COMMENT

        Returns
        ---------

        output value of the final Node executed in the composition : various
        """
        # TBI: Handle trials, timesteps, etc
        self._assign_execution_ids(context)
        context.composition = self

        if scheduler_processing is None:
            scheduler_processing = self.scheduler_processing

        scheduler_processing._init_clock(context)

        if self.learning_enabled:
            if bin_execute is True or str(bin_execute).endswith('Run'):
                # Since the automatically generated llvm function is overwritten in the event that learning_enabled is true, we can just rely on the super function
                results = super(AutodiffComposition, self).run(inputs=inputs,
                                                    scheduler_processing=scheduler_processing,
                                                    termination_processing=termination_processing,
                                                    context=context,
                                                    num_trials=num_trials,
                                                    call_before_time_step=call_before_time_step,
                                                    call_after_time_step=call_after_time_step,
                                                    call_before_pass=call_before_pass,
                                                    call_after_pass=call_after_pass,
                                                    call_before_trial=call_before_trial,
                                                    call_after_trial=call_after_trial,
                                                    clamp_input=clamp_input,
                                                    bin_execute=bin_execute,
                                                    initial_values=initial_values,
                                                    reinitialize_values=reinitialize_values,
                                                    runtime_params=runtime_params,
                                                    )
                self.parameters.pytorch_representation._get(context).copy_weights_to_psyneulink(context)
                # HACK: manually call forward function to get final outputs
                results = []
                self.learning_enabled = False
                input_nodes = self.get_nodes_by_role(NodeRole.INPUT)
                forward_inputs = inputs["inputs"]
                for input_num in range(0,len(forward_inputs[input_nodes[0]])):
                    curr_input_dict = {}
                    for node in input_nodes:
                        curr_input_dict[node] = [forward_inputs[node][input_num]]
                    results.append(self.run(
                        inputs=curr_input_dict,
                        scheduler_processing=scheduler_processing,
                        termination_processing=termination_processing,
                        context=context,
                        num_trials=num_trials,
                        call_before_time_step=call_before_time_step,
                        call_after_time_step=call_after_time_step,
                        call_before_pass=call_before_pass,
                        call_after_pass=call_after_pass,
                        call_before_trial=call_before_trial,
                        call_after_trial=call_after_trial,
                        clamp_input=clamp_input,
                        bin_execute=bin_execute,
                        initial_values=initial_values,
                        reinitialize_values=reinitialize_values,
                        runtime_params=runtime_params,
                    ))
                self.learning_enabled = True
                results = [results]
            else:
                self._analyze_graph()
                self._initialize_from_context(context, base_context=Context(execution_id=None), override=False)
                if self.refresh_losses or (self.parameters.losses._get(context) is None):
                    self.parameters.losses._set([], context)
                if callable(inputs):
                    stimuli = inputs()
                else:
                    stimuli = self._adjust_stimulus_dict(inputs)
                trial_output = self.execute(
                    inputs=stimuli,
                    minibatch_size=minibatch_size,
                    call_before_minibatch=call_before_minibatch,
                    call_after_minibatch=call_after_minibatch,
                    num_trials=num_trials,
                    context=context,
                    do_logging=do_logging,
                    bin_execute=bin_execute
                )
                full_results = self.parameters.results._get(context)
                if full_results is None:
                    full_results = trial_output
                else:
                    full_results.append(trial_output)
                self.parameters.results._set(full_results, context)
                results = full_results
            self.most_recent_context = context
            return results

        else:
            results = super(AutodiffComposition, self).run(inputs=inputs,
                                                    scheduler_processing=scheduler_processing,
                                                    termination_processing=termination_processing,
                                                    context=context,
                                                    num_trials=num_trials,
                                                    call_before_time_step=call_before_time_step,
                                                    call_after_time_step=call_after_time_step,
                                                    call_before_pass=call_before_pass,
                                                    call_after_pass=call_after_pass,
                                                    call_before_trial=call_before_trial,
                                                    call_after_trial=call_after_trial,
                                                    clamp_input=clamp_input,
                                                    bin_execute=bin_execute,
                                                    initial_values=initial_values,
                                                    reinitialize_values=reinitialize_values,
                                                    runtime_params=runtime_params,
                                                    )

        scheduler_processing.get_clock(context)._increment_time(TimeScale.RUN)
        return results

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
        biases = pytorch_representation.get_biases_for_mechanisms()

        return weights, biases

    def _get_param_struct_type(self, ctx):
        # We only need input/output params (rest should be in pytorch model params)
        mech_param_type_list = (ctx.get_param_struct_type(m) if (m is self.input_CIM or m is self.output_CIM)
                                else pnlvm.ir.LiteralStructType([]) for m in self._all_nodes)

        proj_param_type_list = (ctx.get_param_struct_type(p) if (p.sender in self.input_CIM.input_states or p.receiver in self.output_CIM.input_states)
                                else pnlvm.ir.LiteralStructType([]) for p in self.projections)

        self._build_pytorch_representation(self.default_execution_id)
        model = self.parameters.pytorch_representation.get(
            self.default_execution_id)
        pytorch_params = model._get_param_struct_type(ctx)

        input_nodes = self.get_nodes_by_role(NodeRole.INPUT)
        output_nodes = self.get_nodes_by_role(NodeRole.OUTPUT)

        learning_ty = pnlvm.ir.LiteralStructType([
                ctx.int32_ty, # dimensionality
                pnlvm.ir.IntType(64) # Data ptr
            ])
        learning_inputs = pnlvm.ir.LiteralStructType(
            (learning_ty for node in input_nodes))
        learning_targets = pnlvm.ir.LiteralStructType(
            (learning_ty for node in output_nodes))
        learning_params = pnlvm.ir.LiteralStructType([
            ctx.int32_ty, # epochs
            ctx.int32_ty, # number of targets/inputs to train with
            ctx.int32_ty, # number target nodes
            learning_targets, # target struct array
            ctx.int32_ty, # number input nodes
            learning_inputs, # input struct array
        ])
        param_args = [pnlvm.ir.LiteralStructType(mech_param_type_list),
                      pnlvm.ir.LiteralStructType(proj_param_type_list),
                      pytorch_params, learning_params]
        return pnlvm.ir.LiteralStructType(param_args)

    def _get_param_initializer(self, context, simulation=False):
        def _parameterize_node(node):
            if node is self.input_CIM or node is self.output_CIM:
                return tuple(node._get_param_initializer(context))
            else:
                return tuple()

        mech_params = (_parameterize_node(m)
                       for m in self._all_nodes if m is not self.controller or not simulation)
        proj_params = (tuple(p._get_param_initializer(context)) if (p.sender in self.input_CIM.input_states or p.receiver in self.output_CIM.input_states)
                       else tuple() for p in self.projections)
        self._build_pytorch_representation(self.default_execution_id)
        model = self.parameters.pytorch_representation.get(self.default_execution_id)
        pytorch_params = model._get_param_initializer()
        learning_params = (0, 0, 0, (), 0, ())
        param_args = (tuple(mech_params), tuple(proj_params),
                      pytorch_params, learning_params)
        return tuple(param_args)

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

