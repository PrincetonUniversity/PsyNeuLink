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
  * `AutodiffComposition_Execution`
      - `AutodiffComposition_LLVM`
      - `AutodiffComposition_PyTorch`
      - `AutodiffComposition_Nested_Modulation`
      - `AutodiffComposition_Logging`
  * `AutodiffComposition_Examples`
  * `AutodiffComposition_Class_Reference`


.. _AutodiffComposition_Overview:

Overview
--------

AutodiffComposition is a subclass of `Composition` for constructing and training feedforward neural network
either, using either direct compilation (to LLVM) or automatic conversion to `PyTorch <https://pytorch.org/>`_,
both of which considerably accelerate training (by as much as three orders of magnitude) compared to the
`standard implementation of learning  <Composition_Learning_Standard>` in a Composition.  Although an
AutodiffComposition is constructed and executed in much the same way as a standard Composition, it largely restricted
to feedforward neural networks using `supervised learning <Composition_Learning_Supervised>`, and in particular the
the `backpropagation learning algorithm <https://en.wikipedia.org/wiki/Backpropagation>`_. although it can be used for
some forms of `unsupervised learning <Composition_Learning_Unsupervised>` that are supported in PyTorch (e.g.,
`self-organized maps <https://github.com/giannisnik/som>`_).


.. _AutodiffComposition_Creation:

Creating an AutodiffComposition
-------------------------------

An AutodiffComposition can be created by calling its constructor, and then adding `Components <Component>` using
the standard `Composition methods <Composition_Creation>` for doing so (e.g., `add_node <Composition.add_node>`,
`add_projection <Composition.add_projections>`,  `add_linear_processing_pathway
<Composition.add_linear_processing_pathway>`, etc.).  The constructor also includes a number of parameters that are
specific to the AutodiffComposition (see `AutodiffComposition_Class_Reference` for a list of these parameters,
and `examples <AutodiffComposition_Examples>` below).  Note that all of the Components in an AutodiffComposition
must be able to be subject to `learning <Composition_Learning>`, but cannot include any `learning components
<Composition_Learning_Components>` themselves.  Specifically, it cannot include any `ModulatoryMechanisms
<ModulatoryMechanism>`, `LearningProjections <LearningProjection>`, or the ObjectiveMechanism <OBJECTIVE_MECHANISM>`
used to compute the loss for learning.

    .. _Autodiff_Learning_Components_Warning:
    .. warning::
        When an AutodiffComposition is constructed, it creates all of the learning Components
        that are needed, and thus **cannot include** any that are prespecified.

COMMENT:
FIX: IS THIS STILL TRUE? SEEMS TO CONTRADICT STATEMENT BELOW:
This means that it cannot be used with a Composition that contains any `modulatory components
<ModulatorySignal_Anatomy_Figure>` or ones that are subject to modulation, whether by ModulatoryMechanisms within or
outside the Composition;
?MAYBE THE FOLLOWING IS BETTER:
COMMENT
This means that an AutodiffComposition also cannot itself include a `controller <Composition_Controller>` or any
`ControlMechanisms <ControlMechanism>`.  However, it can include Mechanisms that are subject to modulatory control
(see `Figure <ModulatorySignal_Anatomy_Figure>`, and `modulation <ModulatorySignal_Modulation>`) by ControlMechanisms
*outside* the Composition, including the controller of a Composition within which the AutodiffComposition is nested.
That is, an AutodiffComposition can be `nested in a Composition <Composition_Nested>` that has such other Components
(see `AutodiffComposition_Nested_Modulation` below).

A few other restrictions apply to the construction and modification of AutodiffCompositions:

    .. hint:: AutodiffComposition does not (currently) support the *automatic* construction of separate bias parameters.
       Thus, when comparing a model constructed using an AutodiffComposition to a corresponding model in PyTorch, the
       `bias <https://www.pytorch.org/docs/stable/nn.html#torch.nn.Module>` parameter of PyTorch modules should be set
       to `False`.  Trainable biases *can* be specified explicitly in an AutodiffComposition by including a
       TransferMechanism that projects to the relevant Mechanism (i.e., implementing that layer of the network to
       receive the biases) using a `MappingProjection` with a `matrix <MappingProjection.matrix>` parameter that
       implements a diagnoal matrix with values corresponding to the initial value of the biases.

    .. warning:: Mechanisms or Projections should not be added to or deleted from an AutodiffComposition after it
       has been executed. Unlike an ordinary Composition, AutodiffComposition does not support this functionality.


.. _AutodiffComposition_Execution:

Execution
---------

An AutodiffComposition's `run <Composition.run>`, `execute <Composition.execute>`, and `learn <Composition.learn>`
methods are the same as for a `Composition`.  However, the **execution_mode** in the `learn <Composition.learn>`
method has different effects than for a standard Composition, that determine whether it uses `LLVM compilation
<AutodiffComposition_LLVM>` or `translation to PyTorch <AutodiffComposition_PyTorch>` to execute learning.
These are each described in greater detail below, and summarized in this `table <Composition_Compilation_Table>`
which provides a comparison of the different modes of execution for an AutodiffComposition and standard `Composition`.

.. _AutodiffComposition_LLVM:

*LLVM mode*
~~~~~~~~~~~

This is specified by setting **execution_mode** = `ExecutionMode.LLVMRun` in the `learn <Composition.learn>` method
of an AutodiffCompositon.  This provides the fastest performance, but is limited to `supervised learning
<Composition_Learning_Supervised>` using the `BackPropagation` algorithm. This can be run using standard forms of
loss, including mean squared error (MSE) and cross entropy, by specifying this in the **loss_spec** argument of
the constructor (see `AutodiffComposition <AutodiffComposition_Class_Reference>` for additional details, and
`Compilation Modes <Composition_Compiled_Modes>` for more information about executing a Composition in compiled mode.

    .. note::
       Specifying `ExecutionMode.LLVMRUn` in either the `learn <Composition.learn>` and `run <Composition.run>`
       methods of an AutodiffComposition causes it to (attempt to) use compiled execution in both cases; this is
       because LLVM compilation supports the use of modulation in PsyNeuLink models (as compared to `PyTorch mode
       <AutodiffComposition_PyTorch>`; see `note <AutodiffComposition_PyTorch_Note>` below).

.. _AutodiffComposition_PyTorch:

*PyTorch mode*
~~~~~~~~~~~~~~

This is specified by setting **execution_mode = `ExecutionMode.PyTorch` in the `learn <Composition.learn>` method of
an AutodiffCompositon (see `example <BasicsAndPrimer_Rumelhart_Model>` in `BasicsAndPrimer`).  This automatically
translates the AutodiffComposition to a `PyTorch <https://pytorch.org>`_ model and uses that for execution.  This is
almost as fast as `LLVM compilation <_AutodiffComposition_LLVM>`, but provides greater flexiblity.  Although it too is
best suited for use with `supervised learning <Composition_Learning_Supervised>`, it can also be used for some forms
of `unsupervised learning <Composition_Learning_Unsupervised>` that are supported in PyTorch (e.g., `self-organized
maps <https://github.com/giannisnik/som>`_).

    .. _AutodiffComposition_PyTorch_Note:

    .. note::
       While specifying `ExecutionMode.PyTorch` in the `learn <Composition.learn>`  method of an AutodiffComposition
       causes it to use PyTorch for training, specifying this in the `run <Compositon.run>` method causes it to be
       executed using the *Python* interpreter (and not PyTorch);  this is so that any modulation can take effect
       during execution (see `AutodiffComposition_Nested_Modulation` below), which is not supported by PyTorch.

    .. warning::
      * Specifying `ExecutionMode.LLVM` or `ExecutionMode.PyTorch` in the learn() method of a standard
        `Composition` causes an error.

    .. note::
      * Specifying `ExecutionMode.Python` in the learn() method of a `AutodiffComposition` is treated as a
        synonym of `ExecutionMode.PyTorch` (see table).

.. technical_note::
   `ExecutionMode.PyTorch` is a synonym for `ExecutionMode.Python`, that is provided for clarity of the user interface:
   the default for an AutodiffComposition (i.e., if **execution_mode** is not specified, or it is set to
   `ExecutionMode.Python`) is to use PyTorch translation in `learn <Composition.learn>` but the Python interpreter
   for `run <Composition.run>`.  The use of `ExecutionMode.PyTorch` is simply to make it clear that, during learning,
   it will use PyTorch. This contrasts with the use of `ExecutionMode.LLVMrun`, in which case both the `learn
   <Composition.learn>` and `run <Composition.run>` methods use LLVM compilation.

.. _AutodiffComposition_Nested_Modulation:

*Nested Execution and Modulation*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Like any other `Composition`, an AutodiffComposition may be `nested <Composition_Nested>` inside another
(see `example <AutodiffComposition_Nested_Example>` below).  However, learning, none of the internal
Components of the AutodiffComposition (e.g., intermediate layers of a neural network model) are accessible to the
other Components of the outer Composition, (e.g., as sources of information, or for `modulation
<ModulatorySignal_Modulation>`).  However, when
COMMENT:
learning turned off,
COMMENT
it is executed using its `run <Composition.run>` method, then the  AutodiffComposition functions like any other,
and all of its internal Components are accessible to other Components of the outer Composition. Thus, as long as access
to its internal Components is not needed during learning, an `AutodiffComposition` can be trained, and then used to
execute the trained Composition like any other.


.. _AutodiffComposition_Logging:

*Logging*
~~~~~~~~~

Logging in AutodiffCompositions follows the same procedure as `logging in a Composition <Log>`.
However, since an AutodiffComposition internally converts all of its Mechanisms either to LLVM
or to an equivalent PyTorch model, then its inner components are not actually executed. This means that there is
limited support for logging parameters of components inside an AutodiffComposition; Currently, the only supported
parameters are:

1) the `matrix` parameter of Projections

2) the `value` parameter of its inner components


.. _AutodiffComposition_Examples:

Examples
--------

.. _AutodiffComposition_Creation_Example:

The following is an example showing how to create a simple AutodiffComposition, specify its inputs and targets,
and run it with learning enabled and disabled:

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
    >>> # Run Composition in learnng mode
    >>> my_autodiff.learn(inputs = input_dict)
    >>> # Run Composition in test mode
    >>> my_autodiff.run(inputs = input_dict['inputs'])


.. _AutodiffComposition_Nested_Example:

The following shows how the AutodiffComposition created in the previous example can be nested and run inside another
Composition::

    >>> # Create outer composition
    >>> my_outer_composition = pnl.Composition()
    >>> my_outer_composition.add_node(my_autodiff)
    >>> # Specify dict containing inputs and targets for nested Composition
    >>> training_input = {my_autodiff: input_dict}
    >>> # Run in learning mode
    >>> result1 = my_outer_composition.learn(inputs=training_input)
    COMMENT:
    >>> # Run with learning disabled (and standard input format)
    >>> no_training_input = {my_autodiff: my_inputs}
    >>> result2 = parentmy_outer_compositionComposition.run(inputs=no_training_input)
    COMMENT

.. _AutodiffComposition_Class_Reference:

Class Reference
---------------

"""
import logging
import os
import warnings
import numpy as np
from packaging import version
from pathlib import Path, PosixPath

try:
    import torch
    from torch import nn
    import torch.optim as optim
    torch_available = True
except ImportError:
    torch_available = False
else:
    from psyneulink.library.compositions.pytorchmodelcreator import PytorchModelCreator

from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.core.components.mechanisms.modulatory.modulatorymechanism import ModulatoryMechanism_Base
from psyneulink.core.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
from psyneulink.core.compositions.composition import Composition, NodeRole
from psyneulink.core.compositions.composition import CompositionError
from psyneulink.core.compositions.report \
    import ReportOutput, ReportParams, ReportProgress, ReportSimulations, ReportDevices, \
    LEARN_REPORT, EXECUTE_REPORT, PROGRESS_REPORT
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import AUTODIFF_COMPOSITION, SOFT_CLAMP, Loss
from psyneulink.core.scheduling.scheduler import Scheduler
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.core import llvm as pnlvm



logger = logging.getLogger(__name__)


__all__ = [
    'AutodiffComposition'
]


class AutodiffCompositionError(CompositionError):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class AutodiffComposition(Composition):
    """
    Subclass of `Composition` that trains models using either LLVM compilation or `PyTorch <https://pytorch.org>`_;
    see and `Composition <Composition_Class_Reference>` for additional arguments and attributes.

    Arguments
    ---------

    learning_rate : float : default 0.001
        the learning rate passed to the optimizer if none is specified in the learn method of the AutodiffComposition.

    disable_learning : bool: default False
        specifies whether the AutodiffComposition should disable learning when run in `learning mode
        <Composition.learn>`.

    optimizer_type : str : default 'sgd'
        the kind of optimizer used in training. The current options are 'sgd' or 'adam'.

    weight_decay : float : default 0
        specifies the L2 penalty (which discourages large weights) used by the optimizer.

    loss_spec : Loss or PyTorch loss function : default Loss.MSE
        specifies the loss function for training; see `Loss` for arguments.

    Attributes
    ----------

    optimizer : PyTorch optimizer function
        the optimizer used for training. Depends on the **optimizer_type**, **learning_rate**, and **weight_decay**
        arguments from initialization.

    loss : PyTorch loss function
        the loss function used for training. Depends on the **loss_spec** argument from initialization.

    losses : list of floats
        tracks the average loss after each weight update (i.e. each minibatch) during learning.

    last_saved_weights : path
        path for file to which weights were last saved.

    last_loaded_weights : path
        path for file from which weights were last loaded.

    """

    componentCategory = AUTODIFF_COMPOSITION
    class Parameters(Composition.Parameters):
        optimizer = None
        learning_rate = Parameter(.001, fallback_default=True)
        losses = Parameter([])
        trial_losses = Parameter([])
        tracked_loss = Parameter(None, pnl_internal=True)
        tracked_loss_count = Parameter(0, pnl_internal=True)
        pytorch_representation = None

    # TODO (CW 9/28/18): add compositions to registry so default arg for name is no longer needed
    @check_user_specified
    def __init__(self,
                 pathways=None,
                 learning_rate=None,
                 optimizer_type='sgd',
                 weight_decay=0,
                 loss_spec=Loss.MSE,
                 disable_learning=False,
                 refresh_losses=False,
                 disable_cuda=True,
                 cuda_index=None,
                 force_no_retain_graph=False,
                 name="autodiff_composition",
                 **kwargs):

        if not torch_available:
            raise AutodiffCompositionError('Pytorch python module (torch) is not installed. Please install it with '
                                           '`pip install torch` or `pip3 install torch`')

        super(AutodiffComposition, self).__init__(name = name,
                                                  learning_rate = learning_rate,
                                                  optimizer_type = optimizer_type,
                                                  weight_decay = weight_decay,
                                                  loss_spec = loss_spec,
                                                  pathways=pathways,
                                                  **kwargs)

        self.optimizer_type = optimizer_type
        self.loss_spec = loss_spec
        self.refresh_losses = refresh_losses
        self._built_pathways = False
        self.weight_decay = weight_decay
        self.force_no_retain_graph = force_no_retain_graph
        self.loss = None
        self.disable_learning = disable_learning
        self._runtime_learning_rate = None
        self.last_saved_weights = None
        self.last_loaded_weights = None

        # keeps track of average loss per epoch
        self.losses = []

        # ordered execution sets for the pytorch model
        self.execution_sets = None

        if not disable_cuda and torch.cuda.is_available():
            if cuda_index is None:
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cuda:' + str(cuda_index))
        else:
            self.device = torch.device('cpu')

    # CLEANUP: move some of what's done in the methods below to a "validate_params" type of method
    @handle_external_context()
    def _build_pytorch_representation(self, context=None, refresh=False):
        if self.scheduler is None:
            self.scheduler = Scheduler(graph=self.graph_processing)
        if self.parameters.pytorch_representation._get(context=context) is None or refresh:
            model = PytorchModelCreator(composition=self,
                                        device=self.device,
                                        context=context)

            self.parameters.pytorch_representation._set(model, context, skip_history=True, skip_log=True)

        # Set up optimizer function
        old_opt = self.parameters.optimizer._get(context)
        learning_rate = self._runtime_learning_rate or self.learning_rate
        if old_opt is None or refresh:
            opt = self._make_optimizer(self.optimizer_type, learning_rate, self.weight_decay, context)
            self.parameters.optimizer._set(opt, context, skip_history=True, skip_log=True)

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
        if not isinstance(self.loss_spec, (str, Loss)):
            return self.loss_spec
        elif loss_spec == Loss.MSE:
            return nn.MSELoss(reduction='mean')
        elif loss_spec == Loss.SSE:
            return nn.MSELoss(reduction='sum')
        elif loss_spec == Loss.CROSS_ENTROPY:
            if version.parse(torch.version.__version__) >= version.parse('1.12.0'):
                return nn.CrossEntropyLoss()

            # Cross entropy loss is used for multiclass categorization and needs inputs in shape
            # ((# minibatch_size, C), targets) where C is a 1-d vector of probabilities for each potential category
            # and where target is a 1d vector of type long specifying the index to the target category. This
            # formatting is different from most other loss functions available to autodiff compositions,
            # and therefore requires a wrapper function to properly package inputs.
            return lambda x, y: nn.CrossEntropyLoss()(torch.atleast_2d(x), torch.atleast_2d(y.type(x.type())))
        elif loss_spec == Loss.L1:
            return nn.L1Loss(reduction='sum')
        elif loss_spec == Loss.NLL:
            return nn.NLLLoss(reduction='sum')
        elif loss_spec == Loss.POISSON_NLL:
            return nn.PoissonNLLLoss(reduction='sum')
        elif loss_spec == Loss.KL_DIV:
            return nn.KLDivLoss(reduction='sum')
        else:
            raise AutodiffCompositionError(f"Loss type {loss_spec} not recognized. Loss argument must be a "
                                           f"Loss enum or function. Currently, the recognized loss types are: "
                                           f"L1 (Mean), SSE (sum squared error), CROSS_ENTROPY, NLL (negative log "
                                           f"likelihood), POISSONNLL (Poisson negative log likelihood, "
                                           f"and KL_DIV (KL divergence.")

    # performs learning/training on all input-target pairs it recieves for given number of epochs
    def autodiff_training(self, inputs, targets, context=None, scheduler=None):

        # compute total loss across output neurons for current trial
        tracked_loss = self.parameters.tracked_loss._get(context)
        if tracked_loss is None:
            self.parameters.tracked_loss._set(torch.zeros(1, device=self.device).double(),
                                              context=context,
                                              skip_history=True,
                                              skip_log=True)
            tracked_loss = self.parameters.tracked_loss._get(context)

        curr_tensor_inputs = {}
        curr_tensor_targets = {}
        for component in inputs.keys():
            input = inputs[component][0]
            curr_tensor_inputs[component] = torch.tensor(input, device=self.device).double()
        for component in targets.keys():
            target = targets[component][0]
            curr_tensor_targets[component] = torch.tensor(target, device=self.device).double()

        # do forward computation on current inputs
        curr_tensor_outputs = self.parameters.pytorch_representation._get(context).forward(curr_tensor_inputs,
                                                                                           context,
                                                                                           )

        for component in curr_tensor_outputs.keys():
            # possibly add custom loss option, which is a loss function that takes many args
            # (outputs, targets, weights, and more) and returns a scalar
            new_loss = self.loss(curr_tensor_outputs[component], curr_tensor_targets[component])
            tracked_loss += new_loss

        outputs = []
        for input_port in self.output_CIM.input_ports:
            assert (len(input_port.all_afferents) == 1)  # CW 12/05/18, this assert may eventually be outdated
            component = input_port.all_afferents[0].sender.owner
            outputs.append(curr_tensor_outputs[component].detach().cpu().numpy().copy())

        self.parameters.tracked_loss_count._set(self.parameters.tracked_loss_count._get(context=context) + 1,
                                                context=context,
                                                skip_history=True,
                                                skip_log=True)
        return outputs

    def clear_losses(self, context=None):
        self.losses = []
        self.parameters.losses.set([], context=context)

    def _update_learning_parameters(self, context):
        """
        Updates parameters (weights) based on trials run since last update.
        """
        optimizer = self.parameters.optimizer._get(context=context)
        optimizer.zero_grad()

        tracked_loss = self.parameters.tracked_loss._get(context=context) / self.parameters.tracked_loss_count._get(context=context)
        if self.force_no_retain_graph:
            tracked_loss.backward(retain_graph=False)
        else:
            tracked_loss.backward(retain_graph=True)
        self.parameters.losses._get(context=context).append(tracked_loss.detach().cpu().numpy()[0])
        self.parameters.tracked_loss._set(torch.zeros(1, device=self.device).double(), context=context, skip_history=True, skip_log=True)
        self.parameters.tracked_loss_count._set(0, context=context, skip_history=True, skip_log=True)
        optimizer.step()
        self.parameters.pytorch_representation._get(context=context).detach_all()
        self.parameters.pytorch_representation._get(context).copy_weights_to_psyneulink(context)

    def _gen_llvm_function(self, *, ctx:pnlvm.LLVMBuilderContext, tags:frozenset):
        if "run" in tags:
            return pnlvm.codegen.gen_composition_run(ctx, self, tags=tags)
        else:
            return pnlvm.codegen.gen_autodiffcomp_exec(ctx, self, tags=tags)

    def _get_total_loss(self, num_trials: int=1, context:Context=None):
        return sum(self.parameters.trial_losses._get(context)[-num_trials:]) /num_trials

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
            if NodeRole.INPUT in self.get_roles_by_node(node) and NodeRole.TARGET not in self.get_roles_by_node(node):
                ret[node] = values
        return ret

    def learn(self, *args, **kwargs):
        if self._built_pathways is False:
            self.infer_backpropagation_learning_pathways()
            self._built_pathways = True

        if 'execution_mode' in kwargs:
            execution_mode = kwargs['execution_mode']
            if execution_mode == pnlvm.ExecutionMode.Python:
                raise AutodiffCompositionError(f"{self.name} is an AutodiffComposition so its learn() "
                                               f"cannot be called with execution_mode = ExecutionMode.Python; "
                                               f"use ExecutionMode.PyTorch or ExecutionMode.LLVMRun.")
            # OK, now that the user has been advised to use ExecutionMode.PyTorch and warned *not* to ExecutionMdoe.Python,
            #     convert ExecutionMode.PyTorch specification to ExecutionMode.Python for internal use (nice, eh?)
            if execution_mode == pnlvm.ExecutionMode.PyTorch:
                kwargs['execution_mode'] = pnlvm.ExecutionMode.Python

        return super().learn(*args, **kwargs)

    @handle_external_context()
    def execute(self,
                inputs=None,
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
                reset_stateful_functions_to=None,
                context=None,
                base_context=Context(execution_id=None),
                clamp_input=SOFT_CLAMP,
                targets=None,
                runtime_params=None,
                execution_mode:pnlvm.ExecutionMode = pnlvm.ExecutionMode.PyTorch,
                skip_initialization=False,
                report_output:ReportOutput=ReportOutput.OFF,
                report_params:ReportOutput=ReportParams.OFF,
                report_progress:ReportProgress=ReportProgress.OFF,
                report_simulations:ReportSimulations=ReportSimulations.OFF,
                report_to_devices:ReportDevices=None,
                report=None,
                report_num=None,
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

            report(self,
                   LEARN_REPORT,
                   # EXECUTE_REPORT,
                   report_num=report_num,
                   scheduler=scheduler,
                   content='trial_start',
                   context=context)

            self._build_pytorch_representation(context)
            output = self.autodiff_training(autodiff_inputs,
                                            autodiff_targets,
                                            context,
                                            scheduler)

            # FIX 5/28/20:
            # context.add_flag(ContextFlags.PROCESSING)
            execution_phase = context.execution_phase
            context.execution_phase = ContextFlags.PROCESSING

            self.output_CIM.execute(output, context=context)
            # FIX 5/28/20:
            context.execution_phase = execution_phase

            report(self,
                   # [LEARN_REPORT],
                   [EXECUTE_REPORT, PROGRESS_REPORT],
                   report_num=report_num,
                   scheduler=scheduler,
                   content='trial_end',
                   context=context)

            scheduler.get_clock(context)._increment_time(TimeScale.TRIAL)

            return output

        return super(AutodiffComposition, self).execute(inputs=inputs,
                                                        scheduler=scheduler,
                                                        termination_processing=termination_processing,
                                                        call_before_time_step=call_before_time_step,
                                                        call_before_pass=call_before_pass,
                                                        call_after_time_step=call_after_time_step,
                                                        call_after_pass=call_after_pass,
                                                        reset_stateful_functions_to=reset_stateful_functions_to,
                                                        context=context,
                                                        base_context=base_context,
                                                        clamp_input=clamp_input,
                                                        runtime_params=runtime_params,
                                                        execution_mode=execution_mode,
                                                        report=report,
                                                        report_num=report_num
                                                        )

    @handle_external_context(fallback_most_recent=True)
    def save(self, path:PosixPath=None, directory:str=None, filename:str=None, context=None):
        """Saves all weight matrices for all MappingProjections in the AutodiffComposition

        Arguments
        ---------
        path: Path, PosixPath or str : default None
            path specification; must be a legal path specification in the filesystem.
        directory: str : default ``current working directory``
            directory where `matrices <MappingProjection.matrix>` for all MappingProjections
            in the AutodiffComposition are saved.
        filename: str : default ``<name of AutodiffComposition>_matrix_wts.pnl``
            filename in which `matrices <MappingProjection.matrix>` for all MappingProjections
            in the AutodiffComposition are saved.
        .. note::
           Matrices are saved in
           `PyTorch state_dict <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`_ format.

        Return
        ------
        Path

        """
        error_msg = f" (for saving weight matrices for '{self.name}') is not a legal path."

        if path:
            try:
                path = Path(path)
            except:
                raise AutodiffCompositionError(f"'{path}'{error_msg}")
        else:
            try:
                if directory:
                    path = Path(directory)
                else:
                    path = Path(os.getcwd())
                if filename:
                    path = Path(os.path.join(path, filename))
                else:
                    path = Path(os.path.join(path, f'{self.name}_matrix_wts.pnl'))
            except IsADirectoryError:
                raise AutodiffCompositionError(f"'{path}'{error_msg}")
        proj_state = {
            p.name: p.parameters.matrix.get(context=context)
            # p.name: p.matrix.base
            for p in self.projections
            if not (isinstance(p, ModulatoryProjection_Base)
                    or isinstance(p.sender.owner, CompositionInterfaceMechanism)
                    or isinstance(p.receiver.owner, CompositionInterfaceMechanism)
                    or isinstance(p.sender.owner, ModulatoryMechanism_Base)
                    or isinstance(p.receiver.owner, ModulatoryMechanism_Base)
                    or p.sender.owner in self.get_nodes_by_role(NodeRole.LEARNING)
                    or p.receiver.owner in self.get_nodes_by_role(NodeRole.LEARNING)
                )}
        try:
            torch.save(proj_state, path)
        except IsADirectoryError:
            raise AutodiffCompositionError(f"'{path}'{error_msg}")

        self.last_saved_weights = path

        return path

    @handle_external_context(fallback_most_recent=True)
    def load(self, path:PosixPath=None, directory:str=None, filename:str=None, context=None):
        """Loads all weight matrices for all MappingProjections in the AutodiffComposition from file
        Arguments
        ---------
        path: Path : default None
            Path for file in which `MappingProjection` `matrices <MappingProjection.matrix>` are stored.
            This must be a legal PosixPath object; if it is specified **directory** and **filename** are ignored.
        directory: str : default ``current working directory``
            directory where `MappingProjection` `matrices <MappingProjection.matrix>` are stored.
        filename: str : default ``<name of AutodiffComposition>_matrix_wts.pnl``
            name of file in which `MappingProjection` `matrices <MappingProjection.matrix>` are stored.
        .. note::
           Matrices must be stored in
           `PyTorch state_dict <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`_ format.
        """
        error_msg = f" (for loading weight matrices for '{self.name}') is not a legal path."
        if path:
            if not isinstance(path,Path):
                raise AutodiffCompositionError(f"'{path}'{error_msg}")
        else:
            try:
                if directory:
                    path = Path(directory)
                else:
                    path = Path(os.getcwd())
                if filename:
                    path = Path(os.path.join(path, filename))
                else:
                    path = Path(os.path.join(path , f'{self.name}_matrix_wts.pnl'))
            except IsADirectoryError:
                raise AutodiffCompositionError(f"'{path}'{error_msg}")
        try:
            state = torch.load(path)
        except FileNotFoundError:
            raise AutodiffCompositionError(f"'{path}'{error_msg}")

        self.last_loaded_weights = path

        for projection in [p for p in self.projections
                           if not (isinstance(p, ModulatoryProjection_Base)
                                   or isinstance(p.sender.owner, CompositionInterfaceMechanism)
                                   or isinstance(p.receiver.owner, CompositionInterfaceMechanism)
                                   or isinstance(p.sender.owner, ModulatoryMechanism_Base)
                                   or isinstance(p.receiver.owner, ModulatoryMechanism_Base)
                                   or p.sender.owner in self.get_nodes_by_role(NodeRole.LEARNING)
                                   or p.receiver.owner in self.get_nodes_by_role(NodeRole.LEARNING)
            )]:
            matrix = state[projection.name]
            if np.array(matrix).shape != projection.matrix.base.shape:
                raise AutodiffCompositionError(f"Shape of matrix loaded for '{projection.name}' "
                                               f"({np.array(matrix).shape}) "
                                               f"does not match its shape ({projection.matrix.base.shape})")
            projection.matrix.base = matrix
            projection.parameters.matrix.set(matrix, context=context, override=True)
            projection.parameter_ports['matrix'].parameters.value.set(matrix, context=context, override=True)

        self._build_pytorch_representation(context=context, refresh=True)

    def _get_state_ids(self):
        return super()._get_state_ids() + ["optimizer"]

    def _get_state_struct_type(self, ctx):
        comp_state_type_list = ctx.get_state_struct_type(super())
        pytorch_representation = self._build_pytorch_representation()
        optimizer_state_type = pytorch_representation._get_compiled_optimizer()._get_optimizer_struct_type(ctx)

        return pnlvm.ir.LiteralStructType((
            *comp_state_type_list,
            optimizer_state_type))

    def _get_state_initializer(self, context):
        comp_states = super()._get_state_initializer(context)
        optimizer_states = tuple()

        return (*comp_states, optimizer_states)
