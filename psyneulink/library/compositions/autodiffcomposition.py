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
      - `AutodiffComposition_Logging`
      - `AutodiffComposition_Nested_Execution`
  * `AutodiffComposition_Class_Reference`


.. _AutodiffComposition_Overview:

Overview
--------

.. warning:: As of PsyNeuLink 0.7.5, the API for using AutodiffCompositions has been slightly changed! Please see `this link <RefactoredLearningGuide>` for more details!

AutodiffComposition is a subclass of `Composition` used to train feedforward neural network models through integration
with `PyTorch <https://pytorch.org/>`_, a popular machine learning library, which executes considerably more quickly
than using the `standard implementation of learning <Composition_Learning_Standard>` in a Composition, using its
`learning methods <Composition_Learning_Methods>`. An AutodiffComposition is configured and run similarly to a standard
Composition, with some exceptions that are described below.
COMMENT:
FIX: UPDATE runtimes WITH COMPILED VERSION
COMMENT

.. _AutodiffComposition_Creation:

Creating an AutodiffComposition
-------------------------------

An AutodiffComposition can be created by calling its constructor, and then adding `Components <Component>` using the
standard `Composition methods <Composition_Creation>` for doing so.  The constructor also includes an number of
parameters that are specific to the AutodiffComposition. See the <class reference `AutodiffComposition`> for a list of these parameters.

.. warning:: Mechanisms or Projections should not be added to or deleted from an AutodiffComposition after it has
   been run for the first time. Unlike an ordinary Composition, AutodiffComposition does not support this
   functionality.

.. warning:: When comparing models built in PyTorch to those using AutodiffComposition,
   the `bias <https://www.pytorch.org/docs/stable/nn.html#torch.nn.Module>` parameter of PyTorch modules should be set to `False`, as AutodiffComposition does not currently support trainable biases.


.. _AutodiffComposition_Execution:

Execution
---------

An AutodiffComposition's `run <Composition.run>`, `execute <Composition.execute>`, and `learn <Composition.learn>` methods are the same as for a `Composition`.

The following is an example showing how to create a
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
    >>> # Run Composition in learnng mode
    >>> my_autodiff.learn(inputs = input_dict)
    >>> # Run Composition in test mode
    >>> my_autodiff.run(inputs = input_dict['inputs'])

.. _AutodiffComposition_Logging:

Logging
~~~~~~~

Logging in AutodiffCompositions follows the same procedure as `logging in a Composition <Log>`. However, since an AutodiffComposition internally converts all of its mechanisms to an equivalent PyTorch model,
then its inner components are not actually executed. This means that there is limited support for logging parameters of components inside an AutodiffComposition;
Currently, the only supported parameters are:

1) the `matrix` parameter of Projections

2) the `value` parameter of its inner components

.. _AutodiffComposition_Nested_Execution:

Nested Execution
~~~~~~~~~~~~~~~~

Like any other `Composition`, an AutodiffComposition may be `nested inside another <Composition_Nested>`.

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
import numpy as np

import logging
try:
    import torch
    from torch import nn
    import torch.optim as optim
    torch_available = True
except ImportError:
    torch_available = False
else:
    from psyneulink.library.compositions.pytorchmodelcreator import PytorchModelCreator


from psyneulink.core.components.functions.transferfunctions import Linear, Logistic, ReLU
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.compositions.composition import Composition, NodeRole
from psyneulink.core.compositions.composition import CompositionError
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import SOFT_CLAMP, TRAINING_SET
from psyneulink.core.scheduling.scheduler import Scheduler
from psyneulink.core.globals.parameters import Parameter
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
    Subclass of `Composition` that trains models using `PyTorch <https://pytorch.org>`_.
    See `Composition <Composition_Class_Reference>` for additional arguments and attributes.

    Arguments
    ---------

    learning_rate : float : default 0.001
        the learning rate, which is passed to the optimizer.

    disable_learning : bool: default False
        specifies whether the AutodiffComposition should disable learning when run in `learning mode
        <Composition.learn>`.

    optimizer_type : str : default 'sgd'
        the kind of optimizer used in training. The current options are 'sgd' or 'adam'.

    weight_decay : float : default 0
        specifies the L2 penalty (which discourages large weights) used by the optimizer.

    loss_spec : str or PyTorch loss function : default 'mse'
        specifies the loss function for training. The current string options are 'mse' (the default), 'crossentropy',
        'l1', 'nll', 'poissonnll', and 'kldiv'. Any PyTorch loss function can work here, such as ones from
        https://pytorch.org/docs/stable/nn.html#loss-functions

    losses : list of floats
        tracks the average for each weight update (i.e. each minibatch)

    optimizer : PyTorch optimizer function
        the optimizer used for training. Depends on the **optimizer_type**, **learning_rate**, and **weight_decay**
        arguments from initialization.

    loss : PyTorch loss function
        the loss function used for training. Depends on the **loss_spec** argument from initialization.

    """

    class Parameters(Composition.Parameters):
        """"""
        optimizer = None
        learning_rate = Parameter(.001, fallback_default=True)
        losses = Parameter([])
        trial_losses = Parameter([])
        tracked_loss = Parameter(None, pnl_internal=True)
        tracked_loss_count = Parameter(0, pnl_internal=True)
        pytorch_representation = None

    # TODO (CW 9/28/18): add compositions to registry so default arg for name is no longer needed
    def __init__(self,
                 learning_rate=None,
                 optimizer_type='sgd',
                 weight_decay=0,
                 loss_spec='mse',
                 disable_learning=False,
                 refresh_losses=False,
                 disable_cuda=True,
                 cuda_index=None,
                 force_no_retain_graph=False,
                 name="autodiff_composition"):

        if not torch_available:
            raise AutodiffCompositionError('Pytorch python module (torch) is not installed. Please install it with '
                                           '`pip install torch` or `pip3 install torch`')

        super(AutodiffComposition, self).__init__(name = name,
                                                  learning_rate = learning_rate,
                                                  optimizer_type = optimizer_type,
                                                  weight_decay = weight_decay,
                                                  loss_spec = loss_spec)

        self.optimizer_type = optimizer_type
        self.loss_spec = loss_spec
        self.refresh_losses = refresh_losses
        self._built_pathways = False
        self.weight_decay = weight_decay
        self.force_no_retain_graph = force_no_retain_graph
        self.loss = None
        self.disable_learning = disable_learning

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
    def _build_pytorch_representation(self, context=None):
        if self.scheduler is None:
            self.scheduler = Scheduler(graph=self.graph_processing)
        if self.parameters.pytorch_representation._get(context=context) is None:
            model = PytorchModelCreator(composition=self,
                                        device=self.device,
                                        context=context)

            self.parameters.pytorch_representation._set(model, context, skip_history=True, skip_log=True)

        # Set up optimizer function
        old_opt = self.parameters.optimizer._get(context)
        if old_opt is None:
            opt = self._make_optimizer(self.optimizer_type, self.learning_rate, self.weight_decay, context)
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

    # performs learning/training on all input-target pairs it recieves for given number of epochs
    def autodiff_training(self, inputs, targets, context=None, scheduler=None):

        # compute total loss across output neurons for current trial
        tracked_loss = self.parameters.tracked_loss._get(context)
        if tracked_loss is None:
            self.parameters.tracked_loss._set(torch.zeros(1, device=self.device).double(), context=context, skip_history=True, skip_log=True)
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
        curr_tensor_outputs = self.parameters.pytorch_representation._get(context).forward(
            curr_tensor_inputs,
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

        self.parameters.tracked_loss_count._set(self.parameters.tracked_loss_count._get(context=context) + 1, context=context, skip_history=True, skip_log=True)
        return outputs

    def clear_losses(self, context=None):
        self.losses = []
        self.parameters.losses.set([], context=context)

    def _update_learning_parameters(self, context):
        """
        Updates parameters based on trials ran since last update.
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
                                            scheduler)

            # FIX 5/28/20:
            # context.add_flag(ContextFlags.PROCESSING)
            execution_phase = context.execution_phase
            context.execution_phase = ContextFlags.PROCESSING

            self.output_CIM.execute(output, context=context)
            # FIX 5/28/20:
            context.execution_phase = execution_phase


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
                                                        bin_execute=bin_execute,
                                                        )

    def _get_state_struct_type(self, ctx):
        node_state_type_list = (ctx.get_state_struct_type(m) for m in self._all_nodes)
        proj_state_type_list = (ctx.get_state_struct_type(p) for p in self._inner_projections)
        comp_state_type_list = ctx.get_state_struct_type(super())
        pytorch_representation = self._build_pytorch_representation()
        optimizer_state_type = pytorch_representation._get_compiled_optimizer()._get_optimizer_struct_type(ctx)

        return pnlvm.ir.LiteralStructType((
            pnlvm.ir.LiteralStructType(node_state_type_list),
            pnlvm.ir.LiteralStructType(proj_state_type_list),
            *comp_state_type_list,
            optimizer_state_type))

    def _get_state_initializer(self, context):
        node_states = (m._get_state_initializer(context=context) for m in self._all_nodes)
        proj_states = (p._get_state_initializer(context=context) for p in self._inner_projections)
        comp_states = super()._get_state_initializer(context)
        optimizer_states = tuple()

        return (tuple(node_states), tuple(proj_states), *comp_states, optimizer_states)
