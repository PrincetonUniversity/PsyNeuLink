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
    - `AutodiffComposition_`
      - `AutodiffComposition_Modulatory_Mechanisms`
      - `AutodiffComposition_Bias_Parameters`
      - `AutodiffComposition_Nesting`
      - `AutodiffComposition_Post_Construction_Modification`
    * `AutodiffComposition_Execution`
      - `AutodiffComposition_PyTorch`
      - `AutodiffComposition_LLVM`
      - `AutodiffComposition_Python`
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
<Composition.add_linear_processing_pathway>`, etc.). The constructor also includes a number of parameters that are
specific to the AutodiffComposition (see `AutodiffComposition_Class_Reference` for a list of these parameters,
and `examples <AutodiffComposition_Examples>` below). While an AutodiffComposition can generally be created using the
same methods as a standard Composition, there are a few restrictions that apply to its construction, summarized below.

.. _AutodiffComposition_Restrictions:

.. _AutodiffComposition_Modulatory_Mechanisms:

*Only one OutputPort per Node*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `Nodes <Composition_Nodes>` of an AutodiffComposition currently can have only *one* `OutputPort`, though that
can have more than one `efferent <Port_Base.efferents>` `MappingProjection`.  Nodes can also have more than one
`InputPort`, that can receive more than one `afferent `path_afferent <Port_Base.path_afferents>` Projections.

*No Modulatory Components*
~~~~~~~~~~~~~~~~~~~~~~~~~~

All of the Components in an AutodiffComposition must be able to be subjected to `learning <Composition_Learning>`,
which means that no `ModulatoryMechanisms <ModulatoryMechanism>` can be included in an AutodiffComposition.
Specifically, this precludes any `learning components <Composition_Learning_Components>`, `ControlMechanisms
<ControlMechanism>`, or a `controller <Composition_Controller>`.

.. _Autodiff_Learning_Components_Warning:

*Learning Components.*  An AutodiffComposition **cannot include any** `learning components
<Composition_Learning_Components>` themselves (i.e., `LearningMechanisms <LearningMechanism>`, `LearningSignals
<LearningSignal>`, or LearningProjections <LearningProjection>`, nor the `ComparatorMechanism <COMPARATOR_MECHANISM>`
or `ObjectiveMechanism <OBJECTIVE_MECHANISM>` used to compute the loss for learning). These are constructed
automatically when learning is executed in `Python mode <AutodiffComposition_Python>` or `LLVM mode
<AutodiffComposition_LLVM>`, and PyTorch-compatible Components are constructed when it is executed in `PyTorch mode
<AutodiffComposition_PyTorch>`.

COMMENT:
FIX: IS THE FOLLOWING STILL TRUE? SEEMS TO CONTRADICT STATEMENTS BELOW:
This means that it cannot be used with a Composition that contains any `modulatory components
<ModulatorySignal_Anatomy_Figure>` or ones that are subject to modulation, whether by ModulatoryMechanisms within or
outside the Composition;
?MAYBE THE FOLLOWING IS BETTER:
COMMENT
*Control Components.*  An AutodiffComposition also cannot include any `ControlMechanisms <ControlMechanism>` or a
`controller <Composition_Controller>`.  However, it *can* include Mechanisms that are subject to modulatory control
(see `Figure <ModulatorySignal_Anatomy_Figure>`, and `modulation <ModulatorySignal_Modulation>`) by ControlMechanisms
*outside* the Composition, including the controller of a Composition within which the AutodiffComposition is nested.
That is, an AutodiffComposition can be `nested in a Composition <Composition_Nested>` that has such other Components
(see `AutodiffComposition_Nested_Modulation` below).

.. _AutodiffComposition_Bias_Parameters:

*No Bias Parameters*
~~~~~~~~~~~~~~~~~~~~

AutodiffComposition does not (currently) support the *automatic* construction of separate bias parameters.
Thus, when constructing a model using an AutodiffComposition that corresponds to one in PyTorch, the `bias
<https://www.pytorch.org/docs/stable/nn.html#torch.nn.Module>` parameter of PyTorch modules should be set
to `False`. Trainable biases *can* be specified explicitly in an AutodiffComposition by including a
TransferMechanism that projects to the relevant Mechanism (i.e., implementing that layer of the network to
receive the biases) using a `MappingProjection` with a `matrix <MappingProjection.matrix>` parameter that
implements a diagnoal matrix with values corresponding to the initial value of the biases.

.. _AutodiffComposition_Nesting:

*Nesting*
~~~~~~~~~

An AutodiffComposition can be `nested <Composition_Nested>` inside another Composition for learning, and there can
be any level of such nestings.  However, all of the nested Compositions must be AutodiffCompositions. Furthermore, all
nested Compositions use the `learning_rate <AutodiffComposition.learning_rate>` specified for the outermost Composition,
whether this is specified in the call to its `learn <AutodiffComposition.learn>` method, its constructor, or its
default value is being used (see `learning_rate <AutodiffComposition.learning_rate>` below for additional details).

.. technical_note::
   Projections from `Nodes <Composition_Nodes>` in an immediately enclosing outer Composition to the `input_CIM
   <Composition.input_CIM>` of a nested Composition, and from its `output_CIM <Composition.output_CIM>` to Nodes
   in the outer Composition are subject to learning;  however those within the nested Composition itself (i.e.,
   from its input_CIM to its INPUT Nodes and from its OUTPUT Nodes to its output_CIM) are *not* subject to learning,
   as they serve simply as conduits of information between the outer Composition and the nested one.

.. warning::
   Nested Compositions are supported for learning only in `PyTorch mode <AutodiffComposition_PyTorch>`, and will
   cause an error if the `learn <AutodiffComposition.learn>` method of an AutodiffComposition is executed in
   `Python mode <AutodiffComposition_Python>` or `LLVM mode <AutodiffComposition_LLVM>`.

.. _AutodiffComposition_Post_Construction_Modification:

*No Post-construction Modification*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
COMMENT:
IS THIS STILL TRUE?
COMMENT
Mechanisms or Projections should not be added to or deleted from an AutodiffComposition after it has
been executed. Unlike an ordinary Composition, AutodiffComposition does not support this functionality.


.. _AutodiffComposition_Execution:

Execution
---------

An AutodiffComposition's `run <Composition.run>`, `execute <Composition.execute>`, and `learn <Composition.learn>`
methods are the same as for a `Composition`.  However, the **execution_mode** in the `learn <Composition.learn>`
method has different effects than for a standard Composition, that determine whether it uses `LLVM compilation
<AutodiffComposition_LLVM>` or `translation to PyTorch <AutodiffComposition_PyTorch>` to execute learning.
These are each described in greater detail below, and summarized in this `table <Composition_Compilation_Table>`
which provides a comparison of the different modes of execution for an AutodiffComposition and standard `Composition`.

.. _AutodiffComposition_PyTorch:

*PyTorch mode*
~~~~~~~~~~~~~~

This is the default for an AutodiffComposition, but, can be specified explicitly by setting **execution_mode** =
`ExecutionMode.PyTorch` in the `learn <Composition.learn>` method (see `example <BasicsAndPrimer_Rumelhart_Model>`
in `BasicsAndPrimer`).  In this mode, the AutodiffComposition is automatically translated to a `PyTorch
<https://pytorch.org>`_ model for learning.  This is comparable in speed to `LLVM compilation
<_AutodiffComposition_LLVM>`, but provides greater flexiblity, including the ability to include nested
AutoDiffCompositions in learning. Although it is best suited for use with `supervised learning
<Composition_Learning_Supervised>`, it can also be used for some forms of `unsupervised learning
<Composition_Learning_Unsupervised>` that are supported in PyTorch (e.g., `self-organized maps
<https://github.com/giannisnik/som>`_).

    .. _AutodiffComposition_PyTorch_Note:

    .. note::
       While specifying `ExecutionMode.PyTorch` in the `learn <Composition.learn>`  method of an AutodiffComposition
       causes it to use PyTorch for training, specifying this in the `run <Compositon.run>` method causes it to be
       executed using the *Python* interpreter (and not PyTorch);  this is so that any modulation can take effect
       during execution (see `AutodiffComposition_Nested_Modulation` below), which is not supported by PyTorch.

    .. warning::
      * Specifying `ExecutionMode.LLVM` or `ExecutionMode.PyTorch` in the learn() method of a standard
        `Composition` causes an error.

COMMENT:
FIX: ADD MENTION OF TARGET NODES AND PYTORCH WRAPPERS
COMMENT

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


COMMENT:
FIX: 8/13/23 - COMPLETE DOCS HERE
COMMENT

.. _AutodiffComposition_Python:

*Python mode*
~~~~~~~~~~~~~
An AutodiffComposition can also be run using the standard PsyNeuLink learning components.  However, this cannot
be used if the AutodiffComposition has any nested Compositions, irrespective of whether they are ordinary
Compositions or AutodiffCompositions.


.. _AutodiffComposition_Nested_Modulation:

*Nested Execution and Modulation*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# FIX:
Like any other `Composition`, an AutodiffComposition may be `nested <Composition_Nested>` inside another
(see `example <AutodiffComposition_Nested_Example>` below).  However, during learning, none of the internal
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
import collections
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
    from psyneulink.library.compositions.pytorchwrappers import PytorchCompositionWrapper

from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.core.components.mechanisms.modulatory.modulatorymechanism import ModulatoryMechanism_Base
from psyneulink.core.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.compositions.composition import Composition, NodeRole, CompositionError
from psyneulink.core.compositions.report import (ReportOutput, ReportParams, ReportProgress, ReportSimulations,
                                                 ReportDevices, EXECUTE_REPORT, LEARN_REPORT, PROGRESS_REPORT)
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context, CONTEXT
from psyneulink.core.globals.keywords import AUTODIFF_COMPOSITION, SOFT_CLAMP, Loss
from psyneulink.core.globals.utilities import is_numeric_scalar
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
        specifies the learning rate passed to the optimizer if none is specified in the `learn
        <AutdodiffComposition.learn>` method of the AutodiffComposition
        (see `learning_rate <AutodiffComposition.learning_rate>` for additional details).

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

    learning_rate : float
        determines the learning_rate passed the optimizer, and is applied to all `Projection`\\s in the
        AutodiffComposition that are `learnable <MappingProjection.learnable>`.

        .. note::
           At present, the same learning rate is applied to all Components of an AutodiffComposition, irrespective
           of the `learning_rate <`learning_rate <LearningMechanism.learning_rate>` that may be specified for any
           individual Mechanisms or any `nested Compositions <AutodiffComposition_Nesting>`; in the case of the
           latter, the `learning_rate <AutodiffComposition.learning_rate>` of the outermost AutodiffComposition is
           used, whether this is specified in the call to its `learn <AutodiffComposition.learn>` method, its
           constructor, or its default value is being used.

        .. hint::
           To disable updating of a particular `MappingProjection` in an AutodiffComposition, specify the
           **learnable** parameter of its constructor as `False`; this applies to MappingProjections at any
           level of `nesting <AutodiffComposition_Nesting>`.

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
    if torch_available:
        from psyneulink.library.compositions.pytorchEMcompositionwrapper import PytorchCompositionWrapper
        pytorch_composition_wrapper_type = PytorchCompositionWrapper

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

        # if not torch_available:
        #     raise AutodiffCompositionError('Pytorch python module (torch) is not installed. Please install it with '
        #                                    '`pip install torch` or `pip3 install torch`')

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
        self.target_output_map = {}

        # keeps track of average loss per epoch
        self.losses = []

        # ordered execution sets for the pytorch model
        self.execution_sets = None

        if not disable_cuda and torch.cuda.is_available():
            if cuda_index is None:
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cuda:' + str(cuda_index))
        elif torch_available:
            self.device = torch.device('cpu')

        # Set to True after first warning about failure to specify execution mode so warning is issued only once
        self.execution_mode_warned_about_default = False

    def infer_backpropagation_learning_pathways(self, execution_mode, context=None):
        """Create backpropapagation learning pathways for every Input Node --> Output Node pathway
        Flattens nested compositions:
          - only includes the Projections in outer Composition to/from the CIMs of the nested Composition
            (i.e., to input_CIMs and from output_CIMs) -- the ones that should be learned;
          - excludes Projections from/to CIMs in the nested Composition
            (from input_CIMs and to output_CIMs), as those should remain identity Projections;
          see `PytorchCompositionWrapper` for table of how Projections are handled and further details.
        """

        self._analyze_graph()

        def _get_pytorch_backprop_pathway(input_node)->list:
            """Breadth-first search from input_node to find all input -> output pathways
            Uses queue(node, input_port, composition) to traverse all nodes in the graph
            IMPLEMENTATION NOTE:  flattens nested Compositions
            Return a list of all pathways from input_node -> output node
            """
            pathways = []
            prev = {}
            queue = collections.deque([(input_node, None, self)])

            # FIX:  9/17/23 - THIS VERSION FLATTENS NESTED COMPOSITIONS;  MAY NOT STILL BE NEEDED
            #                 SINCE EXECUTION SETS ARE NOW FLATTENED IN PytorchCompositionWrapper
            #                 ?? REVERT TO OLD VERSION (IN PRE-"CLEAN_UP" VERSIONS, OR ON DEVEL?),
            #                 THOUGH DOING SO PREVIOUSLY SEEMED TO LOSE TARGET NODE.
            #                 MAYBE NOT NOW THAT THEY ARE CONSTRUCTED EXPLICITLY BELOW?
            def create_pathway(node)->list:
                pathway = []
                entry = node
                while entry in prev:
                    pathway.insert(0, entry)
                    entry = prev[entry]
                pathway.insert(0, entry)
                # Only consider input -> projection -> ... -> output pathways
                # (since can't learn on only one mechanism)
                if len(pathway) >= 3:
                    return pathway
                else:
                    return []

            # breadth-first search starting with input node
            while len(queue) > 0:
                node, input_port, current_comp = queue.popleft()

                if (isinstance(node, Composition) and node is not self
                        and any(isinstance(proj.sender.owner, CompositionInterfaceMechanism)
                                for proj in node.afferents)):
                    for output_port in node.input_CIM.output_ports:
                        for proj in output_port.efferents:
                            queue.append((proj.receiver.owner, proj.receiver, node))
                    continue

                # node is output_CIM of outer Composition (i.e., end of pathway)
                if isinstance(node, CompositionInterfaceMechanism) and node is self.output_CIM:
                    assert False, (f"PROGRAM ERROR: 'Got to output_CIM of outermost Composition '({self.name})' "
                                   f"without detecting OUTPUT NODE at end of pathway")

                # End of pathway: OUTPUT Node of outer Composition
                if current_comp == self and node in current_comp.get_nodes_by_role(NodeRole.OUTPUT):
                    pathways.append(create_pathway(node))
                    continue

                # Consider all efferent Projections of node
                for efferent_proj, rcvr in [(p, p.receiver.owner)
                                            for p in node.efferents
                                            if p in current_comp.projections]:

                    # Ignore efferent Projections that do not have a learnable attribute
                    #   or are ModulatoryProjections (i.e., including LearningProjections)
                    # Note: if learnable==False, it will be passed along to PyTorch in PytorchProjectionWrapper
                    if not hasattr(efferent_proj,'learnable') or isinstance(efferent_proj,ModulatoryProjection_Base):
                        continue

                    # Deal with Projections to CIMs since nested comps can be learned in PyTorch mode
                    if isinstance(rcvr, CompositionInterfaceMechanism):

                        # Projection to input_CIM, possibly entering a nested Composition
                        if rcvr == rcvr.composition.input_CIM:
                            assert rcvr.composition is not current_comp
                            rcvr_comp = rcvr.composition
                            # FIX: 9/17/23:
                            #FIX: NEED TO BRANCH NOT ON EFFERENTS FROM input_CIM BUT RATHER FROM ITS AFFERENT(S) NODE(S)
                            # Get Node(s) in inner Composition to which Node projects (via input_CIM)
                            receivers = rcvr._get_destination_info_from_input_CIM(efferent_proj.receiver)
                            for _, rcvr, _ in [receivers] if isinstance(receivers, tuple) else receivers:
                                assert rcvr in rcvr_comp.get_nodes_by_role(NodeRole.INPUT), \
                                    f"PROGRAM ERROR: '{rcvr.name}' is not an INPUT Node of '{rcvr_comp.name}'"
                                # Assign efferent_proj (Projection to input_CIM) since it should be learned in PyTorch mode
                                prev[rcvr] = efferent_proj # <- OLD
                                prev[efferent_proj] = node
                                queue.append((rcvr, efferent_proj.receiver, rcvr_comp))

                        # rcvr is Nested Composition output_CIM:
                        # Projection is to output_CIM, possibly exiting from a nested Composition
                        # FIX: 10/1/23 - REVERSE THIS AND NEXT elif?
                        elif rcvr == current_comp.output_CIM and current_comp is not self:

                            # Get output_CIM info for current efferent_proj
                            output_CIM_input_port = efferent_proj.receiver
                            output_CIM = output_CIM_input_port.owner
                            output_CIM_output_port = output_CIM.port_map[efferent_proj.sender][1]

                            # Get all Node(s) in outer Composition to which node projects (via output_CIM)
                            receivers = rcvr._get_destination_info_for_output_CIM(output_CIM_output_port)
                            # Replace efferent_proj(s) with one(s) from output_CIM to rcvr(s) in outer Composition,
                            #   since that(those) is(are the one(s) that should be learned in PyTorch mode
                            # Note:  _get_destination_info_for_output_CIM returns list of destinations
                            #        in order of output_CIM.output_port.efferents
                            if receivers:
                                for efferent_idx, receiver in enumerate(receivers):
                                    if receiver:
                                        _, rcvr, rcvr_comp = receiver
                                        assert rcvr_comp is not current_comp
                                    efferent_proj = output_CIM_output_port.efferents[efferent_idx]
                                    prev[rcvr] = efferent_proj
                                    prev[efferent_proj] = node
                                    queue.append((rcvr, efferent_proj.receiver, rcvr_comp))
                            else:
                                pathways.append(create_pathway(node))

                        # rcvr is Outermost Composition output_CIM:
                        # End of pathway: Direct projection from output_CIM of nested comp to outer comp's output_CIM
                        elif rcvr is self.output_CIM:
                            # Assign node that projects to current node as OUTPUT Node for pathway
                            node_output_port = efferent_proj.sender
                            _, sender, _ = node._get_source_info_from_output_CIM(node_output_port)
                            pathway = create_pathway(node)
                            if pathway:
                                queue.popleft()
                                pathways.append(pathway)

                        else:
                            assert False, f"PROGRAM ERROR:  Unrecognized CompositionInterfaceMechanism: {rcvr}"

                    else:
                        prev[rcvr] = efferent_proj
                        prev[efferent_proj] = node
                        queue.append((rcvr, efferent_proj.receiver, current_comp))
                        continue

            return pathways

        # Construct a pathway for each INPUT Node (except the TARGET Node)
        pathways = [pathway for node in self.get_nodes_by_role(NodeRole.INPUT)
                    if node not in self.get_nodes_by_role(NodeRole.TARGET)
                    for pathway in _get_pytorch_backprop_pathway(node)]

        if execution_mode == pnlvm.ExecutionMode.PyTorch:
            # For PyTorch mode, only need to construct dummy TARGET Nodes, to allow targets to be:
            #  - specified in the same way as for other execution_modes
            #  - trial-by-trial values to kept aligned with inputs in batch / minibatch construction
            #  - tracked for logging (as mechs of a Composition)
            # IMPLEMENTATION NOTE: only add target nodes if not already present
            #    (to avoid duplication in multiple calls, including from command line;
            #     see test_xor_training_identicalness_standard_composition_vs_PyTorch_and_LLVM for example)
            output_mechs = self.get_nested_nodes_output_nodes_at_levels()
            assert set([mech for mech in [pathway[-1] for pathway in pathways]]) == set(output_mechs)
            target_mechs = [ProcessingMechanism(default_variable = np.array([np.zeros_like(value)
                                                                             for value in mech.value],
                                                                            dtype=object),
                                                name= 'TARGET for ' + mech.name)
                            for mech in output_mechs if mech not in self.target_output_map.values()]
            # Suppress warnings about role assignments
            context = Context(source=ContextFlags.METHOD)
            self.add_nodes(target_mechs, required_roles=[NodeRole.TARGET, NodeRole.LEARNING], context=context)
            for target_mech in target_mechs:
                self.exclude_node_roles(target_mech, NodeRole.OUTPUT, context)
                for output_port in target_mech.output_ports:
                    output_port.parameters.require_projection_in_composition.set(False, override=True)
            self.target_output_map.update({target: output for target, output in zip(target_mechs, output_mechs)})
        else:
            # Construct entire PNL backpropagation learning pathways for each INPUT Node
            for pathway in pathways:
                self.add_backpropagation_learning_pathway(pathway=pathway,
                                                          loss_spec=self.loss_spec)

    # CLEANUP: move some of what's done in the methods below to a "validate_params" type of method
    @handle_external_context()
    def _build_pytorch_representation(self, context=None, refresh=False):
        """Builds a Pytorch representation of the AutodiffComposition"""
        if self.scheduler is None:
            self.scheduler = Scheduler(graph=self.graph_processing)
        if self.parameters.pytorch_representation._get(context=context) is None or refresh:
            model = self.pytorch_composition_wrapper_type(composition=self,
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
        if not is_numeric_scalar(learning_rate):
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

    def autodiff_training(self, inputs, targets, context=None, scheduler=None):
        """Perform learning/training on all input-target pairs received for given number of epochs"""

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
            curr_tensor_inputs[component] = torch.tensor(inputs[component], device=self.device).double()

        for component in targets.keys():
            curr_tensor_targets[self.target_output_map[component]] = [torch.tensor(np.atleast_1d(target),
                                                                                   device=self.device).double()
                                                                      for target in targets[component]]

        # do forward computation on current inputs
        #   should return 2d values for each component
        curr_tensor_outputs = self.parameters.pytorch_representation._get(context).forward(curr_tensor_inputs, context)

        for component in curr_tensor_outputs.keys():
            # possibly add custom loss option, which is a loss function that takes many args
            # (outputs, targets, weights, and more) and returns a scalar
            new_loss = 0
            for i in range(len(curr_tensor_outputs[component])):
                new_loss += self.loss(curr_tensor_outputs[component][i],
                                     curr_tensor_targets[component][i])
            tracked_loss += new_loss

        outputs = []
        for input_port in self.output_CIM.input_ports:
            assert (len(input_port.all_afferents) == 1), \
                f"PROGRAM ERROR: {input_port.name} of ouput_CIM for '{self.name}' has more than one afferent."
            port, component, _ = self.output_CIM._get_source_info_from_output_CIM(input_port)
            idx = component.output_ports.index(port)
            outputs += [curr_tensor_outputs[component][idx].detach().cpu().numpy().copy().tolist()]

        self.parameters.tracked_loss_count._set(np.array(self.parameters.tracked_loss_count._get(context=context) + 1),
                                                context=context,
                                                skip_history=True,
                                                skip_log=True)
        return outputs

    def clear_losses(self, context=None):
        self.losses = []
        self.parameters.losses.set([], context=context)

    def _update_learning_parameters(self, context):
        """Carry out backpropagation learning for one or more trials
        Updates parameters (weights) based on trials run since last update.
        Uses Pytorch backward method to compute gradients and update weights
        """
        optimizer = self.parameters.optimizer._get(context=context)
        optimizer.zero_grad()

        tracked_loss = self.parameters.tracked_loss._get(context=context) / int(self.parameters.tracked_loss_count._get(context=context))
        tracked_loss.backward(retain_graph=not self.force_no_retain_graph)
        self.parameters.losses._get(context=context).append(tracked_loss.detach().cpu().numpy()[0])
        self.parameters.tracked_loss._set(torch.zeros(1, device=self.device).double(), context=context, skip_history=True, skip_log=True)
        self.parameters.tracked_loss_count._set(np.array(0), context=context, skip_history=True, skip_log=True)
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

    def _infer_input_nodes(self, input_dict: dict):
        """Remove TARGET Nodes, and return dict with values of INPUT Nodes for single trial
        For nested Compositions, replace input to nested Composition with inputs to its INPUT Nodes
        For InuptPorts, replace with owner

        Returns
        ---------
        A dict mapping INPUT Nodes -> input values for a single trial
        """
        autodiff_input_dict = {}
        for node, values in input_dict.items():
            mech = node.owner if isinstance(node, InputPort) else node
            if (mech in self.get_nested_nodes_input_nodes_at_levels()
                    and mech not in self.get_nodes_by_role(NodeRole.TARGET)):
                # Pass along inputs to all INPUT Nodes except TARGETS
                # (those are handled separately in _infer_output_nodes)
                autodiff_input_dict[node] = values
            # FIX: 11/3/23:  This is handled _parse_learning_spec
            # elif isinstance(node, Composition):
            #     # Replace input to nested Composition with inputs for the InputPorts of its INPUT Nodes
            #     i = 0
            #     for output_port in node.input_CIM.output_ports:
            #         # If node has input from a Node in an outer Composition, no need for input here
            #         if node.input_CIM._get_source_node_for_input_CIM(output_port):
            #             continue
            #         assert len(output_port.efferents) == 1, \
            #             (f"PROGRAM ERROR: {output_port.name} of ouput_CIM for '{node.name}' "
            #              f"has more than one efferent.")
            #         # Get input for destination input_port for every trial in values
            #         #   note: each value (input spec) should be 2d rather than 3d,
            #         #         since it is the input for an InputPort rather than a Mechanism;
            #         #         this gets parsed in PytorchCompositionWrapper.forward()
            #         # autodiff_input_dict[output_port.efferents[0].receiver] = values[i]
            #         # autodiff_input_dict[output_port.efferents[0].receiver] = [value[i] for value in values]
            #         i += 1
        return autodiff_input_dict

    def _infer_output_nodes(self, input_dict: dict):
        """Remove INPUT Nodes, and return dict with values for TARGET Nodes

        Get Inputs to TARGET Nodes and assign to dict mapping them to OUTPUT Nodes of Composition,
        which are used for computation of loss in autodiff_training().

        Returns
        ---------
        A dict mapping TARGET Nodes -> target values corresponding to OUTPUT Nodes of Composition
        """
        # Reduce from 3d inputs to 2d values to match outputs computed in forward computation in autodiff_training()
        return {node:value for node, value in input_dict.items() if node in self.target_output_map}

    def _parse_learning_spec(self, inputs, targets, execution_mode, context):
        stim_input, num_input_trials = super()._parse_learning_spec(inputs, targets, execution_mode, context)

        if not callable(inputs):
            input_ports_for_INPUT_Nodes = self._get_input_receivers()
            nested_inputs = {}
            stim_input_copy = stim_input.copy()
            # Replace input to nested Composition with inputs to its INPUT Nodes (to accommodate flattened version)
            for node in stim_input_copy:
                # If node is a nested Composition
                if isinstance(node, Composition):
                    # If owner of input_port is a Node in the nested Composition, replace entry for nested Composition
                    #   in stim_input with entries for the input_ports of its INPUT Nodes
                    for elem, input_port in enumerate([p for p in input_ports_for_INPUT_Nodes if p.owner in node.nodes]):
                        nested_inputs[input_port] = [entry[elem] for entry in stim_input_copy[node]]
                    stim_input.pop(node)
                    stim_input.update(nested_inputs)

        return stim_input, num_input_trials

    def _check_nested_target_mechs(self):
        pass
    @handle_external_context()
    def learn(self, *args, **kwargs):
        execution_phase_at_entry = kwargs[CONTEXT].execution_phase
        kwargs[CONTEXT].execution_phase = ContextFlags.PREPARING

        execution_mode = self._get_execution_mode(kwargs.pop('execution_mode', None))
        kwargs[CONTEXT].execution_phase = execution_phase_at_entry

        any_nested_comps = [node for node in self.nodes if isinstance(node, Composition)]
        if any_nested_comps:
            # Can't learn in Python mode if any nested Compositions
            if execution_mode is not pnlvm.ExecutionMode.PyTorch:
                nested_comp_names = [f"'{comp.name}'" for comp in any_nested_comps]
                raise AutodiffCompositionError(f"Unable to execute learning in {pnlvm.ExecutionMode.Python.name} mode "
                                               f"for '{self.name}' because it contains one or more nested "
                                               f"Compositions: {' ,'.join(nested_comp_names)}.")

            # Can't learn if any nested comps that are not AutodiffCompositions
            nested_comps = [f"'{comp.name}'" for comp in any_nested_comps if not isinstance(comp, AutodiffComposition)]
            if nested_comps:
                raise AutodiffCompositionError(f"Unable execute learning for '{self.name}' "
                                               f"because it contains nested Composition(s) "
                                               f"that are not AutodiffCompositions: {' ,'.join(nested_comps)}.")

        if self._built_pathways is False:
            self.infer_backpropagation_learning_pathways(execution_mode)
            self._built_pathways = True

        return super().learn(*args, execution_mode=execution_mode, **kwargs)

    def _get_execution_mode(self, execution_mode):
        """Parse execution_mode argument and return a valid execution mode for the learn() method
        Can be overridden by subclasses to change the permitted and/or default execution mode for learning
        """
        if execution_mode is None:
            if self.execution_mode_warned_about_default is False:
                warnings.warn(f"The execution_mode argument was not specified in the learn() method of '{self.name}'; "
                              f"ExecutionMode.PyTorch will be used by default.")
                self.execution_mode_warned_about_default = True
            execution_mode = pnlvm.ExecutionMode.PyTorch

        return execution_mode

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
        """Override to execute autodiff_training() in learning mode if execute_mode is not Python"""

        if (self._is_learning(context) and execution_mode is not pnlvm.ExecutionMode.PyTorch and
                any([isinstance(node, Composition) for node in self.nodes])):
            raise CompositionError(f"Must use execution_mode=ExecutionMode.PyTorch for learning "
                                   f"that includes nested AutodiffComposition(s).")

        if execution_mode is not pnlvm.ExecutionMode.Python:
            self._assign_execution_ids(context)
            context.composition = self
            context.source = ContextFlags.COMPOSITION

            if execution_mode is pnlvm.ExecutionMode.PyTorch and not torch_available:
                raise AutodiffCompositionError(f"'{self.name}.learn()' has been called with ExecutionMode.Pytorch, "
                                               f"but Pytorch module ('torch') is not installed. "
                                               f"Please install it with `pip install torch` or `pip3 install torch`")

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

                execution_phase = context.execution_phase
                context.execution_phase = ContextFlags.PROCESSING

                self.output_CIM.execute(output, context=context)
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

        # Call Composition execute in Python mode
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
