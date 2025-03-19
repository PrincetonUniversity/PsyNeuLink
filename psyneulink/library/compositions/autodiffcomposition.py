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
      - `AutodiffComposition`
          - `AutodiffComposition_Modulatory_Mechanisms`
          - `AutodiffComposition_Bias_Parameters`
          - `AutodiffComposition_Nesting`
          - `AutodiffComposition_Learning_Rates`
          - `AutodiffComposition_Exchange_With_Torch_Parameters`
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

*Only one OutputPort per Node*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `Nodes <Composition_Nodes>` of an AutodiffComposition currently can have only *one* `OutputPort`, though that
can have more than one `efferent <Port_Base.efferents>` `MappingProjection`.  Nodes can also have more than one
`InputPort`, that can receive more than one `afferent `path_afferent <Port_Base.path_afferents>` Projections.

.. _AutodiffComposition_Modulatory_Mechanisms:

*No Modulatory Components*
~~~~~~~~~~~~~~~~~~~~~~~~~~

All of the Components in an AutodiffComposition must be able to be subjected to `learning <Composition_Learning>`,
which means that no `ModulatoryMechanisms <ModulatoryMechanism>` can be included in an AutodiffComposition.
Specifically, this precludes any `learning components <Composition_Learning_Components>`, `ControlMechanisms
<ControlMechanism>`, or a `controller <Composition_Controller>`.

.. _Autodiff_Learning_Components_Warning:

*Learning Components*.  An AutodiffComposition **cannot include any** `learning components
<Composition_Learning_Components>` themselves (i.e., `LearningMechanisms <LearningMechanism>`, `LearningSignals
<LearningSignal>`, or `LearningProjections <LearningProjection>`, nor the `ComparatorMechanism`
or `ObjectiveMechanism` used to compute the loss for learning). These are constructed
automatically when learning is executed in `Python mode <AutodiffComposition_Python>` or `LLVM mode
<AutodiffComposition_LLVM>`, and PyTorch-compatible Components are constructed when it is executed in
`PyTorch mode <AutodiffComposition_PyTorch>`.

*Control Components*. An AutodiffComposition also cannot include any `ControlMechanisms <ControlMechanism>` or a
`controller <Composition_Controller>`.  However, it *can* include Mechanisms that are subject to modulatory control
(see `Figure <ModulatorySignal_Anatomy_Figure>`, and `modulation <ModulatorySignal_Modulation>`) by ControlMechanisms
*outside* the Composition, including the controller of a Composition within which the AutodiffComposition is nested.
That is, an AutodiffComposition can be `nested in a Composition <Composition_Nested>` that has other such Components
(see `AutodiffComposition_Nested_Modulation` below).

.. _AutodiffComposition_Bias_Parameters:

*No Bias Parameters*
~~~~~~~~~~~~~~~~~~~~

AutodiffComposition does not (currently) support the *automatic* construction of separate bias parameters.
Thus, when constructing the PyTorch version of an AutodiffComposition, the `bias
<https://www.pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ parameter of any PyTorch modules are set to False.
However, biases can be implemented using `Composition_Bias_Nodes`.

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

.. _AutodiffComposition_Learning_Rates:

*Learning Rates and Optimizer Params*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **optimizer_params** argument of the constructor can be used to specify parameters for the optimizer used for
learning by the AutodiffComposition. At present, this is restricted to overriding the `learning_rate
<AutodiffComposition.learning_rate>` Parameter of the Composition (used as the default by the `optimizer
<AutodiffComposition.optimizer>`) to assign individual learning rates to specific Projections. This is done by
specifying **optimizer_params** as a dict, in which each key is a reference to a learnable `MappingProjection`
in the AutodiffComposition, and the value of which specifies its learning_rate. Sublcasses of AutodiffComposition may
involve different forms of specification and/or support other parameters for the optimizer. Projections that are not
sepcified in **optimizer_params** use, in order of precedence: the `learning_rate <AutodiffComposition.learning_rate>`
specified in the call to the AutodiffComposition's `learn <AutodiffComposition.learn>` method, the **learning_rate**
argument of its constructor, or the default value for the AutodiffComposition.

.. _AutodiffComposition_Exchange_With_Torch_Parameters:

*Exchanging Parameters with Pytorch Modules*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The AutodiffComposition's `copy_torch_param_to_projection_matrix` and `copy_projection_matrix_to_torch_param` methods
can be used to exchange weight matrices between the parameters of a PyTorch module and the `matrix
<MappingProjection.matrix>` Parameter of a `MappingProjection` in the AutodiffComposition. Pytorch Parameters can
be referenced flexibly, either by the Parameter object itself, or by the module and either the name or index of the
Parameter in the module's state_dict or parameter list, respectively. Slices of PyTorch Parameters can also be used,
for cases in which the matrix of a Project corresponds to only a subpart of the PyTorch Parameter (e.g., for
`GRUComposition`). Both methods return the item assigned.

.. _AutodiffComposition_Post_Construction_Modification:

*No Post-construction Modification*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

COMMENT:
IS THIS STILL TRUE?  TEST?
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

COMMENT:
# 7/10/24 - FIX:
.. _AutodiffComposition_PyTorch_LearningScale:
   ADD DESCRIPTION OF HOW LearningScale SPECIFICATIONS MAP TO EXECUTION OF pytorch_rep:
      OPTIMIZATION STEP:
      for AutodiffCompositions, this corresponds to a single call to `foward()` and `backward()`
            methods of the Pytorch model
COMMENT

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
      * Specifying `ExecutionMode.LLVMRun` or `ExecutionMode.PyTorch` in the learn() method of a standard
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
       Specifying `ExecutionMode.LLVMRun` in either the `learn <Composition.learn>` and `run <Composition.run>`
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
    >>> my_mech_1 = pnl.TransferMechanism(function=pnl.Linear, input_shapes = 3)
    >>> my_mech_2 = pnl.TransferMechanism(function=pnl.Linear, input_shapes = 2)
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
from collections import deque
from typing import Union

try:
    import torch
    from torch import nn
    import torch.optim as optim
    torch_available = True
except ImportError:
    torch_available = False
else:
    from psyneulink.library.compositions.pytorchwrappers import PytorchCompositionWrapper
    from psyneulink.library.compositions.pytorchshowgraph import PytorchShowGraph

from psyneulink._typing import Iterable, Mapping, Optional
from psyneulink.core.components.component import Component
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.core.components.mechanisms.modulatory.modulatorymechanism import ModulatoryMechanism_Base
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.compositions.composition import Composition, NodeRole, CompositionError
from psyneulink.core.compositions.report import (ReportOutput, ReportParams, ReportProgress, ReportSimulations,
                                                 ReportDevices, EXECUTE_REPORT, LEARN_REPORT, PROGRESS_REPORT)
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import (
    AUTODIFF_COMPOSITION, EXECUTION_MODE,
    LEARNING_SCALE_LITERALS, LEARNING_SCALE_NAMES, LEARNING_SCALE_VALUES,
    Loss, LOSSES, MATRIX_WEIGHTS, MINIBATCH, NODE_VALUES, NODE_VARIABLES,
    OPTIMIZATION_STEP, RESULTS, RUN, SOFT_CLAMP, SYNCH_WITH_PNL_OPTIONS,
    RETAIN_IN_PNL_OPTIONS, TARGETS, TRAINED_OUTPUTS, TRIAL, DEFAULT,
)
from psyneulink.core.globals.utilities import is_matrix_keyword, is_numeric_scalar, convert_to_np_array
from psyneulink.core.scheduling.scheduler import Scheduler
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.core import llvm as pnlvm


logger = logging.getLogger(__name__)


__all__ = [
    'AutodiffComposition'
]

def _get_torch_trained_outputs(owning_component=None, context=None):
    if not context.execution_id:
        return None
    pytorch_rep = owning_component.parameters.pytorch_representation._get(context)
    if not pytorch_rep:
        return None
    return np.array(pytorch_rep.retained_trained_outputs)

def _get_torch_targets(owning_component=None, context=None):
    if not context.execution_id:
        return None
    pytorch_rep = owning_component.parameters.pytorch_representation._get(context)
    if not pytorch_rep:
        return None
    return np.array(pytorch_rep.retained_targets)

def _get_torch_losses(owning_component, context):
    if not context.execution_id:
        return None
    pytorch_rep = owning_component.parameters.pytorch_representation._get(context)
    if not pytorch_rep:
        return None
    return np.array(pytorch_rep.retained_losses)

class AutodiffCompositionError(CompositionError):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class AutodiffComposition(Composition):
    """
    AutodiffComposition(                        \
        optimizer_type='sgd',
        loss_spec=Loss.MSE,
        weight_decay=0,
        learning_rate=0.001,
        optimizer_params=None,
        disable_learning=False,
        synch_projection_matrices_with_torch=RUN,
        synch_node_variables_with_torch=None,
        synch_node_values_with_torch=RUN,
        synch_results_with_torch=RUN,
        retain_torch_trained_outputs=MINIBATCH,
        retain_torch_targets=MINIBATCH,
        retain_torch_losses=MINIBATCH,
        device=CPU
        )

    Subclass of `Composition` that trains models using either LLVM compilation or `PyTorch <https://pytorch.org>`_;
    see and `Composition <Composition_Class_Reference>` for additional arguments and attributes.  See `Composition`
    for additional arguments to constructor.

    Arguments
    ---------

    optimizer_type : str : default 'sgd'
        the kind of optimizer used in training. The current options are 'sgd' or 'adam'.

    loss_spec : Loss or PyTorch loss function : default Loss.MSE
        specifies the loss function for training; see `Loss` for arguments.

    weight_decay : float : default 0
        specifies the L2 penalty (which discourages large weights) used by the optimizer.

    learning_rate : float : default 0.001
        specifies the learning rate passed to the optimizer if none is specified in the `learn
        <AutdodiffComposition.learn>` method of the AutodiffComposition;
        see `learning_rate <AutodiffComposition.learning_rate>` for additional details.

    optimizer_params : Dict[str: value]
        specifies parameters for the optimizer used for learning by the GRUComposition
        (see `AutodiffComposition_Learning_Rates` for details of specification.

    disable_learning : bool: default False
        specifies whether the AutodiffComposition should disable learning when run in `learning mode
        <Composition.learn>`.

    synch_projection_matrices_with_torch : `LearningScale` : default RUN
        specifies the default for the AutodiffComposition for when to copy Pytorch parameters to PsyNeuLink
        `Projection matrices <MappingProjection.matrix>` (connection weights), which can be overridden by specifying
        the **synch_projection_matrices_with_torch** argument in the `learn <Composition.learn>` method;
        see `synch_projection_matrices_with_torch <AutodiffComposition.synch_projection_matrices_with_torch>`
        for additional details.

    synch_node_variables_with_torch : `LearningScale` : default None
        specifies the default for the AutodiffComposition for when to copy the current input to Pytorch nodes
        to the PsyNeuLink `variable <Mechanism_Base.value>` attribute of the corresponding PsyNeuLink `nodes
        <Composition_Node>`, which can be overridden by specifying the **synch_node_variables_with_torch** argument
        in the `learn <Composition.learn>` method; see `synch_node_variables_with_torch
        <AutodiffComposition.synch_node_variables_with_torch>` for additional details.

    synch_node_values_with_torch : `LearningScale` : default RUN
        specifies the default for the AutodiffComposition for when to copy the current output of Pytorch nodes to the
        PsyNeuLink `value <Mechanism_Base.value>` attribute of the corresponding PsyNeuLink `nodes <Composition_Node>`,
        which can be overridden by specifying the **synch_node_values_with_torch** argument in the `learn
        <Composition.learn>` method; see `synch_node_values_with_torch
        <AutodiffComposition.synch_node_values_with_torch>` for additional details.

    synch_results_with_torch : `LearningScale` : default RUN
        specifies the default for the AutodiffComposition for when to copy the outputs of the Pytorch model
        to the AutodiffComposition's `results <Composition.results>` attribute, which can be overridden by
        specifying the **synch_results_with_torch** argument in the `learn <Composition.learn>` method.
        Note that this differs from **retain_torch_trained_outputs**, which specifies the frequency at which
        the outputs of the PyTorch model are tracked, all of which are stored in the AutodiffComposition's
        `torch_trained_outputs <AutodiffComposition.torch_trained_outputs>` attribute at the end of the run;
        see `synch_results_with_torch <AutodiffComposition.synch_results_with_torch>` for
        additional details.

    retain_torch_trained_outputs : `LearningScale` : default MINIBATCH
        specifies the default for the AutodiffComposition for scale at which the outputs of the Pytorch
        model are tracked, all of which are stored in the AutodiffComposition's `torch_trained_outputs
        <AutodiffComposition.torch_trained_outputs>` attribute at the end of the run; this can be overridden
        by specifying the **retain_torch_trained_outputs** argument in the `learn <Composition.learn>` method.
        Note that this differs from **synch_results_with_torch**, which specifies the frequency with
        which values are called to the AutodiffComposition's `results` attribute; see `retain_torch_trained_outputs
        <AutodiffComposition.retain_torch_trained_outputs>` for additional details.

    retain_torch_targets : `LearningScale` : default MINIBATCH
        specifies the default for the AutodiffComposition for when to copy the targets used for training the
        Pytorch model to the AutodiffComposition's `torch_targets <Composition.torch_targets>` attribute, which can be
        overridden by specifying the **retain_torch_targets** argument in the `learn <Composition.learn>` method;
        see `retain_torch_targets <AutodiffComposition.retain_torch_targets>` for additional details.

    retain_torch_losses : `LearningScale` : default MINIBATCH
        specifies the default for the AutodiffComposition for the scale at which the losses of the Pytorch model
        are tracked, all of which are stored in the AutodiffComposition's `torch_losses <Composition.torch_losses>`
        attribute at the end of the run; see `retain_torch_losses <AutodiffComposition.retain_torch_losses>` for
        additional details.

    device : torch.device : default device-dependent
        specifies the device on which the model is run. If None, the device is set to 'cuda' if available,
        then 'mps`, otherwise 'cpu'.

    Attributes
    ----------

    pytorch_representation : PytorchCompositionWrapper : default None
        represents the PyTorch model of the AutodiffComposition, which is created when the AutodiffComposition is
        run in `PyTorch mode <AutodiffComposition_PyTorch>`.

    optimizer : PyTorch optimizer function
        the optimizer used for training. Depends on the **optimizer_type**, **learning_rate**, and **weight_decay**
        arguments from initialization.

    loss : PyTorch loss function
        the loss function used for training. Depends on the **loss_spec** argument from initialization.

    learning_rate : float or bool
        determines the default learning_rate passed the optimizer, that is applied to all `Projections <Projection>`
        in the AutodiffComposition that are `learnable <MappingProjection.learnable>`, and for which individual rates
        have not been specified (for how to do the latter, see `AutodiffComposition_Learning_Rates`).

        .. note::
           At present, an outermost Compositon's learning rate is applied to any `nested Compositions
           <AutodiffComposition_Nesting>`, whether this is specified in the call to its `learn
           <AutodiffComposition.learn>` method, its constructor, or its default value is being used.

        .. hint::
           To disable updating of a particular `MappingProjection` in an AutodiffComposition, specify either the
           **learnable** parameter of its constructor or its learning_rate specification in the **optimizer_params**
           argument of the AutodiffComposition's constructor to False  (see `AutodiffComposition_Learning_Rates`);
           this applies to MappingProjections at any level of `nesting <AutodiffComposition_Nesting>`

    synch_projection_matrices_with_torch : OPTIMIZATION_STEP, MINIBATCH, EPOCH or RUN
        determines when to copy PyTorch parameters to PsyNeuLink `Projection matrices <MappingProjection.matrix>`
        (connection weights) if this is not specified in the call to `learn <AutodiffComposition.learn>`. Copying more
        frequently keeps the PsyNeuLink representation more closely synchronized with parameter updates in Pytorch,
        but slows performance (see `AutodiffComposition_PyTorch_LearningScale` for information about settings).

    synch_node_variables_with_torch : OPTIMIZATION_STEP, TRIAL, MINIBATCH, EPOCH, RUN or None
        determines when to copy the current input to Pytorch functions to the PsyNeuLink `variable
        <Mechanism_Base.value>` attribute of the corresponding PsyNeuLink `nodes <Composition_Node>`,
        if this is not specified in the call to `learn <AutodiffComposition.learn>`.
        COMMENT:
        8/8/24 - FIX: 3/15/25 ADD EXPLANATION OF WHY THIS IS NOT GENERALLY USEFUL ALONG THE LINES OF THE FOLLOWING
                 ALSO RELATE TO EXECUTE_NODES OPTION ONCE IMPLEMENTED
        This is supported for inspection and debugging, but is not generally useful, as PsyNeuLink uses `Lazy
        Evaluation <Component_Lazy_Updating>`, in which the variable of a node is determined by the input it receives
        during execution.
        COMMENT
        Copying more frequently keeps the PsyNeuLink representation more closely copying more frequently
        keeps them synchronized with parameter updates in Pytorch, but can slow performance (see
        `AutodiffComposition_PyTorch_LearningScale` for information about settings).

    synch_node_values_with_torch : OPTIMIZATION_STEP, MINIBATCH, EPOCH or RUN
        determines when to copy the current output of Pytorch functions to the PsyNeuLink `value
        <Mechanism_Base.value>` attribute of the corresponding PsyNeuLink `nodes <Composition_Node>`,
        if this is not specified in the call to `learn <AutodiffComposition.learn>`. Copying more
        frequently keeps the PsyNeuLink representation more closely synchronized with parameter
        updates in Pytorch, but can also slow performance (see `AutodiffComposition_PyTorch_LearningScale`
        for information about settings).

    synch_results_with_torch : OPTIMIZATION_STEP, TRIAL, MINIBATCH, EPOCH or RUN
        determines when to copy the current outputs of Pytorch nodes to the PsyNeuLink `results
        <Composition.results>` attribute of the AutodiffComposition if this is not specified in
        the call to `learn <AutodiffComposition.learn>`. Copying more frequently keeps the PsyNeuLink
        representation more closely synchronized with parameter updates in Pytorch, but slows performance
        (see `AutodiffComposition_PyTorch_LearningScale` for information about settings).

    retain_torch_trained_outputs : OPTIMIZATION_STEP, MINIBATCH, EPOCH, RUN or None
        determines the scale at which the outputs of the Pytorch model are tracked, all of which are stored in
        the AutodiffComposition's `results <Composition.results>` attribute at the end of the run if this is not
        specified in the call to `learn <AutodiffComposition.learn>`(see `AutodiffComposition_PyTorch_LearningScale`
        for information about settings)

    retain_torch_targets : OPTIMIZATION_STEP, TRIAL, MINIBATCH, EPOCH, RUN or None
        determines the scale at which the targets used for training the Pytorch model are tracked, all of which
        are stored in the AutodiffComposition's `targets <Composition.targets>` attribute at the end of the run
        if this is not specified in the call to `learn <AutodiffComposition.learn>`
        (see `AutodiffComposition_PyTorch_LearningScale` for information about settings).

    retain_torch_losses : OPTIMIZATION_STEP, MINIBATCH, EPOCH, RUN or None
        determines the scale at which the losses of the Pytorch model are tracked, all of which are stored in
        the AutodiffComposition's `torch_losses <Composition.torch_losses>` attribute at the end of the run
        if this is nota specified in the call to `learn <AutodiffComposition.learn>`
        (see `AutodiffComposition_PyTorch_LearningScale` for information about settings).

    torch_trained_outputs : List[ndarray]
        stores the outputs (converted to np arrays) of the Pytorch model trained during learning, at the frequency
        specified by `retain_torch_trained_outputs <AutodiffComposition.retain_torch_trained_outputs>` if it is set
        to *MINIBATCH*, *EPOCH*, or *RUN*; see `retain_torch_trained_outputs
        <AutodiffComposition.retain_torch_trained_outputs>` for additional details.

    torch_targets : List[ndarray]
        stores the targets used for training the Pytorch model during learning at the frequency specified by
        `retain_torch_targets <AutodiffComposition.retain_torch_targets>` if it is set to *MINIBATCH*, *EPOCH*,
        or *RUN*; see `retain_torch_targets <AutodiffComposition.retain_torch_targets>` for additional details.

    torch_losses : list of floats
        stores the average loss after each weight update (i.e. each minibatch) during learning, at the frequency
        specified by `retain_torch_trained_outputs <AutodiffComposition.retain_torch_trained_outputs>` if it is set to *MINIBATCH*,
        *EPOCH*, or *RUN*; see `retain_torch_losses <AutodiffComposition.retain_torch_losses>` for additonal details.

    COMMENT:  FIX: NOT CURRENTLY BEING POPULTED, BUT SEEMS TO BE USED BY _get_total_loss() and early_stopper
    trial_losses = Parameter([])
    COMMENT

    last_saved_weights : path
        path for file to which weights were last saved.

    last_loaded_weights : path
        path for file from which weights were last loaded.

    device : torch.device
        the device on which the model is run.
    """

    componentCategory = AUTODIFF_COMPOSITION
    if torch_available:
        from psyneulink.library.compositions.pytorchwrappers import PytorchCompositionWrapper, PytorchMechanismWrapper
        pytorch_composition_wrapper_type = PytorchCompositionWrapper
        pytorch_mechanism_wrapper_type = PytorchMechanismWrapper

    class Parameters(Composition.Parameters):
        pytorch_representation = None
        optimizer = None
        learning_rate = Parameter(.001, fallback_value=DEFAULT)
        synch_projection_matrices_with_torch = Parameter(RUN, fallback_value=DEFAULT)
        synch_node_variables_with_torch = Parameter(None, fallback_value=DEFAULT)
        synch_node_values_with_torch = Parameter(RUN, fallback_value=DEFAULT)
        synch_results_with_torch = Parameter(RUN, fallback_value=DEFAULT)
        retain_torch_trained_outputs = Parameter(MINIBATCH, fallback_value=DEFAULT)
        retain_torch_targets = Parameter(MINIBATCH, fallback_value=DEFAULT)
        retain_torch_losses = Parameter(MINIBATCH, fallback_value=DEFAULT)
        torch_trained_outputs = Parameter([], getter=_get_torch_trained_outputs)
        torch_targets = Parameter([], getter=_get_torch_targets)
        torch_losses = Parameter([], getter=_get_torch_losses)
        trial_losses = Parameter([]) # FIX <- related to early_stopper, but not getting assigned anywhere
        device = None

        # def _validate_memory_template(self, device):
        #     if isinstance(device, str) and device not in [CPU, CUDA, MPS]:
        #         raise AutodiffCompositionError(f"Device must be one of {CPU}, {CUDA}, or {MPS}")
        #
        def _validate_synch_projection_matrices_with_torch(self, spec):
            if spec is not None and spec not in LEARNING_SCALE_VALUES:
                raise AutodiffCompositionError(f"Value of 'synch_projection_matrices_with_torch' arg "
                                               f"must be one of the following keywords: "
                                               f"{', '.join(LEARNING_SCALE_NAMES)}")

        def _validate_synch_node_variables_with_torch(self, spec):
            if spec is not None and spec not in LEARNING_SCALE_VALUES:
                raise AutodiffCompositionError(f"Value of 'synch_node_variables_with_torch' arg "
                                               f"must be one of the following keywords: "
                                               f"{', '.join(LEARNING_SCALE_NAMES)}")

        def _validate_synch_node_values_with_torch(self, spec):
            if spec is not None and spec not in LEARNING_SCALE_VALUES:
                raise AutodiffCompositionError(f"Value of 'synch_node_values_with_torch' arg "
                                               f"must be one of the following keywords: "
                                               f"{', '.join(LEARNING_SCALE_NAMES)}")

        def _validate_synch_results_with_torch(self, spec):
            if spec is not None and spec not in LEARNING_SCALE_VALUES:
                raise AutodiffCompositionError(f"Value of 'synch_results_with_torch' arg "
                                               f"must be one of the following keywords: "
                                               f"{', '.join(LEARNING_SCALE_NAMES)}")
            if spec is OPTIMIZATION_STEP:
                arg_vals = LEARNING_SCALE_NAMES.copy()
                arg_vals.remove('OPTIMIZATION_STEP')
                raise AutodiffCompositionError(f"'OPTIMIZATION_STEP can't be used with 'synch_results_with_torch';"
                                               f"use another value of {', '.arg_vals}")


        def _validate_retain_torch_trained_outputs(self, spec):
            if spec is not None and spec not in LEARNING_SCALE_VALUES:
                raise AutodiffCompositionError(f"Value of `retain_torch_trained_outputs` arg "
                                               f"must be one of the following keywords: "
                                               f"{', '.join(LEARNING_SCALE_NAMES)}")

        def _validate_retain_torch_targets(self, spec):
            if spec is not None and spec not in LEARNING_SCALE_VALUES:
                raise AutodiffCompositionError(f"Value of `retain_torch_targets` arg "
                                               f"must be one of the following keywords: "
                                               f"{', '.join(LEARNING_SCALE_NAMES)}")

        def _validate_retain_torch_losses(self, spec):
            if spec is not None and spec not in LEARNING_SCALE_VALUES:
                raise AutodiffCompositionError(f"Value of `retain_torch_losses` arg "
                                               f"must be one of the following keywords: "
                                               f"{', '.join(LEARNING_SCALE_NAMES)}")


    # TODO (CW 9/28/18): add compositions to registry so default arg for name is no longer needed
    @check_user_specified
    def __init__(self,
                 pathways=None,
                 optimizer_type='sgd',
                 loss_spec=Loss.MSE,
                 weight_decay=0,
                 learning_rate=None,
                 optimizer_params:dict=None,
                 disable_learning=False,
                 force_no_retain_graph=False,
                 refresh_losses=False,
                 synch_projection_matrices_with_torch:Optional[str]=RUN,
                 synch_node_variables_with_torch:Optional[str]=None,
                 synch_node_values_with_torch:Optional[str]=RUN,
                 synch_results_with_torch:Optional[str]=RUN,
                 retain_torch_trained_outputs:Optional[str]=MINIBATCH,
                 retain_torch_targets:Optional[str]=MINIBATCH,
                 retain_torch_losses:Optional[str]=MINIBATCH,
                 device=None,
                 disable_cuda=True,
                 cuda_index=None,
                 name="autodiff_composition",
                 **kwargs):

        # if not torch_available:
        #     raise AutodiffCompositionError('Pytorch python module (torch) is not installed. Please install it with '
        #                                    '`pip install torch` or `pip3 install torch`')
        #
        show_graph_attributes = kwargs.pop('show_graph_attributes', {})

        super(AutodiffComposition, self).__init__(
            name = name,
            pathways=pathways,
            optimizer_type = optimizer_type,
            loss_spec = loss_spec,
            weight_decay = weight_decay,
            learning_rate = learning_rate,
            synch_projection_matrices_with_torch = synch_projection_matrices_with_torch,
            synch_node_variables_with_torch = synch_node_variables_with_torch,
            synch_node_values_with_torch = synch_node_values_with_torch,
            synch_results_with_torch = synch_results_with_torch,
            retain_torch_trained_outputs = retain_torch_trained_outputs,
            retain_torch_targets = retain_torch_targets,
            retain_torch_losses = retain_torch_losses,
            **kwargs)

        self._built_pathways = False
        self.targets_from_outputs_map = {} # Map from TARGETS nodes to any OUTPUT nodes from which they receive input
        self.outputs_to_targets_map = {}   # Map from trained OUTPUT nodes to their TARGETS
        self._trained_comp_nodes_to_pytorch_nodes_map = None # Set by subclasses that replace trained OUTPUT Nodes
        self._input_comp_nodes_to_pytorch_nodes_map = None # Set by subclasses that replace INPUT Nodes
        self._pytorch_projections = []
        self.optimizer_type = optimizer_type
        self._optimizer_params = optimizer_params or {}
        self.loss_spec = loss_spec
        self._runtime_learning_rate = None
        self.force_no_retain_graph = force_no_retain_graph
        self.refresh_losses = refresh_losses
        self.weight_decay = weight_decay
        self.disable_learning = disable_learning
        self.loss_function = None
        self.last_saved_weights = None
        self.last_loaded_weights = None

        # keeps track of average loss per epoch
        self.losses = []

        # ordered execution sets for the pytorch model
        self.execution_sets = None

        # # MODIFIED 7/10/24 OLD:
        if not disable_cuda and torch.cuda.is_available():
            if cuda_index is None:
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cuda:' + str(cuda_index))
        elif torch_available:
            self.device = torch.device('cpu')
            self.torch_dtype = self.pytorch_composition_wrapper_type.torch_dtype
        else:
            self.device = device
            self.torch_dtype = None
        # # MODIFIED 7/10/24 NEW: NEEDED FOR torch MPS SUPPORT
        #  FIX: ADD AFTER USE OF utilities.get_torch_tensor() AND COMPATIBLITY WITH MPS IS VALIDATED
        # if device is None:
        #     # Try setting device by default
        #     if not disable_cuda and torch.cuda.is_available():
        #         if cuda_index is None:
        #             self.device = torch.device(CUDA)
        #         else:
        #             self.device = torch.device('cuda:' + str(cuda_index))
        #     elif torch_available:
        #         if torch.backends.mps.is_available():
        #             from psyneulink.core.components.functions.nonstateful.transferfunctions import Linear
        #             try:
        #                 self.device = torch.device(MPS)
        #                 test_pytorch_fct_with_mps = Linear()._gen_pytorch_fct(self.device, Context())
        #             except AssertionError:
        #                 self.device = torch.device(CPU)
        #         else:
        #             self.device = torch.device(CPU)
        # else:
        #     self.device = device
        # # MODIFIED 7/10/24 END

        # Set to True after first warning about failure to specify execution mode so warning is issued only once
        self.execution_mode_warned_about_default = False
        # torch params added when warned in copy_projection_matrix_to_torch_param() to avoid repeats for same param
        self.require_grad_warning = []
        # return self.infer_backpropagation_learning_pathways(pnlvm.ExecutionMode.PyTorch)

        # ShowGraph
        self.assign_ShowGraph(show_graph_attributes)
    def assign_ShowGraph(self, show_graph_attributes):
        """Override to replace assignment of ShowGraph class with PytorchShowGraph if torch is available"""
        show_graph_attributes = show_graph_attributes or {}
        if torch_available:
            self._show_graph = PytorchShowGraph(self, **show_graph_attributes)
        else:
            from psyneulink.core.compositions.showgraph import ShowGraph
            self._show_graph = ShowGraph(self, **show_graph_attributes)

    @handle_external_context()
    def infer_backpropagation_learning_pathways(self, execution_mode, context=None)->list:
        """Create backpropagation learning pathways for every Input Node --> Output Node pathway
        Flattens nested compositions:
          - only includes the Projections in outer Composition to/from the CIMs of the nested Composition
            (i.e., to input_CIMs and from output_CIMs) -- the ones that should be learned;
          - excludes Projections from/to CIMs in the nested Composition
            (from input_CIMs and to output_CIMs), as those should remain identity Projections;
          see `PytorchCompositionWrapper` for table of how Projections are handled and further details.
        Returns list of target nodes for each pathway
        """

        # Construct a pathway(s) for each INPUT Node (including BIAS Nodes), except the TARGET Node)
        pathways = self._get_pytorch_backprop_pathways(context)

        if execution_mode is pnlvm.ExecutionMode.PyTorch:
            # For PyTorch mode, only need to construct dummy TARGET Nodes, to allow targets to be:
            #  - specified in the same way as for other execution_modes
            #  - trial-by-trial values kept aligned with inputs in batch / minibatch construction
            #  - tracked for logging (as mechs of a Composition)
            # IMPLEMENTATION NOTE:
            #    only add target nodes if not already present
            #    (to avoid duplication in multiple calls, including from command line;
            #     see test_xor_training_identicalness_standard_composition_vs_PyTorch_and_LLVM for example)
            # output_mechs_for_learning = self.get_nested_output_nodes_at_all_levels()
            # assert set([mech for mech in [pathway[-1] for pathway in pathways]]) == set(output_mechs_for_learning)
            pathway_terminal_nodes = [mech for mech in [pathway[-1] for pathway in pathways]]
            identified_target_nodes = self._identify_target_nodes(context)
            output_mechs_for_learning = [node for node in identified_target_nodes if node in pathway_terminal_nodes]
            target_mechs = [ProcessingMechanism(default_variable = np.array([np.zeros_like(value)
                                                                             for value in mech.value],
                                                                            dtype=object),
                                                name= 'TARGET for ' + mech.name)
                            for mech in output_mechs_for_learning if mech not in self.targets_from_outputs_map.values()]
            # Suppress warnings about role assignments
            context = Context(source=ContextFlags.METHOD)
            self.add_nodes(target_mechs, required_roles=[NodeRole.TARGET, NodeRole.LEARNING], context=context)
            for target_mech in target_mechs:
                self.exclude_node_roles(target_mech, NodeRole.OUTPUT, context)
                for output_port in target_mech.output_ports:
                    output_port.parameters.require_projection_in_composition.set(False, override=True)
            self.targets_from_outputs_map.update({target: output for target, output
                                           in zip(target_mechs, output_mechs_for_learning)})
        else:
            # Construct entire PNL backpropagation learning pathways for each INPUT Node
            for pathway in pathways:
                self.add_backpropagation_learning_pathway(pathway=pathway,
                                                          loss_spec=self.loss_spec)

        self.outputs_to_targets_map = {output: target for target, output in self.targets_from_outputs_map.items()}
        self._analyze_graph()
        return self.learning_components

    @handle_external_context()
    def _get_pytorch_backprop_pathways(self, context)->list:

        self._analyze_graph()
        return [pathway
                    for node in (self.get_nodes_by_role(NodeRole.INPUT) + self.get_nodes_by_role(NodeRole.BIAS))
                    if node not in self.get_nodes_by_role(NodeRole.TARGET)
                    for pathway in self._get_pytorch_backprop_pathway(node, context)]

    def _get_pytorch_backprop_pathway(self, input_node, context)->list:
        """Breadth-first search from input_node to find all input -> output pathways
        Uses queue(node, composition) to traverse all nodes in the graph
        IMPLEMENTATION NOTE:  flattens nested Compositions, removing any CIMs in the nested Compositions
        Return a list of all pathways from input_node -> output node
        """

        pathways = []  # List of all feedforward pathways from INPUT Node to OUTPUT Node
        dependency_dict = {}      # Dictionary of previous component for each component in every pathway
        queue = deque([(input_node, self)])  # Queue of nodes to visit in breadth-first search

        def create_pathway(current_comp, node)->list:
            """Create pathway starting with node (presumably an output NODE) and working backward via dependency_dict"""
            pathway = []
            entry = node
            while entry in dependency_dict:
                # Prevent cycle from recurrent pathway
                if entry in pathway:
                    break
                pathway.insert(0, entry)
                entry = dependency_dict[entry]
            pathway.insert(0, entry)
            # Only consider pathways with 3 or more components (input -> projection -> ... -> output)
            #    since can't learn on only one mechanism (len==1)
            #    and a pathway can't have just one mechanism and one projection (len==2)
            if len(pathway) >= 3:
                return pathway
            else:
                return []

        # breadth-first search starting with input node
        while len(queue) > 0:
            node, current_comp = queue.popleft()

            # node is nested Composition that is an INPUT node of the immediate outer Composition,
            #   so put that in queue for procsssing in next pass through while loop
            if (isinstance(node, Composition) and node is not self
                    and any(isinstance(proj.sender.owner, CompositionInterfaceMechanism)
                            for proj in node.afferents)):
                for output_port in node.input_CIM.output_ports:
                    for proj in output_port.efferents:
                        queue.append((proj.receiver.owner, node))
                continue

            # node is output_CIM of outer Composition (i.e., end of pathway) which shouldn't happen yet
            if isinstance(node, CompositionInterfaceMechanism) and node is self.output_CIM:
                assert False, (f"PROGRAM ERROR: 'Got to output_CIM of outermost Composition '({self.name})' "
                               f"without detecting OUTPUT NODE at end of pathway")

            # End of pathway: OUTPUT Node of outer Composition
            if current_comp == self and node in current_comp.get_nodes_by_role(NodeRole.OUTPUT):
                pathways.append(create_pathway(current_comp, node))
                continue

            # # Get all efferent Projections of node,
            # #   including direct projections out of a nested Composition implemented in PyTorchCompositionWrapper
            efferent_projs = [(p, p.receiver.owner) for p in node.efferents if p in current_comp.projections]
            if not efferent_projs:
                efferent_projs = [(p, p.receiver.owner) for p in node.efferents
                                  if p in current_comp._pytorch_projections]

            # Follow efferent Projection to next Node in pathway
            for efferent_proj, rcvr in efferent_projs:
                # Ignore efferent Projections that do not have a learnable attribute
                #   or are ModulatoryProjections (i.e., including LearningProjections)
                # Note: if learnable==False, it will be passed along to PyTorch in PytorchProjectionWrapper
                if not hasattr(efferent_proj,'learnable') or isinstance(efferent_proj,ModulatoryProjection_Base):
                    continue

                # Deal with Projections to/from CIMs since nested comps can be learned in PyTorch mode
                if isinstance(rcvr, CompositionInterfaceMechanism):

                    # Projection to input_CIM of a nested Composition
                    if rcvr == rcvr.composition.input_CIM:
                        assert rcvr.composition is not current_comp
                        rcvr_comp = rcvr.composition
                        # Get Node(s) in inner Composition to which Node projects (via input_CIM)
                        receivers = rcvr._get_destination_info_from_input_CIM(efferent_proj.receiver)
                        for _, nested_rcvr, _ in [receivers] if isinstance(receivers, tuple) else receivers:
                            if rcvr_comp._input_comp_nodes_to_pytorch_nodes_map:
                                # If nested comp has _input_comp_nodes_to_pytorch_nodes_map, get nested_rcvr from it
                                nested_rcvr = rcvr_comp._input_comp_nodes_to_pytorch_nodes_map[nested_rcvr]
                            else:
                                # Otherwise, ensure that nested_rcvr is an INPUT Node of rcvr_comp
                                assert nested_rcvr in rcvr_comp.get_nodes_by_role(NodeRole.INPUT), \
                                    f"PROGRAM ERROR: '{nested_rcvr.name}' is not an INPUT Node of '{rcvr_comp.name}'"
                                # Assign efferent_proj (Projection to input_CIM) since it should be learned in PyTorch mode
                            rcvr_comp._add_dependency(node, efferent_proj, nested_rcvr,
                                                      dependency_dict, queue, rcvr_comp)

                    # rcvr is Nested Composition output_CIM:
                    # Projection is to output_CIM exiting from a nested Composition
                    elif rcvr == current_comp.output_CIM and current_comp is not self:

                        # Get output_CIM info for current efferent_proj
                        output_CIM_input_port = efferent_proj.receiver
                        output_CIM = output_CIM_input_port.owner
                        # Get port of output_CIM that efferent_proj sends to, for use in findings its receiver(s) below
                        if efferent_proj in current_comp.projections:
                            output_CIM_output_port = output_CIM.port_map[efferent_proj.sender][1]
                        elif efferent_proj in current_comp._pytorch_projections:
                            # FIX: 3/8/25 - THERE MUST BE AN EASIER WAY TO GET THIS MORE DIRECTLY
                            output_CIM_output_port = \
                                (output_CIM.port_map)[efferent_proj.receiver.path_afferents[0].sender][1]

                        # Get all Node(s) in outer Composition to which node projects (via output_CIM)
                        receivers = rcvr._get_destination_info_for_output_CIM(output_CIM_output_port)
                        # Replace efferent_proj(s) with one(s) from output_CIM to rcvr(s) in outer Composition,
                        #   since that(those) is(are) the one(s) that should be learned in PyTorch mode
                        # Note:  _get_destination_info_for_output_CIM returns list of destinations
                        #        in order of output_CIM.output_port.efferents
                        if receivers:
                            for efferent_idx, receiver in enumerate(receivers):
                                if receiver:
                                    _, rcvr, rcvr_comp = receiver
                                    assert rcvr_comp is not current_comp
                                efferent_proj = output_CIM_output_port.efferents[efferent_idx]
                                rcvr_comp._add_dependency(node, efferent_proj, rcvr,
                                                          dependency_dict, queue, rcvr_comp)
                        else:
                            pathways.append(create_pathway(current_comp, node))

                    # rcvr is Outermost Composition output_CIM:
                    # End of pathway: Direct projection from output_CIM of nested comp to outer comp's output_CIM
                    elif rcvr is self.output_CIM:
                        # Assign node that projects to current node as OUTPUT Node for pathway
                        node_output_port = efferent_proj.sender
                        _, sender, _ = node._get_source_info_from_output_CIM(node_output_port)
                        pathway = create_pathway(current_comp, node)
                        if pathway:
                            queue.popleft()
                            pathways.append(pathway)

                    else:
                        assert False, f"PROGRAM ERROR:  Unrecognized CompositionInterfaceMechanism: {rcvr}"

                else:
                    if rcvr in current_comp.nodes:
                        # rcvr is still in nested Composition, so keep traversing that
                        current_comp._add_dependency(node, efferent_proj, rcvr, dependency_dict, queue, current_comp)
                        continue
                    elif rcvr in self.nodes:
                        # rcvr is in outer Composition (presumably a direct Pytorch Projection out of nested comp)
                        self._add_dependency(node, efferent_proj, rcvr, dependency_dict, queue, self)
                        continue
                    else:
                        assert False, \
                            (f"PROGRAM ERROR:  Unrecognized receiver ('{rcvr.name}') of Projection from '{node.name}'.")

        return pathways

    def _add_dependency(self,
                        sender:ProcessingMechanism,
                        projection:MappingProjection,
                        receiver:ProcessingMechanism,
                        dependency_dict:dict,
                        queue:deque,
                        comp:Composition):
        """Append dependencies to dependency list, and next node to queue used in _get_pytorch_backprop_pathway()
        This uses the Projection from node to receiver to implement the relevant dependencies for construcing the
        pathway;  however, this can be overridden by a subclass of Autodiff to implement a custom pathway
        (see example in GRUComposition).
        """
        dependency_dict[receiver] = projection
        dependency_dict[projection] = sender
        queue.append((receiver, comp))

    # CLEANUP: move some of what's done in the methods below to a "validate_params" type of method
    @handle_external_context()
    def _build_pytorch_representation(self, context=None, refresh=None, base_context=Context(execution_id=None)):
        """Builds a Pytorch representation of the AutodiffComposition"""
        if self.scheduler is None:
            self.scheduler = Scheduler(graph=self.graph_processing)
        if self.parameters.pytorch_representation._get(context=context, fallback_value=None) is None or refresh:
            model = self.pytorch_composition_wrapper_type(composition=self,
                                                          device=self.device,
                                                          context=context,
                                                          base_context=base_context,
                                                          )

        # Set up optimizer function
        learning_rate = self._runtime_learning_rate or self.learning_rate
        old_opt = self.parameters.optimizer._get(context)
        if (old_opt is None or refresh) and refresh is not False:
            self._instantiate_optimizer(refresh, learning_rate, context)
        # Set up loss function
        if self.loss_function is not None:
            logger.warning("Overwriting 'loss_function' for AutodiffComposition {}! Old loss function: {}".format(
                self, self.loss_function))
        if callable(self.loss_spec):
            self.loss_function = self.loss_spec
        else:
            self.loss_function = self._get_loss(self.loss_spec)

        return self.parameters.pytorch_representation._get(context)

    def _instantiate_optimizer(self, refresh, learning_rate, context):
        if not is_numeric_scalar(learning_rate):
            raise AutodiffCompositionError("Learning rate must be an integer or float value.")
        if self.optimizer_type not in ['sgd', 'adam']:
            raise AutodiffCompositionError("Invalid optimizer specified. Optimizer argument must be a string. "
                                           "Currently, Stochastic Gradient Descent and Adam are the only available "
                                           "optimizers (specified as 'sgd' or 'adam').")
        pytorch_rep = self.parameters.pytorch_representation._get(context)
        params = pytorch_rep.parameters()
        if self.optimizer_type == 'sgd':
            opt = optim.SGD(params, lr=learning_rate, weight_decay=self.weight_decay)
        else:
            opt = optim.Adam(params, lr=learning_rate, weight_decay=self.weight_decay)

        pytorch_rep._parse_optimizer_params(context)
        for param_group in pytorch_rep._optimizer_param_groups:
            opt.add_param_group(param_group)

        # Assign optimizer to AutodiffComposition and PytorchCompositionWrapper
        self.parameters.optimizer._set(opt, context, skip_history=True, skip_log=True)
        pytorch_rep.optimizer = opt

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
        elif loss_spec == Loss.BINARY_CROSS_ENTROPY:
            return nn.BCELoss()
        elif loss_spec == Loss.L1:
            return nn.L1Loss(reduction='sum')
        elif loss_spec == Loss.NLL:
            return nn.NLLLoss(reduction='sum')
        elif loss_spec == Loss.POISSON_NLL:
            return nn.PoissonNLLLoss(reduction='sum')
        elif loss_spec == Loss.KL_DIV:
            return nn.KLDivLoss(reduction='sum')
        else:
            raise AutodiffCompositionError(f"Loss type {loss_spec} not recognized. 'loss_function' argument must be a "
                                           f"Loss enum or function. Currently, the recognized loss types are: "
                                           f"L1 (Mean), SSE (sum squared error), CROSS_ENTROPY, NLL (negative log "
                                           f"likelihood), POISSONNLL (Poisson negative log likelihood, "
                                           f"and KL_DIV (KL divergence.")

    def get_target_nodes(self, execution_mode=pnlvm.ExecutionMode.PyTorch):
        """Return `TARGET` `Nodes <Composition_Nodes>` of the AutodiffComposition."""
        self.infer_backpropagation_learning_pathways(execution_mode=execution_mode)
        return super(AutodiffComposition, self).get_target_nodes()

    def autodiff_forward(self, inputs, targets,
                         synch_with_pnl_options, retain_in_pnl_options,
                         execution_mode, scheduler, context):
        """
        Perform forward pass of model and compute loss for a batch of trials in Pytorch mode.
        Losses are then accumulated, error is backpropagated by compositionrunner.run_learning()
          before the next time it calls run(), in a call to backward() by do_gradient_optimization()
          in _batch_inputs() or _batch_function_inputs(),
        """
        assert execution_mode is pnlvm.ExecutionMode.PyTorch
        pytorch_rep = self.parameters.pytorch_representation._get(context)

        # --------- Get current values of nodes  -------------------------------------------------

        #   should return 2d values for each component

        # Get value of INPUT nodes for current trial
        curr_tensors_for_inputs = {}
        for component in inputs.keys():
            if not isinstance(inputs[component], torch.Tensor):
                curr_tensors_for_inputs[component] = torch.tensor(inputs[component], device=self.device).double()
            else:
                curr_tensors_for_inputs[component] = inputs[component]

        # Execute PytorchCompositionWrapper to get value of all OUTPUT nodes for current trial
        curr_tensors_for_outputs = pytorch_rep.forward(curr_tensors_for_inputs, None, synch_with_pnl_options, context)

        # Get value of OUTPUT nodes that are being trained (i.e., for which there are TARGET nodes)
        curr_tensors_for_trained_outputs = {k:v for k,v in curr_tensors_for_outputs.items()
                                            if k in self.outputs_to_targets_map}

        # Get value of TARGET nodes for current trial
        curr_tensors_for_targets = {}
        for component, target in targets.items():
            if isinstance(target, torch.Tensor) or isinstance(target, np.ndarray):
                curr_tensors_for_targets[component] = [target[:, i, :] for i in range(target.shape[1])]
            else:
                # It's  a list, of lists, of torch tensors because it is ragged
                num_outputs = len(targets[component][0])
                curr_tensors_for_targets[component] = [torch.stack([batch_elem[i]
                                                                    for batch_elem in target])
                                                       for i in range(num_outputs)]

        # Map value of TARGET nodes to trained OUTPUT nodes
        curr_target_tensors_for_trained_outputs = {}
        for trained_output, target in self.outputs_to_targets_map.items():
            curr_target_tensors_for_trained_outputs[trained_output] = curr_tensors_for_targets[target]

        # --------- Compute the loss (TARGET-OUTPUT) for each trained OUTPUT node  ---------------------------

        # Calculate and track the loss over the trained OUTPUT nodes:
        #   curr_target_tensors_for_trained_outputs compared against curr_tensors_for_trained_outputs
        for component, outputs in curr_tensors_for_trained_outputs.items():
            trial_loss = 0
            targets = curr_target_tensors_for_trained_outputs[component]
            num_outputs = outputs.shape[1] if type(outputs) is torch.Tensor else len(outputs[0])
            for i in range(num_outputs):
                # loss only accepts 0 or 1d target. reshape assuming pytorch_rep.minibatch_loss dim is correct

                # Get the output, if it's a torch tensor we can slice, if it's a list of list (its ragged) and we
                # need to index
                output = outputs[:, i, :] if type(outputs) is torch.Tensor else torch.stack([batch_elem[i] for batch_elem in outputs])

                comp_loss = self.loss_function(
                    output,
                    torch.atleast_1d(targets[i])
                )
                comp_loss = comp_loss.reshape_as(pytorch_rep.minibatch_loss)
                trial_loss += comp_loss
            pytorch_rep.minibatch_loss += trial_loss
        pytorch_rep.minibatch_loss_count += 1

        # --------- Return the values of output of trained nodes and all nodes  ---------------------------------------

        # IMPLEMENTATION NOTE: Need values in order corresponding to output_CIM Ports.

        # Get output Nodes, their out_ports and corresponding indices
        #     in order of outermost AutodiffComposition's output_CIM Ports
        outputs_idx_port_node_comp = []
        for port in self.output_CIM.input_ports:
            source_info = self.output_CIM._get_source_info_from_output_CIM(port)
            source_ouput_port_idx = source_info[1].output_ports.index(source_info[0])
            outputs_idx_port_node_comp.append(tuple((source_ouput_port_idx, *source_info)))

        # Assign values to trained_output_values and all_output_values
        trained_output_values = []
        all_output_values = []
        for item in outputs_idx_port_node_comp:
            idx, port, node, comp = item
            if comp._trained_comp_nodes_to_pytorch_nodes_map:
                node = comp._trained_comp_nodes_to_pytorch_nodes_map[node]
            outputs = curr_tensors_for_outputs[node]
            if type(outputs) is torch.Tensor:
                output = outputs[:, idx, ...]
            else:
                output = torch.stack([batch_elem[idx] for batch_elem in outputs])
            output = output.detach().cpu().numpy().copy().tolist()
            if self.targets_from_outputs_map.values():
                trained_output_values += [output]
            all_output_values += [output]

        # Turn into a numpy array, possibly ragged
        all_output_values = convert_to_np_array(all_output_values)

        # Swap the first two dimensions (output_port, batch) to (batch, output_port)
        all_output_values = all_output_values.swapaxes(0, 1)

        pytorch_rep.all_output_values = all_output_values

        # Get values of TARGET nodes
        target_values = [value[0].detach().cpu().numpy().copy().tolist()
                         for value in list(curr_tensors_for_targets.values())]
        pytorch_rep.target_values = target_values

        # Synchronize outcomes after every trial if specified
        # IMPLEMENTATION NOTE: RESULTS is not included here as it is handled in call to autodiff._update_results()
        pytorch_rep.synch_with_psyneulink(synch_with_pnl_options,
                                          [OPTIMIZATION_STEP, TRIAL],
                                          context,
                                          [NODE_VARIABLES, NODE_VALUES])
        pytorch_rep.retain_for_psyneulink({TRAINED_OUTPUTS: trained_output_values,
                                           TARGETS: target_values},
                                          retain_in_pnl_options,
                                          context)

        return trained_output_values, all_output_values

    def clear_losses(self, context=None):
        self.losses = []
        if self.pytorch_representation:
            self.pytorch_representation.retained_losses = []

    def do_gradient_optimization(self, retain_in_pnl_options, context, optimization_num=None):
        """Compute loss and use in call to autodiff_backward() to compute gradients and update PyTorch parameters.
        Update parameters (weights) based on trial(s) executed since last optimization,
        Reinitizalize minibatch_loss and minibatch_loss_count
        """
        pytorch_rep = self.parameters.pytorch_representation._get(context=context)
        minibatch_loss = pytorch_rep.minibatch_loss / pytorch_rep.minibatch_loss_count

        self.autodiff_backward(minibatch_loss, context)

        # # Save loss for current round of optimization
        pytorch_rep.retain_for_psyneulink({LOSSES: minibatch_loss}, retain_in_pnl_options, context)

        # Reset minibatch_loss for next round of optimization
        pytorch_rep.minibatch_loss = torch.zeros(1, device=self.device).double()
        pytorch_rep.minibatch_loss_count = 0

    def autodiff_backward(self, minibatch_loss, context):
        """Calculate gradients and apply to PyTorch model parameters (weights)"""
        pytorch_rep = self.parameters.pytorch_representation._get(context=context)
        optimizer = pytorch_rep.optimizer

        # Gradient updates
        optimizer.zero_grad()
        # Compute and log average loss over all trials since last update
        minibatch_loss.backward(retain_graph=not self.force_no_retain_graph)
        # Update weights and copy to PNL
        optimizer.step()
        assert True

    def _gen_llvm_function(self, *, ctx:pnlvm.LLVMBuilderContext, tags:frozenset):
        if "run" in tags:
            return pnlvm.codegen.gen_composition_run(ctx, self, tags=tags)
        else:
            return pnlvm.codegen.gen_autodiffcomp_exec(ctx, self, tags=tags)

    def _get_total_loss(self, num_trials: int=1, context:Context=None):
        return sum(self.parameters.trial_losses._get(context)[-num_trials:]) /num_trials

    def _get_autodiff_inputs_values(self, input_dict: dict):
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
            if (mech in self.get_nested_input_nodes_at_all_levels()
                    and mech not in self.get_nodes_by_role(NodeRole.TARGET)):
                # Pass along inputs to all INPUT Nodes except TARGETS
                # (those are handled separately in _get_autodiff_targets_values)
                if torch_available:
                    # Convert to torch tensor of type expected by PytorchCompositionWrapper
                    # values = torch.tensor(values, dtype=self.torch_dtype, device=self.device)
                    values = values.type(self.torch_dtype)
                autodiff_input_dict[node] = values
        return autodiff_input_dict

    def _get_autodiff_targets_values(self, input_dict):
        """Return dict with input values for TARGET Nodes
        Get inputs to TARGET Nodes used for computation of loss in autodiff_forward().
        Uses input_dict to get input values for TARGET Nodes that are INPUT Nodes of the AutodiffComposition,
        If a TARGET Node is not an INPUT Node, it is assumed to be the target of a projection from an INPUT Node
        and the value is determined by searching recursively for the input Node that projects to the TARGET Node.

        Returns
        ---------
        A dict mapping TARGET Nodes -> target values
        """
        target_values = {}
        def get_target_value(target):
            if target in self.get_nodes_by_role(NodeRole.INPUT):
                return input_dict[target]
            if len(target.path_afferents) > 1:
                raise AutodiffCompositionError(f"TARGET Node '{target.name}' (for '{self.name}')"
                                               f"cannot have more than one afferent projection.")
            target = target.path_afferents[0].sender.owner
            return get_target_value(target)

        for target in self.targets_from_outputs_map:
            target_values[target] = get_target_value(target)
        return target_values

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

    def _identify_target_nodes(self, context)->list:
        """Recursively call all nested AutodiffCompositions to assign TARGET nodes for learning"""
        # Default is to use OUTPUT
        target_nodes = [node for node in self.get_nodes_by_role(NodeRole.OUTPUT)
                        if not isinstance(node, Composition)]
        for node in self.nodes:
            if isinstance(node, AutodiffComposition):
                target_nodes.extend(node._identify_target_nodes(context))
        return target_nodes

    def _get_valid_weights_shape(self, projection):
        pnl_wt_matrix = projection.defaults.matrix
        if not isinstance(pnl_wt_matrix, np.ndarray):
            assert is_matrix_keyword(pnl_wt_matrix)
            pnl_wt_matrix = projection._get_matrix_from_keyword(pnl_wt_matrix)
        return pnl_wt_matrix.shape

    @handle_external_context()
    def set_weights(self, pnl_proj, weights:Union[list, np.ndarray], context=None):
        """Set weights for specified Projection."""
        valid_shape = self._get_valid_weights_shape(pnl_proj)
        assert weights.shape == valid_shape, \
            (f"PROGRAM ERROR: Shape of weights in 'weights' arg of '{self.name}.set_weights' "
             f"Specified weights do not match required shape ({valid_shape}).)")
        pnl_proj.parameters.matrix._set(weights, context)
        pnl_proj.parameter_ports['matrix'].parameters.value._set(weights, context)

    @handle_external_context(fallback_default=True)
    def learn(self,
              *args,
              synch_projection_matrices_with_torch:Optional[LEARNING_SCALE_LITERALS]=NotImplemented,
              synch_node_variables_with_torch:Optional[LEARNING_SCALE_LITERALS]=NotImplemented,
              synch_node_values_with_torch:Optional[LEARNING_SCALE_LITERALS]=NotImplemented,
              synch_results_with_torch:Optional[LEARNING_SCALE_LITERALS]=NotImplemented,
              retain_torch_trained_outputs:Optional[LEARNING_SCALE_LITERALS]=NotImplemented,
              retain_torch_targets:Optional[LEARNING_SCALE_LITERALS]=NotImplemented,
              retain_torch_losses:Optional[LEARNING_SCALE_LITERALS]=NotImplemented,
              context: Context = None,
              base_context: Context = Context(execution_id=None),
              skip_initialization: bool = False,
              **kwargs
              ) -> list:
        """Override to handle synch and retain args
        Note: defaults for synch and retain args are set to NotImplemented, so that the user can specify None if
              they want to locally override the default values for the AutodiffComposition (see docstrings for run()
              and _parse_synch_and_retain_args() for additonal details).
        """
        execution_phase_at_entry = context.execution_phase
        context.execution_phase = ContextFlags.PREPARING

        execution_mode = self._get_execution_mode(kwargs.pop('execution_mode', None))
        context.execution_phase = execution_phase_at_entry

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
            self.infer_backpropagation_learning_pathways(execution_mode, context=context)
            self._built_pathways = True

        synch_with_pnl_options, retain_in_pnl_options = (
            self._parse_synch_and_retain_args(synch_projection_matrices_with_torch,
                                              synch_node_variables_with_torch,
                                              synch_node_values_with_torch,
                                              synch_results_with_torch,
                                              retain_torch_trained_outputs,
                                              retain_torch_targets,
                                              retain_torch_losses,
                                              context=context,
                                              **kwargs))

        if execution_mode == pnlvm.ExecutionMode.PyTorch and not torch_available:
            raise AutodiffCompositionError(f"'{self.name}.learn()' has been called with ExecutionMode.Pytorch, "
                                           f"but Pytorch module ('torch') is not installed. "
                                           f"Please install it with `pip install torch` or `pip3 install torch`")

        return super().learn(*args,
                             synch_with_pnl_options=synch_with_pnl_options,
                             retain_in_pnl_options=retain_in_pnl_options,
                             execution_mode=execution_mode,
                             context=context,
                             base_context=base_context,
                             skip_initialization=skip_initialization,
                             **kwargs)

    def _parse_synch_and_retain_args(self,
                                     synch_projection_matrices_with_torch:Optional[LEARNING_SCALE_LITERALS],
                                     synch_node_variables_with_torch:Optional[LEARNING_SCALE_LITERALS],
                                     synch_node_values_with_torch:Optional[LEARNING_SCALE_LITERALS],
                                     synch_results_with_torch:Optional[LEARNING_SCALE_LITERALS],
                                     retain_torch_trained_outputs:Optional[LEARNING_SCALE_LITERALS],
                                     retain_torch_targets:Optional[LEARNING_SCALE_LITERALS],
                                     retain_torch_losses:Optional[LEARNING_SCALE_LITERALS],
                                     context: Context = None,
                                     **kwargs
                                     )->tuple:
        # Remove args from kwargs in case called from run() (won't be there if called from learn()
        if synch_projection_matrices_with_torch == NotImplemented:
            synch_projection_matrices_with_torch = kwargs.pop('synch_projection_matrices_with_torch', NotImplemented)
            if synch_projection_matrices_with_torch == NotImplemented:
                synch_projection_matrices_with_torch = self.parameters.synch_projection_matrices_with_torch.default_value
        if synch_node_variables_with_torch == NotImplemented:
            synch_node_variables_with_torch = kwargs.pop('synch_node_variables_with_torch', NotImplemented)
            if synch_node_variables_with_torch == NotImplemented:
                synch_node_variables_with_torch = self.parameters.synch_node_variables_with_torch.default_value
        if synch_node_values_with_torch == NotImplemented:
            synch_node_values_with_torch = kwargs.pop('synch_node_values_with_torch', NotImplemented)
            if synch_node_values_with_torch == NotImplemented:
                synch_node_values_with_torch = self.parameters.synch_node_values_with_torch.default_value
        if synch_results_with_torch == NotImplemented:
            synch_results_with_torch = kwargs.pop('synch_results_with_torch', NotImplemented)
            if synch_results_with_torch == NotImplemented:
                synch_results_with_torch = self.parameters.synch_results_with_torch.default_value
        if retain_torch_trained_outputs == NotImplemented:
            retain_torch_trained_outputs = kwargs.pop('retain_torch_trained_outputs', NotImplemented)
            if retain_torch_trained_outputs == NotImplemented:
                retain_torch_trained_outputs = self.parameters.retain_torch_trained_outputs.default_value
        if retain_torch_targets == NotImplemented:
            retain_torch_targets = kwargs.pop('retain_torch_targets', NotImplemented)
            if retain_torch_targets == NotImplemented:
                retain_torch_targets = self.parameters.retain_torch_targets.default_value
        if retain_torch_losses == NotImplemented:
            retain_torch_losses = kwargs.pop('retain_torch_losses', NotImplemented)
            if retain_torch_losses == NotImplemented:
                retain_torch_losses = self.parameters.retain_torch_losses.default_value

        if self.minibatch_size > 1:
            args_str = []
            if retain_torch_trained_outputs in {OPTIMIZATION_STEP, TRIAL}:
                args_str.append('retain_torch_trained_outputs')
            if retain_torch_losses in {OPTIMIZATION_STEP,TRIAL}:
                args_str.append('retain_torch_losses')
            if retain_torch_targets in {OPTIMIZATION_STEP,TRIAL}:
                args_str.append('retain_torch_targets')
            if args_str:
                arg_args = 'args' if len(args_str) == 1 else 'arg'
                is_are = 'is' if len(args_str) == 1 else 'are'
                raise AutodiffCompositionError(f"The {' ,'.join(args_str)} {arg_args} in the learn() method for "
                                               f"'{self.name}' {is_are} specifed as 'OPTIMIZATION' or 'TRIAL', but "
                                               f"'minibatch_size` ({self.minibatch_size}) != 1, so "
                                               f"{', '.join([arg.split('_')[-1] for arg in args_str])} "
                                               f"will be updated only at the end of a minibatch; "
                                               f"use 'MINIBATCH' for the {arg_args} to avoid this warning.")

        # Package options for synching and tracking into dictionaries as arguments to learning and exec methods
        synch_with_pnl_options = {MATRIX_WEIGHTS: synch_projection_matrices_with_torch
                                                  or self.parameters.synch_projection_matrices_with_torch._get(context),
                                  NODE_VARIABLES: synch_node_variables_with_torch
                                               or self.parameters.synch_node_variables_with_torch._get(context),
                                  NODE_VALUES: synch_node_values_with_torch
                                               or self.parameters.synch_node_values_with_torch._get(context),
                                  RESULTS: synch_results_with_torch
                                                    or self.parameters.synch_results_with_torch._get(context)}

        retain_in_pnl_options = {TRAINED_OUTPUTS: retain_torch_trained_outputs
                                                   or self.parameters.retain_torch_trained_outputs._get(context),
                                 TARGETS: retain_torch_targets or self.parameters.retain_torch_targets._get(context),
                                 LOSSES: retain_torch_losses or self.parameters.retain_torch_losses._get(context)}

        return synch_with_pnl_options, retain_in_pnl_options

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

    @handle_external_context(fallback_default=True)
    def execute(self,
                inputs=None,
                num_trials=None,
                minibatch_size=1,
                optimizations_per_minibatch=1,
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
                synch_with_pnl_options:Optional[Mapping]=None,
                retain_in_pnl_options:Optional[Mapping]=None,
                report_output:ReportOutput=ReportOutput.OFF,
                report_params:ReportOutput=ReportParams.OFF,
                report_progress:ReportProgress=ReportProgress.OFF,
                report_simulations:ReportSimulations=ReportSimulations.OFF,
                report_to_devices:ReportDevices=None,
                report=None,
                report_num=None,
                )->np.ndarray:
        """Override to execute autodiff_forward() in learning mode if execute_mode is not Python"""

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

                autodiff_inputs = self._get_autodiff_inputs_values(inputs)
                autodiff_targets = self._get_autodiff_targets_values(inputs)

                # Begin reporting of learning TRIAL:
                report(self,
                       LEARN_REPORT,
                       # EXECUTE_REPORT,
                       report_num=report_num,
                       scheduler=scheduler,
                       content='trial_start',
                       context=context)

                self._build_pytorch_representation(context, base_context=base_context)
                trained_output_values, all_output_values = \
                                                self.autodiff_forward(inputs=autodiff_inputs,
                                                                      targets=autodiff_targets,
                                                                      synch_with_pnl_options=synch_with_pnl_options,
                                                                      retain_in_pnl_options=retain_in_pnl_options,
                                                                      execution_mode=execution_mode,
                                                                      scheduler=scheduler,
                                                                      context=context)
                execution_phase = context.execution_phase
                context.execution_phase = ContextFlags.PROCESSING
                context.execution_phase = execution_phase

                # Complete TRIAL Panel for output report, and report progress
                report(self,
                       # [LEARN_REPORT],
                       [EXECUTE_REPORT, PROGRESS_REPORT],
                       report_num=report_num,
                       scheduler=scheduler,
                       content='trial_end',
                       context=context)

                scheduler.get_clock(context)._increment_time(TimeScale.TRIAL)

                self.most_recent_context = context
                return all_output_values

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

    @handle_external_context(fallback_default=True)
    def run(self, *args,
            synch_projection_matrices_with_torch:Optional[LEARNING_SCALE_LITERALS]=NotImplemented,
            synch_node_variables_with_torch:Optional[LEARNING_SCALE_LITERALS]=NotImplemented,
            synch_node_values_with_torch:Optional[LEARNING_SCALE_LITERALS]=NotImplemented,
            synch_results_with_torch:Optional[LEARNING_SCALE_LITERALS]=NotImplemented,
            retain_torch_trained_outputs:Optional[LEARNING_SCALE_LITERALS]=NotImplemented,
            retain_torch_targets:Optional[LEARNING_SCALE_LITERALS]=NotImplemented,
            retain_torch_losses:Optional[LEARNING_SCALE_LITERALS]=NotImplemented,
            batched_results:bool=False,
            context: Context = None,
            **kwargs):
        """Override to handle synch and retain args if run called directly from run() rather than learn()
        Note: defaults for synch and retain args are NotImplemented, so that the user can specify None if they want
              to locally override the default values for the AutodiffComposition (see _parse_synch_and_retain_args()
              for details). This is distinct from the user assigning the Parameter default_values(s), which is done
              in the AutodiffComposition constructor and handled by the Parameter._specify_none attribute.
        """

        # Store whether we need to return results list with a batch dimension, or flatten it
        self.batched_results = batched_results

        if not (SYNCH_WITH_PNL_OPTIONS in kwargs and RETAIN_IN_PNL_OPTIONS in kwargs):
            # No synch_with_pnl_options and retain_in_pnl_options dicts:
            # - so must have been called from run directly rather than learn
            # - therefore, must validate, parse and package options into those dicts
            if synch_results_with_torch is NotImplemented:
                # IMPLEMENTATION NOTE:
                #     If synch_results_with_torch is not specified by the user in call from run(), set it to
                #     MINIBATCH (rather than RUN, which is the default_value for calls from AutodiffComposition);
                #     this is required for calling _update_results() from Composition.run(), which does not itself
                #     know about synch and retain options, and the expected default behavior of which is to update
                #     results on every try in a call to run().
                synch_results_with_torch = MINIBATCH
            synch_with_pnl_options, retain_in_pnl_options = (
                self._parse_synch_and_retain_args(synch_projection_matrices_with_torch,
                                                   synch_node_variables_with_torch,
                                                   synch_node_values_with_torch,
                                                   synch_results_with_torch,
                                                   retain_torch_trained_outputs,
                                                   retain_torch_targets,
                                                   retain_torch_losses,
                                                  context=context,
                                                   **kwargs))
            kwargs[SYNCH_WITH_PNL_OPTIONS] = synch_with_pnl_options
            kwargs[RETAIN_IN_PNL_OPTIONS] = retain_in_pnl_options

        results = super(AutodiffComposition, self).run(*args, context=context, **kwargs)
        if EXECUTION_MODE in kwargs and kwargs[EXECUTION_MODE] is pnlvm.ExecutionMode.PyTorch:
            # Synchronize specified outcomes at end of run
            pytorch_rep = self.parameters.pytorch_representation.get(context)
            if pytorch_rep:
                pytorch_rep.synch_with_psyneulink(kwargs[SYNCH_WITH_PNL_OPTIONS], RUN, context)

        return results

    def _update_results(self, results, trial_output, execution_mode, synch_with_pnl_options, context):
        """Track results at specified frequency during learning"""
        if execution_mode is pnlvm.ExecutionMode.PyTorch:

            # Check if the trial_output is atleast 3D
            is_output_3d = trial_output.ndim >= 3 or (trial_output.ndim == 2 and len(trial_output) > 0 and
                                                      isinstance(trial_output[0, 0], (np.ndarray, list)))

            if (RESULTS in synch_with_pnl_options
                    and synch_with_pnl_options[RESULTS] in {TRIAL, MINIBATCH}):
                # Use Composition's own _update_results method since no savings when done trial-by-trial
                if not self.batched_results and is_output_3d:
                    for out in trial_output:
                        super()._update_results(results, out, execution_mode, synch_with_pnl_options, context)
                else:
                    super()._update_results(results, trial_output, execution_mode, synch_with_pnl_options, context)

            elif (RESULTS in synch_with_pnl_options
                  and synch_with_pnl_options[RESULTS] == RUN):
                # Use pytorch_reps method to keep a local list of results that are copied to autodiff.results after run
                pytorch_rep = self.parameters.pytorch_representation._get(context)
                if not self.batched_results and is_output_3d:
                    for out in trial_output:
                        pytorch_rep.retain_results(out)
                else:
                    pytorch_rep.retain_results(trial_output)
        else:
            super()._update_results(results, trial_output, execution_mode, synch_with_pnl_options, context)

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
    def load(self, path:PosixPath=None, directory:str=None, filename:str=None, context=None, weights_only:bool=False):
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
            state = torch.load(path, weights_only=weights_only)
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

    if torch_available:
        @handle_external_context(fallback_most_recent=True)
        def copy_torch_param_to_projection_matrix(self,
                                                  projection:Union[str, MappingProjection],
                                                  torch_param:Union[torch.nn.Parameter, torch.Tensor, str, int],
                                                  torch_module:torch.nn.Module=None,
                                                  torch_slice:slice=None,
                                                  validate:bool=True,
                                                  context:Optional[Union[Context, str]]=None)->np.ndarray:
            """Assign torch Parameter to `matrix <MappingProjection.matrix>` Parameter of specified `MappingProjection`.
            Return torch_param as the np.ndarray assigned to `matrix <MappingProjection.matrix>` Parameter of
            **projection**.

            Arguments
            ---------

            projection : str or MappingProjection
               specifies `MappingProjection` to which the torch_param is assigned as its `matrix
               <MappingProjection.matrix>` Parameter;  if specified as a str, it must be the name of a
               MappingProjection in the AutodiffComposition.

            torch_param : torch.nn.Parameter, str or int
               specifies torch_param to assign to the `matrix <MappingProjection.matrix>` Parameter of **projection**;
               if it is a torch.nn.Parameter or torch.Tensor, then the **torch_module** argument does not need to be
               specified; if specified as a str or int, it must be the name of a torch Parameter (used to access it in
               the state_dict) or its index (used to access it in the parameterlist) of the **torch_module** argument,
               which must be also specified.

            torch_module : torch.nn.Module : default None
               specifies a torch.nn.Module containing **torch_param** assigned to the`matrix<MappingProjection.matrix>`
               Parameter of **projection**; this does not need to be specified if **torch_param** is a
               torch.nn.Parameter or torch.Tensor, but must be specified if **torch_param** is a str or int.

            torch_slice : slice : default None
               specifies a slice of **torch_param** to assign to the `matrix <MappingProjection.matrix>` Parameter
               of **projection**; if it is not specified, the entire tensor of **torch_param** is used.

              .. warning::
                 **torch_slice** should not be specified if the specification of **torch_param** already takes this
                 into account.

            validate : bool : default True
               specifies whether to validate the **projection** and **torch_param** arguments; setting it to False
               results in more efficient processing if this method is called frequently; however, invalid arguments will
               raise standard Python exceptions rather than more informative AutodiffComposition errors, and unexpected
               results may go unnoticed.

               .. warning::
                  if validate is False, for efficiency: **projection** *must* be a `MappingProjection`, **torch_param**
                  *must* be a torch.Tensor, and both **torch_module** and **torch_slice** are ignored.

            context : Context or None : default most recent Context
               specifies context to use for the value of Projection.matrix;  if it is not provided, then a default
               `Context` is constructed using the `name <Composition.name>` of the AutodiffComposition as the
               `execution_id <Context.execution_id>`, commensurate with the one used bydefault for its `execution
               <AutodiffComposition_Execution>`.
            """
            if validate:
                torch_tensor, projection = self._validate_torch_param_and_projection(torch_param,
                                                                                     torch_module,
                                                                                     torch_slice,
                                                                                     projection)
            else:
                # Assume **torch_param** is passed in as Tensor and **projection** as Projection if validate is False
                torch_tensor = torch_param[torch_slice] if torch_slice else torch_param

            torch_param_as_pnl_matrix = torch_tensor.detach().cpu().numpy().T
            projection.parameters.matrix._set(torch_param_as_pnl_matrix, context)
            projection.parameter_ports['matrix'].parameters.value._set(torch_param_as_pnl_matrix, context)
            return torch_param_as_pnl_matrix

        def copy_projection_matrix_to_torch_param(self,
                                                  projection:Union[str, MappingProjection],
                                                  torch_param:Union[torch.nn.Parameter, torch.Tensor, str, int],
                                                  torch_module:torch.nn.Module=None,
                                                  torch_slice:slice=None,
                                                  validate:bool=True,
                                                  context:Optional[Union[Context, str]]=None)->torch.Tensor:
            """Assign the `matrix <MappingProjection.matrix>` Parameter of a `MappingProjection` to a Pytorch Parameter.

            .. warning:
               If the PyTorch Parameter has requires_grad=True, this will impact its updating in PyTorch.

            Return torch.Tensor assigned to **torch_param**

            Arguments
            ---------

            projection : str or MappingProjection
               specifies `MappingProjection`, the `matrix <MappingProjection.matrix>` of which is assigned torch_param;
               if specified as a str, it must be the name of a MappingProjection in the AutodiffComposition.

            torch_param : torch.nn.Parameter, str or int
               specifies torch Parameter to which the `matrix <MappingProjection.matrix>` of the Projection is assigned;
               if it is a torch.nn.Parameter or torch.Tensor, then the **torch_module** argument does not need to be
               specified; if specified as a str or int, it must be the name of a torch Parameter (used to access it in
               the state_dict) or its index (used to access it in the parameterlist) of the **torch_module** argument,
               which must be also specified.

            torch_module : torch.nn.Module : default None
               specifies a torch.nn.Module containing **torch_param** to which the **projection**'s `matrix
               <MappingProjection.matrix>` Parameter is assigned; this does not need to be specified if **torch_param**
               is a torch.nn.Parameter or torch.Tensor, but must be specified if **torch_param** is a str or int.

            torch_slice : slice : default None
               specifies a slice of **torch_param** to assign to the `matrix <MappingProjection.matrix>` Parameter
               of **projection**; if it is not specified, the entire tensor of **torch_param** is used.

              .. warning::
                 **torch_slice** should not be specified if the specification of **torch_param** already takes this
                 into account.

            validate : bool : default True
               specifies whether to validate the **projection** and **torch_param** arguments; setting it to False
               results in more efficient processing if this method is called frequently; however, invalid arguments
               then raise standard Python exceptions rather than more informative AutodiffComposition errors,
               and unexpected results may go unnoticed.

               .. warning::
                  if validate is False, for efficiency: **projection** *must* be a `MappingProjection`, **torch_param**
                  *must* be a torch.Tensor, and both **torch_module** and **torch_slice** are ignored.

            context : Context or None : default most recent Context
               specifies context to use for the value of Projection.matrix;  if it is not provided, then a default
               `Context` is constructed using the `name <Composition.name>` of the AutodiffComposition as the
               `execution_id <Context.execution_id>`, commensurate with the one used bydefault for its `execution
               <AutodiffComposition_Execution>`.
            """
            if validate:
                torch_tensor, projection = self._validate_torch_param_and_projection(torch_param,
                                                                                     torch_module,
                                                                                     torch_slice,
                                                                                     projection)
            # Assume **torch_param** is passed in as a Tensor and **projection** as a Projection if validate is False
            else:
                torch_tensor = torch_param
            if slice is not None:
                torch_tensor = torch_tensor[torch_slice]
            matrix = projection.parameters.matrix.get(context).T.squeeze()
            matrix_as_tensor = torch.tensor(matrix, dtype=torch_tensor.dtype)
            torch_tensor.data.copy_(matrix_as_tensor)
            return matrix_as_tensor

        def _validate_torch_param_and_projection(self, torch_param, torch_module, torch_slice, projection_spec)->tuple:
            """Validate torch and projection arguments for copying between PyTorch and AutodiffComposition.
            Return tuple of torch.Tensor and MappingProjection.
            """
            method_name = 'copy_torch_param_to_projection_matrix'

            # Torch Parameter specification is a Tensor or a torch.nn.Parameter
            if isinstance(torch_param, torch.Tensor):
                torch_tensor = torch_param

            # Torch Parameter specification is a Tensor or a torch.nn.Parameter
            elif isinstance(torch_param, type(None)):
                if isinstance(torch_module, (torch.nn.Parameter, torch.Tensor)):
                    raise AutodiffCompositionError(f"Specification of 'torch_module' arg in {method_name}() is a "
                                                   f"torch Parameter or Tensor; this should be specified using the "
                                                   f"'torch_para' arg.")
                raise AutodiffCompositionError(f"The 'torch_param' arg in {method_name}() ({torch_param}) must be "
                                               f"specified, using either a torch.nn.Parameter or torch.Tensor, or a "
                                               f"str or int paired with specification of a torch.nn.Module in the "
                                               f"'torch_module' arg.")
            # Torch Parameter specification is a torch.nn.Module
            elif isinstance(torch_param, torch.nn.Module):
                raise AutodiffCompositionError(f"Specification of 'torch_param' arg in {method_name}() ({torch_param}) "
                                               f"is a Module, but must be a torch.nn.Parameter, torch.Tensor, str or "
                                               f"int; if a Module is intended, use the 'torch_module' arg, and specify "
                                               f"the Parameter name or index in the 'torch_param' arg.")

            elif isinstance(torch_param, (str, int)):
                if torch_module is None:
                    raise AutodiffCompositionError(f"Specifying of the 'torch_param' arg in {method_name}() with a "
                                                   f"string or int ({torch_param}) requires the 'torch_module' "
                                                   f"arg to be specified as well.")
                if not isinstance(torch_module, torch.nn.Module):
                    raise AutodiffCompositionError(f"Specification of 'torch_module' arg in {method_name}() "
                                                   f"({torch_module}) must be a torch.nn.Module.")
                if isinstance(torch_param, str):
                    # Name of Parameter was specified, so get it from Module's state_dict,
                    if torch_param not in torch_module.state_dict():
                        raise AutodiffCompositionError(f"'{torch_param}' specified in 'torch_param' arg of "
                                                       f"{method_name}() is not the name of a Parameter in the "
                                                       f"state_dict() for '{torch_module}'.")
                    torch_tensor = torch_module.state_dict()[torch_param]
                else:
                    # Index of Parameter was specified, so get it from Module's parameters() list
                    try:
                        torch_tensor = list(torch_module.parameters())[torch_param]
                    except IndexError:
                        raise AutodiffCompositionError(f"The value ({torch_param}) specified in the 'torch_param' arg "
                                                       f"of {method_name}() is not an index within the range of the "
                                                       f"ParameterList specified for the Module ('{torch_module}').")
            else:
                # Unrecognized specification for torch_param arg.
                raise AutodiffCompositionError(f"Specification of 'torch_param' arg in {method_name}() ({torch_param}) "
                                               f"must be a torch.nn.Parameter, torch.Tensor, str or int.")

            if torch_slice is not None:
                if not isinstance(torch_slice, slice):
                    if isinstance(torch_param, (str, int)):
                        param_ref = f"'{torch_param}'" if isinstance(torch_param, str) else f"{torch_param}"
                        raise AutodiffCompositionError(f"Specification of 'torch_slice' arg in {method_name}() "
                                                       f"('{torch_slice}') for Parameter {param_ref} of {torch_module} "
                                                       f"must be a slice.")
                    else:
                        raise AutodiffCompositionError(f"Specification of 'torch_slice' arg in {method_name}() "
                                                       f"({torch_slice}) must be a slice.")
                torch_tensor = torch_tensor[torch_slice]

            # Parse and validate projection spec
            if projection_spec not in self.projections:
                if isinstance(projection_spec, str):
                    raise AutodiffCompositionError(f"'{projection_spec}' in {method_name}() "
                                                   f"is not the name of a Projection in '{self.name}'.")
                elif isinstance(projection_spec, MappingProjection):
                    raise AutodiffCompositionError(f"'{projection_spec.name}' in {method_name}() "
                                                   f"is not a Projection in '{self.name}'.")
                else:
                    assert False, f"PROGRAM ERROR: Illegal type for 'projection' ({projection_spec}) in {method_name}."
            projection = self.projections[projection_spec]

            torch_param_as_pnl_matrix = torch_tensor.detach().cpu().numpy().T
            bias_note = ""
            if torch_param_as_pnl_matrix.ndim == 1:
                # Note: torch biases are 1d, but PNL requires matrices to be 2d
                torch_param_as_pnl_matrix = np.atleast_2d(torch_param_as_pnl_matrix)
                bias_note = (f" [Note: torch biases, usually 1d, have already been converted to 2d "
                             f"to match PsyNeuLink BIAS Nodes Projections.]")
            if torch_param_as_pnl_matrix.shape != projection.parameters.matrix.get().shape:
                raise AutodiffCompositionError(
                    f"Shape of torch parameter {torch_param_as_pnl_matrix.shape} in {method_name}() does not match "
                    f"shape of matrix for '{projection.name}' {projection.parameters.matrix.get().shape}.{bias_note}")
            return torch_tensor, projection

    def show_graph(self, *args, **kwargs):
        """Override to use PytorchShowGraph if show_pytorch is True"""
        return self._show_graph.show_graph(*args, **kwargs)

    @property
    def _dependent_components(self) -> Iterable[Component]:
        res = super()._dependent_components

        # NOTE: _dependent_components should possibly be reworked to be
        # a context-dependent method
        for pytorch_repr in self.parameters.pytorch_representation.values.values():
            if pytorch_repr is not None:
                res.extend([w.projection for w in pytorch_repr.projection_wrappers])

        return res
