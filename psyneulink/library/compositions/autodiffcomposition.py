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
          COMMENT:
          - `AutodiffComposition_Optimizer`
          COMMENT
          - `AutodiffComposition_Targets`
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
be any level of such nestings.  However, all of the nested Compositions must be AutodiffCompositions. The learning_rate
for nested Compositions is inherited from the enclosing Composition unless it is set individually (see
`Composition_Learning_Rate` for a full discussion of how learning rates and precedence of assignment;  see
`Composition_Enable_Learning` for enabling and disabling learning in nested Compositions.

.. technical_note::
   Projections from `Nodes <Composition_Nodes>` in an immediately enclosing outer Composition to the `input_CIM
   <Composition.input_CIM>` of a nested Composition, and from its `output_CIM <Composition.output_CIM>` to Nodes
   in the outer Composition are subject to learning; however those within the nested Composition itself (i.e.,
   from its input_CIM to its INPUT Nodes and from its OUTPUT Nodes to its output_CIM) are *not* subject to learning,
   as they serve simply as conduits of information between the outer Composition and the nested one.

.. warning::
   Nested Compositions are supported for learning only in `PyTorch mode <AutodiffComposition_PyTorch>`, and will
   cause an error if the `learn <AutodiffComposition.learn>` method of an AutodiffComposition is executed in
   `Python mode <AutodiffComposition_Python>` or `LLVM mode <AutodiffComposition_LLVM>`.

.. _AutodiffComposition_Learning_Rates:

*Learning Rates*
~~~~~~~~~~~~~~~~

The **learning** argument of the constructor and/or the `learn <AutodiffComposition.learn>` method can be used to
specify both a `learning_rate <AutodiffComposition.learning_rate>` for the entire AutodiffComposition and/or
individual MappingProjections within it (see `Composition_Learning_rate` for details of specification). Learning_rates
specified for individual MappingProjections are passed to the corresponding parameters of the AutodiffComposition's
`pytorch_representation <AutodiffComposition.pytorch_representation>` when it is executed. Specifications made in the
constructor for the AutodiffComposition are used as the default learning_rates for all executions of the `learn
<AutodiffComposition.learn>`; specifications made in the call to the `learn() <AutodiffComposition.learn>` method
override any made in the constructor, but are used only for that execution. A warning is issued if a learning_rate is
specified for a Projection with a `learnable <MappingProjection.learnable>` attribute set to ``False``, and an error
is generated if the Projection is associated with a PyTorch Parameter that is not learnable.
See `Composition_Learning_rate` for additional information about specifying learning_rates, including how the
`learning_rate <MappingProjection.learning_rate>` is determined for Projections that are not expliclity specified.

COMMENT:
.. note::
   An outermost AutodiffComposition's learning rate is applied to any `nested AutodiffCompositions
   <AutodiffComposition_Nesting>`, whether this is specified in the call to its `learn
   <AutodiffComposition.learn>` method, its constructor, or its default value is being used.
COMMENT

.. hint::
   To disable learning for a particular `MappingProjection` in an AutodiffComposition, specify either the
   **learnable** parameter of its constructor or its learning_rate specification in the **learning_rate**
   argument of the AutodiffComposition's constructor to False; this applies to MappingProjections at any level of
   `nesting <AutodiffComposition_Nesting>`.

COMMENT:
.. _AutodiffComposition_Optimizer:

*Optimizer*
~~~~~~~~~~~

In addition to `learning_rate <Projection.learning_rate>`, other parameters can be customized by constructing
a `torch.nn.optimizer <https://pytorch.org/docs/main/optim.html>`_ and assigining it to the **optimizer** argument
of either the AutodiffComposition's constructor or `learn <AutodiffComposition.learn>` method.  This requires creating
and adding ``param_groups`` for the `torch.nn.Parameters
<https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html>`_ corresponding to the Projections to be
specified, which are listed in the AutodiffComposition's `torch_parameters <AutodiffComposition.torch_parameters>`
attribute.
COMMENT

.. _AutodiffComposition_Targets:

*AutodiffComposition Target Specification(s)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TEACHER_TARGET BREADCRUMB: EDIT
Mention student:teacher
Formats:
- LossMechanism
- tuple
- list of LearningMechanisms and/or tuples
- dict of student:teacher pairs
- default loss function for LossMechanisms is loss_spec
- automatically constructs MappingProjections from SAMPLE ("student") and TARGET ("teacher") Nodes
  - technical note::  values of MappingProjection(s) received from TARGET Nodes are detached
      (to prevent gradient propagation)
  - a SAMPLE can receive error signals from multiple LossMechanisms, directly or indirectly;
    gradients combined
- ??handling of multiple targets for a single student??
- comparable to use of ComparatorMechanisms for PNL learning in Composition
- if none is specified:
   - OUTPUT Nodes are used as students
   - TARGET Nodes constructed automatically to receive target inputs specified in **inputs** argument learn()
- must specify targets either in constructor or learn() method

 ----------
 OLD
 This can be specified in any of the following ways:
        - as a `LossMechanism` added to the AutodiffComposition (see `LossMechanism <LossMechanism_Class_Reference>`);
        - as a tuple or list of target values, in which case a `LossMechanism` is created automatically using the
            specified values as its `value <LossMechanism.value>` attribute; or
        - as a dict mapping `Nodes <Composition_Nodes>` in the AutodiffComposition to their target values, in which
            case a `LossMechanism` is created automatically using the specified values as its `value
            <LossMechanism.value>` attribute.
        If None (the default), no targets are used for training; in this case, the AutodiffComposition can only be
        executed in `test mode <Composition_Learning_Test_Mode>`.
------------

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
# 7/10/24 - BREADCRUMB:
.. _AutodiffComposition_PyTorch_LearningScale:
   ADD DESCRIPTION OF HOW LearningScale SPECIFICATIONS MAP TO EXECUTION OF pytorch_rep:
      OPTIMIZATION STEP:
      for AutodiffCompositions, this corresponds to a single call to `forward()` and `backward()`
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
       causes it to use PyTorch for training, specifying this in the `run <Composition.run>` method causes it to be
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
from typing import Any, Dict, Set, Tuple, Union

try:
    import torch
    from torch import nn
    import torch.optim as optim
    torch_available = True
except ImportError:
    torch_available = False
else:
    from psyneulink.library.compositions.pytorchshowgraph import PytorchShowGraph

from psyneulink._typing import Iterable, Mapping, Optional, Literal
from psyneulink.core.components.component import Component
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.mechanisms.processing.processingmechanism import (
    ProcessingMechanism, ProcessingMechanism_Base)
from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.core.components.mechanisms.modulatory.modulatorymechanism import ModulatoryMechanism_Base
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.compositions.composition import (Composition, CompositionError, LearningScale, NodeRole)
from psyneulink.core.compositions.report import (ReportOutput, ReportParams, ReportProgress, ReportSimulations,
                                                 ReportDevices, EXECUTE_REPORT, LEARN_REPORT, PROGRESS_REPORT)
from psyneulink.library.components.mechanisms.processing.objective.lossmechanism import LossMechanism
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import (
    AUTODIFF_COMPOSITION,
    DEFAULT,
    DEFAULT_LEARNING_RATE,
    ERROR,
    EXECUTION_MODE,
    LEARNING_RATE,
    Loss,
    LOSSES,
    MATRIX_WEIGHTS,
    NAME,
    NODE_VALUES,
    NODE_VARIABLES,
    PROJECTION,
    RESULTS,
    RETAIN_IN_PNL_OPTIONS,
    SAMPLE,
    SOFT_CLAMP,
    SYNCH_WITH_PNL_OPTIONS,
    TARGET,
    TARGETS,
    TRAINED_OUTPUTS,
    WARNING
)
from psyneulink.core.globals.utilities import (
    is_identity_matrix, is_matrix_keyword, is_numeric_scalar, convert_to_list, convert_to_np_array, deprecation_warning)
from psyneulink.core.scheduling.scheduler import Scheduler
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.core import llvm as pnlvm


logger = logging.getLogger(__name__)


__all__ = [
    'AutodiffComposition', 'OPTIMIZER_PARAMS', 'EXCLUDE_FROM_GRADIENT_CALC',
]

OPTIMIZER_PARAMS = 'optimizer_params'
EXCLUDE_FROM_GRADIENT_CALC = 'exclude_from_gradient_calc'

SynchRetainArg = Optional[Union[LearningScale, str]]


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
    AutodiffComposition(                           \
        optimizer_type='sgd',                      \
        loss_spec=Loss.MSE,                        \
        targets=None,                              \
        weight_decay=0,                            \
        enable_learning=True,                      \
        learning_rate=0.001,                       \
        synch_projection_matrices_with_torch=RUN,  \
        synch_node_variables_with_torch=None,      \
        synch_node_values_with_torch=RUN,          \
        synch_results_with_torch=RUN,              \
        retain_torch_trained_outputs=MINIBATCH,    \
        retain_torch_targets=MINIBATCH,            \
        retain_torch_losses=MINIBATCH,             \
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
        specifies the default loss function for training; see `Loss` for arguments;
        any specifications in **targets** override this default.

    targets : LossMechanism, tuple, list or dict  : default None
        specifies the target(s) used for training the model; see `AutodiffComposition_Targets` for additional details;

    weight_decay : float : default 0
        specifies the L2 penalty (which discourages large weights) used by the optimizer.

    enable_learning : bool: default True
        specifies whether the AutodiffComposition should enable learning when run in `learning mode
        <Composition.learn>` (see `Composition_Enable_Learning` for additional details).

    learning_rate : float, int, bool or dict : default 0.001
        specifies the learning rate(s) passed to the optimizer; overridden by any specified in the `learn
        <AutdodiffComposition.learn>` method of the AutodiffComposition; if a dict is used, and it does
        not contain an entry for *DEFAULT_LEARNING_RATE*, the default indicated above is used (see `learning_rate
        (see `AutodiffComposition_Learning_Rate` and `Composition_Learning_Rate` for additional details).

    synch_projection_matrices_with_torch : `LearningScale` : default RUN
        specifies the default for the AutodiffComposition for when to copy Pytorch parameters to PsyNeuLink
        `Projection matrices <MappingProjection.matrix>` (connection weights), which can be overridden by specifying
        the **synch_projection_matrices_with_torch** argument in the `learn <Composition.learn>` method
        (see `synch_projection_matrices_with_torch <AutodiffComposition.synch_projection_matrices_with_torch>`
        for additional details).

    synch_node_variables_with_torch : `LearningScale` : default None
        specifies the default for the AutodiffComposition for when to copy the current input to Pytorch nodes
        to the PsyNeuLink `variable <Mechanism_Base.value>` attribute of the corresponding PsyNeuLink `nodes
        <Composition_Node>`, which can be overridden by specifying the **synch_node_variables_with_torch** argument
        in the `learn <Composition.learn>` method (see `synch_node_variables_with_torch
        <AutodiffComposition.synch_node_variables_with_torch>` for additional details).

    synch_node_values_with_torch : `LearningScale` : default RUN
        specifies the default for the AutodiffComposition for when to copy the current output of Pytorch nodes to the
        PsyNeuLink `value <Mechanism_Base.value>` attribute of the corresponding PsyNeuLink `nodes <Composition_Node>`,
        which can be overridden by specifying the **synch_node_values_with_torch** argument in the `learn
        <Composition.learn>` method (see `synch_node_values_with_torch
        <AutodiffComposition.synch_node_values_with_torch>` for additional details).

    synch_results_with_torch : `LearningScale` : default RUN
        specifies the default for the AutodiffComposition for when to copy the outputs of the Pytorch model
        to the AutodiffComposition's `results <Composition.results>` attribute, which can be overridden by
        specifying the **synch_results_with_torch** argument in the `learn <Composition.learn>` method.
        Note that this differs from **retain_torch_trained_outputs**, which specifies the frequency at which
        the outputs of the PyTorch model are tracked, all of which are stored in the AutodiffComposition's
        `torch_trained_outputs <AutodiffComposition.torch_trained_outputs>` attribute at the end of the run
        (see `synch_results_with_torch <AutodiffComposition.synch_results_with_torch>` for
        additional details).

    retain_torch_trained_outputs : `LearningScale` : default MINIBATCH
        specifies the default for the AutodiffComposition for scale at which the outputs of the Pytorch
        model are tracked, all of which are stored in the AutodiffComposition's `torch_trained_outputs
        <AutodiffComposition.torch_trained_outputs>` attribute at the end of the run; this can be overridden
        by specifying the **retain_torch_trained_outputs** argument in the `learn <Composition.learn>` method.
        Note that this differs from **synch_results_with_torch**, which specifies the frequency with
        which values are called to the AutodiffComposition's `results` attribute (see `retain_torch_trained_outputs
        <AutodiffComposition.retain_torch_trained_outputs>` for additional details).

    retain_torch_targets : `LearningScale` : default MINIBATCH
        specifies the default for the AutodiffComposition for when to copy the targets used for training the
        Pytorch model to the AutodiffComposition's `torch_targets <Composition.torch_targets>` attribute, which can be
        overridden by specifying the **retain_torch_targets** argument in the `learn <Composition.learn>` method
        (see `retain_torch_targets <AutodiffComposition.retain_torch_targets>` for additional details).

    retain_torch_losses : `LearningScale` : default MINIBATCH
        specifies the default for the AutodiffComposition for the scale at which the losses of the Pytorch model
        are tracked, all of which are stored in the AutodiffComposition's `torch_losses <Composition.torch_losses>`
        attribute at the end of the run (see `retain_torch_losses <AutodiffComposition.retain_torch_losses>` for
        additional details).

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

    loss_spec : PyTorch loss function
        the loss function used for training. Depends on the **loss_spec** argument from initialization.

    targets : list of LossMechanisms
        each LossMechanism computes the loss for the output of the Node from which it recieves its *SAMPLE* input
        (the "student" Node) by comparing it to the output of the Node from which it receives its *TARGET* input
        (the "teacher" Node), using the specified loss function; see `AutodiffComposition_Targets` for additional
        details.

    learning_rate : float or bool
        determines the default learning_rate passed the `optimizer <PytorchCompositionWrappe.optimizer>`,
        that is applied to all `Projections <Projection>` in the AutodiffComposition that are `learnable
        <MappingProjection.learnable>`, and for which individual rates have not been specified (see
        `AutodiffComposition_Learning_Rates` for additional details).

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

    torch_parameters : List[Tuple[str, torch.nn.parameter]]
        list of PyTorch named_parameters() for `pytorch_representation <AutodiffComposition.pytorch_representation>`
        of AutodiffComposition.

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

    full_sequence_mode: bool : default False
        Whether to run the underlying Composition in full sequence mode or not. In full sequence mode, each element of
    an input sequence for a trial is processed in a separate time step. This is needed only if there are sequential
    dependencies between the mechanisms of the compositions. Note, if the composition contains GRU compositions wrappers
    full sequence mode is not needed (and should be avoided to improve efficiency) because the composition wrapper
    itself handles the  sequential dependencies between the mechanisms of the GRU composition.

    """

    componentCategory = AUTODIFF_COMPOSITION
    if torch_available:
        from psyneulink.library.compositions.pytorchwrappers import PytorchCompositionWrapper, PytorchMechanismWrapper
        pytorch_composition_wrapper_type = PytorchCompositionWrapper
        pytorch_mechanism_wrapper_type = PytorchMechanismWrapper

    class Parameters(Composition.Parameters):
        pytorch_representation = None
        # optimizer = None
        loss_spec = Parameter(Loss.MSE, stateful=False, modulable=False)
        targets = Parameter(None, fallback_value=None, structural=True, stateful=False, modulable=False,
                            dependencies={'loss_spec'}
                            )
        synch_projection_matrices_with_torch = Parameter(LearningScale.RUN, fallback_value=DEFAULT)
        synch_node_variables_with_torch = Parameter(None, fallback_value=DEFAULT)
        synch_node_values_with_torch = Parameter(LearningScale.RUN, fallback_value=DEFAULT)
        synch_results_with_torch = Parameter(LearningScale.RUN, fallback_value=DEFAULT)
        retain_torch_trained_outputs = Parameter(LearningScale.MINIBATCH, fallback_value=DEFAULT)
        retain_torch_targets = Parameter(LearningScale.MINIBATCH, fallback_value=DEFAULT)
        retain_torch_losses = Parameter(LearningScale.MINIBATCH, fallback_value=DEFAULT)
        torch_trained_outputs = Parameter([], getter=_get_torch_trained_outputs)
        torch_targets = Parameter([], getter=_get_torch_targets)
        torch_losses = Parameter([], getter=_get_torch_losses)
        trial_losses = Parameter([]) # FIX <- related to early_stopper, but not getting assigned anywhere
        device = None

        def _validate_loss_spec(self, spec):
            if spec and not isinstance(spec, (Loss, torch.nn.modules.loss._Loss)):
                return f"must be a member of the Loss enum or a PyTorch loss function."

        def _parse_targets(self, specs)->list:
            """Parse targets argument to standardize into list of LossMechanisms or (sample, target) tuples
            Convert Mechanism specs for sample and/or target in a tuple to the corresponding primary port.
            """
            if specs:
                if isinstance(specs, (LossMechanism, tuple, dict)):
                    specs = convert_to_list(specs)
                for i, spec_tuple in enumerate(specs.copy()):
                    sample, target = spec_tuple
                    sample = sample.output_port if isinstance(sample, Mechanism) else sample
                    target = target.output_port if isinstance(target, Mechanism) else target
                    specs[i] = (sample, target)
            return specs

        def _validate_targets(self, spec):
            if spec is None:
                return None
            if isinstance(spec, list):
                for item in spec:
                    if not isinstance(item, (LossMechanism, tuple)):
                        return (f"must be a list of LossMechanisms or a collection of student, teacher node pairs .")
                    assert isinstance(item[0], OutputPort), \
                        "PROGRAM ERROR: 1st item of tuple specifciont for targets arg should be OutputPort by now."
                    if not isinstance(item[0].owner, ProcessingMechanism_Base):
                        return (f"the first item of a tuple or key of a dict entry must be ProcessingMechanism.")
                    assert isinstance(item[1], OutputPort) or item[1] == TARGET, \
                        ("PROGRAM ERROR: 2nd item of tuple specifiction for targets arg should be OutputPort by now.")
                    if not (item[1] == TARGET or isinstance(item[1].owner, ProcessingMechanism_Base)):
                        return (f"the second item of a tuple or value of a dict entry must be "
                                f"a ProcessingMechanism or the keyword 'TARGET'.")
                    return None
            else:
                return (f"must be a LossMechanism, list of them, "
                        f"or a tuple or dict containing pairs of student, teacher nodes.")

        def _parse_LearningScale_param(self, value):
            try:
                return LearningScale(value)
            except ValueError:
                return value

        # TODO: consider implementing a type/enum parser as attr on Parameter
        _parse_synch_projection_matrices_with_torch = _parse_LearningScale_param
        _parse_synch_node_variables_with_torch = _parse_LearningScale_param
        _parse_synch_node_values_with_torch = _parse_LearningScale_param
        _parse_synch_results_with_torch = _parse_LearningScale_param
        _parse_retain_torch_trained_outputs = _parse_LearningScale_param
        _parse_retain_torch_targets = _parse_LearningScale_param
        _parse_retain_torch_losses = _parse_LearningScale_param

        def _validate_LearningScale_param(
            self, value: Any, invalid_members: Optional[Set] = None
        ):
            # NOTE: this occurs after parse, so it will already be a
            # LearningScale if possible
            if value is None:
                return None

            try:
                is_learningscale = value in LearningScale
            except TypeError:
                is_learningscale = False

            try:
                is_invalid_member = value in invalid_members
            except TypeError:
                is_invalid_member = False

            if not is_learningscale:
                return (
                    f"must be a {LearningScale.__name__} or corresponding"
                    f" string: {', '.join(map(str, LearningScale))}"
                )

            if is_invalid_member:
                valid = set(LearningScale).difference(invalid_members)
                return f"can't be used; use another value of: {', '.join(valid)}"

            return None

        def _validate_synch_projection_matrices_with_torch(self, spec):
            return self._validate_LearningScale_param(spec)

        def _validate_synch_node_variables_with_torch(self, spec):
            return self._validate_LearningScale_param(spec)

        def _validate_synch_node_values_with_torch(self, spec):
            return self._validate_LearningScale_param(spec)

        def _validate_synch_results_with_torch(self, spec):
            return self._validate_LearningScale_param(spec, {LearningScale.OPTIMIZATION_STEP})

        def _validate_retain_torch_trained_outputs(self, spec):
            return self._validate_LearningScale_param(spec)

        def _validate_retain_torch_targets(self, spec):
            return self._validate_LearningScale_param(spec)

        def _validate_retain_torch_losses(self, spec):
            return self._validate_LearningScale_param(spec)

    # TODO (CW 9/28/18): add compositions to registry so default arg for name is no longer needed
    @check_user_specified
    def __init__(self,
                 pathways=None,
                 optimizer_type: str = 'sgd',
                 loss_spec: Loss = Loss.MSE, # default is Loss.MSE set in Parameters
                 targets: Union[LossMechanism, tuple, list, dict] = None,
                 weight_decay: float = 0.0,
                 learning_rate: Optional[Union[float,int,bool,dict,]]=.001,
                 enable_learning: bool = True,
                 force_no_retain_graph: bool = False,
                 refresh_losses: bool = False,
                 synch_projection_matrices_with_torch: SynchRetainArg = LearningScale.RUN,
                 synch_node_variables_with_torch: SynchRetainArg = None,
                 synch_node_values_with_torch: SynchRetainArg = LearningScale.RUN,
                 synch_results_with_torch: SynchRetainArg = LearningScale.RUN,
                 retain_torch_trained_outputs: SynchRetainArg = LearningScale.MINIBATCH,
                 retain_torch_targets: SynchRetainArg = LearningScale.MINIBATCH,
                 retain_torch_losses: SynchRetainArg = LearningScale.MINIBATCH,
                 device=None,
                 disable_cuda=True,
                 cuda_index=None,
                 full_sequence_mode: bool = False,
                 name="autodiff_composition",
                 **kwargs):

        show_graph_attributes = kwargs.pop('show_graph_attributes', {})

        # Deal with deprecated arg
        if OPTIMIZER_PARAMS in kwargs:
            opt_params_arg = deprecation_warning(self, kwargs,
                                                 deprecated_args={OPTIMIZER_PARAMS:LEARNING_RATE},
                                                 additional_msg=" Other torch.nn.optimizer parameters are not "
                                                                "currently supported, but will be in a future version.")
            if learning_rate is not None:
                opt_params_arg[DEFAULT_LEARNING_RATE] = learning_rate
            learning_rate = opt_params_arg.pop(LEARNING_RATE)


        super(AutodiffComposition, self).__init__(
            name = name,
            pathways=pathways,
            optimizer_type = optimizer_type,
            loss_spec = loss_spec,
            targets = targets,
            weight_decay = weight_decay,
            enable_learning = enable_learning,
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
        self.loss_mechs_map = {}  # {LossMechanism : (sample, target)} tuple of sender Ports
        self.target_ports_for_samples = {} # {sample OutputPort : TARGET Node OutputPort}
        self._trained_comp_nodes_to_pytorch_nodes_map = None # Set by subclasses that replace trained OUTPUT Nodes
        self._input_comp_nodes_to_pytorch_nodes_map = None # Set by subclasses that replace INPUT Nodes
        self._pytorch_projections = []
        self.optimizer_type = optimizer_type
        self._optimizer_constructor_params = self.parameters.learning_rates_dict.get(None)
        self._runtime_learning_rate = None
        self.force_no_retain_graph = force_no_retain_graph
        self.refresh_losses = refresh_losses
        self.weight_decay = weight_decay
        self.loss_function = None
        self.last_saved_weights = None
        self.last_loaded_weights = None
        self.full_sequence_mode = full_sequence_mode

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
            self.torch_dtype = self.pytorch_composition_wrapper_type.torch_dtype
        else:
            self.device = device
            self.torch_dtype = None

        # Set to True after first warning about failure to specify execution mode so warning is issued only once
        self.execution_mode_warned_about_default = False
        # torch params added when warned in copy_projection_matrix_to_torch_param() to avoid repeats for same param
        self.require_grad_warning = []

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
    def infer_backpropagation_learning_pathways(self, execution_mode, context=None, base_context=None)->list:
        """Create backpropagation learning pathways for every INPUT Node --> OUTPUT Node pathway
        Pathways are constructed in _get_pytorch_backprop_pathways()
            Flattens nested compositions:
              - only includes the Projections in outer Composition to/from the CIMs of the nested Composition
                (i.e., to input_CIMs and from output_CIMs) -- the ones that should be learned;
              - excludes Projections from/to CIMs in the nested Composition
                (from input_CIMs and to output_CIMs), as those should remain identity Projections;
              see `PytorchCompositionWrapper` for table of how Projections are handled and further details.
        For Python mode:
          - calls add_backpropagation_learning_pathway() for each identified pathway
            which also creates TARGET Nodes for TERMINAL Nodes in each pathway
        For PyTorch mode:
          - if **targets** are specified in the AutodiffComposition constructor,
             LossMechanisms and MappingProjections are constructed for them;
          - otherwise, TERMINAL Nodes of each pathway are used to construct LossMechanisms and TARGET Nodes
            with associated MappingProjections) to allow targets to be specified in inputs argument of learn().
          - the above allow:
            - trial-by-trial losses to be kept aligned with inputs in batch / minibatch construction
            - losses to be tracked for logging (as mechs of a Composition)
        TEACHER_TARGET BREADCRUMB:
        Return a list of any TARGET nodes that need to be referenced in inputs argument of learn()
        """
        context = context or Context()
        base_context = base_context or Context()

        # Construct a pathway(s) for each INPUT Node (including BIAS Nodes), except the TARGET Node)
        pathways = self._get_pytorch_backprop_pathways(context)

        if execution_mode is pnlvm.ExecutionMode.PyTorch:
            # Construct LossMechanisms, and TARGET Nodes if needed, for inclusion in pathway construction below
            self._instantiate_loss_components(pathways, context, base_context)

        else:
        # if execution_mode is not pnlvm.ExecutionMode.PyTorch:
            # For non-Pytorch modes, construct and add PNL backpropagation learning pathways for each INPUT Node
            #    that will construct learning components, including TARGET Nodes for all TERMINAL Nodes
            for pathway in pathways:
                self.add_backpropagation_learning_pathway(pathway=pathway,
                                                          loss_spec=self.loss_spec)

        self._analyze_graph()
        return self.learning_components

    @handle_external_context()
    def _get_pytorch_backprop_pathways(self, context)->list:
        """Get backpropagation pathways for all INPUT Nodes of AutodiffComposition
        Return a list of all pathways"""
        self._analyze_graph()
        return [pathway
                    for node in (self.get_nodes_by_role(NodeRole.INPUT) + self.get_nodes_by_role(NodeRole.BIAS))
                    if node not in self.get_nodes_by_role(NodeRole.TARGET)
                    for pathway in self._get_pytorch_backprop_pathway(node, context)]

    def _get_pytorch_backprop_pathway(self, input_node, context)->list:
        """Breadth-first search from input_node to find all input -> <any OUTPUT Node> pathways
        Uses queue(node, composition) to traverse all nodes in the graph
        IMPLEMENTATION NOTE:  flattens nested Compositions, removing any CIMs in the nested Compositions
        Return a list of all pathways from input_node -> any OUTPUT Node
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
            # Only allow odd number of components since there must be one fewer Projections than Mechanisms
            assert len(pathway) % 2, \
                f"PROGRAM ERROR: There are one too many Projections in pathway: {' ,'.join(pathway)}"
            return pathway

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
            # # MODIFIED TEACHER_TARGET OLD:
            # if current_comp == self and node in current_comp.get_nodes_by_role(NodeRole.OUTPUT):
            # MODIFIED TEACHER_TARGET NEW ADD LossMech.SAMPLE :
            if current_comp == self and (node in current_comp.get_nodes_by_role(NodeRole.OUTPUT)
                                         or not node.efferents):
            # MODIFIED TEACHER_TARGET END
                pathways.append(create_pathway(current_comp, node))
                continue

            # # Get all efferent Projections of node,
            # #   including direct projections out of a nested Composition implemented in PyTorchCompositionWrapper
            efferent_projs = [(p, p.receiver.owner) for p in node.efferents if p in current_comp.projections]
            if not efferent_projs:
                # # MODIFIED TEACHER_TARGET OLD:
                # efferent_projs = [(p, p.receiver.owner) for p in node.efferents
                #                   if p in current_comp._pytorch_projections]
                # MODIFIED TEACHER_TARGET NEW:
                efferent_projs = [(p, p.receiver.owner) for p in node.efferents
                                  if (p in current_comp._pytorch_projections
                                      or isinstance(p.receiver.owner, LossMechanism))]
                # MODIFIED TEACHER_TARGET END

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
                                rcvr_comp._add_dependency(node, efferent_proj, rcvr, dependency_dict, queue, rcvr_comp)
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
                        current_comp._pytorch_projections.append(efferent_proj)
                        continue
                    elif rcvr in self.nodes:
                        # rcvr is in outer Composition (presumably a direct Pytorch Projection out of nested comp)
                        self._add_dependency(node, efferent_proj, rcvr, dependency_dict, queue, self)
                        continue
                    else:
                        assert False, \
                            (f"PROGRAM ERROR:  Unrecognized receiver ('{rcvr.name}') of Projection from '{node.name}'.")

        return pathways

    def _mech_in_learnable_pathway(self, mech: ProcessingMechanism_Base) -> bool:
        """Return True if `mech` receives a Project from any pathway that has at least one learnable Projection"""
        for afferent in mech.path_afferents:
            if afferent.learnable:
                return True
            check_afferent_pathway = self._mech_in_learnable_pathway(afferent.sender.owner)
            if check_afferent_pathway:
                return True
        return False

    def _sample_is_in_learnable_pathway(self, sample, target=None, loss_mech=None,
                                        constructed_target_mechs=None,
                                        action:Optional[Union[Literal[ERROR, WARNING]]]=None)->bool:
        """Take specified action if sample has no afferent pathways with any learnable Projections.
        - target argument is used to determine for error_message;
        - if no action is specified, return True or False
        """
        if self._mech_in_learnable_pathway(sample):
            return True
        # Construct relevant error/warning message
        if target:
            # Target was specified in *targets* arg of constructor
            if isinstance(target, LossMechanism):
                target_msg = f"LossMechanism ('{loss_mech.name}')"
            elif target in constructed_target_mechs:
                target_msg = "TARGET input"
            else:
                target_msg = f"TARGET node ('{target.name}')"
            error_msg = (f"A {target_msg} has been assigned to a node ('{sample.name}') for "
                         f"learning that has no afferent pathways with any learnable Projections.")
        else:
            # TARGET Nodes being constructed for all OUTPUT Nodes, so all must be in learnable pathways
            if sample in self.get_nested_nodes_by_roles_at_any_level(self, NodeRole.SINGLETON):
                # Singletons are caught here because they are identified as OUTPUT Nodes,
                #   but are not specified in targets dict of learn() method.
                # Technically, they are not erroneous, so allow construction;
                #   warning about non-learnability is handled in _instantiate_optimizer()
                return False
            # TARGET Nodes being constructed for all OUTPUT Nodes, so all must be in learnable pathways
            error_msg = (f"A target value is specified for '{sample.name}' in the learn() method of '{self.name}', "
                         f"but that Node has no afferent pathways with any learnable Projections.")

        # Take specified action
        if action is ERROR:
            raise AutodiffCompositionError(error_msg)
        elif action is WARNING:
            warnings.warn(error_msg)
        return False

    def _instantiate_loss_components(self, pathways, context, base_context):
        """Instantiate LossMechanisms, and TARGET Nodes if needed, for AutodiffComposition

        TEACHER_TARGET BREADCRUMB CLEANUP:
          ?? POPULATE self.learning_components WITH ANY INSTANTATED TARGET Nodes
          FOR BACKWARD COMPATIBILITY AND COMPATIBILITY WITH OTHER (E.G., PNL) LEARNING MODES
          IF NOT, WHERE IS IT POPULATED?

          DOCUMENT THAT AutoDiff PUTS LOSS AS WELL AS TARGET NODES IN learning_components
          PUT TARGET Nodes in self.target_nodes ATTRIBUTE
          CHANGE TESTS/SCRIPTS THAT USE learning_components TO IDENTIFY TARGET Nodes TO USE self.target_nodes

          ALSO, DEAL WITH NESTED COMPS?  OR ONLY CALL THIS AFTER FLATTENING?

        If **targets** arg of AutodiffComposition constructor:
        - IS specified:
          - construct LossMechanisms:
            {<student Node> : <teacher Node>} ->
                student Node -> LossMechanism.input_port[SAMPLE]
                reacher Node -> LossMechanism.input_port[TARGET]
        - is NOT specified:
          - use TERMINAL nodes of pathways to construct TARGET Nodes and LossMechanisms:
              learn(targets = {<OUTPUT Node> : <value>}) -> TARGET Node (in _map_external_target_values_to_target_nodes)
              TARGET Node -> LossMechanism.input_port[SAMPLE]
              OUTPUT Node  -> LossMechanism.input_port[TARGET]
          - this allows:
            - external targets to be specified in the same way as for other execution_modes
            - trial-by-trial losses to be kept aligned with inputs in batch / minibatch construction
            - losses to be tracked for logging (as mechs of a Composition)

        Construct self.loss_mechs_map: {<LossMechanism: (student Node, teacher Node)}
        Add loss_mechs and any constructed TARGET Nodes to self.learning_components
        """
        context = Context(source=ContextFlags.METHOD, execution_id=context.execution_id)
        constructed_target_mechs = []

        # Determine whether targets were specified by user or OUTPUT Nodes should be used to construct TARGET Nodes
        if self.targets:
            # targets specified by user in **targets** argument of AutodiffComposition constructor,
            #   either as LossMechanism, (sample:target) tuple, or list containing either
            # Get TARGET specs; can be Node or TARGET keyword requiring construction of TARGET Node (below)
            loss_mech_specs = []
            target_mechs = []
            for loss_mech_spec in self.targets:
                if isinstance(loss_mech_spec, LossMechanism):
                    sample_port = loss_mech_spec.sample
                    sample_mech = sample_port.owner
                    target_port = loss_mech_spec.target
                    target_mech = target_port.owner
                    # If sample specified for LossMechanism is not in a pathway with at least one learnable Projection
                    #   then raise error, as executing its LossFunction in pytorch will cause a crash
                    self._sample_is_in_learnable_pathway(sample=sample_mech, target=target_mech, loss_mech=loss_mech,
                                                         constructed_target_mechs=constructed_target_mechs,
                                                         action=ERROR)
                elif isinstance(loss_mech_spec, tuple):
                    sample_port, target_spec = loss_mech_spec
                    sample_mech = sample_port.owner
                    target_mech = target_spec.owner if isinstance(target_spec, OutputPort) else target_spec
                    # If specified sample Mechanism is not in a pathway with at least one learnable Projection
                    #   then raise error, as constructing a LossMechanism with aLossFunction that tries to compute
                    #   loss in pytorch will cause a crash
                    self._sample_is_in_learnable_pathway(sample=sample_mech, target=target_mech, loss_mech=None,
                                                         constructed_target_mechs=None,
                                                         action=ERROR)
                    # Determine whether target is internal node or TARGET keyword
                    if isinstance(target_spec, OutputPort):
                        # target is internal Node
                        self.target_ports_for_samples.update({sample_port: target_spec})
                    elif target_spec == TARGET:
                        # target is TARGET keyword, so construct TARGET Node
                        if sample_port in self.target_ports_for_samples:
                            # TARGET Node has already been constructed for specified sample Port
                            continue
                        sample_name = (sample_port.full_name if len(sample_port.owner.output_ports)>1
                                       else sample_port.owner.name)
                        target_mech = ProcessingMechanism(default_variable = np.array([np.zeros_like(value) for value
                                                                                       in sample_mech.value],
                                                                                      dtype=object),
                                                          name= 'TARGET for ' + sample_name)
                        target_mech._initialize_from_context(context, base_context, override=False)
                        target_port = target_mech.output_port
                        self.target_ports_for_samples.update({sample_port: target_port})
                        self.add_node(target_mech, required_roles=[NodeRole.TARGET, NodeRole.INPUT], context=context)
                        constructed_target_mechs.append(target_mech)
                    else:
                        assert False, (f"PROGRAM_ERROR: unrecognized value of target specification "
                                       f"({loss_mech_spec[1]} for '{self.name}'.")
                else:
                    assert False, (f"PROGRAM_ERROR: unrecognized specification for self.targets "
                                   f"({loss_mech_spec} for '{self.name}'.")

                target_mechs.append(target_mech)
                loss_mech_specs.append((loss_mech_spec[0], target_spec))

        else:
            # No targets specified by user, so construct TARGET Node for all OUTPUT Nodes of the AutodiffComposition
            # IMPLEMENTATION NOTE:
            #    only add target nodes if *not* already present in self.target_ports_for_samples.values()
            #    (to avoid duplication in multiple calls, including from command line;
            #     see test_xor_training_identicalness_standard_composition_vs_PyTorch_and_LLVM for example)
            pathway_terminal_nodes = [mech for mech in [pathway[-1] for pathway in pathways]]
            identified_output_nodes = self._identify_output_nodes(context)
            output_ports_for_learning = []
            for node in [n for n in identified_output_nodes if n in pathway_terminal_nodes]:
                output_ports_for_learning.extend(node.output_ports)
            target_mechs = self.get_nodes_by_role(NodeRole.TARGET)
            for output_port_for_learning in output_ports_for_learning:

                if not self._sample_is_in_learnable_pathway(sample=output_port_for_learning.owner, target=None,
                                                            loss_mech=None,
                                                            constructed_target_mechs=constructed_target_mechs,
                                                            action=ERROR):
                    # If no error is generated in sample_is_in_learnable_pathway(), sample is a singeton;
                    #   warning about non-learnability is handled in _instantiate_optimizer()
                    continue
                # Check for existing TARGET Nodes
                existing_output_ports_for_learnings = [sample for sample, target in  self.loss_mechs_map.values()]
               # Get or construct TARGET Node if none exists for OUTPUT Node
                if output_port_for_learning not in existing_output_ports_for_learnings:
                    # Check that TARGET Node doesn't already exist for OUTPUT Node
                    #    (may have been created for PNL learning in call to add_backpropagation_learning_pathway)
                    existing_comparators = [mech for mech in self.nodes if
                                            isinstance(mech, ComparatorMechanism) and
                                            NodeRole.LEARNING_OBJECTIVE in self.get_roles_by_node(mech)]
                    comparators = [mech for mech in existing_comparators
                                   if mech.input_ports['SAMPLE'].path_afferents[0].sender is output_port_for_learning]
                    assert len(comparators) <= 1, (f"PROGRAM ERROR: multiple ComparatorMechanisms found "
                                                   f"for '{output_port_for_learning.full_name}' in {self.name}'.")
                    if comparators:
                        target_mech = comparators[0].input_ports['TARGET'].path_afferents[0].sender.owner
                        # Autodiff now owns this TARGET Node, so dissociate from learning_components used for Python
                        self.exclude_node_roles(target_mech, [NodeRole.LEARNING], context)
                    else:
                        sample = output_port_for_learning
                        sample_name = sample.full_name if len(sample.owner.output_ports)>1 else sample.owner.name
                        target_mech = ProcessingMechanism(default_variable = np.array([np.zeros_like(value)
                                                                                       for value in output_port_for_learning.value],
                                                                                      dtype=object),
                                                          name= 'TARGET for ' + sample_name)
                        target_mech._initialize_from_context(context, base_context, override=False)
                        # TEACHER_TARGET BREADCRUMB: THIS DOES NOT SEEM TO BE USED... DELETE?
                        constructed_target_mechs.append(target_mech)
                    target_mechs.append(target_mech)
            loss_mech_specs = list(zip(output_ports_for_learning, [target.output_port for target in target_mechs]))
            self.target_ports_for_samples.update({k:v for k,v in zip(output_ports_for_learning,
                                                                     [t.output_port for t in target_mechs])})
            self.add_nodes(target_mechs, required_roles=[NodeRole.TARGET, NodeRole.INPUT], context=context)

        # Validate LossMechanism specs
        if not loss_mech_specs:
            raise AutodiffCompositionError(f"Learning cannot be executed for '{self.name}' "
                                           f"since it does not have any learnable Projections.")
        for loss_mech_spec in list(loss_mech_specs):
            # Assume that self.targets is a list of LossMechanisms and/or tuples specifying sample:target pairs
            assert isinstance(loss_mech_spec, (LossMechanism, tuple)), \
                (f"PROGRAM ERROR: item in self.targets is neither LossMechanism nor 2-item tuple: {loss_mech_spec};"
                 f"should have been caught in targets Parameter validation.")
            if isinstance(loss_mech_spec, tuple):
                assert (len(loss_mech_spec) == 2
                        and all((isinstance(item, OutputPort) or item==TARGET) for item in loss_mech_spec)), \
                    (f"PROGRAM ERROR: tuple in self.targets either doesn't have two items "
                     f"or one is not a Mechanisms: {loss_mech_spec}; "
                     f"should have been caught in targets Parameter validation.")
            else:
                assert False, (f"PROGRAM ERROR: unrecognized item in self.targets: {item}")

        # Construct and/or add LossMechanisms (and their MappingProjections)
        for i, loss_mech_spec in enumerate(loss_mech_specs):
            if isinstance(loss_mech_spec, LossMechanism):
                sample = loss_mech_spec.sample
                target = loss_mech_spec.target
                loss_mech = loss_mech_spec
            elif isinstance(loss_mech_spec, tuple):
                sample, target = loss_mech_spec
                # Get loss_mech for sample if it already has one
                loss_mechs_for_sample = [l for l, sample_and_target in self.loss_mechs_map.items()
                                          if sample in sample_and_target]
                if loss_mechs_for_sample:
                    if len(loss_mechs_for_sample) > 1:
                        errant_target_names = [f"'{mech.target.full_name}'" for mech in loss_mechs_for_sample]
                        raise AutodiffCompositionError(
                            f"'{sample.full_name}' is associated with more than one TARGET in '{self.name}: "
                            f"{' ,'.join(errant_target_names)}.")
                    else:
                        loss_mech = loss_mechs_for_sample[0]
                        continue
                else:
                # if not any(sample in sample_and_target for sample_and_target in self.loss_mechs_map.values()):
                # Construct LossMechanism
                    # If there is no loss_mech for the current sample, instantiate one
                    # IMPLEMENTATION NOTE:
                    #        Don't allow multiple LossMechanisms to train the same SAMPLE Node
                    #        But it IS OK to have multiple LossMechanisms use the same TARGET Node
                    #        (i.e., to train multiple SAMPLES)
                    name = sample.full_name if len(sample.owner.output_ports)>1 else sample.owner.name
                    # TEACH_TARGET BREADCRUMB: HACK TO SEE IF IT WORKS:
                    # sample1 = self.nodes[0].nodes['OUTPUT'].output_port
                    loss_mech = LossMechanism(name=f"LOSS for {name}",
                                              sample=sample,
                                              target=self.target_ports_for_samples[sample],
                                              function=None,
                                              loss=self.loss_spec)
                    loss_mech._initialize_from_context(context, base_context, override=False)
                    for proj in loss_mech.path_afferents:
                        proj.learnable= False
            else:
                assert False, f"PROGRAM ERROR: loss_mech_spec should have been a LossMechanism or tuple by now."

            for proj in loss_mech.path_afferents:
                # TEACHER_TARGET BREADCRUMB: REVISE BELOW TO ENFORCE THESE ON CONSTRUCTION in LearningMechanism
                # IMPLEMENTATION NOTE: This is checked here because the Projections are to the LossMechanism
                #                      are constructed by reference to its afferents (sample and target)

                assert is_identity_matrix(proj.parameters.matrix.get()), \
                    (f"PROGRAM ERROR: Matrix of projection to LossMechanism "
                     f"('{proj.name}') is not an identity matrix. ")
                assert proj.learnable is False, (f"PROGRAM ERROR: The 'learnable' attribute of a projection to a "
                                                 f"LossMechanism ('{proj.name}') is not False")

            self.loss_mechs_map[loss_mech] = (sample, target)

        loss_mechs = list(self.loss_mechs_map.keys())

        # Add LossMechanisms and any TARGET Nodes to AutodiffComposition, with required NodeRoles
        self.add_nodes(loss_mechs, required_roles=[#NodeRole.LOSS,
                                                   NodeRole.LEARNING_OBJECTIVE], context=context)

        # TEACHER_TARGET BREADCRUMB:
        #                  THIS DOESN'T WORK SINCE THERE IS A PROPERTY ON COMPOSITION THAT OVERRIDES IT
        #                  NEED TO DECIDE WHAT SHOULD BE ASSIGNED TO THAT AND ADJUST TESTS ACCORDINGLY.
        # self.learning_components.append(loss_mechs + target_mechs)

        # Exclude LossMechanisms and TARGET Nodes from OUTPUT role and suppress warnings about role assignments
        for mech in loss_mechs + target_mechs:
            self.exclude_node_roles(mech, NodeRole.OUTPUT, context)
            for output_port in mech.output_ports:
                output_port.parameters.require_projection_in_composition.set(False, override=True)

    def _add_dependency(self,
                        sender:ProcessingMechanism_Base,
                        projection:MappingProjection,
                        receiver:ProcessingMechanism_Base,
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

    # BREADCRUMB: move some of what's done in the methods below to a "_validate_params" type of method
    @handle_external_context()
    def _build_pytorch_representation(self,
                                      learning_rate=None,
                                      optimizer_params=None,
                                      context=None,
                                      new=None,
                                      base_context=Context(execution_id=None)):
        """Build a Pytorch representation of the AutodiffComposition
        Construct PytorchCompositionWrapper that is used for learning in PyTorch, which is assigned to
        self.pytorch_representation.

        A new pytorch_representation is constructed if:
            self.pytorch_representation == None
            **new** is specified as True
        If _build_pytorch_representation() is called with **new**==None and a pytorch_representation already exists,
            a warning issued and the call is ignored.

        By default (learning_rate=None), the learning_rates specified in the **learning_rate**
        argument of the constructor for the Composition (and stored in self._learning_rates_dict) are
        used to construct the pytorch_representation. However:
        - if **learning_rate** is specified (in a call from the COMMANDLINE),
           that can be used to override the default values, as described under the learning_rate argument below;
        - if **optimizer_params** is specified (in a call from learn(), that is used to specify the learning_rates;
        - if **optimizer_params** is specified in a call from COMMAND_LINE, an error is returned.

        Arguments
        ---------

        new : bool or None : default None
            specifies creation of a new pytorch_representation, using self._optimizer_constructor_params
            as the base values, and updated with any specified in the **learning_rates** arg.  If the method is called
            from the command_line more than once without **new** specified as `True`, warns and ignores.

        learning_rate : float, int, dict : default None
            if None, then the values in self.learning_rates_dict (and stored in self._optimizer_constructor_params)
            are used to assign learning_rates to all Projections in the Composition (and any nested within it)
            (see `Composition_Learning_Rate` for details of specification); if a numeric values is specified,
            that is used as the default learning_rate for the pytorch_representation (replacing
            composition.learning_rate); if a dict is specified, entries are moved to optmizer_params and replace
            values for the specified Projections as well as the Composition's learning_rate (if DEFAULT_LEARNING_RATE
            is specified in the dict).

            .. note::
               Projection-specific learning_rates specified in a dict assigned to **learning_rate** here, like
               any specified in the constructor for the Composition, are stored in the corresponding Projections'
               `learning_rate <MappingProjection.learning_rate>` Parameter under the context <self.name>.DEFAULT_SUFFIX.
        """
        optimizer_params = optimizer_params or {}
        if self.scheduler is None:
            self.scheduler = Scheduler(graph=self.graph_processing)

        # Construct a new pytorch_representation if none exists or new is specified

        # MODIFIED TEACHER_TARGET NEW:
        # BREADCRUMB: THIS IS A HACK TO INSURE THAT _instantiate_loss_components() IS CALLED
        #             BEFORE THE pytorch_representation IS CONSTRUCTED;
        #             NOT SURE IF THAT IS OK IN GENERAL
        from psyneulink.core.llvm import ExecutionMode
        self.infer_backpropagation_learning_pathways(execution_mode=ExecutionMode.PyTorch,
                                                     context=context,
                                                     base_context=base_context)
        # MODIFIED TEACHER_TARGET END

        if self.parameters.pytorch_representation._get(context=context, fallback_value=None) is None or new:
            # Instantiate pytorch_representation
            context.composition = self
            self.pytorch_composition_wrapper_type(composition=self,
                                                  device=self.device,
                                                  context=context,
                                                  base_context=base_context)
        elif context.flags & ContextFlags.COMMAND_LINE:
            warnings.warn(f"The '_build_pytorch_representation() method for '{self.name}' has already been called "
                          f"directly from the command line; this and any additional calls will be ignored. "
                          f"Make any desired modifications to parameters (e.g., learning_rates) either in the "
                          f"constructor for the AutodiffComposition, or its learn() method.")

        # Get pytorch_representation (assigned in constructor for PytorchCompositionWrapper)
        pytorch_rep = self.parameters.pytorch_representation._get(context)

        # BREADCRUMB: MOVE THIS TO PytorchCompositionWrapper __init__(), since it belongs to that
        # Set up optimizer
        old_opt = pytorch_rep.optimizer
        # Get default learning rate (used for all Parameters for which specific learning_rates are not specified),
        #    giving precedence to learning_rate specified in call to learn() (stored in self._runtime_learning_rate)
        #    over learning_rate specified in constructor (passed in above as learning_rate)
        default_learning_rate = \
            (self._runtime_learning_rate if self._runtime_learning_rate is not None else learning_rate)
        if isinstance(learning_rate, dict):
            if optimizer_params:
                # if learning_rate is a dict, optimizer_params should not have been passed in call
                assert context.flags & ContextFlags.COMMAND_LINE, \
                    ("PROGRAM ERROR: 'optmizer_params' assigned when learning_rate assigned as a dict "
                     "in internal call to _build_pytorch_representation() for '{self.name}'.")
                assert False, \
                    ("PROGRAM ERROR:  Assignment of 'optimizer_params' in a direct call to "
                     "_build_pytorch_representation() from the command line is not currently supported.")
            lr_dict = default_learning_rate
            default_learning_rate = lr_dict.pop(DEFAULT_LEARNING_RATE, self.parameters.learning_rate.get(context))
            optimizer_params = lr_dict
            for proj, lr in lr_dict.items():
                if isinstance(proj, str):
                    proj = next(p.projection for p in pytorch_rep.projection_wrappers if p.projection.name == proj)
                proj.parameters.learning_rate.set(lr, context)
            assert self.parameters.learning_rates_dict.get(None) == self._optimizer_constructor_params

        if default_learning_rate is None:
            default_learning_rate = self.parameters.learning_rate.get(default_learning_rate)
        else:
            self.parameters.learning_rate.set(default_learning_rate, context)

        if self._runtime_learning_rate is not None:
            # If _runtime_learning_rate has been specified in call to learn(), make sure that is used
            optimizer_params.update({DEFAULT_LEARNING_RATE: default_learning_rate})

        if (old_opt is None or new) and new is not False:
            # Instantiate a new optimizer if there isn't one yet or new has been called and is not blocked)
            if context.runmode == ContextFlags.LEARNING_MODE:
                # If optimizer is being constructed de novo in call to learn(),
                #    instantiate it using params specified in constructor (if any) since:
                #   - need those implemented in a params_group to revert back to after execution of learn()
                #   - the ones in the call to learn() will be applied in call to _update_optimizer_params() below
                pytorch_rep.optimizer = self._instantiate_optimizer(default_learning_rate,
                                                                    self._optimizer_constructor_params,
                                                                    context)
                # Then update optimizer params with any specified in the call to learn()
                if optimizer_params:
                    pytorch_rep._update_optimizer_params(pytorch_rep.optimizer,
                                                         optimizer_params,
                                                         Context(source=ContextFlags.METHOD,
                                                                 runmode=context.runmode,
                                                                 execution_id=context.execution_id))
            else:
                # Otherwise, if call is from Composition constructor, use params specified by user in that call
                opt_params = optimizer_params or self._optimizer_constructor_params
                pytorch_rep.optimizer = self._instantiate_optimizer(default_learning_rate,
                                                                    opt_params,
                                                                    context)

        elif context.source is ContextFlags.SHOW_GRAPH:
            # Don't bother updating for call to show_graph()
            pass
        else:
            # Otherwise, just update it
            pytorch_rep._update_optimizer_params(old_opt,
                                                 optimizer_params,
                                                 Context(source=ContextFlags.METHOD,
                                                         runmode=context.runmode,
                                                         execution_id=context.execution_id))
        # Set up loss function
        if self.loss_function is not None:
            logger.warning("Overwriting 'loss_function' for AutodiffComposition {}! Old loss function: {}".format(
                self, self.loss_function))
        if callable(self.loss_spec):
            self.loss_function = self.loss_spec
        else:
            self.loss_function = self._get_loss(self.loss_spec)

        return pytorch_rep

    def _instantiate_optimizer(self, learning_rate, optimizer_params, context):

        if isinstance(learning_rate, dict):
            # If learning_rate is a dict, move to optimizer_params and set self.learning_rate to default value
            optimizer_params = learning_rate
            learning_rate = optimizer_params.pop(DEFAULT_LEARNING_RATE,
                                                 self.parameters.learning_rate.default_value)
            self.parameters.learning_rate.set(learning_rate, context)
        if not is_numeric_scalar(learning_rate):
            raise AutodiffCompositionError(
                f"A value ('{learning_rate}') specified in the 'learning_rate' arg of the learn() method "
                f"for '{self.name}' is not valid; it must be an int, float, bool or None.")
        if self.optimizer_type not in ['sgd', 'adam']:
            raise AutodiffCompositionError("Invalid optimizer specified. Optimizer argument must be a string. "
                                           "Currently, Stochastic Gradient Descent and Adam are the only available "
                                           "optimizers (specified as 'sgd' or 'adam').")
        pytorch_rep = self.parameters.pytorch_representation._get(context)
        params = pytorch_rep.parameters()
        # MODIFIED TEACHER_TARGET OLD:
        if (len(pytorch_rep.state_dict()) == 0):
            # avoid expiring params generator
            assert len(list(params)) == 0, \
                (f"PROGRAM ERROR: '{self.name}'.pytorch_representation has parameters "
                 f"but no learnable Projections or entries in its state_dict()")
        # MODIFIED TEACHER_TARGET NEW:
        # if (len(pytorch_rep.state_dict()) == 0
        #         or not any(any([param.requires_grad, param.grad, param.grad_fn])
        #                    for param in list(pytorch_rep.state_dict().values()))
        # ):
        #     # avoid expiring params generator
        #     assert len(list(params)) == 0 or not any(p for p in self.projections if p.learnable), \
        #         (f"PROGRAM ERROR: '{self.name}'.pytorch_representation has parameters "
        #          f"but no learnable Projections or entries in its state_dict()")
        # MODIFIED TEACHER_TARGET END
            warnings.warn(f"'{self.name}' contains no Projections, so it has no params for Pytorch to learn.")
            return
        if self.optimizer_type == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=self.weight_decay)
        else:
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=self.weight_decay)
        pytorch_rep._update_optimizer_params(optimizer, optimizer_params, context)
        return optimizer

    def get_target_nodes(self, execution_mode=pnlvm.ExecutionMode.PyTorch,
                         context=None, base_context=None):
        """Return `TARGET` `Nodes <Composition_Nodes>` of the AutodiffComposition."""
        self.infer_backpropagation_learning_pathways(execution_mode=execution_mode,
                                                     context=context, base_context=base_context)
        return super(AutodiffComposition, self).get_target_nodes()

    def autodiff_forward(self,
                         inputs, targets,
                         optimization_num,
                         synch_with_pnl_options, retain_in_pnl_options,
                         execution_mode, scheduler, context):
        """
        Perform forward pass of model and compute loss for a batch of trials in Pytorch mode.
        Losses are then accumulated, error is backpropagated by compositionrunner.run_learning()
          before the next time it calls run(), in a call to backward() by do_gradient_optimization()
          in _batch_inputs() or _batch_function_inputs(),

        Returns values of all OUTPUT Nodes of pytorch_representation
        """
        assert execution_mode is pnlvm.ExecutionMode.PyTorch
        pytorch_rep = self.parameters.pytorch_representation._get(context)

        # --------- Get current values of nodes  -------------------------------------------------

        # We need to pass both inputs and targets to the forward method in one dict, convert any numpy arrays to torch
        # tensors
        inputs_and_targets = {**inputs, **targets}
        for component, val in list(inputs_and_targets.items()):
            if isinstance(val, torch.Tensor):
                inputs_and_targets[component] = val.to(device=self.device, dtype=torch.double)
            else:
                 inputs_and_targets[component] = torch.tensor(val, device=self.device, dtype=torch.double)

        # Execute PytorchCompositionWrapper to get value of all OUTPUT nodes for current trial
        output_values = pytorch_rep.forward(inputs=inputs_and_targets,
                                            optimization_num=optimization_num,
                                            synch_with_pnl_options=synch_with_pnl_options,
                                            retain_in_pnl_options=retain_in_pnl_options,
                                            full_sequence_mode=self.full_sequence_mode,
                                            sequence_lengths=(
                                                None if not hasattr(pytorch_rep, '_batch_seq_lengths')
                                                else pytorch_rep._batch_seq_lengths),
                                            context=context)

        # TEACHER_TARGET OLD:
        # BREADCRUMB: MOVE TO ITS OWN METHOD FOR FUTURE SUPPORT / PARSING OF DYNAMIC, RUN-TIME TARGETS
        #                    SPECIFIED IN learn(targets={sample/student:target/teacher})
        # # Get value of OUTPUT nodes that are being trained (i.e., for which there are TARGET nodes)
        # curr_tensors_for_trained_outputs = {k:v for k,v in curr_tensors_for_outputs.items()
        #                                     if k in self.outputs_to_targets_map}
        #
        # # Get value of TARGET nodes for current trial
        # curr_tensors_for_targets = {}
        # for component, target in targets.items():
        #     if isinstance(target, torch.Tensor) or isinstance(target, np.ndarray):
        #         curr_tensors_for_targets[component] = [target[:, :, i, ...] for i in range(target.shape[1])]
        #     else:
        #         # It's  a list, of lists, of torch tensors because it is ragged
        #         num_outputs = len(target[0][0])
        #         curr_tensors_for_targets[component] = [torch.stack([torch.stack([s[i] for s in b]) for b in target]) for i in range(num_outputs)]
        #
        # # Map value of TARGET nodes to trained OUTPUT nodes
        # curr_target_tensors_for_trained_outputs = {}
        # for trained_output, target in self.outputs_to_targets_map.items():
        #     curr_target_tensors_for_trained_outputs[trained_output] = curr_tensors_for_targets[target]
        #
        # # --------- Compute the loss (TARGET-OUTPUT) for each trained OUTPUT node  ---------------------------
        #
        # # Calculate and track the loss over the trained OUTPUT nodes:
        # #   curr_target_tensors_for_trained_outputs compared against curr_tensors_for_trained_outputs
        # for component, outputs in curr_tensors_for_trained_outputs.items():
        #     BREADCRUMB: COMPONENT IS A OUTPUT NODE
        #                 OUTPUTS IS TENSOR FOR OUTPUT Node
        #                 targets IS A TENSOR
        #     trial_loss = 0
        #     targets = curr_target_tensors_for_trained_outputs[component]

        #     num_outputs = outputs.shape[1] if type(outputs) is torch.Tensor else len(outputs[0][0])
        #     for i in range(num_outputs):
        #         # loss only accepts 0 or 1d target. reshape assuming pytorch_rep.minibatch_loss dim is correct
        #
        #         # Get the output, if it's a torch tensor we can slice, if it's a list of list (its ragged) and we
        #         # need to index
        #         output = outputs[:, :, i, ...] if type(outputs) is torch.Tensor else torch.stack([torch.stack([s[i] for s in b]) for b in outputs])
        #
        #         # If the sequence dimension is singleton, it can be dropped
        #         if len(output.shape) > 1 and output.shape[1] == 1:
        #             output = output.squeeze(1)
        #             target = torch.atleast_1d(targets[i].squeeze(1))
        #
        #         comp_loss = self.loss_function(
        #             output,
        #             target
        #         )
        #         comp_loss = comp_loss.reshape_as(pytorch_rep.minibatch_loss)
        #         trial_loss += comp_loss
        #     pytorch_rep.minibatch_loss += trial_loss
        # pytorch_rep.minibatch_loss_count += 1
        #
        # # --------- Return the values of output of trained nodes and all nodes  ---------------------------------------
        #
        # # IMPLEMENTATION NOTE: Need values in order corresponding to output_CIM Ports.
        #
        # # Get output Nodes, their out_ports and corresponding indices
        # #     in order of outermost AutodiffComposition's output_CIM Ports
        # outputs_idx_port_node_comp = []
        # for port in self.output_CIM.input_ports:
        #     source_info = self.output_CIM._get_source_info_from_output_CIM(port)
        #     source_ouput_port_idx = source_info[1].output_ports.index(source_info[0])
        #     # BREADCRUMB: DON'T INCLUDE AS OUTPUT IF IT PROJECTS TO ANOTHER NODE IN AN OUTER COMPOSITION
        #     outputs_idx_port_node_comp.append(tuple((source_ouput_port_idx, *source_info)))
        #
        # # Assign values to trained_output_values and all_output_values
        # trained_output_values = []
        # all_output_values = []
        # for item in outputs_idx_port_node_comp:
        #     idx, port, node, comp = item
        #     if comp._trained_comp_nodes_to_pytorch_nodes_map:
        #         node = comp._trained_comp_nodes_to_pytorch_nodes_map[node]
        #     outputs = curr_tensors_for_outputs[node]
        #     if type(outputs) is torch.Tensor:
        #         output = outputs[:, :, idx, ...]
        #     else:
        #         output = torch.stack([torch.stack([s[idx] for s in b]) for b in outputs])
        #
        #     # If the sequence dimension is singleton, squeeze it away
        #     if output.shape[1] == 1:
        #         output = output.squeeze(1)
        #
        #     output = output.detach().cpu().numpy().copy().tolist()
        #     if self.target_ports_for_samples.values():
        #         trained_output_values += [output]
        #     all_output_values += [output]

        pytorch_rep.minibatch_loss = self.compute_loss(targets, pytorch_rep, context)
        pytorch_rep.minibatch_loss_count += 1

        return output_values

    def compute_loss(self, targets, pytorch_rep, context):
        """Compute loss after execution of autodiff_forward()
        Assume that loss is computed using LossMechanism(s) constructed in _instantiate_loss_components.
        Can be overridden to use direct/dedicated/customized computation of loss by subclasses.
        # IMPLEMENTATION NOTE:
        #  targets arg is included for overrides; LossMechanism uses its target input directly
        """
        trial_loss = 0
        assert self.loss_mechs_map, (f"PROGRAM ERROR: compute_loss() called for '{self.name} which does not "
                                     f"have any LossMechanism(s) or an override to compute loss otherwise.'")
        for loss_node in self.loss_mechs_map:
            # Get output of LossMechanism
            comp_loss = pytorch_rep.nodes_map[loss_node].output
            comp_loss = comp_loss.reshape_as(pytorch_rep.minibatch_loss)
            trial_loss += comp_loss
        return trial_loss

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

        if not self.enable_learning:
            return

        pytorch_rep = self.parameters.pytorch_representation._get(context=context)
        optimizer = pytorch_rep.optimizer

        # Gradient updates
        optimizer.zero_grad()
        # Compute and log average loss over all trials since last update
        minibatch_loss.backward(retain_graph=not self.force_no_retain_graph)
        # Update weights and copy to PNL
        optimizer.step()

    def _gen_llvm_function(self, *, ctx:pnlvm.LLVMBuilderContext, tags:frozenset):
        if "run" in tags:
            return pnlvm.codegen.gen_composition_run(ctx, self, tags=tags)
        else:
            return pnlvm.codegen.gen_autodiffcomp_exec(ctx, self, tags=tags)

    def _get_loss(self, loss_spec):
        if not isinstance(self.loss_spec, (str, Loss)):
            return self.loss_spec
        elif loss_spec == Loss.L1:
            return nn.L1Loss(reduction='sum')
        elif loss_spec == Loss.SSE:
            return nn.MSELoss(reduction='sum')
        elif loss_spec == Loss.MSE:
            return nn.MSELoss(reduction='mean')
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
        BREADCRUMB: THE FOLLOWING IS LIMITING;
                      SHOULD ALLOW USE OF THE TARGET NODE'S VALUE ITSELF,
                      SO THAT IT CAN BE SET INTERNALLY BY OTHER NODES (E.G., FOR TEACHING/CONSOLIDATION)
                      MAY NEED TO INSURE THAT TARGET NODE GETS EXECUTED IN FORWARD PASS
        Use input_dict to get input values for TARGET Nodes that are INPUT Nodes of the AutodiffComposition,
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

        for target_port in self.target_ports_for_samples.values():
            # Safe (and cleaner API) to use TARGET Nodes (Mechanisms) here since only one target_port per TARGET Node
            target_values[target_port.owner] = get_target_value(target_port.owner)
        return target_values

    def _map_external_target_values_to_target_nodes(self, target_specs: dict, execution_mode)->dict:
        """Map target values to target mechanisms (as needed by learning)

        Returns
        ---------

        `dict`:
            Dict mapping TargetMechanisms -> target values
        """
        target_values_for_target_nodes = {}
        target_mechs = self.get_nodes_by_role(NodeRole.TARGET)

        # MODIFIED TEACHER_TARGET OLD: MOVED TO COMPOSITION _parse_targets_spec
        # def validate_targets(target_specs)->bool:
        #     """Validate dict specification for samples and targets in learn() method of AutodiffComposition:
        #     Ensure that:
        #         - number of entries in dict equals number of TARGET_MECHANISMS in Composition
        #         - keys are either sample port, sample node, or TARGET nodes
        #     Warn if keys are TARGET Nodes (OK, but sample Node specifiations are simpler and clearer
        #     """
        #     num_target_mechs_in_comp = len(target_mechs)
        #     num_specified_targets = len(target_specs)
        #     if num_specified_targets != num_target_mechs_in_comp:
        #         raise CompositionError(f"The number of items ({num_specified_targets}) specified in the "
        #                                f"`targets` argment of learn() for '{self.name}' must equal the "
        #                                f"number of TARGET Nodes in the Composition ({num_target_mechs_in_comp}.")
        #
        #     # Check for TARGET Nodes specified in target_specs, rather than samples for which they are TARGETS
        #     target_node_as_sample_spec = [f"'{target.name}'" for target in target_specs if target in target_mechs]
        #     if target_node_as_sample_spec and not self._warned_about_target_node_as_sample_spec_in_targets_arg_of_learn:
        #         warnings.warn(f"The dict specified in the 'targets' argument of learn() for '{self.name}' has entries "
        #                       f"that are TARGET Node(s) ({', '.join(target_node_as_sample_spec)}); while this is OK, "
        #                       f"it might be easier and clearer to use the names of the Nodes being used to train the "
        #                       f"network as the keys of the dict, obviating the need to determine the TARGET Node(s). "
        #                       f"Alternatively, TARGET Nodes can be specified in the 'inputs' argument of the learn() "
        #                       f"method, along with INPUT nodes, obviating the need to use a separate 'targets' arg.")
        #         self._warned_about_target_node_as_sample_spec_in_targets_arg_of_learn = True
        #
        #     # Check for specification of Nodes for which TARGET Nodes have not been constructed, either:
        #     #   - explicitly by specifying TARGET for an entry in the **target_specs** arg of the constructor, or
        #     #   - implicitly, if the **target_specs** arg of the constructor was not specified, in which case
        #     #       TARGET Nodes have been constructed for all OUTPUT Nodes of the AutodiffComposition
        #     # TEACHER_TARGET BREADCRUMB: NEED TO HANLDE PORT SPECIFICATONS IN target_specs ARG OF learn()
        #     bad_target_specs = [f"'{sample.owner.name}'" for sample in target_specs
        #                         if sample.owner not in target_mechs and
        #                         sample not in self.target_ports_for_samples]
        #     if bad_target_specs:
        #         raise AutodiffCompositionError(f"The following Node(s) have been specified to receive target inputs "
        #                                        f"in the learn() method of '{self.name}' but are not TARGET Nodes: "
        #                                        f"{', '.join(bad_target_specs)}.")
        #
        #
        # # Validate keys of target_specs dict specified in learn()
        # validate_targets(target_specs)
        # MODIFIED TEACHER_TARGET END

        if execution_mode is not pnlvm.ExecutionMode.PyTorch:
            return super()._map_external_target_values_to_target_nodes(target_specs, execution_mode)

        # Assign target values specified in learn() to TARGET Nodes
        # MODIFIED TEACHER_TARGET OLD:
        # # BREADCRUMB: THIS RETURNS EMPTY DICT FOR target_values_for_target_nodes
        for port, value in target_specs.copy().items():
            if port in self.target_ports_for_samples:
                # Use TARGET Node (target_port owner) for key
                target_values_for_target_nodes[self.target_ports_for_samples[port].owner] = value
        # # MODIFIED TEACHER_TARGET NEW:
        # # BREADCRUMB: WHY BOTHER WITH ALL THIS SINCE VALUES ARE ALREADY ASSIGNED TO TARGET PORTS IN target_specs
        # for target_port, value in target_specs.copy().items():
        #     if target_port in self.target_ports_for_samples.values():
        #         # Get sample to which target is assigned
        #         sample = next(k for k,v in self.target_ports_for_samples.items() if v is target_port)
        #         # Use TARGET Node (target_port owner) for key
        #         target_values_for_target_nodes[self.target_ports_for_samples[sample]] = value
        # MODIFIED TEACHER_TARGET END

        return target_values_for_target_nodes

    def _parse_targets_spec(self, inputs, targets, execution_mode, context):

        # TEACHER_TARGET BREADCRUMB: NEED TO HANLDE PORT SPECIFICATONS IN targets ARG OF learn()
        # self.targets is from **targets** arg of AutodiffComposition constructor and targets is from learn()
        if self.targets and targets:
            # Check whether any samples with nodes specified as targets in the constructor
            #    also appear in the targets dict of learn(): this should not happen,
            #    as they get their target value from the node specified in the constructor
            uncessary_sample_specs_in_learn = []
            for learn_sample in targets:
                for constructor_sample, constructor_target in [spec for spec in self.targets]:
                    if learn_sample is constructor_sample and constructor_target is not TARGET:
                        uncessary_sample_specs_in_learn.append(f"'{learn_sample.name}'")
                # target_node_names = [f"'{node.name}'" for node in self.get_nodes_by_role(NodeRole.TARGET)]
                if uncessary_sample_specs_in_learn:
                    target_error_msg = (f"The following node(s) were specified in the `targets` argument of the "
                                        f"constructor for '{self.name}' as samples that receive their target "
                                        f"values from another node, so they should not be included in the "
                                        f"dict specified in the 'targets' argument of the learn() method: "
                                        f"{', '.join(uncessary_sample_specs_in_learn)}.")
                    raise AutodiffCompositionError(target_error_msg)

        stim_input, num_input_trials = super()._parse_targets_spec(inputs, targets, execution_mode, context)

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

    def _identify_output_nodes(self, context)->list:
        """Recursively call all nested AutodiffCompositions to assign TARGET nodes for learning"""
        # Default is to use OUTPUT
        output_nodes = set(node for node in self.get_nodes_by_role(NodeRole.OUTPUT)
                           if not isinstance(node, Composition))
        for node in self.nodes:
            if isinstance(node, AutodiffComposition):
                output_nodes = output_nodes.union(node._identify_output_nodes(context))
        return output_nodes

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
              execute_in_additional_optimizations: Optional[dict] = None,
              synch_projection_matrices_with_torch: SynchRetainArg = NotImplemented,
              synch_node_variables_with_torch: SynchRetainArg = NotImplemented,
              synch_node_values_with_torch: SynchRetainArg = NotImplemented,
              synch_results_with_torch: SynchRetainArg = NotImplemented,
              retain_torch_trained_outputs: SynchRetainArg = NotImplemented,
              retain_torch_targets: SynchRetainArg = NotImplemented,
              retain_torch_losses: SynchRetainArg = NotImplemented,
              context: Context = None,
              base_context: Context = Context(execution_id=None),
              skip_initialization: bool = False,
              **kwargs
              ) -> list:
        """Override to handle synch and retain args
        Note: defaults for synch and retain args are set to NotImplemented, so that the user can specify None if
              they want to locally override the default values for the AutodiffComposition (see docstrings for run()
              and parse_synch_and_retain_args() for additonal details).

        Arguments
        ---------

        learning_rate : float, int, bool or dict : default 0.001
            specifies the learning rate(s) passed to the optimizer, that overrides any learning_rate specifications
            made in AutodiffComposition constructor and/or individual MappingProjections. If a value is specified,
            it overrides the default learning rate for the Composition, and is used as the default learning rate for
            all MappingProjections in the Composition (and any nested within it) that do not have a specific
            learning_rate specified in their constructor.  A dict can be used to specify
            `MappingProjection`\\-specific learning_rate(s); if it contains a *DEFAULT_LEARNING_RATE* entry,
            that is used in the same was as specifing numeric value; if the dict does not contain a
            *DEFAULT_LEARNING_RATE* entry, then the default indicated above is used for all MappingProjections
            in the Composition, and MappingProjections in any nested Compositions use their default learning_rate
            (see `AutodiffComposition_Learning_Rate` and `Composition_Learning_Rate` for additional details).

        synch_projection_matrices_with_torch : SynchRetainArg : Default NotImplemented
            overrides specification(s) made in Autodiff constructor; see `synch_projection_matrices_with_torch
            <AutodiffComposition.synch_projection_matrices_with_torch>` for additional details.

        synch_node_variables_with_torch : SynchRetainArg : Default NotImplemented
            overrides specification(s) made in Autodiff constructor; see `synch_node_variables_with_torch
            <AutodiffComposition.synch_node_variables_with_torch>` for additional details.

        synch_node_values_with_torch : SynchRetainArg : Default NotImplemented
            overrides specification(s) made in Autodiff constructor; see `synch_node_values_with_torch
            <AutodiffComposition.synch_node_values_with_torch>` for additional details.

        synch_results_with_torch : SynchRetainArg : Default NotImplemented
            overrides specification(s) made in Autodiff constructor; see `synch_results_with_torch
            <AutodiffComposition.synch_results_with_torch>` for additional details.

        retain_torch_trained_outputs : SynchRetainArg : Default NotImplemented
            overrides specification(s) made in Autodiff constructor; see `retain_torch_trained_outputs
            <AutodiffComposition.retain_torch_trained_outputs>` for additional details.

        retain_torch_targets : SynchRetainArg : Default NotImplemented
            overrides specification(s) made in Autodiff constructor; see `retain_torch_targets
            <AutodiffComposition.retain_torch_targets>` for additional details.

        retain_torch_losses : SynchRetainArg : Default NotImplemented
            overrides specification(s) made in Autodiff constructor; see `retain_torch_losses
            <AutodiffComposition.retain_torch_losses>` for additional details.
        """
        execution_phase_at_entry = context.execution_phase
        context.execution_phase = ContextFlags.PREPARING

        execution_mode = self._get_execution_mode(kwargs.pop('execution_mode', None))
        context.execution_phase = execution_phase_at_entry

        # Deal with deprecated arg (can't use deprecation_warning() since that is for constructors)
        if OPTIMIZER_PARAMS in kwargs:
            default_learning_rate = kwargs.pop(LEARNING_RATE, None)
            learning_rate = deprecation_warning(self, kwargs,
                                                deprecated_args={OPTIMIZER_PARAMS:LEARNING_RATE},
                                                method="learn() method",
                                                additional_msg=" Other torch.nn.optimizer parameters are not "
                                                               "currently supported, but will be in a future version.")
            kwargs.update(learning_rate)
            # Move learning_rate spec into optimizer_params dict
            if default_learning_rate is not None:
                if kwargs[LEARNING_RATE]:
                    kwargs[LEARNING_RATE].update({DEFAULT_LEARNING_RATE: default_learning_rate})
                else:
                    kwargs[LEARNING_RATE] = {DEFAULT_LEARNING_RATE: default_learning_rate}

        if LEARNING_RATE in kwargs and isinstance(kwargs[LEARNING_RATE], dict):
            # If learning_rate is a dict:
            # - move it to optimizer_params;
            # - if it contains DEFAULT_LEARNING_RATE entry, assign that as learning_rate
            kwargs[OPTIMIZER_PARAMS] = kwargs[LEARNING_RATE]
            kwargs[LEARNING_RATE] = kwargs[OPTIMIZER_PARAMS].pop(DEFAULT_LEARNING_RATE, None)

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
            self.infer_backpropagation_learning_pathways(execution_mode, context=context, base_context=base_context)
            self._built_pathways = True

        synch_with_pnl_options, retain_in_pnl_options = self.parse_synch_and_retain_args(
            context,
            synch_projection_matrices_with_torch=synch_projection_matrices_with_torch,
            synch_node_variables_with_torch=synch_node_variables_with_torch,
            synch_node_values_with_torch=synch_node_values_with_torch,
            synch_results_with_torch=synch_results_with_torch,
            retain_torch_trained_outputs=retain_torch_trained_outputs,
            retain_torch_targets=retain_torch_targets,
            retain_torch_losses=retain_torch_losses,
        )

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

    def parse_synch_and_retain_args(
        self,
        context: Context,
        synch_projection_matrices_with_torch: SynchRetainArg = NotImplemented,
        synch_node_variables_with_torch: SynchRetainArg = NotImplemented,
        synch_node_values_with_torch: SynchRetainArg = NotImplemented,
        synch_results_with_torch: SynchRetainArg = NotImplemented,
        retain_torch_trained_outputs: SynchRetainArg = NotImplemented,
        retain_torch_targets: SynchRetainArg = NotImplemented,
        retain_torch_losses: SynchRetainArg = NotImplemented,
    ) -> Tuple[Dict, Dict]:
        return self._parse_synch_and_retain_args(
            context,
            synch_projection_matrices_with_torch=synch_projection_matrices_with_torch,
            synch_node_variables_with_torch=synch_node_variables_with_torch,
            synch_node_values_with_torch=synch_node_values_with_torch,
            synch_results_with_torch=synch_results_with_torch,
            retain_torch_trained_outputs=retain_torch_trained_outputs,
            retain_torch_targets=retain_torch_targets,
            retain_torch_losses=retain_torch_losses,
        )

    def _parse_synch_and_retain_args(
        self, context: Context, **kwargs
    ) -> Tuple[Dict, Dict]:
        # Package options for synching and tracking into dictionaries as arguments to learning and exec methods
        def _get_option_val(arg):
            arg_param = getattr(self.parameters, arg)
            val = kwargs.get(arg, NotImplemented)
            try:
                val = LearningScale(val)
            except ValueError:
                # val could be None, which is an acceptable SynchRetainArg
                pass
            if val is NotImplemented:
                val = arg_param._get(context, fallback_value=NotImplemented)
            if val is NotImplemented:
                val = arg_param.default_value
            return val

        # consider making these Parameter aliases
        synch_with_pnl_options = {
            MATRIX_WEIGHTS: "synch_projection_matrices_with_torch",
            NODE_VARIABLES: "synch_node_variables_with_torch",
            NODE_VALUES: "synch_node_values_with_torch",
            RESULTS: "synch_results_with_torch",
        }
        retain_in_pnl_options = {
            TRAINED_OUTPUTS: "retain_torch_trained_outputs",
            TARGETS: "retain_torch_targets",
            LOSSES: "retain_torch_losses",
        }
        for result_name, arg in synch_with_pnl_options.items():
            synch_with_pnl_options[result_name] = _get_option_val(arg)
        for result_name, arg in retain_in_pnl_options.items():
            retain_in_pnl_options[result_name] = _get_option_val(arg)

        if self.minibatch_size > 1:
            args_str = []
            if retain_in_pnl_options[TRAINED_OUTPUTS] in {LearningScale.OPTIMIZATION_STEP, LearningScale.TRIAL}:
                args_str.append('retain_torch_trained_outputs')
            if retain_in_pnl_options[LOSSES] in {LearningScale.OPTIMIZATION_STEP, LearningScale.TRIAL}:
                args_str.append('retain_torch_losses')
            if retain_in_pnl_options[TARGETS] in {LearningScale.OPTIMIZATION_STEP, LearningScale.TRIAL}:
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
                optimization_num=None,
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
                optimizer_params:dict=None,
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

            if scheduler is None:
                scheduler = self.scheduler

            # TBI: How are we supposed to use base_context and statefulness here?
            if ContextFlags.LEARNING_MODE in context.runmode:
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

                output_values = self.autodiff_forward(inputs=autodiff_inputs,
                                                      targets=autodiff_targets,
                                                      optimization_num=optimization_num,
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
                return output_values


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
                                                        optimizer_params=optimizer_params,
                                                        runtime_params=runtime_params,
                                                        execution_mode=execution_mode,
                                                        report=report,
                                                        report_num=report_num
                                                        )

    @handle_external_context(fallback_default=True)
    def run(self, *args,
            execution_mode: pnlvm.ExecutionMode = pnlvm.ExecutionMode.Python,
            synch_projection_matrices_with_torch: SynchRetainArg = NotImplemented,
            synch_node_variables_with_torch: SynchRetainArg = NotImplemented,
            synch_node_values_with_torch: SynchRetainArg = NotImplemented,
            synch_results_with_torch: SynchRetainArg = NotImplemented,
            retain_torch_trained_outputs: SynchRetainArg = NotImplemented,
            retain_torch_targets: SynchRetainArg = NotImplemented,
            retain_torch_losses: SynchRetainArg = NotImplemented,
            batched_results:bool=False,
            context: Context = None,
            **kwargs):
        """Override to handle synch and retain args if run called directly from run() rather than learn()
        Note: defaults for synch and retain args are NotImplemented, so that the user can specify None if they want
              to locally override the default values for the AutodiffComposition (see parse_synch_and_retain_args()
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
                synch_results_with_torch = LearningScale.MINIBATCH
            synch_with_pnl_options, retain_in_pnl_options = self.parse_synch_and_retain_args(
                context,
                synch_projection_matrices_with_torch=synch_projection_matrices_with_torch,
                synch_node_variables_with_torch=synch_node_variables_with_torch,
                synch_node_values_with_torch=synch_node_values_with_torch,
                synch_results_with_torch=synch_results_with_torch,
                retain_torch_trained_outputs=retain_torch_trained_outputs,
                retain_torch_targets=retain_torch_targets,
                retain_torch_losses=retain_torch_losses,
            )
            kwargs[SYNCH_WITH_PNL_OPTIONS] = synch_with_pnl_options
            kwargs[RETAIN_IN_PNL_OPTIONS] = retain_in_pnl_options

        # In LEARNING_MODE, so check that at least one enable_learning is True (potentially in nested Comp)
        if ContextFlags.LEARNING_MODE in context.runmode and not (self._is_learning(context) or
                                                                  any(comp._is_learning(context)
                                                                      for comp in self._get_nested_compositions())):
            raise AutodiffCompositionError(f"The learn() method of '{self.name}' was called, but its "
                                           f"'enable_learning' Parameter (and the ones for any Compositions "
                                           f"nested within) it are set to 'False'. Either set at least one to "
                                           f"'True', or use {self.name}.run().")

        if execution_mode != pnlvm.ExecutionMode.Python and ContextFlags.LEARNING_MODE in context.runmode:
            self._assign_execution_ids(context)
            context.composition = self
            context.source = ContextFlags.COMPOSITION

            if execution_mode is pnlvm.ExecutionMode.PyTorch and not torch_available:
                raise AutodiffCompositionError(f"'{self.name}.learn()' has been called with ExecutionMode.Pytorch, "
                                               f"but Pytorch module ('torch') is not installed. "
                                               f"Please install it with `pip install torch` or `pip3 install torch`")

            self._build_pytorch_representation(optimizer_params=kwargs.get('optimizer_params', None),
                                               learning_rate=self.parameters.learning_rate.get(context),
                                               context=context,
                                               base_context=Context(execution_id=None))

        # Run AutodiffComposition
        results = super(AutodiffComposition, self).run(*args, execution_mode=execution_mode, context=context, **kwargs)

        if execution_mode == pnlvm.ExecutionMode.PyTorch:
            # Synchronize specified outcomes at end of run
            pytorch_rep = self.parameters.pytorch_representation.get(context)
            if pytorch_rep:
                pytorch_rep.synch_with_psyneulink(kwargs[SYNCH_WITH_PNL_OPTIONS], LearningScale.RUN, context)

        return results

    def _update_results(self, results, trial_output, execution_mode, synch_with_pnl_options, context):
        """Track results at specified frequency during learning"""
        if execution_mode is pnlvm.ExecutionMode.PyTorch:

            # Check if the trial_output is atleast 3D
            is_output_3d = trial_output.ndim >= 3 or (trial_output.ndim == 2 and len(trial_output) > 0 and
                                                      isinstance(trial_output[0, 0], (np.ndarray, list)))

            if (RESULTS in synch_with_pnl_options
                    and synch_with_pnl_options[RESULTS] in {LearningScale.TRIAL, LearningScale.MINIBATCH}):
                # Use Composition's own _update_results method since no savings when done trial-by-trial
                if not self.batched_results and is_output_3d:
                    for out in trial_output:
                        super()._update_results(results, out, execution_mode, synch_with_pnl_options, context)
                else:
                    super()._update_results(results, trial_output, execution_mode, synch_with_pnl_options, context)

            elif (RESULTS in synch_with_pnl_options
                  and synch_with_pnl_options[RESULTS] == LearningScale.RUN):
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
    def sample_nodes(self):
        return [loss_mech.sample for loss_mech in self.loss_mechs_map]

    def target_nodes(self):
        return [loss_mech.target for loss_mech in self.loss_mechs_map]

    @property
    def learning_components(self):
        # MODIFIED TEACHER_TARGET NEW:
        pytorch_learning_components = (self.get_nodes_by_role(NodeRole.LEARNING_OBJECTIVE)
                                       + self.get_nodes_by_role(NodeRole.TARGET))
        return pytorch_learning_components or super().learning_components

    @property
    def torch_parameters(self):
        """Return Pytorch Parameters for pytorch_representation of AutodiffComposition"""
        try:
            if self.pytorch_representation is None:
                self._build_pytorch_representation()
            return list(self.pytorch_representation.named_parameters())
        except:
            raise AutodiffCompositionError(
                f"PROGRAM ERROR:  problem accessing torch.named_parameters() for '{self.name}'.")
    @property
    def _dependent_components(self) -> Iterable[Component]:
        res = super()._dependent_components

        # NOTE: _dependent_components should possibly be reworked to be
        # a context-dependent method
        for pytorch_repr in self.parameters.pytorch_representation.values.values():
            if pytorch_repr is not None:
                res.extend([w.projection for w in pytorch_repr.projection_wrappers])

        return res

    def _get_default_comp_learning_rate(self):
        self._get_nested_compositions()
