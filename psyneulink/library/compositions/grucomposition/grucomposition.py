# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* GRUComposition *************************************************

"""
Contents
--------

  * `GRUComposition_Overview`
  * `GRUComposition_Creation`
  * `GRUComposition_Structure`
  * `GRUComposition_Execution`
     - `Processing <GRUComposition_Processing>`
     - `Learning <GRUComposition_Learning>`
  * `GRUComposition_Examples`
  * `GRUComposition_Class_Reference`

.. _GRUComposition_Overview:

Overview
--------

The GRUComposition a subclass of `AutodiffComposition` that implements a single-layered gated recurrent network,
which uses a set of `GatingMechanisms <GatingMechanism>` to implement gates that  modulate the flow of information
through its `hidden_layer_node <GRUComposition.hidden_layer_node>`. This implements the exact same computations as
a PyTorch `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module, which is used to implement
it when its `learn <GRUComposition.learn>` method is called.  When it is executed in Python model, it functions
in the same way as a `GRUCell <https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html>`_ module, processing
its input one stimulus at a time.  However, when used for `learning <GRUComposition_Learning>`, it is executed as
a PyTorch `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module, so that it can used to
process an entire sequence of stimuli at once, and learn to predict the next stimulus in the sequence.

.. _GRUComposition_Creation:

Creation
--------

An GRUComposition is created by calling its constructor.  When it's `learn <AutoDiffComposition.learn>`
method is called, it automatically creates a PytorchGRUCompositionWrapper that implements the GRUComposition
using the PyTorch `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module, that is trained
using PyTorch. Its constructor takes the following arguments that are in addition to or handled differently
than `AutodiffComposition`:

**input_size** (int) specifies the length of the input array to the GRUComposition, and the size
of the `input_node <GRUComposition.input_node>`, which can be different than **hidden_size**.

**hidden_size** (int) specifies the length of the internal ("hidden") state of the GRUComposition,
and the size of the `hidden_layer_node <GRUComposition.hidden_layer_node>` and all nodes other
than the `input_node<GRUComposition.input_node>`, which can be different than **input_size**.

**bias** (bool) specifies whether the GRUComposition includes `BIAS <NodeRole.BIAS>` `Nodes <Composition_Nodes>`
and, correspondingly, the `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module uses
bias vectors in its computations.

.. _GRUComposition_Learning_Arguments:

**enable_learning** (bool) specifies whether learning is enabled for the GRUComposition;  if it is false,
no learning will occur, even when its `learn <AutodiffComposition.learn>` method is called.

**learning_rate** (bool or float): specifies the default learning_rate for the parameters of the Pytorch `GRU
<https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module that are not specified for individual
parameters in the **optimizer_params** argument of the AutodiffComposition's constructor in the call to its `learn
<AutodiffComposition.learn>` method. If it is an int or a float, that is used as the default learning rate for the
GRUComposition; if it is None or True, the GRUComposition's default `learning_rate <GRUComposition.learning_rate>`
(.001) is used; if it is False, then learning will occur only for parameters for which an explicit learning_rate
has been specified in the **optimizer_params** argument of the GRUComposition's constructor
COMMENT: FIX CORRECT?
or in the call to its `learn <AutodiffComposition.learn>` method
COMMENT

.. _GRUComposition_Individual_Learning_Rates:

**optimizer_params** (dict): used to specify parameter-specific learning rates, which supercede the value of the
GRUCompositon's `learning_rate <GRUComposition.learning_rate>`. Keys of the dict must reference parameters of the
`GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module, and values their learning_rates,
as described below.

  **Keys** for specifying individual parameters in the **optimizer_params** dict:

    - *`w_ih`*: learning rate for the ``weight_ih_l0`` parameter of the PyTorch `GRU
      <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module that corresponds to the weights of the
      efferent projections from the `input_node <GRUComposition.input_node>` of the GRUComposition: `wts_in
      <GRUComposition.wts_in>`, `wts_iu <GRUComposition.wts_iu>`, and `wts_ir <GRUComposition.wts_ir>`; its value
      is stored in the `w_ih_learning_rate <GRUComposition.w_ih_learning_rate>` attribute of the GRUComposition;

    - *`w_hh`*: learning rate for the ``weight_hh_l0`` parameter of the PyTorch `GRU
      <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module that corresponds to the weights of the
      efferent projections from the `hidden_layer_node <GRUComposition.hidden_layer_node>` of the GRUComposition:
      `wts_hn <GRUComposition.wts_hn>`, `wts_hu <GRUComposition.wts_hu>`, `wts_hr <GRUComposition.wts_hr>`; its
      value is stored in the `w_hh_learning_rate <GRUComposition.w_hh_learning_rate>` attribute of the GRUComposition;

    - *`b_ih`*: learning rate for the ``bias_ih_l0`` parameter of the PyTorch `GRU
      <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module that corresponds to the biases of the
      efferent projections from the `input_node <GRUComposition.input_node>` of the GRUComposition: `bias_ir
      <GRUComposition.bias_ir>`, `bias_iu <GRUComposition.bias_iu>`, `bias_in <GRUComposition.bias_in>`; its value
      is stored in the `b_ih_learning_rate <GRUComposition.b_ih_learning_rate>` attribute of the GRUComposition;

    - *`b_hh`*: learning rate for the ``bias_hh_l0`` parameter of the PyTorch `GRU
      <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module that corresponds to the biases of the
      efferent projections from the `hidden_layer_node <GRUComposition.hidden_layer_node>` of the GRUComposition:
      `bias_hr <GRUComposition.bias_hr>`, `bias_hu <GRUComposition.bias_hu>`, `bias_hn <GRUComposition.bias_hn>`; its
      value is stored in the `b_hh_learning_rate <GRUComposition.b_hh_learning_rate>` attribute of theGRUComposition.

  **Values** for specifying an individual parameter's learning_rate in the **optimizer_params** dict

    - *int or float*: the value is used as the learning_rate;

    - *True or None*: the value of the GRUComposition's `learning_rate <GRUComposition.learning_rate>` is used;

    - *False*: the parameter is not learned.


.. _GRUComposition_Structure:

Structure
---------

The GRUComposition assigns a node to each of the computations of the PyTorch `GRU
<https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module, and a Projetion to each of
its weight and bias parameters, as shown in the figure below:

.. figure:: _static/GRUComposition_fig.svg
   :alt: GRU Composition
   :width: 400
   :align: center

   **Structure of a GRUComposition** -- can be seen in more detail using the Composition's s `show_graph
   <ShowGraph.show_graph>` method with its **show_node_structure** argument set to ``True`` or ``ALL``;
   can also be seen with biases added by setting the **show_bias** argument to ``True`` in the constructor.

The `input_node <GRUComposition.input_node>` receives the input to the GRUComposition, and passes it to the
`hidden_layer_node <GRUComposition.hidden_layer_node>`, that implements the recurrence and integration function of
a GRU.  The `reset_node <GRUComposition.reset_node>` gates the input to the `new_node<GRUComposition.new_node>`. The
`update_node <GRUComposition.update_node>` gates the input to the `hidden_layer_node<GRUComposition.hidden_layer_node>`
from the `new_node <GRUComposition.new_node>` (current input) and the prior state of the `hidden_layer_node
<GRUComposition.hidden_layer_node>` (i.e., the input it receives from its recurrent Projection).  The `output_node
<GRUComposition.output_node>` receives the output of current state of the `hidden_layer_node
<GRUComposition.hidden_layer_node>` that is provided as the output of the GRUComposition.  The `reset_gate
<GRUComposition.reset_gate>` and `update_node <GRUComposition.update_node>` are `GatingMechanisms <GatingMechanism>`,
while the other nodes are all `Processing Mechanisms <ProcessingMechanism>`.

.. note::
   The GRUComposition is limited to a single layer GRU at present, thus its ``num_layers`` argument is not
   implemented.  Similarly, ``dropout`` and ``bidirectional`` arguments are not yet implemented.  These will
   be added in a future version.

COMMENT:
FIX: ADD EXPLANATION OF THE FOLLOWING
.. technical_note::
   gru_mech
   target_node
   PytorchGRUProjectionWrappers for nested case
COMMENT

.. _GRUComposition_Execution:

Execution
---------

.. _GRUComposition_Processing:

*Processing*
~~~~~~~~~~~~

The GRUComposition implements the following computations by its `reset <GRUComposition.reset_node>`, `update
<GRUComposition.update_node>`, `new <GRUComposition.new_node>`, and `hidden_layer <GRUComposition.hidden_layer_node>`
`Nodes <Composition_Nodes>` when it is executed:

    `reset <GRUComposition.reset_gate>`\\(t) = `Logistic`\\[(`wts_ir <GRUComposition.wts_ir>` *
    `input <GRUComposition.input_node>`) + `bias_ir <GRUComposition.bias_ir>` +
    (`wts_hr <GRUComposition.wts_hr>` * `hidden_layer <GRUComposition.hidden_layer_node>`\\(t-1)) +
    `bias_hr <GRUComposition.bias_hr>`)]

    `update <GRUComposition.update_node>`\\(t) = `Logistic`\\[(`wts_iu <GRUComposition.wts_iu>` *
    `input <GRUComposition.input_node>`) + `bias_iu <GRUComposition.bias_iu>` + (`wts_hu <GRUComposition.wts_hu>` *
    `hidden_layer <GRUComposition.hidden_layer_node>`\\(t-1)) + `bias_hu <GRUComposition.bias_hu>`]

    `new <GRUComposition.new_node>`\\(t) = :math:`tanh`\\[(`wts_in <GRUComposition.wts_in>` *
    `input <GRUComposition.input_node>`) + `bias_in <GRUComposition.bias_in>` +
    (`reset <GRUComposition.reset_gate>`\\(t) * (`wts_hn <GRUComposition.wts_hn>` *
    `hidden_layer <GRUComposition.hidden_layer_node>`\\(t-1) + `bias_hn <GRUComposition.bias_hn>`)]

    `hidden_layer <GRUComposition.hidden_layer_node>`\\(t) = [(1 - `update <GRUComposition.update_node>`\\(t)) *
    `new <GRUComposition.new_node>`\\(t)] + [`update <GRUComposition.update_node>`\\(t) * `hidden_layer
    <GRUComposition.hidden_layer_node>`\\(t-1)]

COMMENT:
where:
    r(t) = reset gate

    z(t) = update gate

    n(t) = new gate

    h(t) = hidden layer

    x(t) = input

    W_ir, W_iz, W_in, W_hr, W_hz, W_hn = input, update, and reset weights

    b_ir, b_iz, b_in, b_hr, b_hz, b_hn = input, update, and reset biases
COMMENT

This corresponds to the computations of the `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module:

.. math::

   &reset = Logistic(wts\\_ir \\cdot input + bias\\_ir + wts\\_hr \\cdot hidden + bias\\_hr)

   &update = Logistic(wts\\_iu \\cdot input + bias\\_iu + wts\\_hu \\cdot hidden + bias\\_hu)

   &new = Tanh(wts\\_in \\cdot input + bias\\_in + reset \\cdot (wts\\_hn \\cdot hidden + bias\\_hn))

   &hidden = (1 - update) \\odot new + update \\odot hidden

where :math:`\\cdot` is the dot product, :math:`\\odot` is the Hadamard product, and all values are for the
current execution of the Composition *(t)* except for hidden, which uses the value from the prior execution *(t-1)*
(see `Cycles <Composition_Cycle>` for handling of recurrence and cycles).


.. technical_note::
    The `full Composition <GRUComposition_Structure>` is executed when its `run <Composition.run>` method is
    called with **execution_mode** set to `ExecutionMode.Python`, or if ``torch_available`` is False.  Otherwise, and
    always in a call to `learn <AutodiffComposition.learn>`, the GRUComposition is executed using the PyTorch `GRU
    <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module with values of the individual
    computations copied back to Nodes of the full GRUComposition at times determined by the value of the
    `synch_node_values_with_torch <AutodiffComposition.synch_node_values_with_torch>` option.


.. _GRUComposition_Learning:

*Learning*
~~~~~~~~~~

Learning is executed using the `learn` method in same way as a standard `AutodiffComposition`.  For learning to
occur the following conditions must obtain:

  - `enable_learning <GRUComposition.enable_learning>` must be set to `True` (the default);

  - GRUCompositions's `learning_rate <GRUComposition.learning_rate>` must not be False and/or the
    `learning_rate of individual parameters <GRUComposition_Individual_Learning_Rates>` must not all be False;

  - **execution_mode** argument of the `learn <AutodiffComposition.learn>` method must `ExecutionMode.PyTorch`
    (the default).

  .. note:: Because a GRUComposition uses the PyTorch `GRU
     <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module to implement its computations during
     learning, its `learn <AutodiffComposition.learn>` method can only be called with the **execution_mode**
     argument set to `ExecutionMode.PyTorch` (the default).

The GRUComposition uses the PyTorch `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module
to implement its computations during learning. After learning, the values of the module's parameters are copied
to the weight `matrices <MappingProjection.matrix>` of the corresponding `MappingProjections <MappingProjection>`,
and results of computations are copied to the `values <Mechanism_Base.value>` of the corresponding `Nodes
<Composition_Nodes>` in the GRUComposition at times determined by the value of the `synch_node_values_with_torch
<AutodiffComposition.synch_node_values_with_torch>` option.

COMMENT:
.. _GRUComposition_Examples:

Examples
--------

The following are examples of how to configure and initialize a GRUComposition:
COMMENT

.. _GRUComposition_Class_Reference:

Class Reference
---------------
"""
import numpy as np
import warnings
from typing import Union
# from sympy.stats import Logistic
from collections import deque

import psyneulink.core.scheduling.condition as conditions
from psyneulink.core.components.functions.nonstateful.transformfunctions import LinearCombination
from psyneulink.core.components.functions.nonstateful.transferfunctions import Linear, Logistic, Tanh
from psyneulink.core.components.functions.nonstateful.transformfunctions import MatrixTransform
from psyneulink.core.components.functions.function import (
    DEFAULT_SEED, get_matrix, _random_state_getter, _seed_setter)
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.compositions.composition import CompositionError, NodeRole
from psyneulink.library.compositions.autodiffcomposition import AutodiffComposition, torch_available
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.modulatory.control.gating.gatingmechanism import GatingMechanism
from psyneulink.core.components.ports.modulatorysignals.gatingsignal import GatingSignal
from psyneulink.core.components.projections.projection import DuplicateProjectionError
from psyneulink.core.components.projections.modulatory.gatingprojection import GatingProjection
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.keywords import (
    CONTEXT, FULL_CONNECTIVITY_MATRIX, GRU_COMPOSITION, IDENTITY_MATRIX, OUTCOME, SUM)
from psyneulink.core import llvm as pnlvm
from psyneulink.core.llvm import ExecutionMode

__all__ = ['GRUComposition', 'GRUCompositionError',
           'INPUT_NODE', 'HIDDEN_LAYER', 'RESET_NODE',
           'UPDATE_NODE', 'NEW_NODE', 'OUTPUT_NODE', 'GRU_INTERNAL_STATE_NAMES', 'GRU_NODE', 'GRU_TARGET_NODE']

# Node names
INPUT_NODE = 'INPUT'
NEW_NODE = 'NEW'
RESET_NODE = 'RESET'
UPDATE_NODE = 'UPDATE'
HIDDEN_LAYER = 'HIDDEN\nLAYER'
OUTPUT_NODE = 'OUTPUT'
GRU_INTERNAL_STATE_NAMES = [NEW_NODE, RESET_NODE, UPDATE_NODE, HIDDEN_LAYER]
GRU_NODE = 'PYTORCH GRU NODE'
GRU_TARGET_NODE = 'GRU TARGET NODE'

class GRUCompositionError(CompositionError):
    pass


class GRUComposition(AutodiffComposition):
    """
    GRUComposition(                         \
        name="GRU_Composition"              \
        input_size=1,                       \
        hidden_size=1,                      \
        bias=False                          \
        enable_learning=True                \
        learning_rate=.01                   \
        optimizer_params=None               \
        )

    Subclass of `AutodiffComposition` that implements a single-layered gated recurrent network.

    See `GRUComposition_Structure` and technical_note under under `GRUComposition_Execution`
    for a description of when the full Composition is constructed and used for execution
    vs. when the PyTorch `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_
    module is used.

    Note: all exposed methods, attributes and `Parameters <Parameter>`) of the GRUComposition are
          PsyNeuLink elements; all PyTorch-specific elements belong to `pytorch_representation
          <AutodiffComposition.pytorch_representation>` which, for a GRUComposition, is of class
          `PytorchGRUCompositionWrapper`.

    Constructor takes the following arguments in addition to those of `AutodiffComposition`:

    Arguments
    ---------

    input_size : int : default 1
        specifies the length of the input array to the GRUComposition, and the size of the `input_node
        <GRUComposition.input_node>`.

    hidden_size : int : default 1
        specifies the length of the internal state of the GRUComposition, and the size of the `hidden_layer_node
        <GRUComposition.hidden_layer_node>` and all nodes other than the `input_node<GRUComposition.input_node>`.

    bias : bool : default False
        specifies whether the GRUComposition uses bias vectors in its computations.

    COMMENT:
    num_layers : int : default 1
     batch_first : bool : default False
     dropout : float : default 0.0
     bidirectional : bool : default False
    COMMENT

    enable_learning : bool : default True
        specifies whether learning is enabled for the GRUComposition (see `Learning Arguments
        <GRUComposition_Learning_Arguments>` for additional details).

    learning_rate : float : default .001
        specifies the learning_rate for the GRUComposition (see `Learning Arguments
        <GRUComposition_Learning_Arguments>` for additional details).

    optimizer_params : Dict[str: value]
        specifies parameters for the optimizer used for learning by the GRUComposition
        (see `Learning Arguments <GRUComposition_Learning_Arguments>` for details of specification).

    Attributes
    ----------

    input_size : int
        determines the length of the input array to the GRUComposition and size of the `input_node
        <GRUComposition.input_node>`.

    hidden_size : int
        determines the size of the `hidden_layer_node` and all other `INTERNAL` `Nodes <Composition_Nodes>`
        of the GRUComposition.

    bias : bool
        determines whether the GRUComposition uses bias vectors in its computations.

    COMMENT:
    num_layers : int : default 1
     batch_first : bool : default False
     dropout : float : default 0.0
     bidirectional : bool : default False
    COMMENT

    enable_learning : bool
        determines whether learning is enabled for the GRUComposition
        (see `Learning Arguments <GRUComposition_Learning_Arguments>` for additional details).

    learning_rate : float
        determines the default learning_rate for the parameters of the Pytorch `GRU
        <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module that are not specified
        for individual parameters in the **optimizer_params** argument of the AutodiffComposition's
        constructor in the call to its `learn <GRUComposition.learn>` method (see `Learning Arguments
        <GRUComposition_Learning_Arguments>` for additional details).

    w_ih_learning_rate : flot or bool
        determines the learning rate specifically for the weights of the `efferent projections
        <Mechanism_Base.efferents>` from the `input_node <GRUComposition.input_node>`
        of the GRUComposition: `wts_in <GRUComposition.wts_in>`, `wts_iu <GRUComposition.wts_iu>`,
        and `wts_ir <GRUComposition.wts_ir>`; corresponds to the ``weight_ih_l0`` parameter of the
        PyTorch `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module
        (see `Learning Arguments <GRUComposition_Learning_Arguments>` for additional details).

    w_hh_learning_rate : float or bool
        determines the learning rate specifically for the weights of the `efferent projections
        <Mechanism_Base.efferents>` from the `hidden_layer_node <GRUComposition.hidden_layer_node>`
        of the GRUComposition: `wts_hn <GRUComposition.wts_hn>`, `wts_hu <GRUComposition.wts_hu>`,
        `wts_hr <GRUComposition.wts_hr>`; corresponds to the ``weight_hh_l0`` parameter of the
        PyTorch `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module
         (see `Learning Arguments <GRUComposition_Learning_Arguments>` for additional details).

    b_ih_learning_rate : float or bool
        determines the learning rate specifically for the biases influencing the `efferent projections
        <Mechanism_Base.efferents>` from the `input_node <GRUComposition.input_node>` of the GRUComposition:
        `bias_ir <GRUComposition.bias_ir>`, `bias_iu <GRUComposition.bias_iu>`, `bias_in <GRUComposition.bias_in>`;
        corresponds to the ``bias_ih_l0`` parameter of the PyTorch `GRU module (see `Learning Arguments
        <GRUComposition_Learning_Arguments>` for additional details).

    b_hh_learning_rate : float or bool
        determines the learning rate specifically for the biases influencing the `efferent projections
        <Mechanism_Base.efferents>` from the `hidden_layer_node <GRUComposition.hidden_layer_node>` of
        the GRUComposition: `bias_hr <GRUComposition.bias_hr>`, `bias_hu <GRUComposition.bias_hu>`,
        `bias_hn <GRUComposition.bias_hn>`; corresponds to the ``bias_hh_l0`` parameter of the PyTorch
        `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module (see `Learning Arguments
        <GRUComposition_Learning_Arguments>` for additional details).

    input_node : ProcessingMechanism
        `INPUT <NodeRole.INPUT>` `Node <Composition_Nodes>` that receives the input to the GRUComposition and passes
        it to the `hidden_layer_node <GRUComposition.hidden_layer_node>`; corresponds to input *(i)* of the PyTorch
        `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module.

    new_node : ProcessingMechanism
        `ProcessingMechanism` that provides the `hidden_layer_node <GRUComposition.hidden_layer_node>`
        with the input from the `input_node <GRUComposition.input_node>`, gated by the `reset_node
        <GRUComposition.reset_node>`; corresponds to new gate *(n)* of the PyTorch `GRU
        <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module.

    hidden_layer_node : ProcessingMechanism
        `ProcessingMechanism` that implements the recurrent layer of the GRUComposition; corresponds to
        hidden layer *(h)* of the PyTorch `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module.

    reset_node : GatingMechanism
        `GatingMechanism` that that gates the input to the `new_node <GRUComposition.new_node>`; corresponds to reset
        gate *(r)* of the PyTorch `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module.

    update_node : GatingMechanism
        `GatingMechanism` that gates the inputs to the hidden layer from the `new_node <GRUComposition.new_node>`
        and the prior state of the `hidden_layer_node <GRUComposition.hidden_layer_node>` itself (i.e., the input
        it receives from its recurrent Projection); corresponds to update gate *(z)* of the PyTorch `GRU
        <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module.

    output_node : ProcessingMechanism
        `OUTPUT <NodeRole.INPUT>` `Node <Composition_Nodes>` that receives the output of the `hidden_layer_node
        <GRUComposition.hidden_layer_node>`; corresponds to result of the PyTorch `GRU
        <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module.

    learnable_projections : List[MappingProjection]
        list of the `MappingProjections <MappingProjection>` in the GRUComposition that have
        `matrix <MappingProjection.matrix>` parameters that can be learned; these correspond to the learnable
        parameters of the PyTorch `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module.

    wts_in : MappingProjection
        `MappingProjection` with learnable `matrix <MappingProjection.matrix>` ("connection weights") that projects
        from the `input_node <GRUComposition.input_node>` to the `new_node <GRUComposition.new_node>`; corresponds to
        :math:`W_{in}` term in the PyTorch `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module's
        computation (see `GRUComposition_Structure` for additional information).

    wts_iu : MappingProjection
        `MappingProjection` with learnable `matrix <MappingProjection.matrix>` ("connection weights") that projects
        from the `input_node <GRUComposition.input_node>` to the `update_node <GRUComposition.update_node>`; corresponds
        to :math:`W_{iz}` term in the PyTorch `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_
        module's computation (see `GRUComposition_Structure` for additional information).

    wts_ir : MappingProjection
        `MappingProjection` with learnable `matrix <MappingProjection.matrix>` ("connection weights") that projects
        from the `input_node <GRUComposition.input_node>` to the `reset_node <GRUComposition.reset_node>`; corresponds
        to :math:`W_{ir}` term in the PyTorch `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_
        module's computation (see `GRUComposition_Structure` for additional information).

    wts_nh : MappingProjection
        `MappingProjection` with learnable `matrix <MappingProjection.matrix>` ("connection weights") that projects
        from the `new_node <GRUComposition.new_node>` to the `hidden_layer_node <GRUComposition.hidden_layer_node>`.
        (see `GRUComposition_Structure` for additional information).

    wts_hr : MappingProjection
        `MappingProjection` with learnable `matrix <MappingProjection.matrix>` ("connection weights")
        that projects from the `hidden_layer_node <GRUComposition.hidden_layer_node>` to the
        `reset_node <GRUComposition.reset_node>`; corresponds to :math:`W_{hr}` term in the PyTorch
        `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module's computation
        (see `GRUComposition_Structure` for additional information).

    wts_hu : MappingProjection
        `MappingProjection` with learnable `matrix <MappingProjection.matrix>` ("connection weights")
        that projects from the `hidden_layer_node <GRUComposition.hidden_layer_node>` to the
        `update_node <GRUComposition.update_node>`; corresponds to :math:`W_{hz}` in the PyTorch
        `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module's computation
        (see `GRUComposition_Structure` for additional information).

    wts_hn : MappingProjection
        `MappingProjection` with learnable `matrix <MappingProjection.matrix>` ("connection weights")
        that projects from the `hidden_layer_node <GRUComposition.hidden_layer_node>` to the `new_node
        <GRUComposition.new_node>`; corresponds to :math:`W_{hn}` in the PyTorch
        `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module's computation
        (see `GRUComposition_Structure` for additional information).

    wts_hh : MappingProjection
        `MappingProjection` with fixed `matrix <MappingProjection.matrix>` ("connection weights") that projects
        from the `hidden_layer_node <GRUComposition.hidden_layer_node>` to itself (i.e., the recurrent Projection).
        (see `GRUComposition_Structure` for additional information).

    wts_ho : MappingProjection
        `MappingProjection` with fixed `matrix <MappingProjection.matrix>` ("connection weights") that projects from
        the `hidden_layer_node <GRUComposition.hidden_layer_node>` to the `output_node <GRUComposition.output_node>`.
        (see `GRUComposition_Structure` for additional information).

    reset_gate : GatingProjection
        `GatingProjection` that gates the input to the `new_node <GRUComposition.new_node>` from the `input_node
        <GRUComposition.input_node>`; its `value <GatingProjection.value>` is used in the Hadamard product with
        the input to produce the new (external) input to the `hidden_layer_node <GRUComposition.hidden_layer_node>`.
        (see `GRUComposition_Structure` for additional information).

    new_gate : GatingProjection
        `GatingProjection` that gates the input to the `hidden_layer_node <GRUComposition.hidden_layer_node>` from the
        `new_node <GRUComposition.new_node>`; its `value <GatingProjection.value>` is used in the Hadamard product
        with the (external) input to the `hidden_layer_node <GRUComposition.hidden_layer_node>` from the `new_node
        <GRUComposition.new_node>`, which determines how much of the `hidden_layer_node
        <GRUComposition.hidden_layer_node>`\\'s new state is determined by the external input vs. its prior state
        (see `GRUComposition_Structure` for additional information).

    recurrent_gate : GatingProjection
        `GatingProjection` that gates the input to the `hidden_layer_node <GRUComposition.hidden_layer_node>` from its
        recurrent projection (`wts_hh <GRUComposition.wts_hh>`); its `value <GatingProjection.value>` is used in the
        in the Hadamard product with the recurrent input to the `hidden_layer_node <GRUComposition.hidden_layer_node>`,
        which determines how much of the `hidden_layer_node <GRUComposition.hidden_layer_node>`\\'s
        new state is determined by its prior state vs.its external input
        (see `GRUComposition_Structure` for additional information).

    bias_ir_node : ProcessingMechanism
        `BIAS` `Node <Composition_Nodes>`, the Projection from which (`bias_ir <GRUComposition.bias_ir>`) provides the
        the bias to weights (`wts_ir <GRUComposition.wts_ir>`) from the `input_node <GRUComposition.input_node>` to the
        `reset_node <GRUComposition.reset_node>` (see `GRUComposition_Structure` for additional information).

    bias_iu_node : ProcessingMechanism
        `BIAS` `Node <Composition_Nodes>`, the Projection from which (`bias_iu <GRUComposition.bias_iu>`) provides
        the the bias to weights (`wts_iu <GRUComposition.wts_iu>`) from the `input_node <GRUComposition.input_node>`
        to the `update_node <GRUComposition.update_node>` (see `GRUComposition_Structure` for additional information).

    bias_in_node : ProcessingMechanism
        `BIAS` `Node <Composition_Nodes>`, the Projection from which (`bias_in <GRUComposition.bias_in>`) provides the
        the bias to weights (`wts_in <GRUComposition.wts_in>`) from the `input_node <GRUComposition.input_node>` to the
        `new_node <GRUComposition.new_node>` (see `GRUComposition_Structure` for additional information).

    bias_hr_node : ProcessingMechanism
        `BIAS` `Node <Composition_Nodes>`, the Projection from which (`bias_hr <GRUComposition.bias_hr>`) provides the
        the bias to weights (`wts_hr <GRUComposition.wts_hr>`) from the `hidden_layer_node
        <GRUComposition.hidden_layer_node>` to the `reset_node <GRUComposition.reset_node>`
        (see `GRUComposition_Structure` for additional information).

    bias_hu_node : ProcessingMechanism
        `BIAS` `Node <Composition_Nodes>`, the Projection from which (`bias_hu <GRUComposition.bias_hu>`) provides the
        the bias to weights (`wts_hu <GRUComposition.wts_hu>`) from the `hidden_layer_node
        <GRUComposition.hidden_layer_node>` to the `update_node <GRUComposition.update_node>`
        (see `GRUComposition_Structure` for additional information).

    bias_hn_node : ProcessingMechanism
        `BIAS` `Node <Composition_Nodes>`, the Projection from which (`bias_hn <GRUComposition.bias_hn>`) provides the
        the bias to weights (`wts_hn <GRUComposition.wts_hn>`) from the `hidden_layer_node
        <GRUComposition.hidden_layer_node>` to the `new_node <GRUComposition.new_node>`
        (see `GRUComposition_Structure` for additional information).

    biases : List[MappingProjection]
        list of the `MappingProjections <MappingProjection>` from the `BIAS <NodeRole.BIAS>` `Nodes of
        the GRUComposition, all of which have `matrix <MappingProjection.matrix>` parameters if `bias
        <GRUComposition.bias>` is True; these correspond to the learnable biases of the PyTorch `GRU
        <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module
        (see `GRUComposition_Structure` for additional information).

    bias_ir : MappingProjection
        `MappingProjection` with learnable `matrix <MappingProjection.matrix>` ("connection weights") that provides
        the bias to the weights, `wts_ir <GRUComposition.wts_ir>`, from the `input_node <GRUComposition.input_node>`
        to the `reset_node <GRUComposition.reset_node>`; corresponds to the :math:`b_ir` bias parameter of the
        PyTorch `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module
        (see `GRUComposition_Structure` for additional information).

    bias_iu : ProcessingMechanism
        `MappingProjection` with learnable `matrix <MappingProjection.matrix>` ("connection weights") that provides
        the bias to the weights, `wts_iu <GRUComposition.wts_iu>`, from the `input_node <GRUComposition.input_node>`
        to the `update_node <GRUComposition.update_node>`; corresponds to the :math:`b_iz` bias parameter of the
        PyTorch `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module
        (see `GRUComposition_Structure` for additional information).

    bias_in : ProcessingMechanism
        `MappingProjection` with learnable `matrix <MappingProjection.matrix>` ("connection weights") that provides
        the bias to the weights, `wts_in <GRUComposition.wts_in>`, from the `input_node <GRUComposition.input_node>`
        to the `new_node <GRUComposition.new_node>`; corresponds to the :math:`b_in` bias parameter of the
        PyTorch `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module
        (see `GRUComposition_Structure` for additional information).

    bias_hr : ProcessingMechanism
        `MappingProjection` with learnable `matrix <MappingProjection.matrix>` ("connection weights")
        that provides the bias to the weights, `wts_hr <GRUComposition.wts_hr>`, from the `hidden_layer_node
        <GRUComposition.hidden_layer_node>` to the `reset_node <GRUComposition.reset_node>`;
        corresponds to the :math:`b_hr` bias parameter of the PyTorch `GRU
        <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module
        (see `GRUComposition_Structure` for additional information).

    bias_hu : ProcessingMechanism
        `MappingProjection` with learnable `matrix <MappingProjection.matrix>` ("connection weights") that provides
        the bias to the weights, `wts_hu <GRUComposition.wts_hu>`, from the `hidden_layer_node
        <GRUComposition.hidden_layer_node>` to the `update_node <GRUComposition.update_node>`;
        corresponds to the :math:`b_hz` bias parameter of the PyTorch `GRU
        <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module
        (see `GRUComposition_Structure` for additional information).

    bias_hn : ProcessingMechanism
        `MappingProjection` with learnable `matrix <MappingProjection.matrix>` ("connection weights") that provides
        the bias to the weights, `wts_hn <GRUComposition.wts_hn>`, from the `hidden_layer_node
        <GRUComposition.hidden_layer_node>` to the `new_node <GRUComposition.new_node>`; corresponds to the :math:`b_hn`
        bias parameter of the PyTorch `GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module
        (see `GRUComposition_Structure` for additional information).
    """

    componentCategory = GRU_COMPOSITION

    if torch_available:
        from psyneulink.library.compositions.grucomposition.pytorchGRUwrappers import \
            PytorchGRUCompositionWrapper, PytorchGRUMechanismWrapper
        pytorch_composition_wrapper_type = PytorchGRUCompositionWrapper
        pytorch_mechanism_wrapper_type = PytorchGRUMechanismWrapper

    class Parameters(AutodiffComposition.Parameters):
        """
            Attributes
            ----------

                bias
                    see `bias <GRUComposition.bias>`

                    :default value: False
                    :type: ``bool``

                enable_learning
                    see `enable_learning <GRUComposition.enable_learning>`

                    :default value: True
                    :type: ``bool``

                gru_mech
                    see `gru_mech <GRUComposition.gru_mech>`

                    :default value: None
                    :type: ``ProcessingMechanism``

                hidden_biases_learning_rate
                    see `hidden_biases_learning_rate <GRUComposition.hidden_biases_learning_rate>`

                    :default value: True
                    :type: ``bool``

                hidden_size
                    see `hidden_size <GRUComposition.hidden_size>`

                    :default value: 1
                    :type: ``int``

                hidden_state

                    :default value: None
                    :type: ``ndarray``

                hidden_weights_learning_rate
                    see `hidden_weights_learning_rate <GRUComposition.hidden_weights_learning_rate>`

                    :default value: True
                    :type: ``bool``

                input_biases_learning_rate
                    see `input_biases_learning_rate <GRUComposition.input_weights_learning_rate>`

                    :default value: True
                    :type: ``bool``

                input_size
                    see `input_size <GRUComposition.input_size>`

                    :default value: 1
                    :type: ``int``

                input_weights_learning_rate
                    see `input_weights_learning_rate <GRUComposition.input_weights_learning_rate>`

                    :default value: True
                    :type: ``bool``

                learning_rate
                    see `learning_results <GRUComposition.learning_rate>`

                    :default value: []
                    :type: ``list``

                random_state
                    see `random_state <NormalDist.random_state>`

                    :default value: None
                    :type: ``numpy.random.RandomState``

        """
        input_size = Parameter(1, structural=True, stateful=False)
        hidden_size = Parameter(1, structural=True, stateful=False)
        bias = Parameter(False, structural=True, stateful=False)
        gru_mech = Parameter(None, structural=True, stateful=False)
        enable_learning = Parameter(True, structural=True)
        learning_rate = Parameter(.001, modulable=True)
        input_weights_learning_rate = Parameter(True, structural=True)
        hidden_weights_learning_rate = Parameter(True, structural=True)
        input_biases_learning_rate = Parameter(True, structural=True)
        hidden_biases_learning_rate = Parameter(True, structural=True)
        random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies='seed')
        seed = Parameter(DEFAULT_SEED(), modulable=True, setter=_seed_setter)

        def _validate_input_size(self, size):
            if not (isinstance(size, np.ndarray) and isinstance(size.tolist(),int)):
                return 'must be an integer'

        def _validate_hidden_size(self, size):
            if not (isinstance(size, np.ndarray) and isinstance(size.tolist(),int)):
                return 'must be an integer'

        def _validate_bias(self, bias):
            if not isinstance(bias, bool):
                return 'must be a boolean'

        def _validate_input_weights_learning_rate(self, rate):
            if not isinstance(rate, (float, bool)):
                return 'must be a float or a boolean'

        def _validate_hidden_weights_learning_rate(self, rate):
            if not isinstance(rate, (float, bool)):
                return 'must be a float or a boolean'

        def _validate_input_biases_learning_rate(self, rate):
            if not isinstance(rate, (float, bool)):
                return 'must be a float or a boolean'

        def _validate_hidden_biases_learning_rate(self, rate):
            if not isinstance(rate, (float, bool)):
                return 'must be a float or a boolean'

    @check_user_specified
    def __init__(self,
                 input_size=None,
                 hidden_size=None,
                 bias=None,
                 # num_layers:int=1,
                 # batch_first:bool=False,
                 # dropout:float=0.0,
                 # bidirectional:bool=False,
                 enable_learning:bool=True,
                 learning_rate:float=None,
                 optimizer_params:dict=None,
                 random_state=None,
                 seed=None,
                 name="GRU Composition",
                 **kwargs):

        # Instantiate Composition -------------------------------------------------------------------------

        super().__init__(name=name,
                         input_size=input_size,
                         hidden_size=hidden_size,
                         bias=bias,
                         # num_layers=num_layers,
                         # batch_first=batch_first,
                         # dropout=dropout,
                         # bidirectional=bidirectional,
                         enable_learning=enable_learning,
                         learning_rate=learning_rate,
                         optimizer_params=optimizer_params,
                         random_state = random_state,
                         seed = seed,
                         **kwargs
                         )

        input_size = self.input_size
        hidden_size = self.hidden_size

        self._construct_pnl_composition(input_size, hidden_size,
                                    context = Context(source=ContextFlags.COMMAND_LINE, string='FROM GRU'))

        self._assign_gru_specific_attributes(input_size, hidden_size)


    # *****************************************************************************************************************
    # ******************************  Nodes and Pathway Construction Methods  *****************************************
    # *****************************************************************************************************************
    #region
    # Construct Nodes --------------------------------------------------------------------------------

    def _construct_pnl_composition(self, input_size, hidden_size, context):
        """Construct Nodes and Projections for GRUComposition"""
        hidden_shape = np.ones(hidden_size)

        self.input_node = ProcessingMechanism(name=INPUT_NODE,
                                              input_shapes=input_size)

        # Two input_ports are used to separately gate input its recurrent Projection and from new_node
        # LinearCombination function of each InputPort is explicitly specified to allow for gating by a vector
        self.hidden_layer_node = ProcessingMechanism(name=HIDDEN_LAYER,
                                                     input_shapes=[hidden_size, hidden_size],
                                                     input_ports=[
                                                         InputPort(name='NEW INPUT',
                                                                   function=LinearCombination(scale=hidden_shape)),
                                                         InputPort(name='RECURRENT',
                                                                   function=LinearCombination(scale=hidden_shape))],
                                                     function=LinearCombination(operation=SUM))

        # Two input_ports are used to allow the input from the hidden_layer_node to be gated but not the input_node
        # The node's LinearCombination function is then used to combine the two inputs
        # And then Tanh is assigend as the function of the OutputPort to do the nonlinear transform
        self.new_node = ProcessingMechanism(name=NEW_NODE,
                                            input_shapes=[hidden_size, hidden_size],
                                            input_ports=['FROM INPUT',
                                                         InputPort(name='FROM HIDDEN',
                                                                   function=LinearCombination(scale=hidden_shape))],
                                            function=LinearCombination,
                                            output_ports=[OutputPort(name='TO HIDDEN LAYER INPUT',
                                                                     function=Tanh)])

        # Gates input to hidden_layer_node from its recurrent Projection and from new_node
        self.update_node = GatingMechanism(name=UPDATE_NODE,
                                           default_allocation=hidden_shape,
                                           function=Logistic,
                                           gating_signals=[
                                               GatingSignal(name='RECURRENT GATING SIGNAL',
                                                            default_allocation=hidden_shape,
                                                            gate=self.hidden_layer_node.input_ports['RECURRENT']),
                                               GatingSignal(name='NEW GATING SIGNAL',
                                                            default_allocation=hidden_shape,
                                                            transfer_function=Linear(scale=-1,offset=1),
                                                            gate=self.hidden_layer_node.input_ports['NEW INPUT'])])
        self.new_gate = self.update_node.gating_signals['NEW GATING SIGNAL'].efferents[0]
        self.new_gate.name = 'NEW GATE'
        self.recurrent_gate = self.update_node.gating_signals['RECURRENT GATING SIGNAL'].efferents[0]
        self.recurrent_gate.name = 'RECURRENT GATE'

        self.reset_node = GatingMechanism(name=RESET_NODE,
                                          default_allocation=hidden_shape,
                                          function=Logistic,
                                          gating_signals=[
                                              GatingSignal(name='RESET GATING SIGNAL',
                                                           default_allocation=hidden_shape,
                                                           gate=self.new_node.input_ports['FROM HIDDEN'])])
        self.reset_gate = self.reset_node.gating_signals['RESET GATING SIGNAL'].efferents[0]
        self.reset_gate.name = 'RESET GATE'

        self.output_node = ProcessingMechanism(name=OUTPUT_NODE,
                                               input_shapes=hidden_size,
                                               function=Linear)

        self.add_nodes([self.input_node, self.new_node, self.reset_node,
                        self.update_node, self.output_node, self.hidden_layer_node],
                       context=context)

        def init_wts(sender_size, receiver_size):
            """Initialize weights for Projections"""
            sqrt_val = np.sqrt(hidden_size)
            return np.random.uniform(-sqrt_val, sqrt_val, (sender_size, receiver_size))

        # Learnable: wts_in, wts_iu, wts_ir, wts_hn, wts_hu,, wts_hr
        self.wts_in = MappingProjection(name='INPUT TO NEW WEIGHTS',
                                        sender=self.input_node,
                                        receiver=self.new_node.input_ports['FROM INPUT'],
                                        learnable=True,
                                        matrix=init_wts(input_size, hidden_size))

        self.wts_iu = MappingProjection(name='INPUT TO UPDATE WEIGHTS',
                                        sender=self.input_node,
                                        receiver=self.update_node.input_ports[OUTCOME],
                                        learnable=True,
                                        matrix=init_wts(input_size, hidden_size))

        self.wts_ir = MappingProjection(name='INPUT TO RESET WEIGHTS',
                                        sender=self.input_node,
                                        receiver=self.reset_node.input_ports[OUTCOME],
                                        learnable=True,
                                        matrix=init_wts(input_size, hidden_size))

        self.wts_nh = MappingProjection(name='NEW TO HIDDEN WEIGHTS',
                                        sender=self.new_node,
                                        receiver=self.hidden_layer_node.input_ports['NEW INPUT'],
                                        learnable=False,
                                        matrix=IDENTITY_MATRIX)

        self.wts_hh = MappingProjection(name='HIDDEN RECURRENT WEIGHTS',
                                        sender=self.hidden_layer_node,
                                        receiver=self.hidden_layer_node.input_ports['RECURRENT'],
                                        learnable=False,
                                        matrix=IDENTITY_MATRIX)

        self.wts_hn = MappingProjection(name='HIDDEN TO NEW WEIGHTS',
                                        sender=self.hidden_layer_node,
                                        receiver=self.new_node.input_ports['FROM HIDDEN'],
                                        learnable=True,
                                        matrix=init_wts(hidden_size, hidden_size))

        self.wts_hr = MappingProjection(name='HIDDEN TO RESET WEIGHTS',
                                        sender=self.hidden_layer_node,
                                        receiver=self.reset_node.input_ports[OUTCOME],
                                        learnable=True,
                                        matrix=init_wts(hidden_size, hidden_size))

        self.wts_hu = MappingProjection(name='HIDDEN TO UPDATE WEIGHTS',
                                        sender=self.hidden_layer_node,
                                        receiver=self.update_node.input_ports[OUTCOME],
                                        learnable=True,
                                        matrix=init_wts(hidden_size, hidden_size))

        self.wts_ho = MappingProjection(name='HIDDEN TO OUTPUT WEIGHTS',
                                        sender=self.hidden_layer_node,
                                        receiver=self.output_node,
                                        learnable=False,
                                        matrix=IDENTITY_MATRIX)

        self.learnable_projections = [self.wts_in, self.wts_iu, self.wts_ir,
                                      self.wts_hn, self.wts_hr, self.wts_hu]

        self.add_projections([self.wts_in, self.wts_iu, self.wts_ir, self.wts_nh,
                              self.wts_hh, self.wts_hn, self.wts_hr, self.wts_hu, self.wts_ho],
                             context=context)

        if self.bias:
            self.bias_in_node = ProcessingMechanism(name='BIAS NODE IN', default_variable=[1])
            self.bias_in = MappingProjection(name='BIAS IN',
                                             sender=self.bias_in_node,
                                             receiver=self.new_node.input_ports['FROM INPUT'],
                                             learnable=True)

            self.bias_iu_node = ProcessingMechanism(name='BIAS NODE IU', default_variable=[1])
            self.bias_iu = MappingProjection(name='BIAS IU',
                                             sender=self.bias_iu_node,
                                             receiver=self.update_node.input_ports[OUTCOME],
                                             learnable=True)

            self.bias_ir_node = ProcessingMechanism(name='BIAS NODE IR', default_variable=[1])
            self.bias_ir = MappingProjection(name='BIAS IR',
                                             sender=self.bias_ir_node,
                                             receiver=self.reset_node.input_ports[OUTCOME],
                                             learnable=True)

            self.bias_hn_node = ProcessingMechanism(name='BIAS NODE HN', default_variable=[1])
            self.bias_hn = MappingProjection(name='BIAS HN',
                                             sender=self.bias_hn_node,
                                             receiver=self.new_node.input_ports['FROM HIDDEN'],
                                             learnable=True)

            self.bias_hr_node = ProcessingMechanism(name='BIAS NODE HR', default_variable=[1])
            self.bias_hr = MappingProjection(name='BIAS HR',
                                             sender=self.bias_hr_node,
                                             receiver=self.reset_node.input_ports[OUTCOME],
                                             learnable=True)

            self.bias_hu_node = ProcessingMechanism(name='BIAS NODE HU', default_variable=[1])
            self.bias_hu = MappingProjection(name='BIAS HU',
                                             sender=self.bias_hu_node,
                                             receiver=self.update_node.input_ports[OUTCOME],
                                             learnable=True)

            self.add_nodes([(self.bias_ir_node, NodeRole.BIAS),
                            (self.bias_iu_node, NodeRole.BIAS),
                            (self.bias_in_node, NodeRole.BIAS),
                            (self.bias_hr_node, NodeRole.BIAS),
                            (self.bias_hu_node, NodeRole.BIAS),
                            (self.bias_hn_node, NodeRole.BIAS)],
                           context=Context(source=ContextFlags.COMMAND_LINE, string='FROM GRU')
                           )

            self.biases = [self.bias_ir, self.bias_iu, self.bias_in,
                                  self.bias_hr, self.bias_hu, self.bias_hn]
            self.add_projections(self.biases, context=context)

        self.scheduler.add_condition(self.update_node, conditions.AfterNodes(self.reset_node))
        self.scheduler.add_condition(self.new_node, conditions.AfterNodes(self.update_node))
        self.scheduler.add_condition(self.hidden_layer_node, conditions.AfterNodes(self.new_node))

        self._set_learning_attributes()

        self._analyze_graph()

    def _assign_gru_specific_attributes(self, input_size, hidden_size):
        for node in self.nodes:
            node.exclude_from_show_graph = True
        self.gru_mech = ProcessingMechanism(name=GRU_NODE,
                                            input_shapes=input_size,
                                            function=MatrixTransform(
                                                default_variable=np.zeros(input_size),
                                                matrix=get_matrix(FULL_CONNECTIVITY_MATRIX,input_size, hidden_size)))
        self._input_comp_nodes_to_pytorch_nodes_map = {self.input_node: self.gru_mech}
        self._trained_comp_nodes_to_pytorch_nodes_map = {self.output_node: self.gru_mech}
        self.target_node = ProcessingMechanism(default_variable = np.zeros_like(self.gru_mech.value),
                                               name= GRU_TARGET_NODE)

    def _set_learning_attributes(self):
        """Set learning-related attributes for Node and Projections
        """
        learning_rate = self.enable_learning

        for projection in self.learnable_projections:

            if self.enable_learning is False:
                projection.learnable = False
                continue

            if learning_rate is False:
                projection.learnable = False
                continue

            elif learning_rate is True:
                # Default (GRUComposition's learning_rate) is used for all field_weight Projections:
                learning_rate = self.learning_rate

            assert isinstance(learning_rate, (int, float)), \
                (f"PROGRAM ERROR: learning_rate for {projection.sender.owner.name} is not a valid value.")

            projection.learnable = True
            if projection.learning_mechanism:
                projection.learning_mechanism.learning_rate = learning_rate

    def get_weights(self, context=None):
        wts_ir = self.wts_ir.parameters.matrix.get(context)
        wts_iu = self.wts_iu.parameters.matrix.get(context)
        wts_in = self.wts_in.parameters.matrix.get(context)
        wts_hr = self.wts_hr.parameters.matrix.get(context)
        wts_hu = self.wts_hu.parameters.matrix.get(context)
        wts_hn = self.wts_hn.parameters.matrix.get(context)
        return wts_ir, wts_iu, wts_in, wts_hr, wts_hu, wts_hn
    #endregion

    @handle_external_context()
    def set_weights(self, weights:Union[list, np.ndarray], biases:Union[list, np.ndarray], context=None):
        """Set weights for Projections to input_node and hidden_layer_node."""

        # MODIFIED 2/16/25 NEW:
        # FIX: CHECK IF TORCH GRU EXISTS YET (CHECK FOR pytorch_representation != None; i.e., LEARNING HAS OCCURRED;
        #      IF SO, ADD CALL TO PytorchGRUPRojectionWrapper HELPER METHOD TO SET TORCH GRU PARAMETERS
        for wts, proj in zip(weights,
                       [self.wts_ir, self.wts_iu, self.wts_in, self.wts_hr, self.wts_hu, self.wts_hn]):
            valid_shape = self._get_valid_weights_shape(proj)
            assert wts.shape == valid_shape, \
                (f"PROGRAM ERROR: Shape of weights in 'weights' arg of '{self.name}.set_weights' "
                 f"({wts.shape}) does not match required shape ({valid_shape}).)")
            proj.parameters.matrix._set(wts, context)
            proj.parameter_ports['matrix'].parameters.value._set(wts, context)
        # MODIFIED 3/11/25 END

        if biases:
            for torch_bias, pnl_bias in zip(biases, [self.bias_ir, self.bias_iu, self.bias_in,
                                                     self.bias_hr, self.bias_hu, self.bias_hn]):
                valid_shape = self._get_valid_weights_shape(pnl_bias)
                assert torch_bias.shape == valid_shape, \
                    (f"PROGRAM ERROR: Shape of biases in 'bias' arg of '{self.name}.set_weights' "
                     f"({torch_bias.shape}) does not match required shape ({valid_shape}).")
                pnl_bias.parameters.matrix._set(torch_bias, context)
                pnl_bias.parameter_ports['matrix'].parameters.value._set(torch_bias, context)

    @handle_external_context()
    def infer_backpropagation_learning_pathways(self, execution_mode, context=None)->list:
        if execution_mode is not pnlvm.ExecutionMode.PyTorch:
            raise GRUCompositionError(f"Learning in {self.componentCategory} "
                                      f"is not supported for {execution_mode.name}.")

        # Create Mechanism the function fo which will be the Pytorch GRU module
        # Note:  function is a placeholder, to induce proper variable and value dimensions;
        #        will be replaced by PyTorch GRU function in PytorchGRUMechanismWrapper
        target_mech = self.target_node

        # Add target Node to GRUComposition
        self.add_node(target_mech, required_roles=[NodeRole.TARGET, NodeRole.LEARNING],
                      context=Context(source=ContextFlags.METHOD, string='FROM GRU'))
        self.exclude_node_roles(target_mech, NodeRole.OUTPUT, context)

        for output_port in target_mech.output_ports:
            output_port.parameters.require_projection_in_composition.set(False, override=True)
        self.targets_from_outputs_map = {target_mech: self.gru_mech}
        self.outputs_to_targets_map = {self.gru_mech: target_mech}

        return [target_mech]

    def _get_pytorch_backprop_pathway(self, input_node, context)->list:
        return [[self.gru_mech]]

    # *****************************************************************************************************************
    # *********************************** Execution Methods  **********************************************************
    # *****************************************************************************************************************
    #region

    def _get_execution_mode(self, execution_mode):
        """Parse execution_mode argument and return a valid execution mode for the learn() method"""
        if execution_mode is None:
            if self.execution_mode_warned_about_default is False:
                warnings.warn(f"The execution_mode argument was not specified in the learn() method of {self.name}; "
                              f"ExecutionMode.PyTorch will be used by default.")
                self.execution_mode_warned_about_default = True
            execution_mode = ExecutionMode.PyTorch
        return execution_mode

    def _add_dependency(self,
                        sender:ProcessingMechanism,
                        projection:MappingProjection,
                        receiver:ProcessingMechanism,
                        dependency_dict:dict,
                        queue:deque,
                        comp:AutodiffComposition):
        """Override to implement direct pathway through gru_mech for pytorch backprop pathway.
        """
        # FIX: 3/9/25 CLEAN THIS UP: WRT ASSIGNMENT OF _pytorch_projections BELOW:
        if self._pytorch_projections:
            assert len(self._pytorch_projections) == 2, \
                (f"PROGRAM ERROR: {self.name}._pytorch_projections should have only two Projections, but has "
                 f"{len(self._pytorch_projections)}: {' ,'.join([proj.name for proj in self._pytorch_projections])}.")
            direct_proj_in = self._pytorch_projections[0]
            direct_proj_out = self._pytorch_projections[1]

        else:
            try:
                direct_proj_in = MappingProjection(name="Projection to GRU COMP",
                                                   sender=sender,
                                                   receiver=self.gru_mech,
                                                   learnable=projection.learnable)
                self._pytorch_projections.append(direct_proj_in)
            except DuplicateProjectionError:
                assert False, "PROGRAM ERROR: Duplicate Projection to GRU COMP"

            try:
                direct_proj_out = MappingProjection(name="Projection from GRU COMP",
                                                    sender=self.gru_mech,
                                                    receiver=self.output_CIM,
                                                    # receiver=self.output_CIM.input_ports[0],
                                                    learnable=False)
                self._pytorch_projections.append(direct_proj_out)
            except DuplicateProjectionError:
                assert False, "PROGRAM ERROR: Duplicate Projection to GRU COMP"

        # FIX: GET ALL EFFERENTS OF OUTPUT NODE HERE
        # output_node = self.output_CIM.output_port.efferents[0].receiver.owner
        # output_node = self.output_CIM.output_port
        output_node = self.output_CIM

        # GRU pathway:
        dependency_dict[direct_proj_in]=sender
        dependency_dict[self.gru_mech]=direct_proj_in
        dependency_dict[direct_proj_out]=self.gru_mech
        dependency_dict[output_node]=direct_proj_out

        # FIX : ADD ALL EFFERENTS OF OUTPUT NODE HERE:
        queue.append((self.gru_mech, self))

    def _identify_target_nodes(self, context):
        return [self.gru_mech]

    def add_node(self, node, required_roles=None, context=None):
        """Override if called from command line to disallow modification of GRUComposition"""
        if context is None:
            raise CompositionError(f"Nodes cannot be added to a {self.componentCategory}: ('{self.name}').")
        super().add_node(node, required_roles, context)

    def add_projection(self, *args, **kwargs):
        """Override if called from command line to disallow modification of GRUComposition"""
        if CONTEXT not in kwargs or kwargs[CONTEXT] is None:
            raise CompositionError(f"Projections cannot be added to a {self.componentCategory}: ('{self.name}'.")
        return super().add_projection(*args, **kwargs)
