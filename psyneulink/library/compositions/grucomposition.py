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
     - `Organization <GRUComposition_Organization>`
     - `Operation <GRUComposition_Operation>`
  * `GRUComposition_Creation`
     - `Learning <GRUComposition_Learning>`
  * `GRUComposition_Structure`
     - `Input <GRUComposition_Input>`
     - `Hidden Layer <GRUComposition_Hidden_Layer>`
     - `Output <GRUComposition_Output>`
  * `GRUComposition_Execution`
     - `Processing <GRUComposition_Processing>`
     - `Learning <GRUComposition_Training>`
  * `GRUComposition_Examples`
  * `GRUComposition_Class_Reference`

.. _GRUComposition_Overview:

Overview
--------

The GRUComposition a subclass of `AutodiffComposition` that implements a single-layered gated recurrent network,
which combines a `RecurrentTransferMechanism` with a set of `GatingMechanisms <GatingMechanism>` that modulate
the flow of information through the RecurrentTransferMechanism.  This corresponds to the `PyTorch GRUNetwork
<https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_, which is used to implement it when its `learn
<GRUComposition.learn>` method is called with `execution_mode <GRUComposition.execution_mode>` set to *PyTorch*

COMMENT:
FIX: ADD EXPLANATION OF ITS RELATIONSHIP TO PyTorch GRUCell
COMMENT
The GRUComposition implements the following computations by its `reset <GRUComposition.reset_node>`, `update
<GRUComposition.update_node>`, `new <GRUComposition.new_node>`, and `hidden <GRUComposition.hidden_layer_node>`
`Nodes <Composition_Nodes>`, corresponding to the terms of the function in the `PyTorch
<https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ module:

.. math::

   &reset = Logistic(wts\_ir \cdot input + bias\_ir + wts\\_hr \cdot hidden + bias\_hr)

   &update = Logistic(wts\_iu \cdot input + bias\_iu + wts\_hu \cdot hidden + bias\_hu)

   &new = Tanh(wts\_in \cdot input + bias\_in + reset \cdot (wts\_hn \cdot hidden + bias\_hn))

   &hidden = (1 - update) \odot new + update \odot hidden

where :math:`\cdot` is the dot product, :math:`\odot` is the Hadamard product, and all values are for the
current execution of the Composition *(t)* except for hidden, which uses the value from the prior execution *(t-1)*
(see `Cycles <Composition_Cycle>` for handling of recurrence and cycles).

.. _GRUComposition_Organization:

**Organization**


COMMENT:
`reset <GRUComposition.reset_gate>` = `Logistic`\\(`wts_ir <GRUComposition.wts_ir>` *
`input <GRUComposition.input_node>` + `bias_ir <GRUComposition.bias_ir>` + `bias_ir <GRUComposition.bias_ir>` +
`wts_hr <GRUComposition.wts_hr>` * `hidden_layer <GRUComposition.hidden_layer_node>` +
`bias_hr <GRUComposition.bias_hr>`

`update <GRUComposition.update_gate>`\\(t) = `Logistic`(`wts_iu <GRUComposition.wts_iu>` *
`input <GRUComposition.input_node>` + `bias_iu <GRUComposition.bias_iu>` + `wts_hu <GRUComposition.wts_hu>` *
`hidden_layer <GRUComposition.hidden_layer_node>`\\(t-1) + `bias_hu <GRUComposition.bias_hu>`

`new <GRUComposition.new_node>`\\(t) = :math:`tanh`(`wts_in <GRUComposition.wts_in>` *
`input <GRUComposition.input_node>` + `bias_in <GRUComposition.bias_in>` +
`reset <GRUComposition.reset_gate>`\\(t) * (`wts_hn <GRUComposition.wts_hn>` *
`hidden_layer <GRUComposition.hidden_layer_node>`\\(t-1) + `bias_hn <GRUComposition.bias_hn>`)

`hidden_layer <GRUComposition.hidden_layer_node>`\\(t) = (1 - `update <GRUComposition.update_gate>`\\(t)) *
`new <GRUComposition.new_node>`\\(t) + `update <GRUComposition.update_gate>`\\(t) * `hidden_layer
<GRUComposition.hidden_layer_node>`\\(t-1)


where:
    r(t) = reset gate
    z(t) = update gate
    n(t) = new gate
    h(t) = hidden layer
    x(t) = input
    W_ir, W_iz, W_in, W_hr, W_hz, W_hn = input, update, and reset weights
    b_ir, b_iz, b_in, b_hr, b_hz, b_hn = input, update, and reset biases
COMMENT

.. _GRUComposition_Operation:

**Operation**


.. _GRUComposition_Creation:

Creation
--------

An GRUComposition is created by calling its constructor.  There are four major elements that can be configured:


.. _GRUComposition_Learning:

*Learning*
~~~~~~~~~~


.. _GRUComposition_Structure:

Structure
---------

.. figure:: _static/GRUComposition_fig.svg
   :alt: GRU Composition
   :width: 400
   :align: center

   **Structure of a GRUComposition** -- can be seen in more detail using the Composition's s `show_graph
   <ShowGraph.show_graph>` method with its **show_node_structure** argument set to ``True`` or ``ALL``.

.. _GRUComposition_Input:

*Input*
~~~~~~~

The inputs corresponding to each key and value field are represented as `INPUT <NodeRole.INPUT>` `Nodes
<Composition_Nodes>` of the GRUComposition, listed in its `query_input_nodes <GRUComposition.query_input_nodes>`
and `value_input_nodes <GRUComposition.value_input_nodes>` attributes, respectively,

.. _GRUComposition_Hidden_Layer:

*Hidden Layer*
~~~~~~~~~~~~~~


.. _GRUComposition_Output:

*Output*
~~~~~~~~


.. _GRUComposition_Execution:

Execution
---------


.. _GRUComposition_Processing:

*Processing*
~~~~~~~~~~~~


.. _GRUComposition_Training:

*Training*
~~~~~~~~~~

If `learn <Composition.learn>` is called, `enable_learning <GRUComposition.enable_learning>` is True, then errors
will be computed for

.. _GRUComposition_Examples:

Examples
--------

The following are examples of how to configure and initialize a GRUComposition:


.. _GRUComposition_Class_Reference:

Class Reference
---------------
"""
import numpy as np
import warnings
from enum import Enum

from sympy.stats import Logistic

import psyneulink.core.scheduling.condition as conditions

from psyneulink._typing import Optional, Union
from psyneulink.core.components.functions.nonstateful.transformfunctions import LinearCombination
from psyneulink.core.components.functions.nonstateful.transferfunctions import Linear, Logistic, Tanh
from psyneulink.core.components.functions.function import DEFAULT_SEED, _random_state_getter, _seed_setter
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.compositions.composition import CompositionError, NodeRole
from psyneulink.library.compositions.autodiffcomposition import AutodiffComposition, torch_available
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.modulatory.control.gating.gatingmechanism import GatingMechanism
from psyneulink.core.components.ports.modulatorysignals.gatingsignal import GatingSignal
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.context import handle_external_context
from psyneulink.core.globals.keywords import CONTEXT, GRU_COMPOSITION, PRODUCT, SUM, IDENTITY_MATRIX
from psyneulink.core.globals.utilities import \
    ContentAddressableList, convert_all_elements_to_np_array, is_numeric_scalar
from psyneulink.core.llvm import ExecutionMode


__all__ = ['GRUComposition', 'GRUCompositionError']


# Node names
INPUT_NODE_NAME = 'INPUT'
HIDDEN_LAYER_NODE_NAME = 'HIDDEN\nLAYER'
RESET_NODE_NAME = 'RESET'
UPDATE_NODE_NAME = 'UPDATE'
NEW_NODE_NAME = 'NEW'
OUTPUT_NODE_NAME = 'OUTPUT'


class GRUCompositionError(CompositionError):
    def __init__(self, error_value):
        self.error_value = error_value
    def __str__(self):
        return repr(self.error_value)


class GRUComposition(AutodiffComposition):
    """
    GRUComposition(                      \
        name="GRU_Composition"           \
        )

    Subclass of `AutodiffComposition` that implements a single-layered gated recurrent network.

    Takes the following arguments:

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

    learning_rate : float : default .01
        specifies the default learning_rate for `field_weights <GRUComposition.field_weights>` not
        specified in `learn_field_weights <GRUComposition.learn_field_weights>` (see `learning_rate
        <GRUComposition_Field_Weights_Learning>` for additional details).

    enable_learning : bool : default True
        specifies whether learning is enabled for the GRUComposition (see `Learning <GRUComposition_Learning>`
        for additional details)


    Attributes
    ----------

    input_size : int
        determines the length of the input array to the GRUComposition and size of the `input_node
        <GRUComposition.input_node>`.

    hidden_size : int
        determines the size of the `hidden_layer_node` and all other `INTERNAL` `Nodes <Composition_Nodes>`
        of the GRUComposition.

    bias=True : bool
        determines whether the GRUComposition uses bias vectors in its computations.

    COMMENT:
    num_layers : int : default 1
     batch_first : bool : default False
     dropout : float : default 0.0
     bidirectional : bool : default False
    COMMENT

    learning_rate : float
        determines the default learning_rate for `field_weights <GRUComposition.field_weights>`
        not specified in `learn_field_weights <GRUComposition.learn_field_weights>`
        (see `learning_rate <GRUComposition_Field_Weights_Learning>` for additional details).

    enable_learning : bool
        determines whether learning is enabled for the GRUComposition
        (see `Learning <GRUComposition_Learning>` for additional details).

    input_node : ProcessingMechanism
        `INPUT <NodeRole.INPUT>` `Node <Composition_Nodes>` that receives the input to the GRUComposition and passes
        it to the `hidden_layer_node <GRUComposition.hidden_layer_node>`; corresponds to input *(i)* of the `PyTorch
        <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ implementation.

    new_node : ProcessingMechanism
        `ProcessingMechanism` that provides the `hidden_layer_node <GRUComposition.hidden_layer_node>` with the input
        from the `input_node <GRUComposition.input_node>`, gated by the `reset_node <GRUComposition.reset_node>`;
        corresponds to new gate *(n)* of the `PyTorch <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_
        implementation.

    hidden_layer_node : ProcessingMechanism
        `ProcessingMechanism` that implements the recurrent layer of the GRUComposition; corresponds to
        hidden layer *(h)* of the `PyTorch <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_
        implementation.

    reset_node : GatingMechanism
        `GatingMechanism` that that gates the input to the `new_node <GRUComposition.new_node>`; corresponds to reset
        gate *(r)* of the `PyTorch <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ implementation.

    update_node : GatingMechanism
        `GatingMechanism` that gates the inputs to the hidden layer from the `new_node <GRUComposition.new_node>`
        and the prior state of the `hidden_layer_node <GRUComposition.hidden_layer_node>` itself (i.e., the input
        it receives from its recurrent Projection); corresponds to update gate *(z)* of the `PyTorch
        <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ implementation.

    output_node : ProcessingMechanism
        `OUTPUT <NodeRole.INPUT>` `Node <Composition_Nodes>` that receives the output of the `hidden_layer_node
        <GRUComposition.hidden_layer_node>`; corresponds to result of the `PyTorch
        <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ implementation.

    wts_in : MappingProjection
        `MappingProjection` with learnable `matrix <MappingProjection.matrix>` ("connection weights") that projects
        from the `input_node <GRUComposition.input_node>` to the `new_node <GRUComposition.new_node>`.

    wts_nh : MappingProjection
        `MappingProjection` with learnable `matrix <MappingProjection.matrix>` ("connection weights") that projects
        from the `new_node <GRUComposition.new_node>` to the `hidden_layer_node <GRUComposition.hidden_layer_node>`.

    wts_hh : MappingProjection
        `MappingProjection` with fixed `matrix <MappingProjection.matrix>` ("connection weights") that projects
        from the `hidden_layer_node <GRUComposition.hidden_layer_node>` to itself (i.e., the recurrent Projection).

    wts_ho : MappingProjection
        `MappingProjection` with fixed `matrix <MappingProjection.matrix>` ("connection weights") that projects from
        the `hidden_layer_node <GRUComposition.hidden_layer_node>` to the `output_node <GRUComposition.output_node>`.

    wts_iu : MappingProjection
        `MappingProjection` with learnable `matrix <MappingProjection.matrix>` ("connection weights") that projects
        from the `input_node <GRUComposition.input_node>` to the `update_node <GRUComposition.update_node>`.

    wts_ir : MappingProjection
        `MappingProjection` with learnable `matrix <MappingProjection.matrix>` ("connection weights") that projects
        from the `input_node <GRUComposition.input_node>` to the `reset_node <GRUComposition.reset_node>`.

    reset_gate : GatingProjection
        `GatingProjection` that gates the input to the `new_node <GRUComposition.new_node>` from the `input_node
        <GRUComposition.input_node>`; its `value <GatingProjection.value>` is used in the Hadamard product with
        the input to produce the new (external) input to the `hidden_layer_node <GRUComposition.hidden_layer_node>`.

    new_gate : GatingProjection
        `GatingProjection` that gates the input to the `hidden_layer_node <GRUComposition.hidden_layer_node>` from the
        `new_node <GRUComposition.new_node>`; its `value <GatingProjection.value>` is used in the Hadamard product
        with the (external) input to the `hidden_layer_node <GRUComposition.hidden_layer_node>` from the `new_node
        <GRUComposition.new_node>`, which determines how much of the `hidden_layer_node
        <GRUComposition.hidden_layer_node>`\\'s new state is determined by the external input vs. its prior state.

    recurrent_gate : GatingProjection
        `GatingProjection` that gates the input to the `hidden_layer_node <GRUComposition.hidden_layer_node>` from its
        recurrent projection (`wts_hh <GRUComposition.wts_hh>`); its `value <GatingProjection.value>` is used in the
        in the Hadamard product with the recurrent input to the `hidden_layer_node <GRUComposition.hidden_layer_node>`,
        which determines how much of the `hidden_layer_node <GRUComposition.hidden_layer_node>`\\'s
        new state is determined by its prior state vs.its external input.
    """

    componentCategory = GRU_COMPOSITION

    if torch_available:
        from psyneulink.library.compositions.pytorchGRUcompositionwrapper import PytorchGRUCompositionWrapper
        pytorch_composition_wrapper_type = PytorchGRUCompositionWrapper


    class Parameters(AutodiffComposition.Parameters):
        """
            Attributes
            ----------

                enable_learning
                    see `enable_learning <GRUComposition.enable_learning>`

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
        learning_rate = Parameter(.001, modulable=True)
        enable_learning = Parameter(True, structural=True)
        random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies='seed')
        seed = Parameter(DEFAULT_SEED(), modulable=True, setter=_seed_setter)

    @check_user_specified
    def __init__(self,
                 input_size:int=1,
                 hidden_size:int=1,
                 bias:bool=False,
                 # num_layers:int=1,
                 # batch_first:bool=False,
                 # dropout:float=0.0,
                 # bidirectional:bool=False,
                 learning_rate:float=None,
                 enable_learning:bool=True,
                 random_state=None,
                 seed=None,
                 name="GRU Composition",
                 **kwargs):

        # Instantiate Composition -------------------------------------------------------------------------

        super().__init__(name=name,
                         input_size=input_size,
                         hidden_size=hidden_size,
                         # bias=bias,
                         # num_layers=num_layers,
                         # batch_first=batch_first,
                         # dropout=dropout,
                         # bidirectional=bidirectional,
                         learning_rate = learning_rate,
                         enable_learning = enable_learning,
                         random_state = random_state,
                         seed = seed,
                         **kwargs
                         )

        self._construct_composition(input_size, hidden_size)

        # if torch_available:
        #     from psyneulink.library.compositions.pytorchGRUCompositionwrapper import PytorchGRUCompositionWrapper
        #     self.pytorch_composition_wrapper_type = PytorchGRUCompositionWrapper

        # Final Configuration and Clean-up ---------------------------------------------------------------------------


    # *****************************************************************************************************************
    # ******************************  Nodes and Pathway Construction Methods  *****************************************
    # *****************************************************************************************************************
    # Construct Nodes --------------------------------------------------------------------------------

    def _construct_composition(self, input_size, hidden_size):
        """Construct Nodes and Projections for GRUComposition"""
        hidden_shape = np.ones(hidden_size)

        self.input_node = ProcessingMechanism(name=INPUT_NODE_NAME,
                                              input_shapes=input_size)

        self.hidden_layer_node = ProcessingMechanism(name=HIDDEN_LAYER_NODE_NAME,
                                                     input_shapes=[hidden_size, hidden_size],
                                                     input_ports=[
                                                         InputPort(name='NEW INPUT',
                                                                   function=LinearCombination(scale=hidden_shape)),
                                                         InputPort(name='RECURRENT',
                                                                   function=LinearCombination(scale=hidden_shape))],
                                                     function=LinearCombination(operation=SUM))

        self.new_node = ProcessingMechanism(name=NEW_NODE_NAME,
                                            input_shapes=[hidden_size, hidden_size],
                                            input_ports=['FROM INPUT',
                                                         InputPort(
                                                             name='FROM HIDDEN',
                                                             function=LinearCombination(scale=hidden_shape))],
                                            function=LinearCombination,
                                            output_ports=[OutputPort(name='TO HIDDEN LAYER INPUT',
                                                                     function=Tanh)])

        self.update_node = GatingMechanism(name=UPDATE_NODE_NAME,
                                           default_allocation=hidden_shape,
                                           function=Logistic,
                                           gating_signals=[
                                               GatingSignal(name='NEW GATING SIGNAL',
                                                            default_allocation=hidden_shape,
                                                            transfer_function=Linear(scale=-1,offset=1),
                                                            gate=self.hidden_layer_node.input_ports['NEW INPUT']),
                                               GatingSignal(name='RECURRENT GATING SIGNAL',
                                                            default_allocation=hidden_shape,
                                                            gate=self.hidden_layer_node.input_ports['RECURRENT'])])
        self.new_gate = self.update_node.gating_signals['NEW GATING SIGNAL'].efferents[0]
        self.new_gate.name = 'NEW GATE'
        self.recurrent_gate = self.update_node.gating_signals['RECURRENT GATING SIGNAL'].efferents[0]
        self.recurrent_gate.name = 'RECURRENT GATE'

        self.reset_node = GatingMechanism(name=RESET_NODE_NAME,
                                          default_allocation=hidden_shape,
                                          function=Logistic,
                                          gating_signals=[
                                              GatingSignal(
                                                  name='RESET GATING SIGNAL',
                                                  default_allocation=hidden_shape,
                                                  gate=self.new_node.input_ports['FROM HIDDEN'])])
        self.reset_gate = self.reset_node.gating_signals['RESET GATING SIGNAL'].efferents[0]
        self.reset_gate.name = 'RESET GATE'

        self.output_node = ProcessingMechanism(name=OUTPUT_NODE_NAME,
                                               input_shapes=hidden_size,
                                               function=Linear)

        self.add_nodes([self.input_node,
                        self.new_node,
                        self.reset_node,
                        self.update_node,
                        self.output_node,
                        self.hidden_layer_node])

        def init_wts(sender_size, receiver_size):
            """Initialize weights for Projections"""
            sqrt_val = np.sqrt(hidden_size)
            return np.random.uniform(-sqrt_val, sqrt_val, (sender_size, receiver_size))

        # Learnable: wts_in, wts_iu, wts_ir, wts_hn, wts_hu,, wts_hr
        self.wts_in = MappingProjection(name='NEW INPUT WEIGHTS',
                                        sender=self.input_node,
                                        receiver=self.new_node.input_ports['FROM INPUT'],
                                        learnable=True,
                                        matrix=init_wts(input_size, hidden_size))

        self.wts_iu = MappingProjection(name='UPDATE INPUT WEIGHTS',
                                        sender=self.input_node,
                                        receiver=self.update_node.input_ports['OUTCOME'],
                                        learnable=True,
                                        matrix=init_wts(input_size, hidden_size))

        self.wts_ir = MappingProjection(name='RESET INPUT WEIGHTS',
                                        sender=self.input_node,
                                        receiver=self.reset_node.input_ports['OUTCOME'],
                                        learnable=True,
                                        matrix=init_wts(input_size, hidden_size))


        self.wts_nh = MappingProjection(name='HIDDEN INPUT WEIGHTS',
                                        sender=self.new_node,
                                        receiver=self.hidden_layer_node.input_ports['NEW INPUT'],
                                        learnable=False,
                                        matrix=init_wts(hidden_size, hidden_size))

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
                                        receiver=self.reset_node.input_ports['OUTCOME'],
                                        learnable=True,
                                        matrix=init_wts(hidden_size, hidden_size))

        self.wts_hu = MappingProjection(name='HIDDEN TO UPDATE WEIGHTS',
                                        sender=self.hidden_layer_node,
                                        receiver=self.update_node.input_ports['OUTCOME'],
                                        learnable=True,
                                        matrix=init_wts(hidden_size, hidden_size))

        self.wts_ho = MappingProjection(name='HIDDEN OUTPUT WEIGHTS',
                                        sender=self.hidden_layer_node,
                                        receiver=self.output_node,
                                        learnable=False,
                                        matrix=IDENTITY_MATRIX)

        self.add_projections([self.wts_in,
                              self.wts_iu,
                              self.wts_ir,
                              self.wts_nh,
                              self.wts_hh,
                              self.wts_hn,
                              self.wts_hr,
                              self.wts_hu,
                              self.wts_ho])
        self._analyze_graph()
    #region

    def _set_learning_attributes(self):
        """Set learning-related attributes for Node and Projections
        """
        for projection in self.projections:

            projection_is_field_weight = projection.sender.owner in self.field_weight_nodes

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

    #endregion

    # *****************************************************************************************************************
    # *********************************** Execution Methods  **********************************************************
    # *****************************************************************************************************************
    # region
    def execute(self,
                inputs=None,
                context=None,
                **kwargs):
        """Set input to weights of Projections to match_nodes and retrieved_nodes if not use_storage_node."""
        results = super().execute(inputs=inputs, context=context, **kwargs)
        # if not self._use_storage_node:
        #     self._store_memory(inputs, context)
        return results


    @handle_external_context()
    def learn(self, *args, **kwargs)->list:
        """Override to check for inappropriate use of ARG_MAX or PROBABILISTIC options for retrieval with learning"""
        enable_learning = self.parameters.enable_learning.get(kwargs[CONTEXT])

        return super().learn(*args, **kwargs)

    def _get_execution_mode(self, execution_mode):
        """Parse execution_mode argument and return a valid execution mode for the learn() method"""
        if execution_mode is None:
            if self.execution_mode_warned_about_default is False:
                warnings.warn(f"The execution_mode argument was not specified in the learn() method of {self.name}; "
                              f"ExecutionMode.PyTorch will be used by default.")
                self.execution_mode_warned_about_default = True
            execution_mode = ExecutionMode.PyTorch
        return execution_mode

    def _identify_target_nodes(self, context)->list:
        """Identify retrieval_nodes specified by **target_field_weights** as TARGET nodes"""
        target_fields = self.target_fields
        if target_fields is False:
            if self.enable_learning:
                warnings.warn(f"The 'enable_learning' arg for {self.name} is True "
                              f"but its 'target_fields' is False, so enable_learning will have no effect.")
            target_nodes = []
        elif target_fields is True:
            target_nodes = [node for node in self.retrieved_nodes]
        elif isinstance(target_fields, list):
            target_nodes = [node for node in self.retrieved_nodes if target_fields[self.retrieved_nodes.index(node)]]
        else:
            assert False, (f"PROGRAM ERROR: target_fields arg for {self.name}: {target_fields} "
                           f"is neither True, False nor a list of bools as it should be.")
        super()._identify_target_nodes(context)
        return target_nodes

    def infer_backpropagation_learning_pathways(self, execution_mode, context=None):
        if self.concatenate_queries:
            raise GRUCompositionError(f"GRUComposition does not support learning with 'concatenate_queries'=True.")
        return super().infer_backpropagation_learning_pathways(execution_mode, context=context)

    def do_gradient_optimization(self, retain_in_pnl_options, context, optimization_num=None):
        # 7/10/24 - MAKE THIS CONTEXT DEPENDENT:  CALL super() IF BEING EXECUTED ON ITS OWN?
        pass

    #endregion

