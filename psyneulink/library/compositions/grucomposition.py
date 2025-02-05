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
<GRUComposition.learn>` method is colled with `execution_mode <GRUComposition.execution_mode>` set to *PyTorch*


COMMENT:
FIX: ADD EXPLANATION OF ITS RELATIONSHIP TO PyTorch GRUCell
COMMENT

.. _GRUComposition_Organization:

**Organization**



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

import psyneulink.core.scheduling.condition as conditions

from psyneulink._typing import Optional, Union
from psyneulink.core.components.functions.nonstateful.transferfunctions import SoftMax
from psyneulink.core.components.functions.function import DEFAULT_SEED, _random_state_getter, _seed_setter
from psyneulink.core.compositions.composition import CompositionError, NodeRole
from psyneulink.library.compositions.autodiffcomposition import AutodiffComposition, torch_available
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.modulatory.control.gating.gatingmechanism import GatingMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.context import handle_external_context
from psyneulink.core.globals.keywords import GRU_COMPOSITION
from psyneulink.core.globals.utilities import \
    ContentAddressableList, convert_all_elements_to_np_array, is_numeric_scalar
from psyneulink.core.llvm import ExecutionMode


__all__ = ['GRUComposition', 'GRUCompositionError']


# Node names
INPUT_NODE_NAME = 'INPUT'
INPUT_AFFIX = f' [{INPUT_NODE_NAME}]'
HIDDEN_LAYER_NODE_NAME = 'HIDDEN LAYER'
HIDDEN_LAYER_AFFIX = f' [{HIDDEN_LAYER_NODE_NAME}]'
RESET_GATE_NAME = 'RESET GATE'
RESET_GATE_AFFIX = f' [{RESET_GATE_NAME}]'
UPDATE_GATE_NAME = 'UPDATE GATE'
UPDATE_GATE_AFFIX = f' [{UPDATE_GATE_NAME}]'
NEW_GATE_NAME = 'NEW GATE'
NEW_GATE_AFFIX = f' [{NEW_GATE_NAME}]'


class GRUCompositionError(CompositionError):
    def __init__(self, error_value):
        self.error_value = error_value
    def __str__(self):
        return repr(self.error_value)



    @property
    def nodes(self):
        """Return all Nodes assigned to the field."""
        return [node for node in
                [self.input_node,
                self.hidden_layer_node,
                self.reset_gate_node,
                self.update_gate_node,
                self.new_gate_node]
                if node is not None]

    @property
    def projections(self):
        """Return all Projections assigned to the field."""
        return [proj for proj in [self.input_to_hidden_projection,
                                  self.input_to_reset_gate_projection,
                                  self.input_to_update_gate_projection,
                                  self.input_to_new_gate_projection,
                                  self.reset_gating_projection,
                                  self.update_gating_projection,
                                  self.new_gating_projection,
                                  self.input_projection,
                                  self.hidden_layer_recurrent_projection]
                                  if proj is not None]


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
        specifies the size of the input layer.

    hidden_size : int : default 1
        specifies the size of the hidden layer.

    num_layers : int : default 1
        specifies the number of layers in the GRU.

    bias : bool : default True
        specifies whether or not to use a bias in the GRU.

    batch_first : bool : default False
        specifies whether the input and output tensors are provided as (batch, seq, feature).

    dropout : float : default 0.0
        specifies the dropout probability.

    bidrectional : bool : default False
        specifies whether the GRU is bidirectional.

    learning_rate : float : default .01
        specifies the default learning_rate for `field_weights <GRUComposition.field_weights>` not
        specified in `learn_field_weights <GRUComposition.learn_field_weights>` (see `learning_rate
        <GRUComposition_Field_Weights_Learning>` for additional details).

    enable_learning : bool : default True
        specifies whether learning is enabled for the EMCComposition (see `Learning <GRUComposition_Learning>`
        for additional details); **use_gating_for_weighting** must be False.


    Attributes
    ----------

    learning_rate : float
        determines the default learning_rate for `field_weights <GRUComposition.field_weights>`
        not specified in `learn_field_weights <GRUComposition.learn_field_weights>`
        (see `learning_rate <GRUComposition_Field_Weights_Learning>` for additional details).

    enable_learning : bool
        determines whether learning is enabled for the EMCComposition
        (see `Learning <GRUComposition_Learning>` for additional details).

    .. _GRUComposition_Nodes:

    input_node : list[ProcessingMechanism]
        `INPUT <NodeRole.INPUT>` `Node <Composition_Nodes>` that receives...

    bias_node : list[ProcessingMechanism]
        `BIAS <NodeRole.INPUT>` `Node <Composition_Nodes>` that receives...

    hidden_layer_node : list[ProcessingMechanism]
        `INPUT <NodeRole.INPUT>` `Node <Composition_Nodes>` that receives...

    reset_gate_node : list[ProcessingMechanism]
        `INPUT <NodeRole.INPUT>` `Node <Composition_Nodes>` that receives

    update_gate_node : list[ProcessingMechanism]
        `INPUT <NodeRole.INPUT>` `Node <Composition_Nodes>` that receives

    new_gate_node : list[ProcessingMechanism]
        `INPUT <NodeRole.INPUT>` ` <Composition_Nodes>` that receives

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
                 learning_rate:float=None,
                 enable_learning:bool=True,
                 random_state=None,
                 seed=None,
                 name="EM_Composition",
                 **kwargs):

        # Instantiate Composition -------------------------------------------------------------------------

        super().__init__(name=name,
                         learning_rate = learning_rate,
                         enable_learning = enable_learning,
                         random_state = random_state,
                         seed = seed,
                         **kwargs
                         )

        self._construct_pathways()

        # if torch_available:
        #     from psyneulink.library.compositions.pytorchGRUCompositionwrapper import PytorchGRUCompositionWrapper
        #     self.pytorch_composition_wrapper_type = PytorchGRUCompositionWrapper

        # Final Configuration and Clean-up ---------------------------------------------------------------------------


    # *****************************************************************************************************************
    # ******************************  Nodes and Pathway Construction Methods  *****************************************
    # *****************************************************************************************************************
    #region
    def _construct_pathways(self):
        """Construct Nodes and Pathways for GRUComposition"""

        # Construct Nodes --------------------------------------------------------------------------------


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
        if not self._use_storage_node:
            self._store_memory(inputs, context)
        return results


    @handle_external_context()
    def learn(self, *args, **kwargs)->list:
        """Override to check for inappropriate use of ARG_MAX or PROBABILISTIC options for retrieval with learning"""
        softmax_choice = self.parameters.softmax_choice.get(kwargs[CONTEXT])
        use_gating_for_weighting = self._use_gating_for_weighting
        enable_learning = self.parameters.enable_learning.get(kwargs[CONTEXT])

        if use_gating_for_weighting and enable_learning:
            raise GRUCompositionError(f"Field weights cannot be learned when 'use_gating_for_weighting' is True; "
                                     f"Construct '{self.name}' with the 'enable_learning' arg set to False.")

        if softmax_choice in {ARG_MAX, PROBABILISTIC}:
            raise GRUCompositionError(f"The ARG_MAX and PROBABILISTIC options for the 'softmax_choice' arg "
                                     f"of '{self.name}' cannot be used during learning; change to WEIGHTED_AVG.")

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

    # *****************************************************************************************************************
    # ***************************************** Properties  **********************************************************
    # *****************************************************************************************************************
    # region
    #endregion
