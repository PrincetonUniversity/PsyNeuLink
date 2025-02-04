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
HIDDEN_LAYER_NODE_NAME = 'HIDDEN LAYER'
HIDDEN_LAYER_AFFIX = f' [{HIDDEN_LAYER_NODE_NAME}]'
RESET_GATE_NAME = 'RESET GATE'
RESET_GATE_AFFIX = f' [{RESET_GATE_NAME}]'
UPDATE_GATE_NAME = 'UPDATE GATE'
UPDATE_GATE_AFFIX = f' [{UPDATE_GATE_NAME}]
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
                [self.hidden_layer_node,
                 self.reset_gate_node,
                 self.update_gate_node,
                 self.new_gate_node,
                if node is not None]

    @property
    def projections(self):
        """Return all Projections assigned to the field."""
        return [proj for proj in [self.reset_gating_projection,
                                  self.update_gating_projection,
                                  self.new_gating_projection,
                                  self.input_projection,
                                  self.hidden_layer_recurrent_projection]
                                  if proj is not None]
    @property


class GRUComposition(AutodiffComposition):
    """
    GRUComposition(                      \
        name="GRU_Composition"           \
        )

    Subclass of `AutodiffComposition` that implements a single-layered gated recurrent network.

    Takes the following arguments:

    Arguments
    ---------

    memory_template : tuple, list, 2d or 3d array : default [[0],[0]]
        specifies the shape of an item to be stored in the GRUComposition's memory
        (see `memory_template <GRUComposition_Memory_Template>` for details).

    memory_fill : scalar or tuple : default 0
        specifies the value used to fill the memory when it is initialized
        (see `memory_fill <GRUComposition_Memory_Fill>` for details).

    memory_capacity : int : default None
        specifies the number of items that can be stored in the GRUComposition's memory;
        (see `memory_capacity <GRUComposition_Memory_Capacity>` for details).

    fields : dict[tuple[field weight, learning specification]] : default None
        each key must a string that is the name of a field, and its value a dict or tuple that specifies that field's
        `field_weight <GRUComposition.field_weights>`, `learn_field_weights <GRUComposition.learn_field_weights>`, and
        `target_fields <GRUComposition.target_fields>` specifications (see `fields <GRUComposition_Fields>` for details
        of specificaton format). The **fields** arg replaces the **field_names**, **field_weights**
        **learn_field_weights**, and **target_fields** arguments, and specifying any of these raises an error.

    field_names : list or tuple : default None
        specifies the names assigned to each field in the memory_template (see `field names <GRUComposition_Field_Names>`
        for details). If the **fields** argument is specified, this is not necessary and specifying raises an error.

    field_weights : list or tuple : default (1,0)
        specifies the relative weight assigned to each key when matching an item in memory (see `field weights
        <GRUComposition_Field_Weights>` for additional details). If the **fields** argument is specified, this
        is not necessary and specifying raises an error.

    learn_field_weights : bool or list[bool, int, float]: default False
        specifies whether the `field_weights <GRUComposition.field_weights>` are learnable and, if so, optionally what
        the learning_rate is for each field (see `learn_field_weights <GRUComposition_Field_Weights_Learning>` for
        specifications). If the **fields** argument is specified, this is not necessary and specifying raises an error.

    learning_rate : float : default .01
        specifies the default learning_rate for `field_weights <GRUComposition.field_weights>` not
        specified in `learn_field_weights <GRUComposition.learn_field_weights>` (see `learning_rate
        <GRUComposition_Field_Weights_Learning>` for additional details).

    normalize_field_weights : bool : default True
        specifies whether the **fields_weights** are normalized over the number of keys, or used as absolute
        weighting values when retrieving an item from memory (see `normalize_field weights
        <GRUComposition_Normalize_Field_Weights>` for additional details).

    concatenate_queries : bool : default False
        specifies whether to concatenate the keys into a single field before matching them to items in
        the corresponding fields in memory (see `concatenate keys <GRUComposition_Concatenate_Queries>` for details).

    normalize_memories : bool : default True
        specifies whether keys and memories are normalized before computing their dot product (similarity)
        (see `Match memories by field <GRUComposition_Processing>` for additional details).

    softmax_gain : float, ADAPTIVE or CONTROL : default 1.0
        specifies the temperature used for softmax normalizing the distance of queries and keys in memory
        (see `Softmax normalize matches over fields <GRUComposition_Processing>` for additional details).

    softmax_threshold : float : default .0001
        specifies the threshold used to mask out small values in the softmax calculation
        see *mask_threshold* under `Thresholding and Adaptive Gain <SoftMax_AdaptGain>` for details).

    softmax_choice : WEIGHTED_AVG, ARG_MAX, PROBABILISTIC : default WEIGHTED_AVG
        specifies how the softmax over distances of queries and keys in memory is used for retrieval
        (see `softmax_choice <GRUComposition_Softmax_Choice>` for a description of each option).

    storage_prob : float : default 1.0
        specifies the probability that an item will be stored in `memory <GRUComposition.memory>`
        when the GRUComposition is executed (see `Retrieval and Storage <GRUComposition_Storage>` for
        additional details).

    memory_decay_rate : float : AUTO
        specifies the rate at which items in the GRUComposition's memory decay
        (see `memory_decay_rate <GRUComposition_Memory_Decay_Rate>` for details).

    purge_by_field_weights : bool : False
        specifies whether `fields_weights <GRUComposition.field_weights>` are used to determine which memory to
        replace when a new one is stored (see `purge_by_field_weight <GRUComposition_Purge_by_Weight>` for details).

    enable_learning : bool : default True
        specifies whether learning is enabled for the EMCComposition (see `Learning <GRUComposition_Learning>`
        for additional details); **use_gating_for_weighting** must be False.

    target_fields : list[bool]: default None
        specifies whether a learning pathway is constructed for each `field <GRUComposition_Entries_and_Fields>`
        of the GRUComposition.  If it is a list, each item must be ``True`` or ``False`` and the number of items
        must be equal to the number of `fields <GRUComposition_Fields> specified (see `Target Fields
         <GRUComposition_Target_Fields>` for additional details). If the **fields** argument is specified,
         this is not necessary and specifying raises an error.

    # 7/10/24 FIX: STILL TRUE?  DOES IT PRECLUDE USE OF GRUComposition as a nested Composition??
    .. technical_note::
        use_storage_node : bool : default True
            specifies whether to use a `LearningMechanism` to store entries in `memory <GRUComposition.memory>`.
            If False, a method on GRUComposition is used rather than a LearningMechanism. This is meant for
            debugging, and precludes use of `import_composition <Composition.import_composition>` to integrate
            the GRUComposition into another Composition;  to do so, use_storage_node must be True (default).

    use_gating_for_weighting : bool : default False
        specifies whether to use output gating to weight the `match_nodes <GRUComposition.match_node>` instead of
        a standard input (see `Weight distances <GRUComposition_Field_Weighting>` for additional details).

    Attributes
    ----------

    memory : ndarray
        3d array of entries in memory, in which each row (axis 0) is an entry, each column (axis 1) is a field, and
        each item (axis 2) is the value for the corresponding field (see `GRUComposition_Memory_Specification`  for
        additional details).

        .. note::
           This is a read-only attribute;  memories can be added to the GRUComposition's memory either by
           COMMENT:
           using its `add_to_memory <GRUComposition.add_to_memory>` method, or
           COMMENT
           executing its `run <Composition.run>` or learn methods with the entry as the ``inputs`` argument.

    fields : ContentAddressableList[Field]
        list of `Field` objects, each of which contains information about the nodes and values of a field in the
        GRUComposition's memory (see `Field`).

    .. _GRUComposition_Parameters:

    memory_capacity : int
        determines the number of items that can be stored in `memory <GRUComposition.memory>`
        (see `memory_capacity <GRUComposition_Memory_Capacity>` for additional details).

    field_names : list[str]
        determines which names that can be used to label fields in `memory <GRUComposition.memory>`
        (see `field_names <GRUComposition_Field_Names>` for additional details).

    field_weights : tuple[float]
        determines which fields of the input are treated as "keys" (non-zero values) that are used to match entries in
        `memory <GRUComposition.memory>` for retrieval, and which are used as "values" (zero values) that are stored
        and retrieved from memory but not used in the match process (see `Match memories by field
        <GRUComposition_Processing>`; also determines the relative contribution of each key field to the match process;
        see `field_weights <GRUComposition_Field_Weights>` additional details. The field_weights can be changed by
        assigning a new list of weights to the `field_weights <GRUComposition.field_weights>` attribute, however only
        the weights for fields used as `keys <GRUComposition_Entries_and_Fields>` can be changed (see
        `GRUComposition_Field_Weights_Change_Note` for additional details).

    learn_field_weights : bool or list[bool, int, float]
        determines whether the `field_weight <GRUComposition.field_weights>` for each `field <GRUComposition_Fields>
        is learnable (see `learn_field_weights <GRUComposition_Learning>` for additional details).

    learning_rate : float
        determines the default learning_rate for `field_weights <GRUComposition.field_weights>`
        not specified in `learn_field_weights <GRUComposition.learn_field_weights>`
        (see `learning_rate <GRUComposition_Field_Weights_Learning>` for additional details).

    normalize_field_weights : bool
        determines whether `fields_weights <GRUComposition.field_weights>` are normalized over the number of keys, or
        used as absolute weighting values when retrieving an item from memory (see `normalize_field weights
        <GRUComposition_Normalize_Field_Weights>` for additional details).

    concatenate_queries : bool
        determines whether keys are concatenated into a single field before matching them to items in `memory
        <GRUComposition.memory (see `concatenate keys <GRUComposition_Concatenate_Queries>` for additional details).

    normalize_memories : bool
        determines whether keys and memories are normalized before computing their dot product (similarity)
        (see `Match memories by field <GRUComposition_Processing>` for additional details).

    softmax_gain : float, ADAPTIVE or CONTROL
        determines gain (inverse temperature) used for softmax normalizing the summed distances of queries
        and keys in memory by the `SoftMax` Function of the `softmax_node <GRUComposition.softmax_node>`
        (see `Softmax normalize distances <GRUComposition_Processing>` for additional details).

    softmax_threshold : float
        determines the threshold used to mask out small values in the softmax calculation
        (see *mask_threshold* under `Thresholding and Adaptive Gain <SoftMax_AdaptGain>` for details).

    softmax_choice : WEIGHTED_AVG, ARG_MAX or PROBABILISTIC
        determines how the softmax over distances of queries and keys in memory is used for retrieval
        (see `softmax_choice <GRUComposition_Softmax_Choice>` for a description of each option).

    storage_prob : float
        determines the probability that an item will be stored in `memory <GRUComposition.memory>`
        when the GRUComposition is executed (see `Retrieval and Storage <GRUComposition_Storage>` for
        additional details).

    memory_decay_rate : float
        determines the rate at which items in the GRUComposition's memory decay
        (see `memory_decay_rate <GRUComposition_Memory_Decay_Rate>` for details).

    purge_by_field_weights : bool
        determines whether `fields_weights <GRUComposition.field_weights>` are used to determine which memory to
        replace when a new one is stored (see `purge_by_field_weight <GRUComposition_Purge_by_Weight>` for details).

    enable_learning : bool
        determines whether learning is enabled for the EMCComposition
        (see `Learning <GRUComposition_Learning>` for additional details).

    target_fields : list[bool]
        determines which fields convey error signals during learning
        (see `Target Fields <GRUComposition_Target_Fields>` for additional details).

    .. _GRUComposition_Nodes:

    query_input_nodes : list[ProcessingMechanism]
        `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` that receive keys used to determine the item
        to be retrieved from `memory <GRUComposition.memory>`, and then themselves stored in `memory
        <GRUComposition.memory>` (see `Match memories by field <GRUComposition_Processing>` for additional details).
        By default these are assigned the name *KEY_n_INPUT* where n is the field number (starting from 0);
        however, if `field_names <GRUComposition.field_names>` is specified, then the name of each query_input_node
        is assigned the corresponding field name appended with * [QUERY]*.

    value_input_nodes : list[ProcessingMechanism]
        `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` that receive values to be stored in `memory
        <GRUComposition.memory>`; these are not used in the matching process used for retrieval.  By default these
        are assigned the name *VALUE_n_INPUT* where n is the field number (starting from 0);  however, if
        `field_names <GRUComposition.field_names>` is specified, then the name of each value_input_node is assigned
        the corresponding field name appended with * [VALUE]*.

    concatenate_queries_node : ProcessingMechanism
        `ProcessingMechanism` that concatenates the inputs to `query_input_nodes <GRUComposition.query_input_nodes>`
        into a single vector used for the matching processing if `concatenate keys <GRUComposition.concatenate_queries>`
        is True. This is not created if the **concatenate_queries** argument to the GRUComposition's constructor is
        False or is overridden (see `concatenate_queries <GRUComposition_Concatenate_Queries>`), or there is only one
        query_input_node. This node is named *CONCATENATE_QUERIES*

    match_nodes : list[ProcessingMechanism]
        `ProcessingMechanisms <ProcessingMechanism>` that compute the dot product of each query and the key stored in
        the corresponding field of `memory <GRUComposition.memory>` (see `Match memories by field
        <GRUComposition_Processing>` for additional details). These are named the same as the corresponding
        `query_input_nodes <GRUComposition.query_input_nodes>` appended with the suffix *[MATCH to KEYS]*.

    field_weight_nodes : list[ProcessingMechanism or GatingMechanism]
        Nodes used to weight the distances computed by the `match_nodes <GRUComposition.match_nodes>` with the
        `field weight <GRUComposition.field_weights>` for the corresponding `key field <GRUComposition_Fields>`
        (see `Weight distances <GRUComposition_Field_Weighting>` for implementation). These are named the same
        as the corresponding `query_input_nodes <GRUComposition.query_input_nodes>`.

    weighted_match_nodes : list[ProcessingMechanism]
        `ProcessingMechanisms <ProcessingMechanism>` that combine the `field weight <GRUComposition.field_weights>`
        for each `key field <GRUComposition_Fields>` with the dot product computed by the corresponding the
        `match_node <GRUComposition.match_nodes>`. These are only implemented if `use_gating_for_weighting
        <GRUComposition.use_gating_for_weighting>` is False (see `Weight distances <GRUComposition_Field_Weighting>`
        for details), and are named the same as the corresponding `query_input_nodes <GRUComposition.query_input_nodes>`
        appended with the suffix *[WEIGHTED MATCH]*.

    combined_matches_node : ProcessingMechanism
        `ProcessingMechanism` that receives the weighted distances from the `weighted_match_nodes
        <GRUComposition.weighted_match_nodes>` if more than one `key field <GRUComposition_Fields>` is specified
        (or directly from `match_nodes <GRUComposition.match_nodes>` if `use_gating_for_weighting
        <GRUComposition.use_gating_for_weighting>` is True), and combines them into a single vector that is passed
        to the `softmax_node <GRUComposition.softmax_node>` for retrieval. This node is named *COMBINE MATCHES*.

    softmax_node : list[ProcessingMechanism]
        `ProcessingMechanisms <ProcessingMechanism>` that computes the softmax over the summed distances of keys
        and memories (output of the `combined_match_node <GRUComposition.combined_match_node>`)
        from the corresponding `match_nodes <GRUComposition.match_nodes>` (see `Softmax over summed distances
        <GRUComposition_Processing>` for additional details).  This is named *RETRIEVE* (as it yields the
        softmax-weighted average over the keys in `memory <GRUComposition.memory>`).

    softmax_gain_control_node : list[ControlMechanism]
        `ControlMechanisms <ControlMechanism>` that adaptively control the `softmax_gain <GRUComposition.softmax_gain>`
        of the `softmax_node <GRUComposition.softmax_node>`. This is implemented only if `softmax_gain
        <GRUComposition.softmax_gain>` is specified as *CONTROL* (see `softmax_gain <GRUComposition_Softmax_Gain>` for
        details).

    retrieved_nodes : list[ProcessingMechanism]
        `ProcessingMechanisms <ProcessingMechanism>` that receive the vector retrieved for each field in `memory
        <GRUComposition.memory>` (see `Retrieve values by field <GRUComposition_Processing>` for additional details).
        These are assigned the same names as the `query_input_nodes <GRUComposition.query_input_nodes>` and
        `value_input_nodes <GRUComposition.value_input_nodes>` to which they correspond appended with the suffix
        * [RETRIEVED]*, and are in the same order as  `input_nodes <GRUComposition.input_nodes>`
        to which to which they correspond.

    storage_node : EMStorageMechanism
        `EMStorageMechanism` that receives inputs from the `query_input_nodes <GRUComposition.query_input_nodes>` and
        `value_input_nodes <GRUComposition.value_input_nodes>`, and stores these in the corresponding field of`memory
        <GRUComposition.memory>` with probability `storage_prob <GRUComposition.storage_prob>` after a retrieval has been
        made (see `Retrieval and Storage <GRUComposition_Storage>` for additional details). This node is named *STORE*.

        .. technical_note::
           The `storage_node <GRUComposition.storage_node>` is assigned a Condition to execute after the `retrieved_nodes
           <GRUComposition.retrieved_nodes>` have executed, to ensure that storage occurs after retrieval, but before
           any subequent processing is done (i.e., in a composition in which the GRUComposition may be embededded.

    input_nodes : list[ProcessingMechanism]
        Full list of `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` in the same order specified in the
        **field_names** argument of the constructor and in `self.field_names <GRUComposition.field_names>`.

    query_and_value_input_nodes : list[ProcessingMechanism]
        Full list of `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` ordered with query_input_nodes first
        followed by value_input_nodes; used primarily for internal computations.

    """

    componentCategory = GRU_COMPOSITION

    if torch_available:
        from psyneulink.library.compositions.pytorchGRUcompositionwrapper import PytorchGRUCompositionWrapper
        pytorch_composition_wrapper_type = PytorchGRUCompositionWrapper


    class Parameters(AutodiffComposition.Parameters):
        """
            Attributes
            ----------

                concatenate_queries
                    see `concatenate_queries <GRUComposition.concatenate_queries>`

                    :default value: False
                    :type: ``bool``

                enable_learning
                    see `enable_learning <GRUComposition.enable_learning>`

                    :default value: True
                    :type: ``bool``

                field_names
                    see `field_names <GRUComposition.field_names>`

                    :default value: None
                    :type: ``list``

                field_weights
                    see `field_weights <GRUComposition.field_weights>`

                    :default value: None
                    :type: ``numpy.ndarray``

                learn_field_weights
                    see `learn_field_weights <GRUComposition.learn_field_weights>`

                    :default value: True
                    :type: ``numpy.ndarray``

                learning_rate
                    see `learning_results <GRUComposition.learning_rate>`

                    :default value: []
                    :type: ``list``

                memory
                    see `memory <GRUComposition.memory>`

                    :default value: None
                    :type: ``numpy.ndarray``

                memory_capacity
                    see `memory_capacity <GRUComposition.memory_capacity>`

                    :default value: 1000
                    :type: ``int``

                memory_decay_rate
                    see `memory_decay_rate <GRUComposition.memory_decay_rate>`

                    :default value: 0.001
                    :type: ``float``

                memory_template
                    see `memory_template <GRUComposition.memory_template>`

                    :default value: np.array([[0],[0]])
                    :type: ``np.ndarray``

                normalize_field_weights
                    see `normalize_field_weights <GRUComposition.normalize_field_weights>`

                    :default value: True
                    :type: ``bool``

                normalize_memories
                    see `normalize_memories <GRUComposition.normalize_memories>`

                    :default value: True
                    :type: ``bool``

                purge_by_field_weights
                    see `purge_by_field_weights <GRUComposition.purge_by_field_weights>`

                    :default value: False
                    :type: ``bool``

                random_state
                    see `random_state <NormalDist.random_state>`

                    :default value: None
                    :type: ``numpy.random.RandomState``

                softmax_gain
                    see `softmax_gain <GRUComposition.softmax_gain>`
                    :default value: 1.0
                    :type: ``float, ADAPTIVE or CONTROL``

                softmax_choice
                    see `softmax_choice <GRUComposition.softmax_choice>`
                    :default value: WEIGHTED_AVG
                    :type: ``keyword``

                softmax_threshold
                    see `softmax_threshold <GRUComposition.softmax_threshold>`
                    :default value: .001
                    :type: ``float``

                storage_prob
                    see `storage_prob <GRUComposition.storage_prob>`

                    :default value: 1.0
                    :type: ``float``
        """
        memory = Parameter(None, loggable=True, getter=_memory_getter, read_only=True)
        memory_template = Parameter([[0],[0]], structural=True, valid_types=(tuple, list, np.ndarray), read_only=True)
        memory_capacity = Parameter(1000, structural=True)
        field_names = Parameter(None, structural=True)
        field_weights = Parameter([1], setter=field_weights_setter)
        learn_field_weights = Parameter(False, structural=True)
        learning_rate = Parameter(.001, modulable=True)
        normalize_field_weights = Parameter(True)
        concatenate_queries = Parameter(False, structural=True)
        normalize_memories = Parameter(True)
        softmax_gain = Parameter(1.0, modulable=True)
        softmax_threshold = Parameter(.001, modulable=True, specify_none=True)
        softmax_choice = Parameter(WEIGHTED_AVG, modulable=False, specify_none=True)
        storage_prob = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        memory_decay_rate = Parameter(AUTO, modulable=True)
        purge_by_field_weights = Parameter(False, structural=True)
        enable_learning = Parameter(True, structural=True)
        target_fields = Parameter(None, read_only=True, structural=True)
        random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies='seed')
        seed = Parameter(DEFAULT_SEED(), modulable=True, setter=_seed_setter)

        def _validate_memory_template(self, memory_template):
            if isinstance(memory_template, tuple):
                if not len(memory_template) in {2,3}:
                    return f"must be length either 2 or 3 if it is a tuple (used to specify shape)."
                if not all(isinstance(item, int) for item in memory_template):
                    return f"must have only integers as entries."
            if isinstance(memory_template, (list, np.ndarray)):
                memory_template = np.array(memory_template)
                if memory_template.ndim not in {1,2,3}:
                    return f"must be either 2 or 3d."
                if not all(isinstance(item, (list, np.ndarray)) for item in memory_template):
                    return f"must be a list or array of lists or arrays."
                # if not all(isinstance(item, (int, float)) for sublist in memory_template for item in sublist):
                #     return f"must be a list or array of lists or arrays of integers or floats."
            else:
                return f"must be tuple of length 2 or 3, or a list or array that is either 2 or 3d."

        def _validate_field_names(self, field_names):
            if field_names and not all(isinstance(item, str) for item in field_names):
                return f"must be a list of strings."

        def _validate_field_weights(self, field_weights):
            if field_weights is not None:
                if not np.atleast_1d(field_weights).ndim == 1:
                    return f"must be a scalar, list of scalars, or 1d array."
                if len(field_weights) == 1 and field_weights[0] is None:
                    raise GRUCompositionError(f"must be a scalar, since there is only one field specified.")
                if any([field_weight < 0 for field_weight in field_weights if field_weight is not None]):
                    return f"must be all be positive values."

        def _validate_normalize_field_weights(self, normalize_field_weights):
            if not isinstance(normalize_field_weights, bool):
                return f"must be all be a boolean value."

        def _validate_learn_field_weights(self, learn_field_weights):
            if isinstance(learn_field_weights, (list, np.ndarray)):
                if not all(isinstance(item, (bool, int, float)) for item in learn_field_weights):
                    return f"can only contains bools, ints or floats as entries."
            elif not isinstance(learn_field_weights, bool):
                return f"must be a bool or list of bools, ints and/or floats."

        def _validate_memory_decay_rate(self, memory_decay_rate):
            if memory_decay_rate is None or memory_decay_rate == AUTO:
                return
            if not is_numeric_scalar(memory_decay_rate) and not (0 <= memory_decay_rate <= 1):
                return f"must be a float in the interval [0,1]."

        def _validate_softmax_gain(self, softmax_gain):
            if not is_numeric_scalar(softmax_gain) and softmax_gain not in {ADAPTIVE, CONTROL}:
                return f"must be a scalar or one the keywords '{ADAPTIVE}' or '{CONTROL}'."

        def _validate_softmax_threshold(self, softmax_threshold):
            if softmax_threshold is not None and (not is_numeric_scalar(softmax_threshold) or softmax_threshold <= 0):
                return f"must be a scalar greater than 0."

        def _validate_storage_prob(self, storage_prob):
            if not is_numeric_scalar(storage_prob) and not (0 <= storage_prob <= 1):
                return f"must be a float in the interval [0,1]."

    @check_user_specified
    def __init__(self,
                 memory_template:Union[tuple, list, np.ndarray]=[[0],[0]],
                 memory_capacity:Optional[int]=None,
                 memory_fill:Union[int, float, tuple, RANDOM]=0,
                 fields:Optional[dict]=None,
                 field_names:Optional[list]=None,
                 field_weights:Union[int,float,list,tuple]=None,
                 learn_field_weights:Union[bool,list,tuple]=None,
                 learning_rate:float=None,
                 normalize_field_weights:bool=True,
                 concatenate_queries:bool=False,
                 normalize_memories:bool=True,
                 softmax_gain:Union[float, ADAPTIVE, CONTROL]=1.0,
                 softmax_threshold:Optional[float]=.001,
                 softmax_choice:Optional[Union[WEIGHTED_AVG, ARG_MAX, PROBABILISTIC]]=WEIGHTED_AVG,
                 storage_prob:float=1.0,
                 memory_decay_rate:Union[float,AUTO]=AUTO,
                 purge_by_field_weights:bool=False,
                 enable_learning:bool=True,
                 target_fields:Optional[Union[list, tuple, np.ndarray]]=None,
                 use_storage_node:bool=True,
                 use_gating_for_weighting:bool=False,
                 random_state=None,
                 seed=None,
                 name="EM_Composition",
                 **kwargs):

        # Construct memory --------------------------------------------------------------------------------

        memory_fill = memory_fill or 0 # FIX: GET RID OF THIS ONCE IMPLEMENTED AS A Parameter
        self._validate_memory_specs(memory_template,
                                    memory_capacity,
                                    memory_fill,
                                    field_weights,
                                    field_names,
                                    name)

        memory_template, memory_capacity = self._parse_memory_template(memory_template,
                                                                       memory_capacity,
                                                                       memory_fill)

        self.fields = ContentAddressableList(component_type=Field)

        (field_names,
         field_weights,
         learn_field_weights,
         target_fields,
         concatenate_queries) = self._parse_fields(fields,
                                                   field_names,
                                                   field_weights,
                                                   learn_field_weights,
                                                   learning_rate,
                                                   normalize_field_weights,
                                                   concatenate_queries,
                                                   normalize_memories,
                                                   target_fields,
                                                   name)
        if memory_decay_rate is AUTO:
            memory_decay_rate = 1 / memory_capacity

        self._use_storage_node = use_storage_node
        self._use_gating_for_weighting = use_gating_for_weighting

        if softmax_gain == CONTROL:
            self.parameters.softmax_gain.modulable = False

        # Instantiate Composition -------------------------------------------------------------------------

        super().__init__(name=name,
                         memory_template = memory_template,
                         memory_capacity = memory_capacity,
                         field_names = field_names,
                         field_weights = field_weights,
                         learn_field_weights=learn_field_weights,
                         learning_rate = learning_rate,
                         normalize_field_weights = normalize_field_weights,
                         concatenate_queries = concatenate_queries,
                         normalize_memories = normalize_memories,
                         softmax_gain = softmax_gain,
                         softmax_threshold = softmax_threshold,
                         softmax_choice = softmax_choice,
                         storage_prob = storage_prob,
                         memory_decay_rate = memory_decay_rate,
                         purge_by_field_weights = purge_by_field_weights,
                         enable_learning = enable_learning,
                         target_fields = target_fields,
                         random_state = random_state,
                         seed = seed,
                         **kwargs
                         )

        self._validate_options_with_learning(use_gating_for_weighting,
                                             enable_learning,
                                             softmax_choice)

        self._construct_pathways(self.memory_template,
                                 self.memory_capacity,
                                 self.field_weights,
                                 self.concatenate_queries,
                                 self.normalize_memories,
                                 self.softmax_gain,
                                 self.softmax_threshold,
                                 self.softmax_choice,
                                 self.storage_prob,
                                 self.memory_decay_rate,
                                 self._use_storage_node,
                                 self.learn_field_weights,
                                 self.enable_learning,
                                 self._use_gating_for_weighting)

        # if torch_available:
        #     from psyneulink.library.compositions.pytorchGRUCompositionwrapper import PytorchGRUCompositionWrapper
        #     self.pytorch_composition_wrapper_type = PytorchGRUCompositionWrapper

        # Final Configuration and Clean-up ---------------------------------------------------------------------------

        # Assign learning-related attributes
        self._set_learning_attributes()

        if self._use_storage_node:
            # ---------------------------------------
            #
            # CONDITION:
            self.scheduler.add_condition(self.storage_node, conditions.AllHaveRun(*self.retrieved_nodes))
            #
            # Generates expected results, but execution_sets has a second set for INPUT nodes
            #    and the match_nodes again with storage_node
            #
            # ---------------------------------------
            #
            # CONDITION:
            # self.scheduler.add_condition(self.storage_node, conditions.AllHaveRun(*self.retrieved_nodes,
            #                                                               time_scale=TimeScale.PASS))
            # Hangs (or takes inordinately long to run),
            #     and evaluating list(execution_list) at LINE 11233 of composition.py hangs:
            #
            # ---------------------------------------
            # CONDITION:
            # self.scheduler.add_condition(self.storage_node, conditions.JustRan(self.retrieved_nodes[0]))
            #
            # Hangs (or takes inordinately long to run),
            #     and evaluating list(execution_list) at LINE 11233 of composition.py hangs:
            #
            # ---------------------------------------
            # CONDITION:
            # self.scheduler.add_condition_set({n: conditions.BeforeNCalls(n, 1) for n in self.nodes})
            # self.scheduler.add_condition(self.storage_node, conditions.AllHaveRun(*self.retrieved_nodes))
            #
            # Generates the desired execution set for a single pass, and runs with expected results,
            #   but raises a warning messages for every node of the following sort:
            # /Users/jdc/PycharmProjects/PsyNeuLink/psyneulink/core/scheduling/scheduler.py:120:
            #   UserWarning: BeforeNCalls((EMStorageMechanism STORAGE MECHANISM), 1) is dependent on
            #   (EMStorageMechanism STORAGE MECHANISM), but you are assigning (EMStorageMechanism STORAGE MECHANISM)
            #   as its owner. This may result in infinite loops or unknown behavior.
            # super().add_condition_set(conditions)

        # Suppress warnings for no efferent Projections
        for node in self.value_input_nodes:
            node.output_port.parameters.require_projection_in_composition.set(False, override=True)
        self.softmax_node.output_port.parameters.require_projection_in_composition.set(False, override=True)

        # Suppress field_weight_nodes as INPUT nodes of the Composition
        for node in self.field_weight_nodes:
            self.exclude_node_roles(node, NodeRole.INPUT)

        # Suppress value_input_nodes as OUTPUT nodes of the Composition
        for node in self.value_input_nodes:
            self.exclude_node_roles(node, NodeRole.OUTPUT)

        # Warn if divide by zero will occur due to memory initialization
        memory = self.memory
        memory_capacity = self.memory_capacity
        if not np.any([
            np.any([memory[i][j] for i in range(memory_capacity)])
            for j in range(self.num_keys)
        ]):
            warnings.warn(f"Memory initialized with at least one field that has all zeros; "
                          f"a divide by zero will occur if 'normalize_memories' is True. "
                          f"This can be avoided by using 'memory_fill' to initialize memories with non-zero values.")

    # *****************************************************************************************************************
    # ***********************************  Memory Construction Methods  ***********************************************
    # *****************************************************************************************************************
    #region
    def _validate_memory_specs(self, memory_template, memory_capacity, memory_fill, field_weights, field_names, name):
        """Validate the memory_template, field_weights, and field_names arguments
        """

        # memory_template must specify a 2D array:
        if isinstance(memory_template, tuple):
        #     if len(memory_template) != 2 or not all(isinstance(item, int) for item in memory_template):
        #         raise GRUCompositionError(f"The 'memory_template' arg for {name} ({memory_template}) uses a tuple to "
        #                                  f"shape requires but does not have exactly two integers.")
            num_fields = memory_template[0]
            if len(memory_template) == 3:
                num_entries = memory_template[0]
            else:
                num_entries = memory_capacity
        elif isinstance(memory_template, (list, np.ndarray)):
            num_entries, num_fields = self._parse_memory_shape(memory_template)
        else:
            raise GRUCompositionError(f"Unrecognized specification for "
                                     f"the 'memory_template' arg ({memory_template}) of {name}.")

        # If a 3d array is specified (i.e., template has multiple entries), ensure all have the same shape
        if not isinstance(memory_template, tuple) and num_entries > 1:
            for entry in memory_template:
                if not (len(entry) == num_fields
                        and np.all([len(entry[i]) == len(memory_template[0][i]) for i in range(num_fields)])):
                    raise GRUCompositionError(f"The 'memory_template' arg for {name} must specify a list "
                                             f"or 2d array that has the same shape for all entries.")

        # Validate memory_fill specification (int, float, or tuple with two scalars)
        if not (isinstance(memory_fill, (int, float)) or
                (isinstance(memory_fill, tuple) and len(memory_fill)==2) and
                all(isinstance(item, (int, float)) for item in memory_fill)):
            raise GRUCompositionError(f"The 'memory_fill' arg ({memory_fill}) specified for {name} "
                                     f"must be a float, int or len tuple of ints and/or floats.")

        # If learn_field_weights is a list of bools, it must match the len of 1st dimension (axis 0) of memory_template:
        if isinstance(self.learn_field_weights, list) and len(self.learn_field_weights) != num_fields:
            raise GRUCompositionError(f"The number of items ({len(self.learn_field_weights)}) in the "
                                     f"'learn_field_weights' arg for {name} must match the number of "
                                     f"fields in memory ({num_fields}).")

        _field_wts = np.atleast_1d(field_weights)
        _field_wts_len = len(_field_wts)

        # If len of field_weights > 1, must match the len of 1st dimension (axis 0) of memory_template:
        if field_weights is not None:
            if (_field_wts_len > 1 and _field_wts_len != num_fields):
                raise GRUCompositionError(f"The number of items ({_field_wts_len}) in the 'field_weights' arg "
                                         f"for {name} must match the number of items in an entry of memory "
                                         f"({num_fields}).")
            # Deal with this here instead of Parameter._validate_field_weights since this is called before super()
            if all([fw is None for fw in _field_wts]):
                raise GRUCompositionError(f"The entries in 'field_weights' arg for {name} can't all be 'None' "
                                         f"since that will preclude the construction of any keys.")

            if not any(_field_wts):
                warnings.warn(f"All of the entries in the 'field_weights' arg for {name} "
                              f"are either None or set to 0; this will result in no retrievals "
                              f"unless/until one or more of them are changed to a positive value.")

            elif any([fw == 0 for fw in _field_wts if fw is not None]):
                warnings.warn(f"Some of the entries in the 'field_weights' arg for {name} "
                              f"are set to 0; those fields will be ignored during retrieval "
                              f"unless/until they are changed to a positive value.")

        # If field_names has more than one value it must match the first dimension (axis 0) of memory_template:
        if field_names and len(field_names) != num_fields:
            raise GRUCompositionError(f"The number of items ({len(field_names)}) "
                                     f"in the 'field_names' arg for {name} must match "
                                     f"the number of fields ({_field_wts_len}).")

    def _parse_memory_template(self, memory_template, memory_capacity, memory_fill)->(np.ndarray,int):
        """Construct memory from memory_template and memory_fill
        Assign self.memory_template and self.entry_template attributes
        """

        def _construct_entries(entry_template, num_entries, memory_fill=None)->np.ndarray:
            """Construct memory entries from memory_template and memory_fill"""

            # Random fill specification
            if isinstance(memory_fill, tuple):
                entries = [[np.full(len(field),
                                    np.random.uniform(memory_fill[1], # upper bound
                                                      memory_fill[0], # lower bound
                                                      len(field))).tolist()
                            for field in entry_template] for i in range(num_entries)]
            else:
                # Fill with zeros
                if memory_fill is None:
                    entry = entry_template
                # Fill with specified value
                elif isinstance(memory_fill, (list, float, int)):
                    entry = [np.full(len(field), memory_fill).tolist() for field in entry_template]
                entries = [np.array(entry, dtype=object) for _ in range(num_entries)]

            return np.array(np.array(entries,dtype=object), dtype=object)

        # If memory_template is a tuple, create and fill full memory matrix
        if isinstance(memory_template, tuple):
            if len(memory_template) == 2:
                memory_capacity = memory_capacity or self.defaults.memory_capacity
                memory = _construct_entries(np.full(memory_template, 0), memory_capacity, memory_fill)
            else:
                if memory_capacity and memory_template[0] != memory_capacity:
                    raise GRUCompositionError(
                        f"The first item ({memory_template[0]}) of the tuple in the 'memory_template' arg "
                        f"for {self.name} does not match the specification of the 'memory_capacity' arg "
                        f"({memory_capacity}); should remove the latter or use a 2-item tuple, list or array in "
                        f"'memory_template' to specify the shape of entries.")
                memory_capacity = memory_template[0]
                memory = _construct_entries(np.full(memory_template[1:], 0), memory_capacity, memory_fill)

        # If memory_template is a list or array
        else:
            # Determine whether template is a single entry or full/partial memory specification
            num_entries, num_fields = self._parse_memory_shape(memory_template)

            # memory_template specifies a single entry
            if num_entries == 1:
                memory_capacity = memory_capacity or self.defaults.memory_capacity
                if np.array([np.nonzero(field) for field in memory_template],dtype=object).any():
                    memory_fill = None
                # Otherwise, use memory_fill
                memory = _construct_entries(memory_template, memory_capacity, memory_fill)

            # If memory template is a full or partial 3d (matrix) specification
            else:
                # If all entries are zero, create entire memory matrix with memory_fill
                if not any(list(np.array(memory_template, dtype=object).flat)):
                    # Use first entry of zeros as template and replicate for full memory matrix
                    memory = _construct_entries(memory_template[0], memory_capacity, memory_fill)
                # If there are any non-zero values, keep specified entries and create rest using memory_fill
                else:
                    memory_capacity = memory_capacity or num_entries
                    if num_entries > memory_capacity:
                        raise GRUCompositionError(
                            f"The number of entries ({num_entries}) specified in "
                            f"the 'memory_template' arg of  {self.name} exceeds the number of entries specified in "
                            f"its 'memory_capacity' arg ({memory_capacity}); remove the latter or reduce the number"
                            f"of entries specified in 'memory_template'.")
                    num_entries_needed = memory_capacity - len(memory_template)
                    # Get remaining entries populated with memory_fill
                    remaining_entries = _construct_entries(memory_template[0], num_entries_needed, memory_fill)
                    assert bool(num_entries_needed == len(remaining_entries))
                    # If any remaining entries, concatenate them with the entries that were specified
                    if num_entries_needed:
                        memory = np.concatenate((np.array(memory_template, dtype=object),
                                                 np.array(remaining_entries, dtype=object)))
                    # All entries were specivied, so just retun memory_template
                    else:
                        memory = np.array(memory_template, dtype=object)

        # Get shape of single entry
        self.entry_template = memory[0]

        return memory, memory_capacity

    def _parse_fields(self,
                      fields,
                      field_names,
                      field_weights,
                      learn_field_weights,
                      learning_rate,
                      normalize_field_weights,
                      concatenate_queries,
                      normalize_memories,
                      target_fields,
                      name)->(list, list, list, bool):

        def _parse_fields_dict(name, fields, num_fields)->(list,list,list,list):
            """Parse fields dict into field_names, field_weights, learn_field_weights, and target_fields"""
            if len(fields) != num_fields:
                raise GRUCompositionError(f"The number of entries ({len(fields)}) in the dict specified in the 'fields' "
                                         f"arg of '{name}' does not match the number of fields in its memory "
                                         f"({self.num_fields}).")
            field_names = [None] * num_fields
            field_weights = [None] * num_fields
            learn_field_weights = [None] * num_fields
            target_fields = [None] * num_fields
            for i, field_name in enumerate(fields):
                field_names[i] = field_name
                if isinstance(fields[field_name], (tuple, list)):
                    # field specified as tuple or list
                    field_weights[i] = fields[field_name][0]
                    learn_field_weights[i] = fields[field_name][1]
                    target_fields[i] = fields[field_name][2]
                elif isinstance(fields[field_name], dict):
                    # field specified as dict
                    field_weights[i] = fields[field_name][FIELD_WEIGHT]
                    learn_field_weights[i] = fields[field_name][LEARN_FIELD_WEIGHT]
                    target_fields[i] = fields[field_name][TARGET_FIELD]
                else:
                    raise GRUCompositionError(f"Unrecognized specification for field '{field_name}' in the 'fields' "
                                             f"arg of '{name}'; it must be a tuple, list or dict.")
            return field_names, field_weights, learn_field_weights, target_fields

        self.num_fields = len(self.entry_template)

        if fields:
            # If a fields dict has been specified, use that to assign field_names, field_weights & learn_field_weights
            if any([field_names, field_weights, learn_field_weights, target_fields]):
                warnings.warn(f"The 'fields' arg for '{name}' was specified, so any of the 'field_names', "
                              f"'field_weights',  'learn_field_weights' or 'target_fields' args will be ignored.")
            (field_names,
             field_weights,
             learn_field_weights,
             target_fields) = _parse_fields_dict(name, fields, self.num_fields)

        # Deal with default field_weights
        if field_weights is None:
            if len(self.entry_template) == 1:
                field_weights = [1]
            else:
                # Default is to treat all fields as keys except the last one, which is the value
                field_weights = [1] * self.num_fields
                field_weights[-1] = None
        field_weights = np.atleast_1d(field_weights)

        if normalize_field_weights and not all([fw == 0 for fw in field_weights]): # noqa: E127
            fld_wts_0s_for_Nones = [fw if fw is not None else 0 for fw in field_weights]
            parsed_field_weights = list(np.array(fld_wts_0s_for_Nones) / (np.sum(fld_wts_0s_for_Nones) or 1))
            parsed_field_weights = [pfw if fw is not None else None
                                    for pfw, fw in zip(parsed_field_weights, field_weights)]
        else:
            parsed_field_weights = field_weights

        # If only one field_weight was specified, but there is more than one field,
        #    repeat the single weight for each field
        if len(field_weights) == 1 and self.num_fields > 1:
            parsed_field_weights = np.repeat(parsed_field_weights, self.num_fields)

        # Make sure field_weight learning was not specified for any value fields (since they don't have field_weights)
        if isinstance(learn_field_weights, (list, tuple, np.ndarray)):
            for i, lfw in enumerate(learn_field_weights):
                if parsed_field_weights[i] is None and lfw is not False:
                    warnings.warn(f"Learning was specified for field '{field_names[i]}' in the 'learn_field_weights' "
                                  f"arg for '{name}', but it is not allowed for value fields; it will be ignored.")
        elif learn_field_weights in {None, True, False}:
            learn_field_weights = [False] * len(parsed_field_weights)
        else:
            assert False, f"PROGRAM ERROR: learn_field_weights ({learn_field_weights}) is not a list, tuple or bool."

        # Memory structure Parameters
        parsed_field_names = field_names.copy() if field_names is not None else None

        # Set memory field attributes
        keys_weights = [i for i in parsed_field_weights if i is not None]
        self.num_keys = len(keys_weights)

        # Get indices of field_weights that specify keys and values:
        self.key_indices = [i for i, pfw in enumerate(parsed_field_weights) if pfw is not None]
        assert len(self.key_indices) == self.num_keys, \
            f"PROGRAM ERROR: number of keys ({self.num_keys}) does not match number of " \
            f"non-zero values in field_weights ({len(self.key_indices)})."
        self.value_indices = [i for i, pfw in enumerate(parsed_field_weights) if pfw is None]
        self.num_values = self.num_fields - self.num_keys
        assert len(self.value_indices) == self.num_values, \
            f"PROGRAM ERROR: number of values ({self.num_values}) does not match number of " \
            f"zero values in field_weights ({len(self.value_indices)})."

        if parsed_field_names:
            self.key_names = [parsed_field_names[i] for i in self.key_indices]
            # self.value_names = parsed_field_names[self.num_keys:]
            self.value_names = [parsed_field_names[i] for i in range(self.num_fields) if i not in self.key_indices]
        else:
            self.key_names = [f'{i}' for i in range(self.num_keys)] if self.num_keys > 1 else ['KEY']
            if self.num_values > 1:
                self.value_names = [f'{i} [VALUE]' for i in range(self.num_values)]
            elif self.num_values == 1:
                self.value_names = ['VALUE']
            else:
                self.value_names = []
            parsed_field_names = self.key_names + self.value_names

        user_specified_concatenate_queries = concatenate_queries or False
        parsed_concatenate_queries = (user_specified_concatenate_queries
                                    and self.num_keys > 1
                                    and np.all(keys_weights == keys_weights[0])
                                    and normalize_memories)
        # if concatenate_queries was forced to be False when user specified it as True, issue warning
        if user_specified_concatenate_queries and not parsed_concatenate_queries:
            # Issue warning if concatenate_queries is True but:
            #   field weights are not all equal and/or
            #   normalize_memories is False and/or
            #   there is only one key
            if self.num_keys == 1:
                error_msg = f"there is only one key"
                correction_msg = ""
            elif not all(np.all(keys_weights[i] == keys_weights[0] for i in range(len(keys_weights)))):
                error_msg = f" field weights ({field_weights}) are not all equal"
                correction_msg = (f" To use concatenation, remove `field_weights` "
                                     f"specification or make them all the same.")
            elif not normalize_memories:
                error_msg = f" normalize_memories is False"
                correction_msg = f" To use concatenation, set normalize_memories to True."
            warnings.warn(f"The 'concatenate_queries' arg for '{name}' is True but {error_msg}; "
                          f"concatenation will be ignored.{correction_msg}")

        # Deal with default target_fields
        if target_fields is None:
            target_fields = [True] * self.num_fields

        self.learning_rate = learning_rate

        for i, name, weight, learn_weight, target in zip(range(self.num_fields),
                                                         parsed_field_names,
                                                         parsed_field_weights,
                                                         learn_field_weights,
                                                         target_fields):
            self.fields.append(Field(name=name,
                                     index=i,
                                     type=FieldType.KEY if weight is not None else FieldType.VALUE,
                                     weight=weight,
                                     learn_weight=learn_weight,
                                     target=target))

        return (parsed_field_names,
                parsed_field_weights,
                learn_field_weights,
                target_fields,
                parsed_concatenate_queries)

    def _parse_memory_shape(self, memory_template):
        """Parse shape of memory_template to determine number of entries and fields"""
        memory_template_dim = np.array(memory_template, dtype=object).ndim
        if memory_template_dim == 1 or all(isinstance(item, (int, float)) for item in memory_template[0]):
            fields_equal_length = all(len(field) == len(memory_template[0]) for field in memory_template)
        else:
            fields_equal_length = all(len(field) == len(memory_template[0]) for field in memory_template[0])

        single_entry = (((memory_template_dim == 1) and not fields_equal_length) or
                        ((memory_template_dim == 2) and fields_equal_length))
        num_entries = 1 if single_entry else len(memory_template)
        num_fields = len(memory_template) if single_entry else len(memory_template[0])
        return num_entries, num_fields

    #endregion

    # *****************************************************************************************************************
    # ******************************  Nodes and Pathway Construction Methods  *****************************************
    # *****************************************************************************************************************
    #region
    def _construct_pathways(self,
                            memory_template,
                            memory_capacity,
                            field_weights,
                            concatenate_queries,
                            normalize_memories,
                            softmax_gain,
                            softmax_threshold,
                            softmax_choice,
                            storage_prob,
                            memory_decay_rate,
                            use_storage_node,
                            learn_field_weights,
                            enable_learning,
                            use_gating_for_weighting,
                            ):
        """Construct Nodes and Pathways for GRUComposition"""

        # Construct Nodes --------------------------------------------------------------------------------

        self._construct_input_nodes()
        self._construct_concatenate_queries_node(concatenate_queries)
        self._construct_match_nodes(memory_template, memory_capacity, concatenate_queries,normalize_memories)
        self._construct_field_weight_nodes(concatenate_queries, use_gating_for_weighting)
        self._construct_weighted_match_nodes(concatenate_queries)
        self._construct_combined_matches_node(concatenate_queries, memory_capacity, use_gating_for_weighting)
        self._construct_softmax_node(memory_capacity, softmax_gain, softmax_threshold, softmax_choice)
        self._construct_softmax_gain_control_node(softmax_gain)
        self._construct_retrieved_nodes(memory_template)
        self._construct_storage_node(use_storage_node, memory_template, memory_decay_rate, storage_prob)

        # Do some validation and get singleton softmax and match Nodes for concatenated queries
        if self.concatenate_queries:
            assert len(self.match_nodes) == 1, \
                f"PROGRAM ERROR: Too many match_nodes ({len(self.match_nodes)}) for concatenated queries."
            assert not self.field_weight_nodes, \
                f"PROGRAM ERROR: There should be no field_weight_nodes for concatenated queries."


        # Create _field_index_map by first assigning indices for all Field Nodes and their Projections
        self._field_index_map = {node: field.index for field in self.fields for node in field.nodes}
        self._field_index_map.update({proj: field.index for field in self.fields for proj in field.projections})
        if self.concatenate_queries:
            # Add projections to concatenated_queries_node with indices of sender query_input_nodes
            for proj in self.concatenate_queries_node.path_afferents:
                self._field_index_map[proj] = self._field_index_map[proj.sender.owner]
            # No indices for singleton Nodes and Projections from concatenated_queries_node through to softmax_node
            self._field_index_map[self.concatenate_queries_node] = None
            self._field_index_map[self.match_nodes[0]] = None
            self._field_index_map[self.match_nodes[0].path_afferents[0]] = None
            self._field_index_map[self.match_nodes[0].efferents[0]] = None


        # Construct Pathways --------------------------------------------------------------------------------
        # FIX: REFACTOR TO ITERATE OVER Fields

        # LEARNING NOT ENABLED --------------------------------------------------
        # Set up pathways WITHOUT PsyNeuLink learning pathways
        if not self.enable_learning:
            self.add_nodes(self.input_nodes)
            if use_storage_node:
                self.add_node(self.storage_node)
            if self.concatenate_queries_node:
                self.add_node(self.concatenate_queries_node)
            self.add_nodes(self.match_nodes + self.field_weight_nodes + self.weighted_match_nodes)
            if self.combined_matches_node:
                self.add_node(self.combined_matches_node)
            self.add_nodes([self.softmax_node] + self.retrieved_nodes)
            if self.softmax_gain_control_node:
                self.add_node(self.softmax_gain_control_node)

        # LEARNING ENABLED -----------------------------------------------------
        # Set up pathways WITH psyneulink backpropagation learning field weights
        else:
            # Query-specific pathways
            if not self.concatenate_queries:
                if self.num_keys == 1:
                    self.add_linear_processing_pathway([self.query_input_nodes[0],
                                                        self.match_nodes[0],
                                                        self.softmax_node])
                else:
                    for i in range(self.num_keys):
                        pathway = [self.query_input_nodes[i],
                                   self.match_nodes[i],
                                   self.combined_matches_node]
                        if self.weighted_match_nodes:
                            pathway.insert(2, self.weighted_match_nodes[i])
                        self.add_linear_processing_pathway(pathway)
                    self.add_linear_processing_pathway([self.combined_matches_node, self.softmax_node])
            # Query-concatenated pathways
            else:
                for i in range(self.num_keys):
                    pathway = [self.query_input_nodes[i],
                               self.concatenate_queries_node,
                               self.match_nodes[0]]
                    self.add_linear_processing_pathway(pathway)
                self.add_linear_processing_pathway([self.match_nodes[0], self.softmax_node])

            # softmax gain control is specified:
            if self.softmax_gain_control_node:
                self.add_node(self.softmax_gain_control_node)

            # field_weights -> weighted_softmax pathways
            if any(self.field_weight_nodes):
                for i in range(self.num_keys):
                    self.add_linear_processing_pathway([self.field_weight_nodes[i], self.weighted_match_nodes[i]])

            self.add_nodes(self.value_input_nodes)

            # Retrieval pathways
            for i in range(len(self.retrieved_nodes)):
                self.add_linear_processing_pathway([self.softmax_node, self.retrieved_nodes[i]])

            # Storage Nodes
            if use_storage_node:
                self.add_node(self.storage_node)

    def _construct_input_nodes(self):
        """Create one node for each input to GRUComposition and identify as key or value
        """
        assert len(self.key_indices) == self.num_keys, \
            f"PROGRAM ERROR: number of keys ({self.num_keys}) does not match number of " \
            f"non-zero values in field_weights ({len(self.key_indices)})."
        assert len(self.value_indices) == self.num_values, \
            f"PROGRAM ERROR: number of values ({self.num_values}) does not match number of " \
            f"None's in field_weights ({len(self.value_indices)})."

        for field in [self.fields[i] for i in self.key_indices]:
            field.input_node = ProcessingMechanism(name=f'{field.name} [QUERY]',
                                                   input_shapes=len(self.entry_template[field.index]))
            field.type = FieldType.KEY

        for field in [self.fields[i] for i in self.value_indices]:
            field.input_node = ProcessingMechanism(name=f'{field.name} [VALUE]',
                                                   input_shapes=len(self.entry_template[field.index]))
            field.type = FieldType.VALUE

    def _construct_concatenate_queries_node(self, concatenate_queries):
        """Create node that concatenates the inputs for all keys into a single vector
        Used to create a matrix for Projection from match / memory weights from concatenate_node -> match_node
        """
        if concatenate_queries:
            # One node that concatenates inputs from all keys
            self.concatenate_queries_node = (
                ProcessingMechanism(name=CONCATENATE_QUERIES_NAME,
                                    function=Concatenate,
                                    input_ports=[{NAME: 'CONCATENATE',
                                                  INPUT_SHAPES: len(self.query_input_nodes[i].output_port.value),
                                                  PROJECTIONS: MappingProjection(
                                                      name=f'{self.key_names[i]} to CONCATENATE',
                                                      sender=self.query_input_nodes[i].output_port,
                                                      matrix=IDENTITY_MATRIX)}
                                                 for i in range(self.num_keys)]))
            # Add Projections from query_input_nodes to concatenate_queries_node to each Field
            for i, proj in enumerate(self.concatenate_queries_node.path_afferents):
                self.fields[self.key_indices[i]].concatenation_projection = proj

        else:
            self.concatenate_queries_node = None

    def _construct_match_nodes(self, memory_template, memory_capacity, concatenate_queries, normalize_memories):
        """Create nodes that, for each key field, compute the similarity between the input and each item in memory.
        - If self.concatenate_queries is True, then all inputs for keys from concatenated_keys_node are
            assigned a single match_node, and weights from memory_template are assigned to a Projection
            from concatenated_keys_node to that match_node.
        - Otherwise, each key has its own match_node, and weights from memory_template are assigned to a Projection
            from each query_input_node[i] to each match_node[i].
        - Each element of the output represents the similarity between the query_input and one key in memory.
        """
        OPERATION = 0
        NORMALIZE = 1
        # Enforce normalization of memories if key is a scalar
        #   (this is to allow 1-L0 distance to be used as similarity measure, so that better matches
        #   (more similar memories) have higher match values; see `MatrixTransform` for explanation)
        args = [(L0,True) if len(key) == 1 else (DOT_PRODUCT,normalize_memories)
                for key in memory_template[0]]

        if concatenate_queries:
            # Assign one match_node for concatenate_queries_node
            # - get fields of memory structure corresponding to the keys
            # - number of rows should total number of elements over all keys,
            #    and columns should number of items in memory
            matrix =np.array([np.concatenate((memory_template[:,:self.num_keys][i]))
                              for i in range(memory_capacity)]).transpose()
            memory_projection = MappingProjection(name=f'MEMORY',
                                                  sender=self.concatenate_queries_node,
                                                  matrix=np.array(matrix.tolist()),
                                                  function=MatrixTransform(operation=args[0][OPERATION],
                                                                           normalize=args[0][NORMALIZE]))
            self.concatenated_match_node = ProcessingMechanism(name='MATCH',
                                                               input_ports={NAME: 'CONCATENATED_INPUTS',
                                                                            INPUT_SHAPES: memory_capacity,
                                                                            PROJECTIONS: memory_projection})
            # Assign None as match_node for all key Fields (since they first project to concatenate_queries_node)
            for field in [field for field in self.fields if field.type == FieldType.KEY]:
                field.match_node = None

        else:
            # Assign each key Field its own match_node and "memory" Projection to it
            for i in range(self.num_keys):
                key_idx = self.key_indices[i]
                field = self.fields[key_idx]
                memory_projection = (
                    MappingProjection(name=f'MEMORY for {self.key_names[i]} [KEY]',
                                      sender=self.query_input_nodes[i].output_port,
                                      matrix = np.array(memory_template[:,key_idx].tolist()).transpose().astype(float),
                                      function=MatrixTransform(operation=args[key_idx][OPERATION],
                                                               normalize=args[key_idx][NORMALIZE])))
                field.match_node = (ProcessingMechanism(name=self.key_names[i] + MATCH_TO_KEYS_AFFIX,
                                                        input_ports= {INPUT_SHAPES:memory_capacity,
                                                                      PROJECTIONS: memory_projection}))
                field.memory_projection = memory_projection


    def _construct_field_weight_nodes(self, concatenate_queries, use_gating_for_weighting):
        """Create ProcessingMechanisms that weight each key's softmax contribution to the retrieved values."""
        if not concatenate_queries and self.num_keys > 1:
            for field in [self.fields[i] for i in self.key_indices]:
                name = WEIGHT if self.num_keys == 1 else f'{field.name}{WEIGHT_AFFIX}'
                variable = np.array(self.field_weights[field.index])
                params = {DEFAULT_INPUT: DEFAULT_VARIABLE}
                if use_gating_for_weighting:
                    field.weight_node = GatingMechanism(name=name,
                                                        input_ports={NAME: 'OUTCOME',
                                                                     VARIABLE: variable,
                                                                     PARAMS: params},
                                                        gate=field.match_node.output_ports[0])
                else:
                    field.weight_node = ProcessingMechanism(name=name,
                                                            input_ports={NAME: 'FIELD_WEIGHT',
                                                                         VARIABLE: variable,
                                                                         PARAMS: params})

    def _construct_weighted_match_nodes(self, concatenate_queries):
        """Create nodes that weight the output of the match node for each key."""
        if not concatenate_queries and self.num_keys > 1:
            for field in [self.fields[i] for i in self.key_indices]:
                field.weighted_match_node = (
                    ProcessingMechanism(name=field.name + WEIGHTED_MATCH_AFFIX,
                                        default_variable=[field.match_node.output_port.value,
                                                          field.match_node.output_port.value],
                                        input_ports=[{PROJECTIONS:
                                                          MappingProjection(name=(f'{MATCH} to {WEIGHTED_MATCH_NODE_NAME} '
                                                                                  f'for {field.name}'),
                                                                            sender=field.match_node,
                                                                            matrix=IDENTITY_MATRIX)},
                                                     {PROJECTIONS:
                                                          MappingProjection(name=(f'{WEIGHT} to {WEIGHTED_MATCH_NODE_NAME} '
                                                                                  f'for {field.name}'),
                                                                            sender=field.weight_node,
                                                                            matrix=FULL_CONNECTIVITY_MATRIX)}],
                                        function=LinearCombination(operation=PRODUCT)))
                field.match_projection = field.match_node.efferents[0]
                field.weight_projection = field.weight_node.efferents[0]

    def _construct_softmax_gain_control_node(self, softmax_gain):
        """Create nodes that set the softmax gain (inverse temperature) for each softmax_node."""
        node = None
        if softmax_gain == CONTROL:
            node = ControlMechanism(name='SOFTMAX GAIN CONTROL',
                                    monitor_for_control=self.combined_matches_node,
                                    control_signals=[(GAIN, self.softmax_node)],
                                    function=get_softmax_gain)
        self.softmax_gain_control_node = node

    def _construct_combined_matches_node(self,
                                         concatenate_queries,
                                         memory_capacity,
                                         use_gating_for_weighting
                                         ):
        """Create node that combines weighted matches for all keys into one match vector."""
        if self.num_keys == 1 or self.concatenate_queries_node:
            self.combined_matches_node = None
            return

        field_weighting = len([weight for weight in self.field_weights if weight]) > 1 and not concatenate_queries

        if not field_weighting or use_gating_for_weighting:
            input_source = self.match_nodes
        else:
            input_source = self.weighted_match_nodes

        self.combined_matches_node = (
            ProcessingMechanism(name=COMBINE_MATCHES_NODE_NAME,
                                input_ports=[{INPUT_SHAPES:memory_capacity,
                                              PROJECTIONS:[MappingProjection(sender=s,
                                                                             matrix=IDENTITY_MATRIX,
                                                                             name=f'{WEIGHTED_MATCH_NODE_NAME} '
                                                                                  f'for {self.key_names[i]} to '
                                                                                  f'{COMBINE_MATCHES_NODE_NAME}')
                                                           for i, s in enumerate(input_source)]}]))

        for i, proj in enumerate(self.combined_matches_node.path_afferents):
            self.fields[self.key_indices[i]].weighted_match_projection = proj

        assert len(self.combined_matches_node.output_port.value) == memory_capacity, \
            'PROGRAM ERROR: number of items in combined_matches_node ' \
            f'({len(self.combined_matches_node.output_port)}) does not match memory_capacity ({self.memory_capacity})'

    def _construct_softmax_node(self, memory_capacity, softmax_gain, softmax_threshold, softmax_choice):
        """Create node that applies softmax to output of combined_matches_node."""
        if self.num_keys == 1 or self.concatenate_queries_node:
            input_source = self.match_nodes[0]
            proj_name =f'{MATCH} to {SOFTMAX_NODE_NAME}'
        else:
            input_source = self.combined_matches_node
            proj_name =f'{COMBINE_MATCHES_NODE_NAME} to {SOFTMAX_NODE_NAME}'

        if softmax_choice == ARG_MAX:
            # ARG_MAX would return entry multiplied by its dot product
            # ARG_MAX_INDICATOR returns the entry unmodified
            softmax_choice = ARG_MAX_INDICATOR

        self.softmax_node = ProcessingMechanism(name=SOFTMAX_NODE_NAME,
                                                input_ports={INPUT_SHAPES: memory_capacity,
                                                             PROJECTIONS: MappingProjection(
                                                                 sender=input_source,
                                                                 matrix=IDENTITY_MATRIX,
                                                                 name=proj_name)},
                                                function=SoftMax(gain=softmax_gain,
                                                                 mask_threshold=softmax_threshold,
                                                                 output=softmax_choice,
                                                                 adapt_entropy_weighting=.95))

    def _construct_retrieved_nodes(self, memory_template)->list:
        """Create nodes that report the value field(s) for the item(s) matched in memory.
        """
        for field in self.fields:
            field.retrieved_node = (
                ProcessingMechanism(name=field.name + RETRIEVED_AFFIX,
                                    input_ports={INPUT_SHAPES: len(field.input_node.variable[0]),
                                                 PROJECTIONS:
                                                     MappingProjection(
                                                         sender=self.softmax_node,
                                                         matrix=memory_template[:,field.index],
                                                         name=f'MEMORY FOR {field.name} '
                                                              f'[RETRIEVE {field.type.name}]')}))
            field.retrieve_projection = field.retrieved_node.path_afferents[0]

    def _construct_storage_node(self,
                                use_storage_node,
                                memory_template,
                                memory_decay_rate,
                                storage_prob):
        """Create EMStorageMechanism that stores the key and value inputs in memory.
        Memories are stored by adding the current input to each field to the corresponding row of the matrix for
        the Projection from the query_input_node (or concatenate_node) to the matching_node and retrieved_node for keys,
        and from the value_input_node to the retrieved_node for values. The `function <EMStorageMechanism.function>`
        of the `EMSorageMechanism` that takes the following arguments:

         - **variable** -- template for an `entry <GRUComposition_Memory_Specification>`
           in `memory<GRUComposition.memory>`;

         - **fields** -- the `input_nodes <GRUComposition.input_nodes>` for the corresponding `fields
           <GRUComposition_Fields>` of an `entry <EMCmposition_Memory>` in `memory <GRUComposition.memory>`;

         - **field_types** -- a list of the same length as ``fields``, containing 1's for key fields and 0's for
           value fields;

         - **concatenate_queries_node** -- node used to concatenate keys
           (if `concatenate_queries <GRUComposition.concatenate_queries>` is `True`) or None;

         - **memory_matrix** -- `memory_template <GRUComposition.memory_template>`);

         - **learning_signals** -- list of ` `MappingProjection`\\s (or their ParameterPort`\\s) that store each
           `field <GRUComposition_Fields>` of `memory <GRUComposition.memory>`;

         - **decay_rate** -- rate at which entries in the `memory_matrix <GRUComposition.memory_matrix>` decay;

         - **storage_prob** -- probability for storing an entry in `memory <GRUComposition.memory>`.
        """
        if use_storage_node:
            learning_signals = [match_node.input_port.path_afferents[0]
                                for match_node in self.match_nodes] + [retrieved_node.input_port.path_afferents[0]
                                for retrieved_node in self.retrieved_nodes]
            self.storage_node = (
                EMStorageMechanism(default_variable=[field.input_node.value[0] for field in self.fields],
                                   fields=[field.input_node for field in self.fields],
                                   field_types=[1 if field.type is FieldType.KEY else 0 for field in self.fields],
                                   concatenation_node=self.concatenate_queries_node,
                                   memory_matrix=memory_template,
                                   learning_signals=learning_signals,
                                   storage_prob=storage_prob,
                                   decay_rate = memory_decay_rate,
                                   name=STORE_NODE_NAME))
            for field in self.fields:
                field.storage_projection = self.storage_node.path_afferents[field.index]

    def _set_learning_attributes(self):
        """Set learning-related attributes for Node and Projections
        """
        # 7/10/24 FIX: SHOULD THIS ALSO BE CONSTRAINED BY VALUE OF field_weights FOR CORRESPONDING FIELD?
        #         (i.e., if it is zero then not learnable? or is that a valid initial condition?)
        for projection in self.projections:

            projection_is_field_weight = projection.sender.owner in self.field_weight_nodes

            if self.enable_learning is False or not projection_is_field_weight:
                projection.learnable = False
                continue

            # Use globally specified learning_rate
            if self.learn_field_weights is None: # Default, which should be treat same as True
                learning_rate = True
            elif isinstance(self.learn_field_weights, (bool, int, float)):
                learning_rate = self.learn_field_weights
            # Use individually specified learning_rate
            else:
                # FIX: THIS NEEDS TO USE field_index_map, BUT THAT DOESN'T SEEM TO HAVE THE WEIGHT PROJECTION YET
                learning_rate = self.learn_field_weights[self._field_index_map[projection]]

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

    def _validate_options_with_learning(self,
                                        use_gating_for_weighting,
                                        enable_learning,
                                        softmax_choice):
        if use_gating_for_weighting and enable_learning:
            warnings.warn(f"The 'enable_learning' option for '{self.name}' cannot be used with "
                          f"'use_gating_for_weighting' set to True; this will generate an error if its "
                          f"'learn' method is called. Set 'use_gating_for_weighting' to True in order "
                          f"to enable learning of field weights.")

        if softmax_choice in {ARG_MAX, PROBABILISTIC} and enable_learning:
            warnings.warn(f"The 'softmax_choice' arg of '{self.name}' is set to '{softmax_choice}' with "
                          f"'enable_learning' set to True; this will generate an error if its "
                          f"'learn' method is called. Set 'softmax_choice' to WEIGHTED_AVG before learning.")


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

    def _store_memory(self, inputs, context):
        """Store inputs to query and value nodes in memory
        Store memories in weights of Projections to match_nodes (queries) and retrieved_nodes (values).
        Note: inputs argument is ignored (included for compatibility with function of MemoryFunctions class;
              storage is handled by call to EMComopsition._encode_memory
        """
        storage_prob = np.array(self._get_current_parameter_value(STORAGE_PROB, context)).astype(float)
        random_state = self._get_current_parameter_value('random_state', context)

        if storage_prob == 0.0 or (storage_prob > 0.0 and storage_prob < random_state.uniform()):
            return
        # self._encode_memory(inputs, context)
        self._encode_memory(context)

    def _encode_memory(self, context=None):
        """Encode inputs as memories
        For each node in query_input_nodes and value_input_nodes,
        assign its value to afferent weights of corresponding retrieved_node.
        - memory = matrix of entries made up vectors for each field in each entry (row)
        - memory_full_vectors = matrix of entries made up vectors concatentated across all fields (used for norm)
        - entry_to_store = query_input or value_input to store
        - field_memories = weights of Projections for each field
        """

        # Get least used slot (i.e., weakest memory = row of matrix with lowest weights) computed across all fields
        field_norms = np.array([np.linalg.norm(field, axis=1)
                                for field in [row for row in self.parameters.memory.get(context)]])
        if self.purge_by_field_weights:
            field_norms *= self.field_weights
        row_norms = np.sum(field_norms, axis=1)
        idx_of_min = np.argmin(row_norms)

        # If concatenate_queries=True, assign entry to col of matrix for Projection from concatenate_node to match_node
        if self.concatenate_queries_node:
            # Get entry to store from concatenate_queries_node
            entry_to_store = self.concatenate_queries_node.value[0]
            # Get matrix of weights for Projection from concatenate_node to match_node
            field_memories = self.concatenate_queries_node.efferents[0].parameters.matrix.get(context)
            # Decay existing memories before storage if memory_decay_rate is specified
            if self.memory_decay_rate:
                field_memories *= self.parameters.memory_decay_rate._get(context)
            # Assign input vector to col of matrix that has lowest norm (i.e., weakest memory)
            field_memories[:,idx_of_min] = np.array(entry_to_store)
            # Assign updated matrix to Projection
            self.concatenate_queries_node.efferents[0].parameters.matrix.set(field_memories, context)

        # Otherwise, assign input for each key field to col of matrix for Projection from query_input_node to match_node
        else:
            for i, input_node in enumerate(self.query_input_nodes):
                # Get entry to store from query_input_node
                entry_to_store = input_node.value[0]
                # Get matrix of weights for Projection from query_input_node to match_node
                field_memories = input_node.efferents[0].parameters.matrix.get(context)
                # Decay existing memories before storage if memory_decay_rate is specified
                if self.memory_decay_rate:
                    field_memories *= self.parameters.memory_decay_rate._get(context)
                # Assign query_input vector to col of matrix that has lowest norm (i.e., weakest memory)
                field_memories[:,idx_of_min] = np.array(entry_to_store)
                # Assign updated matrix to Projection
                input_node.efferents[0].parameters.matrix.set(field_memories, context)

        # For each key and value field, assign input to row of matrix for Projection to retrieved_nodes
        for i, input_node in enumerate(self.query_input_nodes + self.value_input_nodes):
            # Get entry to store from query_input_node or value_input_node
            entry_to_store = input_node.value[0]
            # Get matrix of weights for Projection from input_node to match_node
            field_memories = self.retrieved_nodes[i].path_afferents[0].parameters.matrix.get(context)
            # Decay existing memories before storage if memory_decay_rate is specified
            if self.memory_decay_rate:
                field_memories *= self.memory_decay_rate
            # Assign input vector to col of matrix that has lowest norm (i.e., weakest memory)
            field_memories[idx_of_min] = np.array(entry_to_store)
            # Assign updated matrix to Projection
            self.retrieved_nodes[i].path_afferents[0].parameters.matrix.set(field_memories, context)

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
    @property
    def input_nodes(self):
        return [field.input_node for field in self.fields]

    @property
    def query_input_nodes(self):
        return [field.input_node for field in self.fields if field.type == FieldType.KEY]

    @property
    def value_input_nodes(self):
        return [field.input_node for field in self.fields if field.type == FieldType.VALUE]

    @property
    def match_nodes(self):
        if self.concatenate_queries_node:
            return [self.concatenated_match_node]
        else:
            return [field.match_node for field in self.fields if field.type == FieldType.KEY]

    @property
    def field_weight_nodes(self):
        return [field.weight_node for field in self.fields
                if field.weight_node and field.type == FieldType.KEY]

    @property
    def weighted_match_nodes(self):
        return [field.weighted_match_node for field in self.fields
                if field.weighted_match_node and (field.type == FieldType.KEY)]

    @property
    def retrieved_nodes(self):
        return [field.retrieved_node for field in self.fields]

    #endregion
