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
                [self.input_node,
                self.hidden_layer_node,
                self.reset_gate_node,
                self.update_gate_node,
                self.new_gate_node,
                if node is not None]

    @property
    def projections(self):
        """Return all Projections assigned to the field."""
        return [proj for proj in [self.input_to_hidden_projection
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
        `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` that receives...

    hidden_layer_node : list[ProcessingMechanism]
        `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` that receives...

    reset_gate_node : list[ProcessingMechanism]
        `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` that receives

    update_gate_node : list[ProcessingMechanism]
        `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` that receives

    new_gate_node : list[ProcessingMechanism]
        `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` that receives

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
