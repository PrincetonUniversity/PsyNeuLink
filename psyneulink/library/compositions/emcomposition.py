# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* EMComposition *************************************************

"""

Contents
--------

  * `EMComposition_Overview`
  * `EMComposition_Creation`
  * `EMComposition_Execution`
  * `EMComposition_Examples`
  * `EMComposition_Class_Reference`


.. _EMComposition_Overview:

Overview
--------

Implements a differentiable version of an `EpisodicMemoryMechanism` as a `Composition`, that can serve as a form
of episodic, or external memory in an `AutodiffComposition` capable of learning. It implements all of the functions
of a `ContentAddressableMemory` `Function` used by an `EpisodicMemoryMechanism`, and takes all of the same arguments.

.. _EMComposition_Creation:

Creation
--------

.. _EMComposition_Execution:

Execution
---------

.. _EMComposition_Examples:

Examples
--------

.. _EMComposition_Class_Reference:

Class Reference
---------------
"""
import numpy as np
import warnings

from psyneulink._typing import Optional, Union

from psyneulink.core.components.functions.nonstateful.transferfunctions import SoftMax
from psyneulink.core.components.functions.nonstateful.combinationfunctions import Concatenate
from psyneulink.core.compositions.composition import Composition, CompositionError
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.mechanisms.modulatory.control.gating.gatingmechanism import GatingMechanism
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.keywords import \
    EM_COMPOSITION, FUNCTION, IDENTITY_MATRIX, NAME, OUTCOME, PROJECTIONS, RESULT, SIZE, VALUE, ZEROS_MATRIX

__all__ = [
    'EMComposition'
]

class EMCompositionError(CompositionError):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class EMComposition(Composition):
    """
    Subclass of `Composition` that implements the functions of an `EpisodicMemoryMechanism` in a differentiable form.

    All of the arguments of the `ContentAddressableMemory` `Function` can be specified in the constructor for the
    EMComposition.  In addition, the following arguments can be specified:

    TODO:
    - DECAY WEIGHTS BY:
      ? 1-SOFTMAX / N (WHERE N = NUMBER OF ITEMS IN MEMORY)
      or
         1/N (where N=number of items in memory, and thus gets smaller as N gets
         larger) on each storage (to some asymptotic minimum value), and store the new memory to the unit with the
         smallest weights (randomly selected among “ties" [i.e., within epsilon of each other]), I think we have a
         mechanism that can adaptively use its limited capacity as sensibly as possible, by re-cycling the units
         that have the least used memories.
    - TEST ADPATIVE TEMPERATURE (SEE BELOW)
    - ADD ALL ARGS FOR CONTENTADDRESSABLEMEMORY FUNCTION TO INIT, AND MAKE THEM Parameters

    Arguments
    ---------

    memory_template : list or 2d array : default [[0],[0]]
        specifies the shape of an items to be stored in the EMComposition's memory;  a template (vs. a shape tuple) is
        used to allow fields of different lengths to be specified (i.e., items to be encoded can be ragged arrays).

    memory_capacity : int : default 1000
        specifies the number of items that can be stored in the EMComposition's memory.

    field_weights : tuple : default (1,0)
        specifies the relative weight assigned to each key when matching an item in memory. this is used both to
        specify which fields are used as keys, and to initialize the Projections to the `retrieval_weighting_node
        <EMComposition.retrieval_weighting_node>`.  The number of values specified must match the number of fields specified
        in the memory_template. Non zero values must be positive, and are used to specify keys — fields that are used
        to match items in memory for retieval.  If a single value is
        specified, it is applied to all fields.  If more than a single value is specified, the number must match the
        number of items in the first dimension (axis 0) of **memory_template**. If all items are identical, the field
        weights are treated as a single field (this is identical to specifying a single value, but can be used to allow
        different keys to receive independent inputs from the network);  in this case, all keys are concatenated before
        a match is made in memory If the items in **field_weights** are non-identical, they are applied to the
        corresponding fields during the matching process.  If a zero is specified, that field is not used in the
        matching process, but the value= corresponding to memory that is selected is still returned; this implements
        a "dictionary" in which the fields with non-zero weights are keys and the one(s) with zero weights are values.
        If **learn_weights** is True, the weights are learned during training;  otherwise they remain fixed.

    learn_weights : bool : default False
        specified whether field_weights are learnable during training.

    learning_rate : float : default 0.001
        the learning rate passed to the optimizer if none is specified in the learn method of the EMComposition.

    Attributes
    ----------


    """

    componentCategory = EM_COMPOSITION
    class Parameters(Composition.Parameters):
        learning_rate = Parameter(.001, fallback_default=True)
        losses = Parameter([])
        pytorch_representation = None
    @check_user_specified
    def __init__(self,
                 memory_template:Union[list, np.ndarray]=[[0],[0]],
                 field_names:Optional[list]=None,
                 field_weights:tuple=None,
                 learn_weights:bool=True,
                 memory_capacity:int=1000,
                 decay_memories:bool=False,
                 memory_decay_rate:Optional[float]=None,
                 learning_rate:Optional[float]=None,
                 name="EM_composition"):

        # memory_template must specify a 2D array:
        if np.array(memory_template).ndim != 2:
            raise EMCompositionError(f"The 'memory_template' arg for {name} ({memory_template}) "
                                     f"must be list of lists or a 2d array.")

        if field_weights is None:
            if len(memory_template) == 1:
                field_weights = [1]
            else:
                # Default is to all fields as keys except the last one, which is the value
                num_fields = len(memory_template)
                field_weights = [1] * num_fields
                field_weights[-1] = 0

        # If field weights has more than one value it must match the first dimension (axis 0) of memory_template:
        if len(field_weights) > 1 and len(field_weights) != len(memory_template):
            raise EMCompositionError(f"The number of items ({len(field_weights)}) "
                                     f"in the 'field_weights' arg for {name} must match the number of items "
                                     f"({len(memory_template)}) in the outer dimension of its 'memory_template' arg.")

        # Memory structure (field) parameters
        self.memory_template = memory_template.copy()
        self.num_fields = len(memory_template)
        self.field_weights = field_weights
        # self.field_names = field_names.copy()
        self.learn_weights = learn_weights
        self.learning_rate = learning_rate # FIX: MAKE THIS A PARAMETER

        # Memory management parameters
        self.memory_capacity = memory_capacity # FIX: MAKE THIS A READ-ONLY PARAMETER
        self.decay_memories = decay_memories # FIX: MAKE THIS A PARAMETER
        self.memory_decay_rate = memory_decay_rate or 1 / self.memory_capacity # FIX: MAKE THIS A PARAMETER
        if self.decay_memories and memory_decay_rate == 0.0:
            raise warnings.warn(f"The 'decay_memories' arg was set to True but 'memory_decay_rate' was set to 0.0; "
                                f"default of 1/memory_capacity ({self.memory_decay_rate}) will be used instead.")

        self.concatenate_keys = len(self.field_weights) == 1 or np.all(self.field_weights == self.field_weights[0])
        self.num_keys = len([i for i in self.field_weights if i != 0])
        self.num_values = self.num_fields - self.num_keys

        pathway = self._construct_pathway()

        super().__init__(pathway,
                         name=name)

        # Turn of learning for all Projections except inputs to retrieval_gating_nodes
        self._set_learnability_of_projections()
        self._initialize_memory()

    def _construct_pathway(self):
        """Construct pathway for EMComposition"""

        # Construct nodes of Composition
        self.key_input_nodes = self._construct_key_input_nodes()
        self.value_input_nodes = self._construct_value_input_nodes()
        self.match_nodes = self._construct_match_nodes()
        self.retrieval_gating_nodes = self._construct_retrieval_gating_nodes()
        self.retrieval_weighting_node = self._construct_retrieval_weighting_node()
        self.retrieval_nodes = self._construct_retrieval_nodes()

        # Construct pathway as a set of nodes, since Projections are specified in the construction of each node
        pathway = set(self.key_input_nodes + self.value_input_nodes + self.match_nodes \
                + [self.retrieval_weighting_node]
                + self.retrieval_gating_nodes + self.retrieval_nodes)

        return pathway

    def _construct_key_input_nodes(self):
        """Create one node for each key to be used as cue for retrieval (and then stored) in memory.
        Used to assign new set of weights for Projection for key_input_node[i] -> match_node[i]
        where i is selected randomly without replacement from (0->memory_capacity)
        """
        
        # Get indices of field_weights that specify keys:
        key_indices = np.nonzero(self.field_weights)[0]

        assert len(key_indices) == self.num_keys, \
            f"PROGRAM ERROR: number of keys ({self.num_keys}) does not match number of " \
            f"non-zero values in field_weights ({len(key_indices)})."

        key_input_nodes = [TransferMechanism(size=len(self.memory_template[key_indices[i]]),
                                               name= 'KEY INPUT NODE' if self.num_keys == 1
                                               else f'KEY INPUT NODE {i}')
                       for i in range(self.num_keys)]

        return key_input_nodes

    def _construct_value_input_nodes(self):
        """Create one input node for each value to be stored in memory.
        Used to assign new set of weights for Projection for retrieval_weighting_node -> retrieval_node[i]
        where i is selected randomly without replacement from (0->memory_capacity)
        """

        # Get indices of field_weights that specify keys:
        value_indices = np.where(np.array(self.field_weights) == 0)[0]

        assert len(value_indices) == self.num_values, \
            f"PROGRAM ERROR: number of values ({self.num_values}) does not match number of " \
            f"non-zero values in field_weights ({len(value_indices)})."

        value_input_nodes = [TransferMechanism(size=len(self.memory_template[value_indices[i]]),
                                               name= 'VALUE INPUT NODE' if self.num_values == 1
                                               else f'VALUE INPUT NODE {i}')
                           for i in range(self.num_values)]

        return value_input_nodes

    def _construct_match_nodes(self):
        """Create nodes that, for each key field, compute the similarity between the input and each item in memory
        and return softmax over those similarities.

        If self.concatenate_keys is True, then all inputs for keys are concatenated into a single vector that is
        the input to a single TransferMechanism;  otherwise, each key has its own TransferMechanism.  The size of
        the input to each TransferMechanism is the number of items allowed in memory, and the weights to each element
        in the weight matrix is a single memory.
        """

        # Get indices of field_weights that specify keys:
        key_indices = np.where(np.array(self.field_weights) != 0)
        key_weights = [self.field_weights[i] for i in key_indices[0]]

        assert len(key_indices[0]) == self.num_keys, \
            f"PROGRAM ERROR: number of keys ({self.num_keys}) does not match number of " \
            f"non-zero values in field_weights ({len(key_indices)})."

        if self.concatenate_keys:
            # One node for single key that concatenates all inputs
            match_nodes = [TransferMechanism(size=self.num_keys * self.memory_capacity,
                                             input_ports={NAME: 'CONCATENATED_INPUTS',
                                                          FUNCTION: Concatenate()},
                                             function=SoftMax(gain=self.softmax_temperature(self.field_weights)),
                                             output_ports=[RESULT,
                                                           {VALUE: key_weights[0],
                                                            NAME: 'KEY_WEIGHT'}],
                                             name='MATCH NODE')]
        else:
            # One node for each key
            match_nodes = [
                TransferMechanism(
                    input_ports=
                    {
                        SIZE:self.memory_capacity,
                        PROJECTIONS: self.key_input_nodes[i].output_port
                        # PROJECTIONS:
                        #     MappingProjection(
                        #         sender=self.key_input_nodes[i].output_port,
                        #         matrix=ZEROS_MATRIX)
                    },
                    # (self.memory_capacity,
                    #  MappingProjection(sender=self.key_input_nodes[i].output_port, matrix=ZEROS_MATRIX)),
                    function=SoftMax(),
                    output_ports=[RESULT,
                                  {VALUE: key_weights[0],
                                   NAME: 'KEY_WEIGHT'}],
                    name='MATCH_NODE ' + str(i))
                for i in range(self.num_keys)
            ]

        return match_nodes

    def _construct_retrieval_gating_nodes(self):
        """Create GatingMechanisms that weight each key's contribution to the retrieved values.
        """
        retrieval_gating_nodes = [GatingMechanism(#input_ports=key_match_pair[0].output_ports['RESULT'],
                                                  input_ports={PROJECTIONS: key_match_pair[0].output_ports['RESULT'],
                                                               NAME: 'OUTCOME'},
                                                 gate=[key_match_pair[1].output_ports[1]],
                                                 # name=f'RETRIEVAL GATING NODE {i}')
                                                 name= 'RETRIEVAL GATING NODE' if self.num_keys == 1
                                                 else f'RETRIEVAL GATING NODE {i}')
                                 for i, key_match_pair in enumerate(zip(self.key_input_nodes, self.match_nodes))]
        return retrieval_gating_nodes

    def _construct_retrieval_weighting_node(self):
        """Create layer that computes the weighting of each item in memory.
        """
        # FIX: THIS SHOULD WORK:
        # retrieval_weighting_node = TransferMechanism(input_ports=[{PROJECTIONS: [m.output_port for m in self.match_nodes]}])
        retrieval_weighting_node = TransferMechanism(input_ports=[m.output_port for m in self.match_nodes],
                                                     name='RETRIEVAL WEIGHTING NODE')

        assert len(retrieval_weighting_node.output_port.value) == self.memory_capacity,\
            f'PROGRAM ERROR: number of items in retrieval_weighting_node ({len(retrieval_weighting_node.output_port)})' \
            f'does not match memory_capacity ({self.memory_capacity})'
                                                            
        return retrieval_weighting_node

    def _construct_retrieval_nodes(self):
        """Create layer that reports the value field(s) for the item(s) matched in memory.
        """
        retrieval_nodes = [TransferMechanism(size=self.memory_capacity,
                                             # input_ports=self.value_input_nodes[i],
                                             input_ports=self.retrieval_weighting_node,
                                             name= 'VALUE NODE' if self.num_values == 1 else f'VALUE NODE {i}')
                           for i in range(self.num_values)]
        return retrieval_nodes

    def _set_learnability_of_projections(self):
        """Turn off learning for all Projections except afferents to retrieval_gating_nodes.
        """
        for node in self.nodes:
            for input_port in node.input_ports:
                for proj in input_port.path_afferents:
                    if node in self.retrieval_gating_nodes:
                        proj.learnable = self.learn_weights
                    else:
                        proj.learnable = False

    def _initialize_memory(self):
        """Initialize memory by zeroing weights from:
        - key_input_node(s) to match_node(s) and
        - retrieval_weight_node to retrieval_node(s) .
        """
        for key_node, proj in zip(self.key_input_nodes, [key_node.efferents[0]
                                                         for key_node in self.key_input_nodes]):
            proj.matrix.base = np.zeros_like(key_node.efferents[0].matrix.base)
            proj.execute() # For clarity, ensure that it reports modulated value as zero as well

        for retrieval_node, proj in zip(self.retrieval_nodes, [retrieval_node.path_afferents[0]
                                                               for retrieval_node in self.retrieval_nodes]):
            proj.matrix.base = np.zeros_like(retrieval_node.path_afferents[0].matrix.base)
            proj.execute() # For clarity, ensure that it reports modulated value as zero as well
            
    def softmax_temperature(self, values, epsilon=1e-8):
        """Compute the softmax temperature based on length of vector and number of (near) zero values.
        """
        n = len(values)
        num_zero = np.count_nonzero(values < epsilon)
        num_non_zero = n - num_zero
        gain = 1 + np.exp(1/num_non_zero) * (num_zero/n)
        return gain

    def execute(self, inputs, **kwargs):
        """Set input to weights of Projection to match_node.
        """
        super().execute(**kwargs)
        self._store_memory(inputs)

    def _store_memory(self, inputs):
        """Store inputs in memory as weights of Projections to match_nodes (keys) and retrieval_nodes (values).
        """
        key_inputs = inputs[:self.num_keys]
        value_inputs = inputs[self.num_keys:]

        # Store memories of keys
        for memory, key_input_node in zip(key_inputs, self.key_input_nodes):
            # Memory = key_input;
            #   assign as weights for first empty row of Projection.matrix from key_input_node to match_node
            memories = key_input_node.efferents[0]
            if self.decay_memories:
                memory.matrix.base *= self.memory_decay_rate
            # Get least used slot (i.e., weakest memory = row of matrix with lowest weights)
            idx_of_min = np.argmin(memories.matrix.base.sum(axis=1))
            memories.matrix.base[idx_of_min] = memory

        # Store memories of values
        for memory, retrieval_node in zip(value_inputs, self.retieval_nodes):
            # Memory = value_input;
            #   assign as weights for 1st empty row of Projection.matrix from retrieval_weighting_node to retrieval_node
            memories = retrieval_node.path_afferents[0]
            if self.decay_memories:
                memories.matrix.base *= self.memory_decay_rate
            # Get least used slot (i.e., weakest memory = row of matrix with lowest weights)
            idx_of_min = np.argmin(memories.matrix.base.sum(axis=1))
            memories.matrix.base[min] = memory
