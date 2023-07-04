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
    EM_COMPOSITION, FUNCTION, IDENTITY_MATRIX, NAME, OUTCOME, PROJECTIONS, RESULT, SIZE, VALUE

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

    disable_learning : bool: default False
        specifies whether the EMComposition should disable learning when run in `learning mode
        <Composition.learn>`.

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
                 learning_rate:Optional[float]=None,
                 disable_learning:bool=False,
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

        self.memory_template = memory_template.copy()
        self.num_fields = len(memory_template)
        self.field_weights = field_weights
        # self.field_names = field_names.copy()

        self.concatenate_keys = len(self.field_weights) == 1 or np.all(self.field_weights == self.field_weights[0])
        self.num_keys = len([i for i in self.field_weights if i != 0])
        self.num_values = self.num_fields - self.num_keys
        self.learn_weights = learn_weights
        self.learning_rate = learning_rate # FIX: MAKE THIS A PARAMETER
        self.memory_capacity = memory_capacity # FIX: MAKE THIS A READ-ONLY PARAMETER
        self.disable_learning = disable_learning

        super().__init__(name=name,
                         pathway=self._create_components())


    def _create_components(self):
        self.key_input_nodes = self._create_key_input_nodes()
        self.value_input_nodes = self._create_value_input_nodes()
        self.match_nodes = self._create_match_nodes()
        self.retrieval_gating_node = self._create_retrieval_gating_node()
        self.retrieval_weighting_node = self._create_retrieval_weighting_node()
        self.retrieval_nodes = self._create_retrieval_nodes()
        nodes = self.key_input_nodes + self.value_input_nodes + self.match_nodes \
                + [self.retrieval_weighting_node] \
                + self.retrieval_nodes
        # Return as a set since Projections are specified in the construction of each node
        return set(nodes)

    def _create_key_input_nodes(self):
        """Create layer with one Input node for each key in the memory template."""
        
        # Get indices of field_weights that specify keys:
        key_indices = np.nonzero(self.field_weights)[0]

        assert len(key_indices) == self.num_keys, \
            f"PROGRAM ERROR: number of keys ({self.num_keys}) does not match number of " \
            f"non-zero values in field_weights ({len(key_indices)})."

        key_input_nodes = [TransferMechanism(size=len(self.memory_template[key_indices[i]]),
                                             name='key_input_node_' + str(i))
                       for i in range(self.num_keys)]

        return key_input_nodes

    def _create_value_input_nodes(self):
        """Create layer with one Input node for each key in the memory template."""

        # Get indices of field_weights that specify keys:
        value_indices = np.where(np.array(self.field_weights) == 0)[0]

        assert len(value_indices) == self.num_values, \
            f"PROGRAM ERROR: number of values ({self.num_values}) does not match number of " \
            f"non-zero values in field_weights ({len(value_indices)})."

        value_input_nodes = [TransferMechanism(size=len(self.memory_template[value_indices[i]]),
                                             name='value_input_node_' + str(i))
                           for i in range(self.num_values)]

        return value_input_nodes

    def _create_match_nodes(self):
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
                                             name='match_node')]
        else:
            # One node for each key
            match_nodes = [TransferMechanism(input_ports={SIZE:self.memory_capacity,
                                                          PROJECTIONS: self.key_input_nodes[i].output_port},
                                             function=SoftMax(),
                                             output_ports=[RESULT,
                                                           {VALUE: key_weights[0],
                                                            NAME: 'KEY_WEIGHT'}],
                                             name='match_node_' + str(i))
                           for i in range(self.num_keys)]

        return match_nodes

    def _create_retrieval_gating_node(self):
        """Create GatingMechanism that weights each key's contribution to the retrieval."""
        retrieval_gating_node = GatingMechanism(default_allocation=[1] * self.num_keys,
                                                # FIX: THIS SHOULD WORK:
                                                # input_ports=[m.output_ports['KEY_WEIGHT'] for m in self.match_nodes],
                                                input_ports=[{FUNCTION: Concatenate(),
                                                              PROJECTIONS: [m.output_ports['KEY_WEIGHT']
                                                                            for m in self.match_nodes],
                                                              NAME: OUTCOME
                                                              }],
                                                gate=[m.output_ports[1] for m in self.match_nodes],
                                                # gating_signals=[m.output_port for m in self.match_nodes],
                                                name='retrieval_gating_node')

    def _create_retrieval_weighting_node(self):
        """Create layer that computes the weighting of each item in memory."""

        # if self.concatenate_keys:
        #     # Only one match_node node (with set of softmax weights across all concatenated keys)
        #     retrieval_weighting_node = TransferMechanism(input_ports=self.match_node[0].output_port)
        #
        # else:
        #     # Sum softmax values for each key weighted by the corresponding field_weight value
        #     key_weights = [w for w in self.field_weights if w != 0]
        #     # Assign an InputPort for each key,
        #     #   that receives a MappingProjection from the corresponding node in the match_node
        #     #   with a weight matrix that is initialized with the corresponding field_weight value
        #     input_ports = [{PROJECTIONS: MappingProjection(sender = m,#.output_port,
        #                                                    matrix = np.identity(self.memory_capacity) * key_weights[0],
        #                                                    learnable = self.learn_weights)}
        #                    for i, m in enumerate(self.match_nodes)]
        #     retrieval_weighting_node = TransferMechanism(input_ports=input_ports,
        #                                                   name='retrieval_weighting_node')
        # else:
        #     # Sum softmax values for each key weighted by the corresponding field_weight value
        #     key_weights = [w for w in self.field_weights if w != 0]
        #     # Assign an InputPort for each key,
        #     #   that receives a MappingProjection from the corresponding node in the match_node
        #     #   with a weight matrix that is initialized with the corresponding field_weight value
        #     input_ports = [{PROJECTIONS: MappingProjection(sender = m,#.output_port,
        #                                                    matrix = np.identity(self.memory_capacity) * key_weights[0],
        #                                                    learnable = self.learn_weights)}
        #                    for i, m in enumerate(self.match_nodes)]
        #     retrieval_weighting_node = TransferMechanism(input_ports=input_ports,
        #                                                   name='retrieval_weighting_node')

        projections = [MappingProjection(sender=m.output_port,matrix=IDENTITY_MATRIX) for m in self.match_nodes]
        retrieval_weighting_node = TransferMechanism(input_port={PROJECTIONS:projections})

        assert len(retrieval_weighting_node.output_port) == self.memory_capacity,\
            f'PROGRAM ERROR: number of items in retrieval_weighting_node ({len(retrieval_weighting_node.output_port)})' \
            f'does not match memory_capacity ({self.memory_capacity})'
                                                            
        return retrieval_weighting_node

    def _create_retrieval_nodes(self):
        """Create layer that reports the value field(s) for the item matched in memory."""
        
        retrieval_nodes = [TransferMechanism(size=self.memory_capacity,
                                             input_port=self.value_input_nodes[i],
                                             name='value_node_' + str(i))
                           for i in range(self.num_values)]
        return retrieval_nodes

    def softmax_temperature(self, values, epsilon=1e-8):
        """Compute the softmax temperature based on length of vector and number of (near) zero values."""
        n = len(values)
        num_zero = np.count_nonzero(values < epsilon)
        num_non_zero = n - num_zero
        gain = 1 + np.exp(1/num_non_zero) * (num_zero/n)
        return gain

    def execute(self, **kwargs):
        """Set input to weights of Projection to match_node."""
        super().execute(**kwargs)
