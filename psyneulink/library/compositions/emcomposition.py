# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* EMComposition *************************************************


# ISSUES:
# - IMPLEMENTATION OF STORAGE IN PYTORCH AND/OR AS LEARNING PROCESS (HEBBIAN?)
# - CONFIDENCE COMPUTATION

# TODO:
# - ACCESSIBILITY OF DISTANCES (SEE BELOW): MAKE IT A LOGGABLE PARAMETER (I.E., WITH APPROPRIATE SETTER)
#   ADD COMPILED VERSION OF NORMED LINEAR_COMBINATION FUNCTION TO LinearCombination FUNCTION: dot / (norm a * norm b)
# - DECAY WEIGHTS BY:
#   ? 1-SOFTMAX / N (WHERE N = NUMBER OF ITEMS IN MEMORY)
#   or
#      1/N (where N=number of items in memory, and thus gets smaller as N gets
#      larger) on each storage (to some asymptotic minimum value), and store the new memory to the unit with the
#      smallest weights (randomly selected among “ties" [i.e., within epsilon of each other]), I think we have a
#      mechanism that can adaptively use its limited capacity as sensibly as possible, by re-cycling the units
#      that have the least used memories.
# - ADAPTIVE TEMPERATURE:
#   - ADD COMPUTATION OF GAIN TO _store_memory METHOD
# - MAKE "_store_memory" METHOD USE LEARNING INSTEAD OF ASSIGNMENT (per Steven's Hebban / DPP model?)
# - ADD ADDITIONAL PARAMETERS FROM CONTENTADDRESSABLEMEMORY FUNCTION
#   - KAMESH FOR FORMULA
#   - ADD SOFTMAX TEMP PARAMETER (with None = Auto) SCALE PARAMETER TO SOFTMAX FUNCTION THAT SCALES GAIN RETURNED
# - ADD MEMORY_DECAY TO ContentAddressableMemory FUNCTION (and compiled version by Samyak)

# FIX: COMPILE
#      LinearMatrix to add normalization
#      _store() method to assign weights to memory

"""

Contents
--------

  * `EMComposition_Overview`
     - `Organization <EMComposition_Organization>`
     - `Operation <EMComposition_Operation>`
  * `EMComposition_Creation`
     - `Fields <EMComposition_Fields>`
     - `Capacity <EMComposition_Capacity>`
     - `Learning <EMComposition_Learning>`
  * `EMComposition_Structure`
  * `EMComposition_Execution`
  * `EMComposition_Examples`
  * `EMComposition_Class_Reference`


.. _EMComposition_Overview:

Overview
--------

The EMComposition implements a configurable, content-addressable form of episodic, or eternal memory, that emulates
the functionality  of an `EpisodicMemoryMechanism` -- reproducing all of the functionality of its
`ContentAddressableMemory` `Function` -- in the form of an `AutodiffComposition` that is capable of learning its
`field_weights <EMComposition.field_weights>` (i.e., that weight the keys used to retrieve items from
memory), and that adds the capability for `memory_decay <EMComposition_Memory_Decay>`.  Its `memory
<EMComposition.memory>` is configured using the **memory_template** argument of its constructor, which defines how
each entry in `memory <EMComposition.memory>` is structured (i.e., the number of fields it has, and the length of
each field), as described below.  The inputs to its `key_input_nodes <EMComposition.key_input_nodes>` are used to
retrieve entries from `memory <EMComposition.memory>`, and its `retrieval_nodes <EMComposition.retrieval_nodes>`
report the results of retrieval.

.. _EMComposition_Organization:

**Organization**

*Entries and Fields*. Each entry in memory can have an arbitrary number of fields, and each field can have an arbitrary
length.  However, all entries must have the same number of fields, and the corresponding fields must all have the same
length across entries. Fields can be weighted to determine the influence they have on retrieval, using the
`field_weights <ContentAddressableMemory.memory>` parameter (see `retrieval <EMComposition_Retrieval>` below). The
number and shape of the fields in each entry is specified in the **memory_template** argument of the EMComposition's
constructor (see `memory_template <EMComposition_Fields>`).

.. _EMComposition_Operation:

**Operation**

*Retrieval.*  Entries are retrieved from `memory <ContentAddressableMemory.memory>` based on their
distance from the cues usde for retrieval, computed as the dot product of the cue and each entry in `memory
<EMComposition.memory>`. If memories have more than one field, then the dot products for each field are computed in
one of two ways:

  * **as a single vector** if the `field_weights <EMComposition.field_weights>` parameter is a single
    value or all of its non-zero values (i.e., the weights for the keys) are all the same; in that case, all of the
    key fields in the cue are concatenated into a single vector, and the dot product is computed for that vector and
    the similarly concatenated key fields of each entry in `memory <EMComposition.memory>`. ii) if `field_weights
    <EMComposition.field_weights>`;

  * **field-by-field** if there is more than one non-zero value in `field_weights <EMComposition.field_weights>` and
    they are not all identical;  in this case, the dot product is computed between each key field in the cue and the
    the corresponding ones of each entry in `memory <ContentAddressableMemory.memory>`, and those dot products are
    then softmaxed, and that softamxed vector is multplied by the corresponding values of `field_weights
    <EMComposition.field_weights>`, which are then summed to produce the distance for each entry in `memory
    <EMComposition.memory>`.

  The distances computed between the cue and each entry in `memory <EMComposition.memory>` are then used to compute
  a weighted average of all entries in `memory <EMComposition.memory>` that is then returned as the `result
  <Composition.result>` of the EMComposition's `execution <Composition_Execution>`.
  COMMENT:
  TBD:
  The distances used for the last retrieval is stored in XXXX and the distances of each of their corresponding fields
  (weighted by `distance_field_weights <ContentAddressableMemory.distance_field_weights>`), are returned in XXX,
  respectively.
  COMMENT

*Storage.*  The `inputs <Composition_Input_External_InputPorts>` to the EMComposition's are stored in `memory
<EMComposition.memory>` after each execution, with a probability determined by `storage_prob
<EMComposition.storage_prob>`.  If `memory_decay <EMComposition.memory_decay>` is specified, then the `memory
<EMComposition.memory>` is decayed by that amount after each execution.

.. _EMComposition_Creation:

Creation
--------

ADD:
- DEFAULT CONFIGURATION IS AS DICTIONARY WITH LAST FIELD AS THE VALUE
- memory_template = TUPLE -> initializes with zeros; restricted to regular array
- memory_template = list or 2d array: can be used to store create a ragged array and/or or store initial value

An EMComposition is created by calling its constructor, that takes the following arguments:

  .. _EMComposition_Fields:

* *Field Specification*



  * **memory_template** : This specifies the shape of the items to be stored in the EMComposition's memory, and can be
    specified in any of the following ways:

      * **2-item tuple** -- this is interpreted as an np.array shape specification, in which the first item specifies
        the number of fields in each memory entry, and the second item specifies the length of each field.  For example,
        the following specification:

          (2,3)

        specifies that each item to be stored in the EMComposition's memory has two fields, each of length 3.  This
        specification is equivalent to the following:

          [[0,0,0],[0,0,0]]

      * **list** or **np.array** -- this is interpreted as a template for the items to be stored in the EMComposition's
        memory, in which each item is a list of the same length.  For example, the following specification:

          [[0,0,0],[0,0,0]]

      specifies that each item to be stored in the EMComposition's memory has two fields, each of length 3.  This
      specification is equivalent to the following:

          (2,3)

      as a 2-item tuple, list, or 2d array.  If it is a tuple

       and a list or 2d array is interpreted as a template that allows fields of
          different lengths to be specified (i.e., items to be encoded can be ragged arrays).


  * **field_weights** : this is used both to
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

  * **field_names** : this is used both to

  .. _EMComposition_Capacity:

  * **memory_capacity**: specifies the maximum number of items that can be stored in the EMComposition's memory.

  * **decay_memories**: specifies whether the EMComposition's memory decays over time.

  * **decay_rate**: specifies the rate at which items in the EMComposition's memory decay.

  .. _EMComposition_Learning:

  * **learn_weights** : specifies whether the weights specified in **field_weights** are learned during training.

  * **learning_rate** : specifies the rate at which **field_weights** are learned if learn_weights is True.


.. _EMComposition_Structure:

Structure
---------

* **memory** -- stores the items encoded by the EMComposition.



.. _EMComposition_Execution:

Execution
---------
The arguments of the `run <EMComposition.run>` and `execute <EMComposition.execute>` methods are the same as those
of a `Composition`.  The only difference in execution is that the values of the key_input_value and value_input_value
nodes are assigned in place of the weakest entry in the EMComposition's memory.

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
from psyneulink.core.components.functions.nonstateful.transferfunctions import LinearMatrix
from psyneulink.core.components.functions.function import \
    DEFAULT_SEED, FunctionError, _random_state_getter, _seed_setter, EPSILON, _noise_setter
from psyneulink.core.compositions.composition import Composition, CompositionError, NodeRole
from psyneulink.library.compositions.autodiffcomposition import AutodiffComposition
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.mechanisms.modulatory.control.gating.gatingmechanism import GatingMechanism
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.keywords import \
    AUTO, EM_COMPOSITION, FUNCTION, MULTIPLICATIVE_PARAM, NAME, PROJECTIONS, RESULT, SIZE, VALUE, ZEROS_MATRIX
from psyneulink.core.globals.utilities import all_within_range

__all__ = [
    'EMComposition'
]

STORAGE_PROB = 'storage_prob'

def _memory_getter(owning_component=None, context=None): # FIX: MAKE THIS A PARAMETER
    """Return memory as a list of the memories stored in the memory nodes.
    """
    memory = None
    for key_node in owning_component.key_input_nodes:
        memory_field = key_node.efferents[0].parameters.matrix.get(context).transpose()
        if memory is None:
            memory = memory_field
        else:
            memory = np.concatenate((memory, memory_field),axis=1)
    for retrieval_node in owning_component.retrieval_nodes:
        memory = np.concatenate((memory, retrieval_node.path_afferents[0].parameters.matrix.get(context)),axis=1)
    return memory


class EMCompositionError(CompositionError):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class EMComposition(AutodiffComposition):
    """
    EMComposition(                  \
        memory_template=[[0],[0]],  \
        field_weights=None,         \
        field_names=None,           \
        learn_weights=True,         \
        learning_rate=True,         \
        memory_capacity=1000,       \
        decay_memories=True,        \
        decay_rate=.001,            \
        storage_prob=1.0,           \
        name="EM_composition"       \
        )

    Subclass of `AutodiffComposition` that implements the functions of an `EpisodicMemoryMechanism` in a
    differentiable form and in which it `field_weights <EMComposition.field_weights>` parameter can be learned.

    Takes only the following arguments, all of which are optional

    Arguments
    ---------

    memory_template : 2-item tuple, list, or 2d array : default [[0],[0]]
        specifies the shape of an items to be stored in the EMComposition's memory;
        see `EMComposition_Memory_Template` for details.

    field_weights : tuple : default (1,0)
        specifies the relative weight assigned to each key when matching an item in memory'
        see `EMComposition_Fields` for details.

    field_names : list : default None
        specifies the optional names assigned to each field in the memory_template;'
        see `EMComposition_Fields` for details.

    learn_weights : bool : default False
        specifies whether `field_weights <EMCompostion.field_weights>` are learnable during training.

    learning_rate : float : default .01
        specifies rate at which`field_weights <EMCompostion.field_weights>` are learned if **learn_weights** is True.

    memory_capacity : int : default 1000
        specifies the number of items that can be stored in the EMComposition's memory;
        see `EMComposition_Memory_Capacity` for details.

    decay_memories : bool : default True
        specifies whether memories decay with each execution of the EMComposition;
        see `EMComposition_Memory_Capacity` for details.

    decay_rate : float : default 1 / `memory_capacity <EMComposition.memory_capacity>`
        specifies the rate at which items in the EMComposition's memory decay;
        see `EMComposition_Memory_Capacity` for details.


    Attributes
    ----------

    key_input_nodes : list[TransferMechanism]

    value_input_nodes : list[TransferMechanism]

    match_nodes : list[TransferMechanism]

    retrieval_nodes : list[TransferMechanism]

    retrieval_weighting_node : TransferMechanism

    retrieval_gating_nodes : list[GatingMechanism]

    memory : 2d np.array

    field_weights : list[float]

    field_names : list[str]

    learn_weights : bool

    learning_rate : float

    memory_capacity : int

    decay_memories : bool

    decay_rate : float

    storage_prob : float
    """

    componentCategory = EM_COMPOSITION
    class Parameters(AutodiffComposition.Parameters):
        """
            Attributes
            ----------

                learn_weights
                    see `learn_weights <EMComposition.learn_weights>`

                    :default value: True
                    :type: ``bool``

                learning_rate
                    see `learning_results <EMComposition.learning_rate>`

                    :default value: []
                    :type: ``list``

                memory
                    see `memory <EMComposition.memory>`

                    :default value: None
                    :type: ``numpy.ndarray``

                memory_capacity
                    see `memory_capacity <EMComposition.memory_capacity>`

                    :default value: 1000
                    :type: ``int``

                decay_memories
                    see `decay_memories <EMComposition.decay_memories>`

                    :default value: False
                    :type: ``bool``

                decay_rate
                    see `decay_rate <EMComposition.decay_rate>`

                    :default value: 0.001
                    :type: ``float``

                field_names
                    see `field_names <EMComposition.field_names>`

                    :default value: None
                    :type: ``list``

                field_weights
                    see `field_weights <EMComposition.field_weights>`

                    :default value: None
                    :type: ``numpy.ndarray``

                normalize_memories
                    see `normalize_memories <EMComposition.normalize_memories>`

                    :default value: True
                    :type: ``bool``

                random_state
                    see `random_state <NormalDist.random_state>`

                    :default value: None
                    :type: ``numpy.random.RandomState``

                softmax_gain
                    :default value: 1.0
                    :type: ``float``

                storage_prob
                    see `storage_prob <EMComposition.storage_prob>`

                    :default value: 1.0
                    :type: ``float``

        """
        memory = Parameter(None, loggable=True, getter=_memory_getter, structural=True)
        # memory_template = Parameter([[0],[0]], structural=True)
        memory_capacity = Parameter(1000, structural=True)
        field_weights = Parameter(None, structural=True)
        field_names = Parameter(None, structural=True)
        decay_memories = Parameter(True, loggable=True, modulable=True, fallback_default=True)
        decay_rate = Parameter(None, loggable=True, modulable=True, fallback_default=True,
                               dependencies='decay_memories')
        normalize_memories = Parameter(True, loggable=False, fallback_default=True)
        learning_weights = Parameter(True, fallback_default=True)
        learning_rate = Parameter(.001, fallback_default=True)
        storage_prob = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        softmax_gain = Parameter(1.0, modulable=True)
        random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies='seed')
        seed = Parameter(DEFAULT_SEED, modulable=True, fallback_default=True, setter=_seed_setter)

        # def _validate_memory_template(self, memory_template):
        #     if not isinstance(memory_template, (list, tuple, np.ndarray)):
        #         return f"must be a list, tuple, or array."
        #
        # def _parse_memory_template(self, memory_template):
        #     if isinstance(memory_template, tuple):
        #         return np.zeros(memory_template)
        #     else:
        #         return np.array(memory_template)

        def _validate_field_weights(self, field_weights):
            if not np.atleast_1d(field_weights).ndim == 1:
                return f"must be a scalar, list of scalars, or 1d array."

        def _validate_field_names(self, field_names):
            if field_names and not all(isinstance(item, str) for item in field_names):
                return f"must be a list of strings."

        def _validate_decay_rate(self, decay_rate):
            if decay_rate is not None:
                decay_rate = float(decay_rate)
                if self.decay_memories.get() and decay_rate == 0.0:
                    return f"is 0.0 and 'decay_memories' arg is True; set it to a positive value " \
                           f"or to None to use the default of 1/memory_capacity."
                if not all_within_range(decay_rate, 0, 1):
                    return f"must be a float in the interval [0,1]."

        def _validate_softmax_gain(self, softmax_gain):
            if softmax_gain is not AUTO and not isinstance(softmax_gain, (float, int)):
                return f"must be a scalar or the keyword 'AUTO'."

        def _validate_storage_prob(self, storage_prob):
            storage_prob = float(storage_prob)
            if not all_within_range(storage_prob, 0, 1):
                return f"must be a float in the interval [0,1]."

    @check_user_specified
    def __init__(self,
                 memory_template:Union[tuple, list, np.ndarray]=[[0],[0]],
                 field_names:Optional[list]=None,
                 field_weights:tuple=None,
                 learn_weights:bool=True,
                 learning_rate:float=None,
                 memory_capacity:int=1000,
                 decay_memories:bool=True,
                 decay_rate:float=None,
                 normalize_memories=True,
                 softmax_temp:float=None,
                 storage_prob:float=None,
                 name="EM_composition"):

        if isinstance(memory_template, tuple):
            memory_template = np.zeros(memory_template)
        else:
            memory_template = np.array(memory_template)

        # Deal with default field_weights
        if field_weights is None:
            if len(memory_template) == 1:
                field_weights = [1]
            else:
                # Default is to all fields as keys except the last one, which is the value
                num_fields = len(memory_template)
                field_weights = [1] * num_fields
                field_weights[-1] = 0

        self._validate_memory_structure(memory_template, field_weights, field_names, name)

        # Memory structure (field) attributes (not Parameters)
        self.memory_template = memory_template.copy() # copy to avoid gotcha since default is a list (mutable)
        self.memory_capacity = memory_capacity # FIX: MAKE THIS A READ-ONLY PARAMETER
        self.memory_dim = np.array(memory_template).size
        self.num_fields = len(memory_template)
        self.field_weights = field_weights
        self.field_names = field_names.copy() if field_names is not None else None
        self.learn_weights = learn_weights

        # Memory management parameters
        if self.parameters.decay_memories.get() and self.parameters.decay_rate.get() is None:
            self.parameters.decay_rate.set(decay_rate or 1 / self.memory_capacity)
        if softmax_temp in (None, AUTO):
            self.parameters.softmax_gain.set(AUTO)
        else:
            self.parameters.softmax_gain.set(1/softmax_temp)

        # Memory field attributes
        keys_weights = [i for i in self.field_weights if i != 0]
        self.num_keys = len(keys_weights)
        self.concatenate_keys = len(self.field_weights) == 1 or  np.all(keys_weights == keys_weights[0])
        self.num_values = self.num_fields - self.num_keys
        if self.field_names:
            self.key_names = self.field_names[:self.num_keys]
            self.value_names = self.field_names[self.num_keys:]
        else:
            self.key_names = [f'KEY {i}' for i in range(self.num_keys)] if self.num_keys > 1 else ['KEY']
            self.value_names = [f'VALUE {i}' for i in range(self.num_values)] if self.num_values > 1 else ['VALUE']

        pathway = self._construct_pathway()

        super().__init__(pathway,
                         name=name)

        # Suppress warnings for no efferent Projections
        for node in self.value_input_nodes:
            node.output_ports['RESULT'].parameters.require_projection_in_composition.set(False, override=True)
        for node in self.match_nodes:
            node.output_ports['KEY_WEIGHT'].parameters.require_projection_in_composition.set(False, override=True)
        for port in self.retrieval_weighting_node.output_ports:
            if 'RESULT' in port.name:
                port.parameters.require_projection_in_composition.set(False, override=True)

        # Suppress value_input_nodes as OUTPUT nodes of the Composition
        for node in self.value_input_nodes:
            self.exclude_node_roles(node, NodeRole.OUTPUT)

        # Turn off learning for all Projections except inputs to retrieval_gating_nodes
        self._set_learnability_of_projections()
        self._initialize_memory()

        # Set normalization if specified
        if self.normalize_memories:
            for node in self.match_nodes:
                node.input_ports[0].path_afferents[0].function.parameters.normalize.set(True)

    def _validate_memory_structure(self, memory_template, field_weights, field_names, name):
        """Validate the memory_template, field_weights, and field_names arguments
        """

        # memory_template must specify a 2D array:
        if memory_template.ndim != 2:
            raise EMCompositionError(f"The 'memory_template' arg for {name} ({memory_template}) "
                                     f"must specifiy a list of lists or a 2d array.")

        # If field weights has more than one value it must match the first dimension (axis 0) of memory_template:
        if len(field_weights) > 1 and len(field_weights) != len(memory_template):
            raise EMCompositionError(f"The number of items ({len(field_weights)}) "
                                     f"in the 'field_weights' arg for {name} must match the number of items "
                                     f"({len(memory_template)}) in the outer dimension of its 'memory_template' arg.")

        # If field weights has more than one value it must match the first dimension (axis 0) of memory_template:
        if field_names and len(field_names) != len(memory_template):
            raise EMCompositionError(f"The number of items ({len(field_names)}) "
                                     f"in the 'field_names' arg for {name} must match the number of items "
                                     f"({len(memory_template)}) in the outer dimension of its 'memory_template' arg.")

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
                                               # name= 'KEY INPUT' if self.num_keys == 1 else f'KEY {i} INPUT')
                                               name= f'{self.key_names[i]} INPUT')
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
                                               # name= 'VALUE INPUT' if self.num_values == 1 else f'VALUE {i} INPUT')
                                               name= f'{self.value_names[i]} INPUT')
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
                                             # function=SoftMax(gain=self.softmax_temperature(self.field_weights)),
                                             function=SoftMax(gain=self.parameters.softmax_gain.get()),
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
                        #         matrix=ZEROS_MATRIX,
                        #         function=LinearMatrix(normalize=True))
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
        # def _get_retrieval_node_name(node_type, len, i):
        #     return f'{node_type} RETRIEVED' if len == 1 else f'{node_type} {i} RETRIEVED'
        #
        self.retrieved_key_nodes = [TransferMechanism(size=len(self.key_input_nodes[i].variable[0]),
                                                      input_ports=self.retrieval_weighting_node,
                                                      # name=_get_retrieval_node_name("KEY", self.num_keys, i))
                                                      name= f'{self.key_names[i]} RETRIEVED')
                                    for i in range(self.num_keys)]

        self.retrieved_value_nodes = [TransferMechanism(size=len(self.value_input_nodes[i].variable[0]),
                                                        input_ports=self.retrieval_weighting_node,
                                                        # name= _get_retrieval_node_name("VALUE", self.num_values, i))
                                                        name= f'{self.value_names[i]} RETRIEVED')
                                      for i in range(self.num_values)]

        return self.retrieved_key_nodes + self.retrieved_value_nodes

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
            
    def execute(self, inputs, context, **kwargs):
        """Set input to weights of Projection to match_node.
        """
        results = super().execute(inputs, **kwargs)
        self._store_memory(inputs, context)
        return results

    def _store_memory(self, inputs, context):
        """Store inputs in memory as weights of Projections to match_nodes (keys) and retrieval_nodes (values).
        """
        storage_prob = np.array(self._get_current_parameter_value(STORAGE_PROB, context)).astype(float)
        random_state = self._get_current_parameter_value('random_state', context)

        if storage_prob == 0.0 or (storage_prob > 0.0 and storage_prob < random_state.uniform()):

            return

        for input_node, memory in inputs.items():
            # Memory = key_input or value_input;
            # memories = weights of Projection from input_node to match_node or retrieval_node for given field
            if input_node in self.key_input_nodes:
                # For key_input:
                #   assign as weights for first empty row of Projection.matrix from key_input_node to match_node
                memories = input_node.efferents[0].parameters.matrix.get(context)
                if self.decay_memories:
                    memories *= self.decay_rate
                # Get least used slot (i.e., weakest memory = row of matrix with lowest weights)
                # idx_of_min = np.argmin(memories.sum(axis=0))
                idx_of_min = np.argmin(np.linalg.norm(memories, axis=0))
                memories[:,idx_of_min] = np.array(memory)
                input_node.efferents[0].parameters.matrix.set(memories, context)

                # Set gain of match_node adaptively
                if self.parameters.softmax_gain.get(context) is AUTO:
                    match_node = self.match_nodes[self.key_input_nodes.index(input_node)]
                    assert match_node == input_node.efferents[0].receiver.owner,\
                        f'PROGRAM ERROR: match_node ({match_node}) is not the owner ' \
                        f'of the Projection from key_input_node ({input_node})'
                    gain = self._get_softmax_gain(np.linalg.norm(memories, axis=0))
                    # gain2 = self._get_softmax_gain(match_node.value[0])
                    match_node.function.parameters.gain.set(gain, context)
                    assert True

            if input_node in self.value_input_nodes:
                # For value_input;
                #   assign as weights for 1st empty row of Projection.matrix from retrieval_weighting_node to retrieval_node
                idx = self.value_input_nodes.index(input_node)
                memories = self.retrieval_nodes[idx].path_afferents[0].parameters.matrix.get(context)
                if self.decay_memories:
                    memories *= self.decay_rate
                # Get least used slot (i.e., weakest memory = row of matrix with lowest weights)
                idx_of_min = np.argmin(memories.sum(axis=1))
                memories[idx_of_min] = np.array(memory)
                self.retrieval_nodes[idx].path_afferents[0].parameters.matrix.set(memories, context)

    def _get_softmax_gain(self, values, epsilon=1e-8):
        """Compute the softmax gain based on length of vector and number of (near) zero values.
        """
        n = len(values)
        num_zero = np.count_nonzero(values < epsilon)
        num_non_zero = n - num_zero
        gain = 1 + np.exp(1/num_non_zero) * (num_zero/n)
        return gain
