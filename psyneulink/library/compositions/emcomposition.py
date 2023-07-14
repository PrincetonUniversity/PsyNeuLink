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
# - memory_field as np.array should store speciied value in memory.
# - FIX: BROADCAST MEMORY_TEMPLATE IF IT IS AN NP.ARRAY
# - FIX: ALWAYS CHECK FOR ZEROS IN KEYS OR ALL MEMORIES FOR NORMALIZATION IN LINEAR MATRIX FUNCTION
# - FIX: CONFIDENCE COMPUTATION (USING SIGMOID ON DOT PRODUCTS) AND REPORT THAT (EVEN ON FIRST CALL)
# - FIX: WRITE TESTS
# - FIX: ALLOW memory TO BE INITIALIZED USING A MATRIX OR FILL VALUE
# - FIX: ALLOW SOFTMAX SPEC TO BE A DICT WITH PARAMETERS FOR _get_softmax_gain() FUNCTION
# - FIX: CONCATENATE ANY FIELDS THAT ARE THE SAME WEIGHT (FOR EFFICIENCY)
# - FIX: WHY IS Concatenate NOT WORKING AS FUNCTION OF AN INPUTPORT (WASN'T THAT USED IN CONTEXT OF BUFFER?)
# - FIX: COMPILE
#      - LinearMatrix to add normalization
#      _store() method to assign weights to memory
# - FIX: AUGMENT LINEARMATRIX NORMALIZATION SO THAT ANYTIME A ROW'S NORM IS 0 REPLACE IT WITH 1
#  1s)
# - FIX: LEARNING:
#        - ADD LEARNING MECHANISMS TO STORE MEMORY AND ADJUST WEIGHTS
#        - DEAL WITH ERROR SIGNALS to retrieval_weighting_node OR AS PASS-THROUGH
# - WRITE TESTS FOR INPUT_PORT and MATRIX SPECS CORRECT IN LATEST BRANCHEs
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
# - MAKE "_store_memory" METHOD USE LEARNING INSTEAD OF ASSIGNMENT
#   - make LearningMechanism that, instead of error, simply adds relevant input to weights (with all others = 0)
#   - (relationship to Steven's Hebbian / DPP model?):

# - ADD ADDITIONAL PARAMETERS FROM CONTENTADDRESSABLEMEMORY FUNCTION
# - ADAPTIVE TEMPERATURE: KAMESH FOR FORMULA
# - ADD MEMORY_DECAY TO ContentAddressableMemory FUNCTION (and compiled version by Samyak)
# - MAKE memory_template A CONSTRUCTOR ARGUMENT FOR default_variable


"""

Contents
--------

  * `EMComposition_Overview`
     - `Organization <EMComposition_Organization>`
     - `Operation <EMComposition_Operation>`
  * `EMComposition_Creation`
     - `Fields <EMComposition_Fields>`
     - `Capacity <EMComposition_Memory_Capacity>`
     - `Storage and Retrieval <EMComposition_Retrieval_Storage>`
     - `Learning <EMComposition_Learning>`
  * `EMComposition_Structure`
     - `Input <EMComposition_Input>`
     - `Memory <EMComposition_Memory>`
     - `Output <EMComposition_Output>`
  * `EMComposition_Execution`
     - `Processing <EMComposition_Processing>`
     - `Learning <EMComposition_Learning>`
  * `EMComposition_Examples`
  * `EMComposition_Class_Reference`


.. _EMComposition_Overview:

Overview
--------

The EMComposition implements a configurable, content-addressable form of episodic, or eternal memory, that emulates
an `EpisodicMemoryMechanism` -- reproducing all of the functionality of its `ContentAddressableMemory` `Function` --
in the form of an `AutodiffComposition` that is capable of learning how to differentially weight different cues used
for retrieval,, and that adds the capability for `memory_decay <EMComposition.memory_decay>`. Its `memory
<EMComposition.memory>` is configured using the **memory_template** argument of its constructor, which defines how
each entry in `memory <EMComposition.memory>` is structured (the number of fields in each entry and the length of
each field), and its **field_weights** argument that defines which fields are used as cues for retrieval -- "keys" --
and whether and how they are differentially weighted in the match process used for retrieval, and which are treated
as "values" that are retrieved but not used for the match process.  The inputs corresponding to each key and each
value are represented as `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` of the EMComposition (listed in its
`key_input_nodes <EMComposition.key_input_nodes>` and `value_input_nodes <EMComposition.value_input_nodes>`
attributes, respectively), and the retrieved values are represented as `OUTPUT <NodeRole.OUTPUT>` `Nodes
<Composition_Nodes>` of the EMComposition.  The `memory <EMComposition.memory>` can be accessed using its `memory
<EMComposition.memory>` attribute.

.. _EMComposition_Organization:

**Organization**

*Entries and Fields*. Each entry in memory can have an arbitrary number of fields, and each field can have an arbitrary
length.  However, all entries must have the same number of fields, and the corresponding fields must all have the same
length across entries. Fields can be weighted to determine the influence they have on retrieval, using the
`field_weights <ContentAddressableMemory.memory>` parameter (see `retrieval <EMComposition_Retrieval>` below). The
number and shape of the fields in each entry is specified in the **memory_template** argument of the EMComposition's
constructor (see `memory_template <EMComposition_Fields>`). Which fields treated as keys (i.e., used as cues for
retrieval) and which are treated as values (i.e., retrieved but not used for matching retrieval) is specified in the
**field_weights** argument of the EMComposition's constructor (see `field_weights <EMComposition_Field_Weights>`).

.. _EMComposition_Operation:

**Operation**

*Retrieval.*  The values retrieved from `memory <ContentAddressableMemory.memory>` (one for each field) are based on
the relative similarity of the keys to the entries in memory, computed as the dot product of each key and the
values in the corresponding field for each entry in memory.  These dot products are then softmaxed, and those
softmax distributions are weighted by the corresponding `field_weights <EMComposition.field_weights>` for each field
and then combined, to produce a single softmax distribution over the entries in memory, that is used to generate a
weighted average as the retrieved value across all fields, and returned as the `result <Composition.result>` of the
EMComposition's `execution <Composition_Execution>`.
  COMMENT:
  TBD:
  The distances used for the last retrieval is stored in XXXX and the distances of each of their corresponding fields
  (weighted by `distance_field_weights <ContentAddressableMemory.distance_field_weights>`), are returned in XXX,
  respectively.
  COMMENT

*Storage.*  The `inputs <Composition_Input_External_InputPorts>` to the EMComposition's fields are stored in `memory
<EMComposition.memory>` after each execution, with a probability determined by `storage_prob
<EMComposition.storage_prob>`.  If `memory_decay <EMComposition.memory_decay>` is specified, then the `memory
<EMComposition.memory>` is decayed by that amount after each execution.  If `memory_capacity
<EMComposition.memory_capacity>` has been reached, then each new memory replaces the weakest entry (i.e., the one
with the smallest norm across all of its fields) in `memory <EMComposition.memory>`.

.. _EMComposition_Creation:

Creation
--------

An EMComposition is created by calling its constructor, that takes the following arguments:

  .. _EMComposition_Fields:

*Field Specification*

.. _EMComposition_Memory_Template:

* **memory_template**: This specifies the shape of the entries to be stored in the EMComposition's `memory
  <EMComposition.memory>`.  The default is to initialize `memory <EMComposition.memory>` with all zeros, but
  it can be initialized with pre-specified entries using the list or array formats described below.  The
  **memory_template* can be specified in one of three ways (see `Examples <EMComposition_Examples> `for
  representative use cases):

  * tuple*: interpreted as an np.array shape specification, in which the 1st item specifies the number of fields in
    each memory entry and the 2nd item specifies the length of each field.  The matrix is filled either with zeros or
    with the value specified in **memory_fill** (see below).

  * *2d list or array*:  interpreted as a template for memory entries.  This can be used to specify fields of
    different lengths (i.e., entries that are ragged arrays), with each item in the list (axis 0 of the array) used to
    specify the length of the corresponding field.  The template is broadcast over the third dimension to generate the
    full matrix used to initialize `memory <EMComposition.memory>`.  If the template uses any non-zero values, then the
    array is replicated for all entries in `memory <EMComposition.memory>`.  If the template has all zeros, then the
    the entire `memory <EMComposition.memory>` is filled with either zeros or the value specified in **memory_fill**.
    COMMENT:
    template is used as the first entry in `memory <EMComposition.memory>`, and the remaining entries are filled with
    either zeros or the value specified in **memory_fill** (see below).  If the template has all zeros, then the
    the entire `memory <EMComposition.memory>` is filled with either zeros or the value specified in **memory_fill**.
    COMMENT

    .. note::
       Use a 3d array (as described below) to specify a single entry, with the remainder of entries filled with zeros
       or the value specified in **memory_fill**.

  * *3d list or array*:  used to initialize `memory <EMComposition.memory>` directly. If the outer dimension of
    the list or array (axis 0) has fewer than **memory_capacity** items, then it is filled with the remaining entries,
    using either zeros or the value specified in **memory_fill** (see below).  If all of thespecified entries are
    zeros and **memory_fill** is specified, then the matrix is filled with the value specified in **memory_fill**.

.. _EMComposition_Memory_Fill:

* **memory_fill**: specifies the value used to fill the `memory <EMComposition.memory>`, based on the shape specified
  in the **memory_template** (see above).  The value can be a scalar, or a tuple to specify an interval over which
  to draw random values to fill `memory <EMComposition.memory>` --- both should be scalars, with the first specifying
  the lower bound and the second the upper bound.

.. _EMComposition_Field_Weights:

* **field_weights**: specifies which fields are used as keys, and how they are weighted during retrieval. The
  number of values specified must match the number of fields specified in **memory_template** (i.e., the size of
  of its first dimension (axis 0)).  All non-zero entries must be positive, and designate *keys* -- fields
  that are used to match items in memory for retrieval (see `Match memories by field <EM_CompositionProcessing>`).
  Entries of 0 designate *values* -- fields that are ignored during the matching process, but the values of which
  are retrieved and assigned as the `value <Mechanism_Base.value>` of the corresponding `retrieval_node
  <EMComposition.retrieval_nodes>`. This distinction between keys and value implements a standard "dictionary; however,
  if all entries are non-zero, then all fields are treated as keys, implemented a full form of content-addressable
  memory.  If **learn_weights** is True, the field_weights can be modified during training; otherwise they remain
  fixed. The following options can be used to specify **field_weights**:

    * *None* (the default): all fields except the last are treated as keys, and are weighted equally for retrieval,
      while the last field is treated as a value field;

    * *single entry*: its value is ignored, and all fields are treated as keys (i.e., used for
      retrieval) and are `concatenated <EMComposition_Concatenate_Keys>` and equally weighted for retrieval;

    * *multiple non-zero entries*: If all entries are identical, the value is ignored and the corresponding keys are
      `concatenated <EMComposition_Concatenate_Keys>` and weighted equally for retrieval; if the non-zero entries are
      non-identical, they are used to weight the corresponding fields during retrieval (see `Weight fields
      <EMComposition_Processing>`).  In either case, the remaining fields (with zero weights) are treated
      as value fields.

.. _EMComposition_Field_Names:

* **field_names**: specifies names that can be assigned to the fields.  The number of names specified must
  match the number of fields specified in the memory_template.  If specified, the names are used to label the
  nodes of the EMComposition.  If not specified, the fields are labeled generically as "Key 0", "Key 1", etc..

.. _EMComposition_Concatenate_Keys:

* **concatenate_keys**:  specifies whether keys are concatenated before a match is made to items in memory.
  If True, all keys are concatenated (i.e., fields for which non-zero weights are specified in field_weights);
  this occurs even if the field_weights are not all the same value (in which case a warning is issued).
  If False, keys are concatenated only if all non-zero field_weights are the same value (see `field_weights
  <EMComposition_Field_Weights>` above).

.. _EMComposition_Memory_Capacity:
  
*Memory Capacity*

* **memory_capacity**: specifies the maximum number of items that can be stored in the EMComposition's memory; when
  `memory_capacity <EMComposition.memory_capacity>` is reached, each new entry overwrites the weakest entry (i.e., the
  one with the smallest norm across all of its fields) in `memory <EMComposition.memory>`.

* **memory_decay**: specifies whether the EMComposition's memory decays over time.

* **memory_decay_rate**: specifies the rate at which items in the EMComposition's memory decay;  the default rate is
  1 / `memory_capacity <EMComposition.memory_capacity>`, such that the oldest memories are the most likely to be
  replaced when `memory_capacity <EMComposition.memory_capacity>` is reached.

  .. _EMComposition_Retrieval_Storage:

*Retrieval and Storage*

* **storage_prob** : specifies the probability that the inputs to the EMCompositoin will be stored as an item in
  `memory <EMComposition.memory>` on each execution.

* **normalize_memories** : specifies whether keys and memories are normalized before computing their dot products.

.. _EMComposition_Softmax_Gain:

* **softmax_gain** : specifies the gain (inverse temperature) used for softmax normalizing the dot products of keys
  and memories (see `EMComposition_Execution` below).  If a value is specified, that is used.  If the keyword *CONTROL*
  is (or the value is None), then the `softmax_gain <EMComposition.softmax_gain>` function is used to adaptively set
  the gain based on the entropy of the dot products, preserving the distribution over non-(or near) zero entries
  irrespective of how many (near) zero entries there are.

* **learn_weights** : specifies whether the weights specified in **field_weights** are modifiable during training.

* **learning_rate** : specifies the rate at which **field_weights** are learned if **learn_weights** is True.


.. _EMComposition_Structure:

Structure
---------

.. _EMComposition_Input:

*Input*
~~~~~~~

The inputs corresponding to each key and value field are represented as `INPUT <NodeRole.INPUT>` `Nodes
<Composition_Nodes>` of the EMComposition, listed in its `key_input_nodes <EMComposition.key_input_nodes>`
and `value_input_nodes <EMComposition.value_input_nodes>` attributes, respectively,

.. _EMComposition_Memory:

*Memory*
~~~~~~~

The `memory <EMComposition.memory>` attribute contains a record of the entries in the EMComposition's memory. This is
in the form of a 2d array, in which rows (axis 0) are entries and columns (axis 1) are fields.  The number of fields
is determined by the `memory_template <EMComposition_Memory_Template>` argument of the EMComposition's constructor,
and the number of entries is determined by the `memory_capacity <EMComposition_Memory_Capacity>` argument.

  .. _EMComposition_Memory_Storage:
  .. technical_note::
     The memories are actually stored in the `matrix <MappingProjection.matrix>` parameters of the `MappingProjections`
     from the `retrieval_weighting_node <EMComposition.retrieval_weighting_node>` to each of the `retrieval_nodes
     <EMComposition.retrieval_nodes>`.  Memories associated with each key are also stored in the `matrix
     <MappingProjection.matrix>` parameters of the `MappingProjections` from the `key_input_nodes
     <EMComposition.key_input_nodes>` to each of the corresponding `match_nodes <EMComposition.match_nodes>`.
     This is done so that the match of each key to the memories for the corresponding field can be computed simply
     by passing the input for each key through the Projection to the corresponding match_node and, similarly,
     retrieivals can be computed by passiing the softmax disintributions and weighting for each field computed
     in the `retrieval_weighting_node <EMComposition.retrieval_weighting_node>` through its Projection to each
     `retrieval_node <EMComposition.retrieval_nodes>` to get the retreieved value for each field.

.. _EMComposition_Output:

*Output*
~~~~~~~

The outputs corresponding to retrieved value for each field are represented as `OUTPUT <NodeRole.INPUT>` `Nodes
<Composition_Nodes>` of the EMComposition, listed in its `retrieval_nodes <EMComposition.retrieval_nodes>` attribute.

.. _EMComposition_Execution:

Execution
---------

The arguments of the `run <Composition.run>` , `learn <Composition.learn>` and `Composition.execute`
methods are the same as those of a `Composition`, and they can be passed any of the arguments valid for
an `AutodiffComposition`.  The details of how the EMComposition executes are described below.

.. _EMComposition_Processing:

*Processing*
~~~~~~~~~~~~

When the EMComposition is executed, the following sequence of operations occur:

* **Concatenation**. If the `field_weights <EMComposition.field_weights>` are the same for all `keys
  <EMComposition_Field_Weights>` or the `concatenate_keys <EMComposition_Concatenate_Keys>` attribute is True,
  then the inputs provided to the `key_input_nodes <EMComposition.key_input_nodes>` are concatenated into a single
  vector in the `concatenation_node <EMComposition.concatenation_node>`, that is provided to a corresponding
  `match_node <EMComposition.match_nodes>`.

# FIX: ADD MENTION OF NORMALIZATION HERE
* **Match memories by field**. The values of each `key_input_node <EMComposition.key_input_nodes>` (or the
  `concatenation_node <EMComposition.concatenation_node>` if `concatenate_keys <EMComposition_Concatenate_Keys>`
  attribute is True) are passed through the corresponding `match_node <EMComposition.match_nodes>`, which computes
  the dot product of the input with each memory for the corresponding field, resulting in a vector of dot products
  for each memory in the corresponding field.

* **Softmax normalize matches over fields**. The dot products of memories for each field are passed to the
  corresponding `softmax_node <EMComposition.softmax_nodes>`, which applies a softmax function to normalize the
  dot products of memories for each field.  If `softmax_gain <EMComposition.softmax_gain>` is specified, it is
  used as the gain (inverse temperature) for the softmax function; if it is specified as *CONTROL* or None, then
  the `softmax_gain <EMComposition.softmax_gain>` function is used to adaptively set the gain (see `softmax_gain
  <EMComposition_Softmax_Gain>` for details).

* **Weight fields**. The softmax normalized dot products of keys and memories for each field are passed to the
  `retrieval_weighting_node <EMComposition.retrieval_weighting_node>`, which applies the corresponding `field_weight
  <EMComposition.field_weights>` to the softmaxed dot products of memories for each field, and then haddamard sums
  those weighted dot products to produce a single weighting for each memory.

* **Retrieve values by field**. The vector of weights for each memory generated by the `retrieval_weighting_node
  <EMComposition.retrieval_weighting_node>` is passed through the Projections to the each of the `retrieval_nodes
  <EMComposition.retrieval_nodes>` to compute the retrieved value for each field.

* **Decay memories**.  If `memory_decay <EMComposition.memory_decay>` is True, then each of the memories is decayed
  by the amount specified in `memory_decay <EMComposition.memory_decay>`.

    .. technical_note::
       This is done by multiplying the `matrix <MappingProjection.matrix>` parameter of the `MappingProjection` from
       the `retrieval_weighting_node <EMComposition.retrieval_weighting_node>` to each of the `retrieval_nodes
       <EMComposition.retrieval_nodes>`, as well as the `matrix <MappingProjection.matrix>` parameter of the
       `MappingProjection` from each `key_input_node <EMComposition.key_input_nodes>` to the corresponding
       `match_node <EMComposition.match_nodes>` by `memory_decay <EMComposition.memory_decay_rate>`,
        by 1 - `memory_decay <EMComposition.memory_decay_rate>`.

* **Store memories**. After the values have been retrieved, the inputs to for each field (i.e., values in the
  `key_input_nodes <EMComposition.key_input_nodes>` and `value_input_nodes <EMComposition.value_input_nodes>`)
  are added as a new entry in `memory <EMComposition.memory>`, replacing the weakest one if `memory_capacity
  <EMComposition_Memory_Capacity>` has been reached.

    .. technical_note::
       This is done by adding the input vectors to the the corresponding rows of the `matrix <MappingProjection.matrix>`
       of the `MappingProjection` from the `retreival_weighting_node <EMComposition.retrieval_weighting_node>` to each
       of the `retrieval_nodes <EMComposition.retrieval_nodes>`, as well as the `matrix <MappingProjection.matrix>`
       parameter of the `MappingProjection` from each `key_input_node <EMComposition.key_input_nodes>` to the
       corresponding `match_node <EMComposition.match_nodes>`.  If `memory_capacity <EMComposition_Memory_Capacity>`
       has been reached, then the weakest memory (i.e., the one with the lowest norm across all fields) is replaced by
       the new memory.

COMMENT:
FROM CodePilot: (OF HISTORICAL INTEREST?)
inputs to its `key_input_nodes <EMComposition.key_input_nodes>` and
`value_input_nodes <EMComposition.value_input_nodes>` are assigned the values of the corresponding items in the
`input <Composition.input>` argument.  The `retrieval_weighting_node <EMComposition.retrieval_weighting_node>`
computes the dot product of each key with each memory, and then applies a softmax function to each row of the
resulting matrix.  The `retrieval_nodes <EMComposition.retrieval_nodes>` then compute the dot product of the
softmaxed values for each memory with the corresponding value for each memory, and the result is assigned to the
corresponding `output <Composition.output>` item.
COMMENT

.. _EMComposition_Learning:

*Learning*
~~~~~~~~~~

If `learn <Composition.learn>` is called and the `learn_weights <EMComposition.learn_weights>` attribute is True,
then the `field_weights <EMComposition.field_weights>` are modified to minimize the error passed to the EMComposition
retrieval nodes, using the learning_rate specified in the `learning_rate <EMComposition.learning_rate>` attribute. If
`learn_weights <EMComposition.learn_weights>` is False (or `run <Composition.run>` is called, then the
`field_weights <EMComposition.field_weights>` are not modified and the EMComposition is simply executed without any
modification, and the error signal is passed to the nodes that project to its `INPUT <NodeRole.INPUT>` `Nodes
<Composition_Nodes>`.

  .. note::
     Although memory storage is implemented as  a form of learning (though modification of MappingProjection
     `matrix <MappingProjection.matrix>` parameters; see `memory storage <EMComposition_Memory_Storage>`),
     this occurs irrespective of how EMComposition is run (i.e., whether `learn <Composition.learn>` or `run
     <Composition.run>` is called), and is not affected by the `learn_weights <EMComposition.learn_weights>`
     or `learning_rate <EMComposition.learning_rate>` attributes, which pertain only to whether the `field_weights
     <EMComposition.field_weights>` are modified during learning.

.. _EMComposition_Examples:

Examples
--------

COMMENT:
*Memory specification*
~~~~~~~~~~~~~~~~~~~~~~

The following tuple::

      memory_tempalte=(2,3)

  specifies that each item to be stored in the EMComposition's memory has two fields, each of length 3.  This
  is equivalent to the following specification as a list::

      [[0,0,0],[0,0,0]]

  and stores this as the first entry in `memory <EMComposition.memory>`.  In constast, a list can be used to specify
  fields of different length (i.e., a ragged array), and to initialize memory with non-zero values as in the
  following::

      [[0,0,0],[0,0]]

  in which the first field is length 3 but the second is length 2.  The following can be use to initialize `memory
  <EMComposition.memory>` with small random values::

      np.random.rand((2,3)) /100

    memory_template=[[0,0,0],[0,0]
    memory_fill=(RANDOM, 100)

  COMMENT


.. _EMComposition_Class_Reference:

Class Reference
---------------
"""
import numpy as np
import warnings

from psyneulink._typing import Optional, Union

from psyneulink.core.components.functions.nonstateful.transferfunctions import SoftMax, LinearMatrix
from psyneulink.core.components.functions.nonstateful.combinationfunctions import Concatenate
from psyneulink.core.components.functions.function import \
    DEFAULT_SEED, _random_state_getter, _seed_setter
from psyneulink.core.compositions.composition import CompositionError, NodeRole
from psyneulink.library.compositions.autodiffcomposition import AutodiffComposition
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.modulatory.control.gating.gatingmechanism import GatingMechanism
from psyneulink.core.components.mechanisms.modulatory.learning.learningmechanism import LearningMechanism
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.keywords import \
    CONTROL, EM_COMPOSITION, FUNCTION, GAIN, IDENTITY_MATRIX, \
    MULTIPLICATIVE_PARAM, NAME, PROJECTIONS, RANDOM, RESULT, SIZE, VALUE, ZEROS_MATRIX
from psyneulink.core.globals.utilities import all_within_range

__all__ = [
    'EMComposition'
]

STORAGE_PROB = 'storage_prob'

def _memory_getter(owning_component=None, context=None): # FIX: MAKE THIS A PARAMETER
    """Return array of memories in which rows (axis 0) are memories for each field (axis 1).
    These are derived from `matrix <MappingProjection.matrix>` parameter of the `afferent
    <Mechanism_Base.afferents>` MappingProjections to each of the `retrieval_nodes <EMComposition.retrieval_nodes>`.
    """
    memory = None
    for retrieval_node in owning_component.retrieval_nodes:
        memory_field = retrieval_node.path_afferents[0].parameters.matrix.get(context)
        if memory is None:
            memory = memory_field
        else:
            memory = np.concatenate((memory, memory_field),axis=1)
    return memory

def get_softmax_gain(v, scale=1, base=1, entropy_weighting=.1)->float:
    """Compute the softmax gain (inverse temperature) based on the entropy of the distribution of values.
    scale * (base + (entropy_weighting * log(entropy(logistic(v))))))))
    """
    v = np.squeeze(v)
    gain = scale * (base +
                    (entropy_weighting *
                     np.log(
                         -1 * np.sum((1 / (1 + np.exp(-1 * v))) * np.log(1 / (1 + np.exp(-1 * v)))))))
    return gain


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
        concatenate_keys=False,     \
        learn_weights=True,         \
        learning_rate=True,         \
        memory_capacity=1000,       \
        memory_decay=True,          \
        memory_decay_rate=.001,     \
        storage_prob=1.0,           \
        name="EM_composition"       \
        )

    Subclass of `AutodiffComposition` that implements the functions of an `EpisodicMemoryMechanism` in a
    differentiable form and in which it `field_weights <EMComposition.field_weights>` parameter can be learned.

    Takes only the following arguments, all of which are optional

    Arguments
    ---------

    memory_template : tuple, list, 2d or 3d array : default [[0],[0]]
        specifies the shape of an items to be stored in the EMComposition's memory;
        see `memory_template <EMComposition_Memory_Template>` for details.

    memory_fill : scalar or tuple : default 0
        specifies the value used to fill the memory when it is initialized;
        see `memory_fill <EMComposition_Memory_Fill>` for details.

    field_weights : tuple : default (1,0)
        specifies the relative weight assigned to each key when matching an item in memory'
        see `field weights <EMComposition_Field_Weights>` for details.

    field_names : list : default None
        specifies the optional names assigned to each field in the memory_template;
        see `field names <EMComposition_Field_Names>` for details.

    concatenate_keys : bool : default False
        specifies whether to concatenate the keys into a single field before matching them to items in
        the corresponding fields in memory; see `concatenate keys <EMComposition_Concatenate_Keys>` for details.

    normalize_memories : bool : default True
        specifies whether keys and memories are normalized before computing their dot product (similarity);
        see `Match memories by field <EMComposition_Processing>` for additional details.

    softmax_gain : float : default CONTROL
        specifies the temperature used for softmax normalizing the dot products of keys and memories;
        see `Softmax normalize matches over fields <EMComposition_Processing>` for additional details.

    learn_weights : bool : default False
        specifies whether `field_weights <EMComposition.field_weights>` are learnable during training;
        see `Learning <EMComposition_Learning>` for additional details.

    learning_rate : float : default .01
        specifies rate at which `field_weights <EMComposition.field_weights>` are learned if **learn_weights** is True.

    memory_capacity : int : default 1000
        specifies the number of items that can be stored in the EMComposition's memory;
        see `memory_capacity <EMComposition_Memory_Capacity>` for details.

    memory_decay : bool : default True
        specifies whether memories decay with each execution of the EMComposition;
        see `memory_decay <EMComposition_Memory_Capacity>` for details.

    memory_decay_rate : float : default 1 / `memory_capacity <EMComposition.memory_capacity>`
        specifies the rate at which items in the EMComposition's memory decay;
        see `memory_decay_rate <EMComposition_Memory_Capacity>` for details.


    Attributes
    ----------

    memory : 2d np.array
        array of memories in which rows (axis 0) are memories for each field (axis 1).

    .. _EMComposition_Parameters:

    field_weights : list[float]
        determines which fields of the input are treated as "keys" (non-zero values), used to match entries in `memory
        <EM_Composition.memory>` for retrieval, and which are used as "values" (zero values), that are stored and
        retrieved from memory, but not used in the match process (see `Match memories by field`
        <EMComposition_Processing>`;  see `field_weights <EMComposition_Field_Weights>` for additional details
        of specification).

    field_names : list[str]
        determines which names that can be used to label fields in `memory <EM_Composition.memory>`;  see
        `field_names <EMComposition_Field_Names>` for additional details.

    learn_weights : bool
        determines whether `field_weights <EMComposition.field_weights>` are learnable during training; see
        `Learning <EMComposition_Learning>` for additional details.

    learning_rate : float
        determines whether the rate at which `field_weights <EMComposition.field_weights>` are learned if
        `learn_weights` is True;  see <EMComposition_Learning>` for additional details.

    concatenate_keys : bool
        determines whether keys are concatenated into a single field before matching them to items in `memory
        <EM_Composition.memory>`; see `concatenate keys <EMComposition_Concatenate_Keys>` for additional details.

    normalize_memories : bool
        determines whether keys and memories are normalized before computing their dot product (similarity);
        see `Match memories by field <EMComposition_Processing>` for additional details.

    softmax_gain : CONTROL
        determines gain (inverse temperature) used for softmax normalizing the dot products of keys and memories
        by the `softmax` function of the `softmax_nodes <EMComposition.softmax_nodes>`; see `Softmax normalize matches
        over fields <EMComposition_Processing>` for additional details.

    storage_prob : float
        determines the probability that an item will be stored in `memory <EM_Composition.memory>`.

    memory_capacity : int
        determines the number of items that can be stored in `memory <EM_Composition.memory>`; see `memory_capacity
        <EMComposition_Memory_Capacity>` for additional details.

    memory_decay : bool
        determines whether memories decay with each execution of the EMComposition.

    memory_decay_rate : float
        determines the rate at which items in the EMComposition's memory decay.

    .. _EMComposition_Nodes:

    key_input_nodes : list[TransferMechanism]
        `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` that receive keys used to determine the item
        to be retrieved from `memory <EM_Composition.memory>`, and then themselves stored in `memory
        <EM_Composition.memory>` (see `Match memories by field` <EMComposition_Processing>` for additional details).

    value_input_nodes : list[TransferMechanism]
        `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` that receive values to be stored in `memory
        <EM_Composition.memory>`; these are not used in the matching process used for retrieval

    match_nodes : list[TransferMechanism]
        `TransferMechanisms <TransferMechanism>` that receive the dot product of each key and those stored in
        the corresponding field of `memory <EM_Composition.memory>` (see `Match memories by field`
        <EMComposition_Processing>` for additional details).

    softmax_control_nodes : list[ControlMechanism]
        `ControlMechanisms <ControlMechanism>` that adaptively control the `softmax_gain <EMComposition.softmax_gain>`
        for the corresponding `softmax_nodes <EM_Composition.softmax_nodes>`. These are implemented only if
        `softmax_gain <EMComposition.softmax_gain>` is specified as *CONTROL* (see `softmax_gain
        <EMComposition_Softmax_Gain>` for details).

    softmax_nodes : list[TransferMechanism]
        `TransferMechanisms <TransferMechanism>` that compute the softmax over the vectors received
        from the corresponding `match_nodes <EM_Composition.match_nodes>` (see `Softmax normalize matches over fields
        <EMComposition_Processing>` for additional details).

    retrieval_gating_nodes : list[GatingMechanism]
        `GatingMechanisms <GatingMechanism>` that uses the `field weight <EMComposition.field_weights>` for each
        field to modulate the output of the corresponding `retrieval_node <EM_Composition.retrieval_nodes>` before
        it is passed to the `retrieval_weighting_node <EM_Composition.retrieval_weighting_node>`.  These are
        implemented only if differential weights are specified for the different fields in `field_weights
        <EMComposition.field_weights>`.

    retrieval_weighting_node : TransferMechanism
        `TransferMechanism` that receives the softmax normalized dot products of the keys and memories
        from the `softmax_nodes <EM_Composition.softmax_nodes>`, weights these using `field_weights
        <EMComposition.field_weights>`, and haddamard sums those weighted dot products to produce a
        single weighting for each memory.


    retrieval_nodes : list[TransferMechanism]
        `TransferMechanisms <TransferMechanism>` that receive the vector retrieved for each field in `memory
        <EM_Composition.memory>` (see `Retrieve values by field` <EMComposition_Processing>` for additional details).

    """

    componentCategory = EM_COMPOSITION
    class Parameters(AutodiffComposition.Parameters):
        """
            Attributes
            ----------

                concatenate_keys
                    see `concatenate_keys <EMComposition.concatenate_keys>`

                    :default value: False
                    :type: ``bool``

                memory_decay
                    see `memory_decay <EMComposition.memory_decay>`

                    :default value: False
                    :type: ``bool``

                memory_decay_rate
                    see `memory_decay_rate <EMComposition.memory_decay_rate>`

                    :default value: 0.001
                    :type: ``float``

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
                    see `random_state <EMComposition.softmax_gain>`
                    :default value: CONTROL
                    :type: ``float or CONTROL``

                storage_prob
                    see `storage_prob <EMComposition.storage_prob>`

                    :default value: 1.0
                    :type: ``float``

        """
        memory = Parameter(None, loggable=True, getter=_memory_getter, structural=True)
        # memory_template = Parameter([[0],[0]], structural=True, valid_types=(tuple, list, np.ndarray))
        memory_capacity = Parameter(1000, structural=True)
        field_weights = Parameter(None, structural=True)
        field_names = Parameter(None, structural=True)
        concatenate_keys = Parameter(False, structural=True)
        memory_decay = Parameter(True, loggable=True, modulable=True, fallback_default=True)
        memory_decay_rate = Parameter(None, loggable=True, modulable=True, fallback_default=True,
                               dependencies='memory_decay')
        normalize_memories = Parameter(True, loggable=False, fallback_default=True)
        learn_weights = Parameter(True, fallback_default=True)
        learning_rate = Parameter(.001, fallback_default=True)
        storage_prob = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        softmax_gain = Parameter(CONTROL, modulable=True, fallback_default=True)
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
            if field_weights is not None:
                if  not np.atleast_1d(field_weights).ndim == 1:
                    return f"must be a scalar, list of scalars, or 1d array."
                if any([field_weight < 0 for field_weight in field_weights]):
                    return f"must be all be postive values."

        def _validate_field_names(self, field_names):
            if field_names and not all(isinstance(item, str) for item in field_names):
                return f"must be a list of strings."

        def _validate_memory_decay_rate(self, memory_decay_rate):
            if memory_decay_rate is not None:
                memory_decay_rate = float(memory_decay_rate)
                if self.memory_decay.get() and memory_decay_rate == 0.0:
                    return f"is 0.0 and 'memory_decay' arg is True; set it to a positive value " \
                           f"or to None to use the default of 1/memory_capacity."
                if not all_within_range(memory_decay_rate, 0, 1):
                    return f"must be a float in the interval [0,1]."

        def _validate_softmax_gain(self, softmax_gain):
            if softmax_gain != CONTROL and not isinstance(softmax_gain, (float, int)):
                return f"must be a scalar or the keyword 'CONTROL'."

        def _validate_storage_prob(self, storage_prob):
            storage_prob = float(storage_prob)
            if not all_within_range(storage_prob, 0, 1):
                return f"must be a float in the interval [0,1]."

    @check_user_specified
    def __init__(self,
                 memory_template:Union[tuple, list, np.ndarray]=[[0],[0]],
                 memory_fill:Union[int, float, RANDOM]=0,
                 field_names:Optional[list]=None,
                 field_weights:tuple=None,
                 concatenate_keys:bool=False,
                 learn_weights:bool=True,
                 learning_rate:float=None,
                 memory_capacity:int=1000,
                 memory_decay:bool=True,
                 memory_decay_rate:float=None,
                 normalize_memories:bool=True,
                 softmax_gain:Union[float, CONTROL]=CONTROL,
                 storage_prob:float=None,
                 name="EM_composition"):

        self._validate_memory_specs(memory_template, memory_fill, field_weights, field_names, name)
        self._parse_memory_template(memory_template, memory_fill, memory_capacity, field_weights)
        self._parse_fields(field_weights, field_names, concatenate_keys, learn_weights, learning_rate,)

        # Memory management parameters
        if self.parameters.memory_decay.get() and self.parameters.memory_decay_rate.get() is None:
            self.parameters.memory_decay_rate.set(memory_decay_rate or 1 / self.memory_capacity)

        self.softmax_gain = softmax_gain
        # self._construct_memory(memory_template, memory_fill)

        pathway = self._construct_pathway()

        super().__init__(pathway,
                         name=name)

        # Suppress warnings for no efferent Projections
        for node in self.value_input_nodes:
            node.output_ports['RESULT'].parameters.require_projection_in_composition.set(False, override=True)
        for node in self.softmax_nodes:
            node.output_ports['KEY_WEIGHT'].parameters.require_projection_in_composition.set(False, override=True)
        for port in self.retrieval_weighting_node.output_ports:
            if 'RESULT' in port.name:
                port.parameters.require_projection_in_composition.set(False, override=True)

        # Suppress value_input_nodes as OUTPUT nodes of the Composition
        for node in self.value_input_nodes:
            self.exclude_node_roles(node, NodeRole.OUTPUT)

        # Turn off learning for all Projections except inputs to retrieval_gating_nodes
        self._set_learnability_of_projections()

        # # Set normalization if specified
        # if self.normalize_memories:
        #     for node in self.softmax_nodes:
        #         node.input_ports[0].path_afferents[0].function.parameters.normalize.set(True)

    def _validate_memory_specs(self, memory_template, memory_fill, field_weights, field_names, name):
        """Validate the memory_template, field_weights, and field_names arguments
        """

        # memory_template must specify a 2D array:
        if not ((isinstance(memory_template, tuple) and len(memory_template) == 2)
                or np.array(memory_template, dtype=object).ndim in (2,3)):
            raise EMCompositionError(f"The 'memory_template' arg for {name} ({memory_template}) must be a "
                                     f"numpy shape specification or list or array that has 2 or 3 dimensions.")

        if isinstance(memory_template, tuple):
            num_fields = memory_template[0]
        else:
            memory_template = np.array(memory_template)
            num_fields = memory_template.shape[0] if memory_template.ndim == 2 else memory_template.shape[1]


        if not isinstance(memory_template, tuple) and memory_template.ndim == 3:
            # 3d array specified (i.e., template has multiple entries), so ensure all have the same shape
            for entry in memory_template:
                if not (len(entry) == num_fields
                        and np.all([len(entry[i]) == len(memory_template[0][i]) for i in range(num_fields)])):
                    raise EMCompositionError(f"The 'memory_template' arg for {self.name} must specify a list "
                                             f"or 2d array that has the same shape for all entries.")

        if not (isinstance(memory_fill, (int, float)) or
                (isinstance(memory_fill, tuple) and len(memory_fill)==2) and
                all(isinstance(item, (int, float)) for item in memory_fill)):
            raise EMCompositionError(f"The 'memory_fill' arg ({memory_fill}) specified for {name} "
                                     f"must be a float, int or len tuple of ints and/or floats.")

        # If field_weights has more than one value it must match the first dimension (axis 0) of memory_template:
        if len(field_weights) > 1 and len(field_weights) != num_fields:
            raise EMCompositionError(f"The number of items ({len(field_weights)}) in the 'field_weights' arg "
                                     f"for {name} must match the number of items in an entry of memory "
                                     f"({num_fields}).")

        # If field_names has more than one value it must match the first dimension (axis 0) of memory_template:
        if field_names and len(field_names) != num_fields:
            raise EMCompositionError(f"The number of items ({len(field_names)}) "
                                     f"in the 'field_names' arg for {name} must match "
                                     f"the number of fields ({len(field_weights)}).")


    def _parse_memory_template(self, memory_template, memory_fill, memory_capacity, field_weights):
        """Construct memory from memory_template and memory_fill
        Assign self.memory_template and self.entry_template attributes"""

        def _construct_entries(entry_template, num_entries, memory_fill=None):
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
                    entry = [np.full_like(field, memory_fill).tolist() for field in entry_template]
                entries = [np.array(entry, dtype=object)] * num_entries

            return np.array(np.array(entries))

        # If memory_template is a tuple, create and fill full memory matrix
        if isinstance(memory_template, tuple):
            memory = np.full(memory_template, memory_fill)

        # If memory_template is a list or array
        else:

            # Determine whether template is a single entry or full/partial memory specification
            memory_template_dim = np.asarray(memory_template, dtype=object).ndim
            fields_equal_length = all(len(field) == len(memory_template[0]) for field in memory_template)
            single_entry = (((memory_template_dim == 1) and not fields_equal_length) or
                            ((memory_template_dim == 2)  and fields_equal_length))

            if single_entry:
                # If any non-zeros, replicate the entry for full matrix
                if any(list(np.array(memory_template).flat)):
                    memory_fill = None
                # Otherwise, use memory_fill
                else:
                    memory_fill = memory_fill
                memory = _construct_entries(memory_template, memory_capacity, memory_fill)

            # If memory template is a full or partial 3d (matrix) specification
            else:
                # If all entries are zero, create entire memory matrix with memory_fill
                if not any(list(np.array(memory_template).flat)):
                    # Use first entry of zeros as template and replicate for full memory matrix
                    memory = _construct_entries(memory_template[0], memory_capacity, memory_fill)
                # If there are any non-zero values, keep specified entries and create rest using memory_fill
                else:
                    num_entries_needed = memory_capacity - len(memory_template)
                    remaining_entries = _construct_entries(memory_template[0], num_entries_needed, memory_fill)
                    memory = np.concatenate((memory_template, remaining_entries))

        # Get shape of single entry
        self.entry_template = memory[0]
        self.memory_template = memory
        self.memory_capacity = memory_capacity

    def _parse_fields(self, field_weights, field_names, concatenate_keys, learn_weights, learning_rate):

        # Deal with default field_weights
        if field_weights is None:
            if len(self.entry_template) == 1:
                field_weights = [1]
            else:
                # Default is to all fields as keys except the last one, which is the value
                num_fields = len(self.entry_template)
                field_weights = [1] * num_fields
                field_weights[-1] = 0
        field_weights = np.atleast_1d(field_weights)
        if len(field_weights) == 1:
            field_weights = np.repeat(field_weights, len(self.entry_template))

        # Memory structure (field) attributes (not Parameters)
        self.num_fields = len(self.entry_template)
        self.field_weights = field_weights
        self.field_names = field_names.copy() if field_names is not None else None

        # Memory field attributes
        keys_weights = [i for i in self.field_weights if i != 0]
        self.num_keys = len(keys_weights)
        self.num_values = self.num_fields - self.num_keys
        if self.field_names:
            self.key_names = self.field_names[:self.num_keys]
            self.value_names = self.field_names[self.num_keys:]
        else:
            self.key_names = [f'KEY {i}' for i in range(self.num_keys)] if self.num_keys > 1 else ['KEY']
            self.value_names = [f'VALUE {i}' for i in range(self.num_values)] if self.num_values > 1 else ['VALUE']

        self.concatenate_keys = concatenate_keys or np.all(keys_weights == keys_weights[0])
        if self.concatenate_keys and not np.all(keys_weights == keys_weights[0]):
            warnings.warn(f"Field weights are not all equal, but 'concatenate_keys' is True; "
                          f"field weights will be ignored and all fields will be concatenated as keys.")

        self.learn_weights = learn_weights
        self.learning_rate = learning_rate

    def _construct_pathway(self)->set:
        """Construct pathway for EMComposition"""

        # Construct nodes of Composition
        self.key_input_nodes = self._construct_key_input_nodes()
        self.value_input_nodes = self._construct_value_input_nodes()
        self.concatenate_keys_node = self._construct_concatenate_keys_node()
        self.match_nodes = self._construct_match_nodes()
        self.softmax_nodes = self._construct_softmax_nodes()
        self.softmax_control_nodes = self._construct_softmax_control_nodes()
        self.retrieval_gating_nodes = self._construct_retrieval_gating_nodes()
        self.retrieval_weighting_node = self._construct_retrieval_weighting_node()
        self.retrieval_nodes = self._construct_retrieval_nodes()
        self.input_nodes = self.key_input_nodes + self.value_input_nodes
        # self.storage_nodes = self._construct_storage_nodes()

        # Construct pathway as a set of nodes, since Projections are specified in the construction of each node
        pathway = set(self.key_input_nodes + self.value_input_nodes
                      + self.concatenate_keys_node if self.concatenate_keys_node is not None else []
                      + self.match_nodes + self.softmax_control_nodes + self.softmax_nodes \
                      + [self.retrieval_weighting_node] + self.retrieval_gating_nodes + self.retrieval_nodes)

        return pathway

    def _construct_key_input_nodes(self)->list:
        """Create one node for each key to be used as cue for retrieval (and then stored) in memory.
        Used to assign new set of weights for Projection for key_input_node[i] -> match_node[i]
        where i is selected randomly without replacement from (0->memory_capacity)
        """
        
        # Get indices of field_weights that specify keys:
        key_indices = np.nonzero(self.field_weights)[0]

        assert len(key_indices) == self.num_keys, \
            f"PROGRAM ERROR: number of keys ({self.num_keys}) does not match number of " \
            f"non-zero values in field_weights ({len(key_indices)})."

        key_input_nodes = [TransferMechanism(size=len(self.entry_template[key_indices[i]]),
                                             name= f'{self.key_names[i]} INPUT')
                       for i in range(self.num_keys)]

        return key_input_nodes

    def _construct_value_input_nodes(self)->list:
        """Create one input node for each value to be stored in memory.
        Used to assign new set of weights for Projection for retrieval_weighting_node -> retrieval_node[i]
        where i is selected randomly without replacement from (0->memory_capacity)
        """

        # Get indices of field_weights that specify keys:
        value_indices = np.where(self.field_weights == 0)[0]

        assert len(value_indices) == self.num_values, \
            f"PROGRAM ERROR: number of values ({self.num_values}) does not match number of " \
            f"non-zero values in field_weights ({len(value_indices)})."

        value_input_nodes = [TransferMechanism(size=len(self.entry_template[value_indices[i]]),
                                               name= f'{self.value_names[i]} INPUT')
                           for i in range(self.num_values)]

        return value_input_nodes

    def _construct_concatenate_keys_node(self)->ProcessingMechanism:
        """Create node that concatenates the inputs for all keys into a single vector
        Used to create a matrix for Projectoin from match / memory weights from concatenate_node -> match_node
        """
        # One node that concatenates inputs from all keys
        if not self.concatenate_keys:
            return None
        else:
            return ProcessingMechanism(function=Concatenate,
                                       input_ports=[{NAME: 'CONCATENATE_KEYS',
                                                     SIZE: len(self.key_input_nodes[i].output_port.value),
                                                     PROJECTIONS: MappingProjection(
                                                         sender=self.key_input_nodes[i].output_port,
                                                         matrix=IDENTITY_MATRIX)}
                                                    for i in range(self.num_keys)],
                                       name='CONCATENATE KEYS')

    def _construct_match_nodes(self)->list:
        """Create nodes that, for each key field, compute the similarity between the input and each item in memory.
        Each element of the output represents the similarity between the key_input and one item in memory.

        If self.concatenate_keys is True, then all inputs for keys are concatenated into a single vector that is
        the input to a single TransferMechanism;  otherwise, each key has its own TransferMechanism.  The size of
        the input to each TransferMechanism is the number of items allowed in memory, and the weights to each element
        in the weight matrix is a single memory.
        """

        if self.concatenate_keys:
            # Get fields of memory structure corresponding to the keys
            matrix = self.memory_template[:,:self.num_keys].transpose()
            match_nodes = [
                TransferMechanism(
                    input_ports={NAME: 'CONCATENATED_INPUTS',
                                 SIZE: self.memory_capacity,
                                 PROJECTIONS: MappingProjection(sender=self.concatenate_keys_node,
                                                                # matrix=ZEROS_MATRIX,
                                                                matrix=matrix,
                                                                function=LinearMatrix(
                                                                    normalize=self.normalize_memories))},
            name='MATCH')]
        else:
            # One node for each key
            match_nodes = [
                TransferMechanism(
                    input_ports=
                    {
                        SIZE:self.memory_capacity,
                        PROJECTIONS: MappingProjection(sender=self.key_input_nodes[i].output_port,
                                                       # matrix=ZEROS_MATRIX,
                                                       matrix=self.memory_template[:,i].transpose(),
                                                       function=LinearMatrix(normalize=self.normalize_memories))},
                    name=f'MATCH {self.key_names[i]}')
                for i in range(self.num_keys)
            ]
        return match_nodes

    def _construct_softmax_control_nodes(self)->list:
        """Create nodes that set the softmax gain (inverse temperature) for each softmax_node."""
        if self.softmax_gain != CONTROL:
            return []
        softmax_control_nodes = [
            ControlMechanism(monitor_for_control=match_node,
                             control_signals=[(GAIN, self.softmax_nodes[i])],
                             function=get_softmax_gain,
                             name='SOFTMAX GAIN CONTROL' if len(self.softmax_nodes) == 1
                             else f'SOFTMAX GAIN CONTROL {i}')

            for i, match_node in enumerate(self.match_nodes)]
        return softmax_control_nodes

    def _construct_softmax_nodes(self)->list:
        # FIX:
        """Create nodes that, for each key field, compute the softmax over the similarities between the input and the
        memories in the corresponding match_node.
        """

        # Get indices of field_weights that specify keys:
        key_indices = np.where(np.array(self.field_weights) != 0)
        key_weights = [self.field_weights[i] for i in key_indices[0]]

        assert len(key_indices[0]) == self.num_keys, \
            f"PROGRAM ERROR: number of keys ({self.num_keys}) does not match number of " \
            f"non-zero values in field_weights ({len(key_indices)})."

        # If softmax_gain is specified as CONTROL, then set to None for now
        if self.parameters.softmax_gain.get() == CONTROL:
            softmax_gain = None
        # Otherwise, assign specified value
        else:
            softmax_gain = self.parameters.softmax_gain.get()

        softmax_nodes = [
            TransferMechanism(
                input_ports={SIZE:self.memory_capacity,
                             PROJECTIONS: match_node.output_port},
                function=SoftMax(gain=softmax_gain),
                output_ports=[RESULT,
                              {VALUE: key_weights[0],
                               NAME: 'KEY_WEIGHT'}],
               name='SOFTMAX' if len(self.match_nodes) == 1
               else f'SOFTMAX {i}')
            for i, match_node in enumerate(self.match_nodes)
        ]

        return softmax_nodes

    def _construct_retrieval_gating_nodes(self)->list:
        """Create GatingMechanisms that weight each key's contribution to the retrieved values.
        """
        # FIX: CONSIDER USING THIS FOR INPUT GATING OF MATCH NODE(S)?
        if self.concatenate_keys:
            retrieval_gating_nodes = []
        else:
            retrieval_gating_nodes = [GatingMechanism(input_ports={PROJECTIONS: key_match_pair[0].output_ports['RESULT'],
                                                                   NAME: 'OUTCOME'},
                                                      gate=[key_match_pair[1].output_ports[1]],
                                                      name= 'RETRIEVAL WEIGHTING' if self.num_keys == 1
                                                      else f'RETRIEVAL WEIGHTING {i}')
                                      for i, key_match_pair in enumerate(zip(self.key_input_nodes, self.softmax_nodes))]
        return retrieval_gating_nodes

    def _construct_retrieval_weighting_node(self)->list:
        """Create nodes that compute the weighting of each item in memory.
        """
        # FIX: THIS SHOULD WORK:
        retrieval_weighting_node = TransferMechanism(input_ports=[m.output_port for m in self.softmax_nodes],
                                                     name='WEIGHT RETRIEVALS')

        assert len(retrieval_weighting_node.output_port.value) == self.memory_capacity,\
            f'PROGRAM ERROR: number of items in retrieval_weighting_node ({len(retrieval_weighting_node.output_port)})' \
            f'does not match memory_capacity ({self.memory_capacity})'
                                                            
        return retrieval_weighting_node

    def _construct_retrieval_nodes(self)->list:
        """Create nodes that report the value field(s) for the item(s) matched in memory.
        """

        self.retrieved_key_nodes = [TransferMechanism(input_ports={SIZE: len(self.key_input_nodes[i].variable[0]),
                                                                   PROJECTIONS:
                                                                       MappingProjection(
                                                                           sender=self.retrieval_weighting_node,
                                                                           # matrix=ZEROS_MATRIX)
                                                                           matrix=self.memory_template[:,i])
                                                                   },
                                                      name= f'{self.key_names[i]} RETRIEVED')
                                    for i in range(self.num_keys)]

        self.retrieved_value_nodes = [TransferMechanism(input_ports={SIZE: len(self.value_input_nodes[i].variable[0]),
                                                                     PROJECTIONS:
                                                                         MappingProjection(
                                                                             sender=self.retrieval_weighting_node,
                                                                             # matrix=ZEROS_MATRIX)
                                                                             matrix=self.memory_template[:,
                                                                                    i+self.num_keys])
                                                                     },
                                                        name= f'{self.value_names[i]} RETRIEVED')
                                      for i in range(self.num_values)]

        return self.retrieved_key_nodes + self.retrieved_value_nodes

    def _construct_storage_nodes(self)->list:
        """Create LearningMechanisms that store the input values in memory.
        Memories are stored by adding the current input to each field to the corresponding row of the matrix
        for the Projection from the key_input_node to the matching_node for keys, and from the value_input_node to
        the retrieval node for values.
        """
        # FIX: ASSIGN RELEVANT PROJECTIONS TO LEARNING MECHANISM ATTRIBUTES, AND ASSIGN FUNCTION THAT USES THOSE
        self.key_storage_nodes = [LearningMechanism(size=len(self.key_input_nodes[i].value[0]),
                                                    input_ports=self.key_input_nodes[i].output_port,
                                                    name= f'{self.key_names[i]} STORAGE')
                                  for i in range(self.num_keys)]

        self.value_storage_nodes = [LearningMechanism(size=len(self.value_input_nodes[i].value[0]),
                                                      name= f'{self.value_names[i]} STORAGE')
                                    for i in range(self.num_values)]

        return self.key_storage_nodes + self.value_storage_nodes


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

    def _construct_memory(self, memory_template, memory_fill):
        """Initialize memory by zeroing weights from:
        - key_input_node(s) to match_node(s) and
        - retrieval_weighting_node to retrieval_node(s)
        and then storing memory_template if it was specified as a list or array (vs. a shape)
        """
        # create inputs:
        inputs = {node:memory for node, memory in zip(self.input_nodes, self.entry_template)}
        self._encode_memory(inputs)

        if isinstance(memory_fill, tuple):
            if memory_fill[0] != RANDOM:
                raise EMCompositionError(f"The 'memory_fill' arg of '{self.name}' is a tuple ({memory_fill}), "
                                         f"so the first item must be the keyword 'RANDOM'")
                # memory_fill = np.random.uniform(*memory_fill[1:], size=memory_template.shape)
            memory_fill = RANDOM
            scale = memory_fill[1]

        memory_field = []
        if memory_template.ndim == 2:
            for field in memory_template:
                memory_field.append(np.full((self.memory_capacity, len(field)),memory_fill))


            for i, key_input_node in enumerate(self.key_input_nodes):
                key_input_node.output_port.value = np.zeros_like(key_input_node.output_port.value)
                key_input_node.output
            memory = np.array(memory_template)


        # FIX: ASSIGN FILL
        # ASSIGN TO PROJECTIONS FROM KEY_INPUT_NODE(S) TO MATCH_NODE(S)
        # AND FROM RETRIEVAL_WEIGHTING_NODE TO RETRIEVAL_NODE(S)

    def execute(self, inputs, context, **kwargs):
        """Set input to weights of Projection to match_node."""
        results = super().execute(inputs, **kwargs)
        self._store_memory(inputs, context)
        return results

    def _store_memory(self, inputs, context):
        """Store inputs in memory as weights of Projections to softmax_nodes (keys) and retrieval_nodes (values).
        """
        storage_prob = np.array(self._get_current_parameter_value(STORAGE_PROB, context)).astype(float)
        random_state = self._get_current_parameter_value('random_state', context)

        if storage_prob == 0.0 or (storage_prob > 0.0 and storage_prob < random_state.uniform()):
            return
        self._encode_memory(inputs, context)

    def _encode_memory(self, inputs, context=None):
        for i, input in enumerate(inputs.items()):
            input_node = input[0]
            memory = input[1]
            # Memory = key_input or value_input
            # memories = weights of Projections for each field

            # Store key_input vector in projections from input_key_nodes to match_nodes
            if input_node in self.key_input_nodes:
                # For key_input:
                #   assign as weights for first empty row of Projection.matrix from key_input_node to match_node
                memories = input_node.efferents[0].parameters.matrix.get(context)
                if self.memory_decay:
                    memories *= self.memory_decay_rate
                # Get least used slot (i.e., weakest memory = row of matrix with lowest weights)
                # idx_of_min = np.argmin(memories.sum(axis=0))
                idx_of_min = np.argmin(np.linalg.norm(memories, axis=0))
                memories[:,idx_of_min] = np.array(memory)
                input_node.efferents[0].parameters.matrix.set(memories, context)

            # For all inputs, assign input vector to afferent weights of corresponding retrieval_node
            memories = self.retrieval_nodes[i].path_afferents[0].parameters.matrix.get(context)
            if self.memory_decay:
                memories *= self.memory_decay_rate
            # Get least used slot (i.e., weakest memory = row of matrix with lowest weights)
            idx_of_min = np.argmin(memories.sum(axis=1))
            memories[idx_of_min] = np.array(memory)
            self.retrieval_nodes[i].path_afferents[0].parameters.matrix.set(memories, context)
