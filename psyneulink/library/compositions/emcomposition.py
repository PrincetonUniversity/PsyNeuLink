# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* EMComposition *************************************************

# TODO:
# - FIX: WRITE TESTS
# - FIX: WARNING NOT OCCURING FOR ZEROS WITH MULTIPLE ENTRIES (HAPPENS IF *ANY* KEY IS EVER ALL ZEROS)
# - FIX: ALLOW memory_template TO BE 3-ITEM TUPLE IN WHICH 1ST ITEM SPECIFIES MEMORY CAPACITY
#        DEFAULTS TO memory_capacity; IF memory_capacity IS USER-SPECIFIED AND THEY CONFLICT -> ERROR MESSAGE
# - FIX: - ADD add_memory() METHOD
# - FIX: - HANDLE Nones in args
# - FIX: LEARNING:
#        - ADD LEARNING MECHANISMS TO STORE MEMORY AND ADJUST WEIGHTS
#        - DEAL WITH ERROR SIGNALS to retrieval_weighting_node OR AS PASS-THROUGH
# - FIX: CONFIDENCE COMPUTATION (USING SIGMOID ON DOT PRODUCTS) AND REPORT THAT (EVEN ON FIRST CALL)
# - FIX: ALLOW SOFTMAX SPEC TO BE A DICT WITH PARAMETERS FOR _get_softmax_gain() FUNCTION
# - FIX: CONCATENATE *ANY* FIELDS THAT ARE THE SAME WEIGHT (FOR EFFICIENCY)
# - FIX: COMPILE
#      - LinearMatrix to add normalization
#      - _store() method to assign weights to memory
# - FIX: AUGMENT LinearMatrix Function:
#        - Normalize as option
#        - Anytime a row's norm is 0, replace with 1s
# - FIX: WHY IS Concatenate NOT WORKING AS FUNCTION OF AN INPUTPORT (WASN'T THAT USED IN CONTEXT OF BUFFER?)
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
     - `Memory Template and Fill <EMComposition_Example_Memory_Template>`
     - `Field Weights <EMComposition_Example_Field_Weights>`
  * `EMComposition_Class_Reference`


.. _EMComposition_Overview:

Overview
--------

The EMComposition implements a configurable, content-addressable form of episodic, or eternal memory, that emulates
an `EpisodicMemoryMechanism` -- reproducing all of the functionality of its `ContentAddressableMemory` `Function` --
in the form of an `AutodiffComposition` that is capable of learning how to differentially weight different cues used
for retrieval,, and that adds the capability for `memory_decay <EMComposition.memory_decay>`. Its `memory
<EMComposition.memory>` is configured using the ``memory_template`` argument of its constructor, which defines how
each entry in `memory <EMComposition.memory>` is structured (the number of fields in each entry and the length of
each field), and its ``field_weights`` argument that defines which fields are used as cues for retrieval -- "keys" --
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
`field_weights <ContentAddressableMemory.memory>` parameter (see `retrieval <EMComposition_Retrieval_Storage>` below).
The number and shape of the fields in each entry is specified in the ``memory_template`` argument of the EMComposition's
constructor (see `memory_template <EMComposition_Fields>`). Which fields treated as keys (i.e., used as cues for
retrieval) and which are treated as values (i.e., retrieved but not used for matching retrieval) is specified in the
``field_weights`` argument of the EMComposition's constructor (see `field_weights <EMComposition_Field_Weights>`).

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
  TBD DISTANCE ATTRIBUTES:
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
  ``memory_template`` can be specified in one of three ways (see `EMComposition_Examples` for
  representative use cases):

  .. hint::
     Using the default initialization of memory with all zeros and ``normalize_memories`` set to ``True``
     (see `below <EMComposition_Retrieval_Storage>`) results in a numpy.linalg warning about divide by zero.
     This can be ignored, as it does not affect the results of execution, but it can be averted by specifying
     `memory_fill <EMComposition_Memory_Fill>` to use small random values (e.g., ``memory_fill=(0,.001)``).

  * **tuple**: interpreted as an np.array shape specification, in which the 1st item specifies the number of fields in
    each memory entry and the 2nd item specifies the length of each field.  The matrix is filled either with zeros or
    with the value specified in ``memory_fill`` (see below).

  * **2d list or array**:  interpreted as a template for memory entries.  This can be used to specify fields of
    different lengths (i.e., entries that are ragged arrays), with each item in the list (axis 0 of the array) used to
    specify the length of the corresponding field.  The template is broadcast over the third dimension to generate the
    full matrix used to initialize `memory <EMComposition.memory>`.  If the template uses any non-zero values, then the
    array is replicated for all entries in `memory <EMComposition.memory>`.  If the template has all zeros, then the
    the entire `memory <EMComposition.memory>` is filled with either zeros or the value specified in ``memory_fill``.

    .. hint::
       To specify a single entry, with all other entries filled with zeros
       or the value specified in ``memory_fill``, use a 3d array as described below.

  * **3d list or array**:  used to initialize `memory <EMComposition.memory>` directly. If the outer dimension of
    the list or array (axis 0) has fewer than ``memory_capacity`` items, then it is filled with the remaining entries,
    using either zeros or the value specified in ``memory_fill`` (see below).  If all of thespecified entries are
    zeros and ``memory_fill`` is specified, then the matrix is filled with the value specified in ``memory_fill``.

.. _EMComposition_Memory_Fill:

* **memory_fill**: specifies the value used to fill the `memory <EMComposition.memory>`, based on the shape specified
  in the ``memory_template`` (see above).  The value can be a scalar, or a tuple to specify an interval over which
  to draw random values to fill `memory <EMComposition.memory>` --- both should be scalars, with the first specifying
  the lower bound and the second the upper bound.

.. _EMComposition_Field_Weights:

* **field_weights**: specifies which fields are used as keys, and how they are weighted during retrieval. The
  number of values specified must match the number of fields specified in ``memory_template`` (i.e., the size of
  of its first dimension (axis 0)).  All non-zero entries must be positive, and designate *keys* -- fields
  that are used to match items in memory for retrieval (see `Match memories by field <EMComposition_Processing>`).
  Entries of 0 designate *values* -- fields that are ignored during the matching process, but the values of which
  are retrieved and assigned as the `value <Mechanism_Base.value>` of the corresponding `retrieval_node
  <EMComposition.retrieval_nodes>`. This distinction between keys and value implements a standard "dictionary; however,
  if all entries are non-zero, then all fields are treated as keys, implemented a full form of content-addressable
  memory.  If ``learn_weights`` is True, the field_weights can be modified during training; otherwise they remain
  fixed. The following options can be used to specify ``field_weights``:

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
  This is True by default, if the `field_weights <EMComposition.field_weights>` for all keys are equal (i.e.,
  all non-zero weights are equal -- see `field_weights <EMComposition_Field_Weights>`) and `normalize_memories
  <EMComposition.normalize_memories>` is True (the default).  However, if the key `field_weights
  <EMComposition.field_weights>` are *not* all equal, or ``normalize_memories`` is set to False, then a warning
  is issued and concatenation is disabled (i.e., all keys are treated separately). If ``concatenate_keys`` is set
  to False, then keys are not concatenated irrepsective of `field_weights <EMComposition.field_weights>` or
  `normalize_memories <EMComposition.normalize_memories>`.

      .. technical_note::
         If `normalize_memories <EMComposition.normalize_memories>` is True, then concatenating keys has no influence
         on the outcome of the `matching process <EMComposition_Processing>`, however it can be more computationally
         efficient. However, if `normalize_memories <EMComposition.normalize_memories>` is False, it can affect the
         outcome of the matching process, since different fields may have different norms that will result in different
         dot products in the `matching process <EMComposition_Processing>`.

      .. note::
         All `key_input_nodes <EMComposition.key_input_nodes>` and `retrieval_nodes <EMComposition.retrieval_nodes>`
         are always preserved, even when `concatenate_keys <EMComposition.concatenate_keys>` is True, so that separate
         inputs can be provided for each key, and the value of each key can be retrieved separately.

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

* **storage_prob** : specifies the probability that the inputs to the EMComposition will be stored as an item in
  `memory <EMComposition.memory>` on each execution.

* **normalize_memories** : specifies whether keys and memories are normalized before computing their dot products.

.. _EMComposition_Softmax_Gain:

* **softmax_gain** : specifies the gain (inverse temperature) used for softmax normalizing the dot products of keys
  and memories (see `EMComposition_Execution` below).  If a value is specified, that is used.  If the keyword *CONTROL*
  is (or the value is None), then the `softmax_gain <EMComposition.softmax_gain>` function is used to adaptively set
  the gain based on the entropy of the dot products, preserving the distribution over non-(or near) zero entries
  irrespective of how many (near) zero entries there are.

* **learn_weights** : specifies whether `field_weights <EMComposition.field_weights>` are modifiable during training.

* **learning_rate** : specifies the rate at which  `field_weights <EMComposition.field_weights>` are learned if
  ``learn_weights`` is True.


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

When the EMComposition is executed, the following sequence of operations occur
(also see `figure <EMComposition_Example_Fig>`):

* **Concatenation**. By default, if the `field_weights <EMComposition.field_weights>` are the same for all `keys
  <EMComposition_Field_Weights>` and `normalize_memories <EMComposition.normalize_memories>` is True then, for
  efficiency of computation, the inputs provided to the `key_input_nodes <EMComposition.key_input_nodes>` are
  concatenated into a single vector in the `concatenate_keys_node <EMComposition.concatenate_keys_node>`, that
  is provided to a single `match_node <EMComposition.match_nodes>`.  However, if either of these conditions is
  not met or `concatenate_keys <EMComposition.concatenate_keys>`is False, then the input to each `key_input_node
  <EMComposition.key_input_nodes>` is provided to its own `match_node <EMComposition.match_nodes>`
  (see `concatenate keys <EMComposition_Concatenate_Keys>` for additional information).

* **Match memories by field**. The values of each `key_input_node <EMComposition.key_input_nodes>` (or the
  `concatenate_keys_node <EMComposition.concatenate_keys_node>` if `concatenate_keys <EMComposition_Concatenate_Keys>`
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

The following are examples of how to configure and initialize the EMComposition's `memory <EMComposition.memory>`:

*Visualizing the EMComposition*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The EMComposition can be visualized graphically, like any `Composition`, using its `show_graph
<Composition.show_graph>` method.  For example, the figure below shows the following EMComposition
that has 2 keys and 1 value::

    >>> import psyneulink as pnl
    >>> em = EMComposition(memory_template=(3,2), memory_capacity=4)
    >>> em.show_graph()
    <BLANKLINE>

.. _EMComposition_Example_fig:

.. figure:: _static/EMComposition_Example_fig.svg
   :alt: Exxample of an EMComposition
   :align: left

       **Example of an EMComposition**

       .. note::
          The order in which the nodes at a given level (e.g., the `INPUT <NodeRole.INPUT>` or `OUTPUT
          <NodeRole.OUTPUT>` `Nodes <Composition_Nodes>`) are shown in the diagram is arbitrary, and does not necessarily
          reflect the order in which they are created or specied in the script.

.. _EMComposition_Example_Memory_Template:

*Memory Template*
~~~~~~~~~~~~~~~~~

The ``memory_template`` argument is used to configure the EMComposition's `memory <EMComposition.memory>`, which
can be specified with by a tuple or a list or array.

.. _EMComposition_Example_Tuple_Spec:

**Tuple specification**

A tuple can be used to specify the number of fields and the length of each field in memory.  In the example above,
a tuple is used to specify that EMComposition's memory should four entries, each of which has two fields of length
3 each.  The contents of `memory <EMComposition.memory>` can be see using it `memory <EMComposition.memory>`
attribute::

    >>> em.memory
    [[[array([0., 0., 0.]), array([0., 0., 0.])]],
     [[array([0., 0., 0.]), array([0., 0., 0.])]],
     [[array([0., 0., 0.]), array([0., 0., 0.])]],
     [[array([0., 0., 0.]), array([0., 0., 0.])]]]

Note that there are four entries (rows) each with two fields (columns) that is each of length 3. The number of entries
was determined by ``memory_capacity``.  The default for ``memory_capacity`` is 1000, but 4 is used here for legibility.
The specification of ``memory_template`` above is equivalent to the following use of a list or array to specify
``memory_template``::

    >>> em = EMComposition(memory_template=[[0,0,0],[0,0,0]], memory_capacity=4)

**List or array specification**

Note that in the example above the two fields have the same length (3). This is always the case when a tuple is used,
as it generates a regular array.  However, a list or array can be used to specify fields of different length (i.e.,
as a ragged array).  For example, the following specifies one field of length 3 and another of length 1::

    >>> em = EMComposition(memory_template=[[0,0,0],[0]], memory_capacity=4)
    >>> em.memory
    [[[array([0., 0., 0.]), array([0.])]],
     [[array([0., 0., 0.]), array([0.])]],
     [[array([0., 0., 0.]), array([0.])]],
     [[array([0., 0., 0.]), array([0.])]]]

.. _EMComposition_Example_Memory_Fill:

**Memory fill**

Note that the examples above generate a warning about the use zeros to initialize the memory.  This is because the
default value for ``memory_fill`` is ``0``, and the default value for ``normalize_memories`` is ``True``, which
will cause a divide by zero warning when memories are normalized  While numpy handles this gracefully, albeit with
a warning, it can be avoided by specifying a non-zero value for ``memory_fill``, such as small number::

    >>> em = EMComposition(memory_template=[[0,0,0],[0]], memory_capacity=4, memory_fill=.001)
    >>> em.memory
    [[[array([0.001, 0.001, 0.001]), array([0.001])]],
     [[array([0.001, 0.001, 0.001]), array([0.001])]],
     [[array([0.001, 0.001, 0.001]), array([0.001])]],
     [[array([0.001, 0.001, 0.001]), array([0.001])]]]

Here, a single value was specified for ``memory_fill`` (which can be a float or int) that is used to fill all values.
A tuple can also be specified, in which case it is used to generate a random number in the internval between the first
and second values of the tuple.  For example, the following uses random values between 0 and 0.01 to fill all entries::

    >>> em = EMComposition(memory_template=[[0,0,0],[0]], memory_capacity=4, memory_fill=(0,0.01))
    >>> em.memory
    [[[array([0.00298981, 0.00563404, 0.00444073]), array([0.00245373])]],
     [[array([0.00148447, 0.00666486, 0.00228882]), array([0.00237541])]],
     [[array([0.00432786, 0.00035378, 0.00265932]), array([0.00980598])]],
     [[array([0.00151163, 0.00889032, 0.00899815]), array([0.00854529])]]]

.. _EMComposition_Example_Multiple_Entries:

**Multiple entries**

In the examples above, a single entry was specified, and that was used as a template for initializing the remaining
entries in memory. However, a list or array can be used to directly initialize any or all entries. For example, the
following initializes memory with two specific entries::

    >>> em = EMComposition(memory_template=[[[1,2,3],[4]],[[100,101,102],[103]]], memory_capacity=4)
    >>> em.memory
    [[[array([1., 2., 3.]), array([4.])]],
     [[array([100., 101., 102.]), array([103.])]],
     [[array([0., 0., 0.]), array([0.])]],
     [[array([0., 0., 0.]), array([0.])]]]

Note that the two entries must have exactly the same shapes. If they do not, and error will be generated.
Also note that the remaining entries are filled with zeros (the default value for ``memory_fill``).
Here again, ``memory_fill`` can be used to specify a different default value::

    >>> em = EMComposition(memory_template=[[[7],[24,5]],[[100],[3,106]]], memory_capacity=4, memory_fill=(0,.01))
    >>> em.memory
    [[[array([7.]), array([24.,  5.])]],
     [[array([100.]), array([  3., 106.])]],
     [[array([0.00803646]), array([0.00341276, 0.00286969])]],
     [[array([0.00143196]), array([0.00079033, 0.00710556])]]]

.. _EMComposition_Example_Field_Weights:

*Field Weights*
~~~~~~~~~~~~~~~

By default, all of the fields specified are treated as keys except the last, which is treated as a "value" field --
that is, one that is not included in the matching process, but for which a value is retrieved along with the key fields.
For example, in the `figure <EMComposition_Example_fig>` above, of the three fields specified, the first two are used as
keys, and the last is used as a value. However, the ``field_weights`` argument can be used to modify this, specifying
which fields should be used as keys, as well as the relative contribution that each makes to the matching process, and
which should be used as value fields.  Non-zero elements in the ``field_weights`` argument designate keys, and zeros
specify value fields.  For example, the following specifies that the first two fields should be used as keys while
the last two should be used as values::

    >>> em = EMComposition(memory_template=[[0,0,0],[0],[0,0],[0,0,0,0]], memory_capacity=3, field_weights=[1,1,0,0])
    >>> em.show_graph()
    <BLANKLINE>


.. _EMComposition_Example_Field_Weights_Equal_fig:

.. figure:: _static/EMComposition_field_weights_equal_fig.svg

    **Use of field_weights to specify keys and values.**

The ``field_weights`` argument can also be used to specify the relative contribution of each field to the matching
process.  By default, all non-zero values are set to 1, but different values can be used to weight the
relative contribution of each field.  The values are normalized so that the sum of all non-zero values is 1, and the
relative contribution of each is determined by the ratio of its value to the sum of all non-zero values.  For example,
the following specifies that the first two fields should be used as keys, with the first contributing 75% to the
matching process and the second field should contribute 25%::

    >>> em = EMComposition(memory_template=[[0,0,0],[0],[0,0]], memory_capacity=3, field_weights=[3,1,0])
    >>> em.show_graph()
    <BLANKLINE>

.. _EMComposition_Example_Field_Weights_Different_fig:

.. figure:: _static/EMComposition_field_weights_different.svg

    **Use of field_weights to specify relative contribution of fields to matching process.**

Note that in this case, the `concatenate_keys_node <EMComposition.concatenate_keys_node>` has been replaced by a
pair of `retreival_weighting_nodes <EMComposition.retrieval_gating_nodes>`, one for each key field.  This is because
the keys were assigned different weights;  when they are assigned equal weights, or if no weights are specified,
and `normalize_memories <EMComposition.normalize_memories>` is `True`, then the keys are concatenated and are
concatenated for efficiency of processing.  This can be suppressed by specifying `concatenate_keys` as `False`
(see `concatenate_keys <EMComposition_Concatenate_Keys>` for additional details).

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

def _memory_getter(owning_component=None, context=None)->list: # FIX: MAKE THIS A PARAMETER
    """Return list of memories in which rows (outer dimension) are memories for each field.
    These are derived from `matrix <MappingProjection.matrix>` parameter of the `afferent
    <Mechanism_Base.afferents>` MappingProjections to each of the `retrieval_nodes <EMComposition.retrieval_nodes>`.
    """
    # Get memory from Projection(s) to each retrieval_node
    memory = [retrieval_node.path_afferents[0].parameters.matrix.get(context)
              for retrieval_node in owning_component.retrieval_nodes]
    # Reorganize memory so that each row is an entry and each column is a field
    return [[memory[j][i] for j in range(owning_component.num_fields)]
              for i in range(owning_component.memory_capacity)]

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
        concatenate_keys=True,      \
        learn_weights=True,         \
        learning_rate=True,         \
        memory_capacity=1000,       \
        memory_decay=True,          \
        memory_decay_rate=.001,     \
        storage_prob=1.0,           \
        name="EM_Composition"       \
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

    concatenate_keys : bool : default True
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
        specifies rate at which `field_weights <EMComposition.field_weights>` are learned if ``learn_weights`` is True.

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

    memory : list[list[list[float]]]
        list of entries in memory, in which each row (outer dimensions) is an entry and each item in the row is the
        value for the corresponding field;  see `EMComposition_Memory` for additional details.

        .. note::
           This is a read-only attribute;  memories can be added to the EMComposition's memory either by
           COMMENT:
           using its `add_to_memory <EMComposition.add_to_memory>` method, or
           COMMENT
           executing its `run <Composition.run>` or learn methods with the entry as the ``inputs`` argument.

    .. _EMComposition_Parameters:

    field_weights : list[float]
        determines which fields of the input are treated as "keys" (non-zero values), used to match entries in `memory
        <EMComposition.memory>` for retrieval, and which are used as "values" (zero values), that are stored and
        retrieved from memory, but not used in the match process (see `Match memories by field
        <EMComposition_Processing>`;  see `field_weights <EMComposition_Field_Weights>` for additional details
        of specification).

    field_names : list[str]
        determines which names that can be used to label fields in `memory <EMComposition.memory>`;  see
        `field_names <EMComposition_Field_Names>` for additional details.

    learn_weights : bool
        determines whether `field_weights <EMComposition.field_weights>` are learnable during training; see
        `Learning <EMComposition_Learning>` for additional details.

    learning_rate : float
        determines whether the rate at which `field_weights <EMComposition.field_weights>` are learned if
        `learn_weights` is True;  see `EMComposition_Learning>` for additional details.

    concatenate_keys : bool
        determines whether keys are concatenated into a single field before matching them to items in `memory
        <EMComposition.memory>`; see `concatenate keys <EMComposition_Concatenate_Keys>` for additional details.

    normalize_memories : bool
        determines whether keys and memories are normalized before computing their dot product (similarity);
        see `Match memories by field <EMComposition_Processing>` for additional details.

    softmax_gain : CONTROL
        determines gain (inverse temperature) used for softmax normalizing the dot products of keys and memories
        by the `softmax` function of the `softmax_nodes <EMComposition.softmax_nodes>`; see `Softmax normalize matches
        over fields <EMComposition_Processing>` for additional details.

    storage_prob : float
        determines the probability that an item will be stored in `memory <EMComposition.memory>`.

    memory_capacity : int
        determines the number of items that can be stored in `memory <EMComposition.memory>`; see `memory_capacity
        <EMComposition_Memory_Capacity>` for additional details.

    memory_decay : bool
        determines whether memories decay with each execution of the EMComposition.

    memory_decay_rate : float
        determines the rate at which items in the EMComposition's memory decay.

    .. _EMComposition_Nodes:

    key_input_nodes : list[TransferMechanism]
        `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` that receive keys used to determine the item
        to be retrieved from `memory <EMComposition.memory>`, and then themselves stored in `memory
        <EMComposition.memory>` (see `Match memories by field <EMComposition_Processing>` for additional details).

    value_input_nodes : list[TransferMechanism]
        `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` that receive values to be stored in `memory
        <EMComposition.memory>`; these are not used in the matching process used for retrieval

    concatenate_keys_node : TransferMechanism
        `TransferMechanism` that concatenates the inputs to `key_input_nodes <EMComposition.key_input_nodes>` into a
        single vector used for the matching processing if `concatenate keys <EMComposition.concatenate_keys>` is True.
        This is not created if the ``contatenate_keys`` argument to the EMComposition's constructor is False or is
        overridden (see `concatenate_keys <EMComposition_Concatenate_Keys>`), or there is only one key_input_node.

    match_nodes : list[TransferMechanism]
        `TransferMechanisms <TransferMechanism>` that receive the dot product of each key and those stored in
        the corresponding field of `memory <EMComposition.memory>` (see `Match memories by field
        <EMComposition_Processing>` for additional details).

    softmax_control_nodes : list[ControlMechanism]
        `ControlMechanisms <ControlMechanism>` that adaptively control the `softmax_gain <EMComposition.softmax_gain>`
        for the corresponding `softmax_nodes <EMComposition.softmax_nodes>`. These are implemented only if
        `softmax_gain <EMComposition.softmax_gain>` is specified as *CONTROL* (see `softmax_gain
        <EMComposition_Softmax_Gain>` for details).

    softmax_nodes : list[TransferMechanism]
        `TransferMechanisms <TransferMechanism>` that compute the softmax over the vectors received
        from the corresponding `match_nodes <EMComposition.match_nodes>` (see `Softmax normalize matches over fields
        <EMComposition_Processing>` for additional details).

    retrieval_gating_nodes : list[GatingMechanism]
        `GatingMechanisms <GatingMechanism>` that uses the `field weight <EMComposition.field_weights>` for each
        field to modulate the output of the corresponding `retrieval_node <EMComposition.retrieval_nodes>` before
        it is passed to the `retrieval_weighting_node <EMComposition.retrieval_weighting_node>`.  These are
        implemented only if differential weights are specified for the different fields in `field_weights
        <EMComposition.field_weights>`.

    retrieval_weighting_node : TransferMechanism
        `TransferMechanism` that receives the softmax normalized dot products of the keys and memories
        from the `softmax_nodes <EMComposition.softmax_nodes>`, weights these using `field_weights
        <EMComposition.field_weights>`, and haddamard sums those weighted dot products to produce a
        single weighting for each memory.

    retrieval_nodes : list[TransferMechanism]
        `TransferMechanisms <TransferMechanism>` that receive the vector retrieved for each field in `memory
        <EMComposition.memory>` (see `Retrieve values by field <EMComposition_Processing>` for additional details).

    """

    componentCategory = EM_COMPOSITION
    class Parameters(AutodiffComposition.Parameters):
        """
            Attributes
            ----------

                concatenate_keys
                    see `concatenate_keys <EMComposition.concatenate_keys>`

                    :default value: True
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

                    :default value: False # False UNTIL IMPLEMENTED
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
        memory = Parameter(None, loggable=True, getter=_memory_getter, read_only=True)
        # memory_template = Parameter([[0],[0]], structural=True, valid_types=(tuple, list, np.ndarray))
        memory_capacity = Parameter(1000, structural=True)
        field_weights = Parameter(None, structural=True)
        field_names = Parameter(None, structural=True)
        concatenate_keys = Parameter(True, structural=True)
        memory_decay = Parameter(True, loggable=True, modulable=True, fallback_default=True)
        memory_decay_rate = Parameter(None, loggable=True, modulable=True, fallback_default=True,
                               dependencies='memory_decay')
        normalize_memories = Parameter(True, loggable=False, fallback_default=True)
        learn_weights = Parameter(False, fallback_default=True) # FIX: False until learning is implemented
        learning_rate = Parameter(.001, fallback_default=True)
        storage_prob = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        softmax_gain = Parameter(CONTROL, modulable=True, fallback_default=True)
        random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies='seed')
        seed = Parameter(DEFAULT_SEED, modulable=True, fallback_default=True, setter=_seed_setter)

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
                 concatenate_keys:bool=True,
                 learn_weights:bool=False, # FIX: False FOR NOW, UNTIL IMPLEMENTED
                 learning_rate:float=None,
                 memory_capacity:int=1000,
                 memory_decay:bool=True,
                 memory_decay_rate:float=None,
                 normalize_memories:bool=True,
                 softmax_gain:Union[float, CONTROL]=CONTROL,
                 storage_prob:float=None,
                 name="EM_Composition"):

        # Construct memory --------------------------------------------------------------------------------

        self._validate_memory_specs(memory_template, memory_fill, field_weights, field_names, name)
        self._parse_memory_template(memory_template, memory_fill, memory_capacity, field_weights)
        self._parse_fields(field_weights, field_names, concatenate_keys, normalize_memories,
                           learn_weights, learning_rate, name)

        if self.parameters.memory_decay.get() and self.parameters.memory_decay_rate.get() is None:
            self.parameters.memory_decay_rate.set(memory_decay_rate or 1 / self.memory_capacity)
        self.softmax_gain = softmax_gain

        # Instantiate Composition -------------------------------------------------------------------------

        pathway = self._construct_pathway()
        super().__init__(pathway,
                         name=name)

        # Clean-up ----------------------------------------------------------------------------------------

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

        # Warn if divide by zero will occur due to memory initialization
        # # MODIFIED 7/16/23 OLD:
        # if not np.any([np.any([self.memory[i][0][j]
        #                        for i in range(self.memory_capacity)])
        #                for j in range(self.num_keys)]):
        # MODIFIED 7/16/23 NEW:
        if not np.any([np.any([self.memory[i][j]
                               for i in range(self.memory_capacity)])
                       for j in range(self.num_keys)]):
        # MODIFIED 7/16/23 END
            warnings.warn(f"Memory initialized with at least one field that has all zeros; "
                          f"a divide by zero will occur if 'normalize_memories' is True. "
                          f"This can be avoided by using 'memory_fill' to initialize memories with non-zero values.")

    # *****************************************************************************************************************
    # ***********************************  Memory Construction Methods  ***********************************************
    # *****************************************************************************************************************

    def _validate_memory_specs(self, memory_template, memory_fill, field_weights, field_names, name):
        """Validate the memory_template, field_weights, and field_names arguments
        """

        # memory_template must specify a 2D array:
        if isinstance(memory_template, tuple):
            if len(memory_template) != 2 or not all(isinstance(item, int) for item in memory_template):
                raise EMCompositionError(f"The 'memory_template' arg for {name} ({memory_template}) uses a tuple to "
                                         f"shape requires but does not have exactly two integers.")
            num_fields = memory_template[0]
        elif isinstance(memory_template, (list, np.ndarray)):
            num_entries, num_fields = self._parse_memory_shape(memory_template)
        else:
            raise EMCompositionError(f"Unrecognized specification for "
                                     f"the 'memory_template' arg ({memory_template}) of {name}.")

        # If a 3d array is specified (i.e., template has multiple entries), ensure all have the same shape
        if not isinstance(memory_template, tuple) and num_entries > 1:
            for entry in memory_template:
                if not (len(entry) == num_fields
                        and np.all([len(entry[i]) == len(memory_template[0][i]) for i in range(num_fields)])):
                    raise EMCompositionError(f"The 'memory_template' arg for {self.name} must specify a list "
                                             f"or 2d array that has the same shape for all entries.")

        # Validate memqory_fill specification (int, float, or tuple with two scalars)
        if not (isinstance(memory_fill, (int, float)) or
                (isinstance(memory_fill, tuple) and len(memory_fill)==2) and
                all(isinstance(item, (int, float)) for item in memory_fill)):
            raise EMCompositionError(f"The 'memory_fill' arg ({memory_fill}) specified for {name} "
                                     f"must be a float, int or len tuple of ints and/or floats.")

        # If field_weights has more than one value it must match the first dimension (axis 0) of memory_template:
        field_weights_len = len(np.atleast_1d(field_weights))
        if field_weights is not None and field_weights_len > 1 and field_weights_len != num_fields:
            raise EMCompositionError(f"The number of items ({field_weights_len}) in the 'field_weights' arg "
                                     f"for {name} must match the number of items in an entry of memory "
                                     f"({num_fields}).")

        # If field_names has more than one value it must match the first dimension (axis 0) of memory_template:
        if field_names and len(field_names) != num_fields:
            raise EMCompositionError(f"The number of items ({len(field_names)}) "
                                     f"in the 'field_names' arg for {name} must match "
                                     f"the number of fields ({field_weights_len}).")

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
                    entry = [np.full(len(field), memory_fill).tolist() for field in entry_template]
                entries = [np.array(entry, dtype=object)] * num_entries

            return np.array(np.array(entries,dtype=object), dtype=object)

        # If memory_template is a tuple, create and fill full memory matrix
        if isinstance(memory_template, tuple):
            memory = _construct_entries(np.full(memory_template, 0), memory_capacity, memory_fill)

        # If memory_template is a list or array
        else:
            # Determine whether template is a single entry or full/partial memory specification
            num_entries, num_fields = self._parse_memory_shape(memory_template)

            # memory_template specifies a single entry
            if num_entries == 1:
                # If any non-zeros, replicate the entry for full matrix
                # if any(np.array(memory_template, dtype=object).any()):
                # if any(np.nonzero(np.array(memory_template, dtype=object))):
                # if np.array(np.nonzero(np.array(memory_template, dtype=object))).any():
                if np.array([np.nonzero(field) for field in memory_template],dtype=object).any():
                    memory_fill = None
                # Otherwise, use memory_fill
                else:
                    memory_fill = memory_fill
                memory = _construct_entries(memory_template, memory_capacity, memory_fill)

            # If memory template is a full or partial 3d (matrix) specification
            else:
                # If all entries are zero, create entire memory matrix with memory_fill
                if not any(list(np.array(memory_template, dtype=object).flat)):
                    # Use first entry of zeros as template and replicate for full memory matrix
                    memory = _construct_entries(memory_template[0], memory_capacity, memory_fill)
                # If there are any non-zero values, keep specified entries and create rest using memory_fill
                else:
                    num_entries_needed = memory_capacity - len(memory_template)
                    # Get remaining entries populated with memory_fill
                    remaining_entries = _construct_entries(memory_template[0], num_entries_needed, memory_fill)
                    assert bool(num_entries_needed == len(remaining_entries))
                    # I any remaining entries, concatenate them with the entries that were specified
                    if num_entries_needed:
                        memory = np.concatenate((np.array(memory_template, dtype=object),
                                                 np.array(remaining_entries, dtype=object)))
                    # All entries were specivied, so just retun memory_template
                    else:
                        memory = np.array(memory_template, dtype=object)

        # Get shape of single entry
        self.entry_template = memory[0]
        self.memory_template = memory
        self.memory_capacity = memory_capacity

    def _parse_fields(self,
                      field_weights,
                      field_names,
                      concatenate_keys,
                      normalize_memories,
                      learn_weights,
                      learning_rate,
                      name):

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

        self.concatenate_keys = (concatenate_keys
                                 and self.num_keys > 1
                                 and np.all(keys_weights == keys_weights[0])
                                 and normalize_memories)
        # if concatenate_keys was forced off above:
        if concatenate_keys and not self.concatenate_keys:
            # Issue warning if concatenate_keys is True but either
            #   field weights are not all equal and/or normalize_memories is False
            fw_error_msg = nm_error_msg = fw_correction_msg = nm_correction_msg = None
            if not all(np.all(keys_weights[i] == keys_weights[0] for i in range(len(keys_weights)))):
                fw_error_msg = f" field weights ({field_weights}) are not all equal"
                fw_correction_msg = f"remove `field_weights` specification or make them all the same."
            if not normalize_memories:
                nm_error_msg = f" normalize_memories is False"
                nm_correction_msg = f" or set normalize_memories to True"
            if fw_error_msg and nm_error_msg:
                error_msg = f"{fw_error_msg} and {nm_error_msg}"
                correction_msg = f"{fw_correction_msg} and/or {nm_correction_msg}"
            else:
                error_msg = fw_error_msg or nm_error_msg
                correction_msg = fw_correction_msg or nm_correction_msg
            warnings.warn(f"The 'concatenate_keys' arg for '{name}' is True but {error_msg}; "
                          f"concatenation will be ignored. To use concatenation, {correction_msg}.")

        # FIX: UNTIL FULLY IMPLEMENTED
        if learn_weights:
            warnings.warn(f"The 'learn_weights' arg for '{name}' is True but not yet implemented; "
                          f"automatically set to False for now;  stay tuned...")
        # self.learn_weights = learn_weights
        self.learning_rate = learning_rate

    def _parse_memory_shape(self, memory_template):
        """Parse shape of memory_template to determine number of entries and fields"""
        memory_template_dim = np.array(memory_template, dtype=object).ndim
        # # MODIFIED 7/16/23 OLD:
        # if memory_template_dim == 1:
        #     fields_equal_length = all(len(field) == len(memory_template[0]) for field in memory_template)
        # else:
        #     if isinstance(memory_template[0], (int, float)):
        #         fields_equal_length = all(len(field) == len(memory_template[0]) for field in memory_template)
        #     else:
        #         fields_equal_length = all(len(field) == len(memory_template[0]) for field in memory_template[0])
        # MODIFIED 7/16/23 NEW:
        if memory_template_dim == 1 or all(isinstance(item, (int, float)) for item in memory_template[0]):
            fields_equal_length = all(len(field) == len(memory_template[0]) for field in memory_template)
        else:
            fields_equal_length = all(len(field) == len(memory_template[0]) for field in memory_template[0])
        # MODIFIED 7/16/23 END

        single_entry = (((memory_template_dim == 1) and not fields_equal_length) or
                        ((memory_template_dim == 2)  and fields_equal_length))
        num_entries = 1 if single_entry else len(memory_template)
        num_fields = len(memory_template) if single_entry else len(memory_template[0])
        return num_entries, num_fields

    # *****************************************************************************************************************
    # ******************************  Nodes and Pathway Construction Methods  *****************************************
    # *****************************************************************************************************************

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
                      + self.match_nodes + self.softmax_control_nodes + self.softmax_nodes \
                      + [self.retrieval_weighting_node] + self.retrieval_gating_nodes + self.retrieval_nodes)
        if self.concatenate_keys_node is not None:
            pathway.add(self.concatenate_keys_node)

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
        - If self.concatenate_keys is True, then all inputs for keys from concatenated_keys_node are assigned a single
            match_node, and weights from memory_template are assigned to a Projection from concatenated_keys_node to
            that match_node.
        - Otherwise, each key has its own match_node, and weights from memory_template are assigned to a Projection
            from each key_input_node[i] to each match_node[i].
        - Each element of the output represents the similarity between the key_input and one item in memory.
        """

        if self.concatenate_keys:
            # Get fields of memory structure corresponding to the keys
            # Number of rows should total number of elements over all keys,
            #    and columns should number of items in memory
            matrix =np.array([np.concatenate((self.memory_template[:,:self.num_keys][i]))
                              for i in range(self.memory_capacity)]).transpose()
            match_nodes = [
                TransferMechanism(
                    input_ports={NAME: 'CONCATENATED_INPUTS',
                                 SIZE: self.memory_capacity,
                                 PROJECTIONS: MappingProjection(sender=self.concatenate_keys_node,
                                                                matrix=matrix,
                                                                function=LinearMatrix(
                                                                    normalize=self.normalize_memories))},
                    name='MATCH')]

        # One node for each key
        else:
            match_nodes = [
                TransferMechanism(
                    input_ports= {
                        SIZE:self.memory_capacity,
                        PROJECTIONS: MappingProjection(sender=self.key_input_nodes[i].output_port,
                                                       matrix = np.array(self.memory_template[:,i].tolist()
                                                                         ).transpose().astype(float),
                                                       function=LinearMatrix(normalize=self.normalize_memories))},
                    name=f'MATCH {self.key_names[i]}')
                for i in range(self.num_keys)
            ]

        return match_nodes

    def _construct_softmax_control_nodes(self)->list:
        """Create nodes that set the softmax gain (inverse temperature) for each softmax_node."""

        softmax_control_nodes = []
        if self.softmax_gain == CONTROL:
            softmax_control_nodes = [ControlMechanism(monitor_for_control=match_node,
                                                      control_signals=[(GAIN, self.softmax_nodes[i])],
                                                      function=get_softmax_gain,
                                                      name='SOFTMAX GAIN CONTROL' if len(self.softmax_nodes) == 1
                                                      else f'SOFTMAX GAIN CONTROL {i}')
                                     for i, match_node in enumerate(self.match_nodes)]

        return softmax_control_nodes

    def _construct_softmax_nodes(self)->list:
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
        retrieval_gating_nodes = []
        if not self.concatenate_keys:
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

    # *****************************************************************************************************************
    # *********************************** Execution Methods  **********************************************************
    # *****************************************************************************************************************

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
