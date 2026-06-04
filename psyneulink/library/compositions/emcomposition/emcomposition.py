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
     - `Organization <EMComposition_Organization>`
     - `Operation <EMComposition_Operation>`
  * `EMComposition_Creation`
     - `Memory Structue Specification <EMComposition_Memory_Specification>`
        • `Memory Template <EMComposition_Memory_Template>`
        • `Memory Fill <EMComposition_Memory_Fill>`
        • `Memory Capacity <EMComposition_Memory_Capacity>`
     - `Field Characteristics <EMComposition_Fields>`
        • `Field Specification Dict <EMComposition_Field_Specification_Dict>`
        • `Field Names <EMComposition_Field_Names>`
        • `Field Weights <EMComposition_Field_Weights>`
        • `Normalize Field Weights <EMComposition_Normalize_Field_Weights>`
        • `Concatenate Queries <EMComposition_Concatenate_Queries>`
     - `Retrieval and Storage <EMComposition_Retrieval_Storage>`
        • `Storage Probability <EMComposition_Storage_Prob>`
        • `Softmax Gain <EMComposition_Softmax_Gain>`
        • `Softmax Threshold <EMComposition_Softmax_Threshold>`
        • `Softmax Choice <EMComposition_Softmax_Choice>`
        • `Memory Decay Rate <EMComposition_Memory_Decay_Rate>`
        • `Purge by Weight <EMComposition_Purge_by_Weight>`
     - `Learning <EMComposition_Learning_Creation>`
  * `EMComposition_Structure`
     - `Input <EMComposition_Input>`
     - `Memory <EMComposition_Memory_Structure>`
     - `Output <EMComposition_Output>`
  * `EMComposition_Execution`
     - `Processing <EMComposition_Processing>`
     - `Learning <EMComposition_Learning_Execution>`
  * `EMComposition_Examples`
     - `Example Case <EMComposition_Example_Case>`
  * `EMComposition_Class_Reference`

.. _EMComposition_Overview:

Overview
--------

EMComposition implements a configurable, content-addressable form of episodic (or external) memory. It is a subclass
of `AutodiffComposition`, which allows it to backpropagate error signals and learn how to differentially weight cues
(queries) used for retrieval. It uses `ExternalMemoryMechanism` to manage field-specific memory matrices.
EMComposition supports one or more memory fields, which can be configured as keys or values. Keys are used to match
queries against stored entries, while values are retrieved but not used for matching.

It also allows several other factors to be configured, including the function used for evaluating the match of
queries to keys ("scoring"), the gain of the `SoftMax` function used to normalize the scores for each memory field,
whether the gain is adapted to the number of entries in memory, and whether the `memory <EMComposition.memory>` is
decayed by a specified amount after each storage. In many of these respects, it implements a variant of a `Modern
Hopfield Network <https://en.wikipedia.org/wiki/Modern_Hopfield_network>`_ (in which storage of entries in the memory
matrix corresponds to fast learning of weights in the weights of a densely recurrent network), as well as important
features of a `Transformer <https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)>`_ (in which
the softmax-weight simiarly based retrieval process implements a form of attention).

.. technical_note::
   EMComposition emulates an `EpisodicMemoryMechanism` -- reproducing all of the functionality of its
   `ContentAddressableMemory` `Function` -- in the form of an `AutodiffComposition`, which allows it to
   backpropagate error signals based on retrieved values to it inputs, and learn how to differentially
   weight cues (queries) used for retrieval.  It also adds the capability for `memory_decay
   <EMComposition.memory_decay_rate>`.


.. _EMComposition_Organization:

**Organization**

.. _EMComposition_Fields_and_Entries:

*Fields and Entries*. The `memory <EMComposition.memory>` of an EMComposition is organized into `fields
<EMComposition_Fields>`), that are used to represent different pieces of information that are stored and
retrieved together as an entry in the EMComposition's memory. Each field receives its own input (`query
<EMComposition.query_input_nodes>` or `value <EMComposition.value_input_nodes>`), generates its own output
(`retrieved value <EMComposition.retrieved_value>`), and its `memory <ExternalMemory.memory>` is managed by
its own `ExternalMemoryMechanism`. The EMComposition coordinates the storage and retrieval of entries across
fields. The contribution of each field to retrieval can be weighted by asisgning `field_weights
<EMComposition.field_weights>`. The EMComposition's full memory can be accessed using its `memory
<EMComposition.memory>` attribute, which returns a 3d array, the rows of which (axis 0) are each entry,
the columns of which (axis 1) are the fields of each entry, and the items of which (axis 2) are the values
of the entries for each field. The EMComposition's fields are listed in its `fields <EMComposition.fields>`
attribute, and its The memory for a given field can be accessed via the `em.fields[*NAME*].memory
<ExternalMemoryMechanism.memory>` attribute.

**Operation**

When an EMComposition is executed, each key field computes the similarity of its input (query) to all of the
values for each entry in its memory, and the set of similarity "scores" are combined across all fields to
generate a set of "combined scores" for each entry in memory, which is then used by each field to generate the
retrieved value for that field. The input to each field is then stored in the memory for that field. By default,
this occurs by replacing the value in each field that belongs to the entry with the weakest value across all fields.
However, this behavior can be modified using the `memory_decay_rate <EMComposition.memory_decay_rate>` and
`scores_metric <EMComposition.scores_metric>` Parameters. The retrieved values for each field are returned as the
`result <Composition_Execution_Results_and_Reporting>` of the EMComposition's `execution <Composition_Execution>`,
and can be accessed using its `results <Composition.results>` or its `output_values <Composition.output_values>`
attributes. During learning, the EMComposition can adjust the `field_weights <EMComposition.field_weights>` to
optimize performance, and the error gradient can be passed through the composition to adjust the weights of the
fields that contribute to retrieval, as well as to any input nodes that project to the EMComposition.


.. _EMComposition_Creation:

Creation
--------

An EMComposition is created by calling its constructor. There are four major elements that can be configured:
the structure of its `memory <EMComposition_Memory_Specification>`; the characteristics of its
fields <EMComposition_Fields>`; how `storage and retrieval <EMComposition_Retrieval_Storage>` operate; and whether
and how `learning <EMComposition_Learning_Creation>` is carried out.

.. _EMComposition_Memory_Specification:

*Memory Structure Specification*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `memory <EMComposition.memory>` of an EMComposition is comprised of `fields and entries
<EMComposition_Fields_and_Entries>`. The number of fields, the "width" of each, and the total number of
entries are specified using three arguments of the EMComposition's constructor: **memory_template**, that
specifies the overall structure of its `memory <EMComposition.memory>` (the number of fields in each entry,
the width of each field, and the number of entries); **memory_fill**, that specifies initial values to assign
to each entry; and **memory_capacity**, that can be used to specify the total number of entries that can be
stored in the EMComposition's `memory <EMComposition.memory>`. These are described in detail below.

.. _EMComposition_Memory_Template:

* **memory_Template**: this specifies the shape of the entries to be stored in the EMComposition's `memory
  <EMComposition.memory>`, and can be used to initialize it with pre-specified entries. The **memory_template**
  argument can be specified in one of three ways (see `EMComposition_Examples` for representative use cases):

  * **tuple**: interpreted as an np.array shape specification, that must be of length 2 or 3. If it is a 3-item tuple,
    then the first item specifies the number of entries in memory, the 2nd the number of fields in each entry, and the
    3rd the length of each field (note that in this case, all the fields will be the same leingth).  If it is a 2-item
    tuple, this specifies the shape of an entry, and the number of entries is specified by `memory_capacity
    <EMComposition_Memory_Capacity>`).  All entries are filled with zeros or the value specified by `memory_fill
    <EMComposition_Memory_Fill>`.

    .. warning::
       If **memory_template** is specified with a 3-item tuple and `memory_capacity <EMComposition_Memory_Capacity>`
       is also specified with a value that does not match the first item of **memory_template**, an error is
       generated indicating the conflict in the number of entries specified.

    .. hint::
       To specify a single field, a list or array must be used (see below), as a 2-item tuple is interpreted as
       specifying the shape of an entry, and so it can't be used to specify the number of entries each of which
       has a single field.

  * **2d list or array**: interpreted as a template for a memory entry. This can be used to specify fields of
    different lengths (i.e., entries that are ragged arrays), with each item in the list (axis 0 of the array) used
    to specify the length of the corresponding field. The template is then used to initialze all entries in `memory
    <EMComposition.memory>`. If the template includes any non-zero elements, then the array is replicated for all
    entries in `memory <EMComposition.memory>`; otherwise, they are filled with either zeros or the value specified
    in `memory_fill <EMComposition_Memory_Fill>`.

    .. hint::
       To specify a single entry, with all other entries filled with zeros
       or the value specified in **memory_fill**, use a 3d array as described below.

  * **3d list or array**: used to initialize `memory <EMComposition.memory>` directly with the entries specified in
    the outer dimension (axis 0) of the list or array.  If `memory_capacity <EMComposition_Memory_Capacity>` is not
    specified, then it is set to the number of entries in the list or array. If **memory_capacity** *is* specified,
    then the number of entries specified in **memory_template** must be less than or equal to **memory_capacity**.
    if is less than **memory_capacity**, then the remaining entries in `memory <EMComposition.memory>` are filled
    with zeros or the value specified in **memory_fill** (see below); if all of the entries specified contain only
    zeros, and **memory_fill** is specified, then the matrix is filled with the value specified in **memory_fill**;
    otherwise, zeros are used to fill all entries.

.. _EMComposition_Memory_Fill:

* **memory_fill**: specifies the value used to fill the `memory <EMComposition.memory>`, based on the shape specified
  in the **memory_template** (see above). The value can be a scalar, or a tuple to specify an interval over which
  to draw uniformly distributed random values to fill `memory <EMComposition_Proj.memory>` --- both should be scalars,
  with the first specifying the lower bound and the second the upper bound.  If **memory_fill** is not specified,
  and no entries are specified in **memory_template**, then `memory <EMComposition_Proj.memory>` is filled with zeros.

  .. hint::
     If memory is initialized with all zeros and **normalize_memories** set to ``True`` (see `below
     <EMComposition_Retrieval_Storage>`) then a numpy.linalg warning is issued about divide by zero.
     This can be ignored, as it does not affect the results of execution, but it can be averted by specifying
     `memory_fill <EMComposition_Memory_Fill>` to use small random values (e.g., ``memory_fill=(0,.001)``).

.. _EMComposition_Memory_Capacity:

* **memory_capacity**: specifies the number of entries that can be stored in the EMComposition's memory; when
  `memory_capacity <EMComposition.memory_capacity>` is reached, each new entry overwrites an existing one. By
  default, this is the weakest entry (i.e., the one with the smallest norm across all of its fields); however
  different behaviors can be configured (see `EMComposition_Memory_Decay_Rate`). If `memory_template
  <EMComposition_Memory_Template>` is specified as a 3-item tuple or 3d list or array (see above), then that is
  used to determine `memory_capacity <EMComposition.memory_capacity>` (if it is specified and conflicts with either
  of those an error is generated). Otherwise, it can be specified using a numerical value, with a default of 1000.

  .. warning::
     The `memory_capacity <EMComposition.memory_capacity>` of an EMComposition cannot be modified after construction.


.. _EMComposition_Fields:

*Field Characteristics*
~~~~~~~~~~~~~~~~~~~~~~~

While the *number* of fields is specified using the arguments described `above <EMComposition_Memory_Specification>`,
the *characteristics* of each field (e.g., whether it is treated as key or value field, how key fields are weighted
for retrieval, and how they are invovled in learning) are specified using either the **fields** argument of the
EMComposition's constructor, or a combination of the **field_names**, **field_weights**, **learn_field_weights** and
**target_fields** arguments, as described below.

.. _EMComposition_Field_Specification_Dict:

* **fields**: a dict that specifies the names of the fields and their attributes. There must be one entry for each
  field specified in the **memory_template**, formatted as {<str: dict or tuple>}, where:

  * *str*: a string that specifies the name of the field.

  * *dict or tuple*: either of which must have three entries; if a dict, the key to each entry must be the keyword
    specified below or, if a tuple, the entries must appear in the following order:

    - *FIELD_WEIGHT* `specification <EMComposition_Field_Weights>` - value must be a scalar or ``None``.
      If it is a scalar, the field is treated as a `retrieval key <EMComposition_Field_Weights>` in `memory
      <EMComposition.memory>` that is weighted by that value during retrieval; if ``None``, it is treated as a
      value in `memory <EMComposition.memory>` and the field cannot be reconfigured later.

    - *LEARN_FIELD_WEIGHT* `specification <EMComposition_Field_Weights_Learning>` - value must be a boolean or a float;
      if ``False``, the field_weight for that field is not learned; if ``True``, the field weight is learned using the
      EMComposition's `learning_rate <EMComposition.learning_rate>`; if a float, that is used as its learning_rate.

    - *TARGET_FIELD* `specification <EMComposition_Target_Fields>` - value must be a boolean; if ``True``,
      the value of the `retrieved_node <EMComposition.retrieved_nodes>` for that field conrtributes to the
      error computed during learning and backpropagated through the EMComposition (see `Backpropagation of
      <EMComposition_Error_BackPropagation>`); if ``False``, the retrieved value for that field does not
      contribute to the error; however, its field_weight can still be learned if that is specfified in
      `learn_field_weight <EMComposition_Field_Weights_Learning>`.

  .. _note:
     The **fields** argument is provided as a simple and convenient way of specifying field attributes;
     the dict itself is not retained as a `Parameter` or attribute of the EMComposition.

  The specifications provided in the **fields** argument are assigned to the corresponding Parameters of
  the EMComposition which, alternatively, can be specified directly using the **field_names**, **field_weights**,
  **learn_field_weights** and **target_fields** arguments of the EMComposition's constructor, as described below.

  .. warning::
     A **fields** dict cannot be specified together with any of the **field_names**, **field_weights**,
     **learn_field_weights** and/or **target_fields** arguments; if that occurs, a warning is issued,
     the assignments specified in the **fields** dict are used, and any specifications made in the
     **field_names**, **field_weights**, **learn_field_weights** and/or **target_fields** arguments are ignored.

.. _EMComposition_Field_Names:

* **field_names**: a list that specifies names to be assigned to the fields; the number of names specified must match
  the number of fields specified in the **memory_template**. If specified, the names are used to label the nodes of the
  EMComposition; otherwise, the fields are labeled generically as "Key 0", "Key 1", and "Value 1", "Value 2", etc..

.. _EMComposition_Field_Weights:

* **field_weights**: specifies which fields are used as keys vs. value fields, and how key fields are weighted during
  retrieval. Fields designated as keys are used to match inputs (queries) against entries in memory for retrieval (see
  `Match memories by field <EMComposition_Processing>`); entries designated as *values* are ignored during the matching
  process, but their stored values are retrieved and assigned as the `value <Mechanism_Base.value>` of the corresponding
  `retrieved_node <EMComposition.retrieved_nodes>`. This distinction between keys and value corresponds to the format
  of a standard "dictionary," though in that case only a single key and value are allowed, whereas in an EMComposition
  there can be one or more keys and any number of values; if all fields are keys, this implements a full form of
  content-addressable memory. The following options can be used to specify **field_weights**:

    * *None* (the default): all fields except the last are treated as keys, and are assigned a weight of 1,
      while the last field is treated as a value field (same as assiging it ``None`` in a list or tuple (see below).

    * *scalar*: all fields are treated as keys (i.e., used for retrieval) and weighted equally for retrieval. If
      `normalize_field_weights <EMComposition_Normalize_Field_Weights>` is ``True``, the value is divided by the number
      of keys, whereas if `normalize_field_weights <EMComposition_Normalize_Field_Weights>` is ``False``, then the value
      specified is used to weight the retrieval of all keys with that value.

      .. note::
         At present these have the same result, since the `SoftMax` function is used to normalize the match
         scores. However, other retrieval functions may be added in the future that would be affected by
         the value of the `field_weights <EMComposition.field_weights>`. Therefore, it is recommended to leave
         `normalize_field_weights <EMComposition_Normalize_Field_Weights>` set to ``True`` (the default) to ensure that
         the `field_weights <EMComposition.field_weights>` are normalized to sum to 1.0.

    * *list or tuple*: the number of entries must match the number of fields specified in **memory_template**, and all
      entries must be either 0, a positive scalar value, ``None``, or ``False``. If all entries are identical, they
      are treated as if a single value was specified (see above); if the entries are non-identical, any entries that
      are not ``None`` or ``False`` are used to weight the corresponding fields during retrieval (see `Weight fields
      <EMComposition_Processing>`), including those that are 0 (though these will not be used in the retrieval
      process unless/until they are changed to a positive value). If `normalize_field_weights
      <EMComposition_Normalize_Field_Weights>` is ``True``, all non-None/non-False field_weight entries are normalized
      so that they sum to 1.0; if `normalize_field_weights <EMComposition_Normalize_Field_Weights>` is ``False``, the
      raw values are used to weight the retrieval of the corresponding fields. All entries of ``None`` or ``False`` are
      treated as value fields, are not assigned a `field_weight_node <EMComposition.field_weight_nodes>`, and are
      ignored during retrieval. These *cannot be modified after the EMComposition has been constructed (see note below).

    .. _EMComposition_No_Field_Weights_For_Single_Key_Note:

    .. note::
       If there is only a single key field, no field_weight is constructed, as in this case weighting would have
       no effect; this also means that **learn_field_weights** has no effect, and a warning is issued if specified.

    .. _EMComposition_Field_Weights_Change_Note:

    .. note::
       The field_weights can be modified after the EMComposition has been constructed, by assigning a new set of weights
       to the `field_weights <EMComposition.field_weights>` `Parameter`.  However, only field_weights associated with
       key fields (i.e., that were initially assigned non-None or non-False field_weights) can be modified; the weights
       for value fields (i.e., ones that were initially assigned a field_weight of ``None`` or ``False``) cannot be
       modified, and doing so raises an error. If a field that will be used initially as a value but may later need
       to be used as a key, it should be assigned a `field_weight <EMComposition.field_weights>` of ``0`` at
       construction (rather than ``None`` or ``False``), which can then later be changed as needed.

    .. technical_note::
       The reason that field_weights can be modified only for keys is that `field_weight_nodes
       <EMComposition.field_weight_nodes>` are constructed only for keys, since ones for values would have no effect
       on the retrieval process and therefore are uncecessary (and can be misleading).

* **learn_field_weights**:  if **enable_learning** is ``True``, this specifies which field_weights are subject to
  learning and optionally the `learning_rate <EMComposition.learning_rate>` for each (see `learn_field_weights
  <EMComposition_Field_Weights_Learning>` below for details of specification); however, this has no effect if there
  is only a single key (see `note <EMComposition_No_Field_Weights_For_Single_Key_Note>` above), and a warning is issued
  if it is specified.

.. _EMComposition_Normalize_Field_Weights:

* **normalize_field_weights**: specifies whether the `field_weights <EMComposition.field_weights>` are normalized or
  their raw values are used. If ``True``, the value of all non-``None`` and non-``False`` `field_weights
  <EMComposition.field_weights>` are normalized so that they sum to 1.0, and the normalized values are used to weight
  (i.e., multiply) the corresponding fields during retrieval (see `Weight fields <EMComposition_Processing>`). If
  `normalize_field_weights <EMComposition_Processing.normalize_field_weights>` is ``False``, the raw values of the
  `field_weights <EMComposition.field_weights>` are used to weight the retrieved value of each field. This setting
  is ignored if **field_weights** is ``None`` or `concatenate_queries <EMComposition_Concatenate_Queries>` is ``True``.


.. _EMComposition_Concatenate_Queries:

COMMENT:
BREADCRUMB: EXPLAIN HERE WHAT HAPPENS RE: field_memory_nodes
COMMENT
* **concatenate_queries**: specifies whether queries are concatenated before they are matched to keys in memory.
  This is ``False`` by default. Setting it to ``True`` causes all the fields to be treated, in effect, as a single
  field for scoring, retrieval and norming, while preserving the ability to provide separate inputs for each field,
  inspect them individually, and keep their retrieved values separate (in the EMComposition's `retrieved_nodes
  <EMComposition.retrieved_nodes>`). This requires that all non-zero `field_weights <EMComposition.field_weights>`
  have the same value (see `field_weights <EMComposition_Field_Weights>`) and that `normalize_memories
  <EMComposition.normalize_memories>` is set to ``True``; otherwise, setting concatenate_queries to ``True`` issues
  a warning, and the setting is ignored. Setting **concatenate_queries** causes a `concatenate_queries_node
  <EMComposition.concatenate_queries_node>` to be created that receives input from all of the `query_input_nodes
  <EMComposition.query_input_nodes>`, and passes them as a single vector to the `combined_scores_node
  <EMComposition.combined_scores_node>`.

      .. note::
         While this is computationally more efficient, it can affect the outcome of the `scoring process
         <EMComposition_Processing>`, since computing the similarity of a single vector comprised of the concatenated
         inputs is not identical to computing the similairty of each field independently and then combining the results.

      .. note::
         All `query_input_nodes <EMComposition.query_input_nodes>` and `retrieved_nodes <EMComposition.retrieved_nodes>`
         are always preserved, even when `concatenate_queries <EMComposition.concatenate_queries>` is ``True``, so that
         separate inputs can be provided for each key, and the value of each key can be retrieved separately.

.. _EMComposition_Retrieval_Storage:

*Retrieval and Storage*
~~~~~~~~~~~~~~~~~~~~~~~

The following arguments can be used to configure how retrieval and storage operate:

* **normalize_memories**: specifies whether query inputs and keys in memory are normalized before computing their
similarity.  If this is set to ``True`` and `scores_metric <EMComposition.scores_metric>` is set to *DOT_PRODUCT*,
then the scores correspond to the cosine similarity of the queries and keys.
COMMENT:
BREADCRUMB FROM AI: ADD WHEN THESE METRICS ARE SUPPORTED
If `scores_metric <EMComposition.scores_metric>` is set to *EUCLIDEAN* or *MANHATTAN*, then the scores correspond
to the negative of the euclidean or manhattan distance, respectively
COMMENT

.. _EMComposition_Storage_Prob:

* **storage_prob**: specifies the probability that the inputs to the EMComposition (queries and values) will be stored
  as an entry in `memory <EMComposition.memory>` on each execution.


.. _EMComposition_Softmax_Choice:

* **softmax_choice**: specifies how the `SoftMax` Function of the EMComposition's `combined_scores_node
  <EMComposition.combined_scores_node>` to weight entries for retrieval based on the combined_scores, which is
  the sum of the match scores between queries and keys generated by each field (see `EMComposition_Combined_Scores`);
  the following are the options that can be used, and the type of retrieved value they produce:

  * *WEIGHTED_AVG* (default): softmax-normalized transform of combined_scores, that is used to produce a
    softmax-weighted average over entries as the retrieved value;

  * *ARG_MAX*: entry with the highest score (one with lowest index in `memory <EMComposition.memory>`)\
               if there are identical ones);

  * *PROBABISTIC*: probabilistically chosen entry based on the softmax-normalized combined scores.

  .. warning::
     Use of the *ARG_MAX* and *PROBABILISTIC* options are not compatible with learning, as these implement a discrete
     choice and thus are not differentiable. Constructing an EMComposition with **softmax_choice** set to either of
     these options and **learn_field_weights** set to ``True` (or a list with any ``True`` entries) will generate a
     warning and calling the EMComposition's `learn() <Composition.learn>` method will generate an error; it must be
     changed to *WEIGHTED_AVG* to execute learning.

  .. technical_note::
     The *WEIGHTED_AVG* option is passed as *ALL* to the **output** argument of the `SoftMax` Function, *ARG_MAX* is
     passed as *ARG_MAX_INDICATOR*; and *PROBALISTIC* is passed as *PROB_INDICATOR*. This mapping is honored for both
     Python execution and the PyTorch execution path (e.g., ``execution_mode=ExecutionMode.PyTorch``); other SoftMax
     output types are not used by EMComposition.

.. _EMComposition_Softmax_Gain:

* **softmax_gain**: specifies the gain (inverse temperature) used for softmax normalizing the combined distances
  used for retrieval when **softamx_choice is *WEIGHTED_AVG* (see `EMComposition_Execution` below); otherwise it is
  ignored.  The following options can be used:

  * numeric value: the value is used as the gain of the `SoftMax` Function for the EMComposition's
    `combined_scores_node <EMComposition.combined_scores_node>`;

  * *ADAPTIVE*: the `adapt_gain <SoftMax.adapt_gain>` method of the `SoftMax` Function is used to adaptively set
    the `softmax_gain <EMComposition.softmax_gain>` based on the entropy of the match scores, in order to preserve
    the distribution over non- (or near) zero entries irrespective of how many (near) zero entries there are
    (see `Thresholding and Adaptive Gain <SoftMax_AdaptGain>` for additional details);

  * *CONTROL*: a `ControlMechanism` is created, and its `ControlSignal` is used to modulate the `softmax_gain
    <EMComposition.softmax_gain>` parameter of the `SoftMax` function of the EMComposition's `combined_scores_node
    <EMComposition.combined_scores_node>`.

  If ``None`` is specified, the default value for the `SoftMax` function is used.

.. _EMComposition_Softmax_Threshold:

* **softmax_threshold**: if this is specified, and **softmax_gain** is specified with a numeric value,
  then any values below the specified threshold are set to 0 before the combined scores are softmaxed
  (see *mask_threhold* under `Thresholding and Adaptive Gain <SoftMax_AdaptGain>` for additional details).

.. _EMComposition_Memory_Decay_Rate:

COMMENT:
BREADCRUMB ADD **forgee** HERE
COMMENT

* **memory_decay_rate**: specifies the rate at which items in the EMComposition's memory decay each time a new entry
  is stored; the default rate is *AUTO*, which sets it to  1 / `memory_capacity <EMComposition_Proj.memory_capacity>`,
  such that the oldest memories are the weakest and
  COMMENT:
  , if `forget <EMComposition.forget>` is set to *WEAKEST*,
  COMMENT
  the most likely to be replaced. If **memory_decay_rate** is set to 0 or ``False``,
  then memories do not decay
  COMMENT:
  REMOVE THE FOLLOWING WHEN **forget** IS IMPLEMENTED
  COMMENT
  and, when `memory_capacity <EMComposition_Proj.memory_capacity>` is reached, the weakest
  memories (ones with the lowest norm) are replaced, irrespective of order of entry.

  .. hint::
     If `normalize_memories <EMComposition.normalize_memories>` is ``True`` and `scores_metric
     <EMComposition.scores_metric>` is *DOT_PRODUCT*, or `scores_metric <EMComposition.scores_metric>` is *COSINE*,
     then memory_decay_rate has no effect, since all memories have the same norm.
     COMMENT: BREADCRUMB TBD
     In that case, a warning is issued, and memories are randomly replace ?? replaced with incrementing index?
     BREADCRUMB: MENTION purge_by_field_weight PER BELOW AND/OR REVISE THE ABOVE WHEN **forget** IS IMPLEMENTED
     COMMENT

COMMENT:
BREADCRUMB: REMOVE OR REVISE WHEN **forget** IS IMPLEMENTED
COMMENT
.. _EMComposition_Purge_by_Weight:

* **purge_by_field_weight**: specifies whether `field_weights <EMComposition.field_weights>` are used in determining
  which memory entry is replaced when a new memory is `stored <EMComposition_Storage>`.  If ``True``, the norm of each
  entry is multiplied by its `field_weight <EMComposition_Field_Weighting>` to determine which entry is the weakest and
  will be replaced.

.. _EMComposition_Learning_Creation:

*Learning*
~~~~~~~~~~

EMComposition supports two functions of learning: error backpropagation through the entire Composition, and
the learning of `field_weights <EMComposition.field_weights>` within it. Learning is enabled by setting the
**enable_learning** argument of the EMComposition's constructor to ``True``, and optionally specifying the
**learn_field_weights** argument (as detailed below). If **enable_learning** is ``False``, no learning of any
kind occurs, no error can be backpropagated through the EMComposition, and so EMComposition cannot be included
in a `learning pathway <AutodiffComposition_Learning_Pathways>`; if it is ``True``, then both forms of learning
are enabled.

.. _EMComposition_Error_BackPropagation

*Backpropagation of error*.  If **enable_learning** is ``True``, then the values retrieved from `memory
<EMComposition.memory>` when the EMComposition is executed during learning can be used for error computation
and backpropagation through the EMComposition to its inputs.
COMMENT:
BREADCRUMB: NEEDS TO BE DEBUGGED:
By default, the values of all of its `retrieved_nodes
<EMComposition.retrieved_nodes>` are included. For those that do not project to an outer Composition (i.e., one in
which the EMComposition is `nested <Composition_Nested>`), and sample-target pairs are not explicitly specified in
the constructor for the EMComposition or any within which it is nested (see `AutodiffComposition_Configuring_Learning`),
then a `TARGET <NodeRole.TARGET_INPUT>` node is constructed for each of the `retrieved_nodes.
<EMComposition.retrieved_nodes>`.  Those are then used to compute errors that are backpropagated through the
EMComposition to its `query_input_nodes <EMComposition.query_input_nodes>` and `value_input_nodes
<EMComposition.value_input_nodes>`, and on to any nodes that project to those from a Composition within which the
EMComposition is `nested <Composition_Nested>`. Retrieved_nodes that *do* project to an outer Composition receive
their errors from those nodes, which are also backpropagated through the EMComposition.
COMMENT
Fields can be selectively
specified for learning in the **fields** argument or the **target_fields** argument of the EMComposition's
constructor, as detailed below.

*Field Weight Learning*.  If **enable_learning** is ``True``, then the `field_weights <EMComposition.field_weights>`
can be learned, by specifing these either in the **fields** argument or the **learn_field_weights** argument of
the EMComposition's constructor, as detailed below.

.. note::
   Learning field_weights implements a function comparable to the learning in an attention head of the `Transformer
   <https://arxiv.org/abs/1706.03762>`_ architecture, although at present the field can only be scalar values rather
   than vectors or matrices, and it cannot receive input. These capabilities may be added in the future.

The following arguments of the EMComposition's constructor can be used to configure learning:

* **enable_learning**: specifies whether any learning is enabled for the EMComposition.  If ``False``,
  no learning occurs; if ``True``, then both error backpropagation and learning of `field_weights
  <EMComposition.field_weights>` can occur. If **enable_learning** is ``True``, **use_gating_for_weighting**
  must be ``False`` (see `note <EMComposition_Gating_For_Weighting>`).

.. _EMComposition_Target_Fields:

COMMENT:
BREADCRUMB: HOW DOES THIS INTERACT WITH **targets** ARGUMENT OF AUTODIFF CONSTRUCTOR AND/OR TARGET NODE CONSTRUCTION?
COMMENT
* **target_fields**: specifies which `retrieved_nodes <EMComposition.retrieved_nodes>` are used to compute
  errors, and propagate these back through the EMComposition to its `query <EMComposition.query_input_nodes>` and
  `value_input_nodes <EMComposition.value_input_nodes>`. If this is ``None`` (the default), all `retrieved_nodes
  <EMComposition.retrieved_nodes>` are used; if it is a list or tuple, then it must have the same number of items
  as there are fields, and each item must be a boolean specifying whether the corresponding `retrieved_nodes
  <EMComposition.retrieved_nodes>` participate in learning, and errors are computed only for those nodes. This can
  also be specified in a dict for the **fields** argument (see `fields <EMComposition_Field_Specification_Dict>`).

.. _EMComposition_Field_Weights_Learning:

* **learn_field_weights**: specifies which field_weights are subject to learning, and optionally the `learning_rate
  <EMComposition.learning_rate>` for each; this can also be specified in a dict for the **fields** argument (see
  `fields <EMComposition_Field_Specification_Dict>`). The following specfications can be used:

  * *None*: all field_weights are subject to learning, and the `learning_rate <EMComposition.learning_rate>` for the
    EMComposition is used as the learning_rate for all field_weights.

  * *bool*: If ``True``, all field_weights are subject to learning, and the `learning_rate
    <EMComposition.learning_rate>` for the EMComposition is used as the learning rate for all
    field_weights; if ``False``, no field_weights are subject to learning, regardless of `enable_learning
    <EMComposition.enable_learning>`.

  * *list* or *tuple*: must be the same length as the number of fields specified in the memory_template, and each entry
    must be either ``True``, ``False`` or a positive scalar value.  If ``True``, the corresponding field_weight is subject
    to learning and the `learning_rate <EMComposition.learning_rate>` for the EMComposition is used to specify the
    learning_ rate for that field; if ``False``, the corresponding field_weight is not subject to learning; if a scalar
    value is specified, it is used as the `learning_rate` for that field.

* **learning_rate**: specifies the learning_rate for any `field_weights <EMComposition.field_weights>` for which a
  learning_rate is not individually specified in the **learn_field_weights** argument (see above).

.. _EMComposition_Structure:

Structure
---------

.. _EMComposition_Input:

*Input*
~~~~~~~

The inputs corresponding to each query and value field are assigned as `INPUT <NodeRole.INPUT>` `Nodes
<Composition_Nodes>` of the EMComposition, and listed in its `query_input_nodes <EMComposition.query_input_nodes>`
and `value_input_nodes <EMComposition.value_input_nodes>` attributes, respectively.

.. _EMComposition_Memory_Structure:

*Memory*
~~~~~~~~

The EMComposition's `memory <EMComposition.memory>` is stored in the `memory <ExternalMemory.memory>` `Parameter`
of the `ExternalMechansim` for each `field <EMComposition_Fields>`. Each field is treated as a separate "channel"
for storage and retrieval, and is associated with its own corresponding input (query or value) and output (retrieved
value) `Node <Composition_Nodes>`. These are coordinated for retrieval and storage by the `combined_scores_node
<EMComposition.combined_scores_node>`. The structure of the memory is configured as descrbied in
`EMComposition_Memory_Specification`. The full contents of the EMComposition's memory can be accessed from its
`memory <EMComposition.memory>` Parameter, which aggregates and formats the `memory <ExternalMemory.memory>` of each
field into a single 3d array, in which rows (axis 0) are entries, columns (axis 1) are fields, and items (axis 2) are
the values for each field of a given entry.  Note that, since fields can have different "widths" (i.e., their values
can different in length), the full memory array can be "ragged." Information about each field can be accessed in the
`fields <EMComposition.fields>` attribute, which is a list of `Field` objects containing information about the nodes
adn values associated with each field.

.. _EMComposition_Output:

*Output*
~~~~~~~~

The `retrieved_nodes <EMComposition.retrieved_nodes>` of the EMComposition are assigned as its `OUTPUT
<NodeRole.OUTPUT>` `Nodes <Composition_Nodes>`, and their `values <OutputPort.value>` as the `output_values
<Composition.output_values>` of the EMComposition as well as to its `results <Composition.results>` attribute
when it executes.

.. _EMComposition_Execution:

Execution
---------

COMMENT:

BREADCRUMB: INTEGRATE WITH BELOW:

The inputs to an EMComposition, comprised of
its queriess and values, are assigned to each of its `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>`:
queries (i.e., that are matched to its keys) are assigned to its `query_input_nodes  <EMComposition.query_input_nodes>`;
and the remaining inputs are assigned to it `value_input_nodes <EMComposition.value_input_nodes>`. When the
EMComposition is executed, the retrieved values for all fields are returned as the result, and recorded in its `results
<Composition.result>` attribute. The value for each field is assigned as the `value <OutputPort.value>` of its
`OUTPUT <NodeRole.OUTPUT>` `Nodes <Composition_Nodes>`. The input is then stored in its `memory <EMComposition.memory>`,
with a probability determined by its `storage_prob <EMComposition.storage_prob>` `Parameter`, and all previous memories
decayed by its `memory_decay_rate <EMComposition.memory_decay_rate>`. The `memory <EMComposition.memory>` can be
accessed using its `memory <EMComposition.memory>` Parameter.

  .. technical_note::
     The memories are actually stored in the `matrix <MappingProjection.matrix>` parameters of the`MappingProjections`
     from the `combined_matches_node <EMComposition.combined_matches_node>` to each of the `retrieved_nodes
     <EMComposition.retrieved_nodes>`. Memories associated with each key are also stored (in inverted form) in the
     `matrix <MappingProjection.matrix>` parameters of the `MappingProjection <MappingProjection>` from the
     `query_input_nodes <EMComposition.query_input_nodes>` to each of the corresponding `match_nodes
     <EMComposition.match_nodes>`. This is done so that the match of each query to the keys in memory for the
     corresponding field can be computed simply by passing the input for each query through the Projection (which
     computes the distance of the input with the Projection's `matrix <MappingProjection.matrix>` parameter) to the
     corresponding match_node; and, similarly, retrieivals can be computed by passing the softmax distributions for
     each field computed in the `combined_matches_node <EMComposition.combined_matches_node>` through its Projection
     to each `retrieved_node <EMComposition.retrieved_nodes>` (which are inverted versions of the matrices of the
     `MappingProjections <MappingProjection>` from the `query_input_nodes <EMComposition.query_input_nodes>` to each
     of the corresponding `match_nodes <EMComposition.match_nodes>`), to compute the distance of the weighted
     softmax over entries with the corresponding field of each entry that yields the retreieved value for each field.
------------------

**Operations**

When an EMComposition is executed, the following operations are carried out:

*Scoring.* The query for each field is compared against the keys in the memory for that field, and a match score is
generated based on the similarity of the query to the keys for that field. By default, for queries and keys that are
vectors, normalized dot products (comparable to cosine similarity) are used to compute the similarity of each query
to each key in memory; and if they are scalars the L0 norm is used.
XXXCOMMENT: EM2 BREADCRUMB -- ADD WHEN IMPLEMENTED
Optionally, COSINE SIMILARITY...
XXXCOMMENT
The scores are assigned as the `value <OutputPort.value>` of the *SCORES* OutputPort for each `field_memory_node
<EMCompositon.field_memory_nodes>`, which is passed to the `combined_scores_node <EMComposition.combined_scores_node>`
to generated a *COMBINED_SCORES* that is used for retrieval (see below).

*Retrieval.*  The values retrieved from `memory <ContentAddressableMemory.memory>` (one for each field) are based
on the relative similarity of the keys to the entries in memory, computed as the distance of each key and the
values in the corresponding field for each entry in memory. By default, for queries and keys that are vectors,
normalized dot products (comparable to cosine similarity) are used to compute the similarity of each query to each
key in memory; and if they are scalars the L0 norm is used.  These distances are then weighted by the corresponding
`field_weights <EMComposition.field_weights>` for each field (if specified) and then summed, and the sum is softmaxed
to produce a softmax distribution over the entries in memory. That is then used to generate a softmax-weighted average
of the retrieved values across all fields, which is returned as the `result <Composition.result>` of the EMComposition's
`execution <Composition_Execution>` (an EMComposition can also be configured to return the exact entry with the lowest
distance (weighted by field), however then it is not compatible with learning; see `softmax_choice
<EMComposition_Softmax_Choice>`).

  XXXCOMMENT:
  TBD DISTANCE ATTRIBUTES:
  The distance used for the last retrieval is stored in XXXX, and the distances of each of their corresponding fields
  (weighted by `distance_field_weights <ContentAddressableMemory.distance_field_weights>`), are returned in XXX,
  respectively.
  XXXCOMMENT

*Storage.*  The `inputs <Composition_Input_External_InputPorts>` to the EMComposition's fields are stored
in `memory <EMComposition.memory>` after each execution, with a probability determined by `storage_prob
<EMComposition.storage_prob>`.  If `memory_decay_rate <EMComposition.memory_decay_rate>` is specified, then
the `memory <EMComposition.memory>` is decayed by that amount after each execution.  If `memory_capacity
<EMComposition.memory_capacity>` has been reached, then each new memory replaces the weakest entry
(i.e., the one with the smallest norm across all of its fields) in `memory <EMComposition.memory>`.
COMMENT

The arguments of the `run() <Composition.run>` , `learn() <Composition.learn>` and `execute() <Composition.execute>`
methods of an EMComposition are the same as those of a `Composition`, and they can be passed any of the arguments
valid for an `AutodiffComposition`. The details of how the EMComposition executes are described below.

.. _EMComposition_Processing:

*Processing*
~~~~~~~~~~~~

When the EMComposition is executed, the following sequence of operations occur
COMMENT:
BREADCRUMB: CHECK FIGURE
COMMENT
(also see `figure <EMComposition_Example_Fig>`):

* **Input**.  The inputs to the EMComposition are provided to the `query_input_nodes <EMComposition.query_input_nodes>`
  and `value_input_nodes <EMComposition.value_input_nodes>`. The former are used to compute the match scores
  for each `key field <EMComposition_Fields_and_Entries>`, while the latter are stored but not used for matching.

* **Compute similarity scores**. By default, the input to every `query_input_node <EMComposition.query_input_nodes>`
  is passed to the corresponding `field_memory_node <EMComposition.field_memory_nodes>`, that scores the similarity
  of the query to each key (i.e., the value of each entry) in memory for that field). The similarity scores for each
  field are assigned as the `value <OutputPort.value>` of the *SCORES* OutputPort for each `field_memory_node
  <EMComposition.field_memory_nodes>`, which is passed to the `combined_scores_node
  <EMComposition.combined_scores_node>` to generated a *COMBINED_SCORES* that is used for retrieval (see below).

  However, if `concatenate_queries <EMComposition.concatenate_queries>` is ``True``, the inputs and scoring are handled
  differently:  In this case, the inputs to all of the `query_input_nodes <EMComposition.query_input_nodes>` are
  concatenated into a single vector in the `concatenate_queries_node <EMComposition.concatenate_queries_node>`,
  which is passed to the `concatenated_memory_node` that is used to compute the similarity of the concatenated query
  against a concatenated version of the keys in memory for each field.



  XXX

--------------
  .  In this way, each
  match is normalized so that, absent `field_weighting <EMComposition_Field_Weights>`, all keys contribute equally to
  retrieval irrespective of relative differences in the norms of the queries or the keys in memory. However, if the
  `field_weights <EMComposition.field_weights>` are the same for all `keys <EMComposition_Field_Weights>` and
  `normalize_memories <EMComposition.normalize_memories>` is True, then the inputs provided to the `query_input_nodes
  <EMComposition.query_input_nodes>` are concatenated into a single vector (in the
  `concatenate_queries_node <EMComposition.concatenate_queries_node>`), which is passed to a single `match_node
  <EMComposition.match_nodes>`.  This may be more computationally efficient than passing each query through its own
  `match_node <EMComposition.match_nodes>`,
  COMMENT:
  FROM CodePilot: (OF HISTORICAL INTEREST?)
  and may also be more effective if the keys are highly correlated (e.g., if they are different representations of
  the same stimulus).
  COMMENT
  however it will not necessarily produce the same results as passing each query through its own `match_node
  <EMComposition.match_nodes>` (see `concatenate keys <`concatenate_queries_node>` for additional information).

.. _EMComposition_Distance_Computation:

* **Match memories by field**. The values of each `query_input_node <EMComposition.query_input_nodes>`
  (or the `concatenate_queries_node <EMComposition.concatenate_queries_node>` if `concatenate_queries
  <EMComposition_Concatenate_Queries>` attribute is True) are passed through a `MappingProjection` that
  computes the distance between the corresponding input (query) and each memory (key) for the corresponding field,
  the result of which is possed to the corresponding `match_node <EMComposition.match_nodes>`. By default, the distance
  is computed as the normalized dot product (i.e., between the normalized query vector and the normalized key for the
  corresponding `field <EMComposition_Fields_and_Entries>`, that is comparable to using cosine similarity). However,
  if `normalize_memories <EMComposition.normalize_memories>` is set to ``False``, just the raw dot product is computed.
  The distance can also be customized by specifying a different `function <MappingProjection.function>` for the
  `MappingProjection` to the `match_node <EMComposition.match_nodes>`. The result is assigned as the `value
  <Mechanism_Base.value>` of the corresponding `match_node <EMComposition.match_nodes>`.

.. _EMComposition_Field_Weighting:

* **Weight distances**. If `field weights <EMComposition_Field_Weights>` are specified, then the distance computed
  by the `MappingProjection` to each `match_node <EMComposition.match_nodes>` is multiplied by the corresponding
  `field_weight <EMComposition.field_weights>` using the `field_weight_node <EMComposition.field_weight_nodes>`.
  By default (if `use_gating_for_weighting <EMComposition.use_gating_for_weighting>` is ``False``), this is done using
  the `weighted_match_nodes <EMComposition.weighted_match_nodes>`, each of which receives a Projection from a
  `match_node <EMComposition.match_nodes>` and the corresponding `field_weight_node <EMComposition.field_weight_nodes>`
  and multiplies them to produce the weighted distance for that field as its output.  However, if
  `use_gating_for_weighting <EMComposition.use_gating_for_weighting>` is ``True``, the `field_weight_nodes` are
  implemented as `GatingMechanisms <GatingMechanism>`, each of which uses its `field weight
  <EMComposition.field_weights>` as a `GatingSignal <GatingSignal>` to output gate (i.e., multiplicatively modulate
  the output of) the corresponding `match_node <EMComposition.match_nodes>`. In this case, the `weighted_match_nodes
  are not implemented, and the output of the `match_node <EMComposition.match_nodes>` is passed directly to the
  `combined_matches_node <EMComposition.combined_matches_node>`.


  .. _EMComposition_Gating_For_Weighting:
  .. note::
     Setting `use_gating_for_weighting <EMComposition.use_gating_for_weighting>` to ``True`` reduces the size and
     complexity of the EMComposition, by eliminating the `weighted_match_nodes <EMComposition.weighted_match_nodes>`.
     However, doing to precludes the ability to learn the `field_weights <EMComposition.field_weights>`,
     since `GatingSignals <GatingSignal>` are  `ModulatorySignal>` that cannot be learned.  If learning is required,
     then `use_gating_for_weighting` should be set to ``False``.

* **Combine distances**.  If `field weights <EMComposition_Field_Weights>` are used to specify more than one `key field
  <EMComposition_Fields>`, then the (weighted) distances computed for each field (see above) are summed across fields
  by the `combined_matches_node <EMComposition.combined_matches_node>`, before being passed to the `softmax_node
  <EMComposition.softmax_node>`. If only one key field is specified, then the output of the `match_node
  <EMComposition.match_nodes>` is passed directly to the `softmax_node <EMComposition.softmax_node>`.

* **Softmax normalize distances**. The distances, passed either from the `combined_matches_node
  <EMComposition.combined_matches_node>`, or directly from the `match_node <EMComposition.match_nodes>` if there is
  only one key field, are passed to the `softmax_node <EMComposition.softmax_node>`, which applies the `SoftMax`
  Function, which generates the softmax distribution used to retrieve entries from `memory <EMComposition.memory>`.
  If a numerical value is specified for `softmax_gain <EMComposition.softmax_gain>`, that is used as the gain (inverse
  temperature) for the SoftMax Function; if *ADAPTIVE* is specified, then the `SoftMax.adapt_gain` function is used
  to adaptively set the gain based on the summed distance (i.e., the output of the `combined_matches_node
  <EMComposition.combined_matches_node>`;  if *CONTROL* is specified, then the summed distance is monitored by a
  `ControlMechanism` that uses the `adapt_gain <Softmax.adapt_gain>` method of the `SoftMax` Function to modulate its
  `gain <Softmax.gain>` parameter; if ``None`` is specified, the default value of the `Softmax` Function is used as
  the `gain <Softmax.gain>` parameter (see `Softmax_Gain <EMComposition_Softmax_Gain>` for additional  details).

.. _EMComposition_Retreived_Values:

* **Retrieve values by field**. The vector of softmax weights for each memory generated by the `softmax_node
  <EMComposition.softmax_node>` is passed through the Projections to the each of the `retrieved_nodes
  <EMComposition.retrieved_nodes>` to compute the retrieved value for each field, which is assigned as the value
  of the corresponding `retrieved_node <EMComposition.retrieved_nodes>`.

* **Decay memories**.  If `memory_decay <EMComposition.memory_decay>` is ``True``, then each of the memories is
  decayed by the amount specified in `memory_decay_rate <EMComposition.memory_decay_rate>`.

    .. technical_note::
       This is done by multiplying the `matrix <MappingProjection.matrix>` parameter of the `MappingProjection` from
       the `combined_matches_node <EMComposition.combined_matches_node>` to each of the `retrieved_nodes
       <EMComposition.retrieved_nodes>`, as well as the `matrix <MappingProjection.matrix>` parameter of the
       `MappingProjection` from each `query_input_node <EMComposition.query_input_nodes>` to the corresponding
       `match_node <EMComposition.match_nodes>` by `memory_decay <EMComposition.memory_decay_rate>`,
        by 1 - `memory_decay <EMComposition.memory_decay_rate>`.

.. _EMComposition_Storage:

* **Store memories**. After the values have been retrieved, the `storage_node <EMComposition.storage_node>`
  adds the inputs to each field (i.e., values in the `query_input_nodes <EMComposition.query_input_nodes>` and
  `value_input_nodes <EMComposition.value_input_nodes>`) as a new entry in `memory <EMComposition.memory>`,
  replacing the weakest one. The weakest memory is the one with the lowest norm, multipled  by its `field_weight
  <EMComposition.field_weights>` if `purge_by_field_weight <EMComposition.purge_by_field_weight>` is ``True``.

    .. technical_note::
       The norm of each entry is calculated by adding the input vectors to the the corresponding rows of
       the `matrix <MappingProjection.matrix>` of the `MappingProjection` from the `combined_matches_node
       <EMComposition.combined_matches_node>` to each of the `retrieved_nodes <EMComposition.retrieved_nodes>`,
       as well as the `matrix <MappingProjection.matrix>` parameter of the `MappingProjection` from each
       `query_input_node <EMComposition.query_input_nodes>` to the corresponding `match_node
       <EMComposition.match_nodes>` (see note `above <EMComposition_Memory_Storage>` for additional details).

  .. note::
     During training, storage occurs after the weights have been updated for a given input (see `note
     <EMComposition_Storage_Learning>` below).

COMMENT:
FROM CodePilot: (OF HISTORICAL INTEREST?)
inputs to its `query_input_nodes <EMComposition.query_input_nodes>` and
`value_input_nodes <EMComposition.value_input_nodes>` are assigned the values of the corresponding items in the
`input <Composition.input>` argument.  The `combined_softmax_node <EMComposition.field_weight_node>`
computes the dot product of each query with each key in memory, and then applies a softmax function to each row of the
resulting matrix.  The `retrieved_nodes <EMComposition.retrieved_nodes>` then compute the dot product of the
softmaxed values for each memory with the corresponding value for each memory, and the result is assigned to the
corresponding `output <Composition.output>` item.
COMMENT

.. _EMComposition_Learning_Execution:

*Learning*
~~~~~~~~~~

If `learn <Composition.learn>` is called, `enable_learning <EMComposition.enable_learning>` is ``True``, then errors
will be computed for each of the `retrieved_nodes <EMComposition.retrieved_nodes>` that is specified for learning
(see `Learning <EMComposition_Learning_Creation>` for details about specification). These errors are derived either
from any errors backprpated to the EMComposition from an outer Composition in which it is `nested <Composition_Nested>`,
or locally by the difference between the `retrieved_nodes <EMComposition.retrieved_nodes>` and the `target_nodes
<EMComposition.target_nodes>` that are created for each of the `retrieved_nodes <EMComposition.retrieved_nodes>`
that do not project to an outer Composition. These errors are then backpropagated through the EMComposition to the
`query_input_nodes <EMComposition.query_input_nodes>` and `value_input_nodes <EMComposition.value_input_nodes>`,
and on to any nodes that project to it from a composition in which the EMComposition is `nested <Composition_Nested>`.

If `learn_field_weights` is also specified, then the corresponding `field_weights <EMComposition.field_weights>` are
modified to minimize the error passed to the EMComposition retrieved nodes that have been specified for learning,
using the `learning_rate <EMComposition.learning_rate>` for them in `learn_field_weights
<EMComposition.learn_field_weights>` or the default `learning rate <EMComposition.learning_rate>` for the EMComposition.
If `enable_learning <EMComposition.enable_learning>` is ``False`` (or `run <Composition.run>` is called rather than
`learn <Composition.learn>`, then the `field_weights <EMComposition.field_weights>` are not modified, and no error
signals are passed to the nodes that project to  its `query_input_nodes <EMComposition.query_input_nodes>` and
`value_input_nodes <EMComposition.value_input_nodes>`.

  .. note::
     The only parameters modifable by learning in the EMComposition are its `field_weights
     <EMComposition.field_weights>`; all other parameters (including all other Projection `matrices
     <MappingProjection.matrix>`) are fixed, and used only to compute gradients and backpropagate errors.

  .. technical_note::
     Although memory storage is implemented as a form of learning (though modification of MappingProjection
     `matrix <MappingProjection.matrix>` parameters; see `memory storage <EMComposition_Memory_Storage>`),
     this occurs irrespective of how EMComposition is run (i.e., whether `learn <Composition.learn>` or `run
     <Composition.run>` is called), and is not affected by the `enable_learning <EMComposition.enable_learning>`
     or `learning_rate <EMComposition.learning_rate>` attributes, which pertain only to whether the `field_weights
     <EMComposition.field_weights>` are modified during learning.  Furthermore, when run in PyTorch mode, storage
     is executed after the forward() and backward() passes are complete, and is not considered as part of the
     gradient calculations.

  .. _EMComposition_Storage_Learning:

  .. note:
     Storage always occurs *after* the learning (gradient calculation and weight updates) has occured for an input;
     if there this is more than one optimization step for a given input (i.e., if `optimizations_per_minibatch
     <Composition.optimizations_per_minibatch>` is greater than 1), then storage occurs on the optimizaton step(s)
     determined by the `store_on_optimization <EMComposition.store_on_optimization>` Parameter, which can have the
     following values:

     * *FIRST* * storage occurs after the first optimization step (weight update); this means that any values
       generated during additional optimization steps will not be stored in EM, and the effect of those optimizations
       on any values stored subsequently will not be observed until the next input is presented (i.e., the next `TRIAL
       <TimeScale.TRIAL>`'

      * *LAST* * storage occurs after the last optimization step (weight update); this means that any values
        generated during the preceding optimization steps will not be stored in EM, and the effect of those
        optimizations will not be observed until the next input is presented (i.e., the next `TRIAL <TimeScale.TRIAL>`'

      * *ALL* * storage occurs after all optimization steps (weight update), so that values generated during preceding
        optimzation steps will impact subsequent optimization steps for the same input (i.e., `TRIAL <TimeScale.TRIAL>`'

      .. technical_note::
         Execution of storage during the first optimization step is implemented in `PytorchEMMechanismWrapper.execute`;
         by default, this is for optimization_num==0, to ensure that the current values of any input nodes
         (reflecting the *previous input*) are stored before their values are updated to the current inputs
         (at the end of a full  execution of `Composition.execute` in the first optimization step), to deal
         with cases in which EM is executed before those, as in the EGO model for *PREVIOUS STATE* and *CONTEXT*:
         <EGO Model>.scheduler.add_condition(em, BeforeNodes(previous_state_layer, context_layer)).

.. _EMComposition_Examples:

Examples
--------

.. _EMComposition_Example_Case:

The following are examples of how to configure and initialize the EMComposition's `memory <EMComposition.memory>`:

*Visualizing the EMComposition*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The EMComposition can be visualized graphically, like any `Composition`, using its `show_graph
<ShowGraph_show_graph_Method>` method.  For example, the figure below shows an EMComposition that
implements a simple dictionary, with one key field and one value field, each of length 5::

    >>> import psyneulink as pnl
    >>> em = EMComposition(memory_template=(2,5))
    >>> em.show_graph()
    <BLANKLINE>

.. _EMComposition_Example_fig:

.. figure:: _static/EMComposition_Example_fig.svg
   :alt: Example of an EMComposition
   :align: left

       **Example of an EMComposition**

       .. note::
          The order in which the nodes at a given level (e.g., the `INPUT <NodeRole.INPUT>` or `OUTPUT
          <NodeRole.OUTPUT>` `Nodes <Composition_Nodes>`) are shown in the diagram is arbitrary, and does not necessarily
          reflect the order in which they are created or specied in the script.

.. _EMComposition_Example_Memory_Template:

*Memory Template*
~~~~~~~~~~~~~~~~~

The `memory_template <EMComposition_Memory_Template>` argument of a EMComposition's constructor is used to configure
it `memory <EMComposition.memory>`, which can be specified using either a tuple or a list or array.

.. _EMComposition_Example_Tuple_Spec:

**Tuple specification**

The simplest form of specification is a tuple, that uses the `numpy shape
<https://numpy.org/doc/stable/reference/generated/numpy.shape.html>`_ format.  If it has two elements (as in the
example above), the first specifies the number of fields, and the second the length of each field.  In this case,
a default number of entries (1000) is created:

    >>> em.memory_capacity
    1000

The number of entries can be specified explicitly in the EMComposition's constructor, using either the
`memory_capacity <EMComposition_Memory_Capacity>` argument, or by using a 3-item tuple to specify the
`memory_template <EMComposition_Memory_Template>` argument, in which case the first element specifies
the  number of entries, while the second and their specify the number of fields and the length of each field,
respectively.  The following are equivalent::

    >>> em = EMComposition(memory_template=(2,5), memory_capcity=4)

and

    >>> em = EMComposition(memory_template=(4,2,5))

both of which create a memory with 4 entries, each with 2 fields of length 5. The contents of `memory
<EMComposition_Memory_Specification>` can be inspected using the `memory <EMComposition.memory>` attribute::

    >>> em.memory
    [[array([0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0.])],
     [array([0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0.])],
     [array([0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0.])],
     [array([0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0.])]]

The default for `memory_capacity <EMComposition.memory_capacity>` is 1000, which is used if it is not otherwise
specified.

**List or array specification**

Note that in the example above the two fields have the same length (5). This is always the case when a tuple is used,
as it generates a regular array.  A list or numpy array can also be used to specify the **memory_template** argument.
For example, the following is equivalent to the examples above::

    >>> em = EMComposition(memory_template=[[0,0,0],[0,0,0]], memory_capacity=4)

However, a list or array can be used to specify fields of different length (i.e., as a ragged array).  For example,
the following specifies one field of length 3 and another of length 1::

    >>> em = EMComposition(memory_template=[[0,0,0],[0]], memory_capacity=4)
    >>> em.memory
    [[[array([0., 0., 0.]), array([0.])]],
     [[array([0., 0., 0.]), array([0.])]],
     [[array([0., 0., 0.]), array([0.])]],
     [[array([0., 0., 0.]), array([0.])]]]

.. _EMComposition_Example_Memory_Fill:

**Memory fill**

Note that the examples above generate a warning about the use of zeros to initialize the memory. This is
because the default value for **memory_fill** is ``0``, and the default value for `normalize_memories
<EMComposition.normalize_memories>` is ``True``, which will cause a divide by zero warning when memories are
normalized. While this doesn't crash, it will result in nan's that are likely to cauase problems elsewhere.
This can be avoided by specifying a non-zero  value for **memory_fill**, such as small number::

    >>> em = EMComposition(memory_template=[[0,0,0],[0]], memory_capacity=4, memory_fill=.001)
    >>> em.memory
    [[[array([0.001, 0.001, 0.001]), array([0.001])]],
     [[array([0.001, 0.001, 0.001]), array([0.001])]],
     [[array([0.001, 0.001, 0.001]), array([0.001])]],
     [[array([0.001, 0.001, 0.001]), array([0.001])]]]

Here, a single value was specified for **memory_fill** (which can be a float or int), that is used to fill all values.
Random values can be assigned using a tuple to specify and internval between the first and second elements.  For
example, the following uses random values between 0 and 0.01 to fill all entries::

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

Note that the two entries must have exactly the same shapes. If they do not, an error is generated.
Also note that the remaining entries are filled with zeros (the default value for **memory_fill**).
Here again, **memory_fill** can be used to specify a different value::

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
For example, in the `figure <EMComposition_Example_fig>` above, the first field specified was used as a key field,
and the last as a value field. However, the **field_weights** argument can be used to modify this, specifying which
fields should be used as keys fields -- including the relative contribution that each makes to the matching process
-- and which should be used as value fields.  Non-zero elements in the **field_weights** argument designate key fields,
and zeros specify value fields. For example, the following specifies that the first two fields should be used as keys
while the last two should be used as values::

    >>> em = EMComposition(memory_template=[[0,0,0],[0],[0,0],[0,0,0,0]], memory_capacity=3, field_weights=[1,1,0,0])
    >>> em.show_graph()
    <BLANKLINE>


.. _EMComposition_Example_Field_Weights_Equal_fig:

.. figure:: _static/EMComposition_field_weights_equal_fig.svg

    **Use of field_weights to specify keys and values.**

Note that the figure now shows `<QUERY> [WEIGHT] <EMComposition.field_weight_nodes>` `Nodes <Composition_Nodes>`,
that are used to implement the relative contribution that each key field makes to the matching process specifed in
`field_weights <EMComposition.field_weights>` argument.  By default, these are equal (all assigned a value of 1),
but different values can be used to weight the relative contribution of each key field.  The values are normalized so
that they sum 1, and the relative contribution of each is determined by the ratio of its value to the sum of all
non-zero values.  For example, the following specifies that the first two fields should be used as keys,
with the first contributing 75% to the matching process and the second field contributing 25%::

    >>> em = EMComposition(memory_template=[[0,0,0],[0],[0,0]], memory_capacity=3, field_weights=[3,1,0])
    <BLANKLINE>

COMMENT:
.. _EMComposition_Example_Field_Weights_Different_fig:

.. figure:: _static/EMComposition_field_weights_different.svg

    **Use of field_weights to specify relative contribution of fields to matching process.**

Note that in this case, the `concatenate_queries_node <EMComposition.concatenate_queries_node>` has been replaced by
a pair of `weighted_match_node <EMComposition.weighted_match_node>`, one for each key field.  This is because
the keys were assigned different weights;  when they are assigned equal weights, or if no weights are specified,
and `normalize_memories <EMComposition.normalize_memories>` is ``True``, then the keys are concatenated and are
concatenated for efficiency of processing.  This can be suppressed by specifying `concatenate_queries` as ``False``
(see `concatenate_queries <EMComposition_Concatenate_Queries>` for additional details).
COMMENT

.. _EMComposition_Class_Reference:

Class Reference
---------------
"""


import copy
import warnings
from typing import Optional, Union
from enum import Enum

import numpy as np
import torch

import psyneulink.core.scheduling.condition as conditions

from psyneulink.core.components.functions.function import DEFAULT_SEED, _random_state_getter, _seed_setter
from psyneulink.core.components.functions.nonstateful.transferfunctions import SoftMax
from psyneulink.core.components.functions.nonstateful.transformfunctions import Concatenate, LinearCombination, MatrixTransform
from psyneulink.core.components.functions.userdefinedfunction import UserDefinedFunction
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.modulatory.control.gating.gatingmechanism import GatingMechanism
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.compositions.composition import CompositionError, NodeRole
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import (
    ADAPTIVE,
    ALL,
    ARG_MAX,
    ARG_MAX_INDICATOR,
    AUTO,
    CONTEXT,
    CONTROL,
    DEFAULT_INPUT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_VARIABLE,
    DOT_PRODUCT,
    EM_COMPOSITION,
    FIRST,
    FULL_CONNECTIVITY_MATRIX,
    GAIN,
    IDENTITY_MATRIX,
    INPUT_SHAPES,
    LAST,
    L0,
    MULTIPLICATIVE_PARAM,
    NAME,
    OWNER_VALUE,
    PARAMS,
    PROB_INDICATOR,
    PRODUCT,
    PROJECTIONS,
    RANDOM,
    VARIABLE,
)
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.utilities import (
    ContentAddressableList,
    convert_all_elements_to_np_array,
    is_iterable,
    is_numeric_scalar,
)
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.core.scheduling.condition import AfterNodes, All, Always, Any, BeforeNCalls, AfterNCalls
from psyneulink.core.llvm import ExecutionMode
from psyneulink.library.components.mechanisms.processing.integrator.externalmemorymechanism import (
    ExternalMemoryMechanism, NORMS, QUERY, SCORES, RETRIEVED, COMBINED_SCORES, COMBINED_NORMS)
from psyneulink.library.compositions.autodiffcomposition import AutodiffComposition, torch_available


__all__ = [
    "EMComposition",
    "EMCompositionError",
    "FieldType",
    "KEY",
    "FIELD_MEMORY",
    "FIELD_WEIGHT",
    'CONCATENATE_QUERIES_NAME',
    "LEARN_FIELD_WEIGHT",
    "PROBABILISTIC",
    "TARGET_FIELD",
    "WEIGHTED_AVG",
    "WEIGHTED_SCORES"
]


KEY = "key"

# softmax_choice options:
STORAGE_PROB = "storage_prob"
WEIGHTED_AVG = ALL
PROBABILISTIC = PROB_INDICATOR

# specs for entry of fields specification dict
FIELD_WEIGHT = "field_weight"
LEARN_FIELD_WEIGHT = "learn_field_weight"
TARGET_FIELD = "target_field"

# Node names
QUERY_NODE_NAME = "QUERY"
QUERY_AFFIX = f" [{QUERY_NODE_NAME}]"
VALUE_NODE_NAME = "VALUE"
VALUE_AFFIX = f" [{VALUE_NODE_NAME}]"
FIELD_MEMORY = "FIELD_MEMORY"
FIELD_MEMORY_AFFIX = f" [{FIELD_MEMORY}]"
WEIGHT = "WEIGHT"
WEIGHT_AFFIX = f" [{WEIGHT}]"
WEIGHTED_SCORES = "WEIGHTED SCORE"
WEIGHTED_SCORES_NODE_NAME = "WEIGHTED SCORES"
WEIGHTED_SCORES_AFFIX = f" [{WEIGHTED_SCORES_NODE_NAME}]"
CONCATENATED = "CONCATENATED"
CONCATENATED_AFFIX = f" [{CONCATENATED}]"
CONCATENATE_QUERIES_NAME = "CONCATENATE QUERIES"
COMBINED_SCORES_NODE_NAME = "COMBINED SCORES"
RETRIEVED_NODE_NAME = "RETRIEVED"
RETRIEVED_AFFIX = " [RETRIEVED]"


class FieldType(Enum):
    KEY = 0
    VALUE = 1


def _memory_getter(owning_component=None, context=None):
    """Return EMComposition memory as a 3d object array: entries x fields x field_values.
    These are derived from the memory attribute of the field_memory_node of each field.
    """
    if owning_component is None or owning_component.is_initializing:
        return None

    field_memories = [
        np.asarray(field.memory_node.function.parameters.memory.get(context))
        for field in owning_component.fields
    ]

    memory_capacity = owning_component.memory_capacity or owning_component.defaults.memory_capacity
    return convert_all_elements_to_np_array([
        [field_memories[field_idx][entry_idx] for field_idx in range(owning_component.num_fields)]
        for entry_idx in range(memory_capacity)
    ])

def field_weights_setter(field_weights, owning_component=None, context=None):
    if (
        owning_component is None
        or not owning_component.parameters.field_weights._has_value(context)
        or owning_component.parameters.field_weights._get(context) is None
    ):
        return field_weights

    if len(field_weights) != len(owning_component.field_weights):
        raise EMCompositionError(
            f"The number of field_weights ({len(field_weights)}) must match "
            f"the number of fields ({len(owning_component.field_weights)})."
        )

    field_weights = list(field_weights)
    for i, fw in enumerate(field_weights.copy()):
        field_weights[i] = None if fw is None else fw

    if owning_component.normalize_field_weights:
        denominator = np.sum([fw if fw is not None else 0 for fw in field_weights]) or 1
        field_weights = [fw / denominator if fw is not None else None for fw in field_weights]

    field_wt_node_idx = 0
    for i, field_weight in enumerate(field_weights):
        if owning_component.parameters.field_weights.default_value[i] is None:
            if field_weight:
                raise EMCompositionError(
                    f"Field '{owning_component.field_names[i]}' of '{owning_component.name}' was originally assigned "
                    f"as a value node (i.e., with a field_weight = None); this cannot be changed after construction. "
                    f"If you want to change it to a key field, you must re-construct the EMComposition using a scalar "
                    f"for its field in the `field_weights` arg (which can be 0).")

            continue
        owning_component.field_weight_nodes[field_wt_node_idx].input_port.defaults.variable = field_weight
        owning_component.fields[i].weight = field_weight
        field_wt_node_idx += 1

    return np.array(field_weights, dtype=object)


def get_softmax_gain(v, scale=1, base=1, entropy_weighting=.1) -> float:
    v = np.squeeze(v)
    # # MODIFIED EM2 OLD:
    # gain = scale * (base +
    #                 (entropy_weighting *
    #                  np.log(
    #                      -1 * np.sum((1 / (1 + np.exp(-1 * v))) * np.log(1 / (1 + np.exp(-1 * v)))))))
    # return gain
    # MODIFIED EM2 NEW:
    logistic = 1 / (1 + np.exp(-1 * v))
    entropy = -1 * np.sum(logistic * np.log(logistic))
    return scale * (base + entropy_weighting * np.log(entropy))
    # MODIFIED EM2 END


class Field:
    """Object that contains information about a field in an EMComposition's memory."""

    name = None

    def __init__(
        self,
        name: str = None,
        index: int = None,
        type: FieldType = None,
        weight: float = None,
        learn_weight: bool = None,
        learning_rate: float = None,
        target: bool = None,
    ):
        self.name = name
        self.index = index
        self.type = type
        self.weight = weight
        self.learn_weight = learn_weight
        self.learning_rate = learning_rate
        self.target = target

        self.input_node = None
        self.memory_node = None
        self.weight_node = None
        self.weighted_scores_node = None
        self.retrieved_node = None

        self.query_projection = None
        self.concatenation_projection = None
        self.scores_projection = None
        self.norms_projection = None
        self.combined_scores_projection = None
        self.combined_norms_projection = None
        self.retrieved_projection = None
        self.weight_projection = None
        self.weighted_scores_projection = None
        self.weighted_norms_projection = None

        self.missing_value = False


    @property
    def nodes(self):
        return [
            node for node in [
                self.input_node,
                self.memory_node,
                self.weight_node,
                self.weighted_scores_node,
                self.retrieved_node,
            ]
            if node is not None
        ]

    @property
    def projections(self):
        return [
            proj for proj in [
                self.query_projection,
                self.concatenation_projection,
                self.scores_projection,
                self.norms_projection,
                self.combined_scores_projection,
                self.combined_norms_projection,
                self.retrieved_projection,
                self.weight_projection,
                self.weighted_scores_projection,
                self.weighted_norms_projection,
            ]
            if proj is not None
        ]

    @property
    def query(self):
        return self.input_node.variable

    @property
    def match(self):
        return self.memory_node.output_ports[SCORES].value

    @property
    def retrieved_memory(self):
        return self.memory_node.output_ports[RETRIEVED].value

    @property
    def memory(self):
        return self.memory_node.memory

    @property
    def memories(self):
        return self.memory_node.function.parameters.memory.get(None)


class EMCompositionError(CompositionError):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class EMComposition(AutodiffComposition):
    """
        EMComposition(                  \
        memory_template=[[0],[0]],      \
        memory_fill=0,                  \
        memory_capacity=None,           \
        fields=None,                    \
        field_names=None,               \
        field_weights=None,             \
        learn_field_weights=False,      \
        learning_rate=True,             \
        normalize_field_weights=True,   \
        concatenate_queries=False,      \
        normalize_memories=True,        \
        softmax_choice=WEIGHTED_AVG,    \
        softmax_gain=1.0,               \
        softmax_threshold=.001,         \
        storage_prob=1.0,               \
        store_on_optimization=FIRST,    \
        memory_decay_rate=AUTO,         \
        enable_learning=True,           \
        target_fields=None,             \
        use_gating_for_weighting=False, \
        name="EM_Composition"           \
        )

    COMMENT:
    Refactored EMComposition.

    This version replaces:
      - match_nodes backed by memory Projection matrices
      - retrieved_nodes backed by memory Projection matrices
      - EMStorageMechanism

    with:
      - one ExternalMemoryMechanism per field, each owning its field memory matrix.
      - storage occurs in each memory_node based on access_condition an its storage_prob

    The externally visible structure is kept similar to the original EMComposition:
      - input_nodes
      - query_input_nodes
      - value_input_nodes
      - field_weight_nodes
      - weighted_scores_nodes
      - combined_scores_node
      - retrieved_nodes

    Internally, field.memory_node is now the memory owner for each field.
    COMMENT

    Subclass of `AutodiffComposition` that implements the functions of an `EpisodicMemoryMechanism` in a
    differentiable form, in which its `field_weights <EMComposition_Proj.field_weights>` parameter can be learned,
    and that supports `memory_decay <EMComposition_Memory_Decay_Rate>`.

    Takes only the following arguments, all of which are optional

    Arguments
    ---------

    memory_template : tuple, list, 2d or 3d array : default [[0],[0]]
        specifies the shape of an item to be stored in the EMComposition_Proj's memory
        (see `memory_template <EMComposition_Proj_Memory_Template>` for details).

    memory_fill : scalar or tuple : default 0
        specifies the value used to fill the memory when it is initialized
        (see `memory_fill <EMComposition_Proj_Memory_Fill>` for details).

    memory_capacity : int : default None
        specifies the number of items that can be stored in the EMComposition_Proj's memory;
        (see `memory_capacity <EMComposition_Proj_Memory_Capacity>` for details).

    fields : dict[tuple[field weight, learning specification]] : default None
        each key must a string that is the name of a field, and its value a dict or tuple that specifies that field's
        `field_weight <EMComposition_Proj.field_weights>`, `learn_field_weights <EMComposition_Proj.learn_field_weights>`, and
        `target_fields <EMComposition_Proj.target_fields>` specifications (see `fields <EMComposition_Proj_Fields>` for details
        of specificaton format). The **fields** arg replaces the **field_names**, **field_weights**
        **learn_field_weights**, and **target_fields** arguments, and specifying any of these raises an error.

    field_names : list or tuple : default None
        specifies the names assigned to each field in the memory_template (see `field names <EMComposition_Proj_Field_Names>`
        for details). If the **fields** argument is specified, specifying **field_names** is not necessary and
        doing so raises a warning.

    field_weights : list or tuple : default (1,0)
        specifies the relative weight assigned to each key when matching an item in memory (see `field weights
        <EMComposition_Proj_Field_Weights>` for additional details). If the **fields** argument is specified, specifying
        **field_weights** is not necessary and doing so raises a warning.

    learn_field_weights : bool or list[bool, int, float]: default False
        specifies whether the `field_weights <EMComposition_Proj.field_weights>` are learnable and, if so, optionally what
        the learning_rate is for each field (see `learn_field_weights <EMComposition_Proj_Field_Weights_Learning>` for
        specifications). If the **fields** argument is specified, specifying **learn_field_weights** is not necessary,
        and doing so raises a warning.

    learning_rate : float : default .01
        specifies the default learning_rate for `field_weights <EMComposition_Proj.field_weights>` not
        specified in `fields <EMComposition_Proj.fields>` or `learn_field_weights <EMComposition_Proj.learn_field_weights>`
        (see `learning_rate <EMComposition_Proj_Field_Weights_Learning>` for additional details).

    normalize_field_weights : bool : default True
        specifies whether the **fields_weights** are normalized over the number of keys, or used as absolute
        weighting values when retrieving an item from memory (see `normalize_field weights
        <EMComposition_Normalize_Field_Weights>` for additional details).

    concatenate_queries : bool : default False
        specifies whether to concatenate the keys into a single field before matching them to items in
        the corresponding fields in memory (see `concatenate keys <EMComposition_Concatenate_Queries>` for details).

    normalize_memories : bool : default True
        specifies whether keys and memories are normalized before computing their dot product (similarity)
        (see `Match memories by field <EMComposition_Proj_Processing>` for additional details).

    softmax_choice : WEIGHTED_AVG, ARG_MAX, PROBABILISTIC : default WEIGHTED_AVG
        specifies how the softmax over distances of queries and keys in memory is used for retrieval
        (see `softmax_choice <EMComposition_Softmax_Choice>` for a description of each option).

    softmax_gain : float, ADAPTIVE or CONTROL : default 1.0
        specifies the temperature used for softmax normalizing the distance of queries and keys in memory
        (see `Softmax normalize matches over fields <EMComposition_Proj_Processing>` for additional details).

    softmax_threshold : float : default .001
        specifies the threshold used to mask out small values in the softmax calculation
        see *mask_threshold* under `Thresholding and Adaptive Gain <SoftMax_AdaptGain>` for details).

    storage_prob : float : default 1.0
        specifies the probability that an item will be stored in `memory <EMComposition_Proj.memory>`
        when the EMComposition_Proj is executed (see `Retrieval and Storage <EMComposition_Storage>` for
        additional details).

    store_on_optimization : FIRST, LAST, ALL : default FIRST
        specifies the optimization step(s) on which items are stored in `memory <EMComposition_Proj.memory>` during
        learning (see `EMComposition_Storage_Learning` for details).

    memory_decay_rate : float : AUTO
        specifies the rate at which items in the EMComposition_Proj's memory decay
        (see `memory_decay_rate <EMComposition_Proj_Memory_Decay_Rate>` for details).

    purge_by_field_weights : bool : False
        specifies whether `fields_weights <EMComposition_Proj.field_weights>` are used to determine which memory to
        replace when a new one is stored (see `purge_by_field_weight <EMComposition_Purge_by_Weight>` for details).

    enable_learning : bool : default True
        specifies whether learning is enabled for the EMCComposition (see `Learning <EMComposition_Proj_Learning_Creation>`
        for additional details); **use_gating_for_weighting** must be ``False``.

    target_fields : list[bool]: default None
        specifies whether a learning pathway is constructed for each `field <EMComposition_Proj_Entries_and_Fields>`
        of the EMComposition_Proj.  If it is a list, each item must be ``True`` or ``False`` and the number of items
        must be equal to the number of `fields <EMComposition_Proj_Fields> specified (see `Target Fields
         <EMComposition_Target_Fields>` for additional details). If the **fields** argument is specified,
         specifying **target_fields** is not necessary and doing so raises a warning.

    # 7/10/24 FIX: STILL TRUE?  DOES IT PRECLUDE USE OF EMComposition_Proj as a nested Composition??
    .. technical_note::
        use_storage_node : bool : default True
            specifies whether to use a `LearningMechanism` to store entries in `memory <EMComposition_Proj.memory>`.
            If False, a method on EMComposition_Proj is used rather than a LearningMechanism. This is meant for
            debugging, and precludes use of `import_composition <Composition.import_composition>` to integrate
            the EMComposition_Proj into another Composition;  to do so, use_storage_node must be ``True`` (default).

    use_gating_for_weighting : bool : default False
        specifies whether to use output gating to weight the `match_nodes <EMComposition_Proj.match_node>` instead of
        a standard input (see `Weight distances <EMComposition_Proj_Field_Weighting>` for additional details).

    Attributes
    ----------

    memory : ndarray
        3d array of entries in memory, in which each row (axis 0) is an entry, each column (axis 1) is a field, and
        each item (axis 2) is the value for the corresponding field (see `EMComposition_Proj_Memory_Specification`  for
        additional details).

        .. note::
           This is a read-only attribute;  memories can be added to the EMComposition_Proj's memory either by
           COMMENT:
           using its `add_to_memory <EMComposition_Proj.add_to_memory>` method, or
           COMMENT
           executing its `run <Composition.run>` or learn methods with the entry as the ``inputs`` argument.

    fields : ContentAddressableList[Field]
        list of `Field` objects, each of which contains information about the nodes and values of a field in the
        EMComposition_Proj's memory (see `Field`).

    .. _EMComposition_Parameters:

    memory_capacity : int
        determines the number of items that can be stored in `memory <EMComposition_Proj.memory>`
        (see `memory_capacity <EMComposition_Proj_Memory_Capacity>` for additional details).

    field_names : list[str]
        determines which names that can be used to label fields in `memory <EMComposition_Proj.memory>`
        (see `field_names <EMComposition_Proj_Field_Names>` for additional details).

    field_weights : tuple[float]
        determines which fields of the input are treated as "keys" (non-zero values) that are used to match entries in
        `memory <EMComposition_Proj.memory>` for retrieval, and which are used as "values" (zero values) that are stored
        and retrieved from memory but not used in the match process (see `Match memories by field
        <EMComposition_Proj_Processing>`; also determines the relative contribution of each key field to the match process;
        see `field_weights <EMComposition_Proj_Field_Weights>` additional details. The field_weights can be changed by
        assigning a new list of weights to the `field_weights <EMComposition_Proj.field_weights>` attribute, however only
        the weights for fields used as `keys <EMComposition_Proj_Entries_and_Fields>` can be changed (see
        `EMComposition_Proj_Field_Weights_Change_Note` for additional details).

    learn_field_weights : bool or list[bool, int, float]
        determines whether the `field_weight <EMComposition_Proj.field_weights>` for each `field <EMComposition_Proj_Fields>
        is learnable (see `learn_field_weights <EMComposition_Proj_Learning_Creation>` for additional details).

    learning_rate : float
        determines the default learning_rate for `field_weights <EMComposition_Proj.field_weights>`
        not specified in `learn_field_weights <EMComposition_Proj.learn_field_weights>`
        (see `learning_rate <EMComposition_Proj_Field_Weights_Learning>` for additional details).

    normalize_field_weights : bool
        determines whether `fields_weights <EMComposition_Proj.field_weights>` are normalized over the number of keys, or
        used as absolute weighting values when retrieving an item from memory (see `normalize_field weights
        <EMComposition_Normalize_Field_Weights>` for additional details).

    concatenate_queries : bool
        determines whether keys are concatenated into a single field before matching them to items in `memory
        <EMComposition_Proj.memory (see `concatenate keys <EMComposition_Concatenate_Queries>` for additional details).

    normalize_memories : bool
        determines whether keys and memories are normalized before computing their dot product (similarity)
        (see `Match memories by field <EMComposition_Proj_Processing>` for additional details).

    softmax_choice : WEIGHTED_AVG, ARG_MAX or PROBABILISTIC
        determines how the softmax over distances of queries and keys in memory is used for retrieval
        (see `softmax_choice <EMComposition_Softmax_Choice>` for a description of each option).

    softmax_gain : float, ADAPTIVE or CONTROL
        determines gain (inverse temperature) used for softmax normalizing the summed distances of queries
        and keys in memory by the `SoftMax` Function of the `softmax_node <EMComposition_Proj.softmax_node>`
        (see `Softmax normalize distances <EMComposition_Proj_Processing>` for additional details).

    softmax_threshold : float
        determines the threshold used to mask out small values in the softmax calculation
        (see *mask_threshold* under `Thresholding and Adaptive Gain <SoftMax_AdaptGain>` for details).

    storage_prob : float
        determines the probability that an item will be stored in `memory <EMComposition_Proj.memory>`
        when the EMComposition_Proj is executed (see `Retrieval and Storage <EMComposition_Storage>` for
        additional details).

    store_on_optimization : str
        determines the optimization step(s) on which items are stored in `memory <EMComposition_Proj.memory>` during
        learning (see `EMComposition_Storage_Learning` for details).

    memory_decay_rate : float
        determines the rate at which items in the EMComposition_Proj's memory decay
        (see `memory_decay_rate <EMComposition_Proj_Memory_Decay_Rate>` for details).

    purge_by_field_weights : bool
        determines whether `fields_weights <EMComposition_Proj.field_weights>` are used to determine which memory to
        replace when a new one is stored (see `purge_by_field_weight <EMComposition_Purge_by_Weight>` for details).

    enable_learning : bool
        determines whether learning is enabled for the EMCComposition
        (see `Learning <EMComposition_Proj_Learning_Creation>` for additional details).

    target_fields : list[bool]
        determines which fields convey error signals during learning
        (see `Target Fields <EMComposition_Target_Fields>` for additional details).

    .. _EMComposition_Nodes:

    query_input_nodes : list[ProcessingMechanism]
        `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` that receive keys used to determine the item
        to be retrieved from `memory <EMComposition_Proj.memory>`, and then themselves stored in `memory
        <EMComposition_Proj.memory>` (see `Match memories by field <EMComposition_Proj_Processing>` for additional details).
        By default these are assigned the name *KEY_n_INPUT* where n is the field number (starting from 0);
        however, if `field_names <EMComposition_Proj.field_names>` is specified, then the name of each query_input_node
        is assigned the corresponding field name appended with * [QUERY]*.

    value_input_nodes : list[ProcessingMechanism]
        `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` that receive values to be stored in `memory
        <EMComposition_Proj.memory>`; these are not used in the matching process used for retrieval.  By default these
        are assigned the name *VALUE_n_INPUT* where n is the field number (starting from 0);  however, if
        `field_names <EMComposition_Proj.field_names>` is specified, then the name of each value_input_node is assigned
        the corresponding field name appended with * [VALUE]*.

    concatenate_queries_node : ProcessingMechanism
        `ProcessingMechanism` that concatenates the inputs to `query_input_nodes <EMComposition_Proj.query_input_nodes>`
        into a single vector used for the matching processing if `concatenate keys <EMComposition_Proj.concatenate_queries>`
        is ``True``. This is not created if the **concatenate_queries** argument to the EMComposition_Proj's constructor is
        ``False`` or is overridden (see `concatenate_queries <EMComposition_Concatenate_Queries>`), or there is only
        one query_input_node. This node is named *CONCATENATE_QUERIES*

    field_memory_nodes : list[ProcessingMechanism]
        `ProcessingMechanisms <ProcessingMechanism>` that compute the dot product of each query and the key stored in
        the corresponding field of `memory <EMComposition_Proj.memory>` (see `Match memories by field
        <EMComposition_Proj_Processing>` for additional details). These are named the same as the corresponding
        `query_input_nodes <EMComposition_Proj.query_input_nodes>` appended with the suffix *[MATCH to KEYS]*.

    field_weight_nodes : list[ProcessingMechanism or GatingMechanism]
        Nodes used to weight the distances computed by the `match_nodes <EMComposition_Proj.match_nodes>` with the
        `field weight <EMComposition_Proj.field_weights>` for the corresponding `key field <EMComposition_Proj_Fields>`
        (see `Weight distances <EMComposition_Proj_Field_Weighting>` for implementation). These are named the same
        as the corresponding `query_input_nodes <EMComposition_Proj.query_input_nodes>`.

    weighted_match_nodes : list[ProcessingMechanism]
        `ProcessingMechanisms <ProcessingMechanism>` that combine the `field weight <EMComposition_Proj.field_weights>`
        for each `key field <EMComposition_Proj_Fields>` with the dot product computed by the corresponding the
        `match_node <EMComposition_Proj.match_nodes>`. These are only implemented if `use_gating_for_weighting
        <EMComposition_Proj.use_gating_for_weighting>` is ``False`` (see `Weight distances <EMComposition_Proj_Field_Weighting>`
        for details), and are named the same as the corresponding `query_input_nodes <EMComposition_Proj.query_input_nodes>`
        appended with the suffix *[WEIGHTED MATCH]*.

    BREADCRUMB: COMBINE THESE INTO combined_scores_node:
    combined_matches_node : ProcessingMechanism
        `ProcessingMechanism` that receives the weighted distances from the `weighted_match_nodes
        <EMComposition_Proj.weighted_match_nodes>` if more than one `key field <EMComposition_Proj_Fields>` is specified
        (or directly from `match_nodes <EMComposition_Proj.match_nodes>` if `use_gating_for_weighting
        <EMComposition_Proj.use_gating_for_weighting>` is ``True``), and combines them into a single vector that is passed
        to the `softmax_node <EMComposition_Proj.softmax_node>` for retrieval. This node is named *COMBINE MATCHES*.

    softmax_node : list[ProcessingMechanism]
        `ProcessingMechanisms <ProcessingMechanism>` that computes the softmax over the summed distances of keys
        and memories (output of the `combined_match_node <EMComposition_Proj.combined_match_node>`)
        from the corresponding `match_nodes <EMComposition_Proj.match_nodes>` (see `Softmax over summed distances
        <EMComposition_Proj_Processing>` for additional details).  This is named *RETRIEVE* (as it yields the
        softmax-weighted average over the keys in `memory <EMComposition_Proj.memory>`).

    softmax_gain_control_node : list[ControlMechanism]
        `ControlMechanisms <ControlMechanism>` that adaptively control the `softmax_gain <EMComposition_Proj.softmax_gain>`
        of the `softmax_node <EMComposition_Proj.softmax_node>`. This is implemented only if `softmax_gain
        <EMComposition_Proj.softmax_gain>` is specified as *CONTROL* (see `softmax_gain <EMComposition_Softmax_Gain>` for
        details).

    retrieved_nodes : list[ProcessingMechanism]
        `ProcessingMechanisms <ProcessingMechanism>` that receive the vector retrieved for each field in `memory
        <EMComposition_Proj.memory>` (see `Retrieve values by field <EMComposition_Proj_Processing>` for additional details).
        These are assigned the same names as the `query_input_nodes <EMComposition_Proj.query_input_nodes>` and
        `value_input_nodes <EMComposition_Proj.value_input_nodes>` to which they correspond appended with the suffix
        * [RETRIEVED]*, and are in the same order as  `input_nodes <EMComposition_Proj.input_nodes>`
        to which to which they correspond.

    storage_node : EMStorageMechanism
        `EMStorageMechanism` that receives inputs from the `query_input_nodes <EMComposition_Proj.query_input_nodes>` and
        `value_input_nodes <EMComposition_Proj.value_input_nodes>`, and stores these in the corresponding field of`memory
        <EMComposition_Proj.memory>` with probability `storage_prob <EMComposition_Proj.storage_prob>` after a retrieval has been
        made (see `Retrieval and Storage <EMComposition_Storage>` for additional details). This node is named *STORE*.

        .. technical_note::
           The `storage_node <EMComposition_Proj.storage_node>` is assigned a Condition to execute after the `retrieved_nodes
           <EMComposition_Proj.retrieved_nodes>` have executed, to ensure that storage occurs after retrieval, but before
           any subequent processing is done (i.e., in a composition in which the EMComposition_Proj may be embededded.

    input_nodes : list[ProcessingMechanism]
        Full list of `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` in the same order specified in the
        **field_names** argument of the constructor and in `self.field_names <EMComposition_Proj.field_names>`.

    query_and_value_input_nodes : list[ProcessingMechanism]
        Full list of `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` ordered with query_input_nodes first
        followed by value_input_nodes; used primarily for internal computations.
    """

    componentCategory = EM_COMPOSITION

    if torch_available:
        from psyneulink.library.compositions.emcomposition.pytorchEMwrappers import (
            PytorchEMCompositionWrapper, PytorchExternalMemoryMechanismWrapper,
        )
        pytorch_composition_wrapper_type = PytorchEMCompositionWrapper
        pytorch_mechanism_wrapper_type = PytorchExternalMemoryMechanismWrapper

    class Parameters(AutodiffComposition.Parameters):
        """
            Attributes
            ----------

                concatenate_queries
                    see `concatenate_queries <EMComposition.concatenate_queries>`

                    :default value: False
                    :type: ``bool``

                field_names
                    see `field_names <EMComposition.field_names>`

                    :default value: None
                    :type: ``list``

                field_weights
                    see `field_weights <EMComposition.field_weights>`

                    :default value: None
                    :type: ``numpy.ndarray``

                learn_field_weights
                    see `learn_field_weights <EMComposition.learn_field_weights>`

                    :default value: True
                    :type: ``numpy.ndarray``

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

                memory_decay_rate
                    see `memory_decay_rate <EMComposition.memory_decay_rate>`

                    :default value: 0.001
                    :type: ``float``

                memory_template
                    see `memory_template <EMComposition.memory_template>`

                    :default value: np.array([[0],[0]])
                    :type: ``np.ndarray``

                normalize_field_weights
                    see `normalize_field_weights <EMComposition.normalize_field_weights>`

                    :default value: True
                    :type: ``bool``

                normalize_memories
                    see `normalize_memories <EMComposition.normalize_memories>`

                    :default value: True
                    :type: ``bool``

                purge_by_field_weights
                    see `purge_by_field_weights <EMComposition.purge_by_field_weights>`

                    :default value: False
                    :type: ``bool``

                random_state
                    see `random_state <NormalDist.random_state>`

                    :default value: None
                    :type: ``numpy.random.RandomState``

                softmax_choice
                    see `softmax_choice <EMComposition.softmax_choice>`
                    :default value: WEIGHTED_AVG
                    :type: ``keyword``

                softmax_gain
                    see `softmax_gain <EMComposition.softmax_gain>`
                    :default value: 1.0
                    :type: ``float, ADAPTIVE or CONTROL``

                softmax_threshold
                    see `softmax_threshold <EMComposition.softmax_threshold>`
                    :default value: .001
                    :type: ``float``

                storage_prob
                    see `storage_prob <EMComposition.storage_prob>`

                    :default value: 1.0
                    :type: ``float``

                store_on_optimization
                    see `store_on_optimization <EMComposition.store_on_optimization>`

                    :default value: FIRST
                    :type: ``str``
        """
        memory = Parameter(None, loggable=True, getter=_memory_getter, read_only=True)
        memory_template = Parameter([[0], [0]], structural=True, valid_types=(tuple, list, np.ndarray), read_only=True)
        memory_capacity = Parameter(1000, structural=True)
        field_names = Parameter(None, structural=True)
        field_weights = Parameter([1], setter=field_weights_setter)
        learn_field_weights = Parameter(False, structural=True)
        normalize_field_weights = Parameter(True)
        concatenate_queries = Parameter(False, structural=True)
        normalize_memories = Parameter(True)
        softmax_choice = Parameter(WEIGHTED_AVG, modulable=False, specify_none=True)
        softmax_gain = Parameter(1.0, modulable=True)
        softmax_threshold = Parameter(.001, modulable=True, specify_none=True)
        storage_prob = Parameter(1.0, modulable=True)
        store_on_optimization = Parameter(FIRST)
        memory_decay_rate = Parameter(AUTO, modulable=True)
        purge_by_field_weights = Parameter(False, structural=True)
        target_fields = Parameter(None, read_only=True, structural=True)
        random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies="seed")
        seed = Parameter(DEFAULT_SEED(), modulable=True, setter=_seed_setter)

        def _validate_memory_template(self, memory_template):
            if isinstance(memory_template, tuple):
                if len(memory_template) not in {2, 3}:
                    return "must be length either 2 or 3 if it is a tuple."
                if not all(isinstance(item, int) for item in memory_template):
                    return "must have only integers as entries."
            elif isinstance(memory_template, (list, np.ndarray)):
                memory_template = np.array(memory_template, dtype=object)
                if memory_template.ndim not in {1, 2, 3}:
                    return "must be either 1d, 2d, or 3d."
            else:
                return "must be tuple, list, or array."

        def _validate_field_weights(self, field_weights):
            if field_weights is not None:
                if not np.atleast_1d(field_weights).ndim == 1:
                    return "must be a scalar, list of scalars, or 1d array."
                if any([field_weight < 0 for field_weight in field_weights if field_weight is not None]):
                    return "must all be positive values."

        def _validate_learn_field_weights(self, learn_field_weights):
            if isinstance(learn_field_weights, (list, np.ndarray)):
                if not all(isinstance(item, (bool, int, float, type(None))) for item in learn_field_weights):
                    return "can only contain bools, ints, floats, or None."
            elif not isinstance(learn_field_weights, bool):
                return "must be a bool or list of bools, ints, floats, or None."

        def _validate_memory_decay_rate(self, memory_decay_rate):
            if memory_decay_rate is None or memory_decay_rate == AUTO:
                return
            if not is_numeric_scalar(memory_decay_rate) or not 0 <= memory_decay_rate <= 1:
                return "must be a float in the interval [0, 1]."

        def _validate_softmax_gain(self, softmax_gain):
            if not is_numeric_scalar(softmax_gain) and softmax_gain not in {ADAPTIVE, CONTROL}:
                return f"must be a scalar or one of '{ADAPTIVE}' or '{CONTROL}'."

        def _validate_softmax_threshold(self, softmax_threshold):
            if softmax_threshold is not None and (not is_numeric_scalar(softmax_threshold) or softmax_threshold <= 0):
                return "must be a scalar greater than 0."

        def _validate_storage_prob(self, storage_prob):
            if not is_numeric_scalar(storage_prob) or not 0 <= storage_prob <= 1:
                return "must be a float in the interval [0, 1]."

        def _validate_store_on_optimization(self, option):
            if option not in {FIRST, LAST, ALL}:
                return "must be one of FIRST, LAST, or ALL."

    @check_user_specified
    def __init__(
        self,
        memory_template: Union[tuple, list, np.ndarray] = [[0], [0]],
        memory_capacity: Optional[int] = None,
        memory_fill: Union[int, float, tuple, RANDOM] = 0,
        fields: Optional[dict] = None,
        field_names: Optional[list] = None,
        field_weights: Union[int, float, list, tuple] = None,
        learn_field_weights: Union[bool, list, tuple] = None,
        learning_rate: Union[float, bool, int, dict] = None,
        normalize_field_weights: bool = True,
        concatenate_queries: bool = False,
        normalize_memories: bool = True,
        softmax_choice: Optional[Union[WEIGHTED_AVG, ARG_MAX, PROBABILISTIC]] = WEIGHTED_AVG,
        softmax_gain: Union[float, ADAPTIVE, CONTROL] = 1.0,
        softmax_threshold: Optional[float] = .001,
        storage_prob: float = 1.0,
        store_on_optimization: Union[FIRST, LAST, ALL] = FIRST,
        memory_decay_rate: Union[float, AUTO] = AUTO,
        purge_by_field_weights: bool = False,
        enable_learning: bool = True,
        target_fields: Optional[Union[list, tuple, np.ndarray]] = None,
        use_gating_for_weighting: bool = False,
        random_state=None,
        seed=None,
        name="EM_Composition",
        **kwargs,
    ):
        memory_fill = memory_fill or 0

        self._validate_memory_specs(
            memory_template,
            memory_capacity,
            memory_fill,
            field_weights,
            field_names,
            name,
            learn_field_weights,
        )

        self._enable_learning_warning_flag = False
        self._use_gating_for_weighting = use_gating_for_weighting

        memory_template, memory_capacity = self._parse_memory_template(memory_template,
                                                                       memory_capacity,
                                                                       memory_fill)

        self.fields = ContentAddressableList(component_type=Field)
        self.entry_template = memory_template[0]
        self.concatenate_queries_node = None

        (field_names,
         field_weights,
         learn_field_weights,
         target_fields,
         concatenate_queries,
         ) = self._parse_fields(fields,
                                field_names,
                                field_weights,
                                learn_field_weights,
                                learning_rate,
                                normalize_field_weights,
                                concatenate_queries,
                                normalize_memories,
                                target_fields,
                                name)

        if softmax_gain == CONTROL:
            self.parameters.softmax_gain.modulable = False

        super().__init__(
            name=name,
            memory_template=memory_template,
            memory_capacity=memory_capacity,
            field_names=field_names,
            field_weights=field_weights,
            learn_field_weights=learn_field_weights,
            learning_rate=learning_rate,
            normalize_field_weights=normalize_field_weights,
            concatenate_queries=concatenate_queries,
            normalize_memories=normalize_memories,
            softmax_choice=softmax_choice,
            softmax_gain=softmax_gain,
            softmax_threshold=softmax_threshold,
            storage_prob=storage_prob,
            store_on_optimization=store_on_optimization,
            memory_decay_rate=memory_decay_rate,
            purge_by_field_weights=purge_by_field_weights,
            enable_learning=enable_learning,
            target_fields=target_fields,
            random_state=random_state,
            seed=seed,
            **kwargs,
        )

        self._validate_options_with_learning(
            use_gating_for_weighting,
            enable_learning,
            softmax_choice,
        )

        self._construct_pathways(
            memory_template=self.memory_template,
            memory_capacity=self.memory_capacity,
            normalize_memories=self.normalize_memories,
            softmax_choice=self.softmax_choice,
            softmax_gain=self.softmax_gain,
            softmax_threshold=self.softmax_threshold,
            storage_prob=self.storage_prob,
            memory_decay_rate=self.memory_decay_rate,
            learn_field_weights=self.learn_field_weights,
            enable_learning=self.enable_learning,
            use_gating_for_weighting=self._use_gating_for_weighting,
            context=Context(source=ContextFlags.COMMAND_LINE, string="FROM EMComposition"),
        )

        self._assign_learning_attributes()
        self._assign_conditions()
        self._assign_node_roles()
        self._assign_attributes_for_show_graph()

        memory = self.memory
        if memory is not None and not np.any([
            np.any([memory[i][j] for i in range(self.memory_capacity)])
            for j in range(self.num_keys)
        ]):
            warnings.warn(
                f"Memory initialized with at least one key field that has all zeros; "
                f"a divide by zero can occur if 'normalize_memories' is True. "
                f"Use 'memory_fill' with non-zero values to avoid this."
            )

    # *****************************************************************************************************************
    # *********************************** Memory Construction Methods **************************************************
    # *****************************************************************************************************************

    def _validate_memory_specs(
        self,
        memory_template,
        memory_capacity,
        memory_fill,
        field_weights,
        field_names,
        name,
        learn_field_weights,
    ):
        if isinstance(memory_template, tuple):
            num_fields = memory_template[1] if len(memory_template) == 3 else memory_template[0]
            num_entries = memory_template[0] if len(memory_template) == 3 else memory_capacity
        elif isinstance(memory_template, (list, np.ndarray)):
            num_entries, num_fields = self._parse_memory_shape(memory_template)
        else:
            raise EMCompositionError(
                f"Unrecognized specification for the 'memory_template' arg ({memory_template}) of {name}."
            )

        if not isinstance(memory_template, tuple) and num_entries > 1:
            for entry in memory_template:
                if not (
                    len(entry) == num_fields
                    and np.all([len(entry[i]) == len(memory_template[0][i]) for i in range(num_fields)])
                ):
                    raise EMCompositionError(
                        f"The 'memory_template' arg for {name} must have the same shape for all entries."
                    )

        if not (
            isinstance(memory_fill, (int, float))
            or (
                isinstance(memory_fill, tuple)
                and len(memory_fill) == 2
                and all(isinstance(item, (int, float)) for item in memory_fill)
            )
        ):
            raise EMCompositionError(
                f"The 'memory_fill' arg ({memory_fill}) specified for {name} "
                f"must be a float, int, or length-2 tuple of numbers."
            )

        if isinstance(learn_field_weights, list) and len(learn_field_weights) != num_fields:
            raise EMCompositionError(
                f"The number of items ({len(learn_field_weights)}) in the "
                f"'learn_field_weights' arg for {name} must match the number "
                f"of fields in memory ({num_fields})."
            )

        if field_weights is not None:
            field_weights = np.atleast_1d(field_weights)
            if len(field_weights) > 1 and len(field_weights) != num_fields:
                raise EMCompositionError(
                    f"The number of items ({len(field_weights)}) in the 'field_weights' arg "
                    f"for {name} must match the number of fields in memory ({num_fields})."
                )
            if all([fw is None for fw in field_weights]):
                raise EMCompositionError(
                    f"The entries in 'field_weights' arg for {name} can't all be 'None' "
                    f"since that will preclude the construction of any keys."
                )
            if not any(field_weights):
                warnings.warn(
                    f"All of the entries in the 'field_weights' arg for {name} "
                    f"are either None or set to 0; this will result in no retrievals "
                    f"unless/until one or more of them are changed to a positive value."
                )
            elif any([fw == 0 for fw in field_weights if fw is not None]):
                warnings.warn(
                    f"Some of the entries in the 'field_weights' arg for {name} "
                    f"are set to 0; those fields will be ignored during retrieval "
                    f"unless/until they are changed to a positive value."
                )

        if field_names and len(field_names) != num_fields:
            raise EMCompositionError(
                f"The number of items ({len(field_names)}) in the 'field_names' arg for {name} "
                f"must match the number of fields ({num_fields})."
            )

    def _parse_memory_template(self, memory_template, memory_capacity, memory_fill):
        def _construct_entries(entry_template, num_entries, memory_fill=None):
            if isinstance(memory_fill, tuple):
                entries = [
                    [
                        np.full(
                            len(field),
                            np.random.uniform(memory_fill[1], memory_fill[0], len(field)),
                        ).tolist()
                        for field in entry_template
                    ]
                    for _ in range(num_entries)
                ]
            else:
                if memory_fill is None:
                    entry = entry_template
                else:
                    entry = [np.full(len(field), memory_fill).tolist() for field in entry_template]
                entries = [np.array(entry, dtype=object) for _ in range(num_entries)]

            return np.array(np.array(entries, dtype=object), dtype=object)

        if isinstance(memory_template, tuple):
            if len(memory_template) == 2:
                memory_capacity = memory_capacity or self.defaults.memory_capacity
                memory = _construct_entries(np.full(memory_template, 0), memory_capacity, memory_fill)
            else:
                if memory_capacity and memory_template[0] != memory_capacity:
                    raise EMCompositionError(
                        f"The first item ({memory_template[0]}) of 'memory_template' does not match "
                        f"'memory_capacity' ({memory_capacity})."
                    )
                memory_capacity = memory_template[0]
                memory = _construct_entries(np.full(memory_template[1:], 0), memory_capacity, memory_fill)
        else:
            num_entries, _ = self._parse_memory_shape(memory_template)

            if num_entries == 1:
                memory_capacity = memory_capacity or self.defaults.memory_capacity
                if any([np.array(field).any() for field in memory_template]):
                    memory_fill = None
                memory = _construct_entries(memory_template, memory_capacity, memory_fill)
            else:
                if not any(list(np.array(memory_template, dtype=object).flat)):
                    memory = _construct_entries(memory_template[0], memory_capacity, memory_fill)
                else:
                    memory_capacity = memory_capacity or num_entries
                    if num_entries > memory_capacity:
                        raise EMCompositionError(
                            f"The number of entries ({num_entries}) specified in 'memory_template' exceeds "
                            f"'memory_capacity' ({memory_capacity})."
                        )
                    num_entries_needed = memory_capacity - len(memory_template)
                    remaining_entries = _construct_entries(memory_template[0], num_entries_needed, memory_fill)
                    memory = (
                        np.concatenate((np.array(memory_template, dtype=object), remaining_entries))
                        if num_entries_needed
                        else np.array(memory_template, dtype=object)
                    )

        self.entry_template = memory[0]
        return memory, memory_capacity

    def _parse_fields(
        self,
        fields,
        field_names,
        field_weights,
        learn_field_weights,
        learning_rate,
        normalize_field_weights,
        concatenate_queries,
        normalize_memories,
        target_fields,
        name,
    ):
        def _parse_fields_dict(fields_dict, num_fields):
            if len(fields_dict) != num_fields:
                raise EMCompositionError(
                    f"The number of entries ({len(fields_dict)}) in the dict specified in the 'fields' arg "
                    f"of '{name}' does not match the number of fields in its memory ({num_fields})."
                )

            parsed_names = [None] * num_fields
            parsed_weights = [None] * num_fields
            parsed_learn = [None] * num_fields
            parsed_targets = [None] * num_fields

            for i, field_name in enumerate(fields_dict):
                parsed_names[i] = field_name
                spec = fields_dict[field_name]
                if isinstance(spec, (tuple, list)):
                    parsed_weights[i], parsed_learn[i], parsed_targets[i] = spec
                elif isinstance(spec, dict):
                    parsed_weights[i] = spec[FIELD_WEIGHT]
                    parsed_learn[i] = spec[LEARN_FIELD_WEIGHT]
                    parsed_targets[i] = spec[TARGET_FIELD]
                else:
                    raise EMCompositionError(
                        f"Unrecognized specification for field '{field_name}' in 'fields' for '{name}'."
                    )

            return parsed_names, parsed_weights, parsed_learn, parsed_targets

        self.num_fields = len(self.entry_template)

        if isinstance(learning_rate, dict):
            raise EMCompositionError(
                f"The 'learning_rate' arg for '{name}' is specified as a dict, "
                f"which is not supported for an EMComposition;  "
                f"use either its 'fields' arg or its 'learn_field_weights' arg instead."
            )

        if fields:
            if any([field_names, field_weights, learn_field_weights, target_fields]):
                warnings.warn(
                    f"The 'fields' arg for '{name}' was specified, so any of the "
                    f"'field_names', 'field_weights',  'learn_field_weights' or "
                    f"'target_fields' args will be ignored."
                )
            field_names, field_weights, learn_field_weights, target_fields = _parse_fields_dict(
                fields,
                self.num_fields,
            )

        if field_weights is None:
            if len(self.entry_template) == 1:
                field_weights = [1]
            else:
                field_weights = [1] * self.num_fields
                field_weights[-1] = None

        field_weights = np.atleast_1d(field_weights)

        if normalize_field_weights and not all([fw == 0 for fw in field_weights if fw is not None]):
            weights_for_sum = [fw if fw is not None else 0 for fw in field_weights]
            denominator = np.sum(weights_for_sum) or 1
            parsed_field_weights = [
                fw / denominator if fw is not None else None
                for fw in field_weights
            ]
        else:
            parsed_field_weights = field_weights

        if len(field_weights) == 1 and self.num_fields > 1:
            parsed_field_weights = np.repeat(parsed_field_weights, self.num_fields)

        individually_specified = True
        if not is_iterable(learn_field_weights) and learn_field_weights in {None, True, False}:
            learn_field_weights = [learn_field_weights] * len(parsed_field_weights)
            individually_specified = False

        if isinstance(learn_field_weights, (list, tuple, np.ndarray)):
            learn_field_weights = list(learn_field_weights)
            for i, (fw, lfw) in enumerate(zip(parsed_field_weights, learn_field_weights)):
                if fw is None:
                    if lfw and individually_specified:
                        warnings.warn(
                            f"A learning_rate was specified for field '{field_names[i] if field_names else i}' "
                            f"in the 'learn_field_weights' arg for '{name}', "
                            f"but it is not allowed for value fields; it will be ignored."
                        )
                    learn_field_weights[i] = False
                elif lfw in {None, True}:
                    learn_field_weights[i] = learning_rate or lfw
        else:
            raise EMCompositionError(
                f"PROGRAM ERROR: learn_field_weights ({learn_field_weights}) is not a valid specification."
            )

        parsed_field_names = field_names.copy() if field_names is not None else None

        self.key_indices = [i for i, fw in enumerate(parsed_field_weights) if fw is not None]
        self.value_indices = [i for i, fw in enumerate(parsed_field_weights) if fw is None]
        self.num_keys = len(self.key_indices)
        self.num_values = len(self.value_indices)

        if parsed_field_names:
            self.key_names = [parsed_field_names[i] for i in self.key_indices]
            self.value_names = [parsed_field_names[i] for i in self.value_indices]
        else:
            self.key_names = [f"{i}" for i in range(self.num_keys)] if self.num_keys > 1 else ["KEY"]
            self.value_names = (
                [f"{i} [VALUE]" for i in range(self.num_values)]
                if self.num_values > 1
                else (["VALUE"] if self.num_values == 1 else [])
            )
            parsed_field_names = self.key_names + self.value_names

        user_specified_concatenate_queries = concatenate_queries or False
        key_weights = [weight for weight in parsed_field_weights if weight is not None]
        concatenate_queries = (
            user_specified_concatenate_queries
            and self.num_keys > 1
            and all(np.all(key_weight == key_weights[0]) for key_weight in key_weights)
            and normalize_memories
        )
        if user_specified_concatenate_queries and not concatenate_queries:
            if self.num_keys == 1:
                error_msg = "there is only one key"
                correction_msg = ""
            elif not all(np.all(key_weight == key_weights[0]) for key_weight in key_weights):
                error_msg = f"field weights ({field_weights}) are not all equal"
                correction_msg = " To use concatenation, remove `field_weights` specification or make them all the same."
            elif not normalize_memories:
                error_msg = "normalize_memories is False"
                correction_msg = " To use concatenation, set normalize_memories to True."
            else:
                error_msg = "it is not supported"
                correction_msg = ""
            warnings.warn(
                f"The 'concatenate_queries' arg for '{name}' is True but {error_msg}; "
                f"concatenation will be ignored.{correction_msg}"
            )

        if target_fields is None:
            target_fields = [True] * self.num_fields

        self.learning_rate = learning_rate

        for i, field_name, weight, learn_weight, target in zip(
            range(self.num_fields),
            parsed_field_names,
            parsed_field_weights,
            learn_field_weights,
            target_fields,
        ):
            self.fields.append(
                Field(
                    name=field_name,
                    index=i,
                    type=FieldType.KEY if weight is not None else FieldType.VALUE,
                    weight=weight,
                    learn_weight=learn_weight,
                    target=target,
                )
            )

        return (
            parsed_field_names,
            parsed_field_weights,
            learn_field_weights,
            target_fields,
            concatenate_queries,
        )

    def _parse_memory_shape(self, memory_template):
        memory_template_dim = np.array(memory_template, dtype=object).ndim

        if memory_template_dim == 1 or all(isinstance(item, (int, float)) for item in memory_template[0]):
            fields_equal_length = all(len(field) == len(memory_template[0]) for field in memory_template)
        else:
            fields_equal_length = all(len(field) == len(memory_template[0]) for field in memory_template[0])

        single_entry = (
            ((memory_template_dim == 1) and not fields_equal_length)
            or ((memory_template_dim == 2) and fields_equal_length)
        )
        num_entries = 1 if single_entry else len(memory_template)
        num_fields = len(memory_template) if single_entry else len(memory_template[0])
        return num_entries, num_fields

    # *****************************************************************************************************************
    # *********************************** Nodes and Pathway Construction Methods ***************************************
    # *****************************************************************************************************************

    def _construct_pathways(
        self,
        memory_template,
        memory_capacity,
        normalize_memories,
        softmax_choice,
        softmax_gain,
        softmax_threshold,
        storage_prob,
        memory_decay_rate,
        learn_field_weights,
        enable_learning,
        use_gating_for_weighting,
        context,
    ):
        self._construct_input_nodes()
        self._construct_concatenate_queries_node()
        self._construct_field_memory_nodes(
            memory_template,
            memory_capacity,
            normalize_memories,
            storage_prob,
            memory_decay_rate,
        )
        self._construct_concatenated_memory_node(
            memory_template,
            normalize_memories,
            storage_prob,
            memory_decay_rate,
        )
        self._construct_field_weight_nodes()
        self._construct_weighted_scores_nodes()
        self._construct_combined_scores_node(memory_capacity, softmax_gain, softmax_threshold, softmax_choice)
        self._construct_softmax_gain_control_node(softmax_gain)
        self._construct_retrieved_nodes()

        self._field_index_map = {
            node: field.index
            for field in self.fields
            for node in field.nodes
        }
        self._field_index_map.update({
            proj: field.index
            for field in self.fields
            for proj in field.projections
        })

        # EM2 BREADCRUMB: THIS NEED TO DEAL WITH MULTIPLE PROJECTIONS BETWEEN MEMORY NODES AND COMBINED_SCORES NODE
        if not self.enable_learning:
            self.add_nodes(self.input_nodes, context=context)
            if self.concatenate_queries_node:
                self.add_node(self.concatenate_queries_node, context=context)
            self.add_nodes(self.field_memory_nodes, context=context)
            if self.concatenated_memory_node:
                self.add_node(self.concatenated_memory_node, context=context)
            self.add_nodes(self.field_weight_nodes + self.weighted_scores_nodes, context=context)
            self.add_nodes([self.combined_scores_node] + self.retrieved_nodes, context=context)
            if self.softmax_gain_control_node:
                self.add_node(self.softmax_gain_control_node, context=context)
            self._add_pathway_projections(context)
            return

        for field in self.fields:
            self.add_linear_processing_pathway([field.input_node,
                                                field.memory_node])

        if self.concatenate_queries:
            for field in self.key_fields:
                self.add_linear_processing_pathway([field.input_node,
                                                    self.concatenate_queries_node])
            self.add_linear_processing_pathway([self.concatenate_queries_node,
                                                self.concatenated_memory_node,
                                                self.combined_scores_node])

        elif self.num_keys == 1:
            self.add_linear_processing_pathway([self.key_fields[0].memory_node,
                                                self.combined_scores_node])
        else:
            for field in self.key_fields:
                pathway = [field.memory_node,
                           self.combined_scores_node]
                if field.weighted_scores_node:
                    pathway.insert(1, field.weighted_scores_node)
                self.add_linear_processing_pathway(pathway)

        for field in self.fields:
            self.add_linear_processing_pathway([self.combined_scores_node,
                                                field.memory_node,
                                                field.retrieved_node])
            # EM2 BREADCRUMB:
            # self.add_projections([self.combined_scores_node.efferents])

        if self.softmax_gain_control_node:
            self.add_node(self.softmax_gain_control_node, context=context)

        for field in self.key_fields:
            if field.weight_node and field.weighted_scores_node:
                self.add_linear_processing_pathway([
                    field.weight_node,
                    field.weighted_scores_node])

        # EM2 BREADCRUMB:
        #     HACK TO DEAL WITH FAILURE OF composition.add_projection() to handle multiple projections between mechs
        for proj in (self.combined_scores_node.path_afferents + self.combined_scores_node.efferents):
            if proj not in self.projections:
                self.add_projection(proj, context=context)
        self._add_pathway_projections(context)

    def _construct_input_nodes(self):
        for field in self.key_fields:
            field.input_node = ProcessingMechanism(name=f"{field.name} [QUERY]",
                                                   input_shapes=len(self.entry_template[field.index]))
            field.type = FieldType.KEY

        for field in self.value_fields:
            field.input_node = ProcessingMechanism(name=f"{field.name} [VALUE]",
                                                   input_shapes=len(self.entry_template[field.index]))
            field.type = FieldType.VALUE

    def _construct_concatenate_queries_node(self):
        if not self.concatenate_queries:
            self.concatenate_queries_node = None
            self.concatenated_memory_node = None
            return

        self.concatenate_queries_node = ProcessingMechanism(
            name=CONCATENATE_QUERIES_NAME,
            function=Concatenate,
            input_ports=[
                {
                    NAME: "CONCATENATE",
                    INPUT_SHAPES: len(field.input_node.output_port.value),
                    PROJECTIONS: MappingProjection(
                        name=f"{field.name} to CONCATENATE",
                        sender=field.input_node.output_port,
                        matrix=IDENTITY_MATRIX,
                    ),
                }
                for field in self.key_fields
            ],
        )
        for field, proj in zip(self.key_fields, self.concatenate_queries_node.path_afferents):
            field.concatenation_projection = proj

    def _construct_field_memory_nodes(
        self,
        memory_template,
        memory_capacity,
        normalize_memories,
        storage_prob,
        memory_decay_rate,
    ):

        for field in self.fields:
            key_len = 1 if is_numeric_scalar(field.query.squeeze()) else len(field.query.squeeze())
            field_memory = np.array(memory_template[:, field.index].tolist()).astype(float)

            field.memory_node = ExternalMemoryMechanism(
                field_type = field.type,
                field_shape = len(self.entry_template[field.index]),
                field_memory = field_memory,
                decay_rate = memory_decay_rate,
                storage_prob = storage_prob,
                scores_metric = L0 if key_len == 1 else DOT_PRODUCT,
                normalize_memories = True if key_len == 1 else normalize_memories,
                name=f"{field.name}{FIELD_MEMORY_AFFIX}",
            )

            field.query_projection = MappingProjection(
                sender=field.input_node,
                receiver=field.memory_node.input_ports[QUERY],
                matrix=IDENTITY_MATRIX,
                name=f"{field.name} QUERY to FIELD MEMORY",
            )

    def _construct_concatenated_memory_node(
        self,
        memory_template,
        normalize_memories,
        storage_prob,
        memory_decay_rate,
    ):
        if not self.concatenate_queries:
            self.concatenated_memory_node = None
            return

        concatenated_memory = np.array([
            np.concatenate([entry[field.index] for field in self.key_fields])
            for entry in memory_template
        ]).astype(float)
        key_len = len(self.entry_template[self.key_fields[0].index])

        self.concatenated_memory_node = ExternalMemoryMechanism(
            field_type=FieldType.KEY,
            field_shape=concatenated_memory.shape[1],
            field_memory=concatenated_memory,
            decay_rate=memory_decay_rate,
            storage_prob=storage_prob,
            scores_metric=L0 if key_len == 1 else DOT_PRODUCT,
            normalize_memories=True if key_len == 1 else normalize_memories,
            name=f"{CONCATENATED}{FIELD_MEMORY_AFFIX}",
        )
        self.concatenated_query_projection = MappingProjection(
            sender=self.concatenate_queries_node,
            receiver=self.concatenated_memory_node.input_ports[QUERY],
            matrix=IDENTITY_MATRIX,
            name=f"{CONCATENATE_QUERIES_NAME} to {CONCATENATED}{FIELD_MEMORY_AFFIX}",
        )

    def _construct_field_weight_nodes(self):
        if self.num_keys <= 1 or self.concatenate_queries:
            return

        for field in self.key_fields:
            name = f"{field.name}{WEIGHT_AFFIX}"
            variable = np.array(field.weight)
            params = {DEFAULT_INPUT: DEFAULT_VARIABLE}

            field.weight_node = ProcessingMechanism(
                name=name,
                input_ports={
                    NAME: "FIELD_WEIGHT",
                    VARIABLE: variable,
                    PARAMS: params,
                },
            )

    def _construct_weighted_scores_nodes(self):
        if self.num_keys <= 1 or self.concatenate_queries:
            return

        for field in self.key_fields:
            field.weighted_scores_node = ProcessingMechanism(
                name=field.name + WEIGHTED_SCORES_AFFIX,
                default_variable=[
                    np.zeros(self.memory_capacity),
                    np.zeros(self.memory_capacity),
                ],
                input_ports=[
                    {
                        PROJECTIONS: MappingProjection(
                            name=f"{field.name} {SCORES} to {WEIGHTED_SCORES_NODE_NAME}",
                            sender=field.memory_node.output_ports[SCORES],
                            matrix=IDENTITY_MATRIX,
                        )
                    },
                    {
                        PROJECTIONS: MappingProjection(
                            name=f"{field.name} {WEIGHT} to {WEIGHTED_SCORES_NODE_NAME}",
                            sender=field.weight_node,
                            matrix=FULL_CONNECTIVITY_MATRIX,
                        )
                    }
                ],
                output_ports={NAME: WEIGHTED_SCORES,
                              VARIABLE: (OWNER_VALUE,0)},
                function=LinearCombination(operation=PRODUCT),
            )
            field.scores_projection = field.weighted_scores_node.path_afferents[0]
            field.weight_projection = field.weighted_scores_node.path_afferents[1]

    def _construct_combined_scores_node(self, memory_capacity, softmax_gain, softmax_threshold,
                                                 softmax_choice):
        """Construct combined_scores_node
        This is constructed even if num_keys == 1, since it computes the softmax over the scores
        IMPLEMENTATION NOTE:  This plays the same role as the softmax_node in emcomposition_proj.py
        """

        if softmax_choice == ARG_MAX:
            softmax_choice = ARG_MAX_INDICATOR
        initial_softmax_gain = 1.0 if softmax_gain == CONTROL else softmax_gain
        softmax_function = SoftMax(gain=initial_softmax_gain,
                                   mask_threshold=softmax_threshold,
                                   output=softmax_choice,
                                   adapt_entropy_weighting=.95)

        # Construct combined_scores_function
        def _combined_scores_function(variable, gain=initial_softmax_gain):
            """Return softmax over combined scores, and index of minimum norm over combined norms
            variable[0] = scores of memory Nodes combined by hadamard addition in the COMBINED_SCORES input_port
            variable[1] = norms of memory Nodes combined by hadamard addition in the COMBINED_NORMS input_port
            """
            assert len(variable) == 2, \
                (f"PROGRAM ERROR: expected variable with 2 items for combined_scores_function; got {len(variable)}")
            return softmax_function(variable[0], params={GAIN: gain}), int(np.argmin(variable[1]))

        def _gen_pytorch_fct(device, context):
            """Return pytorch version of function"""
            # EM2 BREADCRUMB: CONTEXT execution_id NEEDS TO BE SET TO None,
            #                 SINCE _gen_pytorch_fct IS CALLED IN execution context
            #                 BUT SoftMax Function WAS CONSTRUCTED DURING __init__
            #                 AND SO ITS PARAMETERS HAVE NO VALUES FOR execution_id
            #                 ?? COULD BE DUE TO ORDER OF CALLS TO _gen_pytorch_fct IN PytorchFunctionWrapper??
            #                 POTENTIAL PROBLEM: WHEN FUNCTION IS CALLED IN EXECUTION CONTEXT,
            #                    WILL SOFTMAX FUNCTION PARAMS HAVE VALUES FOR CURRENT CONTEXT OR JUST USE None?
            local_context = copy.copy(context)
            local_context.execution_id = None
            softmax_func = softmax_function._gen_pytorch_fct(device, local_context)
            def func(variable):
                scores = variable[:, :, 0, ...]
                norms = variable[:, :, 1, ...]
                softmax_scores = softmax_func(scores)
                weakest_memory_idx = torch.argmin(norms, dim=-1, keepdim=True).to(dtype=softmax_scores.dtype)
                return [[[softmax_scores[b, s, ...], weakest_memory_idx[b, s, ...]]
                         for s in range(softmax_scores.shape[1])]
                        for b in range(softmax_scores.shape[0])]
            return func

        combined_scores_function = UserDefinedFunction(_combined_scores_function,
                                                       default_variable=[np.zeros(memory_capacity),
                                                                         np.zeros(memory_capacity)],
                                                       pytorch_function_generator =_gen_pytorch_fct
                                                       )
        # combined_scores_function._gen_pytorch_fct = _gen_pytorch_fct

        field_weighting = self.num_keys > 1 and not self.concatenate_queries
        assert (self.weighted_scores_nodes and self.field_weight_nodes) if field_weighting else not field_weighting, \
            (f"PROGRAM ERROR: Mismatch between num_keys and presence of weighted_scores_nodes and/or field_weight_nodes")

        if self.concatenate_queries:
            scores_inputs = [self.concatenated_memory_node.output_ports[SCORES]]
            scores_input_names = [CONCATENATE_QUERIES_NAME]
        else:
            scores_inputs = [(field.weighted_scores_node.output_ports[WEIGHTED_SCORES] if field_weighting
                              else field.memory_node.output_ports[SCORES])
                              for field in self.key_fields]
            scores_input_names = [field.name for field in self.key_fields]
        # EM2 BREADCRUMB: THIS WEIGHTS THE NORMS, WHICH IS PROBABLY NOT CORRECT:
        # norms_inputs = [(field.weighted_scores_node if field.type == FieldType.KEY and field_weighting
        #                  else field.memory_node).output_ports[NORMS]
        #                 for field in self.fields]
        norms_inputs = [field.memory_node.output_ports[NORMS] for field in self.fields]
        self.combined_scores_node = ProcessingMechanism(
            name=COMBINED_SCORES_NODE_NAME,
            input_ports=[
                {NAME:SCORES,
                 INPUT_SHAPES: memory_capacity,
                 PROJECTIONS: [
                     MappingProjection(
                         sender=source,
                         matrix=IDENTITY_MATRIX,
                         name=f"{'WEIGHTED' if field_weighting else ''} {SCORES} for {scores_input_names[i]}")
                              # f" to {COMBINED_SCORES_NODE_NAME}")
                     for i, source in enumerate(scores_inputs)]},
                {NAME:NORMS,
                 INPUT_SHAPES: memory_capacity,
                 PROJECTIONS: [
                     MappingProjection(
                         sender=source,
                         matrix=IDENTITY_MATRIX,
                         name=f"{'WEIGHTED' if field_weighting else ''} {NORMS} for {self.fields[i].name}")
                              # f" to {COMBINED_SCORES_NODE_NAME}")
                     for i, source in enumerate(norms_inputs)]},
            ],
            output_ports=[{NAME:COMBINED_SCORES, VARIABLE: (OWNER_VALUE, 0)},
                          {NAME:COMBINED_NORMS, VARIABLE: (OWNER_VALUE, 1)}],
            function=combined_scores_function
        )

        # EM2 BREADCRUMB: MAKE THIS SPECIFIC TO SCORES, AND ADD SIMILAR LOOP FOR NORMS
        if self.concatenate_queries:
            self.concatenated_scores_projection = next(
                proj for proj in self.combined_scores_node.path_afferents
                if proj.sender is self.concatenated_memory_node.output_ports[SCORES]
            )
        for field in self.fields:
            # Assign Projections from memory_nodes to combined_scores nodes to relevant attributes of field
            if field.type == FieldType.KEY and not self.concatenate_queries:
                # EM2 BREADCRUMB: NEED TO GET AFFERENT FROM field_weighted_scores NODE IF field_weighting
                scores_proj = next(proj for proj in self.combined_scores_node.path_afferents
                                   if proj.sender is (field.weighted_scores_node.output_ports[WEIGHTED_SCORES]
                                                      if field_weighting else field.memory_node.output_ports[SCORES]))
                field.weighted_scores_projection = scores_proj
            norms_proj = next(proj for proj in self.combined_scores_node.path_afferents
                              if proj.sender is field.memory_node.output_ports[NORMS])
            field.weighted_norms_projection = norms_proj

            # EM2 BREADCRUMB: NEED TO EXPLICITLY ADD PROJECTIONS TO COMPOSITION,
            #     SINCE THE COMBINED_SCORES ONE DOES NOT SEEM TO BE GETTING ADDED (BLOCKED BY COMBINED_NORMS ONE?)
            # Assign Projections from combined_scores nodes back to COMBINED_SCORES input_ports of field_memory_nodes
            # Note: this has to be constructed here, as it depends on the combined_scores_node being constructed first
            field.combined_scores_projection = MappingProjection(
                sender=self.combined_scores_node.output_ports[COMBINED_SCORES],
                feedback=True,
                receiver=field.memory_node.input_ports[COMBINED_SCORES],
                matrix=IDENTITY_MATRIX,
                name=f"{COMBINED_SCORES_NODE_NAME} to {field.name} COMBINED_SCORES",
            )
            # Assign Projections from combined_scores nodes back to COMBINED_NORMS input_ports of field_memory_nodes
            # Note: this has to be constructed here, as it depends on the combined_scores_node being constructed first
            field.combined_norms_projection = MappingProjection(
                sender=self.combined_scores_node.output_ports[COMBINED_NORMS],
                feedback=True,
                receiver=field.memory_node.input_ports[COMBINED_NORMS],
                matrix=IDENTITY_MATRIX,
                name=f"{COMBINED_SCORES_NODE_NAME} to {field.name} COMBINED_NORMS",
            )

        if self.concatenate_queries:
            self.concatenated_combined_scores_projection = MappingProjection(
                sender=self.combined_scores_node.output_ports[COMBINED_SCORES],
                feedback=True,
                receiver=self.concatenated_memory_node.input_ports[COMBINED_SCORES],
                matrix=IDENTITY_MATRIX,
                name=f"{COMBINED_SCORES_NODE_NAME} to {CONCATENATE_QUERIES_NAME} COMBINED_SCORES",
            )
            self.concatenated_combined_norms_projection = MappingProjection(
                sender=self.combined_scores_node.output_ports[COMBINED_NORMS],
                feedback=True,
                receiver=self.concatenated_memory_node.input_ports[COMBINED_NORMS],
                matrix=IDENTITY_MATRIX,
                name=f"{COMBINED_SCORES_NODE_NAME} to {CONCATENATE_QUERIES_NAME} COMBINED_NORMS",
            )


    def _construct_softmax_gain_control_node(self, softmax_gain):
        node = None
        if softmax_gain == CONTROL:
            node = ControlMechanism(
                name="SOFTMAX GAIN CONTROL",
                monitor_for_control=self.combined_scores_node or self.key_fields[0].memory_node,
                control_signals=[(GAIN, self.combined_scores_node)],
                function=get_softmax_gain,
            )
        self.softmax_gain_control_node = node

    def _construct_retrieved_nodes(self):
        for field in self.fields:
            field.retrieved_node = ProcessingMechanism(
                name=field.name + RETRIEVED_AFFIX,
                input_ports={
                    INPUT_SHAPES: len(field.input_node.variable[0]),
                    PROJECTIONS: MappingProjection(
                        sender=field.memory_node.output_ports[RETRIEVED],
                        matrix=IDENTITY_MATRIX,
                        name=f"{field.name} RETRIEVED to OUTPUT",
                    ),
                },
            )
            field.retrieved_projection = field.retrieved_node.path_afferents[0]

    def _assign_conditions(self):


        for field in self.fields:

            # Input and weight nodes should run only once, at the beginning of the trial
            # EM2 BREADCRUMB: DOES THIS CONDITION NEED A TimeScale SPECIFICATION (I.E., TRIAL)?
            self.scheduler.add_condition(field.input_node, BeforeNCalls(field.input_node, 1))
            if field.weight_node is not None:
                self.scheduler.add_condition(field.weight_node, BeforeNCalls(field.weight_node, 1))

            # Field-memory mechanisms must run once after inputs, then again after RETRIEVE.
            self.scheduler.add_condition(
                field.memory_node,
                Any(All(AfterNCalls(field.input_node, 1),
                        BeforeNCalls(self.combined_scores_node, 1)),
                    All(AfterNCalls(self.combined_scores_node, 1),
                        BeforeNCalls(field.retrieved_node, 1)))
            )

            # Storage should be after RETRIEVAL
            field.memory_node.parameters.access_condition.set(
                conditions.AfterNCalls(self.combined_scores_node, 1),
                context=Context(source=ContextFlags.COMMAND_LINE, string="FROM EMComposition storage conditions"),
                override=True)


            # Retrieved nodes run only after both field-memory mechanisms have run twice.
            if self.concatenated_memory_node:
                self.scheduler.add_condition(
                    field.retrieved_node,
                    All(AfterNCalls(field.memory_node, 2),
                        AfterNCalls(self.concatenated_memory_node, 2))
                )
            else:
                self.scheduler.add_condition(field.retrieved_node, AfterNCalls(field.memory_node, 2))

        if self.concatenated_memory_node:
            self.scheduler.add_condition(
                self.concatenated_memory_node,
                Any(All(AfterNCalls(self.concatenate_queries_node, 1),
                        BeforeNCalls(self.combined_scores_node, 1)),
                    All(AfterNCalls(self.combined_scores_node, 1),
                        BeforeNCalls(self.retrieved_nodes[0], 1)))
            )
            self.concatenated_memory_node.parameters.access_condition.set(
                conditions.AfterNCalls(self.combined_scores_node, 1),
                context=Context(source=ContextFlags.COMMAND_LINE, string="FROM EMComposition storage conditions"),
                override=True)

        # # RETRIEVE runs only after both field-memory mechanisms have run once.
        # args = ([AfterNCalls(node, 1) for node in self.field_memory_nodes]
        #         + [BeforeNCalls(node, 2) for node in self.field_memory_nodes])
        # self.scheduler.add_condition(self.combined_scores_node, All(*args))

        # # Storage should be after RETRIEVAL
        # for field_memory_node in self.field_memory_nodes:
        #     field_memory_node.parameters.access_condition.set(
        #         conditions.AfterNCalls(self.combined_scores_node, 1),
        #         context=Context(source=ContextFlags.COMMAND_LINE, string="FROM EMComposition storage conditions"),
        #         override=True,
        #     )

        # # BREADCRUMB: NECESSARY??
        # # End the trial after all retrieved nodes have executed once.
        # args = [AfterNCalls(node, 1) for node in self.retrieved_nodes]
        # self.scheduler.termination_conds[TimeScale.TRIAL] = (All(*args))


    def _assign_node_roles(self):
        for node in self.field_weight_nodes:
            self.exclude_node_roles(node, NodeRole.INPUT)
        for node in self.value_input_nodes:
            self.exclude_node_roles(node, NodeRole.OUTPUT)
        if self.concatenate_queries_node:
            self.exclude_node_roles(self.concatenate_queries_node, NodeRole.OUTPUT)
        if self.concatenated_memory_node:
            self.exclude_node_roles(self.concatenated_memory_node, NodeRole.OUTPUT)
        self.exclude_node_roles(self.combined_scores_node, NodeRole.OUTPUT)


    def _assign_attributes_for_show_graph(self):
        for node in self.value_input_nodes:
            node.output_port.parameters.require_projection_in_composition.set(False, override=True)
        if self.concatenate_queries_node:
            self.concatenate_queries_node.output_port.parameters.require_projection_in_composition.set(False, override=True)
        if self.concatenated_memory_node:
            for output_port in self.concatenated_memory_node.output_ports:
                output_port.parameters.require_projection_in_composition.set(False, override=True)
        self.combined_scores_node.output_port.parameters.require_projection_in_composition.set(False, override=True)

    def _add_pathway_projections(self, context):
        projections = []
        for field in self.fields:
            projections.extend(field.projections)

        if self.concatenate_queries:
            projections.extend([
                self.concatenated_query_projection,
                self.concatenated_scores_projection,
                self.concatenated_combined_scores_projection,
                self.concatenated_combined_norms_projection,
            ])

        projections.extend(self.combined_scores_node.path_afferents)
        projections.extend(self.combined_scores_node.efferents)

        for proj in [proj for proj in projections if proj is not None]:
            if proj not in self.projections:
                self.add_projection(proj, context=context)

    def _assign_learning_attributes(self):
        self.execute_in_additional_optimizations = {}

        field_weight_projections = []
        for projection in self.projections:
            if projection.sender.owner in self.field_weight_nodes:
                field_weight_projections.append(projection)
            else:
                projection.learnable = False
                projection.learning_rate = False

        learn_field_weights = self.parameters.learn_field_weights.spec
        if not isinstance(learn_field_weights, (list, np.ndarray)):
            assert not self.enable_learning, (
                "PROGRAM ERROR: self.learn_field_weights is not a list, but should be by this point."
            )

        if (
            all(item is False for item in learn_field_weights)
            or len(self.query_input_nodes) == 1
        ):
            lr_dict = {}
            for projection in field_weight_projections:
                projection.learnable = False
                projection.learning_rate = False
                lr_dict[projection] = False
            self._enable_learning_warning_flag = True
        else:
            lr_dict = {}
            constructor_learning_rate = self.parameters.learning_rate.get(None)
            if not isinstance(constructor_learning_rate, dict):
                lr_dict[DEFAULT_LEARNING_RATE] = constructor_learning_rate

            for i, field in enumerate(self.fields):
                if field.type == FieldType.KEY and field.weight_node:
                    proj = field.weight_node.efferents[0]
                    if learn_field_weights[i] is False:
                        lr_dict[proj] = False
                        proj.learnable = False
                    elif is_numeric_scalar(learn_field_weights[i]):
                        lr_dict[proj] = learn_field_weights[i]
                    elif learn_field_weights[i] is None:
                        continue
                    else:
                        raise EMCompositionError(
                            f"PROGRAM ERROR: learning_rate for {field.name} "
                            f"({learn_field_weights[i]}) is not valid."
                        )

        self.parameters.learning_rate._set(lr_dict, context=Context(execution_id=None))

    def _validate_options_with_learning(self, use_gating_for_weighting, enable_learning, softmax_choice):
        if use_gating_for_weighting and enable_learning:
            warnings.warn(
                f"The 'enable_learning' option for '{self.name}' cannot be used with "
                f"'use_gating_for_weighting=True'; this will generate an error if learn() is called."
            )

        if softmax_choice in {ARG_MAX, PROBABILISTIC} and enable_learning:
            warnings.warn(
                f"The 'softmax_choice' arg of '{self.name}' is set to '{softmax_choice}' with "
                f"'enable_learning' set to True; this will generate an error if its "
                f"'learn' method is called. Set 'softmax_choice' to WEIGHTED_AVG before learning."
            )

    # *****************************************************************************************************************
    # ***************************************** Execution Methods ******************************************************
    # *****************************************************************************************************************

    @handle_external_context(fallback_default=True)
    def learn(
        self,
        *args,
        context: Optional[Context] = None,
        base_context: Context = Context(execution_id=None),
        skip_initialization: bool = False,
        **kwargs,
    ) -> list:
        if (
            not skip_initialization
            and (
                context is None
                or ContextFlags.SIMULATION_MODE not in context.runmode
            )
        ):
            self._initialize_from_context(context, base_context, override=False)

        softmax_choice = self.parameters.softmax_choice.get(context)
        use_gating_for_weighting = self._use_gating_for_weighting
        enable_learning = self.parameters.enable_learning.get(context)

        if use_gating_for_weighting and enable_learning:
            raise EMCompositionError(
                f"Field weights cannot be learned when 'use_gating_for_weighting' is True; "
                f"construct '{self.name}' with 'enable_learning=False'."
            )

        if self.concatenate_queries:
            raise EMCompositionError(
                "EMComposition does not support learning with 'concatenate_queries'='True'."
            )

        if softmax_choice in {ARG_MAX, PROBABILISTIC}:
            raise EMCompositionError(
                f"The ARG_MAX and PROBABILISTIC options for the 'softmax_choice' arg of '{self.name}' "
                f"cannot be used during learning; change to WEIGHTED_AVG."
            )

        if self._enable_learning_warning_flag and not self.is_nested:
            if len(self.query_input_nodes) == 1:
                warnings.warn(
                    f"The 'enable_learning' arg of '{self.name}' is True, but it has only one key, "
                    f"so field_weights and field-weight learning have no effect."
                )

        return super().learn(
            *args,
            context=context,
            base_context=base_context,
            skip_initialization=skip_initialization,
            **kwargs,
        )

    def _instantiate_input_dict(self, input_dict):
        """Override to determine — and respond appropriately -- if any KEY and/or VALUE fields are not specified.
        - If any KEY fields are missing, raise error
        - If any VALUE fields are missing, issue warning that the retrieved value will be stored with the specified KEY
        """
        if self.is_nested:
            # EM2 BREADCRUMB: NEED TESTS FOR THIS
            missing_query_nodes = [f"'{node.name}'" for node in self.query_input_nodes
                                   if self.input_CIM._get_source_node_for_input_CIM(node.input_port)]
            missing_value_nodes = [node for node in self.value_input_nodes
                                   if self.input_CIM._get_source_node_for_input_CIM(node.input_port)]
        else:
            missing_query_nodes = [f"'{node.name}'" for node in self.query_input_nodes if node not in input_dict]
            missing_value_nodes = [node for node in self.value_input_nodes if node not in input_dict]

        if missing_query_nodes:
            raise EMCompositionError(
                f"'inputs' argument of call to learn() method for '{self.name}' is missing entries "
                f"for the following query_input_nodes: {', '.join(missing_query_nodes)}")

        if missing_value_nodes:
            for field in [f for f in self.fields if f.input_node in missing_value_nodes]:
                field.input_node.value_input_specified = False
            missing_value_nodes_str = [f"'{node.name}'" for node in missing_value_nodes]
            plural = len(missing_value_nodes) > 1
            query_str = 'queries' if plural else 'query'
            key_str = 'keys' if plural else 'key'
            s = "s" if plural else ""
            their_its = 'their' if plural else 'its'
            entries = "entries" if plural else "an entry"
            warnings.warn(f"'inputs' argument of call to learn() method for '{self.name}' is missing {entries} "
                          f"for the following value_input_node{s}, so the retrieved value{s} will be stored with "
                          f"the specified {query_str} as {their_its} {key_str}: {', '.join(missing_value_nodes_str)}.")

        return super()._instantiate_input_dict(input_dict)

    def _get_execution_mode(self, execution_mode):
        if execution_mode is None:
            if self._warned_about_default_execution_mode is False:
                warnings.warn(
                    f"The execution_mode argument was not specified in learn() for {self.name}; "
                    f"ExecutionMode.PyTorch will be used by default."
                )
                self._warned_about_default_execution_mode = True
            execution_mode = ExecutionMode.PyTorch
        return execution_mode

    def _identify_output_nodes(self, context) -> list:
        target_fields = self.target_fields

        if target_fields is False:
            if self.enable_learning:
                warnings.warn(
                    f"The 'enable_learning' arg for {self.name} is True but 'target_fields' is False, "
                    f"so enable_learning will have no effect."
                )
            target_nodes = []
        elif target_fields is True:
            target_nodes = [node for node in self.retrieved_nodes]
        elif isinstance(target_fields, (list, tuple, np.ndarray)):
            target_nodes = [
                node for node in self.retrieved_nodes
                if target_fields[self.retrieved_nodes.index(node)]
            ]
        else:
            assert False, (
                f"PROGRAM ERROR: target_fields arg for {self.name}: {target_fields} "
                f"is neither True, False, nor a list of bools."
            )

        super()._identify_output_nodes(context)
        return target_nodes

    def infer_backpropagation_learning_pathways(self, execution_mode, context=None, base_context=None):
        return super().infer_backpropagation_learning_pathways(
            execution_mode,
            context=context,
            base_context=base_context,
        )

    def do_gradient_optimization(self, retain_in_pnl_options, context, optimization_num=None):
        # EM storage is field-local and executed by ExternalMemoryMechanism after retrieval.
        # Field-weight learning can be restored by calling super() once the PyTorch wrapper
        # supports ExternalMemoryMechanism as a differentiable memory component.
        pass

    def add_node(self, node, required_roles=None, context=None):
        if context is None:
            raise EMCompositionError(f"Nodes cannot be added to an {self.componentCategory}: ('{self.name}').")
        super().add_node(node, required_roles, context)

    def add_projection(self, *args, **kwargs):
        if CONTEXT not in kwargs or kwargs[CONTEXT] is None:
            raise EMCompositionError(f"Projections cannot be added to an {self.componentCategory}: ('{self.name}').")
        return super().add_projection(*args, **kwargs)

    # *****************************************************************************************************************
    # ***************************************** Properties *************************************************************
    # *****************************************************************************************************************

    @property
    def key_fields(self):
        return [field for field in self.fields if field.type == FieldType.KEY]

    @property
    def value_fields(self):
        return [field for field in self.fields if field.type == FieldType.VALUE]

    @property
    def input_nodes(self):
        return [field.input_node for field in self.fields]

    @property
    def query_input_nodes(self):
        return [field.input_node for field in self.key_fields]

    @property
    def value_input_nodes(self):
        return [field.input_node for field in self.value_fields]

    @property
    def field_memory_nodes(self):
        return [field.memory_node for field in self.fields]

    @property
    def memory_cycle_nodes(self):
        nodes = list(self.field_memory_nodes)
        if getattr(self, "concatenated_memory_node", None) is not None:
            nodes.append(self.concatenated_memory_node)
        return nodes

    @property
    def match_nodes(self):
        # Compatibility alias: the old "match_nodes" are now the key ExternalMemoryMechanisms.
        if self.concatenate_queries and getattr(self, "concatenated_memory_node", None) is not None:
            return [self.concatenated_memory_node]
        return [field.memory_node for field in self.key_fields]

    @property
    def field_weight_nodes(self):
        return [
            field.weight_node
            for field in self.key_fields
            if field.weight_node is not None
        ]

    @property
    def weighted_scores_nodes(self):
        return [
            field.weighted_scores_node
            for field in self.key_fields
            if field.weighted_scores_node is not None
        ]

    @property
    def retrieved_nodes(self):
        return [field.retrieved_node for field in self.fields]
