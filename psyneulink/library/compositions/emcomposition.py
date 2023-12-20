# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* EMComposition *************************************************
# TODO:
# - QUESTION:
#   - SHOULD differential of SoftmaxGainControl Node be included in learning?
#   - SHOULD MEMORY DECAY OCCUR IF STORAGE DOES NOT? CURRENTLY IT DOES NOT (SEE EMStorage Function)

# - FIX: NAMING
# - FIX: Concatenation:
# -      LLVM for function and derivative
# -      Add Concatenate to pytorchcreator_function
# -      Deal with matrix assignment in LearningProjection LINE 643
# -      Reinstate test for execution of Concatenate with learning in test_emcomposition (currently commented out)
# - FIX: Softmax Gain Control:
#        Test if it current works (they are added to Compostion but not in BackProp processing pathway)
#        Does backprop have to run through this if not learnable?
#        If so, need to add PNL Function, with derivative and LLVM and Pytorch implementations
# - FIX: WRITE MORE TESTS FOR EXECUTION, WARNINGS, AND ERROR MESSAGES
#         - learning (with and without learning field weights
#         - 3d tuple with first entry != memory_capacity if specified
#         - list with number of entries > memory_capacity if specified
#         - input is added to the correct row of the matrix for each key and value for
#                for non-contiguous keys (e.g, field_weights = [1,0,1]))
#         - explicitly that storage occurs after retrieval
# - FIX: WARNING NOT OCCURRING FOR Normalize ON ZEROS WITH MULTIPLE ENTRIES (HAPPENS IF *ANY* KEY IS EVER ALL ZEROS)
# - FIX: IMPLEMENT LearningMechanism FOR RETRIEVAL WEIGHTS:
#        - what is learning_update: AFTER doing?  Use for scheduling execution of storage_node?
#        ?? implement derivative for concatenate
# - FIX: implement add_storage_pathway to handle addition of storage_node as learning mechanism
#        - in "_create_storage_learning_components()" assign "learning_update" arg
#          as BEORE OR DURING instead of AFTER (assigned to learning_enabled arg of LearningMechanism)
# - FIX: Add StorageMechanism LearningProjections to Composition? -> CAUSES TEST FAILURES; NEEDS INVESTIGATION
# - FIX: Thresholded version of SoftMax gain (per Kamesh)
# - FIX: DEAL WITH INDEXING IN NAMES FOR NON-CONTIGUOUS KEYS AND VALUES (reorder to keep all keys together?)
# - FIX: _import_composition:
#        - MOVE LearningProjections
#        - MOVE Condition? (e.g., AllHaveRun) (OR PUT ON MECHANISM?)
# - FIX: IMPLEMENT _integrate_into_composition METHOD THAT CALLS _import_composition ON ANOTHER COMPOSITION
# -      AND TRANSFERS RELEVANT ATTRIBUTES (SUCH AS MEMORY, query_input_nodeS, ETC., POSSIBLY APPENDING NAMES)
# - FIX: ADD Option to suppress field_weights when computing norm for weakest entry in EMStorageMechanism
# - FIX: GENERATE ANIMATION w/ STORAGE (uses Learning but not in usual way)
# - IMPLEMENT use OF multiple inheritance of EMComposition from AutoDiff and Composition

# - FIX: DOCUMENTATION:
#        - enable_learning vs. learning_field_weights
#        - USE OF EMStore.storage_location (NONE => LOCAL, SPECIFIED => GLOBAL)
#        - define "keys" and "values" explicitly
#        - define "key weights" explicitly as field_weights for all non-zero values
#        - make it clear that full size of memory is initialized (rather than "filling up" w/ use)
#        - write examples for run()
# - FIX: ADD NOISE (AND/OR SOFTMAX PROBABILISTIC RETRIEVAL MODE)
# - FIX: ?ADD add_memory() METHOD FOR STORING W/O RETRIEVAL, OR JUST ADD retrieval_prob AS modulable Parameter
# - FIX: CONFIDENCE COMPUTATION (USING SIGMOID ON DOT PRODUCTS) AND REPORT THAT (EVEN ON FIRST CALL)
# - FIX: ALLOW SOFTMAX SPEC TO BE A DICT WITH PARAMETERS FOR _get_softmax_gain() FUNCTION
# MISC:
# - WRITE TESTS FOR INPUT_PORT and MATRIX SPECS CORRECT IN LATEST BRANCHs
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

# - FIX: PSYNEULINK:
#      - TESTS:
#        - WRITE TESTS FOR DriftOnASphere variable = scalar, 2d vector or 1d vector of correct and incorrect lengths
#        - WRITE TESTS FOR LEARNING WITH LinearCombination of 1, 2 and 3 inputs
#
#      - COMPILATION:
#        - Remove CIM projections on import to another composition
#        - Autodiff support for IdentityFunction
#        - LinearMatrix to add normalization
#        - _store() method to assign weights to memory
#        - LLVM problem with ComparatorMechanism
#
#      - pytorchcreator_function:
#           SoftMax implementation:  torch.nn.Softmax(dim=0) is not getting passed correctly
#           Implement LinearCombination
#        - LinearMatrix Function:
#
#      - LEARNING - Backpropagation LearningFunction / LearningMechanism
#        - DOCUMENTATION:
#           - weight_change_matrix = gradient (result of delta rule) * learning_rate
#           - ERROR_SIGNAL is OPTIONAL (only implemented when there is an error_source specified)
#        - Backprop: (related to above?) handle call to constructor with default_variable = None
#        - WRITE TESTS FOR USE OF COVARIATES AND RELATED VIOLATIONS: (see ScratchPad)
#          - Use of LinearCombination with PRODUCT in output_source
#          - Use of LinearCombination with PRODUCT in InputPort of output_source
#                  - Construction of LearningMechanism with Backprop:
#        - MappingProjection / LearningMechanism:
#          - Add learning_rate parameter to MappingProjection (if learnable is True)
#          - Refactor LearningMechanism to use MappingProjection learning_rate specification if present
#        - CHECK FOR EXISTING LM ASSERT IN pytests
#
#      - AutodiffComposition:
#         - replace handling / flattening of nested compositions with Pytorch.add_module (which adds "child" modules)
#         - Check that error occurs for adding a controller to an AutodiffComposition
#         - Check that if "epochs" is not in input_dict for Autodiff, then:
#           - set to num_trials as default,
#           - leave it to override num_trials if specified (add this to DOCUMENTATION)
#        - Input construction has to be:
#           - same for Autodiff in Python mode and PyTorch mode
#               (NOTE: used to be that autodiff could get left in Python mode
#                      so only where tests for Autodiff happened did it branch)
#           - AND different from Composition (in Python mode)
#        - support use of pathway argument in Autodff
#        - the following format doesn't work for LLVM (see test_identicalness_of_input_types:
#           xor = pnl.AutodiffComposition(nodes=[input_layer,hidden_layer,output_layer])
#           xor.add_projections([input_to_hidden_wts, hidden_to_output_wts])
#          - DOCUMENTATION: execution_mode=ExecutionMode.Python allowed
#          - Add warning of this on initial call to learn()
#
#      - Composition:
#        - Add default_execution_mode attribute to allow nested Compositions to be executed in
#              different model than outer Composition
#        - _validate_input_shapes_and_expand_for_all_trials: consolidate with get_input_format()
#        - Generalize treatment of FEEDBACK specification:
      #        - FIX: ADD TESTS FOR FEEDBACK TUPLE SPECIFICATION OF Projection, DIRECT SPECIFICATION IN CONSTRUCTOR
#              - FIX: why aren't FEEDBACK_SENDER and FEEDBACK_RECEIVER roles being assigned when feedback is specified?
#        - add property that keeps track of warnings that have been issued, and suppresses repeats if specified
#        - add property of Composition that lists it cycles
#        - Add warning if termination_condition is trigged (and verbosePref is set)
#        - Addition of projections to a ControlMechanism seems too dependent on the order in which the
#              the ControlMechanism is constructed with respect to its afferents (if it comes before one,
#              the projection to it (i.e., for monitoring) does not get added to the Composition
# -      - IMPLEMENTATION OF LEARNING: NEED ERROR IF TRY TO CALL LEARN ON A COMPOSITION THAT HAS NO LEARNING MECHANISMS
#          INCLUDING IN PYTHON MODE??  OR JUST ALLOW IT TO CONSTRUCT THE PATHWAY AUTOMATICALLY?
#        - Change size argument in constructor to use standard numpy shape format if tupe, and PNL format if list
#        - Write convenience Function for returning current time from context
#             - requires it be called from execution within aComposition, error otherwise)
#             - takes argument for time scale (e.g., TimeScale.TRIAL, TimeScale.RUN, etc.)
#             - Add TimeMechanism for which this is the function, and can be configured to report at a timescale
#        - Add Composition.run_status attribute assigned a context flag, with is_preparing property that checks it
#               (paralleling handling of is_initializing)
#        - Allow set of lists as specification for pathways in Composition
#        - Add support for set notation in add_backpropagation_learning_pathway (to match add_linear_processing_pathway)
#             see ScratchPad: COMPOSITION 2 INPUTS UNNESTED VERSION: MANY-TO-MANY
#        - Make sure that shadow inputs (see InputPort_Shadow_Inputs) uses the same matrix as shadowed input.
#        - composition.add_backpropagation_learning_pathway(): support use of set notation for multiple nodes that
#        project to a single one.
#        - add LearningProjections executed in EXECUTION_PHASE to self.projections
#          and then remove MODIFIED 8/1/23 in _check_for_unused_projections
#        - Why can't verbosePref be set directly on a composition?
#        - Composition.add_nodes():
#           - should check, on each call to add_node, to see if one that has a releavantprojection and, if so, add it.
#           - Allow [None] as argument and treat as []
#        - IF InputPort HAS default_input = DEFAULT_VARIABLE,
#           THEN IT SHOULD BE IGNORED AS AN INPUT NODE IN A COMPOSITION
#        - Add use of dict in pathways specification to map outputs from a set to inputs of another set
#            (including nested comps)
#
#      - ShowGraph:  (show_graph)
#        - don't show INPUT/OUTPUT Nodes for nested Comps in green/red
#                (as they don't really receive input or generate output on a run
#        - show feedback projections as pink (shouldn't that already be the case?)
#        - add mode for showing projections as diamonds without show_learning (e.g., "show_projections")
#        - figure out how to get storage_node to show without all other learning stuff
#        - show 'operation' parameter for LinearCombination in show_node_structure=ALL
#        - specify set of nodes to show and only show those
#        - fix: show_learning=ALL (or merge from EM branch)
#
#      - ControlMechanism
#        - refactor ControlMechanism per notes of 11/3/21, including:
#                FIX: 11/3/21 - MOVE _parse_monitor_specs TO HERE FROM ObjectiveMechanism
#      - EpisodicMemoryMechanism:
#        - make storage_prob and retrieval_prob parameters linked to function
#        - make distance_field_weights a parameter linked to function
#
#      - LinearCombination Function:
#        - finish adding derivative (for if exponents are specified)
#        - remove properties (use getter and setter for Parameters)
#
#      - ContentAddressableMemory Function:
#           - rename "cue" -> "query"
#           - add field_weights as parameter of EM, and make it a shared_parameter ?as well as a function_parameter?

#     - DDM:
#        - make reset_stateful_function_when a Parameter and arg in constructor
#          and align with reset Parameter of IntegratorMechanism)
#
#    - FIX: BUGS:
#      - composition:
#          - If any MappingProjection is specified from nested node to outer node,
#            then direct projections are instantiated to the output_CIM of the outer comp, and the
#            nested comp is treated as OUTPUT Node of outer comp even if all its projections are to nodes in outer comp
#            LOOK IN add_projections? for nested comps
#      - composition (?add_backpropagation_learning_pathway?):
#           THIS FAILS:
#             comp = Composition(name='a_outer')
#             comp.add_backpropagation_learning_pathway([input_1, hidden_1, output_1])
#             comp.add_backpropagation_learning_pathway([input_1, hidden_1, output_2])
#           BUT THE FOLLOWING WORKS (WITH IDENTICAL show_graph(show_learning=True)):
#             comp = Composition(name='a_outer')
#             comp.add_backpropagation_learning_pathway([input_1, hidden_1, output_1])
#             comp.add_backpropagation_learning_pathway([hidden_1, output_2])
#      - show_graph(): QUIRK (BUT NOT BUG?):
#           SHOWS TWO PROJECTIONS FROM a_inner.input_CIM -> hidden_x:
#            ?? BECAUSE hidden_x HAS TWO input_ports SINCE ITS FUNCTION IS LinearCombination?
#             a_inner = AutodiffComposition([hidden_x],name='a_inner')
#             a_outer = AutodiffComposition([[input_1, a_inner, output_1],
#                                            [a_inner, output_2]],
#             a_outer.show_graph(show_cim=True)

#      -LearningMechanism / Backpropagation LearningFunction:
#         - Construction of LearningMechanism on its own fails; e.g.:
#             lm = LearningMechanism(learning_rate=.01, learning_function=BackPropagation())
#             causes the following error:
#                TypeError("Logistic.derivative() missing 1 required positional argument: 'self'")
#      - Adding GatingMechanism after Mechanisms they gate fails to implement gating projections
#           (example:  reverse order of the following in _construct_pathways
#                      self.add_nodes(self.softmax_nodes)
#                      self.add_nodes(self.field_weight_nodes)
#           - add Normalize as option
#           - Anytime a row's norm is 0, replace with 1s
#      - WHY IS Concatenate NOT WORKING AS FUNCTION OF AN INPUTPORT (WASN'T THAT USED IN CONTEXT OF BUFFER?
#           SEE NOTES TO KATHERINE
#
#     - TESTS
#       For duplicate Projections (e.g., assign a Mechanism in **monitor** of ControlMechanism
#            and use comp.add_projection(MappingProjection(mointored, control_mech) -> should generate a duplicate
#            then search for other instances of the same error message

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
as "values" that are retrieved but not used by the match process.  The inputs corresponding to each key (i.e., used
as "queries") and value are represented as `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` of the EMComposition
(listed in its `query_input_nodes <EMComposition.query_input_nodes>` and `value_input_nodes
<EMComposition.value_input_nodes>` attributes, respectively), and the retrieved values are represented as `OUTPUT
<NodeRole.OUTPUT>` `Nodes <Composition_Nodes>` of the EMComposition.  The `memory <EMComposition.memory>` can be
accessed using its `memory <EMComposition.memory>` attribute.

.. _EMComposition_Organization:

**Organization**

*Entries and Fields*. Each entry in memory can have an arbitrary number of fields, and each field can have an arbitrary
length.  However, all entries must have the same number of fields, and the corresponding fields must all have the same
length across entries. Fields can be weighted to determine the influence they have on retrieval, using the
`field_weights <ContentAddressableMemory.memory>` parameter (see `retrieval <EMComposition_Retrieval_Storage>` below).
The number and shape of the fields in each entry is specified in the ``memory_template`` argument of the EMComposition's
constructor (see `memory_template <EMComposition_Fields>`). Which fields treated as keys (i.e., matched against
queries during retrieval) and which are treated as values (i.e., retrieved but not used for matching retrieval) is
specified in the ``field_weights`` argument of the EMComposition's constructor (see `field_weights
<EMComposition_Field_Weights>`).

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
  <EMComposition.memory>`, and can be used to initialize `memory <EMComposition.memory>` with pre-specified entries.
  The ``memory_template`` argument can be specified in one of three ways (see `EMComposition_Examples` for
  representative use cases):

  * **tuple**: interpreted as an np.array shape specification, that must be of length 2 or 3.  If it is a 3-item tuple,
    then the first item specifies the number of entries in memory, the 2nd the number of fields in each entry, and the
    3rd the length of each field.  If it is a 2-item tuple, this specifies the shape of an entry, and the number of
    entries is specified by `memory_capacity <EMComposition_Memory_Capacity>`).  All entries are
    filled with zeros or the value specified by `memory_fill <EMComposition_Memory_Fill>`.

    .. warning::
       If the ``memory_template`` is specified with a 3-item tuple and `memory_capacity <EMComposition_Memory_Capacity>`
       is also specified with a value that does not match the first item of ``memory_template``, and error is
       generated indicating the conflict in the number of entries specified.

    .. hint::
       To specify a single field, a list or array must be used (see below), as a 2-item tuple is interpreted as
       specifying the shape of an entry, and so it can't be used to specify the number of entries each of which
       has a single field.

  * **2d list or array**:  interpreted as a template for memory entries.  This can be used to specify fields of
    different lengths (i.e., entries that are ragged arrays), with each item in the list (axis 0 of the array) used
    to specify the length of the corresponding field.  The template is then used to initialze all entries in `memory
    <EMComposition.memory>`.  If the template includes any non-zero elements, then the array is replicated for all
    entries in `memory <EMComposition.memory>`; otherwise, they are filled with either zeros or the value specified
    in `memory_fill <EMComposition_Memory_Fill>`.

    .. hint::
       To specify a single entry, with all other entries filled with zeros
       or the value specified in ``memory_fill``, use a 3d array as described below.

  * **3d list or array**:  used to initialize `memory <EMComposition.memory>` directly with the entries specified in
    the outer dimension (axis 0) of the list or array.  If `memory_capacity <EMComposition_Memory_Capacity>` is not
    specified, then it is set to the number of entries in the list or array. If ``memory_capacity`` *is* specified,
    then the number of entries specified in ``memory_template`` must be less than or equal to ``memory_capacity``.  If
    is less than ``memory_capacity``, then the remaining entries in `memory <EMComposition.memory>` are filled with
    zeros or the value specified in ``memory_fill`` (see below):  if all of the entries specified contain only
    zeros, and ``memory_fill`` is specified, then the matrix is filled with the value specified in ``memory_fill``;
    otherwise, zeros are used to fill all entries.

.. _EMComposition_Memory_Capacity:

*Memory Capacity*

* **memory_capacity**: specifies the number of items that can be stored in the EMComposition's memory; when
  `memory_capacity <EMComposition.memory_capacity>` is reached, each new entry overwrites the weakest entry (i.e., the
  one with the smallest norm across all of its fields) in `memory <EMComposition.memory>`.  If `memory_template
  EMComposition_Memory_Template>` is specified as a 3-item tuple or 3d list or array (see above), then that is used
  to determine `memory_capacity <EMComposition.memory_capacity>` (if it is specified and conflicts with either of those
  an error is generated).  Otherwise, it can be specified using a numerical value, with a default of 1000.  The
  `memory_capacity <EMComposition.memory_capacity>` cannot be modified once the EMComposition has been constructed.

.. _EMComposition_Memory_Fill:

* **memory_fill**: specifies the value used to fill the `memory <EMComposition.memory>`, based on the shape specified
  in the ``memory_template`` (see above).  The value can be a scalar, or a tuple to specify an interval over which
  to draw random values to fill `memory <EMComposition.memory>` --- both should be scalars, with the first specifying
  the lower bound and the second the upper bound.  If ``memory_fill`` is not specified, and no entries are specified
  in ``memory_template``, then `memory <EMComposition.memory>` is filled with zeros.

  .. hint::
     If memory is initialized with all zeros and ``normalize_memories`` set to ``True`` (see `below
     <EMComposition_Retrieval_Storage>`) then a numpy.linalg warning is issued about divide by zero.
     This can be ignored, as it does not affect the results of execution, but it can be averted by specifying
     `memory_fill <EMComposition_Memory_Fill>` to use small random values (e.g., ``memory_fill=(0,.001)``).

.. _EMComposition_Field_Weights:

* **field_weights**: specifies which fields are used as keys, and how they are weighted during retrieval. The
  number of values specified must match the number of fields specified in ``memory_template`` (i.e., the size of
  of its first dimension (axis 0)).  All non-zero entries must be positive, and designate *keys* -- fields
  that are used to match queries agains entries in memory for retrieval (see `Match memories by field
  <EMComposition_Processing>`). Entries of 0 designate *values* -- fields that are ignored during the matching
  process, but the values of which are retrieved and assigned as the `value <Mechanism_Base.value>` of the
  corresponding `retrieved_node <EMComposition.retrieved_nodes>`. This distinction between keys and value implements
  a standard "dictionary; however, if all entries are non-zero, then all fields are treated as keys, implemented a
  full form of content-addressable memory. If ``learn_field_weight`` is True, the field_weights can be modified
  during training, and function like the attention head of a Transformer model); otherwise they remain fixed. The
  following options can be used to specify ``field_weights``:

    * *None* (the default): all fields except the last are treated as keys, and are weighted equally for retrieval,
      while the last field is treated as a value field;

    * *single entry*: its value is ignored, and all fields are treated as keys (i.e., used for
      retrieval) and equally weighted for retrieval;

    * *multiple non-zero entries*: If all entries are identical, the value is ignored and the corresponding keys
      are weighted equally for retrieval; if the non-zero entries are non-identical, they are used to weight the
      corresponding fields during retrieval (see `Weight fields <EMComposition_Processing>`).  In either case,
      the remaining fields (with zero weights) are treated as value fields.

.. _EMComposition_Field_Names:

* **field_names**: specifies names that can be assigned to the fields.  The number of names specified must
  match the number of fields specified in the memory_template.  If specified, the names are used to label the
  nodes of the EMComposition.  If not specified, the fields are labeled generically as "Key 0", "Key 1", etc..

.. _EMComposition_Concatenate_Keys:

* **concatenate_keys**:  specifies whether keys are concatenated before a match is made to items in memory.
  This is False by default. It is also ignored if the `field_weights <EMComposition.field_weights>` for all keys are
  not all equal (i.e., all non-zero weights are not equal -- see `field_weights <EMComposition_Field_Weights>`) and/or
  `normalize_memories <EMComposition.normalize_memories>` is set to False. Setting concatenate_keys to True in either
  of those cases issues a warning, and the setting is ignored. If the key `field_weights <EMComposition.field_weights>`
  (i.e., all non-zero values) are all equal *and* ``normalize_memories`` is set to True, then setting
  ``concatenate_keys`` then a concatenate_keys_node <EMComposition.concatenate_keys_node>` is created that
  receives input from all of the `query_input_nodes <EMComposition.query_input_nodes>` and passes them as a single
  vector to the `mactch_node <EMComposition.match_nodes>`.

      .. note::
         While this is computationally more efficient, it can affect the outcome of the `matching process
         <EMComposition_Processing>`, since computing the normalized dot product of a single vector comprised of the
         concatentated inputs is not identical to computing the normalized dot product of each field independently and
         then combining the results.

      .. note::
         All `query_input_nodes <EMComposition.query_input_nodes>` and `retrieved_nodes <EMComposition.retrieved_nodes>`
         are always preserved, even when `concatenate_keys <EMComposition.concatenate_keys>` is True, so that separate
         inputs can be provided for each key, and the value of each key can be retrieved separately.

.. _EMComposition_Memory_Decay_Rate

* **memory_decay_rate**: specifies the rate at which items in the EMComposition's memory decay;  the default rate
  is *AUTO*, which sets it to  1 / `memory_capacity <EMComposition.memory_capacity>`, such that the oldest memories
  are the most likely to be replaced when `memory_capacity <EMComposition.memory_capacity>` is reached.  If
  ``memory_decay_rate`` is set to 0 None or False, then memories do not decay and, when `memory_capacity
  <EMComposition.memory_capacity>` is reached, the weakest memories are replaced, irrespective of order of entry.

.. _EMComposition_Retrieval_Storage:

*Retrieval and Storage*

* **storage_prob** : specifies the probability that the inputs to the EMComposition will be stored as an item in
  `memory <EMComposition.memory>` on each execution.

* **normalize_memories** : specifies whether queries and keys in memory are normalized before computing their dot
  products.

.. _EMComposition_Softmax_Gain:

* **softmax_gain** : specifies the gain (inverse temperature) used for softmax normalizing the dot products of queries
  and keys in memory (see `EMComposition_Execution` below).  If a value is specified, that is used.  If the keyword
  *CONTROL* is (or the value is None), then the `softmax_gain <EMComposition.softmax_gain>` function is used to
  adaptively set the gain based on the entropy of the dot products, preserving the distribution over non-(or near)
  zero entries irrespective of how many (near) zero entries there are.

* **learn_field_weight** : specifies whether `field_weights <EMComposition.field_weights>` are modifiable during training.

* **learning_rate** : specifies the rate at which  `field_weights <EMComposition.field_weights>` are learned if
  ``learn_field_weight`` is True.


.. _EMComposition_Structure:

Structure
---------

.. _EMComposition_Input:

*Input*
~~~~~~~

The inputs corresponding to each key and value field are represented as `INPUT <NodeRole.INPUT>` `Nodes
<Composition_Nodes>` of the EMComposition, listed in its `query_input_nodes <EMComposition.query_input_nodes>`
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
     from the `combined_softmax_node <EMComposition.combined_softmax_node>` to each of the `retrieved_nodes
     <EMComposition.retrieved_nodes>`. Memories associated with each key are also stored (in inverted form) in the
     `matrix <MappingProjection.matrix>` parameters of the `MappingProjections` from the `query_input_nodes
     <EMComposition.query_input_nodes>` to each of the corresponding `match_nodes <EMComposition.match_nodes>`.
     This is done so that the match of each query to the keys in memory for the corresponding field can be computed
     simply by passing the input for each query through the Projection (which computes the dot product of the input with
     the Projection's `matrix <MappingProjection.matrix>` parameter) to the corresponding match_node; and, similarly,
     retrieivals can be computed by passing the softmax distributions and weighting for each field computed
     in the `combined_softmax_node <EMComposition.combined_softmax_node>` through its Projection to each
     `retrieved_node <EMComposition.retrieved_nodes>` (which are inverted versions of the matrices of the
     `MappingProjections` from the `query_input_nodes <EMComposition.query_input_nodes>` to each of the corresponding
     `match_nodes <EMComposition.match_nodes>`), to compute the dot product of the weighted softmax over
     entries with the corresponding field of each entry that yields the retreieved value for each field.

.. _EMComposition_Output:

*Output*
~~~~~~~

The outputs corresponding to retrieved value for each field are represented as `OUTPUT <NodeRole.INPUT>` `Nodes
<Composition_Nodes>` of the EMComposition, listed in its `retrieved_nodes <EMComposition.retrieved_nodes>` attribute.

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

* **Input**.  The inputs to the EMComposition are provided to the `query_input_nodes <EMComposition.query_input_nodes>`
  and `value_input_nodes <EMComposition.value_input_nodes>`.  The former are used for matching to the corresponding
  `fields <EMComposition_Fields>` of the `memory <EMComposition.memory>`, while the latter are retrieved but not used
  for matching.

* **Concatenation**. By default, the input to every `query_input_node <EMComposition.query_input_nodes>` is passed to a
  to its own `match_node <EMComposition.match_nodes>` through a `MappingProjection` that computes its
  dot product with the corresponding field of each entry in `memory <EMComposition.memory>`.  In this way, each
  match is normalized so that, absent `field_weighting <EMComposition_Field_Weights>`, all keys contribute equally to
  retrieval irrespective of relative differences in the norms of the queries or the keys in memory. However, if the
  `field_weights <EMComposition.field_weights>` are the same for all `keys <EMComposition_Field_Weights>` and
  `normalize_memories <EMComposition.normalize_memories>` is True, then the inputs provided to the `query_input_nodes
  <EMComposition.query_input_nodes>` are concatenated into a single vector (in the
  `concatenate_keys_node <EMComposition.concatenate_keys_node>`), which is passed to a single `match_node
  <EMComposition.match_nodes>`.  This may be more computationally efficient than passing each query through its own
  `match_node <EMComposition.match_nodes>`,
  COMMENT:
  FROM CodePilot: (OF HISTORICAL INTEREST?)
  and may also be more effective if the keys are highly correlated (e.g., if they are different representations of
  the same stimulus).
  COMMENT
  however it will not necessarily produce the same results as passing each query through its own `match_node
  <EMComposition.match_nodes>` (see `concatenate keys <`concatenate_keys_node
  >` for additional
  information).

* **Match memories by field**. The values of each `query_input_node <EMComposition.query_input_nodes>` (or the
  `concatenate_keys_node <EMComposition.concatenate_keys_node>` if `concatenate_keys <EMComposition_Concatenate_Keys>`
  attribute is True) are passed through a `MappingProjection` that computes the dot product of the input with each
  memory for the corresponding field, the result of which is passed to the corresponding `match_node
  <EMComposition.match_nodes>`.

* **Softmax normalize matches over fields**. The dot product for each key field is passed from the `match_node
  <EMComposition.match_nodes>` to the corresponding `softmax_node <EMComposition.softmax_nodes>`, which applies
  a softmax function to normalize the dot products for each key field.  If a numerical value is specified for
  `softmax_gain <EMComposition.softmax_gain>`, that is used as the gain (inverse temperature) for the softmax function;
  otherwise, if it is specified as *CONTROL* or None, then the `softmax_gain <EMComposition.softmax_gain>` function is
  used to adaptively set the gain (see `softmax_gain <EMComposition_Softmax_Gain>` for details).

* **Weight fields**. If `field weights <EMComposition_Field_Weights>` are specified, then the softmax normalized dot
  product for each key field is passed to the corresponding `field_weight_node <EMComposition.field_weight_nodes>`
  where it is multiplied by the corresponding `field_weight <EMComposition.field_weights>` (if
  `use_gating_for_weighting <EMComposition.use_gating_for_weighting>` is True, this is done by using the `field_weight
  <EMComposition.field_weights>` to output gate the `softmax_node <EMComposition.softmax_nodes>`). The weighted softamx
  vectors for all key fields are then passed to the `combined_softmax_node <EMComposition.combined_softmax_node>`,
  where they are haddamard summed to produce a single weighting for each memory.

* **Retrieve values by field**. The vector of softmax weights for each memory generated by the `combined_softmax_node
  <EMComposition.combined_softmax_node>` is passed through the Projections to the each of the `retrieved_nodes
  <EMComposition.retrieved_nodes>` to compute the retrieved value for each field.

* **Decay memories**.  If `memory_decay <EMComposition.memory_decay>` is True, then each of the memories is decayed
  by the amount specified in `memory_decay <EMComposition.memory_decay>`.

    .. technical_note::
       This is done by multiplying the `matrix <MappingProjection.matrix>` parameter of the `MappingProjection` from
       the `combined_softmax_node <EMComposition.combined_softmax_node>` to each of the `retrieved_nodes
       <EMComposition.retrieved_nodes>`, as well as the `matrix <MappingProjection.matrix>` parameter of the
       `MappingProjection` from each `query_input_node <EMComposition.query_input_nodes>` to the corresponding
       `match_node <EMComposition.match_nodes>` by `memory_decay <EMComposition.memory_decay_rate>`,
        by 1 - `memory_decay <EMComposition.memory_decay_rate>`.

.. _EMComposition_Storage:

* **Store memories**. After the values have been retrieved, the inputs to for each field (i.e., values in the
  `query_input_nodes <EMComposition.query_input_nodes>` and `value_input_nodes <EMComposition.value_input_nodes>`)
  are added by the `storage_node <EMComposition.storage_node>` as a new entry in `memory <EMComposition.memory>`,
  replacing the weakest one if `memory_capacity <EMComposition_Memory_Capacity>` has been reached.

    .. technical_note::
       This is done by adding the input vectors to the the corresponding rows of the `matrix <MappingProjection.matrix>`
       of the `MappingProjection` from the `retreival_weighting_node <EMComposition.combined_softmax_node>` to each
       of the `retrieved_nodes <EMComposition.retrieved_nodes>`, as well as the `matrix <MappingProjection.matrix>`
       parameter of the `MappingProjection` from each `query_input_node <EMComposition.query_input_nodes>` to the
       corresponding `match_node <EMComposition.match_nodes>` (see note `above <EMComposition_Memory_Storage>` for
       additional details). If `memory_capacity <EMComposition_Memory_Capacity>` has been reached, then the weakest
       memory (i.e., the one with the lowest norm across all fields) is replaced by the new memory.

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

.. _EMComposition_Learning:

*Learning*
~~~~~~~~~~

FIX: MODIFY TO INDICATE THAT enable_learning ALLOWS PROPAGATION OF ERROR TRHOUGH THE NETWORK,
     WHILE learn_field_weights ALLOWS LEARNING OF THE FIELD_WEIGHTS, WHICH REQUIRES enable_learning TO BE True
If `learn <Composition.learn>` is called and the `learn_field_weights <EMComposition.learn_field_weights>` attribute
is True, then the `field_weights <EMComposition.field_weights>` are modified to minimize the error passed to the
EMComposition retrieved nodes, using the learning_rate specified in the `learning_rate <EMComposition.learning_rate>`
attribute. If `learn_field_weights <EMComposition.learn_field_weights>` is False (or `run <Composition.run>` is called,
then the `field_weights <EMComposition.field_weights>` are not modified and the EMComposition is simply executed
without any modification, and the error signal is passed to the nodes that project to its `INPUT <NodeRole.INPUT>`
`Nodes <Composition_Nodes>`.

  .. note::
     Although memory storage is implemented as  a form of learning (though modification of MappingProjection
     `matrix <MappingProjection.matrix>` parameters; see `memory storage <EMComposition_Memory_Storage>`),
     this occurs irrespective of how EMComposition is run (i.e., whether `learn <Composition.learn>` or `run
     <Composition.run>` is called), and is not affected by the `learn_field_weights <EMComposition.learn_field_weights>`
     or `learning_rate <EMComposition.learning_rate>` attributes, which pertain only to whether the `field_weights
     <EMComposition.field_weights>` are modified during learning.

.. _EMComposition_Examples:

Examples
--------

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
<EMComposition_Memory>` can be inspected using the `memory <EMComposition.memory>` attribute::

    >>> em.memory
    [[array([0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0.])],
     [array([0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0.])],
     [array([0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0.])],
     [array([0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0.])]]

The default for `memory_capacity <EMComposition.memory_capacity>` is 1000, which is used if it is not otherwise
specified.

**List or array specification**

Note that in the example above the two fields have the same length (5). This is always the case when a tuple is used,
as it generates a regular array.  A list or numpy array can also be used to specify the ``memory_template`` argument.
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
because the default value for ``memory_fill`` is ``0``, and the default value for `normalize_memories
<EMComposition.normalize_memories>` is True, which will cause a divide by zero warning when memories are
normalized. While this doesn't crash, it will result in nan's that are likely to cauase problems elsewhere.
This can be avoided by specifying a non-zero  value for ``memory_fill``, such as small number::

    >>> em = EMComposition(memory_template=[[0,0,0],[0]], memory_capacity=4, memory_fill=.001)
    >>> em.memory
    [[[array([0.001, 0.001, 0.001]), array([0.001])]],
     [[array([0.001, 0.001, 0.001]), array([0.001])]],
     [[array([0.001, 0.001, 0.001]), array([0.001])]],
     [[array([0.001, 0.001, 0.001]), array([0.001])]]]

Here, a single value was specified for ``memory_fill`` (which can be a float or int), that is used to fill all values.
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
Also note that the remaining entries are filled with zeros (the default value for ``memory_fill``).
Here again, ``memory_fill`` can be used to specify a different value::

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
and the last as a value field. However, the ``field_weights`` argument can be used to modify this, specifying which
fields should be used as keys fields -- including the relative contribution that each makes to the matching process
-- and which should be used as value fields.  Non-zero elements in the ``field_weights`` argument designate key fields,
and zeros specify value fields. For example, the following specifies that the first two fields should be used as keys
while the last two should be used as values::

    >>> em = EMComposition(memory_template=[[0,0,0],[0],[0,0],[0,0,0,0]], memory_capacity=3, field_weights=[1,1,0,0])
    >>> em.show_graph()
    <BLANKLINE>


.. _EMComposition_Example_Field_Weights_Equal_fig:

.. figure:: _static/EMComposition_field_weights_equal_fig.svg

    **Use of field_weights to specify keys and values.**

Note that the figure now shows `RETRIEVAL WEIGHTING <EMComposition.field_weight_nodes>` `nodes <Composition_Node>`,
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

Note that in this case, the `concatenate_keys_node <EMComposition.concatenate_keys_node>` has been replaced by a
pair of `retreival_weighting_nodes <EMComposition.field_weight_nodes>`, one for each key field.  This is because
the keys were assigned different weights;  when they are assigned equal weights, or if no weights are specified,
and `normalize_memories <EMComposition.normalize_memories>` is `True`, then the keys are concatenated and are
concatenated for efficiency of processing.  This can be suppressed by specifying `concatenate_keys` as `False`
(see `concatenate_keys <EMComposition_Concatenate_Keys>` for additional details).
COMMENT

.. _EMComposition_Class_Reference:

Class Reference
---------------
"""
import numpy as np
import graph_scheduler as gs
import warnings
import psyneulink.core.scheduling.condition as conditions

from psyneulink._typing import Optional, Union

# from psyneulink.library.compositions import torch_available
from psyneulink.core.components.functions.nonstateful.transferfunctions import SoftMax, LinearMatrix
from psyneulink.core.components.functions.nonstateful.combinationfunctions import Concatenate, LinearCombination
from psyneulink.core.components.functions.function import \
    DEFAULT_SEED, _random_state_getter, _seed_setter
from psyneulink.core.compositions.composition import CompositionError, NodeRole
from psyneulink.library.compositions.autodiffcomposition import AutodiffComposition, torch_available
from psyneulink.library.components.mechanisms.modulatory.learning.EMstoragemechanism import EMStorageMechanism
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.modulatory.control.gating.gatingmechanism import GatingMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.keywords import \
    (AUTO, CONTROL, DEFAULT_INPUT, DEFAULT_VARIABLE, EM_COMPOSITION, FULL_CONNECTIVITY_MATRIX,
     GAIN, IDENTITY_MATRIX, MULTIPLICATIVE_PARAM, NAME, PARAMS, PRODUCT, PROJECTIONS, RANDOM, SIZE, VARIABLE)
from psyneulink.core.globals.utilities import all_within_range
from psyneulink.core.llvm import ExecutionMode


__all__ = [
    'EMComposition'
]

STORAGE_PROB = 'storage_prob'

def _memory_getter(owning_component=None, context=None)->list:
    """Return list of memories in which rows (outer dimension) are memories for each field.
    These are derived from `matrix <MappingProjection.matrix>` parameter of the `afferent
    <Mechanism_Base.afferents>` MappingProjections to each of the `retrieved_nodes <EMComposition.retrieved_nodes>`.
    """

    # If storage_node (EMstoragemechanism) is implemented, get memory from that
    if owning_component.is_initializing:
        return None
    if owning_component.use_storage_node:
        return owning_component.storage_node.parameters.memory_matrix.get(context)

    # Otherwise, get memory from Projection(s) to each retrieved_node
    memory = [retrieved_node.path_afferents[0].parameters.matrix.get(context)
              for retrieved_node in owning_component.retrieved_nodes]
    # Reorganize memory so that each row is an entry and each column is a field
    memory_capacity = owning_component.memory_capacity or owning_component.defaults.memory_capacity
    return [[memory[j][i] for j in range(owning_component.num_fields)]
              for i in range(memory_capacity)]

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
    EMComposition(                      \
        memory_template=[[0],[0]],      \
        memory_fill=0,                  \
        memory_capacity=None,           \
        field_weights=None,             \
        field_names=None,               \
        concatenate_keys=False,         \
        normalize_memories=True,        \
        softmax_gain=CONTROL,           \
        storage_prob=1.0,               \
        memory_decay_rate=AUTO,         \
        enable_learning=True,           \
        learn_field_weights=True,       \
        learning_rate=True,             \
        use_gating_for_weighting=False, \
        name="EM_Composition"           \
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

    memory_capacity : int : default None
        specifies the number of items that can be stored in the EMComposition's memory;
        see `memory_capacity <EMComposition_Memory_Capacity>` for details.

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

    storage_prob : float : default 1.0
        specifies the probability that an item will be stored in `memory <EMComposition.memory>`
        when the EMComposition is executed (see `Retrieval and Storage <EMComposition_Storage>` for
        additional details).

    memory_decay_rate : float : AUTO
        specifies the rate at which items in the EMComposition's memory decay;
        see `memory_decay_rate <EMComposition_Memory_Decay_Rate>` for details.

    enable_learning : bool : default True
        specifies whether learning pathway is constructed for the EMComposition (see `enable_learning
        <Composition.enable_learning>` for additional details).

    learn_field_weights : bool : default True
        specifies whether `field_weights <EMComposition.field_weights>` are learnable during training;
        requires **enable_learning** to be True to have any effect;  see `learn_field_weights
        <EMComposition_Learning>` for additional details.

    learning_rate : float : default .01
        specifies rate at which `field_weights <EMComposition.field_weights>` are learned
        if ``learn_field_weights`` is True.

    .. technical_note::
        use_storage_node : bool : default True
            specifies whether to use a `LearningMechanism` to store entries in `memory <EMComposition.memory>`.
            If False, a method on EMComposition is used rather than a LearningMechanism.  This is meant for
            debugging, and precludes use of `import_composition <Composition.import_composition>` to integrate
            the EMComposition into another Composition;  to do so, use_storage_node must be True (default).

    use_gating_for_weighting : bool : default False
        specifies whether to use a `GatingMechanism` to modulate the `combined_softmax_node
        <EMComposition.combined_softmax_node>` instead of a standard ProcessingMechanism.  If True, then
        a GatingMechanism is constructed and used to gate the `OutputPort` of each `field_weight_node
        EMComposition.field_weight_nodes`;  otherwise, the output of each `field_weight_node
        EMComposition.field_weight_nodes` projects to the `InputPort` of the `combined_softmax_node
        EMComposition.combined_softmax_node` that receives a Projection from the corresponding
        `field_weight_node <EMComposition.field_weight_nodes>`, and multiplies its `value
        <Projection_Base.value>`.

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

    memory_capacity : int
        determines the number of items that can be stored in `memory <EMComposition.memory>`; see `memory_capacity
        <EMComposition_Memory_Capacity>` for additional details.

    field_weights : list[float]
        determines which fields of the input are treated as "keys" (non-zero values), used to match entries in `memory
        <EMComposition.memory>` for retrieval, and which are used as "values" (zero values), that are stored and
        retrieved from memory, but not used in the match process (see `Match memories by field
        <EMComposition_Processing>`;  see `field_weights <EMComposition_Field_Weights>` for additional details
        of specification).

    field_names : list[str]
        determines which names that can be used to label fields in `memory <EMComposition.memory>`;  see
        `field_names <EMComposition_Field_Names>` for additional details.

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
        determines the probability that an item will be stored in `memory <EMComposition.memory>`
        when the EMComposition is executed (see `Retrieval and Storage <EMComposition_Storage>` for
        additional details).

    memory_decay_rate : float
        determines the rate at which items in the EMComposition's memory decay (see `memory_decay_rate
        <EMComposition_Memory_Decay_Rate>` for details).

    enable_learning : bool
        determines whether `learning <Composition_Learning>` is enabled for the EMComposition, allowing any error
        received by the `retrieved_nodes <EMComposition.retrieved_nodes>` to be propagated to the `query_input_nodes
        <EMComposition.query_input_nodes>` and `value_input_nodes <EMComposition.value_input_nodes>`, and on to any
        `Nodes <Composition_Nodes>` that project to them.

    learn_field_weights : bool
        determines whether `field_weights <EMComposition.field_weights>` are learnable during training;
        requires `enable_learning <EMComposition.enable_learning>` to be True;  see `Learning
        <EMComposition_Learning>` for additional details.

    learning_rate : float
        determines whether the rate at which `field_weights <EMComposition.field_weights>` are learned
        if `learn_field_weights` is True;  see `EMComposition_Learning>` for additional details.

    .. _EMComposition_Nodes:

    query_input_nodes : list[TransferMechanism]
        `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` that receive keys used to determine the item
        to be retrieved from `memory <EMComposition.memory>`, and then themselves stored in `memory
        <EMComposition.memory>` (see `Match memories by field <EMComposition_Processing>` for additional details).
        By default these are assigned the name *KEY_n_INPUT* where n is the field number (starting from 0);
        however, if `field_names <EMComposition.field_names>` is specified, then the name of each query_input_node
        is assigned the corresponding field name.

    value_input_nodes : list[TransferMechanism]
        `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` that receive values to be stored in `memory
        <EMComposition.memory>`; these are not used in the matching process used for retrieval.  By default these
        are assigned the name *VALUE_n_INPUT* where n is the field number (starting from 0);  however, if
        `field_names <EMComposition.field_names>` is specified, then the name of each value_input_node is assigned

    concatenate_keys_node : TransferMechanism
        `TransferMechanism` that concatenates the inputs to `query_input_nodes <EMComposition.query_input_nodes>` into a
        single vector used for the matching processing if `concatenate keys <EMComposition.concatenate_keys>` is True.
        This is not created if the ``concatenate_keys`` argument to the EMComposition's constructor is False or is
        overridden (see `concatenate_keys <EMComposition_Concatenate_Keys>`), or there is only one query_input_node.

    match_nodes : list[TransferMechanism]
        `TransferMechanisms <TransferMechanism>` that receive the dot product of each key and those stored in
        the corresponding field of `memory <EMComposition.memory>` (see `Match memories by field
        <EMComposition_Processing>` for additional details).  These are assigned names that prepend *MATCH_n* to the
        name of the corresponding `query_input_nodes <EMComposition.query_input_nodes>`.

    softmax_gain_control_nodes : list[ControlMechanism]
        `ControlMechanisms <ControlMechanism>` that adaptively control the `softmax_gain <EMComposition.softmax_gain>`
        for the corresponding `softmax_nodes <EMComposition.softmax_nodes>`. These are implemented only if
        `softmax_gain <EMComposition.softmax_gain>` is specified as *CONTROL* (see `softmax_gain
        <EMComposition_Softmax_Gain>` for details).

    softmax_nodes : list[TransferMechanism]
        `TransferMechanisms <TransferMechanism>` that compute the softmax over the vectors received
        from the corresponding `match_nodes <EMComposition.match_nodes>` (see `Softmax normalize matches over fields
        <EMComposition_Processing>` for additional details).

    field_weight_nodes : list[ProcessingMechanism]
        `ProcessingMechanisms <ProcessingMechanism>`, each of which use the `field weight <EMComposition.field_weights>`
        for a given `field <EMComposition_Fields>` as its (fixed) input and provides this to the corresponding
        `weighted_softmax_node <EMComposition.weighted_softmax_nodes>`. These are implemented only if more than one
        `key field <EMComposition_Fields>` is specified (see `Fields <EMComposition_Fields>` for additional details),
        and are replaced with `retrieval_gating_nodes <EMComposition.retrieval_gating_nodes>` if
        `use_gating_for_weighting <EMComposition.use_gating_for_weighting>` is True.

    weighted_softmax_nodes : list[ProcessingMechanism]
        `ProcessingMechanisms <ProcessingMechanism>`, each of which receives the output of the corresponding
        `softmax_node <EMComposition.softmax_nodes>` and `field_weight_node <EMComposition.field_weight_nodes>`
        for a given `field <EMComposition_Fields>`, and multiplies them to produce the weighted softmax for that field;
        these are implemented only if more than one `key field <EMComposition_Fields>` is specified (see `Fields
        <EMComposition_Fields>` for additional details) and `use_gating_for_weighting
        <EMComposition.use_gating_for_weighting>` is False (in which case, `field_weights <EMComposition.field_weights>`
        are applied through output gating of the `softmax_nodes <EMComposition.softmax_nodes>` by the
        `retrieval_gating_nodes <EMComposition.retrieval_gating_nodes>`).

    retrieval_gating_nodes : list[GatingMechanism]
        `GatingMechanisms <GatingMechanism>` that uses the `field weight <EMComposition.field_weights>` for each
        field to modulate the output of the corresponding `softmax_node <EMComposition.softmax_nodes>` before it
        is passed to the `combined_softmax_node <EMComposition.combined_softmax_node>`. These are implemented
        only if `use_gating_for_weighting <EMComposition.use_gating_for_weighting>` is True and more than one
        `key field <EMComposition_Fields>` is specified (see `Fields <EMComposition_Fields>` for additional details).

    combined_softmax_node : TransferMechanism
        `TransferMechanism` that receives the softmax normalized dot products of the keys and memories
        from the `softmax_nodes <EMComposition.softmax_nodes>`, weighted by the `field_weights_nodes
        <EMComposition.field_weights_nodes>` if more than one `key field <EMComposition_Fields>` is specified
        (or `retrieval_gating_nodes <EMComposition.retrieval_gating_nodes>` if `use_gating_for_weighting
        <EMComposition.use_gating_for_weighting>` is True), and combines them into a single vector that is used to
        retrieve the corresponding memory for each field from `memory <EMComposition.memory>` (see `Retrieve values by
        field <EMComposition_Processing>` for additional details).

    retrieved_nodes : list[TransferMechanism]
        `TransferMechanisms <TransferMechanism>` that receive the vector retrieved for each field in `memory
        <EMComposition.memory>` (see `Retrieve values by field <EMComposition_Processing>` for additional details);
        these are assigned the same names as the `query_input_nodes <EMComposition.query_input_nodes>` and
        `value_input_nodes <EMComposition.value_input_nodes>` to which they correspond appended with the suffix
        *_RETRIEVED*.

    storage_node : EMStorageMechanism
        `EMStorageMechanism` that receives inputs from the `query_input_nodes <EMComposition.query_input_nodes>` and
        `value_input_nodes <EMComposition.value_input_nodes>`, and stores these in the corresponding field of
        `memory <EMComposition.memory>` with probability `storage_prob <EMComposition.storage_prob>` after a retrieval
        has been made (see `Retrieval and Storage <EMComposition_Storage>` for additional details).

        .. technical_note::
           The `storage_node <EMComposition.storage_node>` is assigned a Condition to execute after the `retrieved_nodes
           <EMComposition.retrieved_nodes>` have executed, to ensure that storage occurs after retrieval, but before
           any subequent processing is done (i.e., in a composition in which the EMComposition may be embededded.
    """

    componentCategory = EM_COMPOSITION

    if torch_available:
        from psyneulink.library.compositions.pytorchEMcompositionwrapper import PytorchEMCompositionWrapper
        pytorch_composition_wrapper_type = PytorchEMCompositionWrapper


    class Parameters(AutodiffComposition.Parameters):
        """
            Attributes
            ----------

                concatenate_keys
                    see `concatenate_keys <EMComposition.concatenate_keys>`

                    :default value: False
                    :type: ``bool``

                enable_learning
                    see `enable_learning <EMComposition.enable_learning>`

                    :default value: True
                    :type: ``bool``

                field_names
                    see `field_names <EMComposition.field_names>`

                    :default value: None
                    :type: ``list``

                field_weights
                    see `field_weights <EMComposition.field_weights>`

                    :default value: None
                    :type: ``numpy.ndarray``

                learning_rate
                    see `learning_results <EMComposition.learning_rate>`

                    :default value: []
                    :type: ``list``

                learn_field_weights
                    see `learn_field_weights <EMComposition.learn_field_weights>`

                    :default value: True
                    :type: ``bool``

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
        memory_template = Parameter([[0],[0]], structural=True, valid_types=(tuple, list, np.ndarray), read_only=True)
        memory_capacity = Parameter(1000, structural=True)
        field_weights = Parameter(None)
        field_names = Parameter(None, structural=True)
        concatenate_keys = Parameter(False, structural=True)
        normalize_memories = Parameter(True)
        softmax_gain = Parameter(CONTROL, modulable=True)
        storage_prob = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        memory_decay_rate = Parameter(AUTO, modulable=True)
        enable_learning = Parameter(True, structural=True)
        learn_field_weights = Parameter(True, structural=True)
        learning_rate = Parameter(.001, modulable=True)
        random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies='seed')
        seed = Parameter(DEFAULT_SEED, modulable=True, setter=_seed_setter)

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

        def _validate_field_weights(self, field_weights):
            if field_weights is not None:
                if not np.atleast_1d(field_weights).ndim == 1:
                    return f"must be a scalar, list of scalars, or 1d array."
                if any([field_weight < 0 for field_weight in field_weights]):
                    return f"must be all be positive values."

        def _validate_field_names(self, field_names):
            if field_names and not all(isinstance(item, str) for item in field_names):
                return f"must be a list of strings."

        def _validate_memory_decay_rate(self, memory_decay_rate):
            if memory_decay_rate in {None, AUTO}:
                return
            if not (isinstance(memory_decay_rate, (float, int)) and all_within_range(memory_decay_rate, 0, 1)):
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
                 memory_capacity:Optional[int]=None,
                 memory_fill:Union[int, float, tuple, RANDOM]=0,
                 field_names:Optional[list]=None,
                 field_weights:tuple=None,
                 concatenate_keys:bool=False,
                 normalize_memories:bool=True,
                 softmax_gain:Union[float, CONTROL]=CONTROL,
                 storage_prob:float=1.0,
                 memory_decay_rate:Union[float,AUTO]=AUTO,
                 enable_learning:bool=True,
                 learn_field_weights:bool=True,
                 learning_rate:float=None,
                 use_storage_node:bool=True,
                 use_gating_for_weighting:bool=False,
                 random_state=None,
                 seed=None,
                 name="EM_Composition"):

        # Construct memory --------------------------------------------------------------------------------

        memory_fill = memory_fill or 0 # FIX: GET RID OF THIS ONCE IMPLEMENTED AS A Parameter
        self._validate_memory_specs(memory_template, memory_capacity, memory_fill, field_weights, field_names, name)
        memory_template, memory_capacity = self._parse_memory_template(memory_template,
                                                                       memory_capacity,
                                                                       memory_fill,
                                                                       field_weights)
        field_weights, field_names, concatenate_keys = self._parse_fields(field_weights,
                                                                          field_names,
                                                                          concatenate_keys,
                                                                          normalize_memories,
                                                                          learning_rate,
                                                                          name)
        if memory_decay_rate is AUTO:
            memory_decay_rate = 1 / memory_capacity

        self.use_storage_node = use_storage_node


        # Instantiate Composition -------------------------------------------------------------------------

        super().__init__(name=name,
                         memory_template = memory_template,
                         memory_capacity = memory_capacity,
                         field_weights = field_weights,
                         field_names = field_names,
                         concatenate_keys = concatenate_keys,
                         softmax_gain = softmax_gain,
                         storage_prob = storage_prob,
                         memory_decay_rate = memory_decay_rate,
                         normalize_memories = normalize_memories,
                         enable_learning=enable_learning,
                         learn_field_weights = learn_field_weights,
                         learning_rate = learning_rate,
                         random_state = random_state,
                         seed = seed
                         )

        self._construct_pathways(self.memory_template,
                                 self.memory_capacity,
                                 self.field_weights,
                                 self.concatenate_keys,
                                 self.normalize_memories,
                                 self.softmax_gain,
                                 self.storage_prob,
                                 self.memory_decay_rate,
                                 self.use_storage_node,
                                 self.enable_learning,
                                 self.learn_field_weights,
                                 use_gating_for_weighting)

        # if torch_available:
        #     from psyneulink.library.compositions.pytorchEMcompositionwrapper import PytorchEMCompositionWrapper
        #     self.pytorch_composition_wrapper_type = PytorchEMCompositionWrapper

        # Final Configuration and Clean-up ---------------------------------------------------------------------------

        # Assign learning-related attributes
        self._set_learning_attributes()

        if self.use_storage_node:
            # ---------------------------------------
            #
            # CONDITION:
            self.scheduler.add_condition(self.storage_node, conditions.AllHaveRun(*self.retrieved_nodes))
            #
            # Generates expected results, but execution_sets has a second set for INPUT nodes
            #    and the the match_nodes again with storage_node
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
            #   but generates warning messages for every node of the following sort:
            # /Users/jdc/PycharmProjects/PsyNeuLink/psyneulink/core/scheduling/scheduler.py:120:
            #   UserWarning: BeforeNCalls((EMStorageMechanism STORAGE MECHANISM), 1) is dependent on
            #   (EMStorageMechanism STORAGE MECHANISM), but you are assigning (EMStorageMechanism STORAGE MECHANISM)
            #   as its owner. This may result in infinite loops or unknown behavior.
            # super().add_condition_set(conditions)

        # Suppress warnings for no efferent Projections
        for node in self.value_input_nodes:
            node.output_ports['RESULT'].parameters.require_projection_in_composition.set(False, override=True)
        for port in self.combined_softmax_node.output_ports:
            if 'RESULT' in port.name:
                port.parameters.require_projection_in_composition.set(False, override=True)

        # Suppress field_weight_nodes as INPUT nodes of the Composition
        for node in self.field_weight_nodes:
            self.exclude_node_roles(node, NodeRole.INPUT)

        # Suppress value_input_nodes as OUTPUT nodes of the Composition
        for node in self.value_input_nodes:
            self.exclude_node_roles(node, NodeRole.OUTPUT)

        # Warn if divide by zero will occur due to memory initialization
        if not np.any([np.any([self.memory[i][j]
                               for i in range(self.memory_capacity)])
                       for j in range(self.num_keys)]):
            warnings.warn(f"Memory initialized with at least one field that has all zeros; "
                          f"a divide by zero will occur if 'normalize_memories' is True. "
                          f"This can be avoided by using 'memory_fill' to initialize memories with non-zero values.")

    # *****************************************************************************************************************
    # ***********************************  Memory Construction Methods  ***********************************************
    # *****************************************************************************************************************

    def _validate_memory_specs(self, memory_template, memory_capacity, memory_fill, field_weights, field_names, name):
        """Validate the memory_template, field_weights, and field_names arguments
        """

        # memory_template must specify a 2D array:
        if isinstance(memory_template, tuple):
        #     if len(memory_template) != 2 or not all(isinstance(item, int) for item in memory_template):
        #         raise EMCompositionError(f"The 'memory_template' arg for {name} ({memory_template}) uses a tuple to "
        #                                  f"shape requires but does not have exactly two integers.")
            num_fields = memory_template[0]
            if len(memory_template) == 3:
                num_entries = memory_template[0]
            else:
                num_entries = memory_capacity
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

        # Validate memory_fill specification (int, float, or tuple with two scalars)
        if not (isinstance(memory_fill, (int, float)) or
                (isinstance(memory_fill, tuple) and len(memory_fill)==2) and
                all(isinstance(item, (int, float)) for item in memory_fill)):
            raise EMCompositionError(f"The 'memory_fill' arg ({memory_fill}) specified for {name} "
                                     f"must be a float, int or len tuple of ints and/or floats.")

        # If len of field_weights > 1, must match the len of 1st dimension (axis 0) of memory_template:
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

    def _parse_memory_template(self, memory_template, memory_capacity, memory_fill, field_weights)->(np.ndarray,int):
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
                entries = [np.array(entry, dtype=object)] * num_entries

            return np.array(np.array(entries,dtype=object), dtype=object)

        # If memory_template is a tuple, create and fill full memory matrix
        if isinstance(memory_template, tuple):
            if len(memory_template) == 2:
                memory_capacity = memory_capacity or self.defaults.memory_capacity
                memory = _construct_entries(np.full(memory_template, 0), memory_capacity, memory_fill)
            else:
                if memory_capacity and memory_template[0] != memory_capacity:
                    raise EMCompositionError(
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
                        raise EMCompositionError(
                            f"The number of entries ({num_entries}) specified in "
                            f"the 'memory_template' arg of  {self.name} exceeds the number of entries specified in "
                            f"its 'memory_capacity' arg ({memory_capacity}); remove the latter or reduce the number"
                            f"of entries specified in 'memory_template'.")
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

        return memory, memory_capacity

    def _parse_fields(self,
                      field_weights,
                      field_names,
                      concatenate_keys,
                      normalize_memories,
                      learning_rate,
                      name):

        num_fields = len(self.entry_template)

        # Deal with default field_weights
        if field_weights is None:
            if len(self.entry_template) == 1:
                field_weights = [1]
            else:
                # Default is to treat all fields as keys except the last one, which is the value
                field_weights = [1] * num_fields
                field_weights[-1] = 0
        field_weights = np.atleast_1d(field_weights)
        # Fill out and normalize all field_weights
        if len(field_weights) == 1:
            parsed_field_weights = np.repeat(field_weights / np.sum(field_weights), len(self.entry_template))
        else:
            parsed_field_weights = np.array(field_weights) / np.sum(field_weights)

        # Memory structure Parameters
        parsed_field_names = field_names.copy() if field_names is not None else None

        # Set memory field attributes
        self.num_fields = len(self.entry_template)
        keys_weights = [i for i in parsed_field_weights if i != 0]
        self.num_keys = len(keys_weights)
        self.num_values = self.num_fields - self.num_keys
        if parsed_field_names:
            self.key_names = parsed_field_names[:self.num_keys]
            self.value_names = parsed_field_names[self.num_keys:]
        else:
            self.key_names = [f'{i} [QUERY]' for i in range(self.num_keys)] if self.num_keys > 1 else ['KEY']
            self.value_names = [f'{i} [VALUE]' for i in range(self.num_values)] if self.num_values > 1 else ['VALUE']

        user_specified_concatenate_keys = concatenate_keys or False
        parsed_concatenate_keys = (user_specified_concatenate_keys
                                    and self.num_keys > 1
                                    and np.all(keys_weights == keys_weights[0])
                                    and normalize_memories)
        # if concatenate_keys was forced to be False when user specified it as True, issue warning
        if user_specified_concatenate_keys and not parsed_concatenate_keys:
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

        self.learning_rate = learning_rate
        return parsed_field_weights, parsed_field_names, parsed_concatenate_keys

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

    # *****************************************************************************************************************
    # ******************************  Nodes and Pathway Construction Methods  *****************************************
    # *****************************************************************************************************************

    def _construct_pathways(self,
                            memory_template,
                            memory_capacity,
                            field_weights,
                            concatenate_keys,
                            normalize_memories,
                            softmax_gain,
                            storage_prob,
                            memory_decay_rate,
                            use_storage_node,
                            enable_learning,
                            learn_field_weights,
                            use_gating_for_weighting,
                            ):
        """Construct Nodes and Pathways for EMComposition"""

        # Construct Nodes --------------------------------------------------------------------------------

        field_weighting = len([weight for weight in field_weights if weight]) > 1 and not concatenate_keys

        # First, construct Nodes of Composition with their Projections
        self.query_input_nodes = self._construct_query_input_nodes(field_weights)
        self.value_input_nodes = self._construct_value_input_nodes(field_weights)
        self.input_nodes = self.query_input_nodes + self.value_input_nodes
        self.concatenate_keys_node = self._construct_concatenate_keys_node(concatenate_keys)
        self.match_nodes = self._construct_match_nodes(memory_template, memory_capacity,
                                                                   concatenate_keys,normalize_memories)
        self.softmax_nodes = self._construct_softmax_nodes(memory_capacity, field_weights, softmax_gain)
        self.field_weight_nodes = self._construct_field_weight_nodes(field_weights,
                                                                     concatenate_keys,
                                                                     use_gating_for_weighting)
        self.weighted_softmax_nodes = self._construct_weighted_softmax_nodes(memory_capacity, use_gating_for_weighting)
        self.softmax_gain_control_nodes = self._construct_softmax_gain_control_nodes(softmax_gain)
        self.combined_softmax_node = self._construct_combined_softmax_node(memory_capacity,
                                                                           field_weighting,
                                                                           use_gating_for_weighting)
        self.retrieved_nodes = self._construct_retrieved_nodes(memory_template)
        if use_storage_node:
            self.storage_node = self._construct_storage_node(memory_template, field_weights,
                                                             self.concatenate_keys_node,
                                                             memory_decay_rate, storage_prob)

        # Do some validation and get singleton Nodes for concatenated keys
        if self.concatenate_keys:
            softmax_node = self.softmax_nodes.pop()
            assert not self.softmax_nodes, \
                f"PROGRAM ERROR: Too many softmax_nodes ({len(self.softmax_nodes)}) for concatenated keys."
            assert len(self.softmax_gain_control_nodes) <= 1, \
                (f"PROGRAM ERROR: Too many softmax_gain_control_nodes "
                 f"{len(self.softmax_gain_control_nodes)}) for concatenated keys.")
            match_node = self.match_nodes.pop()
            assert not self.softmax_nodes, \
                f"PROGRAM ERROR: Too many match_nodes ({len(self.match_nodes)}) for concatenated keys."
            assert not self.field_weight_nodes, \
                f"PROGRAM ERROR: There should be no field_weight_nodes for concatenated keys."

        # Construct Pathways --------------------------------------------------------------------------------

        # Set up pathways WITHOUT PsyNeuLink learning pathways
        if not self.enable_learning:
            self.add_nodes(self.query_input_nodes + self.value_input_nodes)
            if self.concatenate_keys:
                self.add_nodes([self.concatenate_keys_node, match_node, softmax_node])
            else:
                self.add_nodes(self.match_nodes +
                               self.softmax_nodes +
                               self.field_weight_nodes +
                               self.weighted_softmax_nodes)
            self.add_nodes(self.softmax_gain_control_nodes +
                           [self.combined_softmax_node] +
                           self.retrieved_nodes)
            if use_storage_node:
                self.add_node(self.storage_node)
                # self.add_projections(proj for proj in self.storage_node.efferents)

        # Set up pathways WITH psyneulink backpropagation learning field weights
        else:

            # Key pathways
            for i in range(self.num_keys):
                # Regular pathways
                if not self.concatenate_keys:
                    pathway = [self.query_input_nodes[i],
                               self.match_nodes[i],
                               self.softmax_nodes[i],
                               self.combined_softmax_node]
                    if self.weighted_softmax_nodes:
                        pathway.insert(3, self.weighted_softmax_nodes[i])
                    # if self.softmax_gain_control_nodes:
                    #     pathway.insert(4, self.softmax_gain_control_nodes[i])
                # Key-concatenated pathways
                else:
                    pathway = [self.query_input_nodes[i],
                               self.concatenate_keys_node,
                               match_node,
                               softmax_node,
                               self.combined_softmax_node]
                    # if self.softmax_gain_control_nodes:
                    #     pathway.insert(4, self.softmax_gain_control_nodes[0]) # Only one, ensured above
                # self.add_backpropagation_learning_pathway(pathway)
                self.add_linear_processing_pathway(pathway)

            # softmax gain control is specified:
            for gain_control_node in self.softmax_gain_control_nodes:
                self.add_node(gain_control_node)

            # field_weights -> weighted_softmax pathways
            if self.field_weight_nodes:
                for i in range(self.num_keys):
                    # self.add_backpropagation_learning_pathway([self.field_weight_nodes[i],
                    #                                            self.weighted_softmax_nodes[i]])
                    self.add_linear_processing_pathway([self.field_weight_nodes[i], self.weighted_softmax_nodes[i]])

            self.add_nodes(self.value_input_nodes)

            # Retrieval pathways
            for i in range(len(self.retrieved_nodes)):
                # self.add_backpropagation_learning_pathway([self.combined_softmax_node, self.retrieved_nodes[i]])
                self.add_linear_processing_pathway([self.combined_softmax_node, self.retrieved_nodes[i]])

            # Storage Nodes
            if use_storage_node:
                self.add_node(self.storage_node)
                # self.add_projections(proj for proj in self.storage_node.efferents)

    def _construct_query_input_nodes(self, field_weights)->list:
        """Create one node for each key to be used as cue for retrieval (and then stored) in memory.
        Used to assign new set of weights for Projection for query_input_node[i] -> match_node[i]
        where i is selected randomly without replacement from (0->memory_capacity)
        """

        # Get indices of field_weights that specify keys:
        key_indices = np.nonzero(field_weights)[0]

        assert len(key_indices) == self.num_keys, \
            f"PROGRAM ERROR: number of keys ({self.num_keys}) does not match number of " \
            f"non-zero values in field_weights ({len(key_indices)})."

        query_input_nodes = [TransferMechanism(size=len(self.entry_template[key_indices[i]]),
                                             name=f'{self.key_names[i]} [QUERY]')
                       for i in range(self.num_keys)]

        return query_input_nodes

    def _construct_value_input_nodes(self, field_weights)->list:
        """Create one input node for each value to be stored in memory.
        Used to assign new set of weights for Projection for combined_softmax_node -> retrieved_node[i]
        where i is selected randomly without replacement from (0->memory_capacity)
        """

        # Get indices of field_weights that specify keys:
        value_indices = np.where(field_weights == 0)[0]

        assert len(value_indices) == self.num_values, \
            f"PROGRAM ERROR: number of values ({self.num_values}) does not match number of " \
            f"non-zero values in field_weights ({len(value_indices)})."

        value_input_nodes = [TransferMechanism(size=len(self.entry_template[value_indices[i]]),
                                               name= f'{self.value_names[i]} [VALUE]')
                           for i in range(self.num_values)]

        return value_input_nodes

    def _construct_concatenate_keys_node(self, concatenate_keys)->ProcessingMechanism:
        """Create node that concatenates the inputs for all keys into a single vector
        Used to create a matrix for Projectoin from match / memory weights from concatenate_node -> match_node
        """
        # One node that concatenates inputs from all keys
        if not concatenate_keys:
            return None
        else:
            return ProcessingMechanism(function=Concatenate,
                                       input_ports=[{NAME: 'CONCATENATE_KEYS',
                                                     SIZE: len(self.query_input_nodes[i].output_port.value),
                                                     PROJECTIONS: MappingProjection(
                                                         name=f'{self.key_names[i]} to CONCATENATE',
                                                         sender=self.query_input_nodes[i].output_port,
                                                         matrix=IDENTITY_MATRIX)}
                                                    for i in range(self.num_keys)],
                                       name='CONCATENATE KEYS')

    def _construct_match_nodes(self, memory_template, memory_capacity, concatenate_keys, normalize_memories)->list:
        """Create nodes that, for each key field, compute the similarity between the input and each item in memory.
        - If self.concatenate_keys is True, then all inputs for keys from concatenated_keys_node are assigned a single
            match_node, and weights from memory_template are assigned to a Projection from concatenated_keys_node to
            that match_node.
        - Otherwise, each key has its own match_node, and weights from memory_template are assigned to a Projection
            from each query_input_node[i] to each match_node[i].
        - Each element of the output represents the similarity between the query_input and one key in memory.
        """

        if concatenate_keys:
            # Get fields of memory structure corresponding to the keys
            # Number of rows should total number of elements over all keys,
            #    and columns should number of items in memory
            matrix =np.array([np.concatenate((memory_template[:,:self.num_keys][i]))
                              for i in range(memory_capacity)]).transpose()
            matrix = np.array(matrix.tolist())
            match_nodes = [
                TransferMechanism(
                    input_ports={NAME: 'CONCATENATED_INPUTS',
                                 SIZE: memory_capacity,
                                 PROJECTIONS: MappingProjection(sender=self.concatenate_keys_node,
                                                                matrix=matrix,
                                                                function=LinearMatrix(
                                                                    normalize=normalize_memories),
                                                                name=f'MEMORY')},
                    name='MATCH')]

        # One node for each key
        else:
            match_nodes = [
                TransferMechanism(
                    input_ports= {
                        SIZE:memory_capacity,
                        PROJECTIONS: MappingProjection(sender=self.query_input_nodes[i].output_port,
                                                       matrix = np.array(
                                                           memory_template[:,i].tolist()).transpose().astype(float),
                                                       function=LinearMatrix(normalize=normalize_memories),
                                                       name=f'MEMORY for {self.key_names[i]} [KEY]')},
                    name=f'{self.key_names[i]} [MATCH to KEYS]')
                for i in range(self.num_keys)
            ]

        return match_nodes

    def _construct_softmax_nodes(self, memory_capacity, field_weights, softmax_gain)->list:
        """Create nodes that, for each key field, compute the softmax over the similarities between the input and the
        memories in the corresponding match_node.
        """

        # Get indices of field_weights that specify keys:
        key_indices = np.where(np.array(field_weights) != 0)
        key_weights = [field_weights[i] for i in key_indices[0]]

        assert len(key_indices[0]) == self.num_keys, \
            f"PROGRAM ERROR: number of keys ({self.num_keys}) does not match number of " \
            f"non-zero values in field_weights ({len(key_indices)})."

        # If softmax_gain is specified as CONTROL, then set to None for now
        #    (will be set in _construct_softmax_gain_control_nodes)
        if softmax_gain == CONTROL:
            softmax_gain = None

        softmax_nodes = [TransferMechanism(input_ports={SIZE:memory_capacity,
                                                        PROJECTIONS: MappingProjection(
                                                            sender=match_node.output_port,
                                                            matrix=IDENTITY_MATRIX,
                                                            name=f'MATCH to SOFTMAX for {self.key_names[i]}')},
                                           function=SoftMax(gain=softmax_gain),
                                           name='SOFTMAX' if len(self.match_nodes) == 1
                                           else f'{self.key_names[i]} [SOFTMAX]')
                         for i, match_node in enumerate(self.match_nodes)]

        return softmax_nodes

    def _construct_softmax_gain_control_nodes(self, softmax_gain)->list:
        """Create nodes that set the softmax gain (inverse temperature) for each softmax_node."""

        softmax_gain_control_nodes = []
        if softmax_gain == CONTROL:
            softmax_gain_control_nodes = [ControlMechanism(monitor_for_control=match_node,
                                                      control_signals=[(GAIN, self.softmax_nodes[i])],
                                                      function=get_softmax_gain,
                                                      name='SOFTMAX GAIN CONTROL' if len(self.softmax_nodes) == 1
                                                      else f'SOFTMAX GAIN CONTROL {self.key_names[i]}')
                                     for i, match_node in enumerate(self.match_nodes)]

        return softmax_gain_control_nodes

    def _construct_field_weight_nodes(self, field_weights, concatenate_keys, use_gating_for_weighting)->list:
        """Create ProcessingMechanisms that weight each key's softmax contribution to the retrieved values."""

        field_weight_nodes = []

        if not concatenate_keys and self.num_keys > 1:
            if use_gating_for_weighting:
                field_weight_nodes = [GatingMechanism(input_ports={VARIABLE: field_weights[i],
                                                                   PARAMS:{DEFAULT_INPUT: DEFAULT_VARIABLE},
                                                                   NAME: 'OUTCOME'},
                                                      gate=[key_match_pair[1].output_ports[0]],
                                                      name= 'RETRIEVAL WEIGHTING' if self.num_keys == 1
                                                      else f'RETRIEVAL WEIGHTING {i}')
                                      for i, key_match_pair in enumerate(zip(self.query_input_nodes,
                                                                             self.softmax_nodes))]
            else:
                field_weight_nodes = [ProcessingMechanism(input_ports={VARIABLE: field_weights[i],
                                                                       PARAMS:{DEFAULT_INPUT: DEFAULT_VARIABLE},
                                                                       NAME: 'FIELD_WEIGHT'},
                                                          name= 'WEIGHT' if self.num_keys == 1
                                                          else f'{self.key_names[i]} [WEIGHT]')
                                      for i in range(self.num_keys)]
        return field_weight_nodes

    def _construct_weighted_softmax_nodes(self, memory_capacity, use_gating_for_weighting)->list:

        if use_gating_for_weighting:
            return []

        weighted_softmax_nodes = \
            [ProcessingMechanism(
                default_variable=[self.softmax_nodes[i].output_port.value,
                                  self.softmax_nodes[i].output_port.value],
                input_ports=[
                    {PROJECTIONS: MappingProjection(sender=sm_fw_pair[0],
                                                    matrix=IDENTITY_MATRIX,
                                                    name=f'SOFTMAX to WEIGHTED SOFTMAX for {self.key_names[i]}')},
                    {PROJECTIONS: MappingProjection(sender=sm_fw_pair[1],
                                                    matrix=FULL_CONNECTIVITY_MATRIX,
                                                    name=f'WEIGHT to WEIGHTED SOFTMAX for {self.key_names[i]}')}],
                function=LinearCombination(operation=PRODUCT),
                name=f'{self.key_names[i]} [WEIGHTED SOFTMAX]')
                for i, sm_fw_pair in enumerate(zip(self.softmax_nodes,
                                                   self.field_weight_nodes))]
        return weighted_softmax_nodes

    def _construct_combined_softmax_node(self,
                                         memory_capacity,
                                         field_weighting,
                                         use_gating_for_weighting
                                         )->ProcessingMechanism:
        """Create nodes that compute the weighting of each item in memory.
        """

        if not field_weighting or use_gating_for_weighting:
            # If use_gating_for_weighting, then softmax_nodes are output gated by gating nodes
            input_source = self.softmax_nodes
        else:
            input_source = self.weighted_softmax_nodes

        combined_softmax_node = (
            ProcessingMechanism(input_ports=[{SIZE:memory_capacity,
                                              # PROJECTIONS:[s for s in input_source]}],
                                              PROJECTIONS:[MappingProjection(sender=s,
                                                                             matrix=IDENTITY_MATRIX,
                                                                             name=f'WEIGHTED SOFTMAX to RETRIEVAL for '
                                                                                  f'{self.key_names[i]}')
                                                           for i, s in enumerate(input_source)]}],
                                name='RETRIEVE'))

        assert len(combined_softmax_node.output_port.value) == memory_capacity, \
            'PROGRAM ERROR: number of items in combined_softmax_node ' \
            '({len(combined_softmax_node.output_port)}) does not match memory_capacity ({self.memory_capacity})'

        return combined_softmax_node

    def _construct_retrieved_nodes(self, memory_template)->list:
        """Create nodes that report the value field(s) for the item(s) matched in memory.
        """

        self.retrieved_key_nodes = \
            [TransferMechanism(input_ports={SIZE: len(self.query_input_nodes[i].variable[0]),
                                            PROJECTIONS:
                                                MappingProjection(
                                                    sender=self.combined_softmax_node,
                                                    matrix=memory_template[:,i],
                                                    name=f'MEMORY FOR {self.key_names[i]} [RETRIEVE KEY]')
                                            },
                               name= f'{self.key_names[i]} [RETRIEVED]')
             for i in range(self.num_keys)]

        self.retrieved_value_nodes = \
            [TransferMechanism(input_ports={SIZE: len(self.value_input_nodes[i].variable[0]),
                                            PROJECTIONS:
                                                MappingProjection(
                                                    sender=self.combined_softmax_node,
                                                    matrix=memory_template[:,
                                                           i + self.num_keys],
                                                    name=f'MEMORY FOR {self.value_names[i]} [RETRIEVE VALUE]')},
                               name= f'{self.value_names[i]} [RETRIEVED]')
             for i in range(self.num_values)]

        return self.retrieved_key_nodes + self.retrieved_value_nodes

    def _construct_storage_node(self,
                                memory_template,
                                field_weights,
                                concatenate_keys_node,
                                memory_decay_rate,
                                storage_prob)->list:
        """Create EMStorageMechanism that stores the key and value inputs in memory.
        Memories are stored by adding the current input to each field to the corresponding row of the matrix for
        the Projection from the query_input_node (or concatenate_node) to the matching_node and retrieved_node for keys,
        and from the value_input_node to the retrieved_node for values. The `function <EMStorageMechanism.function>`
        of the `EMSorageMechanism` that takes the following arguments:

         - **variable* -- template for an `entry <EMComposition_Memory>` in `memory<EMComposition.memory>`;

         - **fields* -- the `input_nodes <EMComposition.input_nodes>` for the corresponding `fields
           <EMComposition_Fields>` of an `entry <EMCmposition_Memory>` in `memory <EMComposition.memory>`;

         - **field_types* -- a list of the same length as ``fields``, containing 1's for key fields and 0's for
           value fields;

         - **concatenate_keys_node* -- node used to concatenate keys (if `concatenate_keys
           <EMComposition.concatenate_keys>` is `True`) or None;

         - **memory_matrix* -- `memory_template <EMComposition.memory_template>`);

         - **learning_signals* -- list of ` `MappingProjection`\\s (or their ParameterPort`\\s) that store each
           `field <EMComposition_Fields>` of `memory <EMComposition.memory>`;

         - **decay_rate** -- rate at which entries in the `memory_matrix <EMComposition.memory_matrix>` decay;

         - **storage_prob** -- probability for storing an entry in `memory <EMComposition.memory>`.
        """

        learning_signals = [match_node.input_port.path_afferents[0]
                            for match_node in self.match_nodes] + \
                           [retrieved_node.input_port.path_afferents[0]
                            for retrieved_node in self.retrieved_nodes]

        storage_node = EMStorageMechanism(default_variable=[self.input_nodes[i].value[0]
                                                            for i in range(self.num_fields)],
                                          fields=[self.input_nodes[i] for i in range(self.num_fields)],
                                          field_types=[0 if weight == 0 else 1 for weight in field_weights],
                                          concatenation_node=self.concatenate_keys_node,
                                          memory_matrix=memory_template,
                                          learning_signals=learning_signals,
                                          storage_prob=storage_prob,
                                          decay_rate = memory_decay_rate,
                                          name='STORE')
        return storage_node

    def _set_learning_attributes(self):
        """Set learning-related attributes for Node and Projections
        """
        # self.require_node_roles(self.storage_node, NodeRole.LEARNING)

        for projection in self.projections:
            if (projection.sender.owner in self.field_weight_nodes
                    and self.enable_learning
                    and self.learn_field_weights):
                projection.learnable = True
            else:
                projection.learnable = False


    # *****************************************************************************************************************
    # *********************************** Execution Methods  **********************************************************
    # *****************************************************************************************************************

    def execute(self,
                inputs=None,
                context=None,
                **kwargs):
        """Set input to weights of Projections to match_nodes and retrieved_nodes if not use_storage_node."""
        results = super().execute(inputs=inputs, context=context, **kwargs)
        if not self.use_storage_node:
            self._store_memory(inputs, context)
        return results

    def _store_memory(self, inputs, context):
        """Store inputs in memory as weights of Projections to softmax_nodes (keys) and retrieved_nodes (values).
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
        purge_by_field_weights = False
        field_norms = np.array([np.linalg.norm(field, axis=1)
                                for field in [row for row in self.parameters.memory.get(context)]])
        if purge_by_field_weights:
            field_norms *= self.field_weights
        row_norms = np.sum(field_norms, axis=1)
        idx_of_min = np.argmin(row_norms)

        # If concatenate_keys is True, assign entry to col of matrix for Projection from concatenate_node to match_node
        if self.concatenate_keys_node:
            # Get entry to store from concatenate_keys_node
            entry_to_store = self.concatenate_keys_node.value[0]
            # Get matrix of weights for Projection from concatenate_node to match_node
            field_memories = self.concatenate_keys_node.efferents[0].parameters.matrix.get(context)
            # Decay existing memories before storage if memory_decay_rate is specified
            if self.memory_decay_rate:
                field_memories *= self.parameters.memory_decay_rate._get(context)
            # Assign input vector to col of matrix that has lowest norm (i.e., weakest memory)
            field_memories[:,idx_of_min] = np.array(entry_to_store)
            # Assign updated matrix to Projection
            self.concatenate_keys_node.efferents[0].parameters.matrix.set(field_memories, context)

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

    def learn(self, *args, **kwargs):
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

    def infer_backpropagation_learning_pathways(self, execution_mode, context=None):
        if self.concatenate_keys:
            raise EMCompositionError(f"EMComposition does not support learning with 'concatenate_keys'=True.")
        super().infer_backpropagation_learning_pathways(execution_mode, context=context)

    def _update_learning_parameters(self, context):
        pass

    def get_output_values(self, context=None):
        """Override to provide ordering of retrieved_nodes that matches order of inputs.
        This is needed since nodes were constructed as sets
        """
        return [retrieved_node.output_port.parameters.value.get(context)
                for retrieved_node in self.retrieved_nodes
                if (not self.output_CIM._sender_is_probe(self.output_CIM.port_map[retrieved_node.output_port][1])
                    or self.include_probes_in_output)]
