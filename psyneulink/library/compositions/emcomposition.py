# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* EMComposition *************************************************
#
# TODO:
# - QUESTION:
#   - SHOULD differential of SoftmaxGainControl Node be included in learning?
#   - SHOULD MEMORY DECAY OCCUR IF STORAGE DOES NOT? CURRENTLY IT DOES NOT (SEE EMStorage Function)

# - FIX: Concatenation:
# -      LLVM for function and derivative
# -      Add Concatenate to pytorchcreator_function
# -      Deal with matrix assignment in LearningProjection LINE 643
# -      Reinstate test for execution of Concatenate with learning in test_emcomposition (currently commented out)
# - FIX: Softmax Gain Control:
#        Test if it current works (they are added to Composition but not in BackProp processing pathway)
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
# - FIX: ADD NOISE
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
#        - MatrixTransform to add normalization
#        - _store() method to assign weights to memory
#        - LLVM problem with ComparatorMechanism
#
#      - pytorchcreator_function:
#           SoftMax implementation:  torch.nn.Softmax(dim=0) is not getting passed correctly
#           Implement LinearCombination
#        - MatrixTransform Function:
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
for retrieval,, and that adds the capability for `memory_decay <EMComposition.memory_decay_rate>`. Its `memory
<EMComposition.memory>` is configured using two arguments of its constructor: **memory_template** argument, that defines
how each entry in `memory <EMComposition.memory>` is structured (the number of fields in each entry and the length
of each field); and **field_weights** argument, that defines which fields are used as cues for retrieval, i.e., "keys",
including whether and how they are differentially weighted in the match process used for retrieval); and which
fields are treated as "values" that are stored retrieved, but not used by the match process. The inputs to an
EMComposition, corresponding to each key ("query") and value field are assigned to each of its `INPUT <NodeRole.INPUT>`
`Nodes <Composition_Nodes>` (listed in its `query_input_nodes <EMComposition.query_input_nodes>` and `value_input_nodes
<EMComposition.value_input_nodes>` attributes, respectively), and the retrieved values are represented as `OUTPUT
<NodeRole.OUTPUT>` `Nodes <Composition_Nodes>` of the EMComposition.  The `memory <EMComposition.memory>` can be
accessed using its `memory <EMComposition.memory>` attribute.

    .. technical_note::
       The memories of an EMComposition are actually stored in the `matrix <EMComposition_Memory>` attribute of a
       set of `MappingProjections <MappingProjection>` (see `note below <EMComposition_Memory_Storage>`).  The `memory
       <EMComposition.memory>` attribute compiles and formats these as a single 3d array, the rows of which (axis 0)
       are each entry, the columns of which (axis 1) are the fields of each entry, and the items of which (axis 2)
       are the values of each field (see `EMComposition_Memory` for additional details).

.. _EMComposition_Organization:

**Organization**

.. _EMComposition_Entries_and_Fields:

*Entries and Fields*. Each entry in memory can have an arbitrary number of fields, and each field can have an arbitrary
length.  However, all entries must have the same number of fields, and the corresponding fields must all have the same
length across entries. Each field is treated as a separate "channel" for storage and retrieval, and is associated with
its own corresponding input (key or value) and output (retrieved value) `Node <Composition_Nodes>` some or all of
which can be used to compute the similarity of the input (key) to entries in memory, that is used for retreieval.
Fields can be differentially weighted to determine the influence they have on retrieval, using the `field_weights
<ContentAddressableMemory.memory>` parameter (see `retrieval <EMComposition_Retrieval_Storage>` below). The number and
shape of the fields in each entry is specified in the **memory_template** argument of the EMComposition's constructor
(see `memory_template <EMComposition_Fields>`). Which fields treated as keys (i.e., matched against queries during
retrieval) and which are treated as values (i.e., retrieved but not used for matching retrieval) is specified in the
**field_weights** argument of the EMComposition's constructor (see `field_weights <EMComposition_Field_Weights>`).

.. _EMComposition_Operation:

**Operation**

*Retrieval.*  The values retrieved from `memory <ContentAddressableMemory.memory>` (one for each field) are based
on the relative similarity of the keys to the entries in memory, computed as the distance of each key and the
values in the corresponding field for each entry in memory. By default, normalized dot products (comparable to cosine
similarity) are used to compute the similarity of each query to each key in memory. These distances are then
weighted by the corresponding `field_weights <EMComposition.field_weights>` for each field (if specified) and then
summed, and the sum is softmaxed to produce a softmax distribution over the entries in memory. That is then used to
generate a softmax-weighted average of the retrieved values across all fields, which is returned as the `result
<Composition.result>` of the EMComposition's `execution <Composition_Execution>` (an EMComposition can also be
configured to return the entry with the lowest distance weighted by field, however then it is not compatible
with learning; see `softmax_choice <EMComposition_Softmax_Choice>`).

  COMMENT:
  TBD DISTANCE ATTRIBUTES:
  The distances used for the last retrieval is stored in XXXX and the distances of each of their corresponding fields
  (weighted by `distance_field_weights <ContentAddressableMemory.distance_field_weights>`), are returned in XXX,
  respectively.
  COMMENT

*Storage.*  The `inputs <Composition_Input_External_InputPorts>` to the EMComposition's fields are stored in `memory
<EMComposition.memory>` after each execution, with a probability determined by `storage_prob
<EMComposition.storage_prob>`.  If `memory_decay_rate <EMComposition.memory_decay_rate>` is specified, then the `memory
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
  The **memory_template** argument can be specified in one of three ways (see `EMComposition_Examples` for
  representative use cases):

  * **tuple**: interpreted as an np.array shape specification, that must be of length 2 or 3.  If it is a 3-item tuple,
    then the first item specifies the number of entries in memory, the 2nd the number of fields in each entry, and the
    3rd the length of each field.  If it is a 2-item tuple, this specifies the shape of an entry, and the number of
    entries is specified by `memory_capacity <EMComposition_Memory_Capacity>`).  All entries are
    filled with zeros or the value specified by `memory_fill <EMComposition_Memory_Fill>`.

    .. warning::
       If **memory_template** is specified with a 3-item tuple and `memory_capacity <EMComposition_Memory_Capacity>`
       is also specified with a value that does not match the first item of **memory_template**, and error is
       generated indicating the conflict in the number of entries specified.

    .. hint::
       To specify a single field, a list or array must be used (see below), as a 2-item tuple is interpreted as
       specifying the shape of an entry, and so it can't be used to specify the number of entries each of which
       has a single field.

  * **2d list or array**: interpreted as a template for memory entries. This can be used to specify fields of
    different lengths (i.e., entries that are ragged arrays), with each item in the list (axis 0 of the array) used
    to specify the length of the corresponding field.  The template is then used to initialze all entries in `memory
    <EMComposition.memory>`.  If the template includes any non-zero elements, then the array is replicated for all
    entries in `memory <EMComposition.memory>`; otherwise, they are filled with either zeros or the value specified
    in `memory_fill <EMComposition_Memory_Fill>`.

    .. hint::
       To specify a single entry, with all other entries filled with zeros
       or the value specified in **memory_fill**, use a 3d array as described below.

  * **3d list or array**: used to initialize `memory <EMComposition.memory>` directly with the entries specified in
    the outer dimension (axis 0) of the list or array.  If `memory_capacity <EMComposition_Memory_Capacity>` is not
    specified, then it is set to the number of entries in the list or array. If **memory_capacity** *is* specified,
    then the number of entries specified in **memory_template** must be less than or equal to **memory_capacity**.  If
    is less than **memory_capacity**, then the remaining entries in `memory <EMComposition.memory>` are filled with
    zeros or the value specified in **memory_fill** (see below):  if all of the entries specified contain only
    zeros, and **memory_fill** is specified, then the matrix is filled with the value specified in **memory_fill**;
    otherwise, zeros are used to fill all entries.

.. _EMComposition_Memory_Capacity:

*Memory Capacity*

* **memory_capacity**: specifies the number of items that can be stored in the EMComposition's memory; when
  `memory_capacity <EMComposition.memory_capacity>` is reached, each new entry overwrites the weakest entry (i.e., the
  one with the smallest norm across all of its fields) in `memory <EMComposition.memory>`.  If `memory_template
  <EMComposition_Memory_Template>` is specified as a 3-item tuple or 3d list or array (see above), then that is used
  to determine `memory_capacity <EMComposition.memory_capacity>` (if it is specified and conflicts with either of those
  an error is generated).  Otherwise, it can be specified using a numerical value, with a default of 1000.  The
  `memory_capacity <EMComposition.memory_capacity>` cannot be modified once the EMComposition has been constructed.

.. _EMComposition_Memory_Fill:

* **memory_fill**: specifies the value used to fill the `memory <EMComposition.memory>`, based on the shape specified
  in the **memory_template** (see above).  The value can be a scalar, or a tuple to specify an interval over which
  to draw random values to fill `memory <EMComposition.memory>` --- both should be scalars, with the first specifying
  the lower bound and the second the upper bound.  If **memory_fill** is not specified, and no entries are specified
  in **memory_template**, then `memory <EMComposition.memory>` is filled with zeros.

  .. hint::
     If memory is initialized with all zeros and **normalize_memories** set to ``True`` (see `below
     <EMComposition_Retrieval_Storage>`) then a numpy.linalg warning is issued about divide by zero.
     This can be ignored, as it does not affect the results of execution, but it can be averted by specifying
     `memory_fill <EMComposition_Memory_Fill>` to use small random values (e.g., ``memory_fill=(0,.001)``).

.. _EMComposition_Field_Weights:

* **field_weights**: specifies which fields are used as keys, and how they are weighted during retrieval. The
  number of entries specified must match the number of fields specified in **memory_template** (i.e., the size of
  of its first dimension (axis 0)). All non-zero entries must be positive; these designate *keys* -- fields
  that are used to match queries against entries in memory for retrieval (see `Match memories by field
  <EMComposition_Processing>`). Entries of 0 designate *values* -- fields that are ignored during the matching
  process, but the values of which are retrieved and assigned as the `value <Mechanism_Base.value>` of the
  corresponding `retrieved_node <EMComposition.retrieved_nodes>`. This distinction between keys and value corresponds
  to the format of a standard "dictionary," though in that case only a single key and value are allowed, whereas
  here there can be one or more keys and any number of values; if all fields are keys, this implements a full form of
  content-addressable memory. If **learn_field_weights** is True (and `enable_learning<EMComposition.enable_learning>`
  is either True or a list with True for at least one entry), then the field_weights can be modified during training
  (this functions similarly to the attention head of a Transformer model, although at present the field can only be
  scalar values rather than vecdtors); if **learn_field_weights** is False, then the field_weights are fixed.
  The following options can be used to specify **field_weights**:

    * *None* (the default): all fields except the last are treated as keys, and are weighted equally for retrieval,
      while the last field is treated as a value field;

    * *single entry*: all fields are treated as keys (i.e., used for retrieval) and weighted equally for retrieval.
      if `normalize_field_weights <EMComposition_Normalize_Field_Weights>` is True, the value is ignored and all
      of keys are weighted by 1 / number of keys (i.e., normalized), whereas if `normalize_field_weights
      <EMComposition_Normalize_Field_Weights>` is False, then the value specified is used to weight the retrieval of
      every keys.

    * *multiple non-zero entries*: If all entries are identical, the value is ignored and the corresponding keys
      are weighted equally for retrieval; if the non-zero entries are non-identical, they are used to weight the
      corresponding fields during retrieval (see `Weight fields <EMComposition_Processing>`).  In either case,
      the remaining fields (with zero weights) are treated as value fields.

.. _EMComposition_Normalize_Field_Weights:

* **normalize_field_weights**: specifies whether the `field_weights <EMComposition.field_weights>` are normalized
    or their raw values are used.  If True, the `field_weights <EMComposition.field_weights>` are normalized so that
    they sum to 1.0, and are used to weight (i.e., multiply) the corresponding fields during retrieval (see `Weight
    fields <EMComposition_Processing>`). If False, the raw values of the `field_weights <EMComposition.field_weights>`
    are used to weight the retrieved value of each field. This setting is ignored if **field_weights**
    is None or `concatenate_queries <EMComposition_Concatenate_Queries>` is in effect.

.. _EMComposition_Field_Names:

* **field_names**: specifies names that can be assigned to the fields.  The number of names specified must
  match the number of fields specified in the memory_template.  If specified, the names are used to label the
  nodes of the EMComposition.  If not specified, the fields are labeled generically as "Key 0", "Key 1", etc..

.. _EMComposition_Concatenate_Queries:

* **concatenate_queries**:  specifies whether keys are concatenated before a match is made to items in memory.
  This is False by default. It is also ignored if the `field_weights <EMComposition.field_weights>` for all keys are
  not all equal (i.e., all non-zero weights are not equal -- see `field_weights <EMComposition_Field_Weights>`) and/or
  `normalize_memories <EMComposition.normalize_memories>` is set to False. Setting concatenate_queries to True in either
  of those cases issues a warning, and the setting is ignored. If the key `field_weights <EMComposition.field_weights>`
  (i.e., all non-zero values) are all equal *and* **normalize_memories** is set to True, then setting
  **concatenate_queries** causes a `concatenate_queries_node <EMComposition.concatenate_queries_node>` to be created
  that receives input from all of the `query_input_nodes <EMComposition.query_input_nodes>` and passes them as a single
  vector to the `mactch_node <EMComposition.match_nodes>`.

      .. note::
         While this is computationally more efficient, it can affect the outcome of the `matching process
         <EMComposition_Processing>`, since computing the distance of a single vector comprised of the concatentated
         inputs is not identical to computing the distance of each field independently and then combining the results.

      .. note::
         All `query_input_nodes <EMComposition.query_input_nodes>` and `retrieved_nodes <EMComposition.retrieved_nodes>`
         are always preserved, even when `concatenate_queries <EMComposition.concatenate_queries>` is True, so that
         separate inputs can be provided for each key, and the value of each key can be retrieved separately.

.. _EMComposition_Memory_Decay_Rate

* **memory_decay_rate**: specifies the rate at which items in the EMComposition's memory decay;  the default rate
  is *AUTO*, which sets it to  1 / `memory_capacity <EMComposition.memory_capacity>`, such that the oldest memories
  are the most likely to be replaced when `memory_capacity <EMComposition.memory_capacity>` is reached.  If
  **memory_decay_rate** is set to 0 None or False, then memories do not decay and, when `memory_capacity
  <EMComposition.memory_capacity>` is reached, the weakest memories are replaced, irrespective of order of entry.

.. _EMComposition_Retrieval_Storage:

*Retrieval and Storage*

* **storage_prob** : specifies the probability that the inputs to the EMComposition will be stored as an item in
  `memory <EMComposition.memory>` on each execution.

* **normalize_memories** : specifies whether queries and keys in memory are normalized before computing their dot
  products.

.. _EMComposition_Softmax_Gain:

* **softmax_gain** : specifies the gain (inverse temperature) used for softmax normalizing the combined distances
  used for retrieval (see `EMComposition_Execution` below).  The following options can be used:

  * numeric value: the value is used as the gain of the `SoftMax` Function for the EMComposition's
    `softmax_node <EMComposition.softmax_node>`.

  * *ADAPTIVE*: the `adapt_gain <SoftMax.adapt_gain>` method of the `SoftMax` Function is used to adaptively set
    the `softmax_gain <EMComposition.softmax_gain>` based on the entropy of the distances, in order to preserve
    the distribution over non- (or near) zero entries irrespective of how many (near) zero entries there are
    (see `Thresholding and Adaptive Gain <SoftMax_AdaptGain>` for additional details).

  * *CONTROL*: a `ControlMechanism` is created, and its `ControlSignal` is used to modulate the `softmax_gain
    <EMComposition.softmax_gain>` parameter of the `SoftMax` function of the EMComposition's `softmax_node
    <EMComposition.softmax_node>`.

  If *None* is specified, the default value for the `SoftMax` function is used.

.. _EMComposition_Softmax_Threshold:

* **softmax_threshold**: if this is specified, and **softmax_gain** is specified with a numeric value,
  then any values below the specified threshold are set to 0 before the distances are softmaxed
  (see *mask_threhold* under `Thresholding and Adaptive Gain <SoftMax_AdaptGain>` for additional details).

.. _EMComposition_Softmax_Choice:

* **softmax_choice** : specifies how the `SoftMax` Function of the EMComposition's `softmax_node
  <EMComposition.softmax_node>` is used, with the combined distances, to generate a retrieved item;
  the following are the options that can be used and the retrieved value they produce:

  * *WEIGHTED_AVG* (default): softmax-weighted average based on combined distances of queries and keys in memory.

  * *ARG_MAX*: entry with the smallest distance (one with lowest index in `memory <EMComposition.memory>`)\
               if there are identical ones).

  * *PROBABISTIC*: probabilistically chosen entry based on softmax-transformed distribution of combined distance.

  .. warning::
     Use of the *ARG_MAX* and *PROBABILISTIC* options is not compatible with learning, as these implement a discrete
     choice and thus are not differentiable. Constructing an EMComposition with **softmax_choice** set to either of
     these options and **enable_learning** set to True (or a list with any True entries) will generate a warning, and
     calling the EMComposition's `learn <Composition.learn>` method will generate an error; it must be changed to
     *WEIGHTED_AVG* to execute learning.

  .. technical_note::
     The *WEIGHTED_AVG* option is passed as *ALL* to the **output** argument of the `SoftMax` Function, *ARG_MAX* is
     passed as *ARG_MAX_INDICATOR*; and *PROBALISTIC* is passed as *PROB_INDICATOR*; the other SoftMax options are
     not currently supported.

.. _EMComposition_Learning:

*Learning*

EMComposition supports two forms of learning -- error backpropagation and the learning of `field_weights
<EMComposition.field_weights>` -- that can be configured by the following arguments of the EMComposition's constructor:

* **enable_learning** : specifies whether learning is enabled for the EMComposition and, if so, which `retrieved_nodes
    <EMComposition.retrieved_nodes>` are used to compute errors, and propagate these back through the network. If
    **enable_learning** is False, then no learning occurs, including of `field_weights <EMComposition.field_weights>`).
    If it is True, then all of the `retrieved_nodes <EMComposition.retrieved_nodes>` participate in learning:  For
    those that do not project to an outer Composition (i.e., one in which the EMComposition is `nested
    <Composition_Nested>`), a `TARGET <NodeRole.TARGET>` node is constructed for each, and used to compute errors that
    are backpropagated through the network to its `query_input_nodes <EMComposition.query_input_nodes>` and
    `value_input_nodes <EMComposition.value_input_nodes>`, and on to any nodes that project to it from a composition
    in which the EMComposition is `nested <Composition_Nested>`; retrieved_nodes that *do* project to an outer
    Composition receive their errors from those nodes, which are also backpropagated through the EMComposition.
    If **enable_learning** is a list, then only the `retrieved_nodes <EMComposition.retrieved_nodes>` specified in the
    list participate in learning, and errors are computed only for those nodes.  The list must contain the same
    number of entries as there are `fields <EMComposition_Fields>` and corresponding `retreived_nodes
    <EMComposition.retrieved_nodes>`, and each entry must be a boolean that specifies whether the corresponding
    `retrieved_node <EMComposition.retrieved_nodes>` is used for learning.

* **learn_field_weights** : specifies whether `field_weights <EMComposition.field_weights>` are modifiable during
    learning (see `field_weights <EMComposition.field_weights>` and `Learning <EMComposition_Learning>` for additional
    information.  For learning of `field_weights <EMComposition.field_weights>` to occur, **enable_learning** must
    also be True, or it must be a list with at least one True entry.  If **learn_field_weights** is True,
    **use_gating_for_weighting** must be False (see `note <EMComposition_Gating_For_Weighting>`).

* **learning_rate** : specifies the rate at which  `field_weights <EMComposition.field_weights>` are learned if
  **learn_field_weights** is True; see `Learning <EMComposition_Learning>` for additional information.

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
~~~~~~~~

The `memory <EMComposition.memory>` attribute contains a record of the entries in the EMComposition's memory. This
is in the form of a 3d array, in which rows (axis 0) are entries, columns (axis 1) are fields, and items (axis 2) are
the values of an entry in a given field.  The number of fields is determined by the `memory_template
<EMComposition_Memory_Template>` argument of the EMComposition's constructor, and the number of entries is determined
by the `memory_capacity <EMComposition_Memory_Capacity>` argument.

  .. _EMComposition_Memory_Storage:
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

.. _EMComposition_Output:

*Output*
~~~~~~~~

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
  distance with the corresponding field of each entry in `memory <EMComposition.memory>`.  In this way, each
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
  the result of which is possed to the corresponding `match_node <EMComposition.match_nodes>`. By default, the
  distance is computed as the normalized dot product (i.e., between the normalized query vector and the normalized
  key for the corresponding `field <EMComposition_Fields>`, that is comparable to using cosine similarity). However,
  if `normalize_memories <EMComposition.normalize_memories>` is set to False, just the raw dot product is computed.
  The distance can also be customized by specifying a different `function <MappingProjection.function>` for the
  `MappingProjection` to the `match_node <EMComposition.match_nodes>`. The result is assigned as the `value
  <Mechanism_Base.value>` of the corresponding `match_node <EMComposition.match_nodes>`.

.. _EMComposition_Field_Weighting:

* **Weight distances**. If `field weights <EMComposition_Field_Weights>` are specified, then the distance computed
  by the `MappingProjection` to each `match_node <EMComposition.match_nodes>` is multiplied by the corresponding
  `field_weight <EMComposition.field_weights>` using the `field_weight_node <EMComposition.field_weight_nodes>`.
  By default (if `use_gating_for_weighting <EMComposition.use_gating_for_weighting>` is False), this is done using
  the `weighted_match_nodes <EMComposition.weighted_match_nodes>`, each of which receives a Projection from a
  `match_node <EMComposition.match_nodes>` and the corresponding `field_weight_node <EMComposition.field_weight_nodes>`
  and multiplies them to produce the weighted distance for that field as its output.  However, if
  `use_gating_for_weighting <EMComposition.use_gating_for_weighting>` is True, the `field_weight_nodes` are implemented
  as `GatingMechanisms <GatingMechanism>`, each of which uses its `field weight <EMComposition.field_weights>` as a
  `GatingSignal <GatingSignal>` to output gate (i.e., multiplicatively modulate the output of) the corresponding
  `match_node <EMComposition.match_nodes>`. In this case, the `weighted_match_nodes` are not implemented,
  and the output of the `match_node <EMComposition.match_nodes>` is passed directly to the `combined_matches_node
  <EMComposition.combined_matches_node>`.


  .. _EMComposition_Gating_For_Weighting:
  .. note::
     Setting `use_gating_for_weighting <EMComposition.use_gating_for_weighting>` to True reduces the size and
     complexity of the EMComposition, by eliminating the `weighted_match_nodes <EMComposition.weighted_match_nodes>`.
     However, doing to precludes the ability to learn the `field_weights <EMComposition.field_weights>`,
     since `GatingSignals <GatingSignal>` are  `ModulatorySignal>` that cannot be learned.  If learning is required,
     then `use_gating_for_weighting` should be set to False.

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
  `gain <Softmax.gain>` parameter; if None is specified, the default value of the `Softmax` Function is used as the
  `gain <Softmax.gain>` parameter (see `Softmax_Gain <EMComposition_Softmax_Gain>` for additional  details).

* **Retrieve values by field**. The vector of softmax weights for each memory generated by the `softmax_node
  <EMComposition.softmax_node>` is passed through the Projections to the each of the `retrieved_nodes
  <EMComposition.retrieved_nodes>` to compute the retrieved value for each field.

* **Decay memories**.  If `memory_decay <EMComposition.memory_decay>` is True, then each of the memories is decayed
  by the amount specified in `memory_decay_rate <EMComposition.memory_decay_rate>`.

    .. technical_note::
       This is done by multiplying the `matrix <MappingProjection.matrix>` parameter of the `MappingProjection` from
       the `combined_matches_node <EMComposition.combined_matches_node>` to each of the `retrieved_nodes
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
       of the `MappingProjection` from the `combined_matches_node <EMComposition.combined_matches_node>` to each
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

.. _EMComposition_Training:

*Training*
~~~~~~~~~~

If `learn <Composition.learn>` is called, `enable_learning <EMComposition.enable_learning>` is True or a list with
any True entries, then errors will be computed for each of the `retrieved_nodes <EMComposition.retrieved_nodes>`
that is specified for learning (see `Learning <EMComposition_Learning>` for details about specification). These errors
are derived either from any errors backprpated to the EMComposition from an outer Composition in which it is `nested
<Composition_Nested>`, or locally by the difference between the `retrieved_nodes <EMComposition.retrieved_nodes>`
and the `target_nodes <EMComposition.target_nodes>` that are created for each of the `retrieved_nodes
<EMComposition.retrieved_nodes>` that do not project to an outer Composition.  These errors are then backpropagated
through the EMComposition to the `query_input_nodes <EMComposition.query_input_nodes>` and `value_input_nodes
<EMComposition.value_input_nodes>`, and on to any nodes that project to it from a composition in which the
EMComposition is `nested <Composition_Nested>`.

If `learn_field_weights <EMComposition.learn_field_weights>` is also True, then the `field_weights
<EMComposition.field_weights>` are modified to minimize the error passed to the EMComposition retrieved nodes, using the
`learning_rate <EMComposition.learning_rate>` specified in the `learning_rate <EMComposition.learning_rate>` attribute.
If `learn_field_weights <EMComposition.learn_field_weights>` is False (or `run <Composition.run>` is called, then the
If `learn_field_weights <EMComposition.learn_field_weights>` is False), then the `field_weights
<EMComposition.field_weights>` are not modified and the EMComposition is simply executed
without any modification, and error signals are passed to the nodes that project to its `query_input_nodes
<EMComposition.query_input_nodes>` and `value_input_nodes <EMComposition.value_input_nodes>`.

  .. note::
     The only parameters modifable by learning in the EMComposition are its `field_weights
     <EMComposition.field_weights>`; all other parameters (including all other Projection `matrices
     <MappingProjection.matrix>`) are fixed, and used only to compute gradients and backpropagate errors.

  .. technical_note::
     Although memory storage is implemented as a form of learning (though modification of MappingProjection
     `matrix <MappingProjection.matrix>` parameters; see `memory storage <EMComposition_Memory_Storage>`),
     this occurs irrespective of how EMComposition is run (i.e., whether `learn <Composition.learn>` or `run
     <Composition.run>` is called), and is not affected by the `learn_field_weights <EMComposition.learn_field_weights>`
     or `learning_rate <EMComposition.learning_rate>` attributes, which pertain only to whether the `field_weights
     <EMComposition.field_weights>` are modified during learning.  Furthermore, when run in PyTorch mode, storage
     is executed after the forward() and backward() passes are complete, and is not considered as part of the
     gradient calculations.

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
<EMComposition.normalize_memories>` is True, which will cause a divide by zero warning when memories are
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

Note that the figure now shows `<QUERY> [WEIGHT] <EMComposition.field_weight_nodes>` `nodes <Composition_Node>`,
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
and `normalize_memories <EMComposition.normalize_memories>` is `True`, then the keys are concatenated and are
concatenated for efficiency of processing.  This can be suppressed by specifying `concatenate_queries` as `False`
(see `concatenate_queries <EMComposition_Concatenate_Queries>` for additional details).
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
from psyneulink.core.components.functions.nonstateful.transferfunctions import SoftMax
from psyneulink.core.components.functions.nonstateful.transformfunctions import (
    Concatenate, LinearCombination, MatrixTransform)
from psyneulink.core.components.functions.function import DEFAULT_SEED, _random_state_getter, _seed_setter
from psyneulink.core.compositions.composition import CompositionError, NodeRole
from psyneulink.library.compositions.autodiffcomposition import AutodiffComposition, torch_available
from psyneulink.library.components.mechanisms.modulatory.learning.EMstoragemechanism import EMStorageMechanism
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.modulatory.control.gating.gatingmechanism import GatingMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.context import handle_external_context
from psyneulink.core.globals.keywords import \
    (ADAPTIVE, ALL, ARG_MAX, ARG_MAX_INDICATOR, AUTO, CONTEXT, CONTROL, DEFAULT_INPUT, DEFAULT_VARIABLE, DOT_PRODUCT,
     EM_COMPOSITION, FULL_CONNECTIVITY_MATRIX, GAIN, IDENTITY_MATRIX, INPUT_SHAPES, L0,
     MULTIPLICATIVE_PARAM, NAME, PARAMS, PROB_INDICATOR, PRODUCT, PROJECTIONS, RANDOM, VARIABLE)
from psyneulink.core.globals.utilities import convert_all_elements_to_np_array, is_numeric_scalar
from psyneulink.core.globals.registry import name_without_suffix
from psyneulink.core.llvm import ExecutionMode


__all__ = ['EMComposition', 'EMCompositionError', 'WEIGHTED_AVG', 'PROBABILISTIC']

STORAGE_PROB = 'storage_prob'
WEIGHTED_AVG = ALL
PROBABILISTIC = PROB_INDICATOR

QUERY_NODE_NAME = 'QUERY'
QUERY_AFFIX = f' [{QUERY_NODE_NAME}]'
VALUE_NODE_NAME = 'VALUE'
VALUE_AFFIX = f' [{VALUE_NODE_NAME}]'
MATCH = 'MATCH'
MATCH_AFFIX = f' [{MATCH}]'
MATCH_TO_KEYS_NODE_NAME = f'{MATCH} to KEYS'
WEIGHT = 'WEIGHT'
WEIGHT_AFFIX = f' [{WEIGHT}]'
MATCH_TO_KEYS_AFFIX = f' [{MATCH_TO_KEYS_NODE_NAME}]'
WEIGHTED_MATCH_NODE_NAME = 'WEIGHTED MATCH'
WEIGHTED_MATCH_AFFIX = f' [{WEIGHTED_MATCH_NODE_NAME}]'
CONCATENATE_QUERIES_NAME = 'CONCATENATE QUERIES'
COMBINE_MATCHES_NODE_NAME = 'COMBINE MATCHES'
COMBINE_MATCHES_AFFIX = f' [{COMBINE_MATCHES_NODE_NAME}]'
SOFTMAX_NODE_NAME = 'RETRIEVE'
SOFTMAX_AFFIX = f' [{SOFTMAX_NODE_NAME}]'
RETRIEVED_NODE_NAME = 'RETRIEVED'
RETRIEVED_AFFIX = ' [RETRIEVED]'
STORE_NODE_NAME = 'STORE'

def _memory_getter(owning_component=None, context=None)->list:
    """Return list of memories in which rows (outer dimension) are memories for each field.
    These are derived from `matrix <MappingProjection.matrix>` parameter of the `afferent
    <Mechanism_Base.afferents>` MappingProjections to each of the `2472s <EMComposition.retrieved_nodes>`.
    """

    # If storage_node (EMstoragemechanism) is implemented, get memory from that
    if owning_component.is_initializing:
        return None
    if owning_component._use_storage_node:
        return owning_component.storage_node.parameters.memory_matrix.get(context)

    # Otherwise, get memory from Projection(s) to each retrieved_node
    memory = [retrieved_node.path_afferents[0].parameters.matrix.get(context)
              for retrieved_node in owning_component.retrieved_nodes]
    # Reorganize memory so that each row is an entry and each column is a field
    memory_capacity = owning_component.memory_capacity or owning_component.defaults.memory_capacity
    return convert_all_elements_to_np_array([
        [memory[j][i] for j in range(owning_component.num_fields)]
        for i in range(memory_capacity)
    ])

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
        normalize_field_weights=True,   \
        field_names=None,               \
        concatenate_queries=False,      \
        normalize_memories=True,        \
        softmax_gain=THRESHOLD,         \
        storage_prob=1.0,               \
        memory_decay_rate=AUTO,         \
        enable_learning=True,           \
        learn_field_weights=True,       \
        learning_rate=True,             \
        use_gating_for_weighting=False, \
        name="EM_Composition"           \
        )

    Subclass of `AutodiffComposition` that implements the functions of an `EpisodicMemoryMechanism` in a
    differentiable form and in which it's `field_weights <EMComposition.field_weights>` parameter can be learned.

    Takes only the following arguments, all of which are optional

    Arguments
    ---------

    memory_template : tuple, list, 2d or 3d array : default [[0],[0]]
        specifies the shape of an item to be stored in the EMComposition's memory;
        see `memory_template <EMComposition_Memory_Template>` for details.

    memory_fill : scalar or tuple : default 0
        specifies the value used to fill the memory when it is initialized;
        see `memory_fill <EMComposition_Memory_Fill>` for details.

    memory_capacity : int : default None
        specifies the number of items that can be stored in the EMComposition's memory;
        see `memory_capacity <EMComposition_Memory_Capacity>` for details.

    field_weights : tuple : default (1,0)
        specifies the relative weight assigned to each key when matching an item in memory;
        see `field weights <EMComposition_Field_Weights>` for additional details.

    normalize_field_weights : bool : default True
        specifies whether the **fields_weights** are normalized over the number of keys, or used as absolute
        weighting values when retrieving an item from memory; see `normalize_field weights
        <EMComposition_Normalize_Field_Weights>` for additional details.

    field_names : list : default None
        specifies the optional names assigned to each field in the memory_template;
        see `field names <EMComposition_Field_Names>` for details.

    concatenate_queries : bool : default False
        specifies whether to concatenate the keys into a single field before matching them to items in
        the corresponding fields in memory; see `concatenate keys <EMComposition_Concatenate_Queries>` for details.

    normalize_memories : bool : default True
        specifies whether keys and memories are normalized before computing their dot product (similarity);
        see `Match memories by field <EMComposition_Processing>` for additional details.

    softmax_gain : float, ADAPTIVE or CONTROL : default 1.0
        specifies the temperature used for softmax normalizing the distance of queries and keys in memory;
        see `Softmax normalize matches over fields <EMComposition_Processing>` for additional details.

    softmax_threshold : float : default .0001
        specifies the threshold used to mask out small values in the softmax calculation;
        see *mask_threshold* under `Thresholding and Adaptive Gain <SoftMax_AdaptGain>` for details).

    softmax_choice : WEIGHTED_AVG, ARG_MAX, PROBABILISTIC : default WEIGHTED_AVG
        specifies how the softmax over distances of queries and keys in memory is used for retrieval;
        see `softmax_choice <EMComposition_Softmax_Choice>` for a description of each option.

    storage_prob : float : default 1.0
        specifies the probability that an item will be stored in `memory <EMComposition.memory>`
        when the EMComposition is executed (see `Retrieval and Storage <EMComposition_Storage>` for
        additional details).

    memory_decay_rate : float : AUTO
        specifies the rate at which items in the EMComposition's memory decay;
        see `memory_decay_rate <EMComposition_Memory_Decay_Rate>` for details.

    enable_learning : bool or list[bool]: default True
        specifies whether a learning pathway is constructed for each `field <EMComposition_Entries_and_Fields>`
        of the EMComposition.  If it is a list, each item must be ``True`` or ``False`` and the number of items
        must be equal to the number of `fields <EMComposition_Fields> specified; see `enable_learning
        <EMComposition.enable_learning>` for additional details.

    learn_field_weights : bool : default True
        specifies whether `field_weights <EMComposition.field_weights>` are learnable during training;
        requires **enable_learning** to be True to have any effect, and **use_gating_for_weighting** must be False;
        see `learn_field_weights <EMComposition_Learning>` for additional details.

    learning_rate : float : default .01
        specifies rate at which `field_weights <EMComposition.field_weights>` are learned
        if `learn_field_weights <EMComposition.learn_field_weights>` is True.

    # 7/10/24 FIX: STILL TRUE?  DOES IT PRECLUDE USE OF EMComposition as a nested Composition??
    .. technical_note::
        use_storage_node : bool : default True
            specifies whether to use a `LearningMechanism` to store entries in `memory <EMComposition.memory>`.
            If False, a method on EMComposition is used rather than a LearningMechanism.  This is meant for
            debugging, and precludes use of `import_composition <Composition.import_composition>` to integrate
            the EMComposition into another Composition;  to do so, use_storage_node must be True (default).

    use_gating_for_weighting : bool : default False
        specifies whether to use output gating to weight the `match_nodes <EMComposition.match_node>` instead of
        a standard input (see `Weight distances <EMComposition_Field_Weighting>` for additional details).

    Attributes
    ----------

    memory : ndarray
        3d array of entries in memory, in which each row (axis 0) is an entry, each column (axis 1) is a field, and
        each item (axis 2) is the value for the corresponding field;  see `EMComposition_Memory` for additional details.

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

    field_weights : tuple[float]
        determines which fields of the input are treated as "keys" (non-zero values) that are used to match entries in
        `memory <EMComposition.memory>` for retrieval, and which are used as "values" (zero values) that are stored
        and retrieved from memory but not used in the match process (see `Match memories by field
        <EMComposition_Processing>`; also determines the relative contribution of each key field to the match process;
        see `field_weights <EMComposition_Field_Weights>` additional details.

    normalize_field_weights : bool : default True
        determines whether `fields_weights <EMComposition.field_weights>` are normalized over the number of keys, or
        used as absolute weighting values when retrieving an item from memory; see `normalize_field weights
        <EMComposition_Normalize_Field_Weights>` for additional details.

    field_names : list[str]
        determines which names that can be used to label fields in `memory <EMComposition.memory>`;  see
        `field_names <EMComposition_Field_Names>` for additional details.

    concatenate_queries : bool
        determines whether keys are concatenated into a single field before matching them to items in `memory
        <EMComposition.memory>`; see `concatenate keys <EMComposition_Concatenate_Queries>` for additional details.

    normalize_memories : bool
        determines whether keys and memories are normalized before computing their dot product (similarity);
        see `Match memories by field <EMComposition_Processing>` for additional details.

    softmax_gain : float, ADAPTIVE or CONTROL
        determines gain (inverse temperature) used for softmax normalizing the summed distances of queries and keys in
        memory by the `SoftMax` Function of the `softmax_node <EMComposition.softmax_node>`; see `Softmax normalize
        distances <EMComposition_Processing>` for additional details.

    softmax_threshold : float
        determines the threshold used to mask out small values in the softmax calculation;
        see *mask_threshold* under `Thresholding and Adaptive Gain <SoftMax_AdaptGain>` for details).

    softmax_choice : WEIGHTED_AVG, ARG_MAX or PROBABILISTIC
        determines how the softmax over distances of queries and keys in memory is used for retrieval;
        see `softmax_choice <EMComposition_Softmax_Choice>` for a description of each option.

    storage_prob : float
        determines the probability that an item will be stored in `memory <EMComposition.memory>`
        when the EMComposition is executed (see `Retrieval and Storage <EMComposition_Storage>` for
        additional details).

    memory_decay_rate : float
        determines the rate at which items in the EMComposition's memory decay (see `memory_decay_rate
        <EMComposition_Memory_Decay_Rate>` for details).

    enable_learning : bool or list[bool]
        determines whether `learning <Composition_Learning>` is enabled for the EMComposition, allowing any error
        received by the `retrieved_nodes <EMComposition.retrieved_nodes>` to be propagated to the corresponding
        `query_input_nodes <EMComposition.query_input_nodes>` and `value_input_nodes
        <EMComposition.value_input_nodes>`, and on to any `Nodes <Composition_Nodes>` that project to them.
        If True, learning is enabled for all fields and if False learning is disabled for all fields; If it is a
        list, then each entry specifies whether learning is enabled or disabled for the corresponding field
        see `Learning <EMComposition_Learning>` and `Fields <EMComposition_Fields>` for additional details.

    learn_field_weights : bool
        determines whether `field_weights <EMComposition.field_weights>` are learnable during training;
        requires `enable_learning <EMComposition.enable_learning>` to be True or a list with at least one True
        entry for the corresponding field; see `Learning <EMComposition_Learning>` for additional details.

    learning_rate : float
        determines whether the rate at which `field_weights <EMComposition.field_weights>` are learned
        if `learn_field_weights` is True;  see `Learning <EMComposition_Learning>` for additional details.

    .. _EMComposition_Nodes:

    query_input_nodes : list[ProcessingMechanism]
        `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` that receive keys used to determine the item
        to be retrieved from `memory <EMComposition.memory>`, and then themselves stored in `memory
        <EMComposition.memory>` (see `Match memories by field <EMComposition_Processing>` for additional details).
        By default these are assigned the name *KEY_n_INPUT* where n is the field number (starting from 0);
        however, if `field_names <EMComposition.field_names>` is specified, then the name of each query_input_node
        is assigned the corresponding field name appended with * [QUERY]*.

    value_input_nodes : list[ProcessingMechanism]
        `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` that receive values to be stored in `memory
        <EMComposition.memory>`; these are not used in the matching process used for retrieval.  By default these
        are assigned the name *VALUE_n_INPUT* where n is the field number (starting from 0);  however, if
        `field_names <EMComposition.field_names>` is specified, then the name of each value_input_node is assigned
        the corresponding field name appended with * [VALUE]*.

    concatenate_queries_node : ProcessingMechanism
        `ProcessingMechanism` that concatenates the inputs to `query_input_nodes <EMComposition.query_input_nodes>`
        into a single vector used for the matching processing if `concatenate keys <EMComposition.concatenate_queries>`
        is True. This is not created if the **concatenate_queries** argument to the EMComposition's constructor is
        False or is overridden (see `concatenate_queries <EMComposition_Concatenate_Queries>`), or there is only one
        query_input_node. This node is named *CONCATENATE_QUERIES*

    match_nodes : list[ProcessingMechanism]
        `ProcessingMechanisms <ProcessingMechanism>` that compute the dot product of each query and the key stored in
        the corresponding field of `memory <EMComposition.memory>` (see `Match memories by field
        <EMComposition_Processing>` for additional details). These are named the same as the corresponding
        `query_input_nodes <EMComposition.query_input_nodes>` appended with the suffix *[MATCH to KEYS]*.

    field_weight_nodes : list[ProcessingMechanism or GatingMechanism]
        Nodes used to weight the distances computed by the `match_nodes <EMComposition.match_nodes>` with the
        `field weight <EMComposition.field_weights>` for the corresponding `key field <EMComposition_Fields>`
        (see `Weight distances <EMComposition_Field_Weighting>` for implementation). These are named the same
        as the corresponding `query_input_nodes <EMComposition.query_input_nodes>`.

    weighted_match_nodes : list[ProcessingMechanism]
        `ProcessingMechanisms <ProcessingMechanism>` that combine the  `field weight <EMComposition.field_weights>`
        for each `key field <EMComposition_Fields>` with the dot product computed by the corresponding the
        `match_node <EMComposition.match_nodes>`. These are only implemented if `use_gating_for_weighting
        <EMComposition.use_gating_for_weighting>` is False (see `Weight distances <EMComposition_Field_Weighting>`
        for details), and are named the same as the corresponding `query_input_nodes <EMComposition.query_input_nodes>`
        appended with the suffix *[WEIGHTED MATCH]*.

    combined_matches_node : ProcessingMechanism
        `ProcessingMechanism` that receives the weighted distances from the `weighted_match_nodes
        <EMComposition.weighted_match_nodes>` if more than one `key field <EMComposition_Fields>` is specified
        (or directly from `match_nodes <EMComposition.match_nodes>` if `use_gating_for_weighting
        <EMComposition.use_gating_for_weighting>` is True), and combines them into a single vector that is passed
        to the `softmax_node <EMComposition.softmax_node>` for retrieval. This node is named *COMBINE MATCHES*.

    softmax_node : list[ProcessingMechanism]
        `ProcessingMechanisms <ProcessingMechanism>` that computes the softmax over the summed distances of keys
        and memories (output of the `combined_match_node <EMComposition.combined_match_node>`)
        from the corresponding `match_nodes <EMComposition.match_nodes>` (see `Softmax over summed distances
        <EMComposition_Processing>` for additional details).  This is named *RETRIEVE* (as it yields the
        softmax-weighted average over the keys in `memory <EMComposition.memory>`).

    softmax_gain_control_node : list[ControlMechanism]
        `ControlMechanisms <ControlMechanism>` that adaptively control the `softmax_gain <EMComposition.softmax_gain>`
        of the `softmax_node <EMComposition.softmax_node>`. This is implemented only if `softmax_gain
        <EMComposition.softmax_gain>` is specified as *CONTROL* (see `softmax_gain <EMComposition_Softmax_Gain>` for
        details).

    retrieved_nodes : list[ProcessingMechanism]
        `ProcessingMechanisms <ProcessingMechanism>` that receive the vector retrieved for each field in `memory
        <EMComposition.memory>` (see `Retrieve values by field <EMComposition_Processing>` for additional details).
        These are assigned the same names as the `query_input_nodes <EMComposition.query_input_nodes>` and
        `value_input_nodes <EMComposition.value_input_nodes>` to which they correspond appended with the suffix
        * [RETRIEVED]*, and are in the same order as  `input_nodes_by_fields <EMComposition.input_nodes_by_fields>`
        to which to which they correspond.

    storage_node : EMStorageMechanism
        `EMStorageMechanism` that receives inputs from the `query_input_nodes <EMComposition.query_input_nodes>` and
        `value_input_nodes <EMComposition.value_input_nodes>`, and stores these in the corresponding field of`memory
        <EMComposition.memory>` with probability `storage_prob <EMComposition.storage_prob>` after a retrieval has been
        made (see `Retrieval and Storage <EMComposition_Storage>` for additional details). This node is named *STORE*.

        .. technical_note::
           The `storage_node <EMComposition.storage_node>` is assigned a Condition to execute after the `retrieved_nodes
           <EMComposition.retrieved_nodes>` have executed, to ensure that storage occurs after retrieval, but before
           any subequent processing is done (i.e., in a composition in which the EMComposition may be embededded.

    input_nodes : list[ProcessingMechanism]
        Full list of `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` ordered with query_input_nodes first
        followed by value_input_nodes; used primarily for internal computations

    input_nodes_by_fields : list[ProcessingMechanism]
        Full list of `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` in the same order specified in the
        **field_names** argument of the constructor and in `self.field_names <EMComposition.field_names>`.

    """

    componentCategory = EM_COMPOSITION

    if torch_available:
        from psyneulink.library.compositions.pytorchEMcompositionwrapper import PytorchEMCompositionWrapper
        pytorch_composition_wrapper_type = PytorchEMCompositionWrapper


    class Parameters(AutodiffComposition.Parameters):
        """
            Attributes
            ----------

                concatenate_queries
                    see `concatenate_queries <EMComposition.concatenate_queries>`

                    :default value: False
                    :type: ``bool``

                enable_learning
                    see `enable_learning <EMComposition.enable_learning>`

                    :default value: True
                    :type: ``bool`` or ``list``

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

                normalize_field_weights
                    see `normalize_field_weights <EMComposition.normalize_field_weights>`

                    :default value: True
                    :type: ``bool``

                normalize_memories
                    see `normalize_memories <EMComposition.normalize_memories>`

                    :default value: True
                    :type: ``bool``

                random_state
                    see `random_state <NormalDist.random_state>`

                    :default value: None
                    :type: ``numpy.random.RandomState``

                softmax_gain
                    see `softmax_gain <EMComposition.softmax_gain>`
                    :default value: 1.0
                    :type: ``float, ADAPTIVE or CONTROL``

                softmax_choice
                    see `softmax_choice <EMComposition.softmax_choice>`
                    :default value: WEIGHTED_AVG
                    :type: ``keyword``

                softmax_threshold
                    see `softmax_threshold <EMComposition.softmax_threshold>`
                    :default value: .001
                    :type: ``float``

                storage_prob
                    see `storage_prob <EMComposition.storage_prob>`

                    :default value: 1.0
                    :type: ``float``
        """
        memory = Parameter(None, loggable=True, getter=_memory_getter, read_only=True)
        memory_template = Parameter([[0],[0]], structural=True, valid_types=(tuple, list, np.ndarray), read_only=True)
        memory_capacity = Parameter(1000, structural=True)
        field_weights = Parameter(None)
        normalize_field_weights = Parameter(True)
        field_names = Parameter(None, structural=True)
        concatenate_queries = Parameter(False, structural=True)
        normalize_memories = Parameter(True)
        softmax_gain = Parameter(1.0, modulable=True)
        softmax_threshold = Parameter(.001, modulable=True, specify_none=True)
        softmax_choice = Parameter(WEIGHTED_AVG, modulable=False, specify_none=True)
        storage_prob = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
        memory_decay_rate = Parameter(AUTO, modulable=True)
        enable_learning = Parameter(True, structural=True)
        learn_field_weights = Parameter(True, structural=True)
        learning_rate = Parameter(.001, modulable=True)
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

        def _validate_field_weights(self, field_weights):
            if field_weights is not None:
                if not np.atleast_1d(field_weights).ndim == 1:
                    return f"must be a scalar, list of scalars, or 1d array."
                if any([field_weight < 0 for field_weight in field_weights]):
                    return f"must be all be positive values."

        def _validate_normalize_field_weights(self, normalize_field_weights):
            if not isinstance(normalize_field_weights, bool):
                return f"must be all be a boolean value."

        def _validate_field_names(self, field_names):
            if field_names and not all(isinstance(item, str) for item in field_names):
                return f"must be a list of strings."

        def _validate_enable_learning(self, enable_learning):
            if isinstance(enable_learning, list):
                if not all(isinstance(item, bool) for item in enable_learning):
                    return f"can only contains bools as entries."
            elif not isinstance(enable_learning, bool):
                return f"must be a bool or list of bools."

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
                 field_names:Optional[list]=None,
                 field_weights:tuple=None,
                 normalize_field_weights:bool=True,
                 concatenate_queries:bool=False,
                 normalize_memories:bool=True,
                 softmax_gain:Union[float, ADAPTIVE, CONTROL]=1.0,
                 softmax_threshold:Optional[float]=.001,
                 softmax_choice:Optional[Union[WEIGHTED_AVG, ARG_MAX, PROBABILISTIC]]=WEIGHTED_AVG,
                 storage_prob:float=1.0,
                 memory_decay_rate:Union[float,AUTO]=AUTO,
                 enable_learning:Union[bool,list]=True,
                 learn_field_weights:bool=True,
                 learning_rate:float=None,
                 use_storage_node:bool=True,
                 use_gating_for_weighting:bool=False,
                 random_state=None,
                 seed=None,
                 name="EM_Composition",
                 **kwargs):

        # Construct memory --------------------------------------------------------------------------------

        memory_fill = memory_fill or 0 # FIX: GET RID OF THIS ONCE IMPLEMENTED AS A Parameter
        self._validate_memory_specs(memory_template, memory_capacity, memory_fill, field_weights, field_names, name)
        memory_template, memory_capacity = self._parse_memory_template(memory_template,
                                                                       memory_capacity,
                                                                       memory_fill)
        field_weights, field_names, concatenate_queries = self._parse_fields(field_weights,
                                                                             normalize_field_weights,
                                                                             field_names,
                                                                             concatenate_queries,
                                                                             normalize_memories,
                                                                             learning_rate,
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
                         field_weights = field_weights,
                         field_names = field_names,
                         normalize_field_weights = normalize_field_weights,
                         concatenate_queries = concatenate_queries,
                         softmax_gain = softmax_gain,
                         softmax_threshold = softmax_threshold,
                         softmax_choice = softmax_choice,
                         storage_prob = storage_prob,
                         memory_decay_rate = memory_decay_rate,
                         normalize_memories = normalize_memories,
                         enable_learning=enable_learning,
                         learn_field_weights = learn_field_weights,
                         learning_rate = learning_rate,
                         random_state = random_state,
                         seed = seed,
                         **kwargs
                         )

        self._validate_options_with_learning(enable_learning,
                                             use_gating_for_weighting,
                                             learn_field_weights,
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
                                 self.enable_learning,
                                 self.learn_field_weights,
                                 self._use_gating_for_weighting)

        # if torch_available:
        #     from psyneulink.library.compositions.pytorchEMcompositionwrapper import PytorchEMCompositionWrapper
        #     self.pytorch_composition_wrapper_type = PytorchEMCompositionWrapper

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
            #   but generates warning messages for every node of the following sort:
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

        # If enable_learning is a list of bools, it must match the len of 1st dimension (axis 0) of memory_template:
        if isinstance(self.enable_learning, list) and len(self.enable_learning) != num_fields:
            raise EMCompositionError(f"The number of items ({len(self.enable_learning)}) in the 'enable_learning' arg "
                                     f"for {name} must match the number of fields in memory "
                                     f"({num_fields}).")

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
                      field_weights,
                      normalize_field_weights,
                      field_names,
                      concatenate_queries,
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
        # Fill out field_weights, normalizing if specified:

        if len(field_weights) == 1:
            if normalize_field_weights:
                parsed_field_weights = np.repeat(field_weights / np.sum(field_weights), len(self.entry_template))
            else:
                parsed_field_weights = np.repeat(field_weights[0], len(self.entry_template))
        else:
            if normalize_field_weights:
                parsed_field_weights = np.array(field_weights) / np.sum(field_weights)
            else:
                parsed_field_weights = field_weights

        # Memory structure Parameters
        parsed_field_names = field_names.copy() if field_names is not None else None

        # Set memory field attributes
        self.num_fields = len(self.entry_template)
        keys_weights = [i for i in parsed_field_weights if i != 0]
        self.num_keys = len(keys_weights)

        # Get indices of field_weights that specify keys and values:
        self.key_indices = np.flatnonzero(parsed_field_weights)
        assert len(self.key_indices) == self.num_keys, \
            f"PROGRAM ERROR: number of keys ({self.num_keys}) does not match number of " \
            f"non-zero values in field_weights ({len(self.key_indices)})."
        self.value_indices = np.where(parsed_field_weights==0)[0]
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
            self.value_names = [f'{i} [VALUE]' for i in range(self.num_values)] if self.num_values > 1 else ['VALUE']
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
            fw_error_msg = nm_error_msg = fw_correction_msg = nm_correction_msg = None
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

        self.learning_rate = learning_rate
        return parsed_field_weights, parsed_field_names, parsed_concatenate_queries

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
                            enable_learning,
                            learn_field_weights,
                            use_gating_for_weighting,
                            ):
        """Construct Nodes and Pathways for EMComposition"""

        # Construct Nodes --------------------------------------------------------------------------------

        field_weighting = len([weight for weight in field_weights if weight]) > 1 and not concatenate_queries

        # First, construct Nodes of Composition with their Projections
        self.query_input_nodes = self._construct_query_input_nodes(field_weights)
        self.value_input_nodes = self._construct_value_input_nodes(field_weights)
        self.input_nodes = self.query_input_nodes + self.value_input_nodes

        # Get list of nodes in order specified in self.field_names
        self.input_nodes_by_fields = [None] * len(field_weights)
        for i in range(self.num_keys):
            self.input_nodes_by_fields[self.key_indices[i]] = self.query_input_nodes[i]
        for i in range(self.num_values):
            self.input_nodes_by_fields[self.value_indices[i]] = self.value_input_nodes[i]
        assert all(self.input_nodes_by_fields), "PROGRAM ERROR: input_nodes_by_fields not fully populated."

        self.concatenate_queries_node = self._construct_concatenate_queries_node(concatenate_queries)
        self.match_nodes = self._construct_match_nodes(memory_template, memory_capacity,
                                                       concatenate_queries,normalize_memories)
        self.field_weight_nodes = self._construct_field_weight_nodes(field_weights,
                                                                     concatenate_queries,
                                                                     use_gating_for_weighting)
        self.weighted_match_nodes = self._construct_weighted_match_nodes(memory_capacity, field_weights)

        self.combined_matches_node = self._construct_combined_matches_node(memory_capacity,
                                                                           field_weighting,
                                                                           use_gating_for_weighting)
        self.softmax_node = self._construct_softmax_node(memory_capacity,
                                                         softmax_gain,
                                                         softmax_threshold,
                                                         softmax_choice)

        self.softmax_gain_control_node = self._construct_softmax_gain_control_node(softmax_gain)

        self.retrieved_nodes = self._construct_retrieved_nodes(memory_template)

        if use_storage_node:
            self.storage_node = self._construct_storage_node(memory_template, field_weights,
                                                             self.concatenate_queries_node,
                                                             memory_decay_rate, storage_prob)

        # Do some validation and get singleton softmax and match Nodes for concatenated queries
        if self.concatenate_queries:
            assert len(self.match_nodes) == 1, \
                f"PROGRAM ERROR: Too many match_nodes ({len(self.match_nodes)}) for concatenated queries."
            assert not self.field_weight_nodes, \
                f"PROGRAM ERROR: There should be no field_weight_nodes for concatenated queries."

        # Construct Pathways --------------------------------------------------------------------------------

        # LEARNING NOT ENABLED --------------------------------------------------
        # Set up pathways WITHOUT PsyNeuLink learning pathways
        if not self.enable_learning:
            self.add_nodes(self.query_input_nodes + self.value_input_nodes)
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
                    self.add_linear_processing_pathway([self.query_input_nodes[i],
                                                        self.match_nodes[i],
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
            if self.field_weight_nodes:
                for i in range(self.num_keys):
                    self.add_linear_processing_pathway([self.field_weight_nodes[i], self.weighted_match_nodes[i]])

            self.add_nodes(self.value_input_nodes)

            # Retrieval pathways
            for i in range(len(self.retrieved_nodes)):
                self.add_linear_processing_pathway([self.softmax_node, self.retrieved_nodes[i]])

            # Storage Nodes
            if use_storage_node:
                self.add_node(self.storage_node)

    def _construct_query_input_nodes(self, field_weights)->list:
        """Create one node for each key to be used as cue for retrieval (and then stored) in memory.
        Used to assign new set of weights for Projection for query_input_node[i] -> match_node[i]
        where i is selected randomly without replacement from (0->memory_capacity)
        """

        assert len(self.key_indices) == self.num_keys, \
            f"PROGRAM ERROR: number of keys ({self.num_keys}) does not match number of " \
            f"non-zero values in field_weights ({len(self.key_indices)})."

        query_input_nodes = [ProcessingMechanism(
            input_shapes=len(self.entry_template[self.key_indices[i]]),
                                             name=f'{self.key_names[i]} [QUERY]')
                       for i in range(self.num_keys)]

        return query_input_nodes

    def _construct_value_input_nodes(self, field_weights)->list:
        """Create one input node for each value to be stored in memory.
        Used to assign new set of weights for Projection for combined_matches_node -> retrieved_node[i]
        where i is selected randomly without replacement from (0->memory_capacity)
        """

        # Get indices of field_weights that specify keys:
        value_indices = np.where(field_weights == 0)[0]

        assert len(value_indices) == self.num_values, \
            f"PROGRAM ERROR: number of values ({self.num_values}) does not match number of " \
            f"non-zero values in field_weights ({len(value_indices)})."

        value_input_nodes = [ProcessingMechanism(
            input_shapes=len(self.entry_template[value_indices[i]]),
                                               name= f'{self.value_names[i]} [VALUE]')
                           for i in range(self.num_values)]

        return value_input_nodes

    def _construct_concatenate_queries_node(self, concatenate_queries)->ProcessingMechanism:
        """Create node that concatenates the inputs for all keys into a single vector
        Used to create a matrix for Projectoin from match / memory weights from concatenate_node -> match_node
        """
        # One node that concatenates inputs from all keys
        if not concatenate_queries:
            return None
        else:
            return ProcessingMechanism(function=Concatenate,
                                       input_ports=[{NAME: 'CONCATENATE',
                                                     INPUT_SHAPES: len(self.query_input_nodes[i].output_port.value),
                                                     PROJECTIONS: MappingProjection(
                                                         name=f'{self.key_names[i]} to CONCATENATE',
                                                         sender=self.query_input_nodes[i].output_port,
                                                         matrix=IDENTITY_MATRIX)}
                                                    for i in range(self.num_keys)],
                                       name=CONCATENATE_QUERIES_NAME)

    def _construct_match_nodes(self, memory_template, memory_capacity, concatenate_queries, normalize_memories)->list:
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
        args = [(L0,False) if len(key) == 1 else (DOT_PRODUCT,normalize_memories) for key in memory_template[0]]

        if concatenate_queries:
            # Get fields of memory structure corresponding to the keys
            # Number of rows should total number of elements over all keys,
            #    and columns should number of items in memory
            matrix =np.array([np.concatenate((memory_template[:,:self.num_keys][i]))
                              for i in range(memory_capacity)]).transpose()
            matrix = np.array(matrix.tolist())
            match_nodes = [
                ProcessingMechanism(
                    input_ports={NAME: 'CONCATENATED_INPUTS',
                                 INPUT_SHAPES: memory_capacity,
                                 PROJECTIONS: MappingProjection(sender=self.concatenate_queries_node,
                                                                matrix=matrix,
                                                                function=MatrixTransform(
                                                                    operation=args[0][OPERATION],
                                                                    normalize=args[0][NORMALIZE]),
                                                                name=f'MEMORY')},
                    name='MATCH')]

        # One node for each key
        else:
            match_nodes = [
                ProcessingMechanism(
                    input_ports= {
                        INPUT_SHAPES:memory_capacity,
                        PROJECTIONS: MappingProjection(sender=self.query_input_nodes[i].output_port,
                                                       matrix = np.array(
                                                           memory_template[:,i].tolist()).transpose().astype(float),
                                                       function=MatrixTransform(operation=args[i][OPERATION],
                                                                                normalize=args[i][NORMALIZE]),
                                                       name=f'MEMORY for {self.key_names[i]} [KEY]')},
                    name=self.key_names[i] + MATCH_TO_KEYS_AFFIX)
                for i in range(self.num_keys)
            ]


        return match_nodes

    # FIX: CONVERT TO _construct_weight_control_nodes
    def _construct_field_weight_nodes(self, field_weights, concatenate_queries, use_gating_for_weighting)->list:
        """Create ProcessingMechanisms that weight each key's softmax contribution to the retrieved values."""

        field_weight_nodes = []

        if not concatenate_queries and self.num_keys > 1:
            if use_gating_for_weighting:
                field_weight_nodes = [GatingMechanism(input_ports={VARIABLE:
                                                                       np.array(field_weights[self.key_indices[i]]),
                                                                   PARAMS:{DEFAULT_INPUT: DEFAULT_VARIABLE},
                                                                   NAME: 'OUTCOME'},
                                                      gate=[key_match_pair[1].output_ports[0]],
                                                      name= 'WEIGHT' if self.num_keys == 1
                                                      else f'{self.key_names[i]}{WEIGHT_AFFIX}')
                                      for i, key_match_pair in enumerate(zip(self.query_input_nodes,
                                                                             self.match_nodes))]
            else:
                field_weight_nodes = [ProcessingMechanism(input_ports={VARIABLE:
                                                                           np.array(field_weights[self.key_indices[i]]),
                                                                       PARAMS: {DEFAULT_INPUT: DEFAULT_VARIABLE},
                                                                       NAME: 'FIELD_WEIGHT'},
                                                          name= WEIGHT if self.num_keys == 1
                                                          else f'{self.key_names[i]}{WEIGHT_AFFIX}')
                                      for i in range(self.num_keys)]
        return field_weight_nodes

    def _construct_weighted_match_nodes(self, memory_capacity, field_weights)->list:
        """Create nodes that weight the output of the match node for each key."""

        weighted_match_nodes = \
            [ProcessingMechanism(default_variable=[self.match_nodes[i].output_port.value,
                                                   self.match_nodes[i].output_port.value],
                                 input_ports=[{PROJECTIONS:
                                                   MappingProjection(sender=match_fw_pair[0],
                                                                     matrix=IDENTITY_MATRIX,
                                                                     name=f'{MATCH} to {WEIGHTED_MATCH_NODE_NAME} '
                                                                          f'for {self.key_names[i]}')},
                                              {PROJECTIONS:
                                                   MappingProjection(sender=match_fw_pair[1],
                                                                     matrix=FULL_CONNECTIVITY_MATRIX,
                                                                     name=f'{WEIGHT} to {WEIGHTED_MATCH_NODE_NAME} '
                                                                          f'for {self.key_names[i]}')}],
                                 function=LinearCombination(operation=PRODUCT),
                                 name=self.key_names[i] + WEIGHTED_MATCH_AFFIX)
             for i, match_fw_pair in enumerate(zip(self.match_nodes,
                                                   self.field_weight_nodes))]

        return weighted_match_nodes

    def _construct_softmax_gain_control_node(self, softmax_gain)->Optional[ControlMechanism]:
        """Create nodes that set the softmax gain (inverse temperature) for each softmax_node."""

        if softmax_gain == CONTROL:
            return ControlMechanism(monitor_for_control=self.combined_matches_node,
                                    control_signals=[(GAIN, self.softmax_node)],
                                    function=get_softmax_gain,
                                    name='SOFTMAX GAIN CONTROL')
        else:
            return None

    def _construct_combined_matches_node(self,
                                         memory_capacity,
                                         field_weighting,
                                         use_gating_for_weighting
                                         )->ProcessingMechanism:
        """Create node that combines weighted matches for all keys into one match vector."""

        if self.num_keys == 1 or self.concatenate_queries_node:
            return

        if not field_weighting or use_gating_for_weighting:
            input_source = self.match_nodes
        else:
            input_source = self.weighted_match_nodes

        combined_matches_node = (
            ProcessingMechanism(input_ports=[{INPUT_SHAPES:memory_capacity,
                                              PROJECTIONS:[MappingProjection(sender=s,
                                                                             matrix=IDENTITY_MATRIX,
                                                                             name=f'{WEIGHTED_MATCH_NODE_NAME} '
                                                                                  f'for {self.key_names[i]} to '
                                                                                  f'{COMBINE_MATCHES_NODE_NAME}')
                                                           for i, s in enumerate(input_source)]}],
                                name=COMBINE_MATCHES_NODE_NAME))

        assert len(combined_matches_node.output_port.value) == memory_capacity, \
            'PROGRAM ERROR: number of items in combined_matches_node ' \
            f'({len(combined_matches_node.output_port)}) does not match memory_capacity ({self.memory_capacity})'

        return combined_matches_node

    def _construct_softmax_node(self, memory_capacity, softmax_gain, softmax_threshold, softmax_choice)->list:
        """Create node that applies softmax to output of combined_matches_node."""

        if self.num_keys == 1 or self.concatenate_queries_node:
            input_source = self.match_nodes[0]
            proj_name =f'{MATCH} to {SOFTMAX_NODE_NAME}'
        # elif self.concatenate_queries_node:
        #     input_source = self.concatenate_queries_node
        #     proj_name =f'{CONCATENATE_QUERIES_NAME} to {SOFTMAX_NODE_NAME}'
        else:
            input_source = self.combined_matches_node
            proj_name =f'{COMBINE_MATCHES_NODE_NAME} to {SOFTMAX_NODE_NAME}'

        if softmax_choice == ARG_MAX:
            # ARG_MAX would return entry multiplied by its dot product
            # ARG_MAX_INDICATOR returns the entry unmodified
            softmax_choice = ARG_MAX_INDICATOR

        softmax_node = ProcessingMechanism(input_ports={INPUT_SHAPES: memory_capacity,
                                                        PROJECTIONS: MappingProjection(
                                                            sender=input_source,
                                                            matrix=IDENTITY_MATRIX,
                                                            name=proj_name)},
                                           function=SoftMax(gain=softmax_gain,
                                                            mask_threshold=softmax_threshold,
                                                            output=softmax_choice,
                                                            adapt_entropy_weighting=.95),
                                           name=SOFTMAX_NODE_NAME)

        return softmax_node

    def _validate_options_with_learning(self,
                                        enable_learning,
                                        use_gating_for_weighting,
                                        learn_field_weights,
                                        softmax_choice):
        if use_gating_for_weighting and learn_field_weights:
            warnings.warn(f"The 'learn_field_weights' option for '{self.name}' cannot be used with "
                          f"'use_gating_for_weighting' set to True; this will generate an error if its "
                          f"'learn' method is called. Set 'use_gating_for_weighting' to True in order "
                          f"to enable learning of field weights.")

        if softmax_choice in {ARG_MAX, PROBABILISTIC} and enable_learning:
            warnings.warn(f"The 'softmax_choice' arg of '{self.name}' is set to '{softmax_choice}' with "
                          f"'enable_learning' set to True (or a list); this will generate an error if its "
                          f"'learn' method is called. Set 'softmax_choice' to WEIGHTED_AVG before learning.")

    def _construct_retrieved_nodes(self, memory_template)->list:
        """Create nodes that report the value field(s) for the item(s) matched in memory.
        """
        self.retrieved_key_nodes = \
            [ProcessingMechanism(input_ports={INPUT_SHAPES: len(self.query_input_nodes[i].variable[0]),
                                            PROJECTIONS:
                                                MappingProjection(
                                                    sender=self.softmax_node,
                                                    matrix=memory_template[:,i],
                                                    name=f'MEMORY FOR {self.key_names[i]} [RETRIEVE KEY]')
                                            },
                               name= self.key_names[i] + RETRIEVED_AFFIX)
             for i in range(self.num_keys)]

        self.retrieved_value_nodes = \
            [ProcessingMechanism(input_ports={INPUT_SHAPES: len(self.value_input_nodes[i].variable[0]),
                                            PROJECTIONS:
                                                MappingProjection(
                                                    sender=self.softmax_node,
                                                    matrix=memory_template[:,
                                                           i + self.num_keys],
                                                    name=f'MEMORY FOR {self.value_names[i]} [RETRIEVE VALUE]')},
                               name= self.value_names[i] + RETRIEVED_AFFIX)
             for i in range(self.num_values)]

        retrieved_nodes = self.retrieved_key_nodes + self.retrieved_value_nodes

        # Return nodes in order sorted by self.field_names
        # (use name_without_suffix as reference in case more than one EMComposition is created,
        #  in which case retrieved_nodes will have "-<int>" appended to their name)
        return [node for name in self.field_names for node in retrieved_nodes
                if node in retrieved_nodes if (name + RETRIEVED_AFFIX) == name_without_suffix(node.name)]

    def _construct_storage_node(self,
                                memory_template,
                                field_weights,
                                concatenate_queries_node,
                                memory_decay_rate,
                                storage_prob)->list:
        """Create EMStorageMechanism that stores the key and value inputs in memory.
        Memories are stored by adding the current input to each field to the corresponding row of the matrix for
        the Projection from the query_input_node (or concatenate_node) to the matching_node and retrieved_node for keys,
        and from the value_input_node to the retrieved_node for values. The `function <EMStorageMechanism.function>`
        of the `EMSorageMechanism` that takes the following arguments:

         - **variable** -- template for an `entry <EMComposition_Memory>` in `memory<EMComposition.memory>`;

         - **fields** -- the `input_nodes <EMComposition.input_nodes>` for the corresponding `fields
           <EMComposition_Fields>` of an `entry <EMCmposition_Memory>` in `memory <EMComposition.memory>`;

         - **field_types** -- a list of the same length as ``fields``, containing 1's for key fields and 0's for
           value fields;

         - **concatenate_queries_node** -- node used to concatenate keys
           (if `concatenate_queries <EMComposition.concatenate_queries>` is `True`) or None;

         - **memory_matrix** -- `memory_template <EMComposition.memory_template>`);

         - **learning_signals** -- list of ` `MappingProjection`\\s (or their ParameterPort`\\s) that store each
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
                                          concatenation_node=concatenate_queries_node,
                                          memory_matrix=memory_template,
                                          learning_signals=learning_signals,
                                          storage_prob=storage_prob,
                                          decay_rate = memory_decay_rate,
                                          name=STORE_NODE_NAME)

        return storage_node

    def _set_learning_attributes(self):
        """Set learning-related attributes for Node and Projections
        """
        # 7/10/24 FIX: SHOULD THIS ALSO BE CONSTRAINED BY VALUE OF field_weights FOR CORRESPONDING FIELD?
        #         (i.e., if it is zero then not learnable? or is that a valid initial condition?)
        for projection in self.projections:
            if (projection.sender.owner in self.field_weight_nodes
                    and self.enable_learning
                    and self.learn_field_weights):
                projection.learnable = True
            else:
                projection.learnable = False

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
        purge_by_field_weights = False
        field_norms = np.array([np.linalg.norm(field, axis=1)
                                for field in [row for row in self.parameters.memory.get(context)]])
        if purge_by_field_weights:
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
        learn_field_weights = self.parameters.learn_field_weights.get(kwargs[CONTEXT])

        if use_gating_for_weighting and learn_field_weights:
            raise EMCompositionError(f"Field weights cannot be learned when 'use_gating_for_weighting' is True; "
                                     f"Construct '{self.name}' with the 'learn_field_weights' arg set to False.")

        if softmax_choice in {ARG_MAX, PROBABILISTIC}:
            raise EMCompositionError(f"The ARG_MAX and PROBABILISTIC options for the 'softmax_choice' arg "
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
        """Identify retrieval_nodes specified by **enable_learning** as TARGET nodes"""
        enable_learning = self.parameters.enable_learning._get(context)
        if enable_learning is False:
            if self.learn_field_weights:
                warnings.warn(f"The 'learn_field_weights' arg for {self.name} is True "
                              f"but its 'enable_learning' is False, so learn_field_weights will have no effect.")
            target_nodes = []
        elif enable_learning is True:
            target_nodes = [node for node in self.retrieved_nodes]
        elif isinstance(enable_learning, list):
            target_nodes = [node for node in self.retrieved_nodes if enable_learning[self.retrieved_nodes.index(node)]]
        else:
            assert False, (f"PROGRAM ERROR: enable_learning arg for {self.name}: {enable_learning} "
                           f"is neither True, False nor a list of bools as it should be.")
        super()._identify_target_nodes(context)
        return target_nodes

    def infer_backpropagation_learning_pathways(self, execution_mode, context=None):
        if self.concatenate_queries:
            raise EMCompositionError(f"EMComposition does not support learning with 'concatenate_queries'=True.")
        super().infer_backpropagation_learning_pathways(execution_mode, context=context)

    def do_gradient_optimization(self, retain_in_pnl_options, context, optimization_num=None):
        # 7/10/24 - MAKE THIS CONTEXT DEPENDENT:  CALL super() IF BEING EXECUTED ON ITS OWN?
        pass

    #endregion
