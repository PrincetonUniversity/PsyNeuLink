# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ******************************************  OutputState *****************************************************

"""

Overview
--------

OutputState(s) represent the result(s) of executing a Mechanism.  This may be the result(s) of its
`function <OutputState.function>` and/or values derived from that result.  The full set of results are stored in the
Mechanism's `output_values <Mechanism_Base.output_values>` attribute.  OutputStates are used to represent
individual items of the Mechanism's `value <Mechanism_Base.value>`, and/or useful quantities derived from
them.  For example, the `function <TransferMechanism.TransferMechanism.function>` of a `TransferMechanism` generates
a single result (the transformed value of its input);  however, a TransferMechanism can also be assigned OutputStates
that represent its mean, variance or other derived values.  In contrast, the `function <DDM.DDM.function>`
of a `DDM` Mechanism generates several results (such as decision accuracy and response time), each of which can be
assigned as the `value <OutputState.value>` of a different OutputState.  The OutputState(s) of a Mechanism can serve
as the input to other  Mechanisms (by way of `projections <Projections>`), or as the output of a Process and/or
System.  The OutputState's `efferents <OutputState.efferents>` attribute lists all of its outgoing
projections.

.. _OutputStates_Creation:

Creating an OutputState
-----------------------

An OutputState can be created by calling its constructor. However, in general this is not necessary, as a Mechanism
automatically creates a default OutputState if none is explicitly specified, that contains the primary result of its
`function <Mechanism_Base.function>`.  For example, if the Mechanism is created within the `pathway` of a
`Process <Process>`, an OutputState is created and assigned as the `sender <MappingProjection.MappingProjection.sender>`
of a `MappingProjection` to the next Mechanism in the pathway, or to the Process' `output <Process_Input_And_Output>`
if the Mechanism is a `TERMINAL` Mechanism for that Process. Other configurations can also easily be specified using
a Mechanism's **output_states** argument (see `OutputState_Specification` below).  If it is created using its
constructor, and a Mechanism is specified in the **owner** argument, it is automatically assigned to that Mechanism.
If its **owner* is not specified, `initialization is deferred.

.. _OutputState_Deferred_Initialization:

Owner Assignment and Deferred Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An OutputState must be owned by a `Mechanism <Mechanism>`.  When OutputState is specified in the constructor for a
`Mechanism <Mechanism>` (see `below <InputState_Specification>`), it is automatically assigned to that Mechanism as
its owner. If the OutputState is created directly, its `owner <OutputState.owner>` Mechanism can specified in the
**owner** argument of its constructor, in which case it is assigned to the specified Mechanism.  Otherwise, its
initialization is `deferred <State_Deferred_Initialization>` until
COMMENT:
TBI: its `owner <State_Base.owner>` attribute is assigned or
COMMENT
the OutputState is assigned to a Mechanism using the Mechanism's `add_states <Mechanism_Base.add_states>` method.

.. _OutputState_Primary:

Primary OutputState
~~~~~~~~~~~~~~~~~~~

Every Mechanism has at least one OutputState, referred to as its *primary OutputState*.  If OutputStates are not
`explicitly specified <OutputState_Specification>` for a Mechanism, a primary OutputState is automatically created
and assigned to its `output_state <Mechanism_Base.output_state>` attribute (note the singular),
and also to the first entry of the Mechanism's `output_states <Mechanism_Base.output_states>` attribute
(note the plural).  The primary OutputState is assigned an `index <OutputState.index>` of '0', and therefore its
`value <OutputState.value>` is assigned as the first (and often only) item of the Mechanism's `value
<Mechanism_Base.value>` attribute.

.. _OutputState_Specification:

OutputState Specification
~~~~~~~~~~~~~~~~~~~~~~~~~

Specifying OutputStates when a Mechanism is created
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OutputStates can be specified for a `Mechanism <Mechanism>` when it is created, in the **output_states** argument of the
Mechanism's constructor (see `examples <State_Constructor_Argument_Examples>` in State), or in an *OUTPUT_STATES* entry
of a parameter dictionary assigned to the constructor's **params** argument.  The latter takes precedence over the
former (that is, if an *OUTPUT_STATES* entry is included in the parameter dictionary, any specified in the
**output_states** argument are ignored).

    .. _OutputState_Replace_Default_Note:

    .. note::
        Assigning OutputStates to a Mechanism in its constructor **replaces** any that are automatically generated for
        that Mechanism (i.e., those that it creates for itself by default).  If any of those need to be retained, they
        must be explicitly specified in the list assigned to the **output_states** argument or the *OUTPUT_STATES* entry
        of the parameter dictionary in the **params** argument).  In particular, if the default OutputState -- that
        usually contains the result of the Mechanism's `function <Mechanism_Base.function>` -- is to be retained,
        it too must be specified along with any additional OutputStates desired.


Adding OutputStates to a Mechanism after it is created
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OutputStates can also be added to a Mechanism, using the Mechanism's `add_states <Mechanism_Base.add_methods>` method.
Unlike specification in the constructor, this **does not** replace any OutputStates already assigned to the Mechanism.
Doing so appends them to the list of OutputStates in the Mechanism's `output_states <Mechanism_Base.output_states>`
attribute, and their values are appended to its `output_values <Mechanism_Base.output_values>` attribute.  If the name
of an OutputState added to a Mechanism is the same as one that is already exists on the Mechanism, its name is suffixed
with a numerical index (incremented for each OutputState with that name; see `Naming`), and the OutputState is added
to the list (that is, it does *not* replace the one that was already there).


.. _OutputState_Variable_and_Value:

*OutputStates* `variable <OutputState.variable>` *and* `value <OutputState.value>`
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Each OutputState created with or assigned to a Mechanism must reference one or more items of the Mechanism's attributes,
that serve as the OutputState's `variable <OutputState.variable>`, and are used by its `function <OutputState.function>`
to generate the OutputState's `value <OutputState.value>`.  By default, it uses the first item of its `owner
<OutputState.owner>` Mechanism's `value <Mechanism_Base.value>`.  However, other attributes (or combinations of them)
can be specified in the **variable** argument of the OutputState's constructor, or the *VARIABLE* entry in an
`OutputState specification dictionary <OutputState_Specification_Dictionary>` (see `OutputState_Customization`).
The specification must be compatible (in the number and type of items it generates) with the input expected by the
OutputState's `function <OutputState.function>`. The OutputState's `variable <OutputState.variable>` is used as the
input to its `function <OutputState.function>` to generate the OutputState's `value <OutputState.value>`, possibly
modulated by a `GatingSignal` (see `below <OutputState_Modulatory_Projections>`).  The OutputState's `value
<OutputState.value>` must, in turn, be compatible with any Projections that are assigned to it, or `used to specify
it <OutputState_Projection_Destination_Specification>`.

The `value <OutputState.value>` of each OutputState of a Mechanism is assigned to a corresponding item of the
Mechanism's `output_values <Mechanism_Base.output_values>` attribute, in the order in which they are assigned in
the **output_states** argument of its constructor, and listed in its `output_states <Mechanism_Base.output_states>`
attribute.

    .. note::
       The `output_values <Mechanism_Base.output_values>` attribute of a Mechanism is **not the same** as its `value
       <Mechanism_Base.value>` attribute:
           * a Mechanism's `value <Mechanism.value>` attribute contains the full and unmodified results of its
             `function <Mechanism_Base.function>`;
           * a Mechanism's `output_values <Mechanism.output_values>` attribute contains a list of the `value
             <OutputState.value>` of each of its OutputStates.

.. _OutputState_Forms_of_Specification:

Forms of Specification
^^^^^^^^^^^^^^^^^^^^^^

OutputStates can be specified in a variety of ways, that fall into three broad categories:  specifying an OutputState
directly; use of a `State specification dictionary <State_Specification>`; or by specifying one or more Components to
which it should project. Each of these is described below:

    .. _OutputState_Direct_Specification:

    **Direct Specification of an OutputState**

    * existing **OutputState object** or the name of one -- it cannot belong to another Mechanism, and the format of
      its `variable <OutputState.variable>` must be compatible with the aributes of the `owner <OutputState.owner>`
      Mechanism specified for the OutputState's `variable <OutputState.variable>` (see `OutputState_Customization`).
    ..
    * **OutputState class**, **keyword** *OUTPUT_STATE*, or a **string** -- creates a default OutputState that uses
      the first item of the `owner <OutputState.owner>` Mechanism's `value <Mechanism_Base.value>` as its `variable
      <OutputState.variable>`, and assigns it as the `owner <OutputState.owner>` Mechanism's `primary OutputState
      <OutputState_Primary>`. If the class name or *OUTPUT_STATE* keyword is used, a default name is assigned to the
      State; if a string is specified, it is used as the `name <OutputState.name>` of the OutputState  (see `Naming`).

    .. _OutputState_Specification_by_Variable:

    * **variable** -- creates an OutputState using the specification as the OutputState's `variable
      <OutputState.variable>` (see `OutputState_Customization`).  This must be compatible with (have the same number
      and type of elements as) the OutputState's `function <OutputState.function>`.  A default name is assigned based
      on the name of the Mechanism (see `Naming`).
    ..
    .. _OutputState_Specification_Dictionary:

    **OutputState Specification Dictionary**

    * **OutputState specification dictionary** -- this can be used to specify the attributes of an OutputState,
      using any of the entries that can be included in a `State specification dictionary <State_Specification>`
      (see `examples <State_Specification_Dictionary_Examples>` in State), including:

      * *VARIABLE*:<keyword or list> - specifies the attribute(s) of its `owner <OutputState.owner>` Mechanism to use
        as the input to the OutputState's `function <OutputState.function>` (see `OutputState_Customization`); this
        must be compatible (in the number and format of the items it specifies) with the OutputState's `function
        <OutputState.function>`.
      |
      * *FUNCTION*:<`Function <Function>`, function or method> - specifies the function used to transform and/or
        combine the item(s) specified for the OutputState's `variable <OutputState.variable>` into its
        `value <OutputState.value>`;  its input must be compatible (in the number and format of elements) with the
        specification of the OutputState's `variable <OutputState.variable>` (see `OutputState_Customization`).
      |
      * *PROJECTIONS* or *MECHANISMS*:<list of `Projections <Projection>` and/or `Mechanisms <Mechanism>`> - specifies
        one or more efferent `MappingProjections <MappingProjection>` from the OutputState, Mechanims that should
        receive them, and/or `ModulatoryProjections <ModulatoryProjection>` for it to receive;  this may be constrained
        by or have consequences for the OutputState's `variable <InputState.variable>` and/or its `value
        <OutputState.value>` (see `OutputState_Compatibility_and_Constraints`).

        .. note::
           The *INDEX* and *ASSIGN* attributes described below have been deprecated in version 0.4.5, and should be
           replaced by use of the *VARIABLE* and *FUNCTION* entries, respectively.  Although use of *INDEX* and *ASSIGN*
           is currently being supported for backward compatibility, this may be eliminated in a future version.

      * *INDEX*:<int> *[DEPRECATED in version 0.4.5]* - specifies the item of the `owner <OutputState.owner>`
        Mechanism's `value <Mechanism_Base.value>` to be used for the OutputState's `variable <OutputState.variable>`;
        equivalent to specifying (OWNER_VALUE, <int>) for *VARIABLE* (see `OutputState_Customization`), which should be
        used for compatibility with future versions.
      |
      * *ASSIGN*:<function> *[DEPRECATED in version 0.4.5]* - specifies the OutputState's `function`
        <OutputState.assign>` attribute;  *FUNCTION* should be used for compatibility with future versions.

    .. _OutputState_Projection_Destination_Specification:

    **Specifying an OutputState by Components to which it Projects**

    COMMENT:
    `examples
      <State_Projections_Examples>` in State)
    COMMENT

    COMMENT:
    ------------------------------------------------------------------------------------------------------------------
    ?? PUT IN ITS OWN SECTION ABOVE OR BELOW??
    Projections from an OutputState can be specified either as attributes, in the constructor for an OutputState (in
    its **projections** argument or in the *PROJECTIONS* entry of an `OutputState specification dictionary
    <OutputState_Specification_Dictionary>`), or used to specify the OutputState itself (using one of the
    `OutputState_Forms_of_Specification` described above. See `State Projections <State_Projections>` for additional
    details concerning the specification of Projections when creating a State.

    .. _OutputState_Projections:

    Projections
    ~~~~~~~~~~~

    When an OutputState is created, it can be assigned one or more `Projections <Projection>`, using either the
    **projections** argument of its constructor, or in an entry of a dictionary assigned to the **params** argument with
    the key *PROJECTIONS*.  An OutputState can be assigned either `MappingProjection(s) <MappingProjection>` or
    `GatingProjection(s) <GatingProjection>`.  MappingProjections are assigned to its `efferents <OutputState.efferents>`
    attribute and GatingProjections to its `mod_afferents <OutputState.mod_afferents>` attribute.  See
    `State Projections <State_Projections>` for additional details concerning the specification of Projections when
    creating a State.
    ------------------------------------------------------------------------------------------------------------------
    COMMENT

    An OutputState can also be specified by specifying one or more Components to or from which it should be assigned
    Projection(s). Specifying an OutputState in this way creates both the OutputState and any of the specified or
    implied Projection(s) (if they don't already exist). `MappingProjections <MappingProjection>`
    are assigned to the OutputState's `efferents <OutputState.efferents>` attribute, and `GatingProjections
    <GatingProjection>` to its `mod_afferents <InputState.mod_afferents>` attribute. Any of the following can be used
    to specify an InputState by the Components that projection to it (see `below
    <OutputState_Compatability_and_Constraints>` for an explanation of the relationship between the `variable` of these
    Components and the OutputState's `value <OutputState.value>`):

    * **InputState, GatingSignal, Mechanism, or list with any of these** -- creates an OutputState with
      the relevant Projection(s).  A `MappingProjection` is created to each InputState or ProcessingMechanism specified
      (for a Mechanism, its `primary InputState <InputState_Primary>` is used). A `GatingProjection` is created for
      each GatingSignal or GatingMechamism specified (for a GatingMechanism, its first GatingSignal is used).
    ..
    * **Projection** -- any form of `Projection specification <Projection_Specification>` can be used; creates an
      OutputState and assigns it as the `sender <MappingProjection.sender>` for any MappingProjections specified, and
      as the `receiver <GatingProjection.receiver>` for any GatingProjections specified.

    .. _OutputState_Tuple_Specification:

    * **OutputState specification tuples** -- these are convenience formats that can be used to compactly specify an
      OutputState along with Projections to or from it in any of the following ways:

        * **2-item tuple:** *(State name or list of State names, Mechanism)* -- 1st item must be the name of an
          `InputState` or `ModulatorySignal`, or a list of such names, and the 2nd item must be the Mechanism to
          which they all belong.  Projections of the relevant types are created for each of the specified States
          (see `State 2-item tuple <State_2_Item_Tuple>` for additional details).
        |
        * **2-item tuple:** *(<State, Mechanism, or list of them>, Projection specification)* -- this is a contracted
          form of the 3-item tuple described below
        |
        * **3-item tuple:** *(<value, State spec, or list of State specs>, variable spec, Projection specification)* --
          this allows the specification of State(s) to which the OutputState should project, together with a
          specification of its `variable <OutputState.variable>` attribute, and (optionally) parameters of the
          Projection(s) to use (e.g., their `weight <Projection_Base.weight>` and/or `exponent
          <Projection_Base.exponent>` attributes.  Each tuple must have at least the first two items (in the
          order listed), and can include the third:

            * **value, State specification, or list of State specifications** -- specifies either the `variable
              <InputState.variable>` of the InputState, or one or more States that should project to it.  The State
              specification(s) can be a (State name, Mechanism) tuple (see above), and/or include Mechanisms, in which
              case their `primary InputState <InputStatePrimary>` is used.  All of the State specifications must be
              consistent with (that is, their `value <State_Base.value>` must be compatible with the `variable
              <Projection_Base.variable>` of) the Projection specified in the fourth item if that is included.
            |
            * **variable spec** -- specifies the attributes of the OutputState's `owner <OutputState.owner>` Mechanism
              used for its `variable <OutputState.variable>` (see `OutputState_Customization`).
            |
            * **Projection specification** (optional) -- `specifies a Projection <Projection_Specification>` that
              must be compatible with the State specification(s) in the 1st item; if there is more than one
              State specified, and the Projection specification is used, all of the States
              must be of the same type (i.e.,either InputStates or GatingSignals), and the `Projection
              Specification <Projection_Specification>` cannot be an instantiated Projection (since a
              Projection cannot be assigned more than one `receiver <Projection_Base.receiver>`).

.. _OutputState_Compatibility_and_Constraints:

OutputState `variable <OutputState.variable>` and `value <OutputState.value>`: Compatibility and Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The format of an OutputStates' `variable <OutputState.variable>` may have consequences that must be taken into
account when `specifying an OutputState by Components to which it projects
<OutputState_Projection_Destination_Specification>`.  These depend on the context in which the specification is
made, and possibly the value of other specifications.  These considerations and how they are handled are described
below, starting with constraints that are given the highest precedence:

  * **OutputState specified in a Mechanism's constructor** -- the specification of the OutputState's `variable
    <OutputState.variable>`, together with its `function <OutputState.function>` determine the OutputState's `value
    <OutputState.value>` (see `above <OutputState_Variable_and_Value>`).  Therefore, any specifications of the
    OutputState relevant to its `value <OutputState.value>` must also be compatible with these factors (for example,
    `specifying it by variable <OutputState_Specification_by_Variable>` or by a `MappingProjection` or an
    `InputState` to which it should project (see `above <OutputState_Projection_Destination_Specification>`).

    COMMENT:
    ***XXX EXAMPLE HERE
    COMMENT
  ..
  * **OutputState specified on its own** -- any direct specification of the OutputState's `variable
    <OutputState.variable>` is used to determine its format (e.g., `specifying it by variable
    <OutputState_Specification_by_Variable>`, or a *VARIABLE* entry in an `OutputState specification dictionary
    <OutputState_Specification_Dictionary>`.  In this case, the value of any `Components used to specify the
    OutputState <OutputState_Projection_Destination_Specification>` must be compatible with the specification of its
    `variable <OutputState.variable>` and the consequences this has for its `value <OutputState.value>` (see below).

    COMMENT:
    ***XXX EXAMPLE HERE
    COMMENT
  ..
  * **OutputState's** `value <OutputState.value>` **not constrained by any of the conditions above** -- then its
    `variable <OutputState.variable>` is determined by the default for an OutputState (the format of the first
    item of its `owner <OutputState.owner>` Mechanism's `value <Mechanism_Base.value>` ). If the OutputState is
    `specified to project to any other Components <OutputState_Projection_Destination_Specification>`, then if the
    Component is a:

    |
    * **InputState or Mechanism** (for which its `primary InputState <InputState_Primary>` is used) -- if its
      `variable <State_Base.variable>` matches the format of the OutputState's `value <OutputState.value>`, a
      `MappingProjection` is created using an `IDENTITY_MATRIX`;  otherwise, a `FULL_CONNECTIVITY_MATRIX` is used
      that maps the OutputState's `value <OutputState.value>` to the InputState's `variable <State_Base.variable>`.
    |
    * **MappingProjection** -- if its `matrix <MappingProjection.matrix>` is specified, then the `sender dimensionality
      <Mapping_Matrix_Dimensionality>` of the matrix must be the same as that of the OutputState's `value
      <OutputState.value>`; if its `receiver <MappingProjection.receiver>` is specified, but not its `matrix
      <MappingProjection.matrix>`, then a matrix is chosen that appropriately maps from the OutputState to the
      receiver (as described just above);  if neither its `matrix <MappingProjection.matrix>` or its `receiver
      <MappingProjection.receiver>` are specified, then the Projection's `initialization is deferred
      <MappingProjection_Deferred_Initialization>` until its `receiver <MappingProjection.receiver>` is specified.
    |
    * **GatingProjection, GatingSignal or GatingMechanism** -- any of these can be used to specify an OutputState;
      their `value` does not need to be compatible with the OutputState's `variable <InputState.variable>` or
      `value <OutputState.value>`, however it does have to be compatible with the `modulatory parameter
      <Function_Modulatory_Params>` of the OutputState's `function <OutputState.function>`.


.. _OutputState_Standard:

Standard OutputStates
^^^^^^^^^^^^^^^^^^^^^

Most types of Mechanisms have a `standard_output_states` class attribute, that contains a list of predefined
OutputStates relevant to that type of Mechanism (for example, the `TransferMechanism` class has OutputStates for
calculating the mean, median, variance, and standard deviation of its result).  The names of these are listed as
attributes of a class with the name *<ABBREVIATED_CLASS_NAME>_OUTPUT*.  For example, the TransferMechanism class
defines `TRANSFER_OUTPUT`, with attributes *MEAN*, *MEDIAN*, *VARIANCE* and *STANDARD_DEVIATION* that are the names of
predefined OutputStates in its `standard_output_states <TransferMechanism.standard_output_states>` attribute.
These can be used in the list of OutputStates specified for a TransferMechanism object, as in the following example::

    >>> import psyneulink as pnl
    >>> my_mech = pnl.TransferMechanism(default_variable=[0,0],
    ...                                 function=pnl.Logistic(),
    ...                                 output_states=[pnl.TRANSFER_OUTPUT.RESULT,
    ...                                                pnl.TRANSFER_OUTPUT.MEAN,
    ...                                                pnl.TRANSFER_OUTPUT.VARIANCE])

In this example, ``my_mech`` is configured with three OutputStates;  the first will be named *RESULT* and will
represent logistic transform of the 2-element input vector;  the second will be named  *MEAN* and will represent mean
of the result (i.e., of its two elements); and the third will be named *VARIANCE* and contain the variance of the
result.

.. _OutputState_Customization:

OutputState Customization
~~~~~~~~~~~~~~~~~~~~~~~~~

An OutputState's `value <OutputState.value>` can be customized by specifying its `variable <OutputState.variable>`
and/or `function <OutputState.function>` in the **variable** and **function** arguments of the OutputState's
constructor, the corresponding entries (*VARIABLE* and *FUNCTION*) of an `OutputState specification
dictionary <OutputState_Specification_Dictionary>`, or in the variable spec (2nd) item of a `3-item tuple
<OutputState_Tuple_Specification>` for the OutputState.

*OutputState* `variable <OutputState.variable>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, an OutputState uses the first (and usually only) item of the owner Mechanism's `value
<Mechanism_Base.value>` as its `variable <OutputState.variable>`.  However, this can be customized by specifying
any other item of its `owner <OutputState.owner>`\\s `value <Mechanism_Base.value>`, the full `value
<Mechanism_Base.value>` itself, other attributes of the `owner <OutputState.owner>`, or any combination of these
using the following keywords:

    *OWNER_VALUE* -- the entire `value <Mechanism_Base.value>` of the OutputState's `owner <OutputState.owner>`.

    *(OWNER_VALUE, <int>)* -- tuple specifying an item of the `owner <OutputState.owner>`\\'s `value
    <Mechanism_Base.value>` indexed by the int;  indexing begins with 0 (e.g.; 1 references the 2nd item).

    *<attribute name>* -- the name of an attribute of the OutputState's `owner <OutputState.owner>` (must be one
    in the `owner <OutputState.owner>`\\'s `params_dict <Mechanism.attributes_dict>` dictionary); returns the value
    of the named attribute for use in the OutputState's `variable <OutputState.variable>`.

    *PARAMS_DICT* -- the `owner <OutputState.owner>` Mechanism's entire `params_dict <Mechanism.attributes_dict>`
    dictionary, that contains entries for all of it accessible attributes.  The OutputState's `function
    <OutputState.function>` must be able to parse the dictionary.
    COMMENT
    ??WHERE CAN THE USER GET THE LIST OF ALLOWABLE ATTRIBUTES?  USER_PARAMS?? aTTRIBUTES_DICT?? USER ACCESSIBLE PARAMS??
    COMMENT

    *List[<any of the above items>]* -- this assigns the value of each item in the list to the corresponding item of
    the OutputState's `variable <OutputState.variable>`.  This must be compatible (in number and type of elements) with
    the input expected by the OutputState's `function <OutputState.function>`.

*OutputState* `function <OutputState.function>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the `function <OutputState.function>` of an OutputState is `Linear`, which simply assigns the
OutputState's `variable <OutputState.variable>` as its `value <OutputState.value>`.  However, a different function
can be assigned, to transform and/or combine the item(s) of the OutputState's `variable <OutputState.variable>`
for use as its `value <OutputState.value>`. The function can be a PsyNeuLink `Function <Function>` or any Python
function or method, so long as the input it expects is compatible (in number and format of elements) with the
OutputState's `variable <OutputState.variable>`.

Examples
--------

In the following example, a `DDM` Mechanism named ``my_mech`` is configured with three OutputStates:

COMMENT:
(also see `OutputState_Structure` below). If the
Mechanism's `function
<Mechanism_Base.function>` returns a value with more than one item (i.e., a list of lists, or a 2d np.array), then an
OutputState can be assigned to any of those items by specifying its `index <OutputState.index>` attribute. An
OutputState can also be configured to transform the value of the item, by specifying a function for its `assign
<OutputState.assign>` attribute; the result will then be assigned as the OutputState's `value <OutputState.value>`.
An OutputState's `index <OutputState.index>` and `assign <OutputState.assign>` attributes can be assigned when
the OutputState is assigned to a Mechanism, by including *INDEX* and *ASSIGN* entries in a `specification dictionary
<OutputState_Specification>` for the OutputState, as in the following example::
COMMENT

    >>> my_mech = pnl.DDM(function=pnl.BogaczEtAl(),
    ...                   output_states=[pnl.DDM_OUTPUT.DECISION_VARIABLE,
    ...                                  pnl.DDM_OUTPUT.PROBABILITY_UPPER_THRESHOLD,
    ...                                  {pnl.NAME: 'DECISION ENTROPY',
    ...                                   pnl.VARIABLE: (pnl.OWNER_VALUE, 2),
    ...                                   pnl.FUNCTION: pnl.Stability(metric=pnl.ENTROPY) }])

COMMENT:
   ADD VERSION IN WHICH INDEX IS SPECIFIED USING DDM_standard_output_states
   CW 3/20/18: TODO: this example is flawed: if you try to execute() it, it gives divide by zero error.
COMMENT

The first two are `Standard OutputStates <OutputState_Standard>` that represent the decision variable of the DDM and
the probability of it crossing of the upper (vs. lower) threshold. The third is a custom OutputState, that computes
the entropy of the probability of crossing the upper threshold.  It uses the 3rd item of the DDM's `value <DDM.value>`
(items are indexed starting with 0), which contains the `probability of crossing the upper threshold
<DDM_PROBABILITY_UPPER_THRESHOLD>`, and uses this as the input to the `Stability` Function assigned as the
OutputState's `function <OutputState.function>`, that computes the entropy of the probability. The three OutputStates
will be assigned to the `output_states <Mechanism_Base.output_states>` attribute of ``my_mech``, and their values
will be assigned as items in its `output_values <Mechanism_Base.output_values>` attribute, in the order in which they
are listed in the **output_states** argument of the constructor for ``my_mech``.

Custom OutputStates can also be created on their own, and separately assigned or added to a Mechanism.  For example,
the ``DECISION ENTROPY`` OutputState could be created as follows::

    >>> decision_entropy_output_state = pnl.OutputState(name='DECISION ENTROPY',
    ...                                                 variable=(OWNER_VALUE, 2),
    ...                                                 function=pnl.Stability(metric=pnl.ENTROPY))

and then assigned either as::

    >>> my_mech = pnl.DDM(function=pnl.BogaczEtAl(),
    ...                   output_states=[pnl.DDM_OUTPUT.DECISION_VARIABLE,
    ...                                  pnl.DDM_OUTPUT.PROBABILITY_UPPER_THRESHOLD,
    ...                                  decision_entropy_output_state])

or::

    >>> another_decision_entropy_output_state = pnl.OutputState(name='DECISION ENTROPY',
    ...                                                variable=(OWNER_VALUE, 2),
    ...                                                function=pnl.Stability(metric=pnl.ENTROPY))
    >>> my_mech2 = pnl.DDM(function=pnl.BogaczEtAl(),
    ...                    output_states=[pnl.DDM_OUTPUT.DECISION_VARIABLE,
    ...                                   pnl.DDM_OUTPUT.PROBABILITY_UPPER_THRESHOLD])

    >>> my_mech2.add_states(another_decision_entropy_output_state) # doctest: +SKIP

Note that another new OutputState had to be used for the second example, as trying to
add the first one created for ``my_mech`` to ``my_mech2`` would have produced an error (since a State already
belonging to one Mechanism can't be added to another.

.. _OutputState_Structure:

Structure
---------

Every OutputState is owned by a `Mechanism <Mechanism>`. It can send one or more `MappingProjections
<MappingProjection>` to other Mechanisms.  If its owner is a `TERMINAL` Mechanism of a Process and/or System, then the
OutputState will also be treated as the output of that `Process <Process_Input_And_Output>` and/or of a System.  It has
the following attributes, some of which can be specified in ways that are specific to, and that can be used to
`customize, the OutputState <OutputState_Customization>`:

COMMENT:
.. _OutputState_Index:

* `index <OutputState.index>`: this determines the item of its owner Mechanism's `value <Mechanism_Base.value>` to
  which it is assigned.  By default, this is set to 0, which assigns it to the first item of the Mechanism's `value
  <Mechanism_Base.value>`.  The `index <Mechanism_Base.index>` must be equal to or less than one minus the number of
  OutputStates listed in the Mechanism's `output_states <Mechanism_Base.output_states>` attribute.  The `variable
  <OutputState.variable>` of the OutputState must also match (in the number and type of its elements) the item of the
  Mechanism's `value <Mechanism_Base.value>` designated by the `index <OutputState.index>`.

.. _OutputState_Assign:

* `assign <OutputState.assign>`:  this specifies a function used to convert the item of the owner Mechanism's
  `value <Mechanism_Base.value>` (designated by the OutputState's `index <OutputState.index>` attribute), before
  providing it to the OutputState's `function <OutputState.function>`.  The `assign <OutputState.assign>`
  attribute can be assigned any function that accept the OutputState's `variable <OutputState.variable>` as its input,
  and that generates a result that can be used the input for the OutputState's `function <OutputState.function>`.
  The default is an identity function (`Linear` with **slope**\\ =1 and **intercept**\\ =0), that simply assigns the
  specified item of the Mechanism's `value <Mechanism_Base.value>` unmodified as the input for OutputState's
  `function <OutputState.function>`.
COMMENT
..
* `variable <OutputState.variable>` -- references attributes of the OutputState's `owner <OutputState.owner>` that
  are used as the input to the OutputState's `function <OutputState.function>`, to determine its `value
  <OutputState.value>`.  The specification must match (in both the number and types of elements it generates)
  the input to the OutputState's `function <OutputState.function>`.  By default, the first item of the `owner
  <OutputState.owner>` Mechanisms' `value <Mechanism_Base.value>` is used.  However, this can be customized as
  described under `OutputState_Customization`.

* `function <OutputState.function>` -- takes the OutputState's `variable <OutputState.variable>` as its input, and
  generates the OutputState's `value <OutputState.value>` as its result.  The default function is `Linear` that simply
  assigns the OutputState's `variable <OutputState.variable>` as its `value <OutputState.value>`.  However, the
  parameters of the `function <OutputState.function>` -- and thus the `value <OutputState.value>` of the OutputState --
  can be modified by `GatingProjections <GatingProjection>` received by the OutputState (listed in its
  `mod_afferents <OutputState.mod_afferents>` attribute.  A custom function can also be specified, so long as it can
  take as its input a value that is compatible with the OutputState's `variable <OutputState.variable>`.

* `projections <OutputState.projections>` -- all of the `Projections <Projection>` sent and received by the OutputState;

.. _OutputState_Efferent_Projections:

* `efferents <OutputState.path_afferents>` -- `MappingProjections <MappingProjection>` that project from the
  OutputState.

.. _OutputState_Modulatory_Projections:

* `mod_afferents <OutputState.mod_afferents>` -- `GatingProjections <GatingProjection>` that project to the OutputState,
  the `value <GatingProjection.value>` of which can modify the OutputState's `value <InputState.value>` (see the
  descriptions of Modulation under `ModulatorySignals <ModulatorySignal_Modulation>` and `GatingSignals
  <GatingSignal_Modulation>` for additional details).  If the OutputState receives more than one GatingProjection,
  their values are combined before they are used to modify the `value <OutputState.value>` of the OutputState.
..
* `value <OutputState.value>`:  assigned the result of the OutputState's `function <OutputState.function>`, possibly
  modified by any `GatingProjections <GatingProjection>` received by the OutputState. It is used as the input to any
  projections that the OutputStatue sends.


.. _OutputState_Execution:

Execution
---------

An OutputState cannot be executed directly.  It is executed when the Mechanism to which it belongs is executed.
When the Mechanism is executed, the values of its attributes specified for the OutputState's `variable
<OutputState.variable>` (see `OutputState_Customization`) are used as the input to the OutputState's `function
<OutputState.function>`. The OutputState is updated by calling its `function <OutputState.function>`.  The result,
modified by any `GatingProjections <GatingProjection>` the OutputState receives (listed in its `mod_afferents
<OutputState.mod_afferents>` attribute), is assigned as the `value <OutputState.value>` of the OutputState.  This is
assigned to a corresponding item of the Mechanism's `output_values <Mechanism_Base.output_values>` attribute,
and is used as the input to any projections for which the OutputState is the `sender <Projection_Base.sender>`.

.. _OutputState_Class_Reference:

Class Reference
---------------


"""

import warnings

import numpy as np
import typecheck as tc

from psyneulink.components.component import Component
from psyneulink.components.functions.function import Function, OneHot, function_type, method_type
from psyneulink.components.shellclasses import Mechanism
from psyneulink.components.states.state import State_Base, _instantiate_state_list, state_type_keywords
from psyneulink.globals.context import ContextFlags
from psyneulink.globals.keywords import ALL, ASSIGN, CALCULATE, COMMAND_LINE, FUNCTION, GATING_SIGNAL, INDEX, INPUT_STATE, INPUT_STATES, MAPPING_PROJECTION, MAX_ABS_INDICATOR, MAX_ABS_VAL, MAX_INDICATOR, MAX_VAL, MEAN, MECHANISM_VALUE, MEDIAN, NAME, OUTPUT_STATE, OUTPUT_STATE_PARAMS, OWNER_VALUE, PARAMS, PARAMS_DICT, PROB, PROJECTION, PROJECTIONS, PROJECTION_TYPE, RECEIVER, REFERENCE_VALUE, RESULT, STANDARD_DEVIATION, STANDARD_OUTPUT_STATES, STATE, VALUE, VARIABLE, VARIANCE
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.globals.utilities import UtilitiesError, is_numeric, iscompatible, recursive_update

__all__ = [
    'make_readonly_property', 'OUTPUTS', 'OutputState', 'OutputStateError', 'PRIMARY', 'SEQUENTIAL',
    'standard_output_states', 'StandardOutputStates', 'StandardOutputStatesError', 'state_type_keywords',
]

state_type_keywords = state_type_keywords.update({OUTPUT_STATE})

# class OutputStateLog(IntEnum):
#     NONE            = 0
#     TIME_STAMP      = 1 << 0
#     ALL = TIME_STAMP
#     DEFAULTS = NONE

OUTPUT_STATE_TYPE = 'outputStateType'

# Used to specify how StandardOutputStates are indexed
PRIMARY = 0
SEQUENTIAL = 'SEQUENTIAL'

DEFAULT_VARIABLE_SPEC = (OWNER_VALUE, 0)

# This is a convenience class that provides list of standard_output_state names in IDE
class OUTPUTS():
    RESULT=RESULT
    MEAN=MEAN
    MEDIAN=MEDIAN
    STANDARD_DEVIATION=STANDARD_DEVIATION
    VARIANCE=VARIANCE
    MECHANISM_VALUE=MECHANISM_VALUE
    MAX_VAL=MAX_VAL
    MAX_ABS_VAL=MAX_VAL
    MAX_INDICATOR=MAX_INDICATOR
    MAX_ABS_INDICATOR=MAX_INDICATOR
    PROB=PROB

standard_output_states = [{NAME: RESULT},
                          {NAME:MEAN,
                           FUNCTION:lambda x: np.mean(x)},
                          {NAME:MEDIAN,
                           FUNCTION:lambda x: np.median(x)},
                          {NAME:STANDARD_DEVIATION,
                           FUNCTION:lambda x: np.std(x)},
                          {NAME:VARIANCE,
                           FUNCTION:lambda x: np.var(x)},
                          {NAME: MECHANISM_VALUE,
                           VARIABLE: OWNER_VALUE},
                          {NAME: OWNER_VALUE,
                           VARIABLE: OWNER_VALUE},
                          {NAME: MAX_VAL,
                           FUNCTION: OneHot(mode=MAX_VAL).function},
                          {NAME: MAX_ABS_VAL,
                           FUNCTION: OneHot(mode=MAX_ABS_VAL).function},
                          {NAME: MAX_INDICATOR,
                           FUNCTION: OneHot(mode=MAX_INDICATOR).function},
                          {NAME: MAX_ABS_INDICATOR,
                           FUNCTION: OneHot(mode=MAX_ABS_INDICATOR).function},
                          {NAME: PROB,
                           FUNCTION: OneHot(mode=PROB).function}
                          ]


class OutputStateError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class OutputState(State_Base):
    """
    OutputState(          \
    owner,                \
    reference_value,      \
    variable=None,        \
    size=None,            \
    function=Linear(),    \
    index=PRIMARY,        \
    assign=None,          \
    projections=None,     \
    params=None,          \
    name=None,            \
    prefs=None,           \
    context=None)

    Subclass of `State <State>` that calculates and represents an output of a `Mechanism <Mechanism>`.

    COMMENT:

        Description
        -----------
            The OutputState class is a type in the State category of Component,
            It is used primarily as the sender for MappingProjections
            Its FUNCTION updates its value:
                note:  currently, this is the identity function, that simply maps variable to self.value

        Class attributes:
            + componentType (str) = OUTPUT_STATES
            + paramClassDefaults (dict)
                + FUNCTION (Linear)
                + FUNCTION_PARAMS   (Operation.PRODUCT)

        Class methods:
            function (executes function specified in params[FUNCTION];  default: Linear)

        StateRegistry
        -------------
            All OutputStates are registered in StateRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances

    COMMENT


    Arguments
    ---------

    owner : Mechanism
        the `Mechanism <Mechanism>` to which the OutputState belongs; it must be specified or determinable from the
        context in which the OutputState is created.

    reference_value : number, list or np.ndarray
        a template that specifies the format of the OutputState's `variable <OutputState.variable>`;  if it is
        specified in addition to the **variable** argument, then these must be compatible (both in the number and
        type of elements).  It is used to insure the compatibility of the source of the input for the OutputState
        with its `function <OutputState.function>`.

    variable : number, list or np.ndarray
        specifies the attributes of the  OutputState's `owner <OutputState.owner>` Mechanism to be used by the
        OutputState's `function <OutputState.function>`  in generating its `value <OutputState.value>`.

    COMMENT:
    size : int, list or ndarray of ints
        specifies variable as array(s) of zeros if **variable** is not passed as an argument;
        if **variable** is specified, it takes precedence over the specification of **size**.
        As an example, the following mechanisms are equivalent::
            T1 = TransferMechanism(size = [3, 2])
            T2 = TransferMechanism(default_variable = [[0, 0, 0], [0, 0]])
    COMMENT

    function : Function, function, or method : default Linear
        specifies the function used to transform and/or combine the items designated by the OutputState's `variable
        <OutputState.variable>` into its `value <OutputState.value>`, under the possible influence of
        `GatingProjections <GatingProjection>` received by the OutputState.

    COMMENT:
    index : int : default PRIMARY
        specifies the item of the owner Mechanism's `value <Mechanism_Base.value>` used as input for the
        function specified by the OutputState's `assign <OutputState.assign>` attribute, to determine the
        OutputState's `value <OutputState.value>`.

    assign : Function, function, or method : default Linear
        specifies the function used to convert the designated item of the owner Mechanism's
        `value <Mechanism_Base.value>` (specified by the OutputState's :keyword:`index` attribute),
        before it is assigned as the OutputState's `value <OutputState.value>`.  The function must accept a value that
        has the same format (number and type of elements) as the item of the Mechanism's
        `value <Mechanism_Base.value>`.
    COMMENT

    projections : list of Projection specifications
        species the `MappingProjection(s) <MappingProjection>` to be sent by the OutputState, and/or
        `GatingProjections(s) <GatingProjection>` to be received (see `OutputState_Projections` for additional details);
        these will be listed in its `efferents <OutputState.efferents>` and `mod_afferents <InputState.mod_afferents>`
        attributes, respectively (see `OutputState_Projections` for additional details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the OutputState, its function, and/or a custom function and its parameters. Values specified for parameters
        in the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default see `name <OutputState.name>`
        specifies the name of the OutputState; see OutputState `name <OutputState.name>` for details.

    prefs : PreferenceSet or specification dict : default State.classPreferences
        specifies the `PreferenceSet` for the OutputState; see `prefs <OutputState.prefs>` for details.


    Attributes
    ----------

    owner : Mechanism
        the Mechanism to which the OutputState belongs.

    mod_afferents : List[GatingProjection]
        `GatingProjections <GatingProjection>` received by the OutputState.

    variable : value, list or np.ndarray
        the value(s) of the item(s) of the `owner <OutputState.owner>` Mechanism's attributes specified in the
        **variable** argument of the constructor, or a *VARIABLE* entry in the `OutputState specification dictionary
        <OutputState_Specification_Dictionary>` used to construct the OutputState.

    COMMENT:
    index : int
        the item of the owner Mechanism's `value <Mechanism_Base.value>` used as input for the function specified by
        its `assign <OutputState.assign>` attribute (see `index <OutputState_Index>` for additional details).

    assign : function or method
        function used to convert the item of the owner Mechanism's `value <Mechanism_Base.value>` specified by
        the OutputState's `index <OutputState.index>` attribute.  The result is combined with the result of the
        OutputState's `function <OutputState.function>` to determine both the `value <OutputState.value>` of the
        OutputState, as well as the value of the corresponding item of the owner Mechanism's `output_values
        <Mechanism_Base.output_values>`. The default (`Linear`) transfers the value unmodified  (see `assign
        <OutputState_Assign>` for additional details)
    COMMENT

    function : Function, function, or method
        function used to transform and/or combine the value of the items of the OutputState's `variable
        <OutputState.variable>` into its `value <OutputState.value>`, under the possible influence of
        `GatingProjections <GatingProjection>` received by the OutputState.

    value : number, list or np.ndarray
        assigned the result of `function <OutputState.function>`;  the same value is assigned to the corresponding item
        of the owner Mechanism's `output_values <Mechanism_Base.output_values>` attribute.

    efferents : List[MappingProjection]
        `MappingProjections <MappingProjection>` sent by the OutputState (i.e., for which the OutputState
        is a `sender <Projection_Base.sender>`).

    projections : List[Projection]
        all of the `Projections <Projection>` received and sent by the OutputState.

    name : str
        the name of the OutputState; if it is not specified in the **name** argument of the constructor, a default is
        assigned by the OutputStateRegistry of the Mechanism to which the OutputState belongs.  Note that most
        Mechanisms automatically create one or more `Standard OutputStates <OutputState_Standard>`, that have
        pre-specified names.  However, if any OutputStates are specified in the **input_states** argument of the
        Mechanism's constructor, those replace its Standard OutputStates (see `note
        <Mechanism_Default_State_Suppression_Note>`);  `standard naming conventions <Naming>` apply to the
        OutputStates specified, as well as any that are added to the Mechanism once it is created.

        .. note::
            Unlike other PsyNeuLink components, State names are "scoped" within a Mechanism, meaning that States with
            the same name are permitted in different Mechanisms.  However, they are *not* permitted in the same
            Mechanism: States within a Mechanism with the same base name are appended an index in the order of their
            creation.

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the OutputState; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    """

    #region CLASS ATTRIBUTES

    componentType = OUTPUT_STATE
    paramsType = OUTPUT_STATE_PARAMS

    # stateAttributes = State_Base.stateAttributes | {INDEX, ASSIGN}

    connectsWith = [INPUT_STATE, GATING_SIGNAL]
    connectsWithAttribute = [INPUT_STATES]
    projectionSocket = RECEIVER
    modulators = [GATING_SIGNAL]

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'OutputStateCustomClassPreferences',
    #     kp<pref>: <setting>...}

    paramClassDefaults = State_Base.paramClassDefaults.copy()
    paramClassDefaults.update({PROJECTION_TYPE: MAPPING_PROJECTION,
                               # DEFAULT_VARIABLE_SPEC: [(OWNER_VALUE, 0)]
                               })
    #endregion

    @tc.typecheck
    def __init__(self,
                 owner=None,
                 reference_value=None,
                 variable=None,
                 size=None,
                 function=None,
                 projections=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None,
                 **kwargs):

        if context is None:
            context = ContextFlags.COMMAND_LINE
            self.context.source = ContextFlags.COMMAND_LINE
            self.context.string = COMMAND_LINE
        else:
            context = ContextFlags.CONSTRUCTOR
            self.context.source = ContextFlags.CONSTRUCTOR

        # For backward compatibility with CALCULATE, ASSIGN and INDEX
        if 'calculate' in kwargs:
            assign = kwargs['calculate']
        if params:
            _maintain_backward_compatibility(params, name, owner)

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(
                function=function,
                params=params)

        # If owner or reference_value has not been assigned, defer init to State._instantiate_projection()
        # if owner is None or reference_value is None:
        if owner is None:
            # Temporarily name OutputState
            self._assign_deferred_init_name(name, context)
            # Store args for deferred initialization
            self.init_args = locals().copy()
            del self.init_args['kwargs']
            self.init_args['context'] = context
            self.init_args['name'] = name
            self.init_args['projections'] = projections

            # Flag for deferred initialization
            self.context.initialization_status = ContextFlags.DEFERRED_INIT
            return

        self.reference_value = reference_value

        # FIX: PUT THIS IN DEDICATED OVERRIDE OF COMPONENT VARIABLE-SETTING METHOD??
        if variable is None:
            if reference_value is None:
                # variable = owner.instance_defaults.value[0]
                # variable = self.paramClassDefaults[DEFAULT_VARIABLE_SPEC] # Default is 1st item of owner.value
                variable = DEFAULT_VARIABLE_SPEC
            else:
                variable = reference_value
        # MODIFIED 3/10/18 OLD:
        if not is_numeric(variable):
            self._variable = variable
        # # MODIFIED 3/10/18 NEW:
        # # FIX: SHOULD HANDLE THIS MORE GRACEFULLY IN _instantiate_state and/or instaniate_output_state
        # # If variable is numeric, assume it is a default spec passed in that had been parsed for initializatoin purposes
        # if is_numeric(variable):
        #     # self._variable = self.paramClassDefaults[DEFAULT_VARIABLE_SPEC]
        #     self._variable = DEFAULT_VARIABLE_SPEC
        # else:
        #     self._variable = variable
        # MODIFIED 3/10/18 END:


        # FIX: 5/26/16
        # IMPLEMENTATION NOTE:
        # Consider adding self to owner.output_states here (and removing from ControlProjection._instantiate_sender)
        #  (test for it, and create if necessary, as per OutputStates in ControlProjection._instantiate_sender),

        # Validate sender (as variable) and params, and assign to variable and paramInstanceDefaults
        super().__init__(owner,
                         variable=variable,
                         size=size,
                         projections=projections,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context,
                         function=function,
                         )

    def _parse_function_variable(self, variable):
        # variable is passed to OutputState by _instantiate_function for OutputState
        if variable is not None:
            return variable
        # otherwise, OutputState uses specified item(s) of owner's value
        else:
            # variable attribute should not be used for computations!
            fct_var = self.variable

            # If variable is not specified, check if OutputState has index attribute
            #    (for backward compatibility with INDEX and ASSIGN)
            if fct_var is None:
                try:
                    # Get indexed item of owner's value
                    fct_var = self.owner.value[self.index]
                except IndexError:
                    # Index is ALL, so use owner's entire value
                    if self.index is ALL:
                        fct_var = self.owner.value
                    else:
                        raise IndexError
                except AttributeError:
                    raise OutputStateError("PROGRAM ERROR: Failure to parse variable for {} of {}".
                                           format(self.name, self.owner.name))

            return fct_var

    def _validate_against_reference_value(self, reference_value):
        """Validate that State.variable is compatible with the reference_value

        reference_value is the value of the Mechanism to which the OutputState is assigned
        """
        return

    def _instantiate_attributes_before_function(self, function=None, context=None):
        """Instantiate default variable if it is None or numeric
        :param function:
        """
        super()._instantiate_attributes_before_function(function=function, context=context)

        # If variable has not been assigned, or it is numeric (in which case it can be assumed that
        #    the value was a reference_value generated during initialization/parsing and passed in the constructor
        if self._variable is None or is_numeric(self._variable):
            self._variable = DEFAULT_VARIABLE_SPEC

    def _instantiate_projections(self, projections, context=None):
        """Instantiate Projections specified in PROJECTIONS entry of params arg of State's constructor

        Assume specification in projections as ModulatoryProjection if it is a:
            ModulatoryProjection
            ModulatorySignal
            AdaptiveMechanism
        Call _instantiate_projections_to_state to assign ModulatoryProjections to .mod_afferents

        Assume all remaining specifications in projections are for outgoing MappingProjections;
            these should be either Mechanisms, States or MappingProjections to one of those
        Call _instantiate_projections_from_state to assign MappingProjections to .efferents

        Store result of function as self.function_value
        function_value is converted to returned value by assign function

        """
        from psyneulink.components.states.modulatorysignals.modulatorysignal import \
            _is_modulatory_spec
        from psyneulink.components.projections.pathway.mappingprojection import MappingProjection

        # Treat as ModulatoryProjection spec if it is a ModulatoryProjection, ModulatorySignal or AdaptiveMechanism
        # or one of those is the first or last item of a ProjectionTuple
        # modulatory_projections = [proj for proj in projections
        #                           if (isinstance(proj, (ModulatoryProjection_Base,
        #                                                ModulatorySignal,
        #                                                AdaptiveMechanism_Base)) or
        #                               (isinstance(proj, ProjectionTuple) and
        #                                any(isinstance(item, (ModulatoryProjection_Base,
        #                                                    ModulatorySignal,
        #                                                    AdaptiveMechanism_Base)) for item in proj)))]
        # modulatory_projections = [proj for proj in projections
        #                           if ((_is_modulatory_spec(proj) and
        #                               isinstance(proj, ProjectionTuple)) or
        #                                any((_is_modulatory_spec(item)
        #                                     or isinstance(item, ProjectionTuple))
        #                                    for item in proj))]
        modulatory_projections = [proj for proj in projections if _is_modulatory_spec(proj)]
        self._instantiate_projections_to_state(projections=modulatory_projections, context=context)

        # Treat all remaining specifications in projections as ones for outgoing MappingProjections
        pathway_projections = [proj for proj in projections if not proj in modulatory_projections]
        for proj in pathway_projections:
            self._instantiate_projection_from_state(projection_spec=MappingProjection,
                                                    receiver=proj,
                                                    context=context)

    def _get_primary_state(self, mechanism):
        return mechanism.output_state

    def _parse_arg_variable(self, default_variable):
        return _parse_output_state_variable(self.owner, default_variable)

    @tc.typecheck
    def _parse_state_specific_specs(self, owner, state_dict, state_specific_spec):
        """Get variable spec and/or connections specified in an OutputState specification tuple

        Tuple specification can be:
            (state_spec, connections)
            (state_spec, variable spec, connections)

        See State._parse_state_specific_spec for additional info.

        Returns:
             - state_spec:  1st item of tuple
             - params dict with VARIABLE and/or PROJECTIONS entries if either of them was specified

        """
        # FIX: ADD FACILITY TO SPECIFY WEIGHTS AND/OR EXPONENTS FOR INDIVIDUAL OutputState SPECS
        #      CHANGE EXPECTATION OF *PROJECTIONS* ENTRY TO BE A SET OF TUPLES WITH THE WEIGHT AND EXPONENT FOR IT
        #      THESE CAN BE USED BY THE InputState's LinearCombination Function
        #          (AKIN TO HOW THE MECHANISM'S FUNCTION COMBINES InputState VALUES)
        #      THIS WOULD ALLOW FULLY GENEREAL (HIEARCHICALLY NESTED) ALGEBRAIC COMBINATION OF INPUT VALUES
        #      TO A MECHANISM
        from psyneulink.components.projections.projection import _parse_connection_specs, ProjectionTuple

        params_dict = {}
        state_spec = state_specific_spec

        if isinstance(state_specific_spec, dict):
            # state_dict[VARIABLE] = _parse_output_state_variable(owner, state_dict[VARIABLE])
            # # MODIFIED 3/10/18 NEW:
            # if state_dict[VARIABLE] is None:
            #     state_dict[VARIABLE] = DEFAULT_VARIABLE_SPEC
            # # MODIFIED 3/10/18 END
            return None, state_specific_spec

        elif isinstance(state_specific_spec, ProjectionTuple):
            state_spec = None
            params_dict[PROJECTIONS] = _parse_connection_specs(self,
                                                               owner=owner,
                                                               connections=[state_specific_spec])

        elif isinstance(state_specific_spec, tuple):
            tuple_spec = state_specific_spec
            state_spec = None
            TUPLE_VARIABLE_INDEX = 1

            if is_numeric(tuple_spec[0]):
                state_spec = tuple_spec[0]
                reference_value = state_dict[REFERENCE_VALUE]
                # Assign value so sender_dim is skipped below
                # (actual assignment is made in _parse_state_spec)
                if reference_value is None:
                    state_dict[REFERENCE_VALUE]=state_spec
                elif  not iscompatible(state_spec, reference_value):
                    raise OutputStateError("Value in first item of 2-item tuple specification for {} of {} ({}) "
                                     "is not compatible with its {} ({})".
                                     format(OutputState.__name__, owner.name, state_spec,
                                            REFERENCE_VALUE, reference_value))
                projection_spec = tuple_spec[1]

            else:
                projection_spec = state_specific_spec if len(state_specific_spec)==2 else (state_specific_spec[0],
                                                                                           state_specific_spec[-1])

            if not len(tuple_spec) in {2,3} :
                raise OutputStateError("Tuple provided in {} specification dictionary for {} ({}) must have "
                                       "either 2 ({} and {}) or 3 (optional additional {}) items, "
                                       "or must be a {}".
                                       format(OutputState.__name__, owner.name, tuple_spec,
                                              STATE, PROJECTION, 'variable spec', ProjectionTuple.__name__))


            params_dict[PROJECTIONS] = _parse_connection_specs(connectee_state_type=self,
                                                               owner=owner,
                                                               connections=projection_spec)


            # Get VARIABLE specification from (state_spec, variable spec, connections) tuple:
            if len(tuple_spec) == 3:

                tuple_variable_spec = tuple_spec[TUPLE_VARIABLE_INDEX]

                # Make sure OutputState's variable has not already been specified
                dict_variable_spec = None
                if VARIABLE in params_dict and params_dict[VARIABLE] is not None:
                    dict_variable_spec = params_dict[VARIABLE]
                elif VARIABLE in state_dict and state_dict[VARIABLE] is not None:
                    dict_variable_spec = params_dict[VARIABLE]
                if dict_variable_spec:
                    name = state_dict[NAME] or self.__name__
                    raise OutputStateError("Specification of {} in item 2 of 3-item tuple for {} ({})"
                                           "conflicts with its specification elsewhere in the constructor for {} ({})".
                                           format(VARIABLE, name, tuple_spec[TUPLE_VARIABLE_INDEX],
                                                  owner.name, dict_variable_spec))

                # Included for backward compatibility with INDEX
                if isinstance(tuple_variable_spec, int):
                    tuple_variable_spec = (OWNER_VALUE, tuple_variable_spec)

                # validate that it is a legitimate spec
                _parse_output_state_variable(owner, tuple_variable_spec)

                params_dict[VARIABLE] = tuple_variable_spec


        elif state_specific_spec is not None:
            raise OutputStateError("PROGRAM ERROR: Expected tuple or dict for {}-specific params but, got: {}".
                                  format(self.__class__.__name__, state_specific_spec))

        return state_spec, params_dict

    @staticmethod
    def _get_state_function_value(owner, function, variable):
        # -- CALL TO GET DEFAULT VALUE AND RETURN THAT (CAN'T USE VARIABLE SINCE DON'T KNOW MECH YET)
        #      THOUGH COULD PASS IN OWNER TO DETERMINE IT
        fct_variable = _parse_output_state_variable(owner, variable)

        # If variable has not been specified, assume it is the default of (OWNER_VALUE,0), and use that value
        if fct_variable is None:
            if owner.value is not None:
                fct_variable = owner.value[0]
            # Get owner's value by calling its function
            else:
                owner.function(owner.variable)[0]

        fct = _parse_output_state_function(owner, OutputState.__name__, function, fct_variable is PARAMS_DICT)

        try:
            return fct(variable=fct_variable)
        except:
            try:
                return fct(fct_variable)
            except TypeError as e:
                raise OutputStateError("Error in function assigned to {} of {}: {}".
                                       format(OutputState.__name__, owner.name, e.args[0]))

    @property
    def variable(self):
        return _parse_output_state_variable(self.owner, self._variable)


    @variable.setter
    def variable(self, variable):
        self._variable = variable

    def _update_variable(self, value):
        '''
            Used to mirror assignments to local variable in an attribute
            Knowingly not threadsafe
        '''
        try:
            return self.variable
        except AttributeError:
            self._variable = value
            return self.variable


    @property
    def owner_value_index(self):
        """Return index or indices of items of owner.value for any to which OutputState's variable has been assigned
        If the OutputState has been assigned to:
        - the entire owner value (i.e., OWNER_VALUE on its own, not in a tuple)
            return owner.value
        - a single item of owner.value (i.e.,  owner.value==(OWNER,index))
            return the index of the item
        - more than one, return a list of indices
        - to no items of owner.value (but possibly other params), return None
        """
        # Entire owner.value
        if isinstance(self._variable, str) and self.variable == OWNER_VALUE:
            return self.owner.value
        elif isinstance(self._variable, tuple):
            return self._variable[1]
        elif isinstance(self._variable, list):
            indices = [item[1] for item in self._variable if isinstance(item, tuple) and OWNER_VALUE in item]
            if len(indices)==1:
                return indices[0]
            elif indices:
                return indices
        else:
            return None

    @property
    def pathway_projections(self):
        return self.efferents

    @pathway_projections.setter
    def pathway_projections(self, assignment):
        self.efferents = assignment

    # For backward compatibility with INDEX and ASSIGN
    @property
    def calculate(self):
        return self.assign


def _instantiate_output_states(owner, output_states=None, context=None):
    """Call State._instantiate_state_list() to instantiate ContentAddressableList of OutputState(s)

    Create ContentAddressableList of OutputState(s) specified in paramsCurrent[OUTPUT_STATES]

    If output_states is not specified:
        - use owner.output_states as list of OutputState specifications
        - if owner.output_states is empty, use owner.value to create a default OutputState

    For each OutputState:
         check for VARIABLE and FUNCTION specifications:
             if it is a State, get from variable and function attributes
             if it is dict, look for VARIABLE and FUNCTION entries (and INDEX and ASSIGN for backward compatibility)
             if it is anything else, assume variable spec is (OWNER_VALUE, 0) and FUNCTION is Linear
         get OutputState's value using _parse_output_state_variable() and append to reference_value
             so that it matches specification of OutputStates (by # and function return values)

    When completed:
        - self.output_states contains a ContentAddressableList of one or more OutputStates;
        - self.output_state contains first or only OutputState in list;
        - paramsCurrent[OUTPUT_STATES] contains the same ContentAddressableList (of one or more OutputStates)
        - each OutputState properly references, for its variable, the specified attributes of its owner Mechanism
        - if there is only one OutputState, it is assigned the full value of its owner.

    (See State._instantiate_state_list() for additional details)

    IMPLEMENTATION NOTE:
        default(s) for self.paramsCurrent[OUTPUT_STATES] (self.value) are assigned here
        rather than in _validate_params, as it requires function to have been instantiated first

    Returns list of instantiated OutputStates
    """

    reference_value = []

    # Get owner.value
    # IMPLEMENTATION NOTE:  ?? IS THIS REDUNDANT WITH SAME TEST IN Mechanism.execute ?  JUST USE RETURN VALUE??
    owner_value = owner.instance_defaults.value

    # IMPLEMENTATION NOTE:  THIS IS HERE BECAUSE IF return_value IS A LIST, AND THE LENGTH OF ALL OF ITS
    #                       ELEMENTS ALONG ALL DIMENSIONS ARE EQUAL (E.G., A 2X2 MATRIX PAIRED WITH AN
    #                       ARRAY OF LENGTH 2), np.array (AS WELL AS np.atleast_2d) GENERATES A ValueError
    if (isinstance(owner_value, list) and
        (all(isinstance(item, np.ndarray) for item in owner_value) and
            all(
                    all(item.shape[i]==owner_value[0].shape[0]
                        for i in range(len(item.shape)))
                    for item in owner_value))):
        pass
    else:
        converted_to_2d = np.atleast_2d(owner.value)
        # If owner_value is a list of heterogenous elements, use as is
        if converted_to_2d.dtype == object:
            owner_value = owner.instance_defaults.value
        # Otherwise, use value converted to 2d np.array
        else:
            owner_value = converted_to_2d

    # This allows method to be called by Mechanism.add_input_states() with set of user-specified output_states,
    #    while calls from init_methods continue to use owner.output_states (i.e., OutputState specifications
    #    assigned in the **output_states** argument of the Mechanism's constructor)
    output_states = output_states or owner.output_states

    # Get the value of each OutputState
    # IMPLEMENTATION NOTE:
    # Should change the default behavior such that, if len(owner_value) == len owner.paramsCurrent[OUTPUT_STATES]
    #        (that is, there is the same number of items in owner_value as there are OutputStates)
    #        then increment index so as to assign each item of owner_value to each OutputState
    # IMPLEMENTATION NOTE:  SHOULD BE REFACTORED TO USE _parse_state_spec TO PARSE ouput_states arg
    if output_states:
        for i, output_state in enumerate(output_states):

            # OutputState object
            if isinstance(output_state, OutputState):
                if output_state.context.initialization_status == ContextFlags.DEFERRED_INIT:
                    try:
                        output_state_value = OutputState._get_state_function_value(owner,
                                                                                   output_state.function,
                                                                                   output_state.init_args[VARIABLE])
                    # For backward compatibility with INDEX and ASSIGN
                    except AttributeError:
                        index = output_state.index
                        output_state_value = owner_value[index]
                elif output_state.value is None:
                    output_state_value = output_state.function()

                else:
                    output_state_value = output_state.value

            else:
                # parse output_state
                from psyneulink.components.states.state import _parse_state_spec
                output_state = _parse_state_spec(state_type=OutputState, owner=owner, state_spec=output_state)

                _maintain_backward_compatibility(output_state, output_state[NAME], owner)

                # If OutputState's name matches the name entry of a dict in standard_output_states:
                #    - use the named Standard OutputState
                #    - merge initial specifications into std_output_state (giving precedence to user's specs)
                if output_state[NAME] and hasattr(owner, STANDARD_OUTPUT_STATES):
                    std_output_state = owner.standard_output_states.get_state_dict(output_state[NAME])
                    if std_output_state is not None:
                        _maintain_backward_compatibility(std_output_state, output_state[NAME], owner)
                        recursive_update(output_state, std_output_state, non_destructive=True)

                if FUNCTION in output_state and output_state[FUNCTION] is not None:
                    output_state_value = OutputState._get_state_function_value(owner,
                                                                               output_state[FUNCTION],
                                                                               output_state[VARIABLE])
                else:
                    output_state_value = _parse_output_state_variable(owner, output_state[VARIABLE])

            output_states[i] = output_state
            reference_value.append(output_state_value)

    else:
        reference_value = owner_value

    if hasattr(owner, OUTPUT_STATE_TYPE):
        outputStateType = owner.outputStateType
    else:
        outputStateType = OutputState

    state_list = _instantiate_state_list(owner=owner,
                                         state_list=output_states,
                                         state_type=outputStateType,
                                         state_param_identifier=OUTPUT_STATE,
                                         reference_value=reference_value,
                                         reference_value_name="output",
                                         context=context)

    # Call from Mechanism.add_states, so add to rather than assign output_states (i.e., don't replace)
    if context & (ContextFlags.COMMAND_LINE | ContextFlags.METHOD):
        owner.output_states.extend(state_list)
    else:
        owner._output_states = state_list

    return state_list


class StandardOutputStatesError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class StandardOutputStates():
    """Collection of OutputState specification dicts for `standard OutputStates <OutputState_Standard>` of a class.

    Parses specification of VARIABLE, assigning indices to OWNER_VALUE if specified.
    Adds <NAME_INDEX> of each OutputState as property of the owner's class, that returns the index of the OutputState
    in the list.


    Arguments
    ---------
    owner : Component
        the Component to which this OutputState belongs

    output_state_dicts : list of dicts
        list of dictionaries specifying OutputStates for the Component specified by `owner`

    indices : PRIMARY,
    SEQUENTIAL, list of ints
        specifies how to assign the (OWNER_VALUE, int) entry for each dict listed in `output_state_dicts`;

        The effects of each value of indices are as follows:

            * *PRIMARY* -- assigns (OWNER_VALUE, PRIMARY) to all output_states for which a VARIABLE entry is not
              specified;

            * *SEQUENTIAL* -- assigns sequentially incremented int to (OWNER_VALUE, int) spec of each OutputState,
              ignoring any VARIABLE entries previously specified for individual OutputStates;

            * list of ints -- assigns each int to an (OWNER_VALUE, int) entry of the corresponding OutputState in
              `output_state_dicts, ignoring any VARIABLE entries previously specified for individual OutputStates;

            * None -- assigns `None` to VARIABLE entries for all OutputStates for which it is not already specified.

    Attributes
    ----------
    data : dict
        dictionary of OutputState specification dictionaries

    indices : list
        list of default indices for each OutputState specified

    names : list
        list of the default names for each OutputState

    Methods
    -------
    get_state_dict(name)
        returns a copy of the designated OutputState specification dictionary
    """

    keywords = {PRIMARY, SEQUENTIAL, ALL}

    @tc.typecheck
    def __init__(self,
                 owner:Component,
                 output_state_dicts:list,
                 indices:tc.optional(tc.any(int, str, list))=None):
        self.owner = owner
        self._instantiate_state_list(output_state_dicts, indices)

    def _instantiate_state_list(self, output_state_dicts, indices):

        self.data = output_state_dicts.copy()

        # Validate that all items in output_state_dicts are dicts
        for item in output_state_dicts:
            if not isinstance(item, dict):
                raise StandardOutputStatesError(
                    "All items of {} for {} must be dicts (but {} is not)".
                    format(self.__class__.__name__, self.owner.componentName, item))

        # Assign indices

        # List was provided, so check that:
        # - it has the appropriate number of items
        # - they are all ints
        # and then assign each int to an (OWNER_VALUE, int) VARIABLE entry in the corresponding dict
        # in output_state_dicts
        # OutputState
        if isinstance(indices, list):
            if len(indices) != len(output_state_dicts):
                raise StandardOutputStatesError("Length of the list of indices "
                                                "provided to {} for {} ({}) "
                                                "must equal the number of "
                                                "OutputStates dicts provided "
                                                "({}) length".format(
                        self.__class__.__name__,
                        self.owner.name,
                        len(indices),
                        len(output_state_dicts)))

            if not all(isinstance(item, int) for item in indices):
                raise StandardOutputStatesError("All the items in the list of "
                                                "indices provided to {} for {} "
                                                "of {}) must be ints".
                                                format(self.__class__.__name__,
                                                       self.name,
                                                       self.owner.name))

            for index, state_dict in zip(indices, self.data):
                state_dict.update({VARIABLE:(OWNER_VALUE, index)})

        # Assign indices sequentially based on order of items in output_state_dicts arg
        elif indices is SEQUENTIAL:
            for index, state_dict in enumerate(self.data):
                state_dict.update({VARIABLE:(OWNER_VALUE, index)})

        # Assign (OWNER_VALUE, PRIMARY) as VARIABLE for all OutputStates in output_state_dicts that don't
        #    have VARIABLE (or INDEX) specified (INDEX is included here for backward compatibility)
        elif indices is PRIMARY:
            for state_dict in self.data:
                if INDEX in state_dict or VARIABLE in state_dict:
                    continue
                state_dict.update({VARIABLE:(OWNER_VALUE, PRIMARY)})

        # Validate all INDEX specification, parse any assigned as ALL, and
        # Add names of each OutputState as property of the owner's class that returns its name string
        for state in self.data:
            if INDEX in state:
                if state[INDEX] in ALL:
                    state.update({VARIABLE:OWNER_VALUE})
                elif state[INDEX] in PRIMARY:
                    state_dict.update({VARIABLE:(OWNER_VALUE, PRIMARY)})
                elif state[INDEX] in SEQUENTIAL:
                    raise OutputStateError("\'{}\' incorrectly assigned to individual {} in {} of {}.".
                                           format(SEQUENTIAL.upper(), OutputState.__name__, OUTPUT_STATE.upper(),
                                                  self.name))
                del state[INDEX]
            setattr(self.owner.__class__, state[NAME], make_readonly_property(state[NAME]))

        # For each OutputState dict with a VARIABLE entry that references it's owner's value (by index)
        # add <NAME_INDEX> as property of the OutputState owner's class that returns its index.
        for state in self.data:
            if isinstance(state[VARIABLE], tuple):
                index = state[VARIABLE][1]
            elif isinstance(state[VARIABLE], int):
                index = state[VARIABLE]
            else:
                continue
            setattr(self.owner.__class__, state[NAME]+'_INDEX', make_readonly_property(index))

    @tc.typecheck
    def add_state_dicts(self, output_state_dicts:list, indices:tc.optional(tc.any(int, str, list))=None):
        self.data.append(self._instantiate_state_list(output_state_dicts, indices))

    @tc.typecheck
    def get_state_dict(self, name:str):
        """Return a copy of the named OutputState dict
        """
        if next((item for item in self.names if name is item), None):
            # assign dict to owner's output_state list
            return self.data[self.names.index(name)].copy()
        # raise StandardOutputStatesError("{} not recognized as name of {} for {}".
        #                                 format(name, StandardOutputStates.__class__.__name__, self.owner.name))
        return None

    # @tc.typecheck
    # def get_dict(self, name:str):
    #     return self.data[self.names.index(name)].copy()
    #
    @property
    def names(self):
        return [item[NAME] for item in self.data]

    # @property
    # def indices(self):
    #     return [item[INDEX] for item in self.data]


def _parse_output_state_variable(owner, variable, output_state_name=None):
    """Return variable for OutputState based on VARIABLE entry of owner's params dict

    The format of the VARIABLE entry determines the format returned:
    - if it is a single item, or a single item in a list, a single item is returned;
    - if it is a list with more than one item, a list is returned.
    :return:
    """

    def parse_variable_spec(spec):
        from psyneulink.components.mechanisms.mechanism import MechParamsDict
        if spec is None or is_numeric(spec) or isinstance(spec, MechParamsDict):
            return spec
        elif isinstance(spec, tuple):
            # Tuple indexing item of owner's attribute (e.g.,: OWNER_VALUE, int))
            try:
                return owner.attributes_dict[spec[0]][spec[1]]
            except TypeError:
                if owner.attributes_dict[spec[0]] is None:
                    return None
                else:
                    raise OutputStateError("Can't parse variable ({}) for {} of {}".
                                           format(spec, output_state_name or OutputState.__name__, owner.name))
        elif isinstance(spec, str) and spec == PARAMS_DICT:
            # Specifies passing owner's params_dict as variable
            return owner.attributes_dict
        elif isinstance(spec, str):
            # Owner's full value or attribute other than its value
            return owner.attributes_dict[spec]
        else:
            raise OutputStateError("\'{}\' entry for {} specification dictionary of {} ({}) must be "
                                   "numeric or a list of {} attribute names".
                                   format(VARIABLE.upper(),
                                          output_state_name or OutputState.__name__,
                                          owner.name, spec,
                                          owner.__class__.__name__))
    if not isinstance(variable, list):
        variable = [variable]

    if len(variable)== 1:
        return parse_variable_spec(variable[0])

    fct_variable = []
    for spec in variable:
        fct_variable.append(parse_variable_spec(spec))
    return fct_variable


def _parse_output_state_function(owner, output_state_name, function, params_dict_as_variable=False):
    """ Parse specification of function as Function, Function class, Function.function, function_type or method_type.

    If params_dict_as_variable is True, and function is a Function, check whether it allows params_dict as variable;
    if it is and does, leave as is,
    otherwise, wrap in lambda function that provides first item of OutputState's value as the functions argument.
    """
    if function is None:
        function = OutputState.ClassDefaults.function

    if isinstance(function, (function_type, method_type)):
        return function

    if isinstance(function, type) and issubclass(function, Function):
        function = function()
    if isinstance(function, Function):
        fct = function.function
    else:
        raise OutputStateError("Specification of \'{}\' for {} of {} must be a {}, the class or function of one "
                               "or a callable object (Python function or method)".
                               format(FUNCTION.upper(), output_state_name, owner.name, Function.__name__))
    if params_dict_as_variable:
        # Function can accept params_dict as its variable
        if hasattr(function, 'params_dict_as_variable'):
            return fct
        # Allow params_dict to be passed to any function, that will use the first item of the owner's value by default
        else:
            if owner.verbosePref is True:
                warnings.warn("{} specified as {} is incompatible with {} specified as {} for {} of {}; "
                              "1st item of {}'s {} attribute will be used instead".
                              format(PARAMS_DICT.upper(), VARIABLE.upper(), function.name, FUNCTION.upper(),
                                     OutputState.name, owner.name, owner.name, VALUE))
            return lambda x : fct(x[OWNER_VALUE][0])
    return fct


def make_readonly_property(val):
    """Return property that provides read-only access to its value
    """

    def getter(self):
        return val

    def setter(self, val):
        raise UtilitiesError("{} is read-only property of {}".format(val, self.__class__.__name__))

    # Create the property
    prop = property(getter).setter(setter)
    return prop


@tc.typecheck
def _maintain_backward_compatibility(d:dict, name, owner):
    """Maintain compatibility with use of INDEX, ASSIGN and CALCULATE in OutputState specification"""

    def replace_entries(x):

        index_present = False
        assign_present = False
        calculate_present = False

        if INDEX in x:
            index_present = True
            # if output_state[INDEX] is SEQUENTIAL:
            #     return
            if x[INDEX] is ALL:
                x[VARIABLE] = OWNER_VALUE
            else:
                x[VARIABLE] = (OWNER_VALUE, x[INDEX])
            del x[INDEX]
        if ASSIGN in x:
            assign_present = True
            x[FUNCTION] = x[ASSIGN]
            del x[ASSIGN]
        if CALCULATE in x:
            calculate_present = True
            x[FUNCTION] = x[CALCULATE]
            del x[CALCULATE]
        return x, index_present, assign_present, calculate_present

    d, i, a, c = replace_entries(d)

    if PARAMS in d and isinstance(d[PARAMS], dict):
        p, i, a, c = replace_entries(d[PARAMS])
        recursive_update(d, p, non_destructive=True)
        for spec in {VARIABLE, FUNCTION}:
            if spec in d[PARAMS]:
                del d[PARAMS][spec]

    if i:
        warnings.warn("The use of \'INDEX\' has been deprecated; it is still supported, but entry in {} specification "
                      "dictionary for {} of {} should be changed to \'VARIABLE: (OWNER_VALUE, <index int>)\' "
                      " for future compatibility.".
                      format(OutputState.__name__, name, owner.name))
        assert False
    if a:
        warnings.warn("The use of \'ASSIGN\' has been deprecated; it is still supported, but entry in {} specification "
                      "dictionary for {} of {} should be changed to \'FUNCTION\' for future compatibility.".
                      format(OutputState.__name__, name, owner.name))
        assert False
    if c:
        warnings.warn("The use of \'CALCULATE\' has been deprecated; it is still supported, but entry in {} "
                      "specification dictionary for {} of {} should be changed to \'FUNCTION\' "
                      "for future compatibility.".format(OutputState.__name__, name, owner.name))

    if name is MECHANISM_VALUE:
        warnings.warn("The name of the \'MECHANISM_VALUE\' StandardOutputState has been changed to \'OWNER_VALUE\';  "
                      "it will still work, but should be changed in {} specification of {} for future compatibility.".
                      format(OUTPUT_STATES, owner.name))
        assert False


